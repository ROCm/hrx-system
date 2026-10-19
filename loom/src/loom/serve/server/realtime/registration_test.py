# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Realtime bindings and history across public registration replacement."""

import base64
import struct

from catalog_test_util import Client as HttpClient
from client_test_util import Client


def replace_model(admin, definition):
    name = definition["name"]
    removed = admin.request(
        "/v1/models/unregister", {"model": name, "mode": "drain"}, status=202
    )
    while True:
        models = admin.request("/v1/status")["data"]
        assert all("error" not in model["loom"] for model in models), models
        if name not in {model["id"] for model in models}:
            break
    replacement = admin.request(
        "/v1/models/register", {**definition, "admission": "open"}, status=202
    )
    assert replacement["loom"]["incarnation"] != removed["loom"]["incarnation"], (
        removed,
        replacement,
    )


def microphone(client, *, automatic):
    if automatic:
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 1,
                            "silence_duration_ms": 1,
                            "create_response": False,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
    samples = [0] * 96 + [64] * 192 + [0] * 480
    client.send(
        "input_audio_buffer.append",
        audio=base64.b64encode(struct.pack(f"<{len(samples)}h", *samples)).decode(),
    )
    if not automatic:
        client.send("input_audio_buffer.commit")


def close(client):
    client.send_frame(8, struct.pack(">H", 1000))
    assert client.read_frame() == (8, struct.pack(">H", 1000))
    assert client.input.read(1) == b""


def registration_replacement(address, trace, definitions):
    admin = HttpClient(address)
    # Activity and transcription have distinct registrations and input owners,
    # even when both use the same recognition-capable source package.
    for name, component in (
        ("speech", "speech"),
        ("fixture-text", "text"),
        ("recognition", "recognition"),
        ("activity", "recognition"),
    ):
        original = Client(address, trace)
        replacement = None
        drained = False
        try:
            assert original.receive()["type"] == "session.created"
            original.message("before", "ab")
            original.respond([{"role": "user", "content": "ab"}])
            definition = next(value for value in definitions if value["name"] == name)
            replace_model(admin, definition)

            replacement = Client(address, trace)
            assert replacement.receive()["type"] == "session.created"
            microphone(replacement, automatic=name == "activity")
            transcript = replacement.until(
                "conversation.item.input_audio_transcription.completed"
            )[-1]
            assert transcript["transcript"] == "A", transcript
            item = replacement.until("conversation.item.done")[-1]["item"]
            assert item["id"] == transcript["item_id"], item
            messages = [{"role": "user", "content": "A"}]
            response = replacement.respond(messages)
            messages.append(
                {
                    "role": "assistant",
                    "content": response["output"][0]["content"][0]["transcript"],
                }
            )

            if name in ("speech", "fixture-text"):
                original.send("response.create")
                response = original.until("response.done")[-1]["response"]
                assert response["status"] == "failed", response
                error = response["status_details"]["error"]
                assert error["code"] == "NOT_FOUND", error
                assert error["message"] == component + " registration has retired", (
                    error
                )

            # Closing an old history must neither wait for reopened inference
            # nor release the replacement conversation's retained prefix.
            admin.request("/v1/models/drain", {"model": "fixture-text"})
            drained = True
            if name in ("recognition", "activity"):
                microphone(original, automatic=name == "activity")
                while True:
                    event = original.receive(allow_error=True)
                    if event["type"] == "error":
                        break
                assert event["error"]["code"] == "NOT_FOUND", event
                assert (
                    event["error"]["message"] == "recognition registration has retired"
                ), event
                assert original.read_frame() == (8, struct.pack(">H", 1011))
                assert original.input.read(1) == b""
            else:
                close(original)
            admin.request("/v1/models/undrain", {"model": "fixture-text"})
            drained = False

            replacement.message("after", "ab")
            messages.append({"role": "user", "content": "ab"})
            response = replacement.respond(messages)
            assert response["usage"]["input_token_details"]["cached_tokens"] > 0, (
                response
            )
            admin.request("/v1/models/drain", {"model": "fixture-text"})
            drained = True
            close(replacement)
        finally:
            original.close()
            if replacement is not None:
                replacement.close()
            if drained:
                admin.request("/v1/models/undrain", {"model": "fixture-text"})
        admin.idle()
