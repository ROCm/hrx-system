# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native PCM conversation through the production CLI and WebSocket endpoint."""

import base64
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

from loom_serve_testing.http_harness import running_server

from loom.src.loom.serve.server.catalog_test_util import (
    Client as HttpClient,
)
from loom.src.loom.serve.server.catalog_test_util import (
    check_pcm,
    complete,
    prepare_catalog,
    write_startup,
)
from loom.src.loom.serve.server.realtime.client_test_util import (
    Client,
    check_response_metadata,
)
from loom.src.loom.serve.server.realtime.independent_test import (
    independent_admission,
    independent_automatic_intent,
    independent_cancel_scope,
    independent_contexts,
    independent_function_input,
    independent_pending_input,
)
from loom.src.loom.serve.server.realtime.registration_test import (
    registration_replacement,
)


def response_policy(address, trace):
    client = Client(address, trace)
    try:
        defaults = client.receive()["session"]
        invalid_metadata = [
            [],
            "metadata",
            1,
            {str(i): "value" for i in range(17)},
            {"k" * 65: "value"},
            {"🦉" * 65: "value"},
            {"key": "v" * 513},
            {"key": "🦉" * 513},
            *({"key": value} for value in (None, True, 1, [], {})),
        ]
        invalid_overrides = [
            *({"metadata": value} for value in invalid_metadata),
            {"type": "realtime"},
            {"audio": {"input": {}}},
            {"audio": {"output": {"voice": ""}}},
            {"audio": {"output": {"speed": 2}}},
            {"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}},
        ]
        for invalid in invalid_overrides:
            client.send(
                "response.create",
                response={
                    "instructions": "x",
                    "audio": {"output": {"voice": "negative"}},
                    **invalid,
                },
            )
            assert client.receive(allow_error=True)["type"] == "error", invalid
        for metadata in (
            r'{"a":"one","\u0061":"two"}',
            r'{"\u0000":"one","\u0000":"two"}',
            r'{"\uD800":"value"}',
            r'{"key":"\uDC00"}',
        ):
            client.send_frame(
                1,
                (
                    '{"type":"response.create","response":{"metadata":'
                    + metadata
                    + "}}"
                ).encode(),
            )
            assert client.receive(allow_error=True)["type"] == "error", metadata
        client.send("session.update", session={"metadata": {}})
        assert client.receive(allow_error=True)["type"] == "error"
        client.send("session.update", session={})
        assert client.until("session.updated")[-1]["session"] == defaults

        client.message("first", "a")
        messages = [{"role": "user", "content": "a"}]
        first = client.respond(messages)
        assert first["status"] == "incomplete", first
        messages.append(
            {
                "role": "assistant",
                "content": first["output"][0]["content"][0]["transcript"],
            }
        )
        client.message("second", "ab")
        messages.append({"role": "user", "content": "ab"})
        # Character limits count Unicode codepoints, including four-byte scalars.
        # Empty and embedded-NUL keys remain distinct after decoding and echoing.
        metadata = {
            "": "",
            "\0": 'quoted " and escaped \\ newline\nNUL\0',
            **{chr(0x1F300 + i) * 64: "🦉" * 512 for i in range(14)},
        }
        second = client.respond(
            messages,
            maximum=16,
            voice="negative",
            overrides={
                "metadata": metadata,
                "audio": {
                    "output": {
                        "voice": "negative",
                        "speed": 1,
                        "format": {"type": "audio/pcm", "rate": 24000},
                    }
                },
            },
        )
        assert second["status"] == "completed", second
        assert second["usage"]["input_token_details"]["cached_tokens"] > 0, second
        messages.append(
            {
                "role": "assistant",
                "content": second["output"][0]["content"][0]["transcript"],
            }
        )
        client.message("third", "x")
        messages.append({"role": "user", "content": "x"})
        third = client.respond(
            [{"role": "system", "content": "x"}, *messages],
            maximum=3,
            overrides={
                "instructions": "x",
                "output_modalities": ["text"],
                "metadata": {},
            },
        )
        assert third["usage"]["input_token_details"]["cached_tokens"] == 0, third
        messages.append(
            {"role": "assistant", "content": third["output"][0]["content"][0]["text"]}
        )
        client.message("fourth", "a")
        messages.append({"role": "user", "content": "a"})
        fourth = client.respond(messages, overrides={"metadata": None})
        assert fourth["usage"]["input_token_details"]["cached_tokens"] > 0, fourth
        client.send("session.update", session={})
        assert client.until("session.updated")[-1]["session"] == defaults
    finally:
        client.close()


def conversation(address, trace):
    client = Client(address, trace)
    try:
        created = client.receive()
        assert created["type"] == "session.created", created
        settings = created["session"]
        assert settings["audio"]["input"]["turn_detection"] is None, settings
        assert settings["audio"]["input"]["format"] == {
            "type": "audio/pcm",
            "rate": 24000,
        }, settings
        for malformed in (
            '{"type":"session.update","type":"response.create","session":{}}',
            '{"type":"session.update","session":{},"se\\u0073sion":{}}',
            '{"type":"session.update","session":{"instructions":"x","max_output_tokens":0}}',
            '{"type":"input_audio_buffer.append","audio":null}',
            '{"type":"response.create","response":{"output_modalities":["audio","text"]}}',
            '{"type":"input_audio_buffer.clear"} {}',
        ):
            client.send_frame(1, malformed.encode())
            assert client.receive(allow_error=True)["type"] == "error", malformed
        client.send("session.update", session={})
        assert client.until("session.updated")[-1]["session"] == settings
        # The 24->16 kHz converter must flush its final lookahead on commit,
        # even when no more microphone frames ever arrive on this connection.
        for count in (1, 7, 93, 139):
            client.send(
                "input_audio_buffer.append",
                audio=base64.b64encode(bytes(2 * count)).decode(),
            )
        client.send("input_audio_buffer.commit")
        events = client.until("conversation.item.input_audio_transcription.completed")
        transcript = events[-1]
        assert transcript["transcript"] == "A", transcript
        assert transcript["usage"] == {"type": "duration", "seconds": 0.01}, transcript
        assert sum(e["type"] == "input_audio_buffer.committed" for e in events) == 1
        assert not any(e["type"] == "response.created" for e in events), events
        messages = [{"role": "user", "content": "A"}]
        response = client.respond(messages)
        assert response["usage"]["input_tokens"] == 1, response
        messages.append(
            {
                "role": "assistant",
                "content": response["output"][0]["content"][0]["transcript"],
            }
        )
        # Voice policy cannot invalidate compatible text history. A later
        # truncation removes only the unaligned transcript of the heard item.
        client.send(
            "session.update", session={"audio": {"output": {"voice": "negative"}}}
        )
        updated = client.until("session.updated")[-1]["session"]
        assert updated["audio"]["output"]["voice"] == "negative", updated
        client.message("second", "aB")
        messages.append({"role": "user", "content": "aB"})
        second = client.respond(messages, maximum=16, voice="negative")
        assert second["usage"]["input_token_details"]["cached_tokens"] > 0, second
        second_item = second["output"][0]
        assert len(second_item["content"][0]["transcript"]) >= 3, second_item
        client.send(
            "conversation.item.truncate",
            item_id=second_item["id"],
            content_index=0,
            audio_end_ms=1,
        )
        truncated = client.until("conversation.item.truncated")[-1]
        assert (
            truncated["item_id"] == second_item["id"] and truncated["audio_end_ms"] == 1
        ), truncated
        client.message("third", "X")
        messages.append({"role": "user", "content": "X"})
        third = client.respond(messages, voice="negative")
        assert third["usage"]["input_token_details"]["cached_tokens"] > 0, third
        admin = HttpClient(address)
        admin.request("/v1/models/drain", {"model": "fixture-text"})
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
        # Close must join checkpoint release even while inference is drained.
        assert client.input.read(1) == b""
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
    finally:
        client.close()


def response_failure(address, trace, *, speech):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        if speech:
            client.send(
                "session.update", session={"audio": {"output": {"voice": "missing"}}}
            )
            client.until("session.updated")
        # Both failures come from real native request preparation: a source-
        # rejected voice or input plus output exceeding the text context.
        client.message("rejected-input", "a" if speech else "a" * 128)
        client.send(
            "input_audio_buffer.append", audio=base64.b64encode(bytes(480)).decode()
        )
        metadata = {
            "input_item_id": "rejected-input",
            "kind": "speech" if speech else "text",
        }
        client.send("response.create", response={"metadata": metadata})
        events = client.until("response.done")
        check_response_metadata(events, metadata)
        response = events[-1]["response"]
        assert response["status"] == "failed", response
        details = response["status_details"]
        assert details["type"] == "failed", details
        error = details["error"]
        assert error["code"] == ("INVALID_ARGUMENT" if speech else "OUT_OF_RANGE"), (
            error
        )
        assert error["message"], error
        assert not any(e["type"] == "response.output_audio.delta" for e in events), (
            events
        )
        if speech:
            # Text supplies the first content before the independently queued
            # speech preparation rejects its voice. The real message survives
            # as incomplete, with its unaligned transcript removed.
            assert response["output"][0]["status"] == "incomplete", response
            assert response["output"][0]["content"][0]["transcript"] is None, response
            assert any(e["type"] == "response.output_item.added" for e in events), (
                events
            )
        else:
            assert response["output"] == [], response
            assert not any(e["type"] == "response.output_item.added" for e in events), (
                events
            )
        client.send("conversation.item.delete", item_id="rejected-input")
        client.until("conversation.item.deleted")
        client.send(
            "session.update", session={"audio": {"output": {"voice": "positive"}}}
        )
        client.until("session.updated")
        # Previously recorded microphone audio survives the failed response.
        # Failed audio contributes no unaligned text to the recovered prompt.
        client.send("input_audio_buffer.commit")
        transcript = client.until(
            "conversation.item.input_audio_transcription.completed"
        )[-1]
        assert transcript["transcript"] == "A", transcript
        client.respond([{"role": "user", "content": "A"}])
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
        assert client.input.read(1) == b""
    finally:
        client.close()


def cancellation(address, trace):
    admin = HttpClient(address)
    admin.request("/v1/models/drain", {"model": "fixture-text"})
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.message("pending", "a")
        metadata = {"input_item_id": "pending"}
        client.send("response.create", response={"metadata": metadata})
        events = client.until("response.created")
        identity = next(
            e["response"]["id"] for e in events if e["type"] == "response.created"
        )
        # Drained native admission is an explicit readiness boundary: this
        # cancellation cannot race with a fixture that already completed.
        client.send("response.cancel", response_id=identity)
        events += client.until("response.done")
        check_response_metadata(events, metadata)
        response = events[-1]["response"]
        assert response["id"] == identity and response["status"] == "cancelled", (
            response
        )
        assert response["status_details"] == {
            "type": "cancelled",
            "reason": "client_cancelled",
        }, response
        assert not any(e["type"].endswith(".delta") for e in events), events
        assert response["output"] == [], response
        assert not any(e["type"] == "response.output_item.added" for e in events), (
            events
        )
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
        client.respond([{"role": "user", "content": "a"}])
        client.send("input_audio_buffer.append", audio="AAAAAA==")
        client.send("input_audio_buffer.clear")
        client.until("input_audio_buffer.cleared")
        client.send("input_audio_buffer.commit", event_id="empty-commit")
        error = client.receive(allow_error=True)
        assert (
            error["type"] == "error" and error["error"]["event_id"] == "empty-commit"
        ), error
        client.send(
            "input_audio_buffer.append",
            audio=base64.b64encode(bytes(480)).decode(),
        )
        client.send("input_audio_buffer.commit")
        transcript = client.until(
            "conversation.item.input_audio_transcription.completed"
        )[-1]
        assert transcript["transcript"] == "A", transcript
        assert transcript["usage"] == {"type": "duration", "seconds": 0.01}, transcript
        client.until("conversation.item.done")
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def pipelined_microphone_items(address, trace):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        admin.request("/v1/models/drain", {"model": "recognition"})
        drained = True
        identities = []
        for frames in (240, 480):
            client.send(
                "input_audio_buffer.append",
                audio=base64.b64encode(bytes(frames * 2)).decode(),
            )
            client.send("input_audio_buffer.commit")
            committed = client.until("input_audio_buffer.committed")[-1]
            identities.append(committed["item_id"])
        admin.request("/v1/models/undrain", {"model": "recognition"})
        drained = False
        transcripts = {}
        done = set()
        while len(transcripts) != 2 or len(done) != 2:
            event = client.receive()
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcripts[event["item_id"]] = event
            elif event["type"] == "conversation.item.done":
                done.add(event["item"]["id"])
        for identity, seconds in zip(identities, (0.01, 0.02), strict=True):
            assert transcripts[identity]["transcript"] == "A", transcripts
            assert transcripts[identity]["usage"] == {
                "type": "duration",
                "seconds": seconds,
            }, transcripts
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "recognition"})
        client.close()


def automatic_microphone(address, trace, *, partitioned, create_response):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        for invalid in (
            {"type": "semantic_vad"},
            {"type": "server_vad", "threshold": 1.1},
            {"type": "server_vad", "prefix_padding_ms": 2000},
            {"type": "server_vad", "silence_duration_ms": -1},
            {"type": "server_vad", "idle_timeout_ms": 1000},
        ):
            client.send(
                "session.update",
                session={"audio": {"input": {"turn_detection": invalid}}},
            )
            assert client.receive(allow_error=True)["type"] == "error", invalid
        policy = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 1,
            "silence_duration_ms": 1,
            "create_response": create_response,
            "interrupt_response": True,
        }
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": policy}}}
        )
        updated = client.until("session.updated")[-1]["session"]
        assert updated["audio"]["input"]["turn_detection"] == policy, updated
        samples = [0] * 96 + [64] * 192 + [0] * 480
        if not create_response:
            samples *= 2
        raw = struct.pack(f"<{len(samples)}h", *samples)
        position = 0
        parts = [1, 7, 93] if partitioned else [len(samples)]
        while position < len(raw):
            count = parts.pop(0) if parts else 137
            payload = raw[position : position + count * 2]
            client.send(
                "input_audio_buffer.append", audio=base64.b64encode(payload).decode()
            )
            position += len(payload)
        count = 1 if create_response else 2
        starts, stops, transcripts, done, events = {}, {}, {}, set(), []
        response_done = False
        while len(done) != count or (create_response and not response_done):
            event = client.receive()
            events.append(event)
            kind = event["type"]
            response_done |= kind == "response.done"
            if kind == "input_audio_buffer.speech_started":
                starts[event["item_id"]] = event["audio_start_ms"]
            elif kind == "input_audio_buffer.speech_stopped":
                stops[event["item_id"]] = event["audio_end_ms"]
            elif kind == "conversation.item.input_audio_transcription.completed":
                transcripts[event["item_id"]] = event
            elif kind == "conversation.item.done" and event["item"]["role"] == "user":
                done.add(event["item"]["id"])
        assert len(starts) == len(stops) == len(transcripts) == count, events
        intervals = []
        for identity, begin in starts.items():
            end = stops[identity]
            assert 0 <= begin < end, events
            assert transcripts[identity]["transcript"] == "A", events
            intervals.append((begin, end, transcripts[identity]["usage"]["seconds"]))
        assert all(
            left[1] <= right[0] for left, right in zip(intervals, intervals[1:])
        ), intervals
        messages = [{"role": "user", "content": "A"}] * count
        if create_response:
            text = "".join(
                e["delta"]
                for e in events
                if e["type"] == "response.output_audio_transcript.delta"
            )
            assert text == complete(messages, 2)[0], events
            audio = b"".join(
                base64.b64decode(e["delta"])
                for e in events
                if e["type"] == "response.output_audio.delta"
            )
            check_pcm(audio, text, 42)
        else:
            assert not any(e["type"] == "response.created" for e in events), events
            client.respond(messages)
        # Changing classified input policy requires an explicit ordered cut.
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": None}}}
        )
        assert client.receive(allow_error=True)["type"] == "error"
        client.send("input_audio_buffer.clear")
        client.until("input_audio_buffer.cleared")
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": None}}}
        )
        updated = client.until("session.updated")[-1]["session"]
        assert updated["audio"]["input"]["turn_detection"] is None, updated
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
        return intervals
    finally:
        client.close()


def automatic_explicit_boundaries(address, trace):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            # Retain the full quiet interval for explicit commit
                            # regardless of concurrent detector progress.
                            "prefix_padding_ms": 20,
                            "silence_duration_ms": 100,
                            "create_response": True,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        raw = struct.pack("<288h", *([64] * 192 + [0] * 96))
        client.send("input_audio_buffer.append", audio=base64.b64encode(raw).decode())
        started = client.until("input_audio_buffer.speech_started")[-1]
        assert started["audio_start_ms"] == 0, started
        client.send("input_audio_buffer.commit")
        client.send("input_audio_buffer.append", audio=base64.b64encode(raw).decode())
        client.send("input_audio_buffer.clear")
        client.send(
            "input_audio_buffer.append", audio=base64.b64encode(bytes(960)).decode()
        )
        client.send("input_audio_buffer.commit")
        transcripts, events, done = [], [], set()
        while len(done) != 2:
            event = client.receive()
            events.append(event)
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcripts.append(event)
            elif event["type"] == "conversation.item.done":
                done.add(event["item"]["id"])
        assert sorted(e["usage"]["seconds"] for e in transcripts) == [0.012, 0.02], (
            transcripts
        )
        assert all(e["transcript"] == "A" for e in transcripts), transcripts
        assert not any(e["type"] == "response.created" for e in events), events
        client.send(
            "session.update", session={"audio": {"input": {"turn_detection": None}}}
        )
        client.until("session.updated")
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def automatic_response_policy(address, trace, *, interrupt):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 1,
                            "silence_duration_ms": 1,
                            "create_response": not interrupt,
                            "interrupt_response": interrupt,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        client.message("before", "a")
        admin.request("/v1/models/drain", {"model": "speech"})
        drained = True
        client.send("response.create")
        response = client.until("response.created")[-1]["response"]
        raw = struct.pack("<768h", *([0] * 96 + [64] * 192 + [0] * 480))
        client.send("input_audio_buffer.append", audio=base64.b64encode(raw).decode())
        events = client.until("conversation.item.input_audio_transcription.completed")
        assert events[-1]["transcript"] == "A", events
        assert any(e["type"] == "input_audio_buffer.speech_started" for e in events), (
            events
        )
        if interrupt:
            while not any(e["type"] == "response.done" for e in events):
                events.append(client.receive())
            cancelled = next(
                e["response"] for e in events if e["type"] == "response.done"
            )
            assert (
                cancelled["id"] == response["id"] and cancelled["status"] == "cancelled"
            ), events
            assert cancelled["status_details"] == {
                "type": "cancelled",
                "reason": "turn_detected",
            }, cancelled
            assert cancelled["output"] == [], cancelled
            client.send("response.create")
            client.until("response.created")
        else:
            assert not any(
                e["type"] in ("response.done", "response.created") for e in events
            ), events
        admin.request("/v1/models/undrain", {"model": "speech"})
        drained = False
        messages = [{"role": "user", "content": "a"}]
        if not interrupt:
            first = client.until("response.done")
            assert first[-1]["response"]["id"] == response["id"], first
            assert first[-1]["response"]["status"] == "incomplete", first
            text = first[-1]["response"]["output"][0]["content"][0]["transcript"]
            assert text == complete(messages, 2)[0], first
            messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "A"})
        second = client.until("response.done")
        response = second[-1]["response"]
        assert response["status"] in ("completed", "incomplete"), response
        text = response["output"][0]["content"][0]["transcript"]
        assert text == complete(messages, 2)[0], response
        audio = b"".join(
            base64.b64decode(e["delta"])
            for e in second
            if e["type"] == "response.output_audio.delta"
        )
        check_pcm(audio, text, 42)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "speech"})
        client.close()


def rejected_activity_geometry(address, trace):
    client = Client(address, trace, model="bounded")
    try:
        assert client.receive()["session"]["model"] == "bounded"
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 0,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        client.send("input_audio_buffer.append", audio="AAAAAA==")
        failure = client.receive(allow_error=True)
        assert failure["type"] == "error", failure
        assert failure["error"]["type"] == "server_error", failure
        assert "input_capacity" in failure["error"]["message"], failure
        assert client.read_frame() == (8, struct.pack(">H", 1011))
        assert client.input.read(1) == b""
        status = HttpClient(address).request("/v1/status")
        assert len(status["data"]) == 3, status
    finally:
        client.close()


def rejected_capture(address, trace):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.message("capture", "a")
        client.send("response.create")
        while True:
            event = client.receive(allow_error=True)
            if event["type"] == "error":
                assert event["error"]["type"] == "server_error", event
                break
        # A profile without retained text cannot capture its completed prefix.
        # That rejected operation owns no pin and must not abort the server.
        assert client.read_frame() == (8, struct.pack(">H", 1011))
        assert client.input.read(1) == b""
        admin = HttpClient(address)
        status = admin.request("/v1/status")
        assert len(status["data"]) == 3, status
        result = admin.request(
            "/v1/chat/completions",
            admin.text_payload([{"role": "user", "content": "a"}], 2),
        )
        assert result["text"] == "bb", result
    finally:
        client.close()


def anonymous_item(address, trace):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.send(
            "conversation.item.create",
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "a"}],
            },
        )
        item = client.until("conversation.item.done")[-1]["item"]
        response = client.respond([{"role": "user", "content": "a"}])
        assert item["id"] != response["output"][0]["id"], response
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def reserved_item_identity(address, trace):
    client = Client(address, trace)
    try:
        session = client.receive()["session"]
        namespace = session["id"].removeprefix("sess_")
        claimed = f"item_{namespace}_1"
        client.message(claimed, "a")
        response = client.respond([{"role": "user", "content": "a"}])
        assert response["output"][0]["id"] == f"item_{namespace}_2", response
        client.send(
            "session.update",
            session={
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "silence_duration_ms": 100,
                            "create_response": False,
                            "interrupt_response": False,
                        }
                    }
                }
            },
        )
        client.until("session.updated")
        raw = struct.pack("<288h", *([64] * 192 + [0] * 96))
        for commit in (True, False):
            client.send(
                "input_audio_buffer.append", audio=base64.b64encode(raw).decode()
            )
            started = client.until("input_audio_buffer.speech_started")[-1]
            identity = started["item_id"]
            client.send(
                "conversation.item.create",
                event_id="reserved-collision",
                item={
                    "id": identity,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "b"}],
                },
            )
            error = client.receive(allow_error=True)
            assert error["error"]["code"] == "INVALID_ARGUMENT", error
            assert error["error"]["event_id"] == "reserved-collision", error
            if commit:
                client.send("input_audio_buffer.commit")
                events = client.until("conversation.item.done")
                transcript = next(
                    e
                    for e in events
                    if e["type"]
                    == "conversation.item.input_audio_transcription.completed"
                )
                assert transcript["item_id"] == identity, transcript
                assert transcript["transcript"] == "A", transcript
            else:
                client.send("input_audio_buffer.clear")
                client.until("input_audio_buffer.cleared")
                # Clear releases the reserved identity without publishing a
                # canonical item; the same explicit ID can now be accepted.
                client.message(identity, "b")
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def deferred_response_frontier(address, trace):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        admin.request("/v1/models/drain", {"model": "recognition"})
        drained = True
        audio = base64.b64encode(bytes(480)).decode()
        client.send("input_audio_buffer.append", audio=audio)
        client.send("input_audio_buffer.commit")
        first = client.until("conversation.item.added")[-1]["item"]["id"]
        client.send("response.create")
        created = client.until("response.created")[-1]["response"]
        client.send("input_audio_buffer.append", audio=audio)
        client.send("input_audio_buffer.commit")
        appended = client.until("conversation.item.added")[-1]
        second = appended["item"]["id"]
        assert appended["previous_item_id"] == first, appended
        admin.request("/v1/models/undrain", {"model": "recognition"})
        drained = False
        events, transcripts, inputs_done = [], set(), set()
        response = None
        while response is None or len(transcripts) != 2 or len(inputs_done) != 2:
            event = client.receive()
            events.append(event)
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcripts.add(event["item_id"])
                assert event["transcript"] == "A", event
            elif (
                event["type"] == "conversation.item.done"
                and event["item"]["role"] == "user"
            ):
                inputs_done.add(event["item"]["id"])
            elif event["type"] == "response.done":
                response = event["response"]
        assert transcripts == {first, second}, events
        assert response["id"] == created["id"], response
        messages = [{"role": "user", "content": "A"}]
        expected = complete(messages, 2)[0]
        assert response["output"][0]["content"][0]["transcript"] == expected, response
        assert response["usage"]["input_tokens"] == 1, response
        added = next(
            e
            for e in events
            if e["type"] == "conversation.item.added"
            and e["item"]["role"] == "assistant"
        )
        assert added["previous_item_id"] == first, added
        pcm = b"".join(
            base64.b64decode(e["delta"])
            for e in events
            if e["type"] == "response.output_audio.delta"
        )
        check_pcm(pcm, expected, 42)
        messages.extend(
            [
                {"role": "assistant", "content": expected},
                {"role": "user", "content": "A"},
            ]
        )
        client.respond(messages)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "recognition"})
        client.close()


def repeated_playback_report(address, trace):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        client.message("before-playback", "a")
        messages = [{"role": "user", "content": "a"}]
        response = client.respond(messages)
        item = response["output"][0]
        report = {"item_id": item["id"], "content_index": 0, "audio_end_ms": 0}
        client.send("conversation.item.truncate", **report)
        client.until("conversation.item.truncated")
        admin.request("/v1/models/drain", {"model": "fixture-text"})
        drained = True
        client.send("response.create")
        pending = client.until("response.created")[-1]["response"]
        # The already removed transcript cannot invalidate or cancel the new
        # response, even when it has not published an output item yet.
        client.send("conversation.item.truncate", **report)
        client.until("conversation.item.truncated")
        admin.request("/v1/models/undrain", {"model": "fixture-text"})
        drained = False
        events = client.until("response.done")
        response = events[-1]["response"]
        expected, finish = complete(messages, 2)
        assert response["id"] == pending["id"], response
        assert response["status"] == (
            "completed" if finish == "stop" else "incomplete"
        ), response
        assert response["output"][0]["content"][0]["transcript"] == expected, response
        pcm = b"".join(
            base64.b64decode(e["delta"])
            for e in events
            if e["type"] == "response.output_audio.delta"
        )
        check_pcm(pcm, expected, 42)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "fixture-text"})
        client.close()


def empty_response(address, trace):
    client = Client(address, trace)
    admin = HttpClient(address)
    drained = False
    try:
        assert client.receive()["type"] == "session.created"
        client.message("empty", "a" * 11)
        admin.request("/v1/models/drain", {"model": "speech"})
        drained = True
        client.send("response.create")
        events = client.until("response.done")
        response = events[-1]["response"]
        assert response["status"] == "completed", events
        assert response["usage"]["input_tokens"] == 11, response
        assert response["usage"]["output_tokens"] == 1, response
        assert response["output"] == [], response
        assert [e["type"] for e in events] == ["response.created", "response.done"], (
            events
        )
        admin.request("/v1/models/undrain", {"model": "speech"})
        drained = False
        # No assistant separator from an invented empty message may enter the
        # next native prompt or its compatible checkpoint.
        client.message("after-empty", "b")
        response = client.respond(
            [
                {"role": "user", "content": "a" * 11},
                {"role": "user", "content": "b"},
            ]
        )
        assert response["usage"]["input_tokens"] == 12, response
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        if drained:
            admin.request("/v1/models/undrain", {"model": "speech"})
        client.close()


def oversized_item_admission(address, trace):
    client = Client(address, trace)
    try:
        assert client.receive()["type"] == "session.created"
        client.message("before", "a")
        messages = [{"role": "user", "content": "a"}]
        for label, unit, width in (("ascii", "a", 1), ("escaped", "\0", 6)):
            item = {
                "id": label,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": ""}],
            }
            event_id = label[0]
            request = {
                "type": "conversation.item.create",
                "event_id": event_id,
                "previous_item_id": "root",
                "item": item,
            }
            overhead = len(json.dumps(request, ensure_ascii=False).encode())
            count = (512 * 1024 - overhead) // width
            item["content"][0]["text"] = unit * count
            payload = json.dumps(request, ensure_ascii=False).encode()
            assert len(payload) <= 512 * 1024
            client.send_frame(1, payload)
            error = client.receive(allow_error=True)
            assert error["type"] == "error", error
            assert error["error"]["code"] == "OUT_OF_RANGE", error
            assert error["error"]["event_id"] == event_id, error
            # Reusing the rejected identity and generating on this same socket
            # prove neither the item nor its oversized text entered history.
            client.message(label, "b")
            messages.append({"role": "user", "content": "b"})
        client.respond(messages)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


FUNCTION_TOOLS = [
    {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    for name, description in (
        ("z_lookup", "First function 🌲"),
        ("a_lookup", "Second function"),
    )
]


def function_oracle(messages, tools, maximum):
    # Independent dense recurrence, with the source's documented reversible
    # rendering: catalogue markers, content, call letters, assistant separator.
    prompt = "#" * len(tools)
    for message in messages:
        prompt += message.get("content") or ""
        prompt += "".join(
            json.loads(call["function"]["arguments"])["value"]
            for call in message.get("tool_calls", [])
        )
        if message["role"] == "assistant":
            prompt += "#"
    raw, finish = complete([{"role": "user", "content": prompt}], maximum)
    boundary = raw.find("x") if tools else -1
    if boundary < 0:
        return raw, [], finish
    calls = [
        {
            "name": tools[0]["name"],
            "arguments": json.dumps({"value": letter}, separators=(",", ":")),
        }
        for letter in raw[boundary:]
    ]
    return raw[:boundary], calls, "tool_calls"


def function_response(
    client, messages, tools, *, maximum=12, overrides=None, results=False
):
    expected, calls, finish = function_oracle(messages, tools, maximum)
    client.send(
        "response.create", response={"max_output_tokens": maximum, **(overrides or {})}
    )
    events = []
    outputs = []
    returned = []
    acknowledgements = set()
    response = None
    while response is None or (results and len(acknowledgements) != len(calls)):
        event = client.receive()
        events.append(event)
        if (
            event["type"] == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        ):
            call = event["item"]
            assert call["status"] == "completed", call
            ordinal = len(outputs)
            assert {key: call[key] for key in ("name", "arguments")} == calls[
                ordinal
            ], (call, calls)
            outputs.append(call)
            if results:
                # Send before response.done, just as Open WebUI executes each
                # completed call. Even empty result strings are canonical input.
                output = "a" if ordinal % 2 == 0 else ""
                identity = f"result_{ordinal}"
                client.send(
                    "conversation.item.create",
                    item={
                        "id": identity,
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": output,
                        "object": "realtime.item",
                        "status": "in_progress",
                    },
                )
                returned.append(
                    {"role": "tool", "tool_call_id": call["call_id"], "content": output}
                )
        elif (
            event["type"] == "conversation.item.done"
            and event["item"]["type"] == "function_call_output"
        ):
            acknowledgements.add(event["item"]["id"])
            assert event["item"]["status"] == "completed", event
        elif event["type"] == "response.done":
            response = event["response"]
    assert len(outputs) == len(calls), (outputs, calls)
    text = "".join(
        event["delta"]
        for event in events
        if event["type"]
        in ("response.output_text.delta", "response.output_audio_transcript.delta")
    )
    assert text == expected, (text, expected, events)
    audio = b"".join(
        base64.b64decode(event["delta"], validate=True)
        for event in events
        if event["type"] == "response.output_audio.delta"
    )
    audio_output = (overrides or {}).get("output_modalities", ["audio"]) == ["audio"]
    if audio_output and expected:
        check_pcm(audio, expected, 42, "positive")
    else:
        assert not audio
    if not expected:
        assert not any(
            "output_audio" in event["type"] or "content_part" in event["type"]
            for event in events
        ), events
    status = "incomplete" if finish == "length" else "completed"
    assert response["status"] == status, response
    assert (
        response["output"]
        == (
            [
                next(
                    event["item"]
                    for event in events
                    if event["type"] == "response.output_item.done"
                    and event["item"]["type"] == "message"
                )
            ]
            if expected
            else []
        )
        + outputs
    ), response
    for index, call in enumerate(outputs, bool(expected)):
        selected = [
            event
            for event in events
            if event.get("item_id", event.get("item", {}).get("id")) == call["id"]
        ]
        kinds = [event["type"] for event in selected]
        assert kinds == [
            "conversation.item.added",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "conversation.item.done",
        ], selected
        for event in selected:
            if event["type"].startswith("response."):
                assert event["output_index"] == index, event
        assert (
            selected[0]["item"]["arguments"] == selected[1]["item"]["arguments"] == ""
        )
        assert selected[2]["delta"] == selected[3]["arguments"] == call["arguments"]
    message = {"role": "assistant", "content": expected}
    if outputs:
        message["tool_calls"] = [
            {
                "id": call["call_id"],
                "type": "function",
                "function": {key: call[key] for key in ("name", "arguments")},
            }
            for call in outputs
        ]
    return response, message, returned


def function_transactions(address, trace, *, content):
    client = Client(address, trace)
    try:
        client.receive()
        client.send(
            "session.update", session={"tools": FUNCTION_TOOLS, "tool_choice": "auto"}
        )
        settings = client.until("session.updated")[-1]["session"]
        assert (
            settings["tools"] == FUNCTION_TOOLS and settings["tool_choice"] == "auto"
        ), settings
        client.message("prompt", content)
        messages = [{"role": "user", "content": content}]
        expected, calls, _ = function_oracle(messages, FUNCTION_TOOLS, 12)
        model = next(
            item["loom"]
            for item in HttpClient(address).request("/v1/status")["data"]
            if item["id"] == "fixture-text"
        )
        assert "error" not in model, model
        try:
            native = HttpClient(address).request(
                "/v1/chat/completions",
                {
                    "model": "fixture-text",
                    "messages": messages,
                    "max_tokens": 12,
                    "seed": 42,
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                key: value
                                for key, value in tool.items()
                                if key != "type"
                            },
                        }
                        for tool in FUNCTION_TOOLS
                    ],
                },
            )
        except AssertionError as error:
            error.add_note(json.dumps(HttpClient(address).request("/v1/status")))
            raise
        choice = native["choices"][0]
        assert choice["finish_reason"] == "tool_calls", native
        assert (choice["message"].get("content") or "") == expected, native
        assert [
            call["function"] for call in choice["message"]["tool_calls"]
        ] == calls, native
        metadata = {"input_item_id": "prompt", "operation": "lookup 🌲"}
        overrides = {"metadata": metadata}
        if not expected:
            # A deliberately nonexistent voice proves tool-only output does not
            # acquire a speech request, independently of emitted audio events.
            overrides["audio"] = {"output": {"voice": "missing-tool-only-voice"}}
        response, message, returned = function_response(
            client, messages, FUNCTION_TOOLS, overrides=overrides, results=True
        )
        assert response["metadata"] == metadata
        messages += [message, *returned]
        spoken, _, _ = function_response(
            client, messages, [], maximum=3, overrides={"tool_choice": "none"}
        )
        assert spoken["usage"]["input_token_details"]["cached_tokens"] == 0, spoken
        # The override did not mutate the default catalogue or the preceding
        # compatible checkpoint. Subsequent canonical history includes results.
        messages.append(
            {
                "role": "assistant",
                "content": spoken["output"][0]["content"][0]["transcript"],
            }
        )
        resumed, _, _ = function_response(client, messages, FUNCTION_TOOLS, maximum=2)
        assert resumed["usage"]["input_token_details"]["cached_tokens"] > 0, resumed
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def function_item_edits(address, trace):
    client = Client(address, trace)
    try:
        client.receive()
        # Unknown calls reject without publishing or reserving the item ID.
        result = {
            "id": "result",
            "type": "function_call_output",
            "call_id": "unknown",
            "output": "",
        }
        client.send("conversation.item.create", item=result, event_id="unknown")
        error = client.receive(allow_error=True)
        assert error["type"] == "error" and error["error"]["event_id"] == "unknown", (
            error
        )
        client.send(
            "conversation.item.create",
            item={
                "id": "call",
                "type": "function_call",
                "name": "z_lookup",
                "arguments": '{"value":"x"}',
                "status": "incomplete",
                "object": "realtime.item",
            },
        )
        call = client.until("conversation.item.done")[-1]["item"]
        assert call["status"] == "completed" and call["call_id"], call
        result["call_id"] = call["call_id"]
        client.send(
            "conversation.item.create",
            item=result,
            previous_item_id="root",
            event_id="before-call",
        )
        error = client.receive(allow_error=True)
        assert (
            error["type"] == "error" and error["error"]["event_id"] == "before-call"
        ), error
        client.send("conversation.item.create", item=result)
        assert client.until("conversation.item.done")[-1]["item"]["output"] == ""
        duplicate = {**call, "id": "duplicate"}
        client.send("conversation.item.create", item=duplicate, event_id="duplicate")
        error = client.receive(allow_error=True)
        assert error["type"] == "error" and error["error"]["event_id"] == "duplicate", (
            error
        )
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": call["call_id"],
                        "function": {"name": "z_lookup", "arguments": '{"value":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call["call_id"], "content": ""},
        ]
        function_response(client, messages, [], maximum=2)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def function_policy_admission(address, trace):
    client = Client(address, trace)
    try:
        client.receive()
        client.send("session.update", session={"tools": FUNCTION_TOOLS})
        settings = client.until("session.updated")[-1]["session"]
        rejected = [
            {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "name": "z_lookup"}},
            {"tools": [{**FUNCTION_TOOLS[0], "strict": True}]},
            {"tools": [FUNCTION_TOOLS[0], FUNCTION_TOOLS[0]]},
            {"tools": [{**FUNCTION_TOOLS[0], "parameters": {"type": "string"}}]},
        ]
        for index, policy in enumerate(rejected):
            identity = f"rejected-policy-{index}"
            client.send("session.update", session=policy, event_id=identity)
            error = client.receive(allow_error=True)
            assert (
                error["type"] == "error" and error["error"]["event_id"] == identity
            ), error
            client.send("session.update", session={})
            assert client.until("session.updated")[-1]["session"] == settings

        # The inbound command fits, but its full effective-settings echo does
        # not. Rejecting the echo must also discard the candidate catalogue.
        tool = {**FUNCTION_TOOLS[0], "description": ""}
        request = {
            "type": "session.update",
            "event_id": "large-policy",
            "session": {"tools": [tool]},
        }

        def encode():
            return json.dumps(
                request, ensure_ascii=False, separators=(",", ":")
            ).encode()

        tool["description"] = "a" * (512 * 1024 - len(encode()))
        payload = encode()
        assert len(payload) == 512 * 1024
        client.send_frame(1, payload)
        error = client.receive(allow_error=True)
        assert error["type"] == "error" and error["error"]["code"] == "OUT_OF_RANGE", (
            error
        )
        assert error["error"]["event_id"] == "large-policy", error
        client.send("session.update", session={})
        assert client.until("session.updated")[-1]["session"] == settings
        client.message("prompt", "a")
        function_response(client, [{"role": "user", "content": "a"}], FUNCTION_TOOLS)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def function_capacity(address, trace):
    client = Client(address, trace)
    try:
        client.receive()
        client.send("session.update", session={"tools": FUNCTION_TOOLS})
        client.until("session.updated")
        # Leave room for the first two of eight generated calls. The complete
        # set must be admitted before any call becomes externally executable.
        for index in range(125):
            client.message(f"empty-{index}", "")
        client.message("prompt", "a")
        client.send("response.create", response={"max_output_tokens": 12})
        events = client.until("response.done")
        assert [event["type"] for event in events] == [
            "response.created",
            "response.done",
        ], events
        response = events[-1]["response"]
        assert response["status"] == "failed" and response["output"] == [], response
        assert response["status_details"]["error"]["code"] == "RESOURCE_EXHAUSTED", (
            response
        )
        for index in range(125):
            client.send("conversation.item.delete", item_id=f"empty-{index}")
            client.until("conversation.item.deleted")
        # No partial calls, assistant separators, or leaked capacity from the
        # failed response may enter the next native prompt.
        function_response(client, [{"role": "user", "content": "a"}], FUNCTION_TOOLS)
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def function_group_edits(address, trace):
    client = Client(address, trace)
    try:
        client.receive()
        client.send("session.update", session={"tools": FUNCTION_TOOLS})
        client.until("session.updated")
        client.message("prompt", "a")
        messages = [{"role": "user", "content": "a"}]
        response, message, _ = function_response(client, messages, FUNCTION_TOOLS)
        first, removed = response["output"][:2]
        client.send(
            "conversation.item.create",
            previous_item_id=first["id"],
            item={
                "type": "function_call_output",
                "call_id": first["call_id"],
                "output": "a",
            },
        )
        client.until("conversation.item.done")
        client.send("conversation.item.delete", item_id=removed["id"])
        client.until("conversation.item.deleted")
        # A result inserted between generated members splits the native
        # assistant message. Deletion removes only its selected call.
        messages += [
            {**message, "tool_calls": message["tool_calls"][:1]},
            {"role": "tool", "tool_call_id": first["call_id"], "content": "a"},
            {**message, "tool_calls": message["tool_calls"][2:]},
        ]
        response, _, _ = function_response(
            client, messages, [], maximum=2, overrides={"tool_choice": "none"}
        )
        assert response["usage"]["input_token_details"]["cached_tokens"] == 0, response
        client.send_frame(8, struct.pack(">H", 1000))
        assert client.read_frame() == (8, struct.pack(">H", 1000))
    finally:
        client.close()


def main():
    executable = str(Path(sys.argv[1]).resolve())
    fixtures = [Path(value).resolve() for value in sys.argv[2:5]]
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    artifacts = Path(
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
        if "TEST_UNDECLARED_OUTPUTS_DIR" in os.environ
        else tempfile.mkdtemp()
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        catalog, flags = prepare_catalog(fixtures, directory)
        text = next(m for m in catalog["models"] if m["definition"]["kind"] == "text")
        text["definition"].update(session_cache_capacity=8, checkpoint_capacity=8)
        write_startup(flags, catalog)
        profile = directory / "assistant.json"
        profile.write_text(
            json.dumps(
                {
                    "name": "assistant",
                    "text": "fixture-text",
                    "recognition": "recognition",
                    "activity": "recognition",
                    "speech": "speech",
                    "voice": "positive",
                    "max_output_tokens": 2,
                    "seed": 42,
                }
            )
        )
        bounded = directory / "bounded.json"
        bounded.write_text(
            json.dumps(
                {
                    **json.loads(profile.read_text()),
                    "name": "bounded",
                    "input_capacity": 4096,
                }
            )
        )
        command = [
            executable,
            f"--flagfile={flags}",
            *sys.argv[6:],
            f"--workers={workers}",
            "--listen=127.0.0.1:0",
            "--heartbeat_ms=0",
            f"--realtime_assistant={profile}",
            f"--realtime_assistant={bounded}",
        ]
        with running_server(command, artifacts / "server.log") as (process, address):
            assert address is not None, process.poll()
            with (artifacts / "wire.jsonl").open("w") as trace:
                independent_contexts(address, trace)
                independent_pending_input(address, trace)
                independent_function_input(address, trace)
                independent_admission(address, trace)
                independent_cancel_scope(address, trace)
                function_transactions(address, trace, content="a")
                function_transactions(address, trace, content="b")
                function_item_edits(address, trace)
                function_policy_admission(address, trace)
                function_capacity(address, trace)
                function_group_edits(address, trace)
                anonymous_item(address, trace)
                reserved_item_identity(address, trace)
                deferred_response_frontier(address, trace)
                repeated_playback_report(address, trace)
                empty_response(address, trace)
                oversized_item_admission(address, trace)
                response_policy(address, trace)
                conversation(address, trace)
                response_failure(address, trace, speech=True)
                response_failure(address, trace, speech=False)
                cancellation(address, trace)
                pipelined_microphone_items(address, trace)
                packed = automatic_microphone(
                    address, trace, partitioned=False, create_response=False
                )
                partitioned = automatic_microphone(
                    address, trace, partitioned=True, create_response=False
                )
                assert packed == partitioned, (packed, partitioned)
                automatic_microphone(
                    address, trace, partitioned=True, create_response=True
                )
                automatic_explicit_boundaries(address, trace)
                automatic_response_policy(address, trace, interrupt=True)
                automatic_response_policy(address, trace, interrupt=False)
                rejected_activity_geometry(address, trace)
        text["definition"].update(session_cache_capacity=0, checkpoint_capacity=0)
        write_startup(flags, catalog)
        with running_server(command, artifacts / "rejection-server.log") as (
            process,
            address,
        ):
            assert address is not None, process.poll()
            with (artifacts / "rejection-wire.jsonl").open("w") as trace:
                rejected_capture(address, trace)

        recognition = next(m for m in catalog["models"] if m["name"] == "recognition")
        catalog["models"].append({**recognition, "name": "activity"})
        text["definition"].update(session_cache_capacity=8, checkpoint_capacity=8)
        write_startup(flags, catalog)
        profile.write_text(
            json.dumps({**json.loads(profile.read_text()), "activity": "activity"})
        )
        with running_server(command, artifacts / "independent-activity-server.log") as (
            process,
            address,
        ):
            assert address is not None, process.poll()
            with (artifacts / "independent-activity-wire.jsonl").open("w") as trace:
                independent_automatic_intent(address, trace)
                registration_replacement(address, trace, catalog["models"])


if __name__ == "__main__":
    main()
