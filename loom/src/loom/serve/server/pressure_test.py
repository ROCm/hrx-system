# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared physical pressure through real source-JIT HTTP model ownership."""

import argparse
import http.client
import json
import struct
import tempfile
import traceback
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import control, running_server, wait_status

# The fixture owns a native page of parameters and an 8 KiB state reservation.
# Limits express one or two complete models, without coarse-slab tail padding.
MODEL_BYTES = 12 * 1024


def completion(address, model, messages, session=None, *, checkpoint=None):
    connection = http.client.HTTPConnection(address.hostname, address.port)
    headers = {"Content-Type": "application/json"}
    if session is not None:
        headers["X-Loom-Session"] = session
    if checkpoint is not None:
        headers["X-Loom-Checkpoint"] = checkpoint
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 2,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ),
            headers=headers,
        )
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def records(payload):
    result = [
        json.loads(line[6:])
        for line in payload.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert "data: [DONE]" in payload, payload
    return result


def output(payload):
    return "".join(
        choice.get("delta", {}).get("content", "")
        for record in records(payload)
        for choice in record.get("choices", [])
    )


def cached_tokens(payload):
    return next(
        record["usage"]["prompt_tokens_details"]["cached_tokens"]
        for record in records(payload)
        if record.get("usage")
    )


def oracle(text):
    hidden = history = total = position = 0

    def consume(token):
        nonlocal hidden, history, total, position
        hidden = (5 * hidden + token + 2) % 97
        history = (7 * history + hidden + position) % 101
        total += history
        position += 1

    tokens = (
        ("abx#".index(character) for character in text)
        if isinstance(text, str)
        else text
    )
    for token in tokens:
        consume(token)
    result = ""
    for _ in range(2):
        if position % 11 == 0:
            break
        token = (hidden + total) % 3
        result += "abx"[token]
        consume(token)
    return result


def check_media(address, memory_bytes):
    pam = (
        "UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUg"
        "UkdCCkVOREhEUgr/AAAA/wAAAP////8="
    )
    image = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/x-portable-arbitrarymap;base64," + pam,
            "detail": "low",
        },
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                image,
                image,
                {"type": "text", "text": "b"},
            ],
        }
    ]
    status, response = completion(address, "alpha", messages)
    expected_status = 503 if memory_bytes == MODEL_BYTES else 200
    assert status == expected_status, response
    if status == 200:
        # batch_units=1 requires two independently retained producer roots.
        # The actual projector fingerprints both pixels and cohort width.
        feature = struct.unpack("<I", struct.pack("<f", 511.0))[0]
        assert output(response) == oracle([0, feature, feature, 1]), response
    else:
        assert "image producer or language continuation" in response, response
        state = control(address, "GET", "/v1/status", 200)
        alpha = next(item["loom"] for item in state["data"] if item["id"] == "alpha")
        for component in ("primary", "image"):
            parameters = alpha["parameters"][component]
            assert not parameters.get("active", False), parameters
            assert parameters.get("activations", 0) == 0, parameters
    print(json.dumps({"media_limit": memory_bytes, "media_status": status}), flush=True)


def check(server, source, directory, device, workers, memory_bytes, retention):
    configuration = directory / "models.flags"
    weights = directory / "weights.safetensors"
    header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
    header += b" " * (-len(header) % 8)
    weights.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<I", 2))
    flags = []
    for name in ("alpha", "beta"):
        model = {
            "name": name,
            "base_directory": str(source.resolve()),
            "definition": {
                **json.loads((source / "model.json").read_text())["definition"],
                "weights": str(weights),
                "tokenizer": str(source / "tokenizer.json"),
                "rows": 2,
                "context_capacity": 128,
                "pool_capacity": 128,
                "prefill_capacity": 32,
                "checkpoint_capacity": 2,
                "session_cache_capacity": 2,
                "max_tokens": 2,
                "draft_depth": 0,
            },
        }
        if retention == "none" and name == "alpha":
            projector = source.parent / "projector"
            model["definition"]["sources"].append(str(projector / "project.loom"))
            model["definition"]["image"] = {
                "prepare": str(projector / "prepare.loom"),
                "control": str(projector / "control.loom"),
                "assets": str(projector),
                "options": {"batch_units": 1},
            }
        document = directory / f"{name}.json"
        document.write_text(json.dumps(model))
        flags.append(f"--model={document.resolve()}")
    configuration.write_text("\n".join(flags) + "\n")
    command = [
        str(server),
        f"--device={device}",
        "--pool_backing=elastic",
        "--weight_slab_bytes=4096",
        "--state_slab_bytes=8192",
        f"--memory_bytes={memory_bytes}",
        f"--workers={workers}",
        f"--flagfile={configuration}",
        "--listen=127.0.0.1:0",
        "--heartbeat_ms=0",
    ]
    with running_server(command, directory / "server.log") as (_, endpoint):
        assert endpoint is not None, (directory / "server.log").read_text()
        address = urlsplit("http://" + endpoint)
        while True:
            models = control(address, "GET", "/v1/status", 200)["data"]
            assert all(not model["loom"].get("error") for model in models), models
            if len(models) == 2 and all(
                model["loom"]["status"]["phase"] == "idle" for model in models
            ):
                break
        messages = [{"role": "user", "content": "abxa"}]
        if retention == "none":
            check_media(address, memory_bytes)
        first = ""
        for index in range(4):
            first_status, first = completion(
                address, "alpha", messages, f"conversation-{index}"
            )
            assert first_status == 200, first
            assert output(first) == oracle("abxa"), first
        wait_status(address, model="alpha", active_requests=0)
        if retention == "parameters":
            for _ in range(2):
                state = control(
                    address,
                    "POST",
                    "/v1/models/pin",
                    200,
                    body=json.dumps({"model": "alpha"}),
                    headers={"Content-Type": "application/json"},
                )
                assert state["loom"]["parameters"]["primary"]["pinned"], state
        elif retention == "checkpoint":
            # The oldest history has detached from its execution row. The
            # explicit pin shares its native state, not a second byte copy.
            control(
                address,
                "POST",
                "/v1/checkpoints/kept",
                201,
                session="conversation-0",
                headers={"X-Loom-Model": "alpha"},
            )
        before = control(address, "GET", "/v1/status", 200)
        second_status, second = completion(address, "beta", messages, "peer")
        after = control(address, "GET", "/v1/status", 200)
        memory_before = before["server"]["resources"]["devices"][0]["memory"][
            "retained"
        ]
        memory_after = after["server"]["resources"]["devices"][0]["memory"]["retained"]
        result = {
            "workers": workers,
            "memory_bytes": memory_bytes,
            "before": memory_before,
            "after": memory_after,
            "peer_status": second_status,
            "retention": retention,
        }
        print(json.dumps(result), flush=True)
        assert memory_after["peak_bytes"] <= memory_bytes, result
        expected = 503 if retention != "none" and memory_bytes == MODEL_BYTES else 200
        assert second_status == expected, second
        beta = next(item["loom"] for item in after["data"] if item["id"] == "beta")
        admission = beta["parameters"]["primary"]["admission"]
        assert not admission["waiting"] and admission["age_seconds"] == 0, beta
        assert beta["parameters"]["primary"]["pins"] == 0, beta
        if expected == 503:
            assert admission["count"] >= 1, beta
            # Complete-unit admission rejects before loading parameters that
            # would leave no room for the request's state.
            parameters = beta["parameters"]["primary"]
            assert not parameters["active"], beta
            assert parameters["activations"] == 0, beta
            assert parameters["committed_bytes"] == parameters["pending_bytes"] == 0, (
                beta
            )
        if expected == 503 and retention == "checkpoint":
            alpha = next(
                item["loom"] for item in after["data"] if item["id"] == "alpha"
            )
            alpha_before = next(
                item["loom"] for item in before["data"] if item["id"] == "alpha"
            )
            # Evicting these weights cannot admit the peer while the checkpoint
            # retains state. Keep them for the checkpoint's next invocation.
            assert alpha["parameters"] == alpha_before["parameters"], alpha
        assert admission["duration_seconds"] >= 0, beta
        connection = http.client.HTTPConnection(address.hostname, address.port)
        try:
            connection.request("GET", "/metrics")
            response = connection.getresponse()
            metrics = response.read().decode()
            assert response.status == 200, metrics
        finally:
            connection.close()
        identity = (
            f'model="beta",kind="text",incarnation="{beta["incarnation"]}",'
            'component="primary"'
        )
        assert (
            f"loom_serve_model_parameter_admission_waits_total{{{identity}}} "
            f"{admission['count']}\n"
        ) in metrics, metrics
        assert (
            f"loom_serve_model_parameter_admission_waiting{{{identity}}} 0\n"
        ) in metrics, metrics
        if expected == 200:
            assert output(second) == oracle("abxa"), second
        history = [
            *messages,
            {"role": "assistant", "content": output(first)},
            {"role": "user", "content": "b"},
        ]
        if retention == "parameters":
            alpha = next(
                item["loom"] for item in after["data"] if item["id"] == "alpha"
            )
            alpha_before = next(
                item["loom"] for item in before["data"] if item["id"] == "alpha"
            )
            assert alpha["parameters"] == alpha_before["parameters"], alpha
            sessions = control(
                address,
                "GET",
                "/v1/sessions",
                200,
                headers={"X-Loom-Model": "alpha"},
            )
            # Pinned weights plus the peer's complete requirement already
            # exceed the small cap. Evicting history cannot make it fit, so
            # admission preserves those histories instead of discarding them
            # to partially activate an impossible peer.
            assert {session["name"] for session in sessions["data"]} == {
                f"conversation-{index}" for index in range(4)
            }, sessions
            for _ in range(2):
                unpinned = control(
                    address,
                    "POST",
                    "/v1/models/unpin",
                    200,
                    body=json.dumps({"model": "alpha"}),
                    headers={"Content-Type": "application/json"},
                )["loom"]
                assert not unpinned["parameters"]["primary"]["pinned"], unpinned
                assert unpinned["parameters"]["primary"] == {
                    **alpha["parameters"]["primary"],
                    "pinned": False,
                }, unpinned
                assert unpinned["incarnation"] == alpha["incarnation"], unpinned
            second_status, second = completion(address, "beta", messages, "peer")
            assert second_status == 200, second
            assert output(second) == oracle("abxa"), second
        if retention == "checkpoint":
            # A pin must survive failed peer pressure even after all automatic
            # owners release their references. Deletion then unblocks the peer.
            resumed_status, resumed = completion(
                address, "alpha", history, checkpoint="kept"
            )
            assert resumed_status == 200, resumed
            assert output(resumed) == oracle("abxa" + output(first) + "#b"), resumed
            assert cached_tokens(resumed) == 5, resumed
            control(
                address,
                "DELETE",
                "/v1/checkpoints/kept",
                200,
                headers={"X-Loom-Model": "alpha"},
            )
            second_status, second = completion(address, "beta", messages, "peer")
            assert second_status == 200, second
            assert output(second) == oracle("abxa"), second
        else:
            resumed_status, resumed = completion(
                address, "alpha", history, "conversation-0"
            )
            assert resumed_status == 200, resumed
            assert output(resumed) == oracle("abxa" + output(first) + "#b"), resumed
            assert cached_tokens(resumed) == (
                0 if memory_bytes == MODEL_BYTES else 5
            ), resumed
        # Removal owns pinned state as well as the service. A peer submitted
        # immediately after accepted unregister must complete without waiting
        # for a client-side polling loop to observe final model destruction.
        status, response = completion(address, "alpha", messages, "removing")
        assert status == 200, response
        assert output(response) == oracle("abxa"), response
        control(
            address,
            "POST",
            "/v1/checkpoints/removal",
            201,
            session="removing",
            headers={"X-Loom-Model": "alpha"},
        )
        control(
            address,
            "POST",
            "/v1/models/pin",
            200,
            body=json.dumps({"model": "alpha"}),
            headers={"Content-Type": "application/json"},
        )
        control(
            address,
            "POST",
            "/v1/models/unregister",
            202,
            body=json.dumps({"model": "alpha", "mode": "drain"}),
            headers={"Content-Type": "application/json"},
        )
        status, response = completion(address, "beta", messages, "replacement")
        assert status == 200, response
        assert output(response) == oracle("abxa"), response
        # Joined removal returns backing to the shared cache. Explicit trim
        # returns that idle physical storage while keeping the server usable.
        control(
            address,
            "POST",
            "/v1/models/unregister",
            202,
            body=json.dumps({"model": "beta", "mode": "drain"}),
            headers={"Content-Type": "application/json"},
        )
        while True:
            final = control(address, "GET", "/v1/status", 200)
            memory = final["server"]["resources"]["devices"][0]["memory"]["retained"]
            if (
                not final["data"]
                and memory["committed_bytes"] == memory["cached_bytes"]
                and not memory["releasing_bytes"]
            ):
                break
            assert all(not model["loom"].get("error") for model in final["data"]), final
        assert memory["pending_bytes"] == 0, memory
        # Model removal leaves the shared workspace's reusable address ranges.
        # Physical trim must preserve those addresses while returning backing.
        workspace_virtual_bytes = memory["reserved_bytes"]
        assert workspace_virtual_bytes > 0, memory
        control(
            address,
            "POST",
            "/v1/devices/trim",
            200,
            body=json.dumps({"device": 0, "target_bytes": 0}),
            headers={"Content-Type": "application/json"},
        )
        while memory["committed_bytes"]:
            final = control(address, "GET", "/v1/status", 200)
            memory = final["server"]["resources"]["devices"][0]["memory"]["retained"]
        assert memory["reserved_bytes"] == workspace_virtual_bytes, memory
        assert memory["committed_bytes"] == 0, memory
        assert memory["peak_bytes"] <= memory_bytes, memory
        assert control(address, "GET", "/readyz", 200) == {"status": "ready"}


def run_from_command_line(cases):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", required=True)
    arguments = parser.parse_args()
    server, source = arguments.server.absolute(), arguments.source.absolute().parent
    failures = []
    with tempfile.TemporaryDirectory(prefix="loom pressure ") as temporary:
        root = Path(temporary)
        for workers, memory_bytes, retention in cases:
            directory = root / f"{workers}-{memory_bytes}-{retention}"
            directory.mkdir()
            try:
                check(
                    server,
                    source,
                    directory,
                    arguments.device,
                    workers,
                    memory_bytes,
                    retention,
                )
            except AssertionError:
                failures.append(
                    (workers, memory_bytes, retention, traceback.format_exc())
                )
    assert not failures, failures


# Pairwise coverage across worker count, physical memory limit, and retained
# state policy. The manual pressure_stress_test target retains the complete
# Cartesian product for explicit stress qualification.
_CASES = (
    # Retain both state slabs while swapping weights; media also fits as
    # sequential phases without requiring their combined declared footprint.
    (1, 20 * 1024, "none"),
    (1, MODEL_BYTES, "checkpoint"),
    (1, 2 * MODEL_BYTES, "parameters"),
    (4, MODEL_BYTES, "none"),
    (4, 2 * MODEL_BYTES, "checkpoint"),
    (4, MODEL_BYTES, "parameters"),
)


if __name__ == "__main__":
    run_from_command_line(_CASES)
