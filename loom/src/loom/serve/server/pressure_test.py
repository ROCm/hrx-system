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
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http import control, running_server, wait_status


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

    for character in text:
        consume("abx#".index(character))
    result = ""
    for _ in range(2):
        if position % 11 == 0:
            break
        token = (hidden + total) % 3
        result += "abx"[token]
        consume(token)
    return result


def check(server, source, directory, workers, memory_bytes, retention):
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
        document = directory / f"{name}.json"
        document.write_text(json.dumps(model))
        flags.append(f"--model={document.resolve()}")
    configuration.write_text("\n".join(flags) + "\n")
    command = [
        str(server),
        "--device=amdgpu",
        "--pool_backing=elastic",
        "--slab_bytes=2097152",
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
                assert state["loom"]["parameter_pinned"], state
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
        expected = 503 if retention != "none" and memory_bytes == 4194304 else 200
        assert second_status == expected, second
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
            # Weight pinning is not a session pin: capacity pressure can discard
            # all idle histories while the parameters remain resident.
            assert bool(sessions["data"]) == (memory_bytes == 8388608), sessions
            for _ in range(2):
                unpinned = control(
                    address,
                    "POST",
                    "/v1/models/unpin",
                    200,
                    body=json.dumps({"model": "alpha"}),
                    headers={"Content-Type": "application/json"},
                )["loom"]
                assert not unpinned["parameter_pinned"], unpinned
                assert unpinned["parameters"] == alpha["parameters"], unpinned
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
            assert cached_tokens(resumed) == (0 if memory_bytes == 4194304 else 5), (
                resumed
            )
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
        # Joined removal retires cache callbacks as well as physical storage.
        # The shared server and its pool remain available afterward.
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
            if not final["data"]:
                break
            assert all(not model["loom"].get("error") for model in final["data"]), final
        memory = final["server"]["resources"]["devices"][0]["memory"]["retained"]
        assert memory["reserved_bytes"] == memory["committed_bytes"] == 0, memory
        assert memory["peak_bytes"] <= memory_bytes, memory
        assert control(address, "GET", "/readyz", 200) == {"status": "ready"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    server, source = arguments.server.absolute(), arguments.source.absolute().parent
    failures = []
    with tempfile.TemporaryDirectory(prefix="loom pressure ") as temporary:
        root = Path(temporary)
        for workers in (1, 4):
            for memory_bytes in (8388608, 4194304):
                for retention in ("none", "checkpoint", "parameters"):
                    directory = root / f"{workers}-{memory_bytes}-{retention}"
                    directory.mkdir()
                    try:
                        check(
                            server, source, directory, workers, memory_bytes, retention
                        )
                    except AssertionError as error:
                        failures.append((workers, memory_bytes, retention, str(error)))
    assert not failures, failures


if __name__ == "__main__":
    main()
