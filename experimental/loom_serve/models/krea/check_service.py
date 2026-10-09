# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify the real retained image HTTP service, including ownership failures.

Requires the base short/long and adapted short isolated-0/1/2.ppm images
produced by check_model.py, the same checkpoint/source geometry, and an
exclusive device lease. The production server performs every image operation
natively. Pillow independently decodes returned PNGs for exact comparison with
isolated CLI pixels.
Current owner status/counters synchronize the client without GPU wait deadlines.
Native retained history proves coverage and fails explicitly on observation gaps.
Final images and concise logs stay below 16 MiB; no checkpoint copies.
"""

import argparse
import base64
import http.client
import io
import json
import socket
import struct
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from experimental.loom_serve.testing.http import (
    Events,
    running_server,
    wait_counters,
    wait_status,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", nargs="+", required=True)
    parser.add_argument(
        "--workers", type=int, default=1, help="Shared JIT/serving CPU workers."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--isolated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = []
    for index in range(3):
        with Image.open(args.isolated / f"isolated-{index}.ppm") as image:
            assert image.size == (384, 384) and image.mode == "RGB"
            expected.append(image.tobytes())
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        *args.server,
        f"--workers={args.workers}",
        f"--model={args.model}",
        f"--checkpoint={args.checkpoint}",
        f"--adapter={args.adapter}",
        "--width=384",
        "--height=384",
        "--text_tokens=512",
        "--port=0",
        "--pending_requests=1",
        "--connections=16",
        "--heartbeat_ms=100",
    ]
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    events = Events()
    with (
        ExitStack() as peers,
        running_server(command, args.output / "server.log", events.record) as (
            _,
            ready,
        ),
    ):
        if ready is None:
            raise RuntimeError("image server did not reach readiness")
        address = urlsplit(f"http://{ready}")
        print(json.dumps(dict(event="ready", address=ready)), flush=True)

        def send(body=None, path="/v1/images/generations", slow=False):
            peer = peers.enter_context(socket.socket())
            if slow:
                peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            peer.connect((address.hostname, address.port))
            method = "GET" if body is None else "POST"
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            body = body or b""
            header = (
                f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            peer.sendall(header + body)
            return peer

        def receive(peer, code=200):
            with http.client.HTTPResponse(peer) as response:
                response.begin()
                value = json.loads(response.read())
                assert response.status == code, (response.status, value)
            peer.close()
            return value

        def reset(peer):
            peer.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            peer.close()

        def pixels(peer, name, baseline):
            value = receive(peer)
            assert isinstance(value["created"], int)
            assert len(value["data"]) == 1
            png = base64.b64decode(value["data"][0]["b64_json"], validate=True)
            (args.output / f"{name}.png").write_bytes(png)
            with Image.open(io.BytesIO(png)) as image:
                assert image.format == "PNG" and image.mode == "RGB"
                assert image.size == (384, 384)
                assert image.tobytes() == expected[baseline], name
            print(json.dumps(dict(event="exact_http_pixels", case=name)), flush=True)

        models = receive(send(path="/v1/models"))
        assert len(models["data"]) == 1
        assert models["data"][0]["id"] == "krea2-turbo"
        for body in (
            b"{}",
            b'{"model":"krea2-turbo","prompt":"x","prompt":"y"}',
            b'{"model":"krea2-turbo","prompt":"x","strength":1e100}',
            b'{"model":"krea2-turbo","prompt":"x","seed":18446744073709551616}',
            b'{"model":"krea2-turbo","prompt":"x","n":2}',
            b'{"model":"krea2-turbo","prompt":"x","size":"512x512"}',
            b'{"model":"krea2-turbo","prompt":"x","reference_images":[]}',
            b'{"model":"krea2-turbo","prompt":"\xff"}',
        ):
            assert "error" in receive(send(body), 400)
        receive(send(path="/unknown"), 404)
        health = receive(send(path="/healthz"))
        assert health == {"status": "ready"}
        initial = wait_counters(address, lambda _: True)
        # Rejected model-protocol requests consume serials too. Start from the
        # actual owner count, not a generation-only numbering convention.
        first_id = initial["requests"] + 1

        short = dict(
            prompt="a red fox in the snow",
            seed="0",
            strength=0,
            model="krea2-turbo",
            size="384x384",
            n=1,
            response_format="b64_json",
        )
        long = dict(short, prompt=" ".join(["red"] * 65), seed=42)
        first = send(short, slow=True)
        wait_status(address, active_requests=1, queued_requests=0, phase="generating")
        health = receive(send(path="/healthz"))
        assert health == {"status": "ready"}
        queued = send(long)
        wait_status(address, active_requests=1, queued_requests=1)
        health = receive(send(path="/healthz"))
        assert health == {"status": "ready"}
        receive(send(short), 503)
        reset(queued)
        wait_counters(address, lambda counts: counts["cancelled"] == 1)
        replacement = send(long)
        wait_status(address, active_requests=1, queued_requests=1)
        # Neither socket has consumed response bytes. In particular, the first
        # peer's deliberately small receive window cannot pin the model output.
        pixels(replacement, "queued-long", 1)
        pixels(first, "slow-short", 0)

        style = send(dict(short, strength=1))
        pixels(style, "style", 2)
        repeated = send(short)
        pixels(repeated, "base-after-style", 0)

        abandoned = send(long)
        wait_status(address, active_requests=1, queued_requests=0, phase="generating")
        reset(abandoned)
        after_reset = send(short)
        # Cancellation retires only after the accepted device/CPU work. The
        # next request must queue while that work still owns its invocation.
        wait_status(address, active_requests=1, queued_requests=1)
        health = receive(send(path="/healthz"))
        assert health == {"status": "ready"}
        pixels(after_reset, "base-after-reset", 0)
        counts = wait_counters(address, lambda counts: counts["completed"] == 5)
        assert counts["requests"] == initial["requests"] + 8
        assert counts["rejected"] == initial["rejected"] + 1
        assert counts["cancelled"] == 2 and counts["failed"] == 0
        assert counts["batches"] == 6

        send(dict(short, strength=1))
        wait_status(address, active_requests=1, queued_requests=0, phase="generating")
        # The harness captures a fixed history head, then signals and joins the
        # child. A pull endpoint cannot promise post-shutdown observations.
    text = (args.output / "server.log").read_text()
    cold = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    assert text.count('"event":"jit_stage"') == 2
    residencies = [event for event in cold if event["event"] == "image_residency"]
    assert len(residencies) == 1
    residency = residencies[0]
    assert [stage["tag"] for stage in residency["stages"]] == [512, 128]
    assert residency["workspace_bytes"] == max(
        stage["workspace_bytes"] for stage in residency["stages"]
    )
    assert residency["workspace_alignment"] == max(
        stage["workspace_alignment"] for stage in residency["stages"]
    )
    assert residency["kernels"] == sum(
        stage["kernels"] for stage in residency["stages"]
    )
    assert residency["entries"] == sum(
        stage["entries"] for stage in residency["stages"]
    )
    records = events.values
    activity = [event for event in records if event["event"] == "weight_activity"]
    loaded = [event for event in activity if event["activations"] == 4]
    assert loaded and loaded[0]["read_bytes"] > 0 and loaded[0]["groups"] > 0
    for event in loaded:
        for field in (
            "activations",
            "deactivations",
            "read_bytes",
            "prepared_bytes",
            "groups",
        ):
            assert event[field] == loaded[0][field], (field, event)
    assert activity[-1]["activations"] == 4 and activity[-1]["deactivations"] == 0
    prepared = [event for event in records if event["event"] == "image_prepared"]
    assert [event["stage"] for event in prepared] == [1, 0, 1, 1, 0, 1, 1]
    assert [event["input_bytes"] for event in prepared] == [
        16 + (text + 34) * 4 for text in (128, 512, 128, 128, 512, 128, 128)
    ]
    generated = [
        event["request"] for event in records if event["event"] == "image_generated"
    ]
    expected_ids = [first_id + offset for offset in (0, 3, 4, 5, 6, 7)]
    assert generated in (expected_ids, expected_ids + [first_id + 8]), generated
    print(
        "PASS: exact native HTTP images, health during compute, bounded admission, "
        "slow-reader independence, queued/active reset and joined shutdown; "
        "two cold text shapes share one model residency.",
        flush=True,
    )


if __name__ == "__main__":
    main()
