# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify the real retained image HTTP service, including ownership failures.

Requires the isolated-0/1/2.ppm images produced by check_model.py, the same
checkpoint/source geometry, and an exclusive device lease. The production
server performs every image operation natively. Pillow only independently
decodes the returned PNGs for exact comparison with isolated CLI pixels.
Lifecycle events synchronize the client without sleeps or GPU wait deadlines.
Final images and concise logs stay below 16 MiB; no checkpoint copies.
"""

import argparse
import base64
import http.client
import io
import json
import signal
import socket
import struct
import subprocess
import threading
from pathlib import Path

from PIL import Image


class Journal:
    def __init__(self, process, path):
        self.process = process
        self.path = path
        self.condition = threading.Condition()
        self.events = []
        self.lines = []
        self.closed = False
        self.failure = None
        self.thread = threading.Thread(target=self.read)
        self.thread.start()

    def read(self):
        try:
            with self.path.open("w") as log:
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    event = json.loads(line) if line.startswith("{") else None
                    with self.condition:
                        self.lines.append(line)
                        if event:
                            self.events.append(event)
                            if event.get("event", "").startswith("image_"):
                                print(line, end="", flush=True)
                        self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                self.failure = error
        finally:
            with self.condition:
                self.closed = True
                self.condition.notify_all()

    def wait(self, event, **fields):
        with self.condition:
            while True:
                for record in self.events:
                    if record.get("event") == event and all(
                        record.get(key) == value for key, value in fields.items()
                    ):
                        return record
                if self.failure:
                    raise self.failure
                if self.closed:
                    raise AssertionError(f"server stopped before {event}: {fields}")
                self.condition.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", nargs="+", required=True)
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
        f"--model={args.model}",
        f"--checkpoint={args.checkpoint}",
        f"--adapter={args.adapter}",
        "--width=384",
        "--height=384",
        "--text_tokens=512",
        "--port=0",
        "--pending_requests=1",
        "--connections=16",
    ]
    (args.output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    journal = Journal(process, args.output / "server.log")
    sockets = []
    try:
        ready = journal.wait("image_ready")
        host, port = ready["address"].rsplit(":", 1)
        address = host, int(port)

        def send(body=None, path="/v1/images/generations", slow=False):
            peer = socket.socket()
            sockets.append(peer)
            if slow:
                peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            peer.connect(address)
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
        assert models["data"][0]["reference_images"] is False
        assert models["data"][0]["adapter_strength"] is True
        for body in (
            b"{}",
            b'{"prompt":"x","prompt":"y"}',
            b'{"prompt":"x","strength":1e100}',
            b'{"prompt":"x","seed":18446744073709551616}',
            b'{"prompt":"x","n":2}',
            b'{"prompt":"x","size":"512x512"}',
            b'{"prompt":"x","reference_images":[]}',
            b'{"prompt":"\xff"}',
        ):
            assert "error" in receive(send(body), 400)
        receive(send(path="/unknown"), 404)
        health = receive(send(path="/healthz"))
        assert (health["active"], health["queued"], health["completed"]) == (0, 0, 0)

        deer = dict(
            prompt="A deer grazing in the forest, Art Deco watercolor style",
            seed="0",
            strength=0,
            model="krea2-turbo",
            size="384x384",
            n=1,
            response_format="b64_json",
        )
        robot = dict(
            deer,
            prompt="A small brass robot tending red flowers in a sunlit greenhouse, watercolor illustration",
            seed=42,
        )
        first = send(deer, slow=True)
        journal.wait("image_started", request=1)
        health = receive(send(path="/healthz"))
        assert health["active"] == 1 and health["completed"] == 0
        queued = send(robot)
        journal.wait("image_accepted", request=2)
        health = receive(send(path="/healthz"))
        assert health["active"] == 1 and health["queued"] == 1
        receive(send(deer), 503)
        reset(queued)
        journal.wait("image_cancelled", request=2, phase="queued")
        replacement = send(robot)
        journal.wait("image_accepted", request=3)
        journal.wait("image_generated", request=1)
        journal.wait("image_generated", request=3)
        # Neither socket has consumed response bytes. In particular, the first
        # peer's deliberately small receive window cannot pin the model output.
        pixels(replacement, "queued-robot", 1)
        pixels(first, "slow-deer", 0)

        style = send(dict(deer, strength=1))
        journal.wait("image_generated", request=4)
        pixels(style, "style", 2)
        repeated = send(deer)
        journal.wait("image_generated", request=5)
        pixels(repeated, "base-after-style", 0)

        abandoned = send(robot)
        journal.wait("image_started", request=6)
        reset(abandoned)
        journal.wait("image_cancelled", request=6, phase="active")
        after_reset = send(deer)
        journal.wait("image_accepted", request=7)
        health = receive(send(path="/healthz"))
        assert health["active"] == 1 and health["queued"] == 1
        journal.wait("image_generated", request=6)
        journal.wait("image_generated", request=7)
        pixels(after_reset, "base-after-reset", 0)

        send(dict(deer, strength=1))
        journal.wait("image_started", request=8)
        process.send_signal(signal.SIGTERM)
        journal.wait("image_shutdown", active=8)
        journal.wait("image_generated", request=8)
        stopped = journal.wait("image_stopped")
        assert stopped["completed"] == 7
        assert process.wait() == 0
        journal.thread.join()
        text = "".join(journal.lines)
        assert text.count('"event":"jit_stage"') == 1
        assert text.count('"event":"image_residency"') == 1
        assert text.count("Streaming ") == 4
        events = [record["event"] for record in journal.events]
        assert (
            events.index("image_shutdown")
            < len(events) - 1 - events[::-1].index("image_generated")
            < events.index("image_stopped")
        )
        (args.output / "events.json").write_text(
            json.dumps(journal.events, indent=2) + "\n"
        )
        print(
            "PASS: exact native HTTP images, health during compute, bounded admission, "
            "slow-reader independence, queued/active reset and joined shutdown; one model residency.",
            flush=True,
        )
    finally:
        for peer in sockets:
            peer.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        journal.thread.join()
        process.stdout.close()


if __name__ == "__main__":
    main()
