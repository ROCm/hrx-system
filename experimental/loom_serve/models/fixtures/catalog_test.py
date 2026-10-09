# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Mixed HTTP lifecycle churn against the real source-JIT serving executable.

The fixtures use independent arithmetic oracles, a tiny parameter checkpoint,
and actual GPU commands. This is ownership/correctness evidence, not a neural
quality or throughput benchmark. The outer runner owns hang detection.
"""

import argparse
import base64
import contextlib
import http.client
import io
import json
import socket
import struct
import tempfile
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.http.logs import LogClient
from experimental.loom_serve.testing.http import Events, wait_counters, wait_status
from experimental.loom_serve.testing.http import running_server as child_server
from experimental.loom_serve.tools.observe import EventLog, record_streams


@contextlib.contextmanager
def stalled_observer(client, rounds):
    """Run the real pull consumer with its output sink held across model work."""
    stopped, blocked, released = (threading.Event() for _ in range(3))

    class Sink(io.StringIO):
        def __init__(self):
            super().__init__()
            self.expected = None
            self.seen = {}

        def write(self, line):
            value = json.loads(line)
            model = value.get("stream", {}).get("model")
            event = value["data"]
            if (
                model == "fixture-text"
                and event["event"] == "cursor"
                and not blocked.is_set()
            ):
                blocked.set()
                released.wait()
            if (
                self.expected is not None
                and model is not None
                and event["event"] == "counters"
            ):
                self.seen[model] = event["values"]
                if self.seen == self.expected:
                    stopped.set()
            return super().write(line)

    sink = Sink()
    url = "http://" + client.address

    def observe():
        try:
            record_streams(LogClient(url, timeout=None), EventLog(sink), stopped)
        finally:
            blocked.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        observer = executor.submit(observe)
        try:
            blocked.wait()
            if observer.done():
                observer.result()
                raise AssertionError("observer exited before its sink was blocked")
            yield
            fast = LogClient(url, timeout=None)
            expected = {}
            for model in fast.models():
                batch = fast.poll(model)
                expected[model] = next(
                    record["values"]
                    for record in batch.records
                    if record["event"] == "counters"
                )
                counts = expected[model]
                assert counts["requests"] == sum(
                    counts[name]
                    for name in ("completed", "rejected", "cancelled", "failed")
                ), counts
            for model in ("pixels", "vectors", "judge"):
                assert (
                    expected[model]["requests"]
                    == expected[model]["completed"]
                    == rounds
                ), expected[model]
                assert expected[model]["output_tokens"] == 0, expected[model]
            assert expected["pixels"]["batches"] == rounds
            assert expected["vectors"]["input_tokens"] == sum(
                iteration + 6 for iteration in range(rounds)
            )
            total = {
                key: sum(counts[key] for counts in expected.values())
                for key in expected["pixels"]
            }
            for _ in range(2):
                batch = fast.poll()
                assert batch.records[0]["counters_available"]
                assert (
                    next(
                        record["values"]
                        for record in batch.records
                        if record["event"] == "counters"
                    )
                    == total
                )
                metrics = client.request("/metrics")
                assert "loom_serve_server_counters_available 1\n" in metrics
                for key, value in total.items():
                    assert f"loom_serve_server_{key}_total {value}\n" in metrics
                for model, counts in expected.items():
                    for key, value in counts.items():
                        prefix = f'loom_serve_model_{key}_total{{model="{model}",kind="'
                        assert any(
                            line.startswith(prefix) and line.endswith(f"}} {value}")
                            for line in metrics.splitlines()
                        )
                # Router-owned utilities do not become model-protocol work.
                assert client.request(
                    "/tokenize", {"model": "vectors", "content": "ab"}
                ) == {"tokens": [0, 1]}
            sink.expected = expected
            released.set()
            observer.result()
            records = [json.loads(line) for line in sink.getvalue().splitlines()]
            assert any(
                record["data"]["event"] == "gap"
                and record.get("stream", {}).get("model") == "fixture-text"
                for record in records
            )
            assert records[-1]["data"]["event"] == "observation_end"
            assert sink.seen == expected
            # Stopping the observer has no authority over the serving process.
            assert client.request("/readyz") == {"status": "ready"}
        finally:
            stopped.set()
            released.set()
            observer.result()


def complete(messages, maximum):
    """Dense recurrence with no knowledge of pages, row IDs or epoch shapes."""
    prompt = "".join(
        message["content"] + ("#" if message["role"] == "assistant" else "")
        for message in messages
    )
    hidden = history = total = position = 0

    def consume(token):
        nonlocal hidden, history, total, position
        hidden = (5 * hidden + token + 2) % 97
        history = (7 * history + hidden + position) % 101
        total += history
        position += 1

    for character in prompt:
        consume("abx#".index(character))
    output = ""
    for _ in range(maximum):
        if position % 11 == 0:
            return output, "stop"
        token = (hidden + total) % 3
        output += "abx"[token]
        consume(token)
    return output, "length"


def check_png(encoded, prompt, seed):
    image = base64.b64decode(encoded, validate=True)
    assert image[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    compressed = bytearray()
    header = None
    while position < len(image):
        length = int.from_bytes(image[position : position + 4], "big")
        kind = image[position + 4 : position + 8]
        payload = image[position + 8 : position + 8 + length]
        checksum = int.from_bytes(
            image[position + 8 + length : position + 12 + length], "big"
        )
        assert zlib.crc32(kind + payload) == checksum
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        if kind == b"IDAT":
            compressed.extend(payload)
        position += 12 + length
    assert header == (7, 5, 8, 2, 0, 0, 0), header
    assert position == len(image)
    expected = bytearray()
    for row in range(5):
        expected.append(0)
        for column in range(7):
            expected.extend(
                (
                    255 * (seed & 1),
                    255 * (len(prompt) & 1),
                    255 * ((row * 7 + column) & 1),
                )
            )
    assert zlib.decompress(compressed) == expected


class Client:
    def __init__(self, address):
        self.address = address

    def request(self, target, value=None, *, method=None, headers=None, status=200):
        connection = http.client.HTTPConnection(self.address)
        try:
            connection.request(
                method or ("POST" if value is not None else "GET"),
                target,
                None if value is None else json.dumps(value),
                {"Content-Type": "application/json", **(headers or {})},
            )
            response = connection.getresponse()
            body = response.read().decode()
            assert response.status == status, (target, response.status, body)
            if response.getheader("Content-Type") == "text/event-stream":
                events = [
                    json.loads(line[6:])
                    for line in body.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                assert "data: [DONE]" in body, body
                assert not any("error" in event for event in events), events
                choices = [
                    choice for event in events for choice in event.get("choices", [])
                ]
                usage = [event["usage"] for event in events if event.get("usage")]
                finishes = [
                    choice["finish_reason"]
                    for choice in choices
                    if choice.get("finish_reason")
                ]
                assert len(usage) == len(finishes) == 1, events
                assert all(event["model"] == "fixture-text" for event in events), events
                return {
                    "text": "".join(
                        choice["delta"].get("content", "") for choice in choices
                    ),
                    "finish": finishes[0],
                    "usage": usage[0],
                }
            if target == "/metrics":
                return body
            return json.loads(body)
        finally:
            connection.close()

    @staticmethod
    def text_payload(messages, maximum):
        return {
            "model": "fixture-text",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": maximum,
            "temperature": 0,
        }

    def text(self, session, messages, maximum=6, checkpoint=None):
        headers = {"X-Loom-Session": session}
        if checkpoint is not None:
            headers["X-Loom-Checkpoint"] = checkpoint
        result = self.request(
            "/v1/chat/completions",
            self.text_payload(messages, maximum),
            headers=headers,
        )
        assert (result["text"], result["finish"]) == complete(messages, maximum), (
            messages,
            result,
        )
        return result

    def checkpoint(self, name, method, session=None, status=200):
        headers = {"X-Loom-Model": "fixture-text"}
        if session is not None:
            headers["X-Loom-Session"] = session
        return self.request(
            "/v1/checkpoints/" + name, method=method, headers=headers, status=status
        )

    def reset_peer(self, session):
        connection = http.client.HTTPConnection(self.address)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(
                    self.text_payload([{"role": "user", "content": "abxaba"}], 16)
                ),
                {"Content-Type": "application/json", "X-Loom-Session": session},
            )
            peer = connection.sock
            response = connection.getresponse()
            assert response.status == 200
            # Header admission is real, but scheduling is free: this reset may
            # race with completion. Held-work tests cover exact cancel phases.
            peer.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
            response.close()
        finally:
            connection.close()

    def idle(self):
        # HTTP itself yields to the actual owner. No timing assumption or
        # arbitrary deadline substitutes for completed request ownership.
        while True:
            status = self.request("/v1/status")
            if all(
                item["loom"]["status"]["phase"] == "idle"
                and item["loom"]["status"]["queued_requests"] == 0
                and item["loom"]["status"]["active_requests"] == 0
                for item in status["data"]
            ):
                return status


@contextlib.contextmanager
def running_server(server, catalog, workers):
    events = Events()
    log_path = catalog.parent / f"server-{workers}.log"
    with child_server(
        [
            str(server),
            f"--models={catalog}",
            "--port=0",
            f"--workers={workers}",
            "--pool_backing=fixed",
            "--heartbeat_ms=0",
        ],
        log_path,
        events.record,
    ) as (_, address):
        try:
            assert address is not None, log_path.read_text()
            yield Client(address), events
        except BaseException:
            print(log_path.read_text(), flush=True)
            raise


def run_round(client, iteration, observations):
    root = [{"role": "user", "content": "abxa"}]
    seed = client.text("root", root, 2)
    assert seed["text"] == "bx"
    if iteration == 0:
        # Actual HTTP completion is the readiness contract. The independent
        # reader then checks native coverage without fabricating old schemas.
        completed = observations.wait("request_stats", "root", model="fixture-text")
        address = urlsplit("http://" + client.address)
        wait_status(
            address,
            model="fixture-text",
            active_requests=0,
            queued_requests=0,
            phase="idle",
        )
        counts = wait_counters(
            address, lambda counts: counts["completed"] == 1, model="fixture-text"
        )
        assert counts["requests"] == counts["completed"] == 1
        assert completed["counters"]["completed"] == 1
        assert (
            completed["counters"]["output_tokens"] == seed["usage"]["completion_tokens"]
        )
        history = observations.values
        admitted = next(event for event in history if event["event"] == "admit")
        spans = [event for event in history if event["event"] == "span"]
        epochs = [event for event in history if event["event"] == "epoch"]
        assert admitted["request"] == completed["request"]
        assert spans and epochs
        assert all(span["request"] == completed["request"] for span in spans)
        assert {span["epoch"] for span in spans} == {epoch["epoch"] for epoch in epochs}
    pin = client.checkpoint("base", "POST", "root", status=201)
    assert pin["position"] == 5, pin
    history = root + [{"role": "assistant", "content": seed["text"]}]

    def conversation(index):
        prompt = "abx" * (4 + (index + iteration) % 17) + "ab"[: index % 3]
        messages = [{"role": "user", "content": prompt}]
        session = f"row-{index}"
        first = client.text(session, messages)
        messages += [
            {"role": "assistant", "content": first["text"]},
            {"role": "user", "content": "ba"},
        ]
        client.text(session, messages)

    def fork(index):
        messages = history + [{"role": "user", "content": "abx"[: index + 1]}]
        result = client.text(f"fork-{index}", messages, checkpoint="base")
        assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 5, result

    def embeddings():
        inputs = ["a" * (iteration + 1), "bax", "xx"]
        result = client.request(
            "/v1/embeddings",
            {
                "model": "vectors",
                "input": inputs,
                "dimensions": 2,
            },
        )
        assert [row["index"] for row in result["data"]] == [0, 1, 2]
        assert [row["embedding"] for row in result["data"]] == [
            [
                len(value) + sum("abx".index(character) for character in value),
                len(value),
            ]
            for value in inputs
        ], result
        count = sum(map(len, inputs))
        assert result["usage"] == {"prompt_tokens": count, "total_tokens": count}

    def image():
        prompt = "ab" + "a" * (iteration % 2)
        result = client.request(
            "/v1/images/generations",
            {
                "model": "pixels",
                "prompt": prompt,
                "seed": iteration,
                "response_format": "b64_json",
            },
        )
        check_png(result["data"][0]["b64_json"], prompt, iteration)

    def decision():
        # One decision request is live, so the independent cohort oracle is 1.
        result = client.request(
            "/v1/systemone",
            {
                "model": "judge",
                "state": "a",
                "questions": {
                    "truth": {
                        "type": "noul",
                        "instructions": "???",
                        "criteria": {"false": "", "true": "x"},
                    },
                },
            },
        )
        assert result["answers"]["truth"] == {"type": "noul", "noul": 0.666666687}, (
            result
        )

    def utilities():
        assert client.request("/tokenize", {"model": "vectors", "content": "abxa"}) == {
            "tokens": [0, 1, 2, 0]
        }
        assert client.request(
            "/detokenize", {"model": "pixels", "tokens": [1, 0, 1]}
        ) == {"content": "bab"}
        assert client.request("/readyz") == {"status": "ready"}

    operations = [lambda index=index: conversation(index) for index in range(12)]
    operations += [lambda index=index: fork(index) for index in range(3)]
    operations += [
        embeddings,
        image,
        decision,
        utilities,
        lambda: client.reset_peer("reset"),
    ]
    barrier = threading.Barrier(len(operations))

    def execute(operation):
        barrier.wait()
        operation()

    with ThreadPoolExecutor(max_workers=len(operations)) as clients:
        futures = [clients.submit(execute, operation) for operation in operations]
        for future in futures:
            future.result()

    client.idle()
    # Invalid input may neither consume the checkpoint nor poison future work.
    client.request(
        "/v1/chat/completions",
        Client.text_payload(
            history + [{"role": "user", "content": "?"}],
            6,
        ),
        headers={"X-Loom-Session": "root", "X-Loom-Checkpoint": "base"},
        status=400,
    )
    rewind = client.text(
        "root", history + [{"role": "user", "content": "xx"}], checkpoint="base"
    )
    assert rewind["usage"]["prompt_tokens_details"]["cached_tokens"] == 5
    client.checkpoint("base", "DELETE")
    client.checkpoint("base", "DELETE", status=404)
    client.idle()
    # This request reserves the entire 128-position logical pool. Idle rows
    # can be evicted, but a leaked pin/reservation would prevent admission.
    client.text("whole-pool", [{"role": "user", "content": "xba" * 41 + "x"}], 4)
    # Reuse the reset peer's identity only after the real owner reports idle.
    client.text("reset", root, 2)
    status = client.idle()
    metrics = client.request("/metrics")
    for item in status["data"]:
        labels = f'model="{item["id"]}",kind="{item["loom"]["kind"]}"'
        for gauge in ("queued_requests", "active_requests"):
            assert f"loom_serve_model_{gauge}{{{labels}}} 0\n" in metrics
    return {
        "round": iteration,
        "concurrent_clients": len(operations),
        "resident_rows": 2,
        "pool_positions": 128,
        "exact_outputs": True,
        "whole_pool_reused": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4, choices=range(1, 17))
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="loom-catalog-churn-") as temporary:
        directory = Path(temporary)
        weights = directory / "text.safetensors"
        header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        weights.write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<I", 2)
        )
        catalog = json.loads(arguments.catalog.read_text())
        text = arguments.catalog.parent / "text"
        catalog["models"].append(
            {
                "kind": "text",
                "name": "fixture-text",
                "source": str(text),
                "weights": str(weights),
                "tokenizer": str(text / "tokenizer.json"),
                "rows": 2,
                "context_capacity": 128,
                "pool_capacity": 128,
                "prefill_capacity": 32,
                "checkpoint_capacity": 2,
                "max_tokens": 16,
                "mtp_depth": 0,
                "pending_requests": 32,
            }
        )
        catalog_path = directory / "models.json"
        catalog_path.write_text(json.dumps(catalog))
        for workers in (1, 4):
            with running_server(arguments.server, catalog_path, workers) as (
                client,
                observations,
            ):
                assert client.request("/readyz") == {"status": "ready"}
                discovered = client.request("/v1/models")
                assert [item["id"] for item in discovered["data"]] == [
                    "pixels",
                    "vectors",
                    "judge",
                    "fixture-text",
                ]
                with stalled_observer(client, arguments.rounds):
                    for iteration in range(arguments.rounds):
                        print(
                            json.dumps(
                                {
                                    "event": "catalog_churn",
                                    "workers": workers,
                                    **run_round(client, iteration, observations),
                                }
                            ),
                            flush=True,
                        )
            print(
                json.dumps({"event": "catalog_joined", "workers": workers}), flush=True
            )


if __name__ == "__main__":
    main()
