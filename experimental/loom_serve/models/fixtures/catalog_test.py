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
import shutil
import socket
import struct
import tempfile
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http import Events, wait_counters, wait_status
from loom_serve_testing.http import running_server as child_server
from loom_serve_testing.logs import LogClient

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


def check_decision(result):
    # The single-question input gives true/false weights of 2 and 1.
    answer = result["answers"]["truth"]
    assert set(answer) == {"type", "noul"}, result
    assert answer["type"] == "noul", result
    # Vulkan F32 division permits 2.5 ULP; at 2/3, one ULP is 2^-24.
    assert abs(answer["noul"] - 2 / 3) <= 2.5 * 2**-24, result


class TelemetryObservation:
    """A real SSE reader checking contiguous batches, not a model dependency."""

    def __init__(self, response):
        self.response = response
        self.identity = None
        self.position = 0
        self.cursor = None
        self.counters = None

    def read(self):
        fields, records = {}, []
        while True:
            line = self.response.readline()
            assert line, "telemetry ended without a stream_end event"
            if line == b"\n":
                break
            key, value = line.decode().rstrip("\n").split(": ", 1)
            if key == "data":
                records.append(json.loads(value))
            else:
                assert key not in fields, (key, fields)
                fields[key] = value
        if fields["event"] == "stream_end":
            assert records == [
                {"identity": self.identity, "reason": "model_unregistered"}
            ], records
            assert self.response.read() == b""
            return False
        assert fields["event"] == "telemetry", fields
        header, footer = records[0], records[-1]
        assert header["event"] == "stream" and footer["event"] == "cursor", records
        if self.identity is None:
            self.identity = header["identity"]
        assert self.identity == header["identity"], header
        for record in records[1:-1]:
            if record["event"] == "counters":
                self.counters = record["values"]
            else:
                assert record["event"] != "gap", record
                assert record["sequence"] == self.position, record
                self.position += 1
        self.cursor = fields["id"]
        assert self.cursor == footer["value"] == f"{self.identity}:{self.position}"
        assert footer["next"] == self.position, footer
        return True

    def completed(self, count):
        while self.counters is None or self.counters["completed"] < count:
            assert self.read(), "model retired before its request completed"
        assert self.counters["completed"] == count, self.counters


@contextlib.contextmanager
def telemetry_observation(connection, model):
    try:
        connection.request("GET", "/logz/stream", headers={"X-Loom-Model": model})
        response = connection.getresponse()
        assert response.status == 200, response.status
        assert response.getheader("Content-Type") == "text/event-stream"
        observer = TelemetryObservation(response)
        assert observer.read()
        yield observer
    finally:
        connection.close()


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
                assert all(event["model"] == value["model"] for event in events), events
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

    def wait_activation(self, name):
        while True:
            state = next(
                item["loom"]
                for item in self.request("/v1/status")["data"]
                if item["id"] == name
            )
            assert "error" not in state, state
            if state["parameter_operation"] == "idle":
                assert state["registration"] == "ready", state
                assert state["status"]["admission"] == "drained", state
                assert state["parameters"]["active"], state
                return state

    def activate(self, name):
        self.request("/v1/models/activate", {"model": name}, status=202)
        return self.wait_activation(name)

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
def running_server(server_command, directory, catalog, workers):
    events = Events()
    log_path = catalog.parent / f"server-{workers}.log"
    with child_server(
        [
            *server_command,
            f"--flagfile={catalog.relative_to(directory)}",
            "--listen=127.0.0.1:0",
            f"--workers={workers}",
            "--heartbeat_ms=0",
        ],
        log_path,
        events.record,
        cwd=directory,
    ) as (_, address):
        try:
            assert address is not None, log_path.read_text()
            client = Client(address)
            client.idle()
            yield client, events
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
        assert all(epoch["device_epochs"] == 1 for epoch in epochs)
        assert all(epoch["token_slots"] == epoch["token_capacity"] for epoch in epochs)
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
        check_decision(result)

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
        if "incarnation" in item["loom"]:
            labels += f',incarnation="{item["loom"]["incarnation"]}"'
        for gauge in ("queued_requests", "active_requests"):
            assert f"loom_serve_model_{gauge}{{{labels}}} 0\n" in metrics
    return {
        "round": iteration,
        "concurrent_clients": len(operations),
        "resident_rows": 2,
        "pool_positions": 128,
        "outputs_verified": True,
        "whole_pool_reused": True,
    }


def check_parameter_residency(client, backing, workers):
    """Keep source programs and a pinned conversation across weight release."""
    root = [{"role": "user", "content": "abxa"}]
    seed = client.text("residency-root", root, 2)
    pin = client.checkpoint("resident-prefix", "POST", "residency-root", status=201)
    assert pin["position"] == 5, pin
    history = root + [
        {"role": "assistant", "content": seed["text"]},
        {"role": "user", "content": "xx"},
    ]
    client.idle()

    def snapshot(name="fixture-text"):
        state = next(
            item["loom"]
            for item in client.request("/v1/status")["data"]
            if item["id"] == name
        )
        assert "error" not in state, state
        return state

    def retained_memory():
        devices = client.request("/v1/status")["server"]["resources"]["devices"]
        assert len(devices) == 1, devices
        assert devices[0]["memory"]["backing"] == backing, devices
        return devices[0]["memory"]["retained"]

    before = snapshot()
    memory_before = retained_memory()
    assert before["parameters"]["active"], before
    assert before["parameters"]["backing"] == backing, before
    selector = {"model": "fixture-text"}
    client.request("/v1/models/deactivate", selector, status=409)
    client.request("/v1/models/drain", selector)
    with ThreadPoolExecutor(max_workers=1) as clients:
        continuation = clients.submit(
            client.text, "residency-root", history, checkpoint="resident-prefix"
        )
        try:
            wait_status(
                urlsplit("http://" + client.address),
                model="fixture-text",
                held_requests=1,
            )
            client.request(
                "/v1/models/deactivate",
                selector,
                status=202 if backing == "elastic" else 409,
            )
            while True:
                cold = snapshot()
                if cold["parameter_operation"] == "idle":
                    break
            parameters = cold["parameters"]
            assert cold["incarnation"] == before["incarnation"], cold
            assert cold["compilation"] == before["compilation"], cold
            assert cold["status"]["admission"] == "drained", cold
            assert cold["status"]["held_requests"] == 1, cold
            inventory_headers = {"X-Loom-Model": "fixture-text"}
            requests = client.request("/v1/requests", headers=inventory_headers)
            assert requests["incarnation"] == before["incarnation"], requests
            assert requests["held_requests"] == 1, requests
            assert requests["data"] == [], requests
            sessions = client.request("/v1/sessions", headers=inventory_headers)
            retained = [
                entry for entry in sessions["data"] if entry["name"] == "residency-root"
            ]
            assert len(retained) == 1 and retained[0]["request"] is None, sessions
            assert retained[0]["position"] == 5, retained
            assert retained[0]["counters"]["completed"] == 1, retained
            memory_cold = retained_memory()
            if backing == "elastic":
                assert not parameters["active"], cold
                assert parameters["committed_bytes"] == 0, cold
                assert before["parameters"]["committed_bytes"] > 0, before
                assert parameters["reserved_bytes"] > 0, cold
                assert memory_before["limit_bytes"] is None, memory_before
                assert memory_cold["reserved_bytes"] == memory_before["reserved_bytes"]
                released = before["parameters"]["committed_bytes"]
                assert (
                    memory_before["committed_bytes"] - memory_cold["committed_bytes"]
                    == released
                ), (memory_before, memory_cold)
                assert (
                    memory_cold["released_bytes"] - memory_before["released_bytes"]
                    == released
                ), (memory_before, memory_cold)
                assert parameters["deactivations"] == (
                    before["parameters"]["deactivations"] + 1
                ), cold
                # Repeating the operation cannot discard retained state or
                # count another physical release of already-inactive weights.
                client.request("/v1/models/deactivate", selector, status=202)
                while snapshot()["parameter_operation"] != "idle":
                    pass
                assert snapshot()["parameters"] == parameters
            else:
                assert memory_cold is memory_before is None
                assert parameters == before["parameters"], cold
                assert "committed_bytes" not in parameters, cold
                assert "reserved_bytes" not in parameters, cold
            peer = client.request(
                "/v1/embeddings",
                {"model": "vectors", "input": "ab", "dimensions": 2},
            )
            assert peer["data"][0]["embedding"] == [3, 2], peer
            assert not continuation.done()
            warmed = client.activate("fixture-text")
            memory_warmed = retained_memory()
            assert warmed["incarnation"] == before["incarnation"], warmed
            assert warmed["compilation"] == before["compilation"], warmed
            assert warmed["status"]["held_requests"] == 1, warmed
            assert warmed["parameters"]["activations"] == (
                before["parameters"]["activations"] + (backing == "elastic")
            ), warmed
            if backing == "elastic":
                assert (
                    memory_warmed["committed_bytes"] == memory_before["committed_bytes"]
                ), (memory_before, memory_warmed)
                assert (
                    warmed["parameters"]["committed_bytes"]
                    == (before["parameters"]["committed_bytes"])
                ), warmed
            assert client.activate("fixture-text")["parameters"] == warmed["parameters"]
            assert not continuation.done()
        finally:
            client.request("/v1/models/undrain", selector)
        resumed = continuation.result()
    assert resumed["usage"]["prompt_tokens_details"]["cached_tokens"] == 5, resumed
    client.idle()
    after = snapshot()
    assert after["incarnation"] == before["incarnation"], after
    assert after["compilation"] == before["compilation"], after
    assert after["parameters"]["active"], after
    assert after["parameters"]["activations"] == (
        before["parameters"]["activations"] + (backing == "elastic")
    ), after
    if backing == "elastic":
        assert (
            after["parameters"]["committed_bytes"]
            == (before["parameters"]["committed_bytes"])
        ), after
    assert client.request("/tokenize", {"model": "fixture-text", "content": "ab"}) == {
        "tokens": [0, 1]
    }
    client.checkpoint("resident-prefix", "DELETE")
    print(
        json.dumps(
            {
                "event": "parameter_residency_resumed",
                "workers": workers,
                "backing": backing,
                "before": before["parameters"],
                "cold": cold["parameters"],
                "warmed": warmed["parameters"],
                "after": after["parameters"],
                "cached_tokens": resumed["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ],
                "outputs_verified": True,
                "incarnation_preserved": True,
                "compilation_preserved": True,
                "shared_retained_memory": {
                    "before": memory_before,
                    "cold": memory_cold,
                    "warmed": memory_warmed,
                },
            }
        ),
        flush=True,
    )
    # Every native registration adapter owns this same boundary, including
    # source programs without a parameter checkpoint. Exercise their operation
    # and final HTTP result rather than inferring support from the text adapter.
    for model, target, payload in (
        ("vectors", "/v1/embeddings", {"input": "ab", "dimensions": 2}),
        (
            "pixels",
            "/v1/images/generations",
            {"prompt": "ab", "seed": 1, "response_format": "b64_json"},
        ),
        (
            "judge",
            "/v1/systemone",
            {
                "state": "a",
                "questions": {
                    "truth": {
                        "type": "noul",
                        "instructions": "???",
                        "criteria": {"false": "", "true": "x"},
                    }
                },
            },
        ),
    ):
        selector = {"model": model}
        identity = snapshot(model)["incarnation"]
        client.request("/v1/models/drain", selector)
        client.request(
            "/v1/models/deactivate",
            selector,
            status=202 if backing == "elastic" else 409,
        )
        while True:
            state = snapshot(model)
            parameters = state["parameters"]
            if state["parameter_operation"] == "idle":
                break
        if backing == "elastic":
            assert not parameters["active"], parameters
            assert parameters["committed_bytes"] == 0, parameters
        warmed = client.activate(model)
        assert warmed["incarnation"] == identity, warmed
        client.request("/v1/models/undrain", selector)
        result = client.request(target, {**selector, **payload})
        if model == "vectors":
            assert result["data"][0]["embedding"] == [3, 2], result
        elif model == "pixels":
            check_png(result["data"][0]["b64_json"], "ab", 1)
        else:
            check_decision(result)
        assert snapshot(model)["incarnation"] == identity


def check_device_trim(client, backing, workers):
    """Release eligible shared backing while a retained continuation is held."""
    selector = {"model": "fixture-text"}
    root = [{"role": "user", "content": "abxa"}]
    seed = client.text("trim-root", root, 2)
    client.checkpoint("trim-prefix", "POST", "trim-root", status=201)
    history = root + [
        {"role": "assistant", "content": seed["text"]},
        {"role": "user", "content": "xx"},
    ]
    client.idle()

    def snapshot():
        state = next(
            item["loom"]
            for item in client.request("/v1/status")["data"]
            if item["id"] == "fixture-text"
        )
        assert "error" not in state, state
        return state

    before = snapshot()
    workspace_before = client.idle()["server"]["resources"]["devices"][0]["memory"][
        "workspace"
    ]
    counter_names = (
        "reserve_count",
        "release_count",
        "reuse_count",
        "reuse_miss_count",
        "fresh_count",
        "exhausted_count",
        "over_budget_count",
        "wait_count",
        "quarantine_eviction_count",
    )
    assert workspace_before["reserve_count"] > 0, workspace_before
    client.request("/v1/models/pin", selector)
    client.request("/v1/models/drain", selector)
    with ThreadPoolExecutor(max_workers=1) as clients:
        continuation = clients.submit(
            client.text, "trim-root", history, checkpoint="trim-prefix"
        )
        try:
            wait_status(
                urlsplit("http://" + client.address),
                model="fixture-text",
                held_requests=1,
            )
            pinned = client.request(
                "/v1/devices/trim", {"device": 0, "target_bytes": 0}
            )
            assert pinned["target_bytes"] == pinned["device"]["index"] == 0, pinned
            assert snapshot()["parameters"] == before["parameters"]
            unpinned = client.request("/v1/models/unpin", selector)["loom"]
            assert unpinned["parameters"] == before["parameters"], unpinned
            trimmed = client.request(
                "/v1/devices/trim", {"device": 0, "target_bytes": 0}
            )
            cold = snapshot()
            assert cold["incarnation"] == before["incarnation"], cold
            assert cold["compilation"] == before["compilation"], cold
            assert cold["status"]["admission"] == "drained", cold
            assert cold["status"]["held_requests"] == 1, cold
            memory_pinned = pinned["device"]["memory"]["retained"]
            memory_cold = trimmed["device"]["memory"]["retained"]
            workspace_cold = trimmed["device"]["memory"]["workspace"]
            # Counter ownership is the device, not a trim generation or model.
            for name in counter_names:
                assert workspace_cold[name] >= workspace_before[name], (
                    name,
                    workspace_before,
                    workspace_cold,
                )
            if backing == "elastic":
                released = before["parameters"]["committed_bytes"]
                assert released > 0, before
                assert not cold["parameters"]["active"], cold
                assert cold["parameters"]["committed_bytes"] == 0, cold
                assert memory_pinned["reserved_bytes"] == memory_cold["reserved_bytes"]
                assert (
                    memory_pinned["committed_bytes"] - memory_cold["committed_bytes"]
                    == released
                ), (memory_pinned, memory_cold)
                assert (
                    memory_cold["released_bytes"] - memory_pinned["released_bytes"]
                    == released
                ), (memory_pinned, memory_cold)
                # The target is deliberately unreachable: retained KV survives.
                assert memory_cold["committed_bytes"] > trimmed["target_bytes"], trimmed
            else:
                assert memory_pinned is memory_cold is None
                assert cold["parameters"] == before["parameters"], cold
            client.request("/v1/devices/trim", {"device": 0, "target_bytes": 0})
            assert snapshot()["parameters"] == cold["parameters"]
            assert not continuation.done()
        finally:
            client.request("/v1/models/undrain", selector)
        resumed = continuation.result()
    assert resumed["usage"]["prompt_tokens_details"]["cached_tokens"] == 5, resumed
    client.idle()
    after = snapshot()
    assert after["incarnation"] == before["incarnation"], after
    assert after["compilation"] == before["compilation"], after
    assert after["parameters"]["activations"] == (
        before["parameters"]["activations"] + (backing == "elastic")
    ), after
    client.checkpoint("trim-prefix", "DELETE")
    vectors = client.request(
        "/v1/embeddings", {"model": "vectors", "input": "ab", "dimensions": 2}
    )
    assert vectors["data"][0]["embedding"] == [3, 2], vectors
    workspace_after = client.idle()["server"]["resources"]["devices"][0]["memory"][
        "workspace"
    ]
    assert workspace_after["reserve_count"] > workspace_cold["reserve_count"], (
        workspace_cold,
        workspace_after,
    )
    print(
        json.dumps(
            {
                "event": "device_trim_resumed",
                "workers": workers,
                "backing": backing,
                "pinned": pinned,
                "trimmed": trimmed,
                "cached_tokens": 5,
                "outputs_verified": True,
                "incarnation_preserved": True,
                "compilation_preserved": True,
                "workspace_after": workspace_after,
            }
        ),
        flush=True,
    )


def copy_package(source, directory, kind):
    """Relocate a fixture with numerical providers outside its model directory.

    The separate prepare/control/weight-policy leaves and tokenizer stay in the
    model package. The model document locates the sibling numerical providers
    independently of the child's working directory.
    Copies dereference runfiles symlinks without preserving their read-only cache
    permissions, so the child reads independent, editable deployment files.
    """
    package = directory / "models" / kind
    shutil.copytree(source, package, copy_function=shutil.copyfile)
    providers = directory / "providers" / kind
    providers.mkdir(parents=True)
    catalog = package / "model.json"
    model = json.loads(catalog.read_text())
    names = model["definition"]["sources"]
    for name in names:
        (package / name).rename(providers / name)
    model["definition"]["sources"] = [
        f"../../providers/{kind}/{name}" for name in names
    ]
    catalog.write_text(json.dumps(model))
    return package.relative_to(directory)


def write_startup(path, catalog):
    """Write ordinary process flags referencing complete individual documents."""
    flags = []
    for index, model in enumerate(catalog["models"]):
        document = path.with_name(f"{path.stem}-{index}.json")
        document.write_text(json.dumps(model))
        flags.append(f"--model={document.resolve()}")
    path.write_text("\n".join(flags) + "\n")


def check_startup_failures(server_command, directory, catalog):
    """Keep an incomplete registration inspectable while peers serve output."""
    models = catalog["models"]
    cases = [
        (model, Path(model["base_directory"]) / "control.loom") for model in models
    ]
    cases.append((models[-1], Path("providers/text/advance.loom")))
    for model, missing in cases:
        source = directory / missing
        saved = source.with_suffix(".saved")
        broken = {"models": [entry for entry in models if entry is not model] + [model]}
        name = f"missing-{model['definition']['kind']}-{missing.stem}"
        catalog_path = directory / "configuration" / f"{name}.json"
        write_startup(catalog_path, broken)
        log_path = catalog_path.with_suffix(".log")
        source.rename(saved)
        try:
            with child_server(
                [
                    *server_command,
                    f"--flagfile={catalog_path.relative_to(directory)}",
                    "--listen=127.0.0.1:0",
                    "--workers=1",
                ],
                log_path,
                cwd=directory,
            ) as (_, address):
                assert address is not None, log_path.read_text()
                client = Client(address)
                while True:
                    listing = client.request("/v1/status")
                    failed = next(
                        item["loom"]
                        for item in listing["data"]
                        if item["id"] == model["name"]
                    )
                    if failed["registration"] == "failed":
                        break
                assert missing.name in failed["error"]["message"], failed
                assert failed["error"]["code"] == 5, failed
                assert client.request("/readyz") == {"status": "ready"}
                # A failed source package does not terminate peer models.
                if model["definition"]["kind"] != "text":
                    client.text(
                        "load-failure-peer", [{"role": "user", "content": "ab"}]
                    )
                else:
                    result = client.request(
                        "/v1/embeddings",
                        {"model": "vectors", "input": "ab", "dimensions": 2},
                    )
                    assert result["data"][0]["embedding"] == [3, 2], result
            diagnostic = log_path.read_text()
            assert (
                "Sanitizer" not in diagnostic and "runtime error:" not in diagnostic
            ), diagnostic
        finally:
            saved.rename(source)
        print(
            json.dumps(
                {
                    "event": "catalog_startup_rejected",
                    "model": model["name"],
                    "missing": str(missing),
                    "earlier_models": len(models) - 1,
                }
            ),
            flush=True,
        )


def check_dynamic_observer(server_command, directory, catalog):
    """The shipping observer follows real model registration, not a static list."""
    declaration = next(
        model
        for model in catalog["models"]
        if model["definition"]["kind"] == "embedding"
    )
    log_path = directory / "dynamic-observer.log"
    socket_path = directory / 'observer "local".sock'
    events, stopped = Events(), threading.Event()

    class Sink(io.StringIO):
        def write(self, line):
            record = json.loads(line)
            data, stream = record["data"], record.get("stream", {})
            if record["source"] == "observer":
                events.record({**data, **stream})
            elif data["event"] == "cursor" and "model" in stream:
                events.record({**data, **stream, "event": "model_cursor"})
            return super().write(line)

    with child_server(
        [
            *server_command,
            "--listen=127.0.0.1:0",
            f"--listen=unix:{socket_path}",
            "--workers=1",
        ],
        log_path,
        cwd=directory,
    ) as (_, address):
        assert address is not None, log_path.read_text()
        client = Client(address)
        reader = LogClient(
            "http://observer.invalid", timeout=None, unix_socket=str(socket_path)
        )
        sink = Sink()

        def observe():
            try:
                record_streams(reader, EventLog(sink), stopped, interval=0.01)
            finally:
                events.record(None)

        with ThreadPoolExecutor(max_workers=1) as executor:
            observer = executor.submit(observe)
            try:
                events.wait(
                    "observation_begin", models=[], unix_socket=str(socket_path)
                )
                client.request("/v1/models/register", declaration, status=202)
                first = events.wait("model_available", model="vectors")["identity"]
                events.wait("model_cursor", model="vectors", identity=first)
                output = client.request(
                    "/v1/embeddings",
                    {"model": "vectors", "input": "ab", "dimensions": 2},
                )
                assert output["data"][0]["embedding"] == [3, 2], output
                client.request(
                    "/v1/models/unregister", {"model": "vectors"}, status=202
                )
                absent = events.wait("model_unavailable", model="vectors")
                assert absent["previous_cursor"].startswith(first + ":"), absent
                client.request("/v1/models/register", declaration, status=202)
                second = (
                    LogClient("http://" + address, timeout=None)
                    .poll("vectors")
                    .stream["identity"]
                )
                assert first != second
                events.wait("model_available", model="vectors", identity=second)
                events.wait("model_cursor", model="vectors", identity=second)
                output = client.request(
                    "/v1/embeddings",
                    {"model": "vectors", "input": "bx", "dimensions": 2},
                )
                assert output["data"][0]["embedding"] == [5, 2], output
            finally:
                stopped.set()
                observer.result()
        assert set(reader.cursors) == {None, "vectors"}, reader.cursors
        events.wait("observation_end")
    assert not socket_path.exists()
    print(json.dumps({"event": "dynamic_observer_followed_reincarnation"}), flush=True)


def check_empty_server(server_command, directory, device, backing):
    catalog = directory / "empty.flags"
    catalog.write_text("# No startup models\n")
    for mode, arguments in (
        ("omitted", []),
        ("empty", [f"--flagfile={catalog.name}"]),
    ):
        log_path = directory / f"{mode}-models.log"
        memory_limit = (
            16 * 1024 * 1024 if mode == "empty" and backing == "elastic" else 0
        )
        with child_server(
            [
                *server_command,
                *arguments,
                "--listen=127.0.0.1:0",
                "--workers=1",
                "--connections=19",
                "--held_requests=7",
                "--model_capacity=9",
                "--request_body_bytes=65536",
                f"--memory_bytes={memory_limit}",
            ],
            log_path,
            cwd=directory,
        ) as (_, address):
            assert address is not None, log_path.read_text()
            client = Client(address)
            for target in ("/healthz", "/readyz"):
                assert client.request(target) == {"status": "ready"}
            assert client.request("/v1/models") == {"object": "list", "data": []}
            status = client.request("/v1/status")
            server = status.pop("server")
            assert status == {"object": "list", "data": []}, status
            resources = server.pop("resources")
            assert server == {
                "worker_count": 1,
                "model_capacity": 9,
                "connection_capacity": 19,
                "request_limits": {
                    "header_byte_capacity": 16384,
                    "body_byte_capacity": 65536,
                    "header_capacity": 64,
                },
                "held_request_capacity": 7,
                "held_requests": 0,
                "observer_capacity": 18,
                "observers": 0,
            }, server
            assert resources["listeners"] == [
                {"address": address, "waiting_for_capacity": False}
            ], resources
            assert len(resources["devices"]) == 1, resources
            actual = resources["devices"][0]
            assert actual["index"] == 0 and actual["driver"] == device, actual
            assert actual["id"] and actual["name"] and actual["backend"], actual
            assert actual["physical_devices"], actual
            for physical in actual["physical_devices"]:
                assert physical["ordinal"] >= 0 and physical["name"], physical
            memory = actual["memory"]
            assert memory["backing"] == backing, memory
            assert memory["retained"] == (
                {
                    "limit_bytes": memory_limit or None,
                    "reserved_bytes": 0,
                    "committed_bytes": 0,
                    "peak_bytes": 0,
                    "released_bytes": 0,
                }
                if backing == "elastic"
                else None
            ), memory
            assert memory["workspace"] == {
                "reserved_bytes": 0,
                "committed_bytes": 0,
                "free_bytes": 0,
                "quarantined_bytes": 0,
                "budget_limit": 0,
                "reservation_count": 0,
                "slab_count": 0,
                "reserve_count": 0,
                "release_count": 0,
                "reuse_count": 0,
                "reuse_miss_count": 0,
                "fresh_count": 0,
                "exhausted_count": 0,
                "over_budget_count": 0,
                "wait_count": 0,
                "quarantine_eviction_count": 0,
            }, memory
            for target in (0, 2**64 - 1):
                assert client.request(
                    "/v1/devices/trim", {"device": 0, "target_bytes": target}
                ) == {"target_bytes": target, "device": actual}
            for value in (
                {},
                {"device": 1, "target_bytes": 0},
                {"device": 0, "target_bytes": -1},
                {"device": 0, "target_bytes": 1.5},
                {"device": 0, "target_bytes": 2**64},
            ):
                client.request("/v1/devices/trim", value, status=400)
            for target in ("/v1/embeddings", "/tokenize", "/detokenize"):
                rejected = client.request(target, {"model": "absent"}, status=404)
                assert rejected == {
                    "error": {
                        "message": "model is not registered",
                        "type": "loom_route_error",
                    }
                }
            observer = LogClient("http://" + address, timeout=None)
            assert observer.models() == []
            batch = observer.poll()
            counters = next(
                record["values"]
                for record in batch.records
                if record["event"] == "counters"
            )
            assert counters and all(value == 0 for value in counters.values())
            metrics = client.request("/metrics").splitlines()
            assert "loom_serve_server_counters_available 1" in metrics
            assert not any(line.startswith("loom_serve_model_") for line in metrics)
            for name in counters:
                assert f"loom_serve_server_{name}_total 0" in metrics
        locators = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.startswith('{"event":"ready"')
        ]
        assert locators == [{"event": "ready", "address": address, "models": 0}]
        print(
            json.dumps(
                {
                    "event": "empty_server_joined",
                    "catalog": mode,
                    "resources": resources,
                }
            ),
            flush=True,
        )


def thread_snapshot(process):
    # Linux exposes kernel-created io_uring workers in the process task list.
    # PF_IO_WORKER (include/linux/sched.h) distinguishes their elastic lifetime
    # from the userspace threads owned by serving and its native dependencies.
    snapshot = {"userspace": {}, "kernel_io": {}, "retired_during_sample": []}
    for entry in Path(f"/proc/{process.pid}/task").iterdir():
        try:
            state = (entry / "stat").read_text()
        except FileNotFoundError:
            # An I/O worker may retire between enumeration and the stat read.
            # Retain that discontinuity; a lost required userspace identity
            # still fails the caller's equality check.
            snapshot["retired_during_sample"].append(entry.name)
            continue
        end = state.rindex(")")
        flags = int(state[end + 2 :].split()[6])
        kind = "kernel_io" if flags & 0x10 else "userspace"
        snapshot[kind][entry.name] = state[state.index("(") + 1 : end]
    return snapshot


def check_sporadic_connections(process, client, model, workers):
    expected_threads = thread_snapshot(process)

    address = urlsplit("http://" + client.address)
    peers = []
    try:
        for target_count in (64, 256):
            while len(peers) < target_count:
                peer = socket.create_connection((address.hostname, address.port))
                peers.append(peer)
                # A partial header retains the connection but creates no
                # application request or worker task waiting on this client.
                peer.sendall(b"GET /healthz HTTP/1.1\r\nHost: local\r\n")
            vectors = client.request(
                "/v1/embeddings",
                {"model": model, "input": ["ab", "bbb"], "dimensions": 2},
            )
            assert [item["embedding"] for item in vectors["data"]] == [[3, 2], [6, 3]]
            current_threads = thread_snapshot(process)
            assert (
                current_threads["userspace"].keys()
                == expected_threads["userspace"].keys()
            ), (
                expected_threads,
                current_threads,
            )
        # Complete the most recently connected peers first; incomplete earlier
        # requests cannot hold later clients or inference behind them.
        completed = 0
        for offset in range(len(peers), 0, -16):
            batch = peers[max(0, offset - 16) : offset]
            for peer in batch:
                peer.sendall(b"\r\n")
            for peer in batch:
                response = http.client.HTTPResponse(peer)
                try:
                    response.begin()
                    assert response.status == 200
                    assert json.loads(response.read()) == {"status": "ready"}
                    completed += 1
                finally:
                    response.close()
                    peer.close()
            assert client.request("/readyz") == {"status": "ready"}
        assert completed == 256
        current_threads = thread_snapshot(process)
        assert (
            current_threads["userspace"].keys() == expected_threads["userspace"].keys()
        ), (
            expected_threads,
            current_threads,
        )
        print(
            json.dumps(
                {
                    "event": "sporadic_connections_joined",
                    "connections": completed,
                    "task_workers": workers,
                    "userspace_threads": len(current_threads["userspace"]),
                    "kernel_io_workers": len(current_threads["kernel_io"]),
                    "added_userspace_threads": 0,
                }
            ),
            flush=True,
        )
    finally:
        for peer in peers:
            peer.close()


def check_dynamic_catalog(server_command, directory, catalog, workers):
    """Twelve real model owners share workers, clients and retirement accounting."""
    declarations = [
        {
            **model,
            "name": f"{model['definition']['kind']}-{index}",
            "admission": "drained",
        }
        for index in range(3)
        for model in catalog["models"]
    ]
    log_path = directory / f"dynamic-catalog-{workers}.log"
    arguments = [
        "--listen=127.0.0.1:0",
        f"--workers={workers}",
        "--model_capacity=12",
        "--connections=320",
        "--heartbeat_ms=0",
    ]
    if workers == 4:
        # Process settings are separate from the dynamically registered catalog.
        settings = directory / "configuration" / "serve flags.txt"
        settings.write_text(
            "# Shared process resources\n" + "\n".join(arguments) + "\n"
        )
        arguments = [f"--flagfile={settings.relative_to(directory)}"]
    with (
        child_server(
            [*server_command, *arguments],
            log_path,
            cwd=directory,
        ) as (process, address),
        contextlib.ExitStack() as observations,
    ):
        assert address is not None, log_path.read_text()
        client = Client(address)
        initial_threads = thread_snapshot(process)
        assert client.request("/v1/models")["data"] == []
        initial_status = client.request("/v1/status")["server"]
        assert initial_status["worker_count"] == workers, initial_status
        assert initial_status["model_capacity"] == 12, initial_status
        assert initial_status["connection_capacity"] == 320, initial_status
        for declaration in declarations:
            client.request("/v1/models/register", declaration, status=202)
        client.request(
            "/v1/models/register",
            {**declarations[0], "name": "beyond-capacity"},
            status=429,
        )
        registered = client.request("/v1/status")["data"]
        identities = {item["id"]: item["loom"]["incarnation"] for item in registered}
        assert len(identities) == 12
        assert all(
            item["loom"]["registration"] == "registered"
            and item["loom"]["status"]["admission"] == "drained"
            for item in registered
        ), registered
        # Metadata-only registrations are already cold; deactivation must not
        # construct a model, prepare sources or change its registration identity.
        client.request(
            "/v1/models/deactivate", {"model": declarations[0]["name"]}, status=202
        )
        assert client.request("/v1/status")["data"] == registered
        # Retention policy can precede native preparation and does not warm or
        # open any modality. Every declared model uses the same owner contract.
        for declaration in declarations:
            pinned = client.request("/v1/models/pin", {"model": declaration["name"]})[
                "loom"
            ]
            assert pinned["parameter_pinned"], pinned
            assert pinned["registration"] == "registered", pinned
            assert pinned["parameter_operation"] == "idle", pinned
            assert "parameters" not in pinned, pinned
            assert "compilation" not in pinned, pinned
        registered_threads = thread_snapshot(process)
        assert (
            registered_threads["userspace"].keys()
            == initial_threads["userspace"].keys()
        ), (
            initial_threads,
            registered_threads,
        )

        # Preparation and parameter I/O need no inference or open admission.
        # Submit all twelve intents before awaiting any individual completion.
        for declaration in declarations:
            client.request(
                "/v1/models/activate", {"model": declaration["name"]}, status=202
            )
        for declaration in declarations:
            warmed = client.wait_activation(declaration["name"])
            assert warmed["incarnation"] == identities[declaration["name"]], warmed
            assert warmed["parameter_pinned"], warmed
            selector = {"model": declaration["name"]}
            client.request("/v1/models/deactivate", selector, status=409)
            unpinned = client.request("/v1/models/unpin", selector)["loom"]
            assert not unpinned["parameter_pinned"], unpinned
            assert unpinned["parameters"] == warmed["parameters"], unpinned
        reader = LogClient("http://" + address, timeout=None)
        warm_counters = next(
            record["values"]
            for record in reader.poll().records
            if record["event"] == "counters"
        )
        assert warm_counters["requests"] == warm_counters["batches"] == 0, warm_counters
        print(
            json.dumps(
                {
                    "event": "drained_catalog_activated",
                    "models": len(declarations),
                    "workers": workers,
                    "inference_requests": warm_counters["requests"],
                    "inference_batches": warm_counters["batches"],
                    "all_admission_closed": True,
                }
            ),
            flush=True,
        )

        def infer(declaration, iteration):
            model = declaration["name"]
            kind = declaration["definition"]["kind"]
            if kind == "text":
                messages = [{"role": "user", "content": "abx" * (iteration + 1)}]
                result = client.request(
                    "/v1/chat/completions",
                    {**Client.text_payload(messages, 6), "model": model},
                    headers={"X-Loom-Session": "shared-name"},
                )
                assert (result["text"], result["finish"]) == complete(messages, 6)
            elif kind == "image":
                prompt = "ab" + "a" * (iteration % 2)
                result = client.request(
                    "/v1/images/generations",
                    {
                        "model": model,
                        "prompt": prompt,
                        "seed": iteration,
                        "response_format": "b64_json",
                    },
                )
                check_png(result["data"][0]["b64_json"], prompt, iteration)
            elif kind == "embedding":
                result = client.request(
                    "/v1/embeddings",
                    {"model": model, "input": ["ab", "bbb"], "dimensions": 2},
                )
                assert [row["embedding"] for row in result["data"]] == [[3, 2], [6, 3]]
            elif kind == "decision":
                result = client.request(
                    "/v1/systemone",
                    {
                        "model": model,
                        "state": "a",
                        "questions": {
                            "truth": {
                                "type": "noul",
                                "instructions": "???",
                                "criteria": {"false": "", "true": "x"},
                            }
                        },
                    },
                )
                check_decision(result)
            else:
                raise AssertionError(kind)

        # Activate the shared device queue with one real request before checking
        # model-count scaling. Vulkan starts its queue completion thread lazily.
        # Names may change during native initialization; identities must remain.
        first = next(
            item for item in declarations if item["definition"]["kind"] == "embedding"
        )
        observer = observations.enter_context(
            telemetry_observation(http.client.HTTPConnection(address), first["name"])
        )
        assert observer.counters["requests"] == observer.counters["completed"] == 0
        client.request("/v1/models/undrain", {"model": first["name"]})
        infer(first, 0)
        observer.completed(1)
        wait_status(
            urlsplit("http://" + address),
            model=first["name"],
            phase="idle",
            queued_requests=0,
            active_requests=0,
        )
        activated_threads = thread_snapshot(process)
        assert (
            initial_threads["userspace"].keys() <= activated_threads["userspace"].keys()
        ), (
            initial_threads,
            activated_threads,
        )
        for declaration in declarations:
            if declaration is not first:
                client.request("/v1/models/undrain", {"model": declaration["name"]})

        with ThreadPoolExecutor(max_workers=12) as executor:
            work = [
                executor.submit(infer, declaration, 0) for declaration in declarations
            ]
            for future in work:
                future.result()
            client.idle()
            observer.completed(2)
            loaded_threads = thread_snapshot(process)
            assert (
                loaded_threads["userspace"].keys()
                == activated_threads["userspace"].keys()
            ), (
                activated_threads,
                loaded_threads,
            )
            check_sporadic_connections(process, client, "embedding-0", workers)

            # Keep independent model work live while the first copy of each
            # modality retires. No client, public name or model kind owns peers.
            work = [
                executor.submit(infer, declaration, 1)
                for declaration in declarations[4:]
            ]
            for declaration in declarations[:4]:
                client.request(
                    "/v1/models/unregister", {"model": declaration["name"]}, status=202
                )
            for future in work:
                future.result()
            peers = {declaration["name"] for declaration in declarations[4:]}
            while True:
                discovered = client.request("/v1/status")["data"]
                assert all(
                    item["loom"]["registration"] != "release_failed"
                    for item in discovered
                ), discovered
                if {item["id"] for item in discovered} == peers:
                    break
            # Retirement has already reclaimed the model; a lagging observer
            # owns only copied output and its connection, not that model/ring.
            while observer.read():
                pass
            for declaration in declarations[:4]:
                client.request(
                    "/v1/models/register",
                    {**declaration, "admission": "open"},
                    status=202,
                )
            replaced = client.request("/v1/status")["data"]
            for item in replaced:
                if item["id"] not in peers:
                    assert item["loom"]["incarnation"] != identities[item["id"]]
                    assert not item["loom"]["parameter_pinned"], item
            work = [
                executor.submit(infer, declaration, 2)
                for declaration in declarations[:4]
            ]
            for future in work:
                future.result()
        client.idle()
        reader = LogClient("http://" + address, timeout=None)
        before = next(
            record["values"]
            for record in reader.poll().records
            if record["event"] == "counters"
        )
        # One activation request, twelve initial requests, eight peers, four
        # replacements and two embedding probes while transport requests park.
        assert before["requests"] == before["completed"] == 27, before
        for index, declaration in enumerate(declarations):
            client.request(
                "/v1/models/unregister",
                {
                    "model": declaration["name"],
                    "mode": "drain" if index % 2 else "abort",
                },
                status=202,
            )
        while True:
            discovered = client.request("/v1/status")["data"]
            assert all(
                item["loom"]["registration"] != "release_failed" for item in discovered
            ), discovered
            if not discovered:
                break
        after = next(
            record["values"]
            for record in reader.poll().records
            if record["event"] == "counters"
        )
        assert before == after, (before, after)
        retired_status = client.request("/v1/status")["server"]
        assert retired_status["resources"]["listeners"] == [
            {"address": address, "waiting_for_capacity": False}
        ]
        for device in retired_status["resources"]["devices"]:
            retained = device["memory"]["retained"]
            if retained is not None:
                assert retained["reserved_bytes"] == retained["committed_bytes"] == 0
        final_threads = thread_snapshot(process)
        assert (
            final_threads["userspace"].keys() == activated_threads["userspace"].keys()
        ), (
            activated_threads,
            final_threads,
        )
        assert client.request("/healthz") == {"status": "ready"}
        print(
            json.dumps(
                {
                    "event": "dynamic_catalog_joined",
                    "models": len(declarations),
                    "workers": workers,
                    "userspace_threads": len(final_threads["userspace"]),
                    "kernel_io_workers": len(final_threads["kernel_io"]),
                    "activation_added_userspace_threads": len(
                        activated_threads["userspace"]
                    )
                    - len(initial_threads["userspace"]),
                    "added_userspace_threads": 0,
                    "completed_requests": after["completed"],
                    "outputs_verified": True,
                    "retired_counters_preserved": True,
                    "sse_model_lifetime_verified": True,
                    "thread_samples": [
                        initial_threads,
                        activated_threads,
                        loaded_threads,
                        final_threads,
                    ],
                }
            ),
            flush=True,
        )


def check_multiple_bindings(server_command, directory, catalog):
    socket_path = directory / 'serve "local".sock'
    log_path = directory / "multiple-bindings.log"
    settings = directory / "configuration" / "listeners.flags"
    settings.write_text(
        "\n".join(
            [
                f"--flagfile={catalog.relative_to(directory)}",
                "--listen=127.0.0.1:0",
                f"--listen=unix:{socket_path}",
                "--workers=1",
                "--connections=320",
                "--heartbeat_ms=0",
            ]
        )
        + "\n"
    )
    with child_server(
        [*server_command, f"--flagfile={settings.relative_to(directory)}"],
        log_path,
        cwd=directory,
    ) as (process, address):
        assert address is not None, log_path.read_text()
        client = Client(address)
        client.idle()
        expected_models = client.request("/v1/models")
        resources = client.request("/v1/status")["server"]["resources"]
        assert resources["listeners"] == [
            {"address": address, "waiting_for_capacity": False},
            {"address": "unix:" + str(socket_path), "waiting_for_capacity": False},
        ], resources
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.connect(str(socket_path))
            peer.sendall(b"GET /v1/status HTTP/1.1\r\nHost: local\r\n\r\n")
            with http.client.HTTPResponse(peer) as response:
                response.begin()
                assert response.status == 200
                unix_resources = json.loads(response.read())["server"]["resources"]
        assert unix_resources["listeners"] == resources["listeners"]
        for tcp_device, unix_device in zip(
            resources["devices"], unix_resources["devices"], strict=True
        ):
            # Atomic workspace observations may change between owner turns.
            assert {
                key: value for key, value in tcp_device.items() if key != "memory"
            } == {key: value for key, value in unix_device.items() if key != "memory"}
        connection = http.client.HTTPConnection("localhost")
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.sock.connect(str(socket_path))
            connection.request(
                "POST",
                "/v1/devices/trim",
                json.dumps({"device": 0, "target_bytes": 0}),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert response.status == 200
            trimmed = json.loads(response.read())
        finally:
            connection.close()
        assert trimmed["target_bytes"] == trimmed["device"]["index"] == 0, trimmed
        assert [
            (item["id"], item["loom"]["incarnation"])
            for item in client.request("/v1/models")["data"]
        ] == [
            (item["id"], item["loom"]["incarnation"])
            for item in expected_models["data"]
        ]
        for model in expected_models["data"]:
            name = model["id"]
            headers = {"X-Loom-Model": name}
            for target in ("/v1/requests", "/v1/sessions"):
                expected_status = (
                    501
                    if target == "/v1/sessions" and model["loom"]["kind"] != "text"
                    else 200
                )
                tcp_inventory = client.request(
                    target, headers=headers, status=expected_status
                )
                connection = http.client.HTTPConnection("localhost")
                connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    connection.sock.connect(str(socket_path))
                    connection.request("GET", target, headers=headers)
                    response = connection.getresponse()
                    assert response.status == expected_status
                    unix_inventory = json.loads(response.read())
                finally:
                    connection.close()
                # Captures are distinct owner turns; identity and empty
                # ownership agree, not the monotonic capture timestamp.
                assert {
                    key: value
                    for key, value in tcp_inventory.items()
                    if key != "time_ns"
                } == {
                    key: value
                    for key, value in unix_inventory.items()
                    if key != "time_ns"
                }, (tcp_inventory, unix_inventory)
                if expected_status == 200:
                    assert tcp_inventory["data"] == []
                    assert tcp_inventory["next_offset"] is None
        observation_connection = http.client.HTTPConnection("localhost")
        observation_connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        observation_connection.sock.connect(str(socket_path))
        connection = http.client.HTTPConnection("localhost")
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with telemetry_observation(observation_connection, "vectors") as observer:
            try:
                connection.sock.connect(str(socket_path))
                connection.request(
                    "POST",
                    "/v1/embeddings",
                    json.dumps(
                        {"model": "vectors", "input": ["a", "bbb"], "dimensions": 2}
                    ),
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                assert response.status == 200
                result = json.loads(response.read())
                assert [entry["embedding"] for entry in result["data"]] == [
                    [1, 1],
                    [6, 3],
                ]
                assert result["usage"]["prompt_tokens"] == 4
                observer.completed(1)
                assert observer.counters["input_tokens"] == 4, observer.counters
            finally:
                connection.close()
        # The request submitted over Unix belongs to the same model observed
        # over TCP, not a second independently constructed deployment.
        wait_counters(
            urlsplit("http://" + address),
            lambda counts: counts["completed"] == 1,
            model="vectors",
        )
        assert client.request("/v1/models") == expected_models
        check_sporadic_connections(process, client, "vectors", 1)
    assert not socket_path.exists()
    locators = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.startswith('{"event":"ready"')
    ]
    assert [locator["address"] for locator in locators] == [
        address,
        "unix:" + str(socket_path),
    ]
    print(json.dumps({"event": "multiple_bindings_joined", "bindings": 2}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--device", default="amdgpu")
    parser.add_argument("--pool_backing", choices=("fixed", "elastic"), default="fixed")
    parser.add_argument("--rounds", type=int, default=4, choices=range(1, 17))
    arguments = parser.parse_args()
    server_command = [
        str(arguments.server.absolute()),
        f"--device={arguments.device}",
        f"--pool_backing={arguments.pool_backing}",
    ]
    with tempfile.TemporaryDirectory(prefix="loom catalog ") as temporary:
        directory = Path(temporary)
        check_empty_server(
            server_command, directory, arguments.device, arguments.pool_backing
        )
        weights = directory / "text.safetensors"
        header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        weights.write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<I", 2)
        )
        catalog = {"models": []}
        for fixture in arguments.fixture:
            kind = json.loads(fixture.read_text())["definition"]["kind"]
            package = copy_package(fixture.parent, directory, kind)
            model = json.loads((directory / package / "model.json").read_text())
            model["base_directory"] = str(directory / package)
            if kind == "text":
                model["definition"].update(
                    weights=str(weights),
                    rows=2,
                    context_capacity=128,
                    pool_capacity=128,
                    checkpoint_capacity=2,
                )
            catalog["models"].append(model)
        catalog_path = directory / "configuration" / "models.flags"
        catalog_path.parent.mkdir()
        write_startup(catalog_path, catalog)
        check_dynamic_observer(server_command, directory, catalog)
        check_startup_failures(server_command, directory, catalog)
        check_multiple_bindings(server_command, directory, catalog_path)
        for workers in (1, 4):
            check_dynamic_catalog(server_command, directory, catalog, workers)
            with running_server(server_command, directory, catalog_path, workers) as (
                client,
                observations,
            ):
                assert client.request("/readyz") == {"status": "ready"}
                discovered = client.request("/v1/models")
                assert sorted(item["id"] for item in discovered["data"]) == sorted(
                    [
                        "pixels",
                        "vectors",
                        "judge",
                        "fixture-text",
                    ]
                )
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
                check_parameter_residency(client, arguments.pool_backing, workers)
                check_device_trim(client, arguments.pool_backing, workers)
            print(
                json.dumps(
                    {
                        "event": "catalog_joined",
                        "workers": workers,
                        "external_sources": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
