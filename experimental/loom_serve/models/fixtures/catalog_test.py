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


def check_decision(result):
    # The single-question input gives true/false weights of 2 and 1.
    answer = result["answers"]["truth"]
    assert set(answer) == {"type", "noul"}, result
    assert answer["type"] == "noul", result
    # Vulkan F32 division permits 2.5 ULP; at 2/3, one ULP is 2^-24.
    assert abs(answer["noul"] - 2 / 3) <= 2.5 * 2**-24, result


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
            f"--models={catalog.relative_to(directory)}",
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

    before = snapshot()
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
            if backing == "elastic":
                assert not parameters["active"], cold
                assert parameters["committed_bytes"] == 0, cold
                assert before["parameters"]["committed_bytes"] > 0, before
                assert parameters["reserved_bytes"] > 0, cold
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
            assert warmed["incarnation"] == before["incarnation"], warmed
            assert warmed["compilation"] == before["compilation"], warmed
            assert warmed["status"]["held_requests"] == 1, warmed
            assert warmed["parameters"]["activations"] == (
                before["parameters"]["activations"] + (backing == "elastic")
            ), warmed
            if backing == "elastic":
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


def copy_package(source, directory, kind):
    """Relocate a fixture with numerical providers outside its model directory.

    The separate prepare/control/weight-policy leaves and tokenizer stay in the
    model package. Only sources.txt locates the sibling numerical providers;
    neither the child's cwd nor its deployment catalog can supply that base.
    Copies dereference runfiles symlinks without preserving their read-only cache
    permissions, so the child reads independent, editable deployment files.
    """
    package = directory / "models" / kind
    shutil.copytree(source, package, copy_function=shutil.copyfile)
    providers = directory / "providers" / kind
    providers.mkdir(parents=True)
    catalog = package / "sources.txt"
    names = [
        line.strip()
        for line in catalog.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for name in names:
        (package / name).rename(providers / name)
    catalog.write_text("".join(f"../../providers/{kind}/{name}\n" for name in names))
    return package.relative_to(directory)


def check_startup_failures(server_command, directory, catalog):
    """Keep an incomplete registration inspectable while peers serve output."""
    models = catalog["models"]
    cases = [(model, Path(model["source"]) / "control.loom") for model in models]
    cases.append((models[-1], Path("providers/text/advance.loom")))
    for model, missing in cases:
        source = directory / missing
        saved = source.with_suffix(".saved")
        broken = {"models": [entry for entry in models if entry is not model] + [model]}
        name = f"missing-{model['kind']}-{missing.stem}"
        catalog_path = directory / "configuration" / f"{name}.json"
        catalog_path.write_text(json.dumps(broken))
        log_path = catalog_path.with_suffix(".log")
        source.rename(saved)
        try:
            with child_server(
                [
                    *server_command,
                    f"--models={catalog_path.relative_to(directory)}",
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
                if model["kind"] != "text":
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
        model for model in catalog["models"] if model["kind"] == "embedding"
    )
    log_path = directory / "dynamic-observer.log"
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
            "--workers=1",
        ],
        log_path,
        cwd=directory,
    ) as (_, address):
        assert address is not None, log_path.read_text()
        client = Client(address)
        reader = LogClient("http://" + address, timeout=None)
        sink = Sink()

        def observe():
            try:
                record_streams(reader, EventLog(sink), stopped, interval=0.01)
            finally:
                events.record(None)

        with ThreadPoolExecutor(max_workers=1) as executor:
            observer = executor.submit(observe)
            try:
                events.wait("observation_begin", models=[])
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
    print(json.dumps({"event": "dynamic_observer_followed_reincarnation"}), flush=True)


def check_empty_server(server_command, directory):
    catalog = directory / "empty.json"
    catalog.write_text('{"models":[]}')
    for mode, arguments in (
        ("omitted", []),
        ("empty", [f"--models={catalog.name}"]),
    ):
        log_path = directory / f"{mode}-models.log"
        with child_server(
            [
                *server_command,
                *arguments,
                "--listen=127.0.0.1:0",
                "--workers=1",
            ],
            log_path,
            cwd=directory,
        ) as (_, address):
            assert address is not None, log_path.read_text()
            client = Client(address)
            for target in ("/healthz", "/readyz"):
                assert client.request(target) == {"status": "ready"}
            for target in ("/v1/models", "/v1/status"):
                assert client.request(target) == {"object": "list", "data": []}
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
        print(json.dumps({"event": "empty_server_joined", "catalog": mode}), flush=True)


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
        {**model, "name": f"{model['kind']}-{index}", "admission": "drained"}
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
    with child_server(
        [*server_command, *arguments],
        log_path,
        cwd=directory,
    ) as (process, address):
        assert address is not None, log_path.read_text()
        client = Client(address)
        initial_threads = thread_snapshot(process)
        assert client.request("/v1/models")["data"] == []
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
            kind = declaration["kind"]
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
        first = next(item for item in declarations if item["kind"] == "embedding")
        client.request("/v1/models/undrain", {"model": first["name"]})
        infer(first, 0)
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
                discovered = client.request("/v1/models")["data"]
                assert all(
                    item["loom"]["registration"] != "release_failed"
                    for item in discovered
                ), discovered
                if {item["id"] for item in discovered} == peers:
                    break
            for declaration in declarations[:4]:
                client.request(
                    "/v1/models/register",
                    {**declaration, "admission": "open"},
                    status=202,
                )
            replaced = client.request("/v1/models")["data"]
            for item in replaced:
                if item["id"] not in peers:
                    assert item["loom"]["incarnation"] != identities[item["id"]]
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
            discovered = client.request("/v1/models")["data"]
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
                f"--models={catalog.relative_to(directory)}",
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
        connection = http.client.HTTPConnection("localhost")
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
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
            assert [entry["embedding"] for entry in result["data"]] == [[1, 1], [6, 3]]
            assert result["usage"]["prompt_tokens"] == 4
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
    parser.add_argument("--catalog", type=Path, required=True)
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
        check_empty_server(server_command, directory)
        weights = directory / "text.safetensors"
        header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        weights.write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<I", 2)
        )
        catalog = json.loads(arguments.catalog.read_text())
        for model in catalog["models"]:
            package = copy_package(Path(model["source"]), directory, model["kind"])
            model["source"] = str(package)
            for field in ("checkpoint", "assets"):
                if field in model:
                    model[field] = str(package)
        text = copy_package(arguments.catalog.parent / "text", directory, "text")
        catalog["models"].append(
            {
                "kind": "text",
                "name": "fixture-text",
                "source": str(text),
                "weights": weights.name,
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
        catalog_path = directory / "configuration" / "models.json"
        catalog_path.parent.mkdir()
        catalog_path.write_text(json.dumps(catalog))
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
