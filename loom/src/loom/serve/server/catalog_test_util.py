# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Mixed-model fixture deployment, HTTP clients and independent oracles."""

import base64
import contextlib
import http.client
import json
import shutil
import socket
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from loom_serve_testing.http import Events, wait_counters, wait_status
from loom_serve_testing.http import running_server as child_server
from loom_serve_testing.logs import LogClient


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


def prepare_catalog(fixtures, directory):
    """Prepare external model documents and a tiny retained-text checkpoint."""
    weights = directory / "text.safetensors"
    header = b'{"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
    header += b" " * (-len(header) % 8)
    weights.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<I", 2))
    catalog = {"models": []}
    for fixture in fixtures:
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
    return catalog, catalog_path


def check_counters(client, rounds):
    """Compare independent model snapshots, server totals and metrics."""
    url = "http://" + client.address
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
            counts[name] for name in ("completed", "rejected", "cancelled", "failed")
        ), counts
    for model in ("pixels", "vectors", "judge"):
        assert expected[model]["requests"] == expected[model]["completed"] == rounds, (
            expected[model]
        )
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
        assert client.request("/tokenize", {"model": "vectors", "content": "ab"}) == {
            "tokens": [0, 1]
        }
    return expected
