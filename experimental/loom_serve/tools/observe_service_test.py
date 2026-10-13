# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""The actual observer follows dynamic models and survives a blocked sink."""

import argparse
import contextlib
import io
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from catalog_test_util import (
    Client,
    check_counters,
    prepare_catalog,
    run_round,
    running_server,
)
from loom_serve_testing.http import Events
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
            expected = check_counters(client, rounds)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--pool_backing", choices=("fixed", "elastic"), required=True)
    parser.add_argument("--rounds", type=int, default=4, choices=range(1, 17))
    arguments = parser.parse_args()
    server_command = [
        str(arguments.server.absolute()),
        f"--device={arguments.device}",
        f"--pool_backing={arguments.pool_backing}",
    ]
    with tempfile.TemporaryDirectory(prefix="loom observer ") as temporary:
        directory = Path(temporary)
        catalog, catalog_path = prepare_catalog(arguments.fixture, directory)
        check_dynamic_observer(server_command, directory, catalog)
        for workers in (1, 4):
            with running_server(server_command, directory, catalog_path, workers) as (
                client,
                observations,
            ):
                with stalled_observer(client, arguments.rounds):
                    for iteration in range(arguments.rounds):
                        result = run_round(client, iteration, observations)
                        print(
                            json.dumps(
                                {
                                    "event": "observer_workload",
                                    "workers": workers,
                                    **result,
                                }
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
