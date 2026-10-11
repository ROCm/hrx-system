# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""HTTP model-check ownership and explicitly fallible telemetry history.

The test harness owns its child, unlike the independent production observer.
Only the cold ready locator comes from stderr. Model readiness comes from HTTP
responses/status/counters; retained history can establish coverage, but a gap
invalidates that evidence rather than changing the server's retention policy.
The outer runner owns hang detection for valid submitted work.
"""

import contextlib
import http.client
import json
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from experimental.loom_serve.http.logs import LogClient


def control(address, method, path, expected, *, session=None, body=None, headers=None):
    connection = http.client.HTTPConnection(address.hostname, address.port)
    fields = dict(headers or {})
    if session is not None:
        fields["X-Loom-Session"] = session
    try:
        connection.request(method, path, body=body, headers=fields)
        response = connection.getresponse()
        payload = response.read()
        if response.status != expected:
            raise RuntimeError(f"{method} {path}: {response.status}, {payload!r}")
        return json.loads(payload)
    finally:
        connection.close()


def wait_status(address, *, model=None, **fields):
    """Wait for current owner state, independently of lossy retained events."""
    while True:
        models = control(address, "GET", "/v1/status", 200)["data"]
        if model is None:
            if len(models) != 1:
                raise ValueError("status wait requires a model in a routed server")
            selected = models[0]
        else:
            selected = next(item for item in models if item["id"] == model)
        state = selected["loom"]["status"]
        if all(state[key] == value for key, value in fields.items()):
            return state


def wait_counters(address, predicate, *, model=None):
    """Wait on always-counted outcomes, not delivery of request-end events."""
    client = LogClient(address.geturl(), timeout=None)
    if model is None:
        models = client.models()
        if len(models) != 1:
            raise ValueError("counter wait requires a model in a routed server")
        model = models[0]
    while True:
        batch = client.poll(model)
        counters = next(
            record["values"]
            for record in batch.records
            if record["event"] == "counters"
        )
        if predicate(counters):
            return counters


class Events:
    """A check's retained evidence, never a promise of lossless publication."""

    def __init__(self):
        self.condition = threading.Condition()
        self._values = []
        self.closed = False
        self.failure = None

    def record(self, event):
        with self.condition:
            if event is None:
                self.closed = True
            else:
                self._values.append(event)
                if event["event"] in ("gap", "stream_reset"):
                    self.failure = RuntimeError(
                        f"incomplete telemetry history cannot prove coverage: {event}"
                    )
            self.condition.notify_all()

    @property
    def values(self):
        with self.condition:
            if self.failure:
                raise self.failure
            return list(self._values)

    def wait(self, kind, session=None, *, model=None, **fields):
        """Inspect a native record; optional session names resolve through starts.

        This is suitable for inspecting an already completed HTTP operation.
        Scheduling and cancellation gates use status/counters instead. A missed
        record or closed observer fails explicitly, rather than waiting on a
        session name that the producer no longer carries on every event.
        """
        with self.condition:
            while True:
                if self.failure:
                    raise self.failure
                names = {
                    (event.get("model"), event["session"]): event["name"]
                    for event in self._values
                    if event["event"] == "session_begin"
                }
                for event in self._values:
                    if (
                        event["event"] == kind
                        and (model is None or event.get("model") == model)
                        and (
                            session is None
                            or names.get((event.get("model"), event.get("session")))
                            == session
                        )
                        and all(
                            event.get(key) == value for key, value in fields.items()
                        )
                    ):
                        return event
                if self.closed:
                    raise RuntimeError(f"observer closed before {kind}: {session}")
                self.condition.wait()


def collect_events(client, models, stopped, record, *, ready=None):
    """Capture native history, then finish through one fixed head per stream.

    Poll envelopes and fresh counter probes are not history. Omitting them keeps
    idle polling from growing a check artifact; wait_counters reads owner totals
    independently. Native records and explicit gaps/resets retain their identity.
    """
    try:
        while True:
            finishing = stopped.is_set()
            for model in models:
                batch = client.poll(model)
                identity = batch.stream["identity"]
                # A fixed fence drains responses capped below the ring capacity
                # without following a heartbeat producer forever at shutdown.
                head = next(
                    event["head"]
                    for event in batch.records
                    if event["event"] == "stream"
                )
                while True:
                    for event in batch.records:
                        if event["event"] not in ("stream", "cursor", "counters"):
                            record({**event, **batch.stream, "model": model})
                    if batch.stream["identity"] != identity:
                        raise RuntimeError("stream restarted while capturing its head")
                    if batch.records[-1]["next"] >= head:
                        break
                    batch = client.poll(model)
            if ready is not None:
                ready.set()
            if finishing:
                break
            stopped.wait(0.01)
    finally:
        try:
            record(None)
        finally:
            if ready is not None:
                ready.set()


@contextlib.contextmanager
def running_server(command, log_path, on_event=None, *, cwd=None):
    """Own a child and optional independent HTTP capture through joined shutdown.

    log_path holds cold stdout/stderr diagnostics. Its .events.jsonl sibling
    holds scoped native /logz records. Both are test artifacts, not server sinks.
    Observation ends before shutdown: teardown-only records are not promised by
    a pull endpoint whose lifetime ends with the server.
    An explicit cwd selects the child's deployment root without changing the
    client's working directory or interpreting its catalog paths.
    """
    with log_path.open("x") as log, ThreadPoolExecutor(max_workers=2) as readers:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        address = None
        drain = observer = None
        stopped = threading.Event()
        attached = threading.Event()

        def consume():
            for line in process.stderr:
                log.write(line)
                log.flush()

        def observe(client, models):
            try:
                with log_path.with_suffix(".events.jsonl").open("x") as output:

                    def record(event):
                        if event is not None:
                            output.write(json.dumps(event) + "\n")
                            output.flush()
                        on_event(event)

                    collect_events(client, models, stopped, record, ready=attached)
            finally:
                attached.set()

        try:
            for line in process.stderr:
                log.write(line)
                log.flush()
                if line.startswith('{"event":"ready"'):
                    address = json.loads(line)["address"]
                    drain = readers.submit(consume)
                    if on_event:
                        client = LogClient(f"http://{address}", timeout=None)
                        observer = readers.submit(
                            observe, client, [None, *client.models()]
                        )
                        attached.wait()
                        if observer.done():
                            observer.result()
                    break
            yield process, address
        finally:
            # Close the HTTP consumer while the server can still answer it. Its
            # error cannot bypass termination and joining of the child we own.
            try:
                stopped.set()
                if observer:
                    observer.result()
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                returncode = process.wait()
                try:
                    if drain:
                        drain.result()
                finally:
                    process.stderr.close()
                if address is not None and returncode:
                    raise RuntimeError(
                        f"server retirement failed: {returncode}; {log_path}\n"
                        f"{log_path.read_text()}"
                    )
