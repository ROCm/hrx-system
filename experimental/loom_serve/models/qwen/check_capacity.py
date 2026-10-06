# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real TCP reservation, queued ownership, eviction and retained-output witness.

The caller supplies a built server and qualified GPU environment. Sequential
dense output is the oracle for the same requests under concurrent pooled load.
Events establish readiness; an outer runner owns any hang deadline.
"""

import argparse
import http.client
import json
import socket
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen import benchmark_service, check_service


class Events:
    def __init__(self):
        self.condition = threading.Condition()
        self.values = []
        self.closed = False

    def record(self, event):
        with self.condition:
            if event is None:
                self.closed = True
            else:
                self.values.append(event)
            self.condition.notify_all()

    def wait(self, kind, session):
        def match():
            return next(
                (
                    event
                    for event in self.values
                    if event.get("event") == kind and event.get("session") == session
                ),
                None,
            )

        with self.condition:
            self.condition.wait_for(lambda: match() or self.closed)
            event = match()
            if event is None:
                raise RuntimeError(f"server closed before {kind}: {session}")
            return event


def payload(messages, maximum):
    return json.dumps(
        {
            "model": "qwen3.8-27b",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": maximum,
        }
    )


def reject(address, session, messages, maximum, expected, diagnostic):
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            payload(messages, maximum),
            {"Content-Type": "application/json", "X-Loom-Session": session},
        )
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != expected or diagnostic not in body:
            raise RuntimeError(f"wrong rejection: {response.status}: {body}")
    finally:
        connection.close()


def messages(background=0):
    return [
        {"role": "system", "content": "Follow instructions exactly. Do not use tools."},
        {
            "role": "user",
            "content": (
                "Background: this repository has kernels and documentation.\n"
                * background
                + "Count from 1 through 1000, one number per line, without commentary."
            ),
        },
    ]


def retained(address, pooled):
    history = [
        {"role": "system", "content": "Follow instructions exactly. Do not use tools."},
        {"role": "user", "content": "Remember my codeword MAPLE. Reply with only OK."},
    ]
    first = benchmark_service.request(address, "retained", history, 16)
    history.extend(
        [
            {"role": "assistant", "content": first["text"]},
            {
                "role": "user",
                "content": "What is my codeword? Reply with only that word.",
            },
        ]
    )
    if pooled:
        # An impossible request must leave the existing checkpoint intact.
        reject(address, "retained", history, 512, 400, "shared pool holds")
    second = benchmark_service.request(address, "retained", history, 16)
    if second["text"].strip() != "MAPLE":
        raise RuntimeError(f"retained content mismatch: {second['text']!r}")
    if not second["usage"]["prompt_tokens_details"]["cached_tokens"]:
        raise RuntimeError("rejected admission discarded the retained checkpoint")
    return [first, second]


def cohort(address, events, pooled):
    requests = [(messages(), 192)] + [
        (messages(index % 4), (32, 48, 64)[index % 3]) for index in range(6)
    ]
    if not pooled:
        return [
            benchmark_service.request(address, f"row-{index}", prompt, maximum)
            for index, (prompt, maximum) in enumerate(requests)
        ]
    with ThreadPoolExecutor(max_workers=len(requests)) as clients:
        anchor = clients.submit(
            benchmark_service.request, address, "row-0", *requests[0]
        )
        admitted = events.wait("admit", "row-0")
        if admitted["reservation_tokens"] * 2 <= 320:
            raise RuntimeError(
                "anchor does not establish the constrained-capacity witness"
            )
        # This identical request cannot fit beside the anchor. Its confirmed
        # queued claim is then cancelled without ever taking a recurrent row.
        cancelled = http.client.HTTPConnection(address.hostname, address.port)
        try:
            cancelled.request(
                "POST",
                "/v1/chat/completions",
                payload(*requests[0]),
                {"Content-Type": "application/json", "X-Loom-Session": "cancelled"},
            )
            events.wait("enqueue", "cancelled")
            reject(address, "cancelled", *requests[0], 409, "active or queued")
            cancelled.sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        finally:
            cancelled.close()
        events.wait("cancel_queued", "cancelled")
        futures = [anchor] + [
            clients.submit(
                benchmark_service.request, address, f"row-{index}", prompt, maximum
            )
            for index, (prompt, maximum) in enumerate(requests[1:], 1)
        ]
        return [future.result() for future in futures]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "weights", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--continuation-epochs", type=int, choices=(1, 2), default=1)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = [str(arguments.server.resolve())] + [
        f"--{name}={getattr(arguments, name).resolve()}"
        for name in ("model", "weights", "tokenizer")
    ]
    command += [
        "--rows=4",
        "--context_capacity=2048",
        "--prefill_capacity=128",
        "--epoch=32:4",
        "--epoch=128:4",
        "--mtp",
        "--mtp_depth=3",
        "--port=0",
        "--connections=24",
        "--pending_requests=8",
        "--heartbeat_ms=1000",
    ]
    reference = None
    for capacity in (0, 320):
        events = Events()
        log_path = arguments.output / f"pool-{capacity}.log"
        continuation_epochs = arguments.continuation_epochs if capacity else 1
        with check_service.running_server(
            command
            + [
                f"--pool_capacity={capacity}",
                f"--continuation_epochs={continuation_epochs}",
            ],
            log_path,
            events.record,
        ) as (_, endpoint):
            if endpoint is None:
                raise RuntimeError(f"server did not become ready: {log_path}")
            address = urlsplit(f"http://{endpoint}")
            results = cohort(address, events, capacity != 0)
            results.extend(retained(address, capacity != 0))
        (arguments.output / f"pool-{capacity}.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        outputs = [
            {key: row[key] for key in ("text", "usage", "finish_reason")}
            for row in results
        ]
        if reference is None:
            reference = outputs
        elif outputs != reference:
            raise RuntimeError(
                "pooled concurrent output differs from sequential dense output"
            )
        if capacity:
            if arguments.continuation_epochs == 2 and not any(
                row["verification_epochs"] == 2
                for event in events.values
                if event.get("event") == "epoch"
                for row in event["rows"]
            ):
                raise RuntimeError("workload did not exercise device continuation")
            admissions = [e for e in events.values if e.get("event") == "admit"]
            if any(e["pool_reserved_tokens"] > capacity for e in admissions):
                raise RuntimeError("completion reservations exceeded physical capacity")
            if any(e.get("session") == "cancelled" for e in admissions):
                raise RuntimeError("cancelled queued request acquired a model row")
            if not any(e.get("event") == "evict" for e in events.values):
                raise RuntimeError("workload did not exercise idle cache eviction")
            heartbeats = [e for e in events.values if e.get("event") == "heartbeat"]
            if any(
                max(e["pool"]["reserved_tokens"], e["pool"]["resident_tokens"])
                > capacity
                for e in heartbeats
            ):
                raise RuntimeError("pool accounting exceeded the configured budget")
            if not any(e["queued_requests"] for e in heartbeats):
                raise RuntimeError("workload did not observe queued admission")
        print(
            json.dumps({"event": "pass", "pool_capacity": capacity, "requests": 9}),
            flush=True,
        )


if __name__ == "__main__":
    main()
