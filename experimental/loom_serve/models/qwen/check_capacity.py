# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real TCP reservation, queued ownership, eviction and retained-output witness.

The caller supplies a built server and qualified GPU environment. Sequential
dense output is the oracle for the same requests under concurrent pooled load.
HTTP owner state establishes readiness; an outer runner owns hang detection.
"""

import argparse
import http.client
import json
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.http import chat
from experimental.loom_serve.testing.http import (
    Events,
    running_server,
    wait_counters,
    wait_status,
)


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
    first = chat.request(address, "retained", history, 16, model="qwen3.8-27b")
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
    second = chat.request(address, "retained", history, 16, model="qwen3.8-27b")
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
            chat.request(address, f"row-{index}", prompt, maximum, model="qwen3.8-27b")
            for index, (prompt, maximum) in enumerate(requests)
        ]
    with ThreadPoolExecutor(max_workers=len(requests)) as clients:
        anchor = clients.submit(
            chat.request, address, "row-0", *requests[0], model="qwen3.8-27b"
        )
        wait_status(address, active_requests=1, queued_requests=0)
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
            wait_status(address, active_requests=1, queued_requests=1)
            reject(address, "cancelled", *requests[0], 409, "active or queued")
            cancelled.sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        finally:
            cancelled.close()
        wait_counters(address, lambda counts: counts["cancelled"] == 1)
        admitted = events.wait("admit", "row-0")
        if admitted["reservation_tokens"] * 2 <= 320:
            raise RuntimeError(
                "anchor does not establish the constrained-capacity witness"
            )
        futures = [anchor] + [
            clients.submit(
                chat.request,
                address,
                f"row-{index}",
                prompt,
                maximum,
                model="qwen3.8-27b",
            )
            for index, (prompt, maximum) in enumerate(requests[1:], 1)
        ]
        return [future.result() for future in futures]


def cancel_active(address):
    # Cancel only after the GPU has produced actual content. Header readiness
    # alone would permit cancellation before any row backing was committed.
    name = "active-cancel"
    cancelled_before = wait_counters(address, lambda _: True)["cancelled"]
    body = payload(messages(), 192).encode()
    peer = socket.create_connection((address.hostname, address.port))
    reader = peer.makefile("rb")
    try:
        peer.sendall(
            (
                "POST /v1/chat/completions HTTP/1.1\r\nHost: local\r\n"
                "Content-Type: application/json\r\n"
                f"X-Loom-Session: {name}\r\nContent-Length: {len(body)}\r\n\r\n"
            ).encode()
            + body
        )
        if not reader.readline().startswith(b"HTTP/1.1 200 "):
            raise RuntimeError("active cancellation request was not admitted")
        for line in reader:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                break
        else:
            raise RuntimeError("active cancellation received no model output")
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    finally:
        reader.close()
        peer.close()
    wait_counters(address, lambda counts: counts["cancelled"] == cancelled_before + 1)
    wait_status(address, active_requests=0, queued_requests=0)
    result = chat.request(
        address,
        name,
        [{"role": "user", "content": "What is 2+2? Reply with only one number."}],
        8,
        model="qwen3.8-27b",
    )
    if result["text"].strip() != "4":
        raise RuntimeError("cancelled row reuse changed output")
    if result["usage"]["prompt_tokens_details"]["cached_tokens"]:
        raise RuntimeError("cancelled checkpoint was retained")
    return result


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
        with running_server(
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
            results.append(cancel_active(address))
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
        prepared = [e for e in events.values if e.get("event") == "prepared"]
        identities = {e["request"] for e in prepared}
        if len(identities) != len(prepared):
            raise RuntimeError("capacity retries repeated source input preparation")
        for admitted in (e for e in events.values if e.get("event") == "admit"):
            plan = next(
                (e for e in prepared if e["request"] == admitted["request"]), None
            )
            if plan is None or any(
                plan[field] != admitted[field]
                for field in (
                    "row",
                    "retained_tokens",
                    "appended_tokens",
                    "reservation_tokens",
                )
            ):
                raise RuntimeError("admission changed its prepared input identity")
        heartbeats = [e for e in events.values if e["event"] == "text_memory"]
        shared = {
            e["time_ns"]: e
            for e in events.values
            if e["event"] == "shared_device_memory"
        }
        if not heartbeats:
            raise RuntimeError("workload did not observe memory heartbeats")
        for event in heartbeats:
            memory = shared[event["time_ns"]]
            retained_bytes = (
                event["elastic_state"]["committed_bytes"]
                + event["elastic_parameters"]["committed_bytes"]
            )
            if memory["retained_committed_bytes"] != retained_bytes:
                raise RuntimeError("single-model device backing disagrees with owners")
            if memory["workspace_live_bytes"] > memory["workspace_committed_bytes"]:
                raise RuntimeError("live workspace exceeds reported shared backing")
        if not any(
            e["workspace_committed_bytes"] and e["workspace_reuse_count"]
            for e in shared.values()
        ):
            raise RuntimeError("workload did not observe shared workspace reuse")
        if capacity:
            if arguments.continuation_epochs == 2 and not any(
                event["verification_epochs"] == 2
                for event in events.values
                if event["event"] == "span"
            ):
                raise RuntimeError("workload did not exercise device continuation")
            admissions = [e for e in events.values if e.get("event") == "admit"]
            if any(
                e["pool_reserved_tokens"] + e["pool_resident_tokens"] > capacity
                for e in admissions
            ):
                raise RuntimeError("completion reservations exceeded physical capacity")
            if any(
                e["event"] == "session_begin" and e["name"] == "cancelled"
                for e in events.values
            ):
                raise RuntimeError("cancelled queued request acquired a model row")
            if not any(
                e["event"] == "session_end"
                and e["counters"]["completed"]
                and not e["counters"]["cancelled"]
                and e["resident_tokens"]
                for e in events.values
            ):
                raise RuntimeError("workload did not exercise idle cache eviction")
            trims = [e for e in events.values if e.get("event") == "state_trim"]
            if not any(e["released_bytes"] for e in trims):
                raise RuntimeError("idle eviction did not release physical backing")
            if any(
                e["pool"]["reserved_tokens"] + e["pool"]["resident_tokens"] > capacity
                for e in heartbeats
            ):
                raise RuntimeError("pool accounting exceeded the configured budget")
            if not any(e["queued"] for e in events.values if e["event"] == "snapshot"):
                raise RuntimeError("workload did not observe queued admission")
        print(
            json.dumps({"event": "pass", "pool_capacity": capacity, "requests": 10}),
            flush=True,
        )


if __name__ == "__main__":
    main()
