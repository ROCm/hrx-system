# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real HTTP pin/fork/summary-rewind witness with a tight shared KV pool."""

import argparse
import http.client
import json
import socket
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen.benchmark_service import request
from experimental.loom_serve.models.qwen.check_service import Events, running_server


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


def same_output(actual, reference):
    for field in ("text", "tool_calls", "finish_reason"):
        if actual[field] != reference[field]:
            raise RuntimeError(
                f"checkpoint replay differs in {field}: {actual} {reference}"
            )


def tool_rewind(address):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Read benchmark evidence for a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "The project codeword is MAPLE. Call lookup exactly once when asked. "
            "After receiving its result, follow instructions without further tools.",
        },
        {"role": "user", "content": "Call lookup with topic kernels."},
    ]
    first = request(address, "tool-root", messages, 128, tools=tools)
    calls = first["tool_calls"]
    if (
        len(calls) != 1
        or calls[0]["function"]["name"] != "lookup"
        or json.loads(calls[0]["function"]["arguments"]) != {"topic": "kernels"}
    ):
        raise RuntimeError(f"incorrect generated tool call: {first}")
    base = messages + [
        {"role": "assistant", "content": first["text"], "tool_calls": calls}
    ]
    receipt = control(address, "POST", "/v1/checkpoints/tool", 201, session="tool-root")
    temporary = base + [
        {
            "role": "tool",
            "tool_call_id": calls[0]["id"],
            "content": "The benchmark passed. Staging reduced weight reads.\n" * 12,
        },
        {"role": "user", "content": "Summarize this evidence in one short sentence."},
    ]
    summary = request(address, "tool-root", temporary, 48, tools=tools)
    if not summary["text"].strip() or summary["tool_calls"]:
        raise RuntimeError(f"tool evidence did not produce a plain summary: {summary}")
    revised = base + [
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": summary["text"]},
        {"role": "user", "content": "Reply with only the project codeword."},
    ]
    rewound = request(address, "tool-root", revised, 16, tools=tools, checkpoint="tool")
    if (
        rewound["usage"]["prompt_tokens_details"]["cached_tokens"]
        != receipt["position"]
    ):
        raise RuntimeError("tool rewind did not restore the pre-result endpoint")
    control(address, "DELETE", "/v1/checkpoints/tool", 200)
    same_output(rewound, request(address, "", revised, 16, tools=tools))
    print(
        json.dumps(
            {
                "event": "tool_rewind",
                "call": first,
                "summary": summary,
                "result": rewound,
                "position": receipt["position"],
            }
        ),
        flush=True,
    )


def queued_pin(address, events, base):
    history = base + [
        {
            "role": "user",
            "content": "Count from 1 through 1000, one number per line without commentary.",
        }
    ]
    with ThreadPoolExecutor(max_workers=1) as clients:
        active = clients.submit(
            request, address, "pin-anchor", history, 256, checkpoint="base"
        )
        events.wait("admit", "pin-anchor")
        queued = http.client.HTTPConnection(address.hostname, address.port)
        try:
            body = {
                "model": "qwen3.8-27b",
                "messages": history,
                "stream": True,
                "max_tokens": 128,
                "temperature": 0,
            }
            queued.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(body),
                {
                    "Content-Type": "application/json",
                    "X-Loom-Session": "pin-queued",
                    "X-Loom-Checkpoint": "base",
                },
            )
            events.wait("enqueue", "pin-queued")
            control(address, "DELETE", "/v1/checkpoints/base", 409)
            queued.sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        finally:
            queued.close()
        events.wait("cancel_queued", "pin-queued")
        result = active.result()
    print(
        json.dumps(
            {
                "event": "queued_pin_retired",
                "output_tokens": result["usage"]["completion_tokens"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("server", "model", "weights", "tokenizer", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(arguments.server.resolve()),
        f"--model={arguments.model.resolve()}",
        f"--weights={arguments.weights.resolve()}",
        f"--tokenizer={arguments.tokenizer.resolve()}",
        "--rows=4",
        "--port=0",
        "--heartbeat_ms=1000",
        "--context_capacity=2048",
        "--prefill_capacity=128",
        "--epoch=32:4",
        "--epoch=128:4",
        "--pool_capacity=1024",
        "--pool_backing=elastic",
        "--memory_bytes=23622320128",
        "--checkpoint_capacity=1",
        "--mtp",
        "--mtp_depth=3",
        "--continuation_epochs=2",
    ]
    log = arguments.output / "server.log"
    events = Events()
    with running_server(command, log, events.record) as (_, endpoint):
        if endpoint is None:
            raise RuntimeError(f"server did not become ready: {log}")
        address = urlsplit(f"http://{endpoint}")
        messages = [
            {
                "role": "system",
                "content": "The project codeword is MAPLE. Follow instructions exactly.",
            },
            {
                "role": "user",
                "content": "Background entry: kernels use shared tiles and streamed weights.\n"
                * 40
                + "Reply with only READY.",
            },
        ]
        first = request(address, "root", messages, 12)
        base = messages + [{"role": "assistant", "content": first["text"]}]
        receipt = control(address, "POST", "/v1/checkpoints/base", 201, session="root")
        print(json.dumps({"event": "pinned", **receipt}), flush=True)
        control(address, "POST", "/v1/checkpoints/base", 409, session="root")
        control(address, "POST", "/v1/checkpoints/extra", 503, session="root")
        control(address, "DELETE", "/v1/checkpoints/absent", 404)
        control(address, "POST", "/v1/checkpoints/bad/name", 400, session="root")
        replay = json.dumps(
            {
                "model": "qwen3.8-27b",
                "messages": messages,
                "stream": True,
                "max_tokens": 32,
                "temperature": 0,
            }
        )
        # Replaying the original prompt cannot share its completed endpoint.
        # A refused replacement preserves the selected retained continuation.
        impossible = control(
            address,
            "POST",
            "/v1/chat/completions",
            503,
            session="root",
            body=replay,
            headers={"Content-Type": "application/json"},
        )
        if (
            "pinned checkpoints leave insufficient"
            not in impossible["error"]["message"]
        ):
            raise RuntimeError(f"independent replay rejected incorrectly: {impossible}")
        bad = {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": "Different history"}],
            "stream": True,
            "max_tokens": 8,
            "temperature": 0,
        }
        mismatch = control(
            address,
            "POST",
            "/v1/chat/completions",
            400,
            session="root",
            body=json.dumps(bad),
            headers={"Content-Type": "application/json", "X-Loom-Checkpoint": "base"},
        )
        if "history does not extend" not in mismatch["error"]["message"]:
            raise RuntimeError(f"history rejected for the wrong reason: {mismatch}")

        # Actual model-generated summary becomes the only retained tool evidence.
        temporary = base + [
            {
                "role": "user",
                "content": "Tool output:\n"
                + "The tile benchmark passed; staging reduces weight reads.\n" * 12
                + "Summarize this evidence in one short sentence.",
            }
        ]
        summary = request(address, "root", temporary, 32)
        if (
            summary["usage"]["prompt_tokens_details"]["cached_tokens"]
            != receipt["position"]
        ):
            raise RuntimeError("capacity refusal replaced the selected continuation")
        # Unrelated admission pressure may evict best-effort session retention;
        # the explicit pin still has to survive and restore the endpoint below.
        control(
            address,
            "POST",
            "/v1/chat/completions",
            503,
            session="independent",
            body=replay,
            headers={"Content-Type": "application/json"},
        )
        revised = base + [
            {
                "role": "user",
                "content": f"Retained summary: {summary['text']}\nReply with only the project codeword.",
            }
        ]
        rewound = request(address, "root", revised, 16, checkpoint="base")
        if (
            rewound["usage"]["prompt_tokens_details"]["cached_tokens"]
            != receipt["position"]
        ):
            raise RuntimeError("rewind did not restore the exact pinned frontier")
        print(
            json.dumps({"event": "rewind", "summary": summary, "result": rewound}),
            flush=True,
        )

        barrier = threading.Barrier(3)
        suffixes = [
            "Reply with only the project codeword.",
            "Reply with the project codeword followed by the word done.",
        ]

        def branch(index):
            history = base + [{"role": "user", "content": suffixes[index]}]
            barrier.wait()
            return request(address, f"fork-{index}", history, 32, checkpoint="base")

        with ThreadPoolExecutor(max_workers=2) as clients:
            futures = [clients.submit(branch, index) for index in range(2)]
            barrier.wait()
            branches = [future.result() for future in futures]
        for result in branches:
            if (
                result["usage"]["prompt_tokens_details"]["cached_tokens"]
                != receipt["position"]
            ):
                raise RuntimeError("fork replayed instead of sharing its endpoint")
        queued_pin(address, events, base)
        control(address, "DELETE", "/v1/checkpoints/base", 200)
        # The tight pool cannot hold the pin plus an independent full prefix.
        # Drop the explicit pin before running fresh replays of every output.
        same_output(rewound, request(address, "", revised, 16))
        for index, result in enumerate(branches):
            history = base + [{"role": "user", "content": suffixes[index]}]
            same_output(result, request(address, "", history, 32))
        control(address, "DELETE", "/v1/checkpoints/base", 404)
        control(
            address,
            "POST",
            "/v1/chat/completions",
            404,
            session="root",
            body=json.dumps(bad),
            headers={"Content-Type": "application/json", "X-Loom-Checkpoint": "base"},
        )

        tool_rewind(address)

    events = events.values
    admits = [
        event
        for event in events
        if event["event"] == "admit" and event["session"].startswith("fork-")
    ]
    if len(admits) != 2 or sum(event["reservation_tokens"] for event in admits) <= 1024:
        raise RuntimeError(
            "fixture did not exceed duplicated private completion reservations"
        )
    serials = {event["request"] for event in admits}
    last_admit = max(
        i
        for i, event in enumerate(events)
        if event["event"] == "admit" and event["request"] in serials
    )
    first_complete = min(
        i
        for i, event in enumerate(events)
        if event["event"] == "complete" and event["request"] in serials
    )
    if last_admit >= first_complete:
        raise RuntimeError("shared branches never held completion credit concurrently")
    for event in events:
        if event["event"] == "heartbeat":
            pool = event["pool"]
            if pool["reserved_tokens"] + pool["resident_tokens"] > 1024:
                raise RuntimeError(f"shared admission overcommitted: {event}")
        if event["event"] == "admit" and (
            event["pool_reserved_tokens"] + event["pool_resident_tokens"] > 1024
        ):
            raise RuntimeError(f"shared admission overcommitted: {event}")
    print(
        json.dumps({"event": "pass", "branches": branches, "admissions": admits}),
        flush=True,
    )


if __name__ == "__main__":
    main()
