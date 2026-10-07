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
            parked = control(address, "POST", "/v1/checkpoints/base/suspend", 200)
            if not parked["host_snapshot_bytes"]:
                raise RuntimeError("queued endpoint did not retain a cold image")
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


def cold_endpoint(address, events, messages, base, receipt):
    for method, target, expected, body in (
        ("POST", "absent/suspend", 404, None),
        ("DELETE", "base/suspend", 400, None),
        ("POST", "base/suspend", 400, "not empty"),
        ("POST", "base/suspend/suspend", 400, None),
    ):
        control(address, method, f"/v1/checkpoints/{target}", expected, body=body)
    parked = control(address, "POST", "/v1/checkpoints/base/suspend", 200)
    if (
        not parked["pinned"]
        or not parked["host_snapshot_bytes"]
        or parked["position"] != receipt["position"]
        or receipt["position"] <= 512
        or receipt["position"] % 64 == 0
    ):
        raise RuntimeError(f"cold fixture needs a large partial prefix: {parked}")
    if control(address, "POST", "/v1/checkpoints/base/suspend", 200) != parked:
        raise RuntimeError("repeated suspension changed the endpoint")

    # A different complete history reuses the old prefix storage. The attempted
    # replacement needs all 1024 positions plus the immutable partial tail;
    # even recycling this selected row cannot fund that completion guarantee.
    competing = [dict(message) for message in messages]
    competing[0]["content"] = competing[0]["content"].replace("MAPLE", "CEDAR")
    selected = request(address, "cold-selected", competing, 12)
    frontier = events.wait("complete", "cold-selected")["position"]
    history = base + [
        {"role": "user", "content": "Reply with only the project codeword."}
    ]
    control(
        address,
        "POST",
        "/v1/chat/completions",
        503,
        session="cold-selected",
        body=json.dumps(
            {
                "model": "qwen3.8-27b",
                "messages": history,
                "stream": True,
                "max_tokens": 480,
                "temperature": 0,
            }
        ),
        headers={"Content-Type": "application/json", "X-Loom-Checkpoint": "base"},
    )
    if control(address, "POST", "/v1/checkpoints/base/suspend", 200) != parked:
        raise RuntimeError("refused wake changed the cold image")
    selected_history = competing + [
        {"role": "assistant", "content": selected["text"]},
        {"role": "user", "content": "Reply with only the project codeword."},
    ]
    kept = request(address, "cold-selected", selected_history, 16)
    if kept["usage"]["prompt_tokens_details"]["cached_tokens"] != frontier:
        raise RuntimeError("refused cold wake replaced the selected continuation")

    # With a fresh destination, the competing idle row can yield its pages.
    restored = request(address, "cold-restored", history, 16, checkpoint="base")
    admitted = events.wait("admit", "cold-restored")
    if (
        restored["usage"]["prompt_tokens_details"]["cached_tokens"]
        != receipt["position"]
        or admitted["restored_bytes"] != parked["host_snapshot_bytes"]
    ):
        raise RuntimeError(f"cold endpoint was not restored exactly: {admitted}")
    if control(address, "POST", "/v1/checkpoints/base/suspend", 200) != parked:
        raise RuntimeError("resident/cold transition changed the immutable endpoint")
    print(
        json.dumps({"event": "cold_resume", "parked": parked, "admit": admitted}),
        flush=True,
    )
    return [(restored, history), (kept, selected_history)]


def fixed_suspension(command, log):
    fixed = [
        "--pool_backing=fixed" if flag == "--pool_backing=elastic" else flag
        for flag in command
        if not flag.startswith("--memory_bytes=")
    ]
    with running_server(fixed, log) as (_, endpoint):
        if endpoint is None:
            raise RuntimeError(f"fixed server did not become ready: {log}")
        address = urlsplit(f"http://{endpoint}")
        history = [{"role": "user", "content": "Reply with only READY."}]
        first = request(address, "fixed", history, 8)
        receipt = control(
            address, "POST", "/v1/checkpoints/fixed", 201, session="fixed"
        )
        rejected = control(address, "POST", "/v1/checkpoints/fixed/suspend", 400)
        if "requires elastic backing" not in rejected["error"]["message"]:
            raise RuntimeError(f"fixed suspension rejected incorrectly: {rejected}")
        history += [
            {"role": "assistant", "content": first["text"]},
            {"role": "user", "content": "Reply with only OK."},
        ]
        restored = request(address, "fixed-fork", history, 8, checkpoint="fixed")
        if (
            restored["usage"]["prompt_tokens_details"]["cached_tokens"]
            != receipt["position"]
        ):
            raise RuntimeError("fixed rejection changed the endpoint")
        control(address, "DELETE", "/v1/checkpoints/fixed", 200)
        same_output(restored, request(address, "", history, 8))
    print(json.dumps({"event": "fixed_suspend_rejected"}), flush=True)


def physical_completion(command, messages, output, parameter_bytes):
    """Admission covers weights, immutable anchors and future private writers."""
    common = [
        flag
        for flag in command
        if not flag.startswith(("--rows=", "--epoch=", "--memory_bytes="))
    ] + ["--rows=2", "--epoch=32:2", "--epoch=128:2"]
    for headroom_mib in (400, 256):
        budget = parameter_bytes + headroom_mib * 1024 * 1024
        events = Events()
        log = output / f"physical-{headroom_mib}.log"
        with running_server(
            common + [f"--memory_bytes={budget}"], log, events.record
        ) as (_, endpoint):
            if endpoint is None:
                raise RuntimeError(f"physical-budget server failed startup: {log}")
            address = urlsplit(f"http://{endpoint}")
            first = request(address, "root", messages, 12)
            history = messages + [
                {"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "Reply with only the project codeword."},
            ]
            receipt = control(
                address, "POST", "/v1/checkpoints/base", 201, session="root"
            )
            body = json.dumps(
                {
                    "model": "qwen3.8-27b",
                    "messages": history,
                    "stream": True,
                    "max_tokens": 16,
                    "temperature": 0,
                }
            )
            if headroom_mib == 256:
                # Logical KV fits, but the shared recurrent reader plus its
                # first private writer cannot coexist with these weights.
                control(
                    address,
                    "POST",
                    "/v1/chat/completions",
                    503,
                    session="root",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
            parked = control(address, "POST", "/v1/checkpoints/base/suspend", 200)
            if not parked["host_snapshot_bytes"]:
                raise RuntimeError("physical pressure fixture did not suspend")
            if headroom_mib == 400:
                # Two tiny idle histories hide recurrent-state pressure from
                # token-only admission. One must yield before opening SSE.
                for name, token in (("small-a", "A"), ("small-b", "B")):
                    request(
                        address,
                        name,
                        [{"role": "user", "content": f"Reply with only {token}."}],
                        8,
                    )
                result = request(address, "restored", history, 16, checkpoint="base")
                admitted = events.wait("admit", "restored")
                if admitted["restored_bytes"] != parked["host_snapshot_bytes"]:
                    raise RuntimeError(f"physical wake replayed: {admitted}")
            else:
                control(
                    address,
                    "POST",
                    "/v1/chat/completions",
                    503,
                    session="root",
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Loom-Checkpoint": "base",
                    },
                )
                if (
                    control(address, "POST", "/v1/checkpoints/base/suspend", 200)
                    != parked
                ):
                    raise RuntimeError("physical denial discarded the cold endpoint")
            control(address, "DELETE", "/v1/checkpoints/base", 200)
            if headroom_mib == 256:
                # Last shared reader is gone; in-place continuation now fits.
                result = request(address, "root", history, 16)
            if (
                result["usage"]["prompt_tokens_details"]["cached_tokens"]
                != receipt["position"]
            ):
                raise RuntimeError("physical pressure discarded the selected frontier")
            same_output(result, request(address, "", history, 16))
        heartbeats = [e for e in events.values if e["event"] == "heartbeat"]
        if not heartbeats or any(
            e["elastic_parameters"]["released_bytes"] for e in heartbeats
        ):
            raise RuntimeError("mutable admission evicted its own required weights")
        peak = max(e["elastic_state"]["peak_bytes"] for e in heartbeats)
        if peak > headroom_mib * 1024 * 1024:
            raise RuntimeError(f"physical completion exceeded its budget: {peak}")
        print(
            json.dumps(
                {
                    "event": "physical_completion",
                    "headroom_mib": headroom_mib,
                    "peak_state_bytes": peak,
                    "result": result,
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
        if "retained state leaves insufficient" not in impossible["error"]["message"]:
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
        cold_results = cold_endpoint(address, events, messages, base, receipt)
        released = control(address, "DELETE", "/v1/checkpoints/base", 200)
        if released["pinned"] or released["host_snapshot_bytes"]:
            raise RuntimeError("cold release retained its image")
        # The tight pool cannot hold the pin plus an independent full prefix.
        # Drop the explicit pin before running fresh replays of every output.
        same_output(rewound, request(address, "", revised, 16))
        for index, result in enumerate(branches):
            history = base + [{"role": "user", "content": suffixes[index]}]
            same_output(result, request(address, "", history, 32))
        for result, history in cold_results:
            same_output(result, request(address, "", history, 16))
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

    fixed_suspension(command, arguments.output / "fixed.log")
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
    heartbeats = [event for event in events if event["event"] == "heartbeat"]
    if not any(event["checkpoint_host_bytes"] for event in heartbeats):
        raise RuntimeError("cold payload was not visible in heartbeat accounting")
    if heartbeats[-1]["checkpoint_host_bytes"]:
        raise RuntimeError("endpoint release leaked host accounting")
    parameter_bytes = max(
        event["elastic_parameters"]["committed_bytes"] for event in heartbeats
    )
    physical_completion(command, messages, arguments.output, parameter_bytes)
    print(
        json.dumps({"event": "pass", "branches": branches, "admissions": admits}),
        flush=True,
    )


if __name__ == "__main__":
    main()
