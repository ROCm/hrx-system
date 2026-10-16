# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measured chat-completion client shared by model checks and benchmarks."""

import http.client
import json
import time


def request(
    address,
    session,
    messages,
    maximum,
    *,
    model,
    tools=None,
    checkpoint=None,
    enable_thinking=False,
    on_content=None,
):
    """Consume one greedy chat stream and measure its complete HTTP lifetime.

    The caller selects the deployment and owns any outer deadline. Session and
    checkpoint headers carry Loom retention identity; other compatible servers
    can ignore them. The returned text, tool calls, usage, terminal reason and
    performance-clock timestamps are copied before the connection closes.
    Incomplete streams and endpoint errors raise instead of successful samples.
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "enable_thinking": enable_thinking,
        "max_tokens": maximum,
    }
    if tools is not None:
        body["tools"] = tools
    headers = {"X-Loom-Session": session}
    if checkpoint is not None:
        headers["X-Loom-Checkpoint"] = checkpoint
    return request_body(address, body, headers, on_content=on_content)


def request_body(address, body, headers=None, *, on_content=None):
    """Measure a complete single-choice stream without rewriting its request.

    Unlike the greedy convenience caller, this path preserves all model policy
    and media fields in a captured request. The address is an HTTP(S) origin;
    the caller owns its validation and any outer deadline.

    on_content observes each nonempty answer delta and its absolute performance
    clock timestamp in nanoseconds. Reasoning and tool fragments remain separate.
    Calls are synchronous, so observer work contributes to measured request time.
    Observer exceptions propagate after closing the connection.
    """
    connection_type = (
        http.client.HTTPSConnection
        if address.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(address.hostname, address.port)
    model = body["model"]
    start = time.perf_counter_ns()
    first = None
    chunks = []
    text = []
    reasoning = []
    tool_calls = {}
    usage = None
    reason = None
    done = False
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            {"Content-Type": "application/json", **(headers or {})},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()!r}")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("model") != model:
                raise RuntimeError(f"response was routed to the wrong model: {event}")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("index") != 0:
                    raise RuntimeError("Expected a single choice at index zero")
                if choice.get("finish_reason"):
                    reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                thought = delta.get("reasoning_content", "")
                if content or thought or delta.get("tool_calls"):
                    received = time.perf_counter_ns()
                    chunks.append((received - start) / 1e6)
                if content:
                    if first is None:
                        first = received
                    text.append(content)
                    if on_content is not None:
                        on_content(content, received)
                if thought:
                    reasoning.append(thought)
                for fragment in delta.get("tool_calls", []):
                    call = tool_calls.setdefault(
                        fragment["index"],
                        {
                            "id": "",
                            "type": "",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    for field in ("id", "type"):
                        if field in fragment:
                            call[field] += fragment[field]
                    for field in ("name", "arguments"):
                        call["function"][field] += fragment.get("function", {}).get(
                            field, ""
                        )
        if not done or usage is None or reason not in ("stop", "length", "tool_calls"):
            raise RuntimeError(
                f"Incomplete successful SSE: {done=}, {usage=}, {reason=}"
            )
        if bool(tool_calls) != (reason == "tool_calls"):
            raise RuntimeError("Tool deltas and terminal finish reason disagree")
        if sorted(tool_calls) != list(range(len(tool_calls))):
            raise RuntimeError("Tool delta indexes are not contiguous")
        end = time.perf_counter_ns()
        return {
            "text": "".join(text),
            "reasoning_content": "".join(reasoning),
            "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
            "usage": usage,
            "finish_reason": reason,
            "request_ms": (end - start) / 1e6,
            "first_text_ms": None if first is None else (first - start) / 1e6,
            "output_chunk_ms": chunks,
            "started_ns": start,
            "completed_ns": end,
        }
    finally:
        connection.close()
