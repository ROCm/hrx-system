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
):
    """Consume one greedy chat stream and measure its complete HTTP lifetime.

    The caller selects the deployment and owns any outer deadline. Session and
    checkpoint headers carry Loom retention identity; other compatible servers
    can ignore them. The returned text, tool calls, usage, terminal reason and
    monotonic timestamps are copied before the connection closes. Incomplete
    streams and endpoint errors raise instead of becoming successful samples.
    """
    connection = http.client.HTTPConnection(address.hostname, address.port)
    start = time.monotonic_ns()
    first = None
    chunks = []
    text = []
    tool_calls = {}
    usage = None
    reason = None
    done = False
    try:
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": maximum,
        }
        if tools is not None:
            body["tools"] = tools
        headers = {"Content-Type": "application/json", "X-Loom-Session": session}
        if checkpoint is not None:
            headers["X-Loom-Checkpoint"] = checkpoint
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            headers,
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
                if choice.get("finish_reason"):
                    reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content or delta.get("tool_calls"):
                    chunks.append((time.monotonic_ns() - start) / 1e6)
                if content:
                    if first is None:
                        first = time.monotonic_ns()
                    text.append(content)
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
        end = time.monotonic_ns()
        return {
            "text": "".join(text),
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
