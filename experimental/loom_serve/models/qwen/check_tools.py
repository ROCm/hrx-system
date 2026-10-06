# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real generated tool calls followed by retained client-serialized history.

The model emits typed arguments; the client supplies a deterministic tool result
without executing model-chosen code. The next request must reuse that session's
retained state and answer from the result. An invalid intervening history must
be rejected without discarding the completed checkpoint. Logs preserve every
request/response and server heartbeat. The caller owns GPU exclusion.
"""

import argparse
import copy
import http.client
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen import benchmark_service, check_service

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read text from a file at the given offset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "literal": {"type": "boolean"},
                },
                "required": ["path", "offset", "literal"],
            },
        },
    }
]


def rejected_history(address, session, messages):
    malformed = copy.deepcopy(messages)
    malformed[-2]["tool_calls"][0]["function"]["arguments"] = "{} trailing"
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(
                {
                    "model": "qwen3.8-27b",
                    "messages": malformed,
                    "tools": TOOLS,
                    "stream": True,
                    "temperature": 0,
                    "max_tokens": 16,
                }
            ),
            {"Content-Type": "application/json", "X-Loom-Session": session},
        )
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != 400 or "trailing" not in body:
            raise RuntimeError(
                f"Malformed history was not rejected: {response.status} {body}"
            )
        return {"status": response.status, "body": body}
    finally:
        connection.close()


def client(address, index, output):
    session = f"source-tool-{index}"
    expected = {"path": f"ledger-{index}.txt", "offset": index + 7, "literal": False}
    codeword = f"COBALT{index}"
    messages = [
        {
            "role": "system",
            "content": (
                "Use the read tool exactly once with the requested arguments. "
                "After receiving its result, reply with only the codeword from "
                "that result, without explanation or further tools."
            ),
        },
        {
            "role": "user",
            "content": f"Call read with these exact arguments: {json.dumps(expected)}",
        },
    ]
    path = output / f"{session}.jsonl"
    with path.open("x") as log:

        def record(event):
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
            log.flush()

        record({"event": "request", "messages": messages, "tools": TOOLS})
        first = benchmark_service.request(address, session, messages, 192, tools=TOOLS)
        record({"event": "response", **first})
        calls = first["tool_calls"]
        if first["finish_reason"] != "tool_calls" or len(calls) != 1:
            raise RuntimeError(f"{session}: expected one tool call: {first}")
        call = calls[0]
        if (
            call["type"] != "function"
            or not call["id"]
            or call["function"]["name"] != "read"
            or json.loads(call["function"]["arguments"]) != expected
        ):
            raise RuntimeError(f"{session}: incorrect generated arguments: {call}")
        messages.extend(
            [
                {"role": "assistant", "content": first["text"], "tool_calls": calls},
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": f"The codeword is {codeword}.",
                },
            ]
        )
        rejected = rejected_history(address, session, messages)
        record({"event": "rejected_history", **rejected})
        record({"event": "request", "messages": messages, "tools": TOOLS})
        second = benchmark_service.request(address, session, messages, 32, tools=TOOLS)
        record({"event": "response", **second})
        if second["finish_reason"] != "stop" or second["text"].strip() != codeword:
            raise RuntimeError(f"{session}: incorrect tool-result answer: {second}")
        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        if not cached:
            raise RuntimeError(f"{session}: canonical history missed retained state")
        return {
            "session": session,
            "typed_arguments": expected,
            "codeword": codeword,
            "cached_tokens": cached,
            "output_tokens": sum(
                turn["usage"]["completion_tokens"] for turn in (first, second)
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=3)
    arguments = parser.parse_args()
    if not 1 <= arguments.clients <= 16:
        parser.error("clients must be in [1, 16]")
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(arguments.server.resolve()),
        f"--model={arguments.model.resolve()}",
        f"--weights={arguments.weights.resolve()}",
        f"--tokenizer={arguments.tokenizer.resolve()}",
        "--prefill_capacity=128",
        "--context_capacity=2048",
        f"--rows={arguments.clients}",
        f"--pool_capacity={arguments.clients * 2048}",
        "--mtp",
        "--mtp_depth=3",
        "--continuation_epochs=2",
        "--port=0",
        "--heartbeat_ms=1000",
    ]

    def progress(event):
        if event and event.get("event") in ("ready", "heartbeat"):
            print(json.dumps(event), flush=True)

    with check_service.running_server(
        command, arguments.output / "server.jsonl", on_event=progress
    ) as (_, address):
        if address is None:
            raise RuntimeError("tool-call server did not reach readiness")
        address = urlsplit(f"http://{address}")
        with ThreadPoolExecutor(max_workers=arguments.clients) as pool:
            futures = [
                pool.submit(client, address, index, arguments.output)
                for index in range(arguments.clients)
            ]
            for future in as_completed(futures):
                print(
                    json.dumps({"event": "client_pass", **future.result()}), flush=True
                )
    print(json.dumps({"event": "pass", "clients": arguments.clients}), flush=True)


if __name__ == "__main__":
    main()
