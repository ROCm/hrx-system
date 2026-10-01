# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bounded concurrent HTTP workload with retained, independently checked turns.

Run against an already ready qwen_server. Timing covers the complete client
cohort, including SSE and continuation requests; model loading is excluded.
Server epoch events provide the actual prompt/decode mix and traversal costs.
"""

import argparse
import hashlib
import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

WORDS = ("MAPLE", "COBALT", "CEDAR", "QUARTZ", "AMBER", "CORAL", "JADE", "ONYX")


def request(address, session, messages, maximum):
    connection = http.client.HTTPConnection(address.hostname, address.port)
    start = time.monotonic_ns()
    first = None
    text = []
    usage = None
    reason = None
    done = False
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(
                {
                    "model": "qwen3.8-27b",
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "temperature": 0,
                    "max_tokens": maximum,
                }
            ),
            {"Content-Type": "application/json", "X-Loom-Session": session},
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
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    reason = choice["finish_reason"]
                content = choice.get("delta", {}).get("content", "")
                if content:
                    if first is None:
                        first = time.monotonic_ns()
                    text.append(content)
        if not done or usage is None or reason not in ("stop", "length"):
            raise RuntimeError(
                f"Incomplete successful SSE: {done=}, {usage=}, {reason=}"
            )
        end = time.monotonic_ns()
        return {
            "text": "".join(text),
            "usage": usage,
            "finish_reason": reason,
            "request_ms": (end - start) / 1e6,
            "first_text_ms": None if first is None else (first - start) / 1e6,
        }
    finally:
        connection.close()


def client(arguments, address, index, barrier):
    word = WORDS[index]
    # Alternating short/long inputs create real overlap between decoding peers
    # and peers still consuming prompt chunks, without pre-recorded arrivals.
    background = (
        "Background: the repository contains kernels, programs, tests, and documentation.\n"
        * (arguments.long_lines if index % 2 else 0)
    )
    messages = [
        {
            "role": "system",
            "content": "Follow the user's instructions exactly. Do not use tools.",
        },
        {
            "role": "user",
            "content": (
                f"Remember my secret codeword {word} for later.\n{background}\n"
                "Now count from 1 through 30, one integer per line, without commentary."
            ),
        },
    ]
    barrier.wait()
    session = f"{arguments.session_prefix}-{index}"
    first = request(address, session, messages, arguments.max_tokens)
    if not first["text"].strip():
        raise RuntimeError(f"{session}: empty first response")
    messages.extend(
        [
            {"role": "assistant", "content": first["text"]},
            {
                "role": "user",
                "content": "What is my secret codeword? Reply with only that word.",
            },
        ]
    )
    second = request(address, session, messages, 16)
    if second["text"].strip() != word:
        raise RuntimeError(f"{session}: expected {word!r}, got {second['text']!r}")
    if not second["usage"]["prompt_tokens_details"]["cached_tokens"]:
        raise RuntimeError(f"{session}: continuation did not reuse retained state")
    return {"session": session, "verified_codeword": word, "turns": [first, second]}


def load_workload(path):
    """Loads frozen text turns; responses become retained history at run time."""
    payload = path.read_bytes()
    workload = json.loads(payload)
    if not isinstance(workload, dict) or not isinstance(workload.get("name"), str):
        raise ValueError("workload needs a name and sessions")
    sessions = workload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("workload sessions must be a nonempty list")
    for index, session in enumerate(sessions):
        if not isinstance(session, dict) or not isinstance(session.get("system"), str):
            raise ValueError(f"session {index} needs a system string")
        turns = session.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"session {index} needs nonempty turns")
        for turn in turns:
            if (
                not isinstance(turn, dict)
                or not isinstance(turn.get("content"), str)
                or not turn["content"]
                or type(turn.get("max_tokens")) is not int
                or not 1 <= turn["max_tokens"] <= 16384
            ):
                raise ValueError(
                    f"session {index}: each turn needs content and max_tokens in [1, 16384]"
                )
    return workload, hashlib.sha256(payload).hexdigest()


def workload_client(arguments, address, index, barrier, workload):
    fixture = workload["sessions"][index]
    messages = [{"role": "system", "content": fixture["system"]}]
    session = f"{arguments.session_prefix}-{index}"
    results = []
    barrier.wait()
    for turn in fixture["turns"]:
        messages.append({"role": "user", "content": turn["content"]})
        input_digest = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        result = request(address, session, messages, turn["max_tokens"])
        if not result["text"].strip():
            raise RuntimeError(f"{session}: empty response")
        if results and not result["usage"]["prompt_tokens_details"]["cached_tokens"]:
            raise RuntimeError(f"{session}: continuation did not reuse retained state")
        results.append(
            {
                **result,
                "input_sha256": input_digest,
                "max_tokens": turn["max_tokens"],
            }
        )
        messages.append({"role": "assistant", "content": result["text"]})
    return {"session": session, "session_index": index, "turns": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--long-lines", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--session-prefix", default="service-benchmark")
    parser.add_argument(
        "--workload",
        type=Path,
        help="Frozen JSON text-turn corpus; replaces the codeword workload.",
    )
    arguments = parser.parse_args()
    address = urlsplit(arguments.url)
    if (
        address.scheme != "http"
        or not address.hostname
        or address.path not in ("", "/")
    ):
        parser.error("url must be an HTTP origin")
    if not 1 <= arguments.clients <= len(WORDS):
        parser.error("clients must be in [1, 8]")
    if arguments.long_lines < 0 or arguments.max_tokens < 1:
        parser.error("long-lines must be nonnegative and max-tokens positive")
    workload = None
    workload_digest = None
    if arguments.workload:
        workload, workload_digest = load_workload(arguments.workload)
        if arguments.clients > len(workload["sessions"]):
            parser.error("workload contains fewer sessions than --clients")
    barrier = threading.Barrier(arguments.clients + 1)
    with ThreadPoolExecutor(max_workers=arguments.clients) as pool:
        futures = [
            pool.submit(workload_client, arguments, address, index, barrier, workload)
            if workload
            else pool.submit(client, arguments, address, index, barrier)
            for index in range(arguments.clients)
        ]
        start = time.monotonic_ns()
        barrier.wait()
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": "client_complete", **result}), flush=True)
    seconds = (time.monotonic_ns() - start) / 1e9
    turns = [turn for result in results for turn in result["turns"]]
    outputs = sum(turn["usage"]["completion_tokens"] for turn in turns)
    inputs = sum(
        turn["usage"]["prompt_tokens"]
        - turn["usage"]["prompt_tokens_details"]["cached_tokens"]
        for turn in turns
    )
    print(
        json.dumps(
            {
                "event": "summary",
                "workload": workload["name"] if workload else "codeword",
                "workload_sha256": workload_digest,
                "clients": arguments.clients,
                "requests": len(turns),
                "elapsed_seconds": seconds,
                "appended_prompt_tokens": inputs,
                "output_tokens_including_eos": outputs,
                "aggregate_output_tokens_per_second": outputs / seconds,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
