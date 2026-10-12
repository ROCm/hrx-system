# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bounded concurrent HTTP workload with retained, independently checked turns.

Run against an already ready text deployment. Timing covers the complete client
cohort, including SSE and continuation requests; model loading is excluded.
Server epoch events provide the actual prompt/decode mix and traversal costs.
"""

import argparse
import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.http.chat import request

WORDS = ("MAPLE", "COBALT", "CEDAR", "QUARTZ", "AMBER", "CORAL", "JADE", "ONYX")


def client(arguments, address, index, barrier):
    word = WORDS[index % len(WORDS)]
    if index >= len(WORDS):
        word += f"-{index // len(WORDS)}"
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
    first = request(
        address, session, messages, arguments.max_tokens, model=arguments.model
    )
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
    second = request(address, session, messages, 16, model=arguments.model)
    if second["text"].strip() != word:
        raise RuntimeError(f"{session}: expected {word!r}, got {second['text']!r}")
    if (
        arguments.retention == "required"
        and not second["usage"]["prompt_tokens_details"]["cached_tokens"]
    ):
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
        if (
            type(session.get("start_after_ms", 0)) is not int
            or session.get("start_after_ms", 0) < 0
        ):
            raise ValueError(
                f"session {index}: start_after_ms must be nonnegative integer"
            )
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
            if type(turn.get("pause_ms", 0)) is not int or turn.get("pause_ms", 0) < 0:
                raise ValueError(
                    f"session {index}: pause_ms must be nonnegative integer"
                )
    return workload, hashlib.sha256(payload).hexdigest()


def workload_client(arguments, address, index, barrier, workload):
    fixture = workload["sessions"][index]
    messages = [{"role": "system", "content": fixture["system"]}]
    session = f"{arguments.session_prefix}-{index}"
    results = []
    barrier.wait()
    if fixture.get("start_after_ms", 0):
        time.sleep(fixture["start_after_ms"] / 1000)
    for turn in fixture["turns"]:
        if turn.get("pause_ms", 0):
            time.sleep(turn["pause_ms"] / 1000)
        messages.append({"role": "user", "content": turn["content"]})
        input_digest = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        result = request(
            address, session, messages, turn["max_tokens"], model=arguments.model
        )
        if not result["text"].strip():
            raise RuntimeError(f"{session}: empty response")
        if (
            results
            and arguments.retention == "required"
            and not result["usage"]["prompt_tokens_details"]["cached_tokens"]
        ):
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


def latency_distribution(values):
    """Nearest-rank milliseconds from complete client samples, not lossy events."""
    samples = sorted(value for value in values if value is not None)
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "minimum": samples[0],
        "p50": samples[math.ceil(len(samples) * 0.50) - 1],
        "p95": samples[math.ceil(len(samples) * 0.95) - 1],
        "p99": samples[math.ceil(len(samples) * 0.99) - 1],
        "maximum": samples[-1],
    }


def followup_cache_summary(results):
    """Measures reuse only on subsequent turns, including partial prefix hits."""
    followups = [turn for result in results for turn in result["turns"][1:]]
    groups = {"hits": [], "misses": []}
    for turn in followups:
        cached = turn["usage"]["prompt_tokens_details"]["cached_tokens"]
        groups["hits" if cached else "misses"].append(turn)
    summary = {"requests": len(followups)}
    for name, turns in groups.items():
        cached = sum(
            turn["usage"]["prompt_tokens_details"]["cached_tokens"] for turn in turns
        )
        summary[name] = {
            "requests": len(turns),
            "cached_prompt_tokens": cached,
            "appended_prompt_tokens": sum(
                turn["usage"]["prompt_tokens"] for turn in turns
            )
            - cached,
            "request_latency_ms": latency_distribution(
                turn["request_ms"] for turn in turns
            ),
            "first_text_latency_ms": latency_distribution(
                turn["first_text_ms"] for turn in turns
            ),
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--model", default="qwen3.8-27b", help="Registered deployment name."
    )
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--long-lines", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--session-prefix", default="service-benchmark")
    parser.add_argument(
        "--retention",
        choices=("required", "observe"),
        default="required",
        help="Require every follow-up to reuse state, or record misses under pressure.",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        help="Frozen JSON text-turn corpus; replaces the codeword workload.",
    )
    arguments = parser.parse_args(argv)
    address = urlsplit(arguments.url)
    if (
        address.scheme != "http"
        or not address.hostname
        or address.path not in ("", "/")
    ):
        parser.error("url must be an HTTP origin")
    if arguments.clients < 1:
        parser.error("clients must be positive")
    if not arguments.model:
        parser.error("model must be nonempty")
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
    cached = sum(
        turn["usage"]["prompt_tokens_details"]["cached_tokens"] for turn in turns
    )
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
                "model": arguments.model,
                "retention": arguments.retention,
                "clients": arguments.clients,
                "requests": len(turns),
                "elapsed_seconds": seconds,
                "appended_prompt_tokens": inputs,
                "cached_prompt_tokens": cached,
                "aggregate_appended_prompt_tokens_per_second": inputs / seconds,
                "output_tokens_including_eos": outputs,
                "aggregate_output_tokens_per_second": outputs / seconds,
                "followup_cache": followup_cache_summary(results),
                "request_latency_ms": latency_distribution(
                    turn["request_ms"] for turn in turns
                ),
                "first_text_latency_ms": latency_distribution(
                    turn["first_text_ms"] for turn in turns
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
