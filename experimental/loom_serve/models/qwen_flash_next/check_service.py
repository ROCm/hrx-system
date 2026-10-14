# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Original-checkpoint text and retained-state checks against a dedicated server.

This uses the ordinary HTTP API and the shared streaming client. The caller
owns the server, assets, and outer hang detection. No timing is a benchmark.
"""

import argparse
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from loom_serve_testing.chat import request
from loom_serve_testing.http_harness import control, wait_status


def messages(text):
    return [{"role": "user", "content": text}]


def append(history, reply, text):
    return history + [
        {"role": "assistant", "content": reply["text"]},
        {"role": "user", "content": text},
    ]


def cached(result):
    return result["usage"]["prompt_tokens_details"]["cached_tokens"]


def same_output(actual, expected):
    for field in ("text", "reasoning_content", "tool_calls", "finish_reason"):
        assert actual[field] == expected[field], (field, actual, expected)
    for field in ("prompt_tokens", "completion_tokens"):
        assert actual["usage"][field] == expected["usage"][field], (
            field,
            actual,
            expected,
        )


def run(address, model):
    prefix = "flash-check-" + uuid.uuid4().hex
    headers = {"Content-Type": "application/json", "X-Loom-Model": model}
    selector = json.dumps({"model": model})

    def chat(label, history, maximum=2, checkpoint=None):
        result = request(
            address,
            prefix + "-" + label,
            history,
            maximum,
            model=model,
            checkpoint=checkpoint,
        )
        print(
            json.dumps(
                {
                    "case": label,
                    "text": result["text"],
                    "usage": result["usage"],
                    "finish_reason": result["finish_reason"],
                }
            ),
            flush=True,
        )
        return result

    def operation(name, expected=200):
        return control(
            address,
            "POST",
            "/v1/models/" + name,
            expected,
            headers=headers,
            body=selector,
        )

    def settled():
        while True:
            status = control(address, "GET", "/v1/status", 200)
            state = next(item["loom"] for item in status["data"] if item["id"] == model)
            if "error" in state:
                raise RuntimeError(state)
            if state["parameter_operation"] == "idle":
                return state

    wait_status(address, model=model, phase="idle", active_requests=0)
    cases = [
        ("arithmetic", "What is 2+2? Reply with only the number.", "4"),
        ("instruction", "Reply with exactly the word COBALT.", "COBALT"),
    ]
    with ThreadPoolExecutor(max_workers=2) as clients:
        pending = [
            clients.submit(chat, label, messages(text), 8) for label, text, _ in cases
        ]
        replies = [future.result() for future in pending]
    for (_, _, expected), reply in zip(cases, replies):
        assert reply["text"].strip() == expected, reply

    history = messages("green trees " * 40)
    first = chat("root", history, 1)
    assert first["usage"]["prompt_tokens"] > 64, first
    checkpoint_path = "/v1/checkpoints/" + prefix
    receipt = control(
        address,
        "POST",
        checkpoint_path,
        201,
        session=prefix + "-root",
        headers=headers,
    )
    try:
        assert receipt["position"] > 64, receipt
        assert receipt["host_snapshot_bytes"] == 0, receipt
        branches = [
            append(history, first, text)
            for text in (
                "What is next?",
                "Describe a forest in winter and the quiet snow.",
            )
        ]
        with ThreadPoolExecutor(max_workers=2) as clients:
            pending = [
                clients.submit(chat, f"fork-{i}", branch, 2, prefix)
                for i, branch in enumerate(branches)
            ]
            forks = [future.result() for future in pending]
        for fork in forks:
            assert cached(fork) == receipt["position"], (fork, receipt)

        # Continue before fresh replays can evict idle execution rows.
        continuation = append(branches[0], forks[0], "And then?")
        continued = chat("fork-0", continuation)
        assert cached(continued) > receipt["position"], continued
        for i, branch in enumerate(branches):
            replay = chat(f"replay-{i}", branch)
            same_output(replay, forks[i])
            assert cached(replay) == 0, replay
        replay = chat("replay-continuation", continuation)
        same_output(replay, continued)
        assert cached(replay) == 0, replay

        # Weight ownership is independent of retained endpoints and compiled
        # commands. Release and reload the original shards, then reuse the pin.
        wait_status(address, model=model, phase="idle", active_requests=0)
        before = settled()
        operation("drain")
        try:
            operation("deactivate", 202)
            cold = settled()
            assert not cold["parameters"]["active"], cold
            assert cold["parameters"]["committed_bytes"] == 0, cold
            operation("activate", 202)
            warm = settled()
            assert warm["parameters"]["active"], warm
            assert warm["incarnation"] == before["incarnation"], (before, warm)
            assert warm["compilation"] == before["compilation"], (before, warm)
            assert (
                warm["parameters"]["committed_bytes"]
                == (before["parameters"]["committed_bytes"])
            ), (before, warm)
        finally:
            operation("undrain")
        restored = chat("restored", branches[0], checkpoint=prefix)
        same_output(restored, forks[0])
        assert cached(restored) == receipt["position"], restored
        print(
            json.dumps(
                {
                    "qualified": "text, concurrent forks, page boundary, continuation, replay, and weight reactivation",
                    "parameter_bytes": warm["parameters"]["committed_bytes"],
                    "checkpoint_position": receipt["position"],
                }
            ),
            flush=True,
        )
    finally:
        control(address, "DELETE", checkpoint_path, 200, headers=headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="flash-next")
    args = parser.parse_args()
    run(urlsplit(args.url), args.model)


if __name__ == "__main__":
    main()
