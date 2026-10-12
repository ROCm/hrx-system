# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify variable resident-row counts through real concurrent HTTP output.

Sequential dense execution with explicit shapes provides the text/usage oracle.
Concurrent pooled execution must advance every configured resident row, exercise
MTP and mixed epochs, and reproduce that output. The candidate uses the automatic
catalog by default. The caller owns GPU exclusion and the outer hang deadline;
the client barrier establishes arrivals without timed sleeps.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.http import chat
from experimental.loom_serve.testing.http import Events, running_server


def request(address, index, barrier=None, *, long_prompt=False):
    messages = [
        {"role": "system", "content": "Follow instructions exactly. Do not use tools."},
        {
            "role": "user",
            "content": (
                "Background: this repository has kernels and documentation.\n"
                * (128 if long_prompt else index % 4)
                + "Count from 1 through 1000, one number per line, without commentary."
            ),
        },
    ]
    if barrier:
        barrier.wait()
    return chat.request(
        address, f"row-{index}", messages, 48 + 8 * (index % 4), model="qwen3.8-27b"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "weights", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--continuation-epochs", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--catalog", choices=("automatic", "explicit"), default="automatic"
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = [str(arguments.server.resolve())] + [
        f"--{name}={getattr(arguments, name).resolve()}"
        for name in ("model", "weights", "tokenizer")
    ]
    command += [
        f"--rows={arguments.rows}",
        "--context_capacity=2048",
        "--prefill_capacity=512",
        "--mtp",
        "--mtp_depth=3",
        "--port=0",
        "--heartbeat_ms=1000",
    ]
    reference = None
    # One prompt exceeds two largest-shape epochs while the short rows decode.
    # Even if it is packed first, the rotating next epoch finishes short rows
    # before it can finish, leaving real decode work to mix into its final chunk.
    # Arrival skew alone cannot establish mixed-work coverage: all short prompts
    # may be ready together and fit in a single epoch. Completion reservations
    # cover the long prompt too, so admission does not serialize this witness.
    pooled_capacity = 256 * arguments.rows + (1024 if arguments.rows > 1 else 0)
    for capacity in (0, pooled_capacity):
        observations = Events()
        log_path = arguments.output / f"pool-{capacity}.log"
        shapes = (
            [f"--epoch=64:{arguments.rows}", f"--epoch=128:{arguments.rows}"]
            if not capacity or arguments.catalog == "explicit"
            else []
        )
        continuation_epochs = arguments.continuation_epochs if capacity else 1
        with running_server(
            command
            + shapes
            + [
                f"--pool_capacity={capacity}",
                f"--continuation_epochs={continuation_epochs}",
            ],
            log_path,
            observations.record,
        ) as (_, endpoint):
            if endpoint is None:
                raise RuntimeError(f"server did not become ready: {log_path}")
            address = urlsplit(f"http://{endpoint}")
            if capacity:
                barrier = threading.Barrier(arguments.rows)
                with ThreadPoolExecutor(max_workers=arguments.rows) as clients:
                    futures = [
                        clients.submit(
                            request,
                            address,
                            index,
                            barrier,
                            long_prompt=arguments.rows > 1
                            and index == arguments.rows - 1,
                        )
                        for index in range(arguments.rows)
                    ]
                    results = [future.result() for future in futures]
            else:
                results = [
                    request(
                        address,
                        index,
                        long_prompt=arguments.rows > 1 and index == arguments.rows - 1,
                    )
                    for index in range(arguments.rows)
                ]
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
            raise RuntimeError("concurrent pooled output differs from dense control")
        # This coverage proof requires complete history. A slow observer fails
        # explicitly on a gap; HTTP output remains the independent oracle.
        events = observations.values
        epochs = [event for event in events if event["event"] == "epoch"]
        spans = [event for event in events if event["event"] == "span"]
        verifiers = {
            epoch["epoch"]: sum(
                span["verification_epochs"] != 0
                for span in spans
                if span["epoch"] == epoch["epoch"]
            )
            for epoch in epochs
        }
        if capacity:
            if arguments.continuation_epochs == 2 and not any(
                span["verification_epochs"] == 2 for span in spans
            ):
                raise RuntimeError("workload did not execute device-fed continuation")
            if arguments.continuation_epochs == 2:
                if not any(
                    span["kind"] == "prefill"
                    and span["known_tokens"]
                    and span["verification_epochs"]
                    for span in spans
                ):
                    raise RuntimeError(
                        "workload did not promote a prompt tail on device"
                    )
                if arguments.rows > 1 and not any(
                    epoch["continued_prefill_tokens"] for epoch in epochs
                ):
                    raise RuntimeError("continuation did not carry ready prompt work")
            visited = {span["row"] for span in spans}
            if visited != set(range(arguments.rows)):
                raise RuntimeError(f"resident rows not exercised: {visited}")
            if max(epoch["spans"] for epoch in epochs) != arguments.rows:
                raise RuntimeError("workload did not advance a full resident cohort")
            if arguments.rows not in verifiers.values():
                raise RuntimeError("workload did not verify a full resident cohort")
            if arguments.rows > 1 and not any(
                epoch["prefill_tokens"] and epoch["decode_tokens"] for epoch in epochs
            ):
                raise RuntimeError("workload did not mix prompt and generated input")
            if arguments.catalog == "automatic":
                for epoch in epochs:
                    expected_spans = 1
                    while expected_spans < epoch["spans"]:
                        expected_spans = min(expected_spans * 2, arguments.rows)
                    if epoch["span_capacity"] != expected_spans:
                        raise RuntimeError("catalog did not shrink to the ready cohort")
                if arguments.rows == 16 and not any(
                    epoch["token_capacity"] == 512 for epoch in epochs
                ):
                    raise RuntimeError(
                        "workload did not exercise the largest token shape"
                    )
            for event in events:
                if event["event"] == "text_memory":
                    pool = event["pool"]
                    if pool["reserved_tokens"] + pool["resident_tokens"] > capacity:
                        raise RuntimeError("pool accounting exceeded its budget")
        print(
            json.dumps(
                {
                    "event": "pass",
                    "rows": arguments.rows,
                    "pool_capacity": capacity,
                    "epochs": len(epochs),
                    "device_epochs": sum(epoch["traversals"] for epoch in epochs),
                    "continuation_epochs": continuation_epochs,
                    "maximum_spans": max(epoch["spans"] for epoch in epochs),
                    "maximum_verifiers": max(verifiers.values()),
                    "shapes": sorted(
                        {
                            (epoch["token_capacity"], epoch["span_capacity"])
                            for epoch in epochs
                        }
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
