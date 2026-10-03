# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify variable resident-row counts through real concurrent HTTP output.

Sequential dense execution provides the text/usage oracle. Concurrent pooled
execution must advance every configured resident row, exercise MTP and mixed
epochs, and reproduce that output. The caller owns GPU exclusion and the outer
hang deadline; the client barrier establishes arrivals without timed sleeps.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve import benchmark_service, check_service


def request(address, index, barrier=None):
    messages = [
        {"role": "system", "content": "Follow instructions exactly. Do not use tools."},
        {
            "role": "user",
            "content": (
                "Background: this repository has kernels and documentation.\n"
                * (index % 4)
                + "Count from 1 through 1000, one number per line, without commentary."
            ),
        },
    ]
    if barrier:
        barrier.wait()
    return benchmark_service.request(
        address, f"row-{index}", messages, 48 + 8 * (index % 4)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "weights", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=range(1, 17), default=16)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = [str(arguments.server.resolve())] + [
        f"--{name}={getattr(arguments, name).resolve()}"
        for name in ("model", "weights", "tokenizer")
    ]
    command += [
        f"--rows={arguments.rows}",
        "--context_capacity=2048",
        "--prefill_capacity=128",
        f"--epoch=64:{arguments.rows}",
        f"--epoch=128:{arguments.rows}",
        "--mtp",
        "--mtp_depth=3",
        "--port=0",
        "--heartbeat_ms=1000",
    ]
    reference = None
    for capacity in (0, 256 * arguments.rows):
        events = []
        log_path = arguments.output / f"pool-{capacity}.log"
        with check_service.running_server(
            command + [f"--pool_capacity={capacity}"], log_path, events.append
        ) as (_, endpoint):
            if endpoint is None:
                raise RuntimeError(f"server did not become ready: {log_path}")
            address = urlsplit(f"http://{endpoint}")
            if capacity:
                barrier = threading.Barrier(arguments.rows)
                with ThreadPoolExecutor(max_workers=arguments.rows) as clients:
                    futures = [
                        clients.submit(request, address, index, barrier)
                        for index in range(arguments.rows)
                    ]
                    results = [future.result() for future in futures]
            else:
                results = [request(address, index) for index in range(arguments.rows)]
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
        epochs = [event for event in events if event and event.get("event") == "epoch"]
        if capacity:
            visited = {row["row"] for epoch in epochs for row in epoch["rows"]}
            if visited != set(range(arguments.rows)):
                raise RuntimeError(f"resident rows not exercised: {visited}")
            if max(epoch["spans"] for epoch in epochs) != arguments.rows:
                raise RuntimeError("workload did not advance a full resident cohort")
            if not any(epoch["mtp"]["rows"] == arguments.rows for epoch in epochs):
                raise RuntimeError("workload did not verify a full resident cohort")
            if arguments.rows > 1 and not any(
                epoch["prefill_tokens"] and epoch["decode_tokens"] for epoch in epochs
            ):
                raise RuntimeError("workload did not mix prompt and generated input")
            for event in events:
                if event and event.get("event") == "heartbeat":
                    pool = event["pool"]
                    if max(pool["reserved_tokens"], pool["resident_tokens"]) > capacity:
                        raise RuntimeError("pool accounting exceeded its budget")
        print(
            json.dumps(
                {
                    "event": "pass",
                    "rows": arguments.rows,
                    "pool_capacity": capacity,
                    "epochs": len(epochs),
                    "maximum_spans": max(epoch["spans"] for epoch in epochs),
                    "maximum_verifiers": max(epoch["mtp"]["rows"] for epoch in epochs),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
