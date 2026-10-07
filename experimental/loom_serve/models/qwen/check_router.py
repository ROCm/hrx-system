# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real multi-model HTTP admission, reload and independent retained-state witness.

The same source/weights are deployed with two names and different shape/context
settings. Sequential requests provide a token/usage oracle for concurrent requests
under a retained budget fitting only one model. This is correctness, not timing.
The caller supplies an exact built server and qualified, exclusive GPU runner.
"""

import argparse
import http.client
import json
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from experimental.loom_serve.models.qwen.benchmark_service import request
from experimental.loom_serve.models.qwen.check_service import (
    Events,
    control,
    running_server,
)


def history(word):
    return [
        {"role": "system", "content": "Follow instructions exactly. Do not use tools."},
        {
            "role": "user",
            "content": f"Remember my secret codeword {word}. "
            "Count from 1 through 1000, one number per line, without commentary.",
        },
    ]


def continuation(messages, first):
    return messages + [
        {"role": "assistant", "content": first["text"]},
        {"role": "user", "content": "Reply with only my secret codeword."},
    ]


def run(command, log, concurrent):
    events = Events()
    results = {}
    with running_server(command, log, events.record) as (_, ready):
        if ready is None:
            raise RuntimeError(f"router did not reach readiness: {log}")
        address = urlsplit(f"http://{ready}")
        catalog = control(address, "GET", "/v1/models", 200)
        if [item["id"] for item in catalog["data"]] != ["scout", "researcher"]:
            raise RuntimeError(f"wrong route catalog: {catalog}")
        messages = {"scout": history("MAPLE"), "researcher": history("COBALT")}
        if concurrent:
            with ThreadPoolExecutor(max_workers=2) as clients:
                a = clients.submit(
                    request, address, "shared", messages["scout"], 128, model="scout"
                )
                events.wait("admit", "shared", model="scout")
                b = clients.submit(
                    request,
                    address,
                    "shared",
                    messages["researcher"],
                    96,
                    model="researcher",
                )
                events.wait("enqueue", "shared", model="researcher")
                cancelled = http.client.HTTPConnection(address.hostname, address.port)
                try:
                    cancelled.request(
                        "POST",
                        "/v1/chat/completions",
                        json.dumps(
                            {
                                "model": "researcher",
                                "messages": messages["researcher"],
                                "stream": True,
                                "max_tokens": 16,
                                "temperature": 0,
                            }
                        ),
                        {
                            "Content-Type": "application/json",
                            "X-Loom-Session": "cancelled",
                        },
                    )
                    events.wait("enqueue", "cancelled", model="researcher")
                    cancelled.sock.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                finally:
                    cancelled.close()
                events.wait("cancel_queued", "cancelled", model="researcher")
                control(address, "GET", "/healthz", 200)
                results["scout"] = [a.result()]
                results["researcher"] = [b.result()]
        else:
            for name, maximum in (("scout", 128), ("researcher", 96)):
                results[name] = [
                    request(address, "shared", messages[name], maximum, model=name)
                ]
        for name, word in (("scout", "MAPLE"), ("researcher", "COBALT")):
            receipt = control(
                address,
                "POST",
                "/v1/checkpoints/shared",
                201,
                session="shared",
                headers={"X-Loom-Model": name},
            )
            followup = continuation(messages[name], results[name][0])
            second = request(
                address, "shared", followup, 16, checkpoint="shared", model=name
            )
            cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
            if second["text"].strip() != word or cached != receipt["position"]:
                raise RuntimeError(f"{name}: lost independent pinned state: {second}")
            results[name].append(second)
        # Identical checkpoint/session keys belong to different routes.
        control(
            address,
            "DELETE",
            "/v1/checkpoints/shared",
            200,
            headers={"X-Loom-Model": "scout"},
        )
        control(
            address,
            "DELETE",
            "/v1/checkpoints/shared",
            200,
            headers={"X-Loom-Model": "researcher"},
        )
        control(address, "POST", "/v1/checkpoints/shared", 400, session="shared")
        control(
            address,
            "POST",
            "/v1/chat/completions",
            404,
            body=json.dumps({"model": "unknown"}),
            headers={"Content-Type": "application/json"},
        )
    # A must relinquish its pin before the queued B can admit; no new request
    # is sent to B to wake it. Completed A state survives B's weight activation.
    if concurrent:
        admitted = {
            event["model"]: event
            for event in events.values
            if event.get("event") == "admit" and event.get("cache") == "replay"
        }
        if admitted["researcher"]["queue_ms"] <= 0:
            raise RuntimeError("second model did not wait on shared pressure")
        handoff = events.values.index(admitted["researcher"])
        produced = sum(
            event["selected_tokens_including_eos"]
            for event in events.values[:handoff]
            if event.get("model") == "scout" and event.get("event") == "epoch"
        )
        if produced != results["scout"][0]["usage"]["completion_tokens"]:
            raise RuntimeError("cap did not force a model handoff")
    heartbeats = [event for event in events.values if event.get("event") == "heartbeat"]
    if not all(
        any(event["model"] == name for event in heartbeats)
        for name in ("scout", "researcher")
    ):
        raise RuntimeError("missing per-route telemetry")
    return results, heartbeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "weights", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--memory-bytes", type=int, default=22 * 1024**3)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    models = []
    for name, context, prefill, depth in (
        ("scout", 2048, 128, 3),
        ("researcher", 4096, 64, 0),
    ):
        models.append(
            {
                "kind": "text",
                "name": name,
                "source": str(arguments.model.resolve()),
                "weights": str(arguments.weights.resolve()),
                "tokenizer": str(arguments.tokenizer.resolve()),
                "rows": 2,
                "context_capacity": context,
                "pool_capacity": 4096,
                "prefill_capacity": prefill,
                "checkpoint_capacity": 1,
                "mtp_depth": depth,
                "continuation_epochs": 2 if depth else 1,
            }
        )
    catalog = arguments.output / "models.json"
    catalog.write_text(json.dumps({"models": models}, indent=2) + "\n")
    command = [
        str(arguments.server.resolve()),
        f"--models={catalog.resolve()}",
        "--port=0",
        "--heartbeat_ms=250",
        "--pool_backing=elastic",
        f"--memory_bytes={arguments.memory_bytes}",
    ]
    baseline, _ = run(command, arguments.output / "sequential.log", False)
    print(json.dumps({"event": "sequential_complete", "results": baseline}), flush=True)
    candidate, heartbeats = run(command, arguments.output / "concurrent.log", True)
    for name in baseline:
        for expected, actual in zip(baseline[name], candidate[name], strict=True):
            for field in ("text", "tool_calls", "usage", "finish_reason"):
                if expected[field] != actual[field]:
                    raise RuntimeError(f"{name}: concurrent {field} differs")
    peak = max(
        event["device_memory"]["retained_committed_bytes"] for event in heartbeats
    )
    reuse = max(event["device_memory"]["workspace_reuse_count"] for event in heartbeats)
    if peak > arguments.memory_bytes or reuse == 0:
        raise RuntimeError(f"shared pool contract failed: {peak=}, {reuse=}")
    print(
        json.dumps(
            {
                "event": "pass",
                "models": 2,
                "matched_requests": 4,
                "retained_peak_observed": peak,
                "workspace_reuse_count": reuse,
                "results": candidate,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
