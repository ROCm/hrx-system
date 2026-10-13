# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Real Qwen/image/Qwen HTTP ownership witness under shared memory pressure.

Requires an exact built shared server, Qwen and Krea source/assets, Pillow,
and check_model.py's independent isolated-0.ppm reference. Run from the repo
root with python -m and PYTHONPATH=loom/src/loom/serve/testing on a qualified
exclusive GPU runner. One worker must serve
health and cancellation while device work is outstanding. Text-only execution
provides the exact continuation oracle. This checks ownership, not performance.
Artifacts are catalogs, JSONL and one PNG, under 16 MiB; weights are not copied.
"""

import argparse
import base64
import http.client
import io
import json
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.chat import request
from loom_serve_testing.http import (
    Events,
    control,
    running_server,
    wait_counters,
    wait_status,
)
from PIL import Image

from experimental.loom_serve.models.qwen.check_router import continuation, history


def send_image(address):
    peer = socket.create_connection((address.hostname, address.port))
    body = json.dumps(
        {
            "model": "illustrator",
            "prompt": "a red fox in the snow",
            "seed": "0",
            "strength": 0,
            "size": "384x384",
            "response_format": "b64_json",
        }
    ).encode()
    peer.sendall(
        b"POST /v1/images/generations HTTP/1.1\r\nHost: localhost\r\n"
        b"Content-Type: application/json\r\nConnection: close\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    return peer


def cancel(peer):
    peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    peer.close()


def receive_image(peer, isolated, output):
    with http.client.HTTPResponse(peer) as response:
        response.begin()
        value = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"image failed: {response.status}, {value}")
    if len(value["data"]) != 1:
        raise RuntimeError(f"wrong image response: {value}")
    png = base64.b64decode(value["data"][0]["b64_json"], validate=True)
    output.write_bytes(png)
    with Image.open(io.BytesIO(png)) as image, Image.open(isolated) as expected:
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (384, 384)
            or image.tobytes() != expected.tobytes()
        ):
            raise RuntimeError("mixed-server PNG differs from isolated native pixels")
    print(json.dumps({"event": "exact_http_pixels", "case": "mixed"}), flush=True)


def pin(address):
    return control(
        address,
        "POST",
        "/v1/checkpoints/anchor",
        201,
        session="conversation",
        headers={"X-Loom-Model": "scout"},
    )


def compare(first, second):
    for field in ("text", "tool_calls", "usage", "finish_reason"):
        if first[field] != second[field]:
            raise RuntimeError(f"mixed-server {field} differs: {first}, {second}")


def run(command, directory, isolated=None):
    events = Events()
    messages = history("MAPLE")
    with running_server(command, directory / "server.log", events.record) as (_, ready):
        if ready is None:
            raise RuntimeError(f"server did not reach readiness: {directory}")
        address = urlsplit(f"http://{ready}")
        expected_names = ["scout", "illustrator"] if isolated else ["scout"]
        catalog = control(address, "GET", "/v1/models", 200)
        if [item["id"] for item in catalog["data"]] != expected_names:
            raise RuntimeError(f"wrong deployment catalog: {catalog}")
        print(json.dumps({"event": "ready", "models": expected_names}), flush=True)
        with ThreadPoolExecutor(max_workers=1) as client:
            first_task = client.submit(
                request, address, "conversation", messages, 128, model="scout"
            )
            if isolated:
                wait_status(
                    address, model="scout", active_requests=1, queued_requests=0
                )
                with send_image(address) as waiting:
                    wait_status(
                        address,
                        model="illustrator",
                        active_requests=1,
                        queued_requests=0,
                        phase="waiting",
                    )
                    cancel(waiting)
                wait_counters(
                    address,
                    lambda counts: counts["cancelled"] == 1,
                    model="illustrator",
                )
                print(
                    json.dumps({"event": "cancelled_before_image_admission"}),
                    flush=True,
                )
            first = first_task.result()
            receipt = pin(address)
            followup = continuation(messages, first)
            if isolated:
                with send_image(address) as image:
                    wait_status(
                        address,
                        model="illustrator",
                        active_requests=1,
                        queued_requests=0,
                        phase="generating",
                    )
                    # A second pending image must own no device storage and be
                    # cancellable without delaying the active image or text.
                    with send_image(address) as queued:
                        wait_status(
                            address,
                            model="illustrator",
                            active_requests=1,
                            queued_requests=1,
                        )
                        cancel(queued)
                    wait_counters(
                        address,
                        lambda counts: counts["cancelled"] == 2,
                        model="illustrator",
                    )
                    second_task = client.submit(
                        request,
                        address,
                        "conversation",
                        followup,
                        16,
                        checkpoint="anchor",
                        model="scout",
                    )
                    wait_status(
                        address, model="scout", active_requests=0, queued_requests=1
                    )
                    if control(address, "GET", "/healthz", 200) != {"status": "ready"}:
                        raise RuntimeError("health failed during image work")
                    receive_image(image, isolated, directory / "image.png")
                    second = second_task.result()
            else:
                second = request(
                    address,
                    "conversation",
                    followup,
                    16,
                    checkpoint="anchor",
                    model="scout",
                )
        cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        if second["text"].strip() != "MAPLE" or cached != receipt["position"]:
            raise RuntimeError(
                f"checkpoint did not preserve the conversation: {second}"
            )
        control(
            address,
            "DELETE",
            "/v1/checkpoints/anchor",
            200,
            headers={"X-Loom-Model": "scout"},
        )
    results = [first, second]
    (directory / "responses.json").write_text(json.dumps(results, indent=2) + "\n")
    return results, events.values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "server",
        "text-source",
        "weights",
        "tokenizer",
        "image-source",
        "checkpoint",
        "adapter",
        "isolated",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-bytes", type=int, default=36 * 1024**3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text = dict(
        name="scout",
        base_directory=str(args.text_source.resolve()),
        definition=json.loads((args.text_source / "model.json").read_text())[
            "definition"
        ],
    )
    text["definition"].update(
        weights=str(args.weights.resolve()),
        tokenizer=str(args.tokenizer.resolve()),
        rows=2,
        context_capacity=2048,
        pool_capacity=4096,
        prefill_capacity=128,
        checkpoint_capacity=1,
        draft_depth=3,
        continuation_epochs=2,
    )
    image = dict(
        name="illustrator",
        base_directory=str(args.image_source.resolve()),
        definition=json.loads((args.image_source / "model.json").read_text())[
            "definition"
        ],
    )
    image["definition"].update(
        checkpoint=str(args.checkpoint.resolve()),
        adapter=str(args.adapter.resolve()),
        width=384,
        height=384,
        text_tokens=512,
        pending_requests=1,
    )
    runs = []
    for name, models, isolated in (
        ("text-only", [text], None),
        ("mixed", [text, image], args.isolated / "isolated-0.ppm"),
    ):
        directory = args.output / name
        directory.mkdir()
        documents = []
        for model in models:
            document = directory / f"{model['name']}.json"
            document.write_text(json.dumps(model, indent=2) + "\n")
            documents.append(document.resolve())
        command = [
            str(args.server.resolve()),
            "--device=amdgpu",
            *[f"--model={path}" for path in documents],
            "--listen=127.0.0.1:0",
            "--heartbeat_ms=25",
            f"--workers={args.workers}",
            "--pool_backing=elastic",
            f"--memory_bytes={args.memory_bytes}",
        ]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        runs.append(run(command, directory, isolated))
        print(json.dumps({"event": "run_complete", "case": name}), flush=True)
    for baseline, candidate in zip(runs[0][0], runs[1][0], strict=True):
        compare(baseline, candidate)
    sequence = runs[1][1]
    image_completions = [
        event
        for event in sequence
        if event["event"] == "image_generated" and event["model"] == "illustrator"
    ]
    admissions = [
        event
        for event in sequence
        if event["event"] == "admit" and event["model"] == "scout"
    ]
    # Separate stream reads have no cross-model order. The native producer
    # timestamps establish successful pin release before text readmission.
    if (
        len(image_completions) != 1
        or len(admissions) != 2
        or not admissions[1]["retained_tokens"]
        or admissions[1]["time_ns"] <= image_completions[0]["time_ns"]
    ):
        raise RuntimeError("cancellation or shared-pressure handoff failed")
    memory = [
        event
        for event in sequence
        if event["event"] == "text_memory" and event["model"] == "scout"
    ]
    if not any(
        event["elastic_parameters"]["committed_bytes"] == 0
        and event["elastic_parameters"]["released_bytes"] > 0
        for event in memory
    ):
        raise RuntimeError("text weights were not evicted during the image")
    shared = [event for event in sequence if event["event"] == "shared_device_memory"]
    peak = max(event["retained_committed_bytes"] for event in shared)
    reuse = max(event["workspace_reuse_count"] for event in shared)
    if peak > args.memory_bytes or reuse == 0:
        raise RuntimeError(f"shared pool contract failed: {peak=}, {reuse=}")
    print(
        json.dumps(
            {
                "event": "pass",
                "matched_text_requests": 2,
                "matched_images": 1,
                "retained_peak_observed": peak,
                "workspace_reuse_count": reuse,
                "workers": args.workers,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
