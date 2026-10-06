# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare retained native images with isolated prompt-to-image invocations.

Uses the real checkpoint and source commands, never a fixture model. An older
qualified CLI may be supplied as --generator for a fixed-shape counterfactual.
Keep below 16 MiB of final images/logs; requires a device lease.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", nargs="+", required=True)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def run(name, command):
        (args.output / f"{name}.json").write_text(json.dumps(command, indent=2) + "\n")
        print(json.dumps(dict(event="begin", check=name)), flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        text = result.stdout + result.stderr
        (args.output / f"{name}.log").write_text(text)
        print(text, end="", flush=True)
        result.check_returncode()
        return text

    prompts = ("a red fox in the snow", " ".join(["red"] * 65))
    common = [f"--model={args.model}", f"--checkpoint={args.checkpoint}"]
    isolated_cases = [
        (prompt, adapted) for adapted in (False, True) for prompt in (0, 1)
    ]
    if not args.adapter:
        isolated_cases = isolated_cases[:2]
    for index, (prompt, adapted) in enumerate(isolated_cases):
        run(
            f"isolated-{index}",
            [
                *args.generator,
                *common,
                f"--prompt={prompts[prompt]}",
                f"--seed={42 if prompt else 0}",
                *([f"--adapter={args.adapter}", "--strength=1"] if adapted else []),
                "--height=384",
                "--width=384",
                "--text_tokens=512",
                f"--output={args.output / f'isolated-{index}.ppm'}",
            ],
        )
    retained = run(
        "retained",
        [
            *args.checker,
            *common,
            *([f"--adapter={args.adapter}"] if args.adapter else []),
            f"--output={args.output}",
        ],
    )
    # Model construction is cold; both commands share the same parameter bank.
    events = [
        json.loads(line) for line in retained.splitlines() if line.startswith("{")
    ]
    residencies = [event for event in events if event.get("event") == "image_residency"]
    prepared = [event for event in events if event.get("event") == "image_prepared"]
    assert retained.count('"event":"jit_stage"') == 2
    assert len(residencies) == 1
    residency = residencies[0]
    assert [stage["tag"] for stage in residency["stages"]] == [512, 128]
    assert residency["workspace_bytes"] == max(
        stage["workspace_bytes"] for stage in residency["stages"]
    )
    assert residency["workspace_alignment"] == max(
        stage["workspace_alignment"] for stage in residency["stages"]
    )
    assert residency["kernels"] == sum(
        stage["kernels"] for stage in residency["stages"]
    )
    assert residency["entries"] == sum(
        stage["entries"] for stage in residency["stages"]
    )
    assert retained.count("Streaming ") == (4 if args.adapter else 3)
    rounds = 2 if args.adapter else 1
    assert retained.count("strength must be finite") == 3 * rounds
    assert [event["stage"] for event in prepared] == [1, 0, 1] * rounds
    assert [event["input_bytes"] for event in prepared] == [
        16 + (text + 34) * 4 for text in (128, 512, 128)
    ] * rounds
    for first in range(0, 3 * rounds, 3):
        assert (args.output / f"image-{first}.f32").read_bytes() == (
            args.output / f"image-{first + 2}.f32"
        ).read_bytes()
    references = (0, 1, 0, 2, 3, 2) if args.adapter else (0, 1, 0)
    for index, isolated in enumerate(references):
        rgb = np.fromfile(args.output / f"image-{index}.f32", dtype="<f4")
        assert np.isfinite(rgb).all() and (abs(rgb) <= 1).all()
        pixels = np.rint((rgb.reshape(3, 384, 384).transpose(1, 2, 0) / 2 + 0.5) * 255)
        expected = np.array(Image.open(args.output / f"isolated-{isolated}.ppm"))
        assert np.array_equal(pixels.astype(np.uint8), expected), index
        print(json.dumps(dict(image=index, exact_pixels=True)), flush=True)
    print(
        "PASS: short/long/short images equal isolated output; "
        "invalid requests preserve one shared residency.",
        flush=True,
    )


if __name__ == "__main__":
    main()
