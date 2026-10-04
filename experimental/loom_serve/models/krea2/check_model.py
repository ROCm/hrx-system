# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare retained native images with isolated prompt-to-image invocations.

Uses the real checkpoint and source commands, never a fixture model. An older
qualified CLI may be supplied as --generator for a controlled extraction
comparison. Keep below 16 MiB of final images/logs; requires a device lease.
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
    parser.add_argument("--adapter", type=Path, required=True)
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

    prompts = (
        "A deer grazing in the forest, Art Deco watercolor style",
        "A small brass robot tending red flowers in a sunlit greenhouse, watercolor illustration",
    )
    common = [f"--model={args.model}", f"--checkpoint={args.checkpoint}"]
    for index in range(3):
        run(
            f"isolated-{index}",
            [
                *args.generator,
                *common,
                f"--prompt={prompts[index == 1]}",
                f"--seed={42 if index == 1 else 0}",
                *([f"--adapter={args.adapter}", "--strength=1"] if index == 2 else []),
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
            f"--adapter={args.adapter}",
            f"--output={args.output}",
        ],
    )
    # Model construction happens once even across invalid, base and LoRA calls.
    assert retained.count('"event":"jit_stage"') == 1
    assert retained.count('"event":"image_residency"') == 1
    assert retained.count("Streaming ") == 4
    assert retained.count("adapter strength must be finite") == 4
    first = (args.output / "image-0.f32").read_bytes()
    assert first == (args.output / "image-3.f32").read_bytes()
    for index, isolated in enumerate((0, 1, 2, 0)):
        rgb = np.fromfile(args.output / f"image-{index}.f32", dtype="<f4")
        assert np.isfinite(rgb).all() and (abs(rgb) <= 1).all()
        pixels = np.rint((rgb.reshape(3, 384, 384).transpose(1, 2, 0) / 2 + 0.5) * 255)
        expected = np.array(Image.open(args.output / f"isolated-{isolated}.ppm"))
        assert np.array_equal(pixels.astype(np.uint8), expected), index
        print(json.dumps(dict(image=index, exact_pixels=True)), flush=True)
    print(
        "PASS: retained base/novel/LoRA/base images equal isolated output; invalid requests preserve residency.",
        flush=True,
    )


if __name__ == "__main__":
    main()
