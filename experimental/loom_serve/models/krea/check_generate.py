# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native fresh-prompt images, LoRA and exact input-to-output wiring.

Runs the production native CLI, an independent reference subprocess, then
the native component observer with independently qualified request bytes.
Requires an exclusive device lease. Retains below 64 MiB without copying
checkpoints. Numeric reference differences are observations, not primitive
tolerance passes. The production CLI has no Python inference dependency.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs="+", required=True)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def run(name, command):
        (args.output / f"{name}.json").write_text(json.dumps(command, indent=2) + "\n")
        print(json.dumps(dict(event="begin", check=name)), flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        (args.output / f"{name}.log").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, end="", flush=True)
        result.check_returncode()
        return result.stdout

    cases = {}
    for name in ("canonical-512", "novel-32"):
        request = json.loads((args.requests / name / "request.json").read_text())
        cases[name] = request
        phases = (
            ("base", "repeat", "zero", "style")
            if name == "canonical-512"
            else ("base", "style")
        )
        for phase in phases:
            adapted = phase in ("zero", "style")
            run(
                name + "-" + phase,
                [
                    *args.native,
                    f"--model={args.model}",
                    f"--checkpoint={args.checkpoint}",
                    f"--prompt={request['prompt']}",
                    f"--seed={request['seed']}",
                    f"--height={request['height']}",
                    f"--width={request['width']}",
                    f"--text_tokens={request['text_tokens']}",
                    *(
                        [
                            f"--adapter={args.adapter / 'softwatercolor.safetensors'}",
                            f"--strength={0 if phase == 'zero' else 1}",
                        ]
                        if adapted
                        else []
                    ),
                    f"--output={args.output / (name + '-' + phase + '.ppm')}",
                ],
            )
        base = (args.output / f"{name}-base.ppm").read_bytes()
        assert base != (args.output / f"{name}-style.ppm").read_bytes(), (
            "active LoRA has no effect"
        )
        if name == "canonical-512":
            assert base == (args.output / f"{name}-repeat.ppm").read_bytes(), (
                "native repeated request differs"
            )
            assert base == (args.output / f"{name}-zero.ppm").read_bytes(), (
                "zero strength changes native image"
            )
        print(
            json.dumps(
                dict(
                    event="native_images",
                    case=name,
                    adapter_changes_image=True,
                    repeated_zero_exact=name == "canonical-512",
                )
            ),
            flush=True,
        )

    reference = args.output / "reference"
    run(
        "reference",
        [
            sys.executable,
            "-B",
            str(args.model / "reference_request.py"),
            f"--checkpoint={args.checkpoint}",
            f"--adapter={args.adapter}",
            f"--requests={args.requests}",
            f"--output={reference}",
        ],
    )
    for name, request in cases.items():
        texts, height, width = (
            request[key] for key in ("text_tokens", "height", "width")
        )
        images = (height // 16) * (width // 16)
        for phase in ("base", "style"):
            adapted = phase == "style"
            weights = [
                args.checkpoint / "text_encoder/model.safetensors",
                args.checkpoint / "turbo.safetensors",
            ]
            if adapted:
                weights.append(args.adapter / "softwatercolor.safetensors")
            weights.append(args.checkpoint / "vae/diffusion_pytorch_model.safetensors")
            actual = args.output / f"{name}-{phase}.f32"
            output = run(
                name + "-" + phase + "-component",
                [
                    *args.checker,
                    f"--model={args.model}",
                    f"--root={'generate_image_adapted' if adapted else 'generate_image'}",
                    f"--weight_policy={args.model / 'weights.loom'}",
                    *[f"--weights={path}" for path in weights],
                    *[
                        f"--config=krea2.{key}={value}"
                        for key, value in dict(
                            block_tokens=texts + images,
                            image_tokens=images,
                            text_tokens=texts,
                            time_count=8,
                            latent_height=height // 8,
                            latent_width=width // 8,
                        ).items()
                    ],
                    *[
                        f"--input={args.requests / name / ('input-' + str(i))}"
                        for i in range(2)
                    ],
                    f"--expected={reference / (name + '-' + phase + '.f32')}",
                    f"--actual={actual}",
                    "--output_type=f32",
                    "--report_only",
                ],
            )
            records = [
                json.loads(line)
                for line in output.splitlines()
                if line.startswith('{"iteration"')
            ]
            assert [record["iteration"] for record in records] == [0, 1]
            rgb = (
                np.fromfile(actual, dtype="<f4")
                .reshape(3, height, width)
                .transpose(1, 2, 0)
            )
            pixels = np.rint((rgb / 2 + 0.5) * 255).astype(np.uint8)
            cli = np.array(Image.open(args.output / f"{name}-{phase}.ppm"))
            assert np.array_equal(cli, pixels), (
                "production caller differs from native command observer"
            )
            Image.fromarray(cli).save(args.output / f"{name}-{phase}.png")
            expected = np.array(Image.open(reference / f"{name}-{phase}.png"))
            mse = float(
                np.square(cli.astype(np.float64) - expected.astype(np.float64)).mean()
            )
            print(
                json.dumps(
                    dict(
                        event="image_comparison",
                        case=name,
                        phase=phase,
                        native_pixels_exact=True,
                        reference_pixel_rmse=math.sqrt(mse),
                        reference_psnr_db=10 * math.log10(255**2 / mse)
                        if mse
                        else None,
                    )
                ),
                flush=True,
            )
    print(
        "PASS: fresh native prompt/seed images, adapter identity and exact final-pixel composition.",
        flush=True,
    )


if __name__ == "__main__":
    main()
