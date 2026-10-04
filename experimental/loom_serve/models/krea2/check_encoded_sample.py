# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native token-IDs-to-RGB against independently executed native stages.

The encoder's retained native taps feed a qualification-only image root. The
production command must reproduce that final RGB bit-for-bit twice, including
base/zero/active LoRA, source-owned tap lifetime and derived key mask. Pixel
differences from canonical and previous native images remain observations.
No external inference framework is loaded. Request construction is still an
external fixture boundary; this is not a prompt-to-image serving check.
Retained RGB, PNGs and commands stay below 96 MiB; taps/weights are reused.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from check_sample import captured_tensor, pixels
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--encoder_results", type=Path, required=True)
    parser.add_argument("--denoise_results", type=Path, required=True)
    parser.add_argument("--stack_reference", type=Path, required=True)
    parser.add_argument("--input_results", type=Path, required=True)
    parser.add_argument("--sample_results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    encoder_inputs = [
        arguments.encoder_results / ("512-" + name)
        for name in ("tokens.i32", "cosine.bf16", "sine.bf16", "mask.u8")
    ]
    token_ids = np.fromfile(encoder_inputs[0], dtype="<i4")
    encoder_mask = np.fromfile(encoder_inputs[3], dtype="u1")
    if (
        token_ids.shape != (546,)
        or (token_ids < 0).any()
        or (token_ids >= 151936).any()
        or encoder_mask.shape != (560,)
        or (encoder_mask > 1).any()
        or not encoder_mask[0]
        or encoder_mask[-14:].any()
    ):
        raise ValueError("expected validated 546-row IDs and padded 560-row mask")
    for phase in ("base", "style"):
        mask = np.fromfile(
            arguments.stack_reference / f"{phase}-block27-mask.u8", dtype="u1"
        )
        if (
            mask.shape != (1088,)
            or not np.array_equal(mask[:512], encoder_mask[34:546])
            or not (mask[512:] == 1).all()
        ):
            raise ValueError(
                f"{phase}: combined mask does not derive from encoder mask"
            )

    def execute(
        name,
        phase,
        adapted,
        expected,
        *,
        integrated,
        overrides=None,
        reference_taps=None,
    ):
        exact = integrated or reference_taps is not None
        directory = arguments.denoise_results / phase
        weights = []
        if integrated:
            weights.append(arguments.checkpoint / "text_encoder/model.safetensors")
        weights.append(arguments.checkpoint / "turbo.safetensors")
        if adapted:
            weights.append(arguments.adapter / "softwatercolor.safetensors")
        weights.append(arguments.checkpoint / "vae/diffusion_pytorch_model.safetensors")
        inputs = [(directory / "sample.bf16", 576 * 64 * 2)]
        if integrated:
            inputs.extend(
                zip(encoder_inputs, (546 * 4, 560 * 128 * 2, 560 * 128 * 2, 560))
            )
            root = "sample_image_adapted" if adapted else "sample_image"
        else:
            inputs.append(
                (
                    reference_taps or arguments.encoder_results / "taps-512.bf16",
                    512 * 12 * 2560 * 2,
                )
            )
            root = (
                "qualify.sample_from_taps_adapted"
                if adapted
                else "qualify.sample_from_taps"
            )
        inputs.append((directory / "times.bf16", 8 * 2))
        inputs.extend(
            (arguments.stack_reference / f"{phase}-block27-{kind}.f32", 1088 * 128 * 4)
            for kind in ("cosine", "sine")
        )
        if not integrated:
            inputs.append(
                (arguments.stack_reference / f"{phase}-block27-mask.u8", 1088)
            )
        inputs.append((directory / "deltas.f32", 8 * 4))
        if adapted:
            inputs.append((directory / "strength.f32", 4))
        inputs.append(
            (arguments.input_results / f"{phase}-48x48-composed/input-1", 2 * 16 * 4)
        )
        for path, size in [*inputs, (expected, 3 * 384 * 384 * 4)]:
            if path.stat().st_size != size:
                raise ValueError(f"{path}: expected {size} bytes for this request")
        configuration = dict(
            block_tokens=1088,
            image_tokens=576,
            text_tokens=512,
            time_count=8,
            latent_height=48,
            latent_width=48,
        )
        configuration.update(overrides or {})
        invocation = [
            *arguments.checker,
            f"--model={arguments.model / 'qualification'}",
            f"--root={root}",
            f"--weight_policy={arguments.model / 'weights.loom'}",
            *[f"--weights={path}" for path in weights],
            *[f"--config=krea2.{key}={value}" for key, value in configuration.items()],
            *[f"--input={path}" for path, _ in inputs],
            f"--expected={expected}",
            f"--actual={arguments.output / (name + '.f32')}",
            "--output_type=f32",
            *(["--atol=0", "--rtol=0"] if exact else ["--report_only"]),
        ]
        (arguments.output / (name + ".json")).write_text(
            json.dumps(dict(check=name, command=invocation), indent=2) + "\n"
        )
        print(json.dumps(dict(check=name, root=root, exact=exact)), flush=True)
        result = subprocess.run(invocation, capture_output=True, text=True)
        (arguments.output / (name + ".log")).write_text(result.stdout + result.stderr)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        if overrides:
            diagnostic = result.stdout + result.stderr
            if (
                result.returncode == 0
                or not all(
                    marker in diagnostic
                    for marker in ("krea2.sample_image", "all_rejected")
                )
                or '"workspace_bytes"' in diagnostic
            ):
                raise AssertionError(f"{name}: expected early geometry rejection")
            print(json.dumps(dict(check=name, geometry_rejected=True)), flush=True)
            return
        result.check_returncode()
        records = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        comparisons = [record for record in records if "iteration" in record]
        if [record["iteration"] for record in comparisons] != [0, 1]:
            raise AssertionError("both native executions must complete")
        if exact and (
            any(record["different"] for record in comparisons)
            or (arguments.output / (name + ".f32")).read_bytes()
            != expected.read_bytes()
        ):
            raise AssertionError(
                f"{name}: final RGB differs bitwise from native stages"
            )
        footprint = next(record for record in records if "workspace_bytes" in record)
        expected_footprint = dict(
            kernels=(95 if adapted else 71) + (14 if integrated else 0),
            parameters=(1062 if adapted else 534) + (386 if integrated else 0),
            parameter_roots=(3 if adapted else 2) + int(integrated),
            parameter_bytes=(27038612748 if adapted else 26569389580)
            + (7843069440 if integrated else 0),
        )
        # Bound phase-packing fragmentation by one largest VAE feature plane.
        maximum_workspace = (
            (275251200 if adapted else 273678336) + 512 * 2560 * 2 + 96 * 384 * 384 * 4
        )
        if integrated:
            maximum_workspace += 512 * 12 * 2560 * 2 + 1280
        if footprint["workspace_bytes"] > maximum_workspace or any(
            footprint[key] != value for key, value in expected_footprint.items()
        ):
            raise AssertionError(f"{name}: unexpected model residency: {footprint}")
        print(json.dumps(dict(check=name, accepted=True, **footprint)), flush=True)

    for adapted in (False, True):
        for constraint, overrides in (
            ("latent_grid", dict(latent_width=46)),
            ("text_prefix", dict(block_tokens=560)),
            ("text_rows", dict(text_tokens=496)),
        ):
            execute(
                f"reject-{constraint}-{'adapted' if adapted else 'base'}",
                "base",
                adapted,
                arguments.sample_results / "base.f32",
                integrated=True,
                overrides=overrides,
            )

    reference_taps = arguments.output / "canonical-taps.bf16"
    reference_taps.write_bytes(
        captured_tensor(
            arguments.reference / "inputs.safetensors",
            "encoder_hidden",
            "BF16",
            [1, 512, 12, 2560],
        )
    )
    cases = (
        ("base", "base", False),
        ("zero", "base", True),
        ("style", "style", True),
    )
    for name, phase, adapted in cases:
        execute(
            "regression-" + name,
            phase,
            adapted,
            arguments.sample_results / (phase + ".f32"),
            integrated=False,
            reference_taps=reference_taps,
        )

    for name, phase, adapted in cases:
        if name != "zero":
            execute(
                phase + "-staged",
                phase,
                adapted,
                arguments.sample_results / (phase + ".f32"),
                integrated=False,
            )
        execute(
            name,
            phase,
            adapted,
            arguments.output / (phase + "-staged.f32"),
            integrated=True,
        )
        if name != "zero":
            image = pixels(arguments.output / (name + ".f32"))
            Image.fromarray(image).save(arguments.output / (name + ".png"))
            for label, previous in (
                (
                    "prior_native_external_encoder",
                    pixels(arguments.sample_results / (phase + ".f32")),
                ),
                (
                    "canonical",
                    np.array(Image.open(arguments.reference / (phase + ".png"))),
                ),
            ):
                error = image.astype(np.float64) - previous.astype(np.float64)
                mse = float(np.square(error).mean())
                print(
                    json.dumps(
                        dict(
                            phase=phase,
                            check=label,
                            pixel_rmse=math.sqrt(mse),
                            psnr_db=10 * math.log10(255**2 / mse) if mse else None,
                        )
                    ),
                    flush=True,
                )
    if (arguments.output / "base.f32").read_bytes() == (
        arguments.output / "style.f32"
    ).read_bytes():
        raise AssertionError("active LoRA does not change the image")
    print(
        "PASS: native token-IDs-to-RGB is exact for base, zero and active LoRA; request construction remains external.",
        flush=True,
    )


if __name__ == "__main__":
    main()
