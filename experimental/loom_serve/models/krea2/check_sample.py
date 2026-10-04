# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify one native text-projection-to-RGB command against native stages.

This consumes the eight-step, 384x384 base/LoRA captures. Text encoding and
fusion remain external. No reference framework is loaded: the native stage
outputs are an exact composition oracle, not an independent numerical oracle.
"""

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def captured_features(path):
    """Read the raw BF16 payload from the pinned single-tensor capture."""
    with path.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        tensor = json.loads(source.read(header_length))["text_fusion"]
        first, last = tensor["data_offsets"]
        if (
            tensor["dtype"] != "BF16"
            or tensor["shape"] != [1, 512, 2560]
            or last - first != 512 * 2560 * 2
        ):
            raise ValueError(f"{path}: capture does not contain 512x2560 BF16 features")
        source.seek(8 + header_length + first)
        data = source.read(last - first)
        if len(data) != last - first:
            raise ValueError(f"{path}: truncated capture")
        return data


def pixels(path):
    value = np.fromfile(path, dtype="<f4").reshape(3, 384, 384).transpose(1, 2, 0)
    return np.rint(np.clip(value / 2 + 0.5, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--denoise_results", type=Path, required=True)
    parser.add_argument("--stack_reference", type=Path, required=True)
    parser.add_argument("--input_results", type=Path, required=True)
    parser.add_argument("--decoder_results", type=Path, required=True)
    parser.add_argument("--text_results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    for phase in ("base", "style"):
        (arguments.output / (phase + "-features.bf16")).write_bytes(
            captured_features(arguments.reference / f"{phase}-text_fusion.safetensors")
        )

    def command(phase, adapted, name, *, expected, stage="image", overrides=None):
        directory = arguments.denoise_results / phase
        weights = [arguments.checkpoint / "turbo.safetensors"]
        if adapted:
            weights.append(arguments.adapter / "softwatercolor.safetensors")
        vae_weights = (
            arguments.checkpoint / "vae" / "diffusion_pytorch_model.safetensors"
        )
        affine = arguments.input_results / f"{phase}-48x48-composed/input-1"
        features = arguments.output / (phase + "-features.bf16")
        projected = arguments.text_results / f"{phase}-512.bf16"
        inputs = [
            (directory / "sample.bf16", 576 * 64 * 2),
            (projected, 512 * 6144 * 2)
            if stage == "denoise"
            else (features, 512 * 2560 * 2),
            (directory / "times.bf16", 8 * 2),
            *[
                (
                    arguments.stack_reference / f"{phase}-block27-{kind}.f32",
                    1088 * 128 * 4,
                )
                for kind in ("cosine", "sine")
            ],
            (arguments.stack_reference / f"{phase}-block27-mask.u8", 1088),
            (directory / "deltas.f32", 8 * 4),
        ]
        if adapted:
            inputs.append((directory / "strength.f32", 4))
        output_type = "f32"
        output_bytes = 3 * 384 * 384 * 4
        if stage == "denoise":
            root = "denoise_adapted" if adapted else "denoise"
            output_type, output_bytes = "bf16", 576 * 64 * 2
        elif stage == "vae":
            root, weights = "vae_decode", [vae_weights]
            inputs = [
                (arguments.output / (phase + "-latent.bf16"), 576 * 64 * 2),
                (affine, 2 * 16 * 4),
            ]
        else:
            root = "sample_image_adapted" if adapted else "sample_image"
            weights.append(vae_weights)
            inputs.append((affine, 2 * 16 * 4))
        for path, size in [*inputs, (expected, output_bytes)]:
            if path.stat().st_size != size:
                raise ValueError(f"{path}: expected {size} bytes for this capture")
        configuration = {
            "block_tokens": 1088,
            "image_tokens": 576,
            "text_tokens": 512,
            "time_count": 8,
            "latent_height": 48,
            "latent_width": 48,
        }
        configuration.update(overrides or {})
        return [
            *arguments.checker,
            f"--model={arguments.model}",
            f"--root={root}",
            f"--weight_policy={arguments.model / 'weights.loom'}",
            *[f"--weights={path}" for path in weights],
            *[f"--config=krea2.{key}={value}" for key, value in configuration.items()],
            *[f"--input={path}" for path, _ in inputs],
            f"--expected={expected}",
            f"--actual={arguments.output / (name + '.' + output_type)}",
            f"--output_type={output_type}",
            *(["--atol=0", "--rtol=0"] if stage == "image" else ["--report_only"]),
        ]

    def execute(name, invocation):
        (arguments.output / (name + ".json")).write_text(
            json.dumps(dict(check=name, command=invocation), indent=2) + "\n"
        )
        print(json.dumps(dict(check=name)), flush=True)
        result = subprocess.run(invocation, capture_output=True, text=True)
        (arguments.output / (name + ".log")).write_text(result.stdout + result.stderr)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        return result

    for adapted in (False, True):
        for constraint, overrides in (
            ("latent_grid", {"latent_width": 46}),
            ("text_prefix", {"block_tokens": 560}),
            ("text_rows", {"text_tokens": 496}),
        ):
            name = f"reject-{constraint}-{'adapted' if adapted else 'base'}"
            result = execute(
                name,
                command(
                    "base",
                    adapted,
                    name,
                    expected=arguments.decoder_results / "base-48x48-native.f32",
                    overrides=overrides,
                ),
            )
            diagnostics = result.stdout + result.stderr
            if result.returncode == 0 or not all(
                marker in diagnostics
                for marker in ("krea2.sample_image", "all_rejected")
            ):
                raise AssertionError(f"{name}: expected geometry-provider rejection")
            if '"workspace_bytes"' in diagnostics:
                raise AssertionError(f"{name}: rejection occurred after JIT selection")
            print(json.dumps(dict(check=name, geometry_rejected=True)), flush=True)

    for name, phase, adapted in (
        ("base", "base", False),
        ("zero", "base", True),
        ("style", "style", True),
    ):
        if name != "zero":
            for stage, suffix, previous in (
                (
                    "denoise",
                    "latent",
                    arguments.denoise_results / phase / "trajectory.bf16",
                ),
                (
                    "vae",
                    "staged",
                    arguments.decoder_results / f"{phase}-48x48-native.f32",
                ),
            ):
                result = execute(
                    phase + "-" + suffix,
                    command(
                        phase,
                        adapted,
                        phase + "-" + suffix,
                        expected=previous,
                        stage=stage,
                    ),
                )
                result.check_returncode()
                records = [
                    json.loads(line)
                    for line in result.stdout.splitlines()
                    if line.startswith("{")
                ]
                if [
                    record["iteration"] for record in records if "iteration" in record
                ] != [0, 1]:
                    raise AssertionError("both staged executions must complete")
        expected = arguments.output / (phase + "-staged.f32")
        result = execute(name, command(phase, adapted, name, expected=expected))
        result.check_returncode()
        records = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        comparisons = [record for record in records if "iteration" in record]
        if [record["iteration"] for record in comparisons] != [0, 1] or any(
            record["different"] for record in comparisons
        ):
            raise AssertionError(f"{name}: both executions must be bitwise exact")
        actual = arguments.output / (name + ".f32")
        if actual.read_bytes() != expected.read_bytes():
            raise AssertionError(f"{name}: output bytes differ from native stages")
        footprint = next(record for record in records if "workspace_bytes" in record)
        expected_counts = dict(
            kernels=(33 if adapted else 20) + 30 + (7 if adapted else 4),
            parameters=(834 if adapted else 376) + 104 + (9 if adapted else 5),
            parameter_roots=3 if adapted else 2,
            parameter_bytes=(25821294848 if adapted else 25382416640)
            + 286100492
            + (216655872 if adapted else 213968896),
        )
        if any(footprint[key] != value for key, value in expected_counts.items()):
            raise AssertionError(f"{name}: shared residency changed: {footprint}")
        # Bound phase-packing fragmentation by one largest VAE feature plane.
        # This is less than keeping both stages' independent scratch alive.
        maximum_workspace = 219709440 + 96 * 384 * 384 * 4 + 576 * 128 + 512 * 6144 * 2
        if footprint["workspace_bytes"] > maximum_workspace:
            raise AssertionError(f"{name}: phase workspace reuse failed: {footprint}")
        print(json.dumps(dict(check=name, accepted=True, **footprint)), flush=True)
        if name != "zero":
            image = pixels(actual)
            Image.fromarray(image).save(arguments.output / (phase + ".png"))
            for label, previous in (
                (
                    "prior_native_external_text",
                    pixels(arguments.decoder_results / f"{phase}-48x48-native.f32"),
                ),
                (
                    "external_reference",
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
        raise AssertionError("active LoRA does not change final RGB")
    print(
        "PASS: native text-projection-to-RGB is exact for base, zero and active LoRA; "
        "invalid geometry is rejected before weight loading. Text encoding/fusion remain external.",
        flush=True,
    )


if __name__ == "__main__":
    main()
