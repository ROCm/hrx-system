# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify one native denoise-to-RGB command against qualified native stages.

This consumes the eight-step, 384x384 base/LoRA captures. Text conditioning
remains external. No reference framework is loaded: the native stage outputs
are an exact composition oracle, not an independent numerical oracle.
"""

import argparse
import json
import subprocess
from pathlib import Path


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
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)

    def command(phase, adapted, name, overrides=None):
        directory = arguments.denoise_results / phase
        weights = [arguments.checkpoint / "turbo.safetensors"]
        if adapted:
            weights.append(arguments.adapter / "softwatercolor.safetensors")
        weights.append(
            arguments.checkpoint / "vae" / "diffusion_pytorch_model.safetensors"
        )
        inputs = [
            (directory / "sample.bf16", 576 * 64 * 2),
            (directory / "text.bf16", 512 * 6144 * 2),
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
        inputs.append(
            (
                arguments.input_results / f"{phase}-48x48-composed/input-1",
                2 * 16 * 4,
            )
        )
        expected = arguments.decoder_results / f"{phase}-48x48-native.f32"
        for path, size in [*inputs, (expected, 3 * 384 * 384 * 4)]:
            if path.stat().st_size != size:
                raise ValueError(f"{path}: expected {size} bytes for this capture")
        configuration = {
            "block_tokens": 1088,
            "image_tokens": 576,
            "time_count": 8,
            "latent_height": 48,
            "latent_width": 48,
        }
        configuration.update(overrides or {})
        return [
            *arguments.checker,
            f"--model={arguments.model}",
            f"--root={'sample_image_adapted' if adapted else 'sample_image'}",
            f"--weight_policy={arguments.model / 'weights.loom'}",
            *[f"--weights={path}" for path in weights],
            *[f"--config=krea2.{key}={value}" for key, value in configuration.items()],
            *[f"--input={path}" for path, _ in inputs],
            f"--expected={expected}",
            f"--actual={arguments.output / (name + '.f32')}",
            "--output_type=f32",
            "--atol=0",
            "--rtol=0",
        ]

    def execute(name, invocation):
        (arguments.output / (name + ".json")).write_text(
            json.dumps(dict(check=name, command=invocation), indent=2) + "\n"
        )
        print(json.dumps(dict(check=name)), flush=True)
        result = subprocess.run(invocation, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", flush=True)
        (arguments.output / (name + ".log")).write_text(result.stdout + result.stderr)
        return result

    for adapted in (False, True):
        for constraint, overrides in (
            ("latent_grid", {"latent_width": 46}),
            ("text_prefix", {"block_tokens": 560}),
        ):
            name = f"reject-{constraint}-{'adapted' if adapted else 'base'}"
            result = execute(name, command("base", adapted, name, overrides))
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
        result = execute(name, command(phase, adapted, name))
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
        expected = arguments.decoder_results / f"{phase}-48x48-native.f32"
        if actual.read_bytes() != expected.read_bytes():
            raise AssertionError(f"{name}: output bytes differ from native stages")
        footprint = next(record for record in records if "workspace_bytes" in record)
        expected_counts = dict(
            kernels=(33 if adapted else 20) + 30,
            parameters=(834 if adapted else 376) + 104,
            parameter_roots=3 if adapted else 2,
            parameter_bytes=(25821294848 if adapted else 25382416640) + 286100492,
        )
        if any(footprint[key] != value for key, value in expected_counts.items()):
            raise AssertionError(f"{name}: shared residency changed: {footprint}")
        # Bound phase-packing fragmentation by one largest VAE feature plane.
        # This is less than keeping both stages' independent scratch alive.
        maximum_workspace = 219709440 + 96 * 384 * 384 * 4 + 576 * 128
        if footprint["workspace_bytes"] > maximum_workspace:
            raise AssertionError(f"{name}: phase workspace reuse failed: {footprint}")
        print(json.dumps(dict(check=name, accepted=True, **footprint)), flush=True)

    if (arguments.output / "base.f32").read_bytes() == (
        arguments.output / "style.f32"
    ).read_bytes():
        raise AssertionError("active LoRA does not change final RGB")
    print(
        "PASS: native denoise-to-RGB is exact for base, zero and active LoRA; "
        "invalid geometry is rejected before weight loading. Text remains external.",
        flush=True,
    )


if __name__ == "__main__":
    main()
