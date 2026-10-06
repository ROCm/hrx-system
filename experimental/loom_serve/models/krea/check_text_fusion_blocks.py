# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native fusion blocks on real encoder taps and canonical intermediates.

F64 primitives establish numerical acceptance; the complete native block must
then match the independently executed native stages bit-for-bit, twice. Base,
zero-strength and active LoRA exercise both layer-axis and masked token-axis
blocks at small and full extents. This is correctness evidence, not timing.

One overwritten fixture holds at most five operands, expected and actual.
Together with retained final block outputs this uses less than 1 GiB on disk
at the qualified 16/512-token shapes. Intermediate tensors remain in RAM.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from text_fusion_reference import evaluate_fusion_block


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite text-fusion tensor")
    value = value.contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().astype("<u2").tobytes()
    if value.dtype == torch.float32:
        return value.numpy().astype("<f4").tobytes()
    if value.dtype == torch.bool:
        return value.numpy().astype("u1").tobytes()
    raise ValueError(f"unsupported fusion fixture dtype: {value.dtype}")


def load_bf16(path, shape):
    return (
        torch.from_numpy(np.fromfile(path, dtype="<u2"))
        .view(torch.bfloat16)
        .reshape(shape)
    )


def report(context, name, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite fusion comparison")
    error = actual.double() - expected.double()
    print(
        json.dumps(
            dict(
                **context,
                check=name,
                relative_l2=float(
                    error.norm() / expected.double().norm().clamp_min(1e-30)
                ),
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fusion_reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    working = arguments.output / "working"
    working.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.set_grad_enabled(False)
    captures = load_file(str(arguments.reference / "inputs.safetensors"))
    taps = captures["encoder_hidden"]
    mask = captures["encoder_mask"].reshape(-1).bool()
    if (
        taps.shape != (1, 512, 12, 2560)
        or taps.dtype != torch.bfloat16
        or mask.numel() != 512
    ):
        raise ValueError(
            "qualification requires the original 512-token encoder capture"
        )
    checkpoints = {
        "base": arguments.checkpoint / "turbo.safetensors",
        "adapter": arguments.adapter / "softwatercolor.safetensors",
    }
    with (
        safe_open(str(checkpoints["base"]), framework="pt") as weights,
        safe_open(str(checkpoints["adapter"]), framework="pt") as adapter,
    ):
        for count in (16, 512):
            for axis in ("layerwise", "refiner"):
                rows = count * 12 if axis == "layerwise" else count
                for layer in (0, 1):
                    for phase in ("base", "style"):
                        adapted = phase == "style"
                        context = dict(
                            tokens=count, rows=rows, axis=axis, layer=layer, phase=phase
                        )
                        if axis == "layerwise" and layer == 0:
                            source = taps[0, :count].reshape(rows, 2560).contiguous()
                        else:
                            previous = (
                                "layerwise0"
                                if axis == "layerwise"
                                else "projected"
                                if layer == 0
                                else "refiner0"
                            )
                            full_rows = 512 * 12 if axis == "layerwise" else 512
                            source = load_bf16(
                                arguments.fusion_reference / f"{phase}-{previous}.bf16",
                                (full_rows, 2560),
                            )[:rows].contiguous()
                        selected_mask = mask[:count].contiguous()
                        if not selected_mask.any():
                            raise ValueError(
                                "refiner witness requires at least one live key"
                            )

                        def execute(
                            name,
                            expected,
                            operands,
                            *,
                            parameters=("base",),
                            exact=False,
                        ):
                            paths = []
                            for index, value in enumerate(operands):
                                path = working / f"input-{index}"
                                path.write_bytes(encode(value))
                                paths.append(path)
                            expected_path = working / "expected.bf16"
                            actual_path = working / "actual.bf16"
                            expected_bytes = encode(expected)
                            expected_path.write_bytes(expected_bytes)
                            prefix = "qualify." if parameters else "krea2."
                            invocation = [
                                *arguments.checker,
                                f"--model={arguments.model / 'qualification'}",
                                f"--weight_policy={arguments.model / 'weights.loom'}",
                                *[
                                    f"--weights={checkpoints[key]}"
                                    for key in parameters
                                ],
                                f"--root={prefix}fusion_{axis}_{name}",
                                f"--config=krea2.text_tokens={count}",
                                f"--config=qualify.fusion_layer={layer}",
                                *[f"--input={path}" for path in paths],
                                f"--expected={expected_path}",
                                f"--actual={actual_path}",
                                *(["--atol=0", "--rtol=0"] if exact else []),
                            ]
                            (working / "check.json").write_text(
                                json.dumps(
                                    dict(**context, check=name, command=invocation),
                                    indent=2,
                                )
                                + "\n"
                            )
                            print(json.dumps(dict(**context, check=name)), flush=True)
                            result = subprocess.run(
                                invocation, capture_output=True, text=True
                            )
                            print(result.stdout, end="", flush=True)
                            print(result.stderr, end="", flush=True)
                            result.check_returncode()
                            records = [
                                json.loads(line)
                                for line in result.stdout.splitlines()
                                if line.startswith("{")
                            ]
                            comparisons = [
                                record for record in records if "iteration" in record
                            ]
                            if [record["iteration"] for record in comparisons] != [
                                0,
                                1,
                            ]:
                                raise AssertionError(
                                    "both native executions must complete"
                                )
                            if exact and (
                                any(record["different"] for record in comparisons)
                                or actual_path.read_bytes() != expected_bytes
                            ):
                                raise AssertionError(
                                    f"{name}: native composition is not bitwise"
                                )
                            footprint = next(
                                record
                                for record in records
                                if "workspace_bytes" in record
                            )
                            return load_bf16(actual_path, expected.shape), footprint

                        def observe(name, expected, operands, **metadata):
                            return execute(name, expected, operands, **metadata)[0]

                        operands = (
                            source,
                            selected_mask,
                            weights,
                            adapter if adapted else None,
                            axis,
                            layer,
                            float(adapted),
                        )
                        oracle = evaluate_fusion_block(*operands)
                        native = evaluate_fusion_block(*operands, observe=observe)
                        whole_inputs = (
                            [source, selected_mask] if axis == "refiner" else [source]
                        )
                        if adapted:
                            whole_inputs.append(
                                torch.tensor([1.0], dtype=torch.float32)
                            )
                        whole, footprint = execute(
                            "block_adapted" if adapted else "block",
                            native,
                            whole_inputs,
                            parameters=("base", "adapter") if adapted else ("base",),
                            exact=True,
                        )
                        if not adapted:
                            execute(
                                "block_adapted",
                                whole,
                                [
                                    *whole_inputs,
                                    torch.tensor([0.0], dtype=torch.float32),
                                ],
                                parameters=("base", "adapter"),
                                exact=True,
                            )
                        if footprint["parameters"] != (28 if adapted else 12):
                            raise AssertionError(
                                f"fusion block parameter residency changed: {footprint}"
                            )
                        (
                            arguments.output / f"{phase}-{axis}{layer}-{count}.bf16"
                        ).write_bytes(encode(whole))
                        report(context, "native_block_vs_f64", whole, oracle)
                        if count == 512 or axis == "layerwise":
                            full_rows = 512 * 12 if axis == "layerwise" else 512
                            external = load_bf16(
                                arguments.fusion_reference
                                / f"{phase}-{axis}{layer}.bf16",
                                (full_rows, 2560),
                            )[:rows]
                            report(context, "native_block_vs_external", whole, external)
                        print(
                            json.dumps(dict(**context, accepted=True, **footprint)),
                            flush=True,
                        )
    print(
        "PASS: native fusion blocks, base/zero/LoRA and exact composition.", flush=True
    )


if __name__ == "__main__":
    main()
