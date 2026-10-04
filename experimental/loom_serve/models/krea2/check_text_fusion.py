# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify the learned layer mixer and complete native text fusion.

F64 projector primitives and independently materialized native A/B results
establish the register-only LoRA fusion. The complete command then must match
separately executed native stages bit-for-bit, including in-place block
advancement and zero strength. Aggregate chain errors remain diagnostics.

The overwritten working fixture and final features retain less than 384 MiB.
Only qualification materializes the 80 MiB rank tensor at 512 tokens.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from check_text_fusion_blocks import encode, load_bf16, report
from safetensors import safe_open
from safetensors.torch import load_file
from text_fusion_reference import evaluate_fusion_block


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
    capture = load_file(str(arguments.reference / "inputs.safetensors"))
    taps = capture["encoder_hidden"]
    mask = capture["encoder_mask"].reshape(-1).bool()
    if taps.shape != (1, 512, 12, 2560) or taps.dtype != torch.bfloat16:
        raise ValueError("qualification requires the original BF16 encoder taps")
    if mask.numel() != 512 or not mask[:16].any():
        raise ValueError("qualification requires live keys at both token extents")
    checkpoints = {
        "base": arguments.checkpoint / "turbo.safetensors",
        "adapter": arguments.adapter / "softwatercolor.safetensors",
    }
    with (
        safe_open(str(checkpoints["base"]), framework="pt") as weights,
        safe_open(str(checkpoints["adapter"]), framework="pt") as adapter,
    ):
        coefficient = (
            weights.get_tensor("txtfusion.projector.weight").bfloat16().double()
        )
        adapter_prefix = "transformer.text_fusion.projector.lora_"
        down = adapter.get_tensor(adapter_prefix + "A.weight").bfloat16().double()
        up = adapter.get_tensor(adapter_prefix + "B.weight").bfloat16().double()

        def project(value, adapted):
            transposed = value.reshape(-1, 12, 2560).transpose(1, 2).double()
            base = (transposed @ coefficient.T).squeeze(-1).bfloat16()
            if not adapted:
                return base
            low = (transposed @ down.T).bfloat16()
            delta = (low.double() @ up.T).squeeze(-1).bfloat16()
            return base + delta

        for count in (16, 512):
            for phase in ("base", "style"):
                adapted = phase == "style"
                context = dict(tokens=count, phase=phase)
                domains = ("base", "adapter") if adapted else ("base",)
                strength = torch.tensor([float(adapted)], dtype=torch.float32)
                selected_mask = mask[:count].contiguous()

                def execute(
                    root,
                    expected,
                    operands,
                    *,
                    parameters=("base",),
                    layer=None,
                    comparison="elementwise",
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
                    invocation = [
                        *arguments.checker,
                        f"--model={arguments.model / 'qualification'}",
                        f"--weight_policy={arguments.model / 'weights.loom'}",
                        *[f"--weights={checkpoints[key]}" for key in parameters],
                        f"--root={root}",
                        f"--config=krea2.text_tokens={count}",
                        *(
                            [f"--config=qualify.fusion_layer={layer}"]
                            if layer is not None
                            else []
                        ),
                        *[f"--input={path}" for path in paths],
                        f"--expected={expected_path}",
                        f"--actual={actual_path}",
                    ]
                    if comparison == "exact":
                        invocation.extend(["--atol=0", "--rtol=0"])
                    elif comparison == "report":
                        invocation.append("--report_only=true")
                    elif comparison != "elementwise":
                        raise ValueError(f"unknown comparison: {comparison}")
                    (working / "check.json").write_text(
                        json.dumps(
                            dict(**context, root=root, command=invocation), indent=2
                        )
                        + "\n"
                    )
                    print(
                        json.dumps(dict(**context, root=root, comparison=comparison)),
                        flush=True,
                    )
                    result = subprocess.run(invocation, capture_output=True, text=True)
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
                    if [record["iteration"] for record in comparisons] != [0, 1]:
                        raise AssertionError("both native executions must complete")
                    if comparison == "exact" and (
                        any(record["different"] for record in comparisons)
                        or actual_path.read_bytes() != expected_bytes
                    ):
                        raise AssertionError(f"{root}: composition differs bitwise")
                    footprint = next(
                        record for record in records if "workspace_bytes" in record
                    )
                    return load_bf16(actual_path, expected.shape), footprint

                def qualify_projector(value, scales):
                    transposed = value.reshape(count, 12, 2560).transpose(1, 2).double()
                    base, _ = execute(
                        "qualify.fusion_projector",
                        (transposed @ coefficient.T).squeeze(-1).bfloat16(),
                        [value],
                    )
                    if not scales:
                        return base
                    low, _ = execute(
                        "qualify.fusion_projector_down",
                        (transposed @ down.T).bfloat16(),
                        [value],
                        parameters=("adapter",),
                    )
                    delta, _ = execute(
                        "qualify.fusion_projector_up",
                        (low.double() @ up.T).squeeze(-1).bfloat16(),
                        [low],
                        parameters=("adapter",),
                    )
                    del low, transposed
                    result = base
                    for scale in scales:
                        result, _ = execute(
                            "qualify.fusion_projector_adapted",
                            base if scale == 0 else base + delta * scale,
                            [value, torch.tensor([scale], dtype=torch.float32)],
                            parameters=("base", "adapter"),
                            comparison="exact",
                        )
                    return result

                canonical = load_bf16(
                    arguments.fusion_reference / f"{phase}-layerwise1.bf16",
                    (512, 12, 2560),
                )[:count].contiguous()
                qualify_projector(canonical, (0.0, 0.5, 1.0))
                del canonical

                source = taps[0, :count].reshape(count * 12, 2560).contiguous()
                native = source
                oracle = source
                for axis in ("layerwise", "refiner"):
                    if axis == "refiner":
                        native = qualify_projector(native, (1.0,) if adapted else ())
                        oracle = project(oracle, adapted)
                    for layer in (0, 1):
                        expected = evaluate_fusion_block(
                            native,
                            selected_mask,
                            weights,
                            adapter if adapted else None,
                            axis,
                            layer,
                            float(adapted),
                        )
                        block_inputs = [native]
                        if axis == "refiner":
                            block_inputs.append(selected_mask)
                        if adapted:
                            block_inputs.append(strength)
                        native, _ = execute(
                            f"qualify.fusion_{axis}_block"
                            + ("_adapted" if adapted else ""),
                            expected,
                            block_inputs,
                            parameters=domains,
                            layer=layer,
                            comparison="report",
                        )
                        oracle = evaluate_fusion_block(
                            oracle,
                            selected_mask,
                            weights,
                            adapter if adapted else None,
                            axis,
                            layer,
                            float(adapted),
                        )
                full_inputs = [source, selected_mask]
                if adapted:
                    full_inputs.append(strength)
                whole, footprint = execute(
                    "text_fusion_adapted" if adapted else "text_fusion",
                    native,
                    full_inputs,
                    parameters=domains,
                    comparison="exact",
                )
                if not adapted:
                    execute(
                        "text_fusion_adapted",
                        whole,
                        [
                            source,
                            selected_mask,
                            torch.tensor([0.0], dtype=torch.float32),
                        ],
                        parameters=("base", "adapter"),
                        comparison="exact",
                    )
                if footprint["parameters"] != (115 if adapted else 49):
                    raise AssertionError(
                        f"full fusion parameter residency changed: {footprint}"
                    )
                (arguments.output / f"{phase}-{count}.bf16").write_bytes(encode(whole))
                report(context, "native_fusion_vs_f64", whole, oracle)
                if count == 512:
                    external = load_bf16(
                        arguments.fusion_reference / f"{phase}-refiner1.bf16",
                        (512, 2560),
                    )
                    report(context, "native_fusion_vs_external", whole, external)
                print(
                    json.dumps(dict(**context, accepted=True, **footprint)), flush=True
                )
    print(
        "PASS: layer projector and complete native fusion, base/zero/LoRA.", flush=True
    )


if __name__ == "__main__":
    main()
