# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify queued Krea stacks against native composition and F64 arithmetic.

Python submits individual blocks only to establish an independent composition
oracle. The production stack is one source-JIT command, with no host traversal.
Scratch files are overwritten per layer; only final tensors are retained.
"""

import argparse
import json
import pathlib
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from block_reference import evaluate_block
from diffusers import Krea2Transformer2DModel
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--rows", type=int, nargs="+", default=[16, 1088])
parser.add_argument("--phase", choices=("base", "style", "both"), default="both")
parser.add_argument("--image_rows", type=int, required=True)
args = parser.parse_args()
if args.image_rows <= 0:
    parser.error("--image_rows must describe the positive final-head input length")
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_grad_enabled(False)


def load(path, dtype="<u2"):
    tensor = torch.from_numpy(np.fromfile(path, dtype=dtype))
    if dtype == "<u2":
        tensor = tensor.view(torch.bfloat16)
    elif dtype == "u1":
        tensor = tensor.bool()
    return tensor


def save(path, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"nonfinite tensor: {path.name}")
    path.write_bytes(
        value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()
    )


def report(phase, rows, layer, name, actual, expected, *, domain="all"):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"nonfinite {name}")
    error = actual.double() - expected.double()
    relative_l2 = float(error.norm() / expected.double().norm().clamp_min(1e-30))
    print(
        json.dumps(
            dict(
                phase=phase,
                rows=rows,
                layer=layer,
                check=name,
                domain=domain,
                relative_l2=relative_l2,
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )
    return relative_l2


def execute(
    root,
    count,
    inputs,
    expected,
    actual,
    *,
    adapted=False,
    layer=None,
    comparison="exact",
):
    command = [
        *args.checker,
        "--model="
        + str(args.model / "qualification" if layer is not None else args.model),
        "--weight_policy=" + str(args.model / "weights.loom"),
        "--weights=" + str(args.checkpoint / "turbo.safetensors"),
        "--root=" + root,
        f"--config=krea2.block_tokens={count}",
        *["--input=" + str(path) for path in inputs],
        "--expected=" + str(expected),
        "--actual=" + str(actual),
    ]
    if adapted:
        command.append("--weights=" + str(args.adapter / "softwatercolor.safetensors"))
    if layer is not None:
        command.append(f"--config=krea2.block_index={layer}")
    if comparison == "report":
        command.append("--report_only")
    else:
        command += ["--atol=0", "--rtol=0"]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    comparisons = [record for record in records if "iteration" in record]
    if [record["iteration"] for record in comparisons] != [0, 1]:
        raise AssertionError("checker did not complete both executions")
    if comparison == "exact" and any(record["different"] for record in comparisons):
        raise AssertionError("exact composition changed output bits")
    footprint = next(record for record in records if "workspace_bytes" in record)
    expected_kernels = 17 if adapted else 11
    if footprint["kernels"] != expected_kernels:
        raise AssertionError(f"expected {expected_kernels} cached kernels: {footprint}")
    if footprint["parameter_roots"] != (2 if adapted else 1):
        raise AssertionError(f"checkpoint roots do not match the model: {footprint}")
    return footprint


transformer = (
    Krea2Transformer2DModel.from_single_file(
        str(args.checkpoint / "turbo.safetensors"),
        config=str(args.checkpoint / "transformer"),
        torch_dtype=torch.bfloat16,
        device="cuda",
        local_files_only=True,
    )
    .eval()
    .requires_grad_(False)
)
if len(transformer.transformer_blocks) != 28:
    raise ValueError("qualification requires the pinned 28-layer transformer")

for phase in ("base", "style"):
    if args.phase not in (phase, "both"):
        continue
    adapted = phase == "style"
    if adapted:
        transformer.load_lora_adapter(
            str(args.adapter),
            weight_name="softwatercolor.safetensors",
            local_files_only=True,
            adapter_name="watercolor",
        )
        transformer.set_adapters("watercolor", weights=1.0)
    with (
        safe_open(
            str(args.checkpoint / "turbo.safetensors"), framework="pt"
        ) as weights,
        safe_open(str(args.adapter / "softwatercolor.safetensors"), framework="pt")
        if adapted
        else nullcontext() as adapter,
    ):
        for count in args.rows:
            image_rows = min(count, args.image_rows)
            reference = args.reference / str(count)
            prefix = f"{phase}-block27-"
            input_path = reference / f"{phase}-stack-input.bf16"
            initial = load(input_path).reshape(count, 6144)
            modulation_path = reference / (prefix + "modulation.bf16")
            cosine_path = reference / (prefix + "cosine.f32")
            sine_path = reference / (prefix + "sine.f32")
            mask_path = reference / (prefix + "mask.u8")
            modulation = load(modulation_path).reshape(6, 6144)
            cosine = load(cosine_path, "<f4").reshape(count, 128)
            sine = load(sine_path, "<f4").reshape(count, 128)
            mask = load(mask_path, "u1")
            directory = args.output / f"{phase}-{count}"
            directory.mkdir()
            strength_path = directory / "strength.f32"
            strength_path.write_bytes(np.array([float(adapted)], dtype="<f4").tobytes())
            shared_inputs = [modulation_path, cosine_path, sine_path, mask_path]
            if adapted:
                shared_inputs.append(strength_path)
            native_path = directory / "input.bf16"
            expected_path = directory / "expected.bf16"
            actual_path = directory / "actual.bf16"
            native, oracle = initial, initial
            external_chain = initial[None].cuda()
            external_modulation = modulation.reshape(1, 1, 6 * 6144).cuda()
            external_rotary = (cosine.cuda(), sine.cuda())
            external_mask = mask.reshape(1, 1, 1, count).cuda()
            block_footprint = None
            for layer, block in enumerate(transformer.transformer_blocks):
                print(
                    json.dumps(dict(phase=phase, rows=count, layer=layer, stage="f64")),
                    flush=True,
                )
                arguments = (
                    modulation,
                    cosine,
                    sine,
                    mask,
                    weights,
                    adapter,
                    layer,
                    float(adapted),
                )
                local_oracle = evaluate_block(native, *arguments)
                oracle = (
                    local_oracle if layer == 0 else evaluate_block(oracle, *arguments)
                )
                external = (
                    block(
                        native[None].cuda(),
                        external_modulation,
                        external_rotary,
                        external_mask,
                    )
                    .cpu()
                    .reshape(count, 6144)
                )
                report(
                    phase, count, layer, "external_local_vs_f64", external, local_oracle
                )
                save(native_path, native)
                save(expected_path, local_oracle)
                footprint = execute(
                    "qualify.block_forward_adapted"
                    if adapted
                    else "qualify.block_forward",
                    count,
                    [native_path, *shared_inputs],
                    expected_path,
                    actual_path,
                    adapted=adapted,
                    layer=layer,
                    comparison="report",
                )
                if block_footprint is None:
                    block_footprint = footprint
                elif footprint != block_footprint:
                    raise AssertionError("layer identity changed the block footprint")
                native = load(actual_path).reshape(count, 6144)
                report(phase, count, layer, "native_local_vs_f64", native, local_oracle)
                external_chain = block(
                    external_chain, external_modulation, external_rotary, external_mask
                )
                report(phase, count, layer + 1, "native_prefix_vs_f64", native, oracle)
                report(
                    phase,
                    count,
                    layer + 1,
                    "external_prefix_vs_f64",
                    external_chain.cpu().reshape(count, 6144),
                    oracle,
                )
                report(
                    phase,
                    count,
                    layer + 1,
                    "native_prefix_vs_f64",
                    native[-image_rows:],
                    oracle[-image_rows:],
                    domain="image",
                )
                report(
                    phase,
                    count,
                    layer + 1,
                    "external_prefix_vs_f64",
                    external_chain[0, -image_rows:].cpu(),
                    oracle[-image_rows:],
                    domain="image",
                )

            oracle_path = directory / "oracle.bf16"
            save(oracle_path, oracle)
            report(phase, count, 28, "native_stack_vs_f64", native, oracle)
            external = load(reference / (prefix + "output.bf16")).reshape(count, 6144)
            if not torch.equal(
                external_chain.cpu().reshape_as(external).view(torch.uint16),
                external.view(torch.uint16),
            ):
                raise AssertionError("external stack changed from the pinned capture")
            report(phase, count, 28, "external_stack_vs_f64", external, oracle)
            # The model discards text rows before its final head. Keep all-row
            # diagnostics, but qualify the values the real consumer receives.
            native_error = report(
                phase,
                count,
                28,
                "native_stack_vs_f64",
                native[-image_rows:],
                oracle[-image_rows:],
                domain="image",
            )
            external_error = report(
                phase,
                count,
                28,
                "external_stack_vs_f64",
                external[-image_rows:],
                oracle[-image_rows:],
                domain="image",
            )
            stack_path = directory / "stack.bf16"
            footprint = execute(
                "transformer_stack_adapted" if adapted else "transformer_stack",
                count,
                [input_path, *shared_inputs],
                actual_path,
                stack_path,
                adapted=adapted,
            )
            if stack_path.read_bytes() != actual_path.read_bytes():
                raise AssertionError("queued stack differs from separate native blocks")
            if footprint["workspace_bytes"] != block_footprint["workspace_bytes"]:
                raise AssertionError(
                    "stack workspace exceeds the single-block workspace"
                )
            for field in ("parameter_bytes", "parameters"):
                if footprint[field] != 28 * block_footprint[field]:
                    raise AssertionError(
                        f"stack does not have 28 unique layers of {field}"
                    )
            if not adapted:
                execute(
                    "transformer_stack_adapted",
                    count,
                    [input_path, *shared_inputs, strength_path],
                    stack_path,
                    directory / "zero.bf16",
                    adapted=True,
                )
                if (directory / "zero.bf16").read_bytes() != stack_path.read_bytes():
                    raise AssertionError("zero-strength stack differs from base")
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        rows=count,
                        check="stack_composition_exact",
                        **footprint,
                    )
                ),
                flush=True,
            )
            if native_error > external_error:
                raise AssertionError(
                    "native image rows exceed external accumulated error"
                )
            print(
                json.dumps(dict(phase=phase, rows=count, check="stack_accepted")),
                flush=True,
            )

print(
    "PASS: queued stacks equal native composition and meet independent F64 accuracy gates.",
    flush=True,
)
