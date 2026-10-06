# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify the native velocity head on actual external and native stack states.

Independent F64 reductions retain BF16 tensor boundaries. Exact composition
uses separately qualified native primitives, not an expanded error tolerance.
The external transformer supplies the captured step's time embedding only;
native time conditioning is a separate production boundary.
"""

import argparse
import json
import pathlib
import subprocess

import numpy as np
import torch
from diffusers import Krea2Transformer2DModel
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--base_stack", type=pathlib.Path, required=True)
parser.add_argument("--style_stack", type=pathlib.Path, required=True)
parser.add_argument("--image_rows", type=int, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
if args.image_rows < 16 or args.image_rows % 16:
    parser.error("--image_rows must describe complete 16-row image tiles")
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_grad_enabled(False)


def load(path):
    return torch.from_numpy(np.fromfile(path, dtype="<u2")).view(torch.bfloat16)


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite head tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def report(phase, source, rows, check, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite head comparison")
    difference = actual.double() - expected.double()
    print(
        json.dumps(
            dict(
                phase=phase,
                source=source,
                rows=rows,
                check=check,
                different=int((difference != 0).sum()),
                relative_l2=float(
                    difference.norm() / expected.double().norm().clamp_min(1e-30)
                ),
                maximum_absolute_error=float(difference.abs().max()),
            )
        ),
        flush=True,
    )


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
base_checkpoint = args.checkpoint / "turbo.safetensors"
adapter_checkpoint = args.adapter / "softwatercolor.safetensors"
with (
    safe_open(str(base_checkpoint), framework="pt") as weights,
    safe_open(str(adapter_checkpoint), framework="pt") as adapter,
):
    scale = weights.get_tensor("last.norm.scale").bfloat16().add(1).double()
    table = weights.get_tensor("last.modulation.lin").bfloat16().reshape(2, 6144)
    matrix = weights.get_tensor("last.linear.weight").bfloat16().double()
    bias = weights.get_tensor("last.linear.bias").bfloat16().double()
    down = (
        adapter.get_tensor("transformer.final_layer.linear.lora_A.weight")
        .bfloat16()
        .double()
    )
    up = (
        adapter.get_tensor("transformer.final_layer.linear.lora_B.weight")
        .bfloat16()
        .double()
    )

    def normalize(value):
        value = value.double()
        inverse = (value.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        return (value * inverse * scale).bfloat16()

    for phase, stack_directory in (
        ("base", args.base_stack),
        ("style", args.style_stack),
    ):
        adapted = phase == "style"
        if adapted:
            transformer.load_lora_adapter(
                str(args.adapter),
                weight_name="softwatercolor.safetensors",
                local_files_only=True,
                adapter_name="watercolor",
            )
            transformer.set_adapters("watercolor", weights=1.0)
        embedding_device = transformer.time_embed(
            torch.ones(1, device="cuda", dtype=torch.bfloat16), dtype=torch.bfloat16
        )
        modulation = transformer.time_mod_proj(
            torch.nn.functional.gelu(embedding_device, approximate="tanh")
        ).cpu()
        captured_modulation = load(
            args.reference / f"{phase}-block27-modulation.bf16"
        ).reshape_as(modulation)
        if encode(modulation) != encode(captured_modulation):
            raise AssertionError("reconstructed first-step conditioning differs")
        embedding = embedding_device.cpu().reshape(6144)
        coefficients = embedding + table

        def affine(value):
            return value * (coefficients[0] + 1) + coefficients[1]

        def project(value):
            result = (value.double() @ matrix.T + bias).bfloat16()
            if adapted:
                low = (value.double() @ down.T).bfloat16()
                result = result + (low.double() @ up.T).bfloat16()
            return result

        external_state = load(args.reference / f"{phase}-block27-output.bf16").reshape(
            -1, 6144
        )[-args.image_rows :]
        native_state = load(stack_directory / "stack.bf16").reshape(-1, 6144)[
            -args.image_rows :
        ]
        oracle_state = load(stack_directory / "oracle.bf16").reshape(-1, 6144)[
            -args.image_rows :
        ]
        if any(
            state.shape[0] != args.image_rows
            for state in (external_state, native_state, oracle_state)
        ):
            raise ValueError("stack captures do not contain the requested image suffix")
        oracle_velocity = project(affine(normalize(oracle_state)))
        external_velocity = (
            transformer.final_layer(external_state[None].cuda(), embedding_device)
            .cpu()
            .squeeze(0)
        )
        report(
            phase,
            "external",
            args.image_rows,
            "velocity_vs_f64_stack",
            external_velocity,
            oracle_velocity,
        )

        for source, state in (("external", external_state), ("native", native_state)):
            for count in dict.fromkeys((16, args.image_rows)):
                directory = args.output / f"{phase}-{source}-{count}"
                directory.mkdir()
                selected = state[-count:]
                paths = {}
                for name, value in (("input", selected), ("embedding", embedding)):
                    paths[name] = directory / f"{name}.bf16"
                    paths[name].write_bytes(encode(value))
                paths["strength"] = directory / "strength.f32"
                paths["strength"].write_bytes(
                    np.array([float(adapted)], dtype="<f4").tobytes()
                )
                paths["zero"] = directory / "zero.f32"
                paths["zero"].write_bytes(np.array([0.0], dtype="<f4").tobytes())

                def execute(name, root, expected, inputs, checkpoints, *, exact=False):
                    expected_path = directory / "expected.bf16"
                    actual_path = directory / f"{name}.bf16"
                    expected_path.write_bytes(encode(expected))
                    command = [
                        *args.checker,
                        "--model="
                        + str(
                            args.model / "qualification"
                            if root.startswith("qualify.")
                            else args.model
                        ),
                        "--weight_policy=" + str(args.model / "weights.loom"),
                        *["--weights=" + str(path) for path in checkpoints],
                        "--root=" + root,
                        f"--config=krea2.image_tokens={count}",
                        *["--input=" + str(paths[key]) for key in inputs],
                        "--expected=" + str(expected_path),
                        "--actual=" + str(actual_path),
                    ]
                    if exact:
                        command += ["--atol=0", "--rtol=0"]
                    print(
                        json.dumps(
                            dict(phase=phase, source=source, rows=count, component=name)
                        ),
                        flush=True,
                    )
                    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
                    print(result.stdout, end="", flush=True)
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
                        raise AssertionError("checker did not complete both executions")
                    if exact and any(record["different"] for record in comparisons):
                        raise AssertionError("exact component comparison failed")
                    if exact and actual_path.read_bytes() != encode(expected):
                        raise AssertionError("non-bitwise head composition")
                    paths[name] = actual_path
                    footprint = next(
                        record for record in records if "workspace_bytes" in record
                    )
                    return load(actual_path).reshape_as(expected), footprint

                normalized, _ = execute(
                    "normalized",
                    "qualify.head_normalize",
                    normalize(selected),
                    ["input"],
                    [base_checkpoint],
                )
                conditioned, _ = execute(
                    "conditioned",
                    "qualify.head_input",
                    affine(normalized),
                    ["input", "embedding"],
                    [base_checkpoint],
                    exact=True,
                )
                projected, _ = execute(
                    "projected",
                    "krea2.head_projection",
                    (conditioned.double() @ matrix.T + bias).bfloat16(),
                    ["conditioned"],
                    [base_checkpoint],
                )
                base, base_footprint = execute(
                    "base",
                    "velocity_head",
                    projected,
                    ["input", "embedding"],
                    [base_checkpoint],
                    exact=True,
                )
                low, _ = execute(
                    "low",
                    "krea2.head_adapter_down",
                    (conditioned.double() @ down.T).bfloat16(),
                    ["conditioned"],
                    [adapter_checkpoint],
                )
                delta, _ = execute(
                    "delta",
                    "qualify.head_adapter_up",
                    (low.double() @ up.T).bfloat16(),
                    ["low"],
                    [adapter_checkpoint],
                )
                combined = base + delta if adapted else base
                combined, _ = execute(
                    "combined",
                    "krea2.head_adapter_add",
                    combined,
                    ["low", "strength", "base"],
                    [adapter_checkpoint],
                    exact=True,
                )
                actual, footprint = execute(
                    "velocity",
                    "velocity_head_adapted",
                    combined,
                    ["input", "embedding", "strength"],
                    [base_checkpoint, adapter_checkpoint],
                    exact=True,
                )
                execute(
                    "zero_velocity",
                    "velocity_head_adapted",
                    base,
                    ["input", "embedding", "zero"],
                    [base_checkpoint, adapter_checkpoint],
                    exact=True,
                )
                expected_base = dict(
                    kernels=2,
                    parameters=4,
                    parameter_roots=1,
                    workspace_bytes=count * 6144 * 2,
                )
                expected_adapted = dict(
                    kernels=4,
                    parameters=6,
                    parameter_roots=2,
                    workspace_bytes=count * (6144 + 32) * 2,
                )
                for observed, expected in (
                    (base_footprint, expected_base),
                    (footprint, expected_adapted),
                ):
                    if any(observed[key] != value for key, value in expected.items()):
                        raise AssertionError(f"unexpected head footprint: {observed}")
                independent = project(affine(normalize(selected)))
                report(phase, source, count, "head_vs_local_f64", actual, independent)
                if count == args.image_rows:
                    report(
                        phase,
                        source,
                        count,
                        "velocity_vs_f64_stack",
                        actual,
                        oracle_velocity,
                    )
                print(
                    json.dumps(
                        dict(
                            phase=phase,
                            source=source,
                            rows=count,
                            check="head_accepted",
                            **footprint,
                        )
                    ),
                    flush=True,
                )

print(
    "PASS: velocity heads meet primitive, exact-fusion, composition and zero-identity gates.",
    flush=True,
)
