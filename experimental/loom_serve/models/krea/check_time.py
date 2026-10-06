# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualify native batched time conditioning and its packed output ownership."""

import argparse
import json
import math
import pathlib
import subprocess

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--checker", nargs="+", required=True)
parser.add_argument("--model", type=pathlib.Path, required=True)
parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
parser.add_argument("--adapter", type=pathlib.Path, required=True)
parser.add_argument("--reference", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 16, 17])
args = parser.parse_args()
if any(count < 1 or count > 17 for count in args.counts):
    parser.error("qualification supplies between 1 and 17 actual/boundary times")
args.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.set_grad_enabled(False)


def load(path):
    return torch.from_numpy(np.fromfile(path, dtype="<u2")).view(torch.bfloat16)


def encode(value):
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite conditioning tensor")
    return value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()


def gelu(value):
    value = value.double()
    argument = math.sqrt(2 / math.pi) * (value + 0.044715 * value.pow(3))
    return (0.5 * value * (1 + argument.tanh())).bfloat16()


def report(phase, count, name, actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite conditioning comparison")
    error = actual.double() - expected.double()
    print(
        json.dumps(
            dict(
                phase=phase,
                count=count,
                check=name,
                different=int((error != 0).sum()),
                relative_l2=float(
                    error.norm() / expected.double().norm().clamp_min(1e-30)
                ),
                maximum_absolute_error=float(error.abs().max()),
            )
        ),
        flush=True,
    )


scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    args.checkpoint / "scheduler", local_files_only=True
)
scheduler.set_timesteps(8, device="cpu", sigmas=np.linspace(1.0, 1 / 8, 8), mu=1.15)
times = (scheduler.timesteps / scheduler.config.num_train_timesteps).bfloat16()
print(json.dumps(dict(check="turbo_times", times=times.float().tolist())), flush=True)
times = torch.cat((times, torch.linspace(0, 1, 9).bfloat16()))
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
stages = (
    ("first", "tmlp.0", "time_embed.linear_1"),
    ("embedding", "tmlp.2", "time_embed.linear_2"),
    ("modulation", "tproj.1", "time_mod_proj"),
)

with (
    safe_open(str(base_checkpoint), framework="pt") as weights,
    safe_open(str(adapter_checkpoint), framework="pt") as adapter,
):
    matrices = {
        name: (
            weights.get_tensor(key + ".weight").bfloat16().double(),
            weights.get_tensor(key + ".bias").bfloat16().double(),
            adapter.get_tensor("transformer." + adapter_key + ".lora_A.weight")
            .bfloat16()
            .double(),
            adapter.get_tensor("transformer." + adapter_key + ".lora_B.weight")
            .bfloat16()
            .double(),
        )
        for name, key, adapter_key in stages
    }
    for phase in ("base", "style"):
        adapted = phase == "style"
        if adapted:
            transformer.load_lora_adapter(
                str(args.adapter),
                weight_name="softwatercolor.safetensors",
                local_files_only=True,
                adapter_name="watercolor",
            )
            transformer.set_adapters("watercolor", weights=1.0)
        single_embedding = transformer.time_embed(
            torch.ones(1, device="cuda", dtype=torch.bfloat16), dtype=torch.bfloat16
        )
        single_modulation = transformer.time_mod_proj(
            torch.nn.functional.gelu(single_embedding, approximate="tanh")
        ).cpu()
        if (
            encode(single_modulation)
            != (args.reference / f"{phase}-block27-modulation.bf16").read_bytes()
        ):
            raise AssertionError("external time path changed from first-step capture")

        for count in args.counts:
            capacity = (count + 15) // 16 * 16
            directory = args.output / f"{phase}-{count}"
            directory.mkdir()
            paths = {
                "times": directory / "times.bf16",
                "strength": directory / "strength.f32",
            }
            paths["times"].write_bytes(encode(times[:count]))
            paths["strength"].write_bytes(
                np.array([float(adapted)], dtype="<f4").tobytes()
            )

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
                    f"--config=krea2.time_count={count}",
                    *["--input=" + str(paths[key]) for key in inputs],
                    "--expected=" + str(expected_path),
                    "--actual=" + str(actual_path),
                ]
                if exact:
                    command += ["--atol=0", "--rtol=0"]
                print(
                    json.dumps(dict(phase=phase, count=count, component=name)),
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
                comparisons = [record for record in records if "iteration" in record]
                if [record["iteration"] for record in comparisons] != [0, 1]:
                    raise AssertionError("checker did not complete both executions")
                if exact and (
                    any(record["different"] for record in comparisons)
                    or actual_path.read_bytes() != encode(expected)
                ):
                    raise AssertionError("non-bitwise conditioning composition")
                paths[name] = actual_path
                footprint = next(
                    record for record in records if "workspace_bytes" in record
                )
                return load(actual_path).reshape_as(expected), footprint

            padded_times = torch.zeros(capacity, dtype=torch.bfloat16)
            padded_times[:count] = times[:count]
            # Frequency and angle construction have explicit F32 boundaries in
            # the model. Transcendentals below are independently evaluated F64.
            exponent = -math.log(1e4) * torch.arange(128, dtype=torch.float32) / 128
            frequencies = exponent.double().exp().float()
            angles = (padded_times.float() * 1000)[:, None] * frequencies
            oracle = torch.cat(
                (angles.double().cos(), angles.double().sin()), -1
            ).bfloat16()
            native, _ = execute(
                "features", "qualify.time_sinusoidal", oracle, ["times"], []
            )
            native_name = "features"
            native_embedding = oracle_embedding = None
            for stage_index, (name, _, _) in enumerate(stages):
                if stage_index:
                    oracle = gelu(oracle)
                    native, _ = execute(
                        name + "_input",
                        "qualify.time_gelu",
                        gelu(native),
                        [native_name],
                        [],
                    )
                    native_name = name + "_input"
                matrix, bias, down, up = matrices[name]
                base, _ = execute(
                    name + "_base",
                    "qualify.time_" + name,
                    (native.double() @ matrix.T + bias).bfloat16(),
                    [native_name],
                    [base_checkpoint],
                )
                oracle_base = (oracle.double() @ matrix.T + bias).bfloat16()
                if adapted:
                    low, _ = execute(
                        name + "_low",
                        "qualify.time_" + name + "_adapter_down",
                        (native.double() @ down.T).bfloat16(),
                        [native_name],
                        [adapter_checkpoint],
                    )
                    delta, _ = execute(
                        name + "_delta",
                        "qualify.time_" + name + "_adapter_up",
                        (low.double() @ up.T).bfloat16(),
                        [name + "_low"],
                        [adapter_checkpoint],
                    )
                    combined, _ = execute(
                        name + "_combined",
                        "qualify.time_" + name + "_adapter_add",
                        base + delta,
                        [name + "_low", "strength", name + "_base"],
                        [adapter_checkpoint],
                        exact=True,
                    )
                    native, _ = execute(
                        name,
                        "qualify.time_" + name + "_adapted",
                        combined,
                        [native_name, "strength"],
                        [base_checkpoint, adapter_checkpoint],
                        exact=True,
                    )
                    oracle_low = (oracle.double() @ down.T).bfloat16()
                    oracle = oracle_base + (oracle_low.double() @ up.T).bfloat16()
                    native_name = name
                else:
                    native, oracle = base, oracle_base
                    native_name = name + "_base"
                report(phase, count, name + "_vs_f64", native[:count], oracle[:count])
                if name == "embedding":
                    native_embedding, oracle_embedding = native, oracle

            expected = torch.cat((native_embedding.flatten(), native.flatten()))
            whole, footprint = execute(
                "conditioning",
                "time_conditioning_adapted" if adapted else "time_conditioning",
                expected,
                ["times", "strength"] if adapted else ["times"],
                [base_checkpoint, adapter_checkpoint] if adapted else [base_checkpoint],
                exact=True,
            )
            if not adapted:
                execute(
                    "zero",
                    "time_conditioning_adapted",
                    whole,
                    ["times", "strength"],
                    [base_checkpoint, adapter_checkpoint],
                    exact=True,
                )
            if (
                footprint["kernels"] != (9 if adapted else 5)
                or footprint["parameters"] != (12 if adapted else 6)
                or footprint["parameter_roots"] != (2 if adapted else 1)
            ):
                raise AssertionError(
                    f"conditioning weights/kernels are not shared: {footprint}"
                )
            if footprint["workspace_bytes"] > capacity * (2 * 6144 + 256 + 32) * 2:
                raise AssertionError(
                    f"conditioning workspace exceeds live bound: {footprint}"
                )
            external_embedding = transformer.time_embed(
                padded_times.cuda(), dtype=torch.bfloat16
            )
            external_modulation = transformer.time_mod_proj(
                torch.nn.functional.gelu(external_embedding, approximate="tanh")
            )
            external = torch.cat(
                (
                    external_embedding[:count].cpu().flatten(),
                    external_modulation[:count].cpu().flatten(),
                )
            )
            # Complete physical storage is checked above. Accumulated model
            # accuracy describes logical steps, not deterministic padding.
            embedding_end = capacity * 6144
            logical = torch.cat(
                (
                    whole[:embedding_end].reshape(capacity, 6144)[:count].flatten(),
                    whole[embedding_end:].reshape(capacity, 36864)[:count].flatten(),
                )
            )
            independent = torch.cat(
                (oracle_embedding[:count].flatten(), oracle[:count].flatten())
            )
            report(phase, count, "native_conditioning_vs_f64", logical, independent)
            report(phase, count, "external_conditioning_vs_f64", external, independent)
            print(
                json.dumps(
                    dict(
                        phase=phase,
                        count=count,
                        check="conditioning_accepted",
                        **footprint,
                    )
                ),
                flush=True,
            )

print(
    "PASS: native time conditioning meets primitive, fusion, composition and zero-identity gates.",
    flush=True,
)
