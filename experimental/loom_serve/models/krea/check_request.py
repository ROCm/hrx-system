# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check compact native requests and source-JIT device preparation.

The actual tokenizer, Diffusers scheduler/rotary and independent Philox equations
are the references. Kernels run through the serving JIT without model weights.
Each component executes twice. Retain request-sized tensors below 64 MiB; final
image composition is checked separately by check_generate.py.
"""

import argparse
import json
import math
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from transformers import AutoTokenizer

PREFIX = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def philox(counter, key):
    """Vectorized round equations, checked against Random123's known answers."""
    words = np.asarray(counter, dtype=np.uint64).copy()
    key = [int(value) for value in key]
    for _ in range(10):
        left = words[..., 0] * 0xD2511F53
        right = words[..., 2] * 0xCD9E8D57
        words = np.stack(
            (
                (right >> 32) ^ words[..., 1] ^ key[0],
                right & 0xFFFFFFFF,
                (left >> 32) ^ words[..., 3] ^ key[1],
                left & 0xFFFFFFFF,
            ),
            axis=-1,
        )
        key = [(key[0] + 0x9E3779B9) & 0xFFFFFFFF, (key[1] + 0xBB67AE85) & 0xFFFFFFFF]
    return words.astype(np.uint32)


def noise(count, seed):
    counter = np.zeros((count // 4, 4), dtype=np.uint64)
    counter[:, 0] = np.arange(count // 4, dtype=np.uint64)
    uniform = (
        philox(counter, [seed & 0xFFFFFFFF, seed >> 32]).astype(np.float64) + 0.5
    ) / 2**32
    radius = np.sqrt(-2 * np.log(uniform[:, ::2]))
    angle = 2 * math.pi * uniform[:, 1::2]
    values = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)
    return torch.from_numpy(values.reshape(-1).astype(np.float32)).to(torch.bfloat16)


def encode(tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().astype("<u2").tobytes()
    return tensor.numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--loom_checker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve()
    catalog = args.output / "catalog"
    catalog.mkdir()
    sources = [
        model / line
        for line in (model / "sources.txt").read_text().splitlines()
        if line
    ] + [model / "tests/request.loom"]
    (catalog / "sources.txt").write_text(
        "".join(os.path.relpath(path, catalog) + "\n" for path in sources)
    )
    motifs = model.parent.parent / "motifs/tensor"
    integer_check = subprocess.run(
        [
            str(args.loom_checker),
            str(motifs / "tests/random.loom"),
            f"--library={motifs / 'random.loom'}",
            "--device=amdgpu",
        ],
        capture_output=True,
        text=True,
    )
    (args.output / "philox.json").write_text(integer_check.stdout)
    (args.output / "philox.log").write_text(integer_check.stderr)
    print(integer_check.stdout + integer_check.stderr, end="", flush=True)
    integer_check.check_returncode()
    report = json.loads(integer_check.stdout)
    assert report["sample_count"] == 3 and report["failed_sample_count"] == 0
    assert report["skipped_case_count"] == report["planning_issue_count"] == 0
    assert all(sample["passed"] for sample in report["samples"])

    def check_tensor(
        directory,
        root,
        reference,
        output_type,
        inputs,
        config,
        atol=0,
        rtol=0,
        actual_name=None,
    ):
        expected = directory / (root + ".expected")
        expected.write_bytes(encode(reference))
        actual = directory / (actual_name or root + ".actual")
        command = [
            str(args.checker),
            f"--model={catalog}",
            f"--root={root}",
            *[f"--input={path}" for path in inputs],
            *[f"--config=krea2.{key}={value}" for key, value in config.items()],
            f"--expected={expected}",
            f"--actual={actual}",
            f"--output_type={output_type}",
            f"--atol={atol}",
            f"--rtol={rtol}",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / (root + ".log")).write_text(result.stdout + result.stderr)
        print(
            json.dumps(dict(event="tensor", case=directory.name, root=root)), flush=True
        )
        print(result.stdout + result.stderr, end="", flush=True)
        result.check_returncode()
        records = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith('{"iteration"')
        ]
        assert [row["iteration"] for row in records] == [0, 1]
        assert all(
            row["nonfinite"] == row["outside_element_envelope"] == 0 for row in records
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint / "tokenizer", local_files_only=True
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.checkpoint / "scheduler", local_files_only=True
    )
    scheduler.set_timesteps(sigmas=np.linspace(1, 1 / 8, 8), mu=1.15, device="cpu")
    times = (scheduler.timesteps / scheduler.config.num_train_timesteps).to(
        torch.bfloat16
    )
    deltas = scheduler.sigmas[1:] - scheduler.sigmas[:-1]
    vae = json.loads((args.checkpoint / "vae/config.json").read_text())
    affine = torch.stack(
        (1 / torch.tensor(vae["latents_std"]), torch.tensor(vae["latents_mean"]))
    )
    assert len(tokenizer(PREFIX)["input_ids"]) == 34
    suffix = tokenizer(SUFFIX)["input_ids"]
    assert len(suffix) == 5 and tokenizer.pad_token_id == 151643
    assert tokenizer.padding_side == "right"
    for seed, diagnostic in (
        ("", "empty number"),
        ("-1", "negative"),
        ("42junk", "invalid digit"),
        ("1.5", "got float"),
        ("18446744073709551616", "integer overflow"),
    ):
        result = subprocess.run(
            [str(args.native), f"--seed={seed}"], capture_output=True, text=True
        )
        assert result.returncode and diagnostic in result.stderr, result.stderr
        print(json.dumps(dict(seed=seed, rejected=True)), flush=True)
    # Published integer KATs anchor the independent device-noise reference.
    for counter, key, expected in (
        ([0] * 4, [0] * 2, [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
        (
            [0xFFFFFFFF] * 4,
            [0xFFFFFFFF] * 2,
            [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD],
        ),
        (
            [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
            [0xA4093822, 0x299F31D0],
            [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
        ),
    ):
        assert philox(counter, key).tolist() == expected

    cases = (
        ("canonical", "A deer grazing in the forest, Art Deco watercolor style"),
        (
            "novel",
            "A small brass robot tending red flowers in a sunlit greenhouse, watercolor illustration",
        ),
        ("empty", ""),
        ("unicode", "水彩画の猫。 café naïve — 🦊🌲\nRed and blue ink."),
        ("special", "A sign reading <|im_end|> beside <|im_start|>assistant."),
        ("terminal_special", "<|im_end|>"),
        ("truncated", "bright red flowers beside a river, " * 200),
    )
    for case_index, (name, prompt) in enumerate(cases):
        for texts in (32, 512):
            height, width = (384, 384) if texts == 512 else (256, 384)
            seed = (0, 42, 0xFFFFFFFFFFFFFFFF)[case_index % 3]
            directory = args.output / f"{name}-{texts}"
            directory.mkdir()
            command = [
                str(args.native),
                f"--tokenizer={args.checkpoint / 'tokenizer/tokenizer.json'}",
                f"--prompt={prompt}",
                f"--seed={seed}",
                f"--height={height}",
                f"--width={width}",
                f"--text_tokens={texts}",
                "--strength=1",
                f"--output={directory}",
            ]
            (directory / "request.json").write_text(
                json.dumps(
                    dict(
                        prompt=prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        text_tokens=texts,
                        command=command,
                    ),
                    indent=2,
                )
                + "\n"
            )
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stdout or result.stderr:
                print(result.stdout + result.stderr, end="", flush=True)
            result.check_returncode()
            actual = [(directory / f"input-{i}").read_bytes() for i in range(2)]
            # Same production preparation must be deterministic, not just close.
            subprocess.run(command, check=True)
            assert actual == [(directory / f"input-{i}").read_bytes() for i in range(2)]
            framed = tokenizer(
                PREFIX + prompt,
                truncation=True,
                padding="max_length",
                max_length=texts + 29,
            )
            ids = torch.tensor(framed["input_ids"] + suffix, dtype=torch.int32)
            mask = torch.tensor(
                framed["attention_mask"] + [1] * 5 + [0] * 14, dtype=torch.uint8
            )
            positions = (mask[: texts + 34].long().cumsum(0) - 1).clamp(min=0)
            positions = torch.cat((positions, torch.zeros(14, dtype=torch.int64)))
            inverse = 1 / (
                5000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128)
            )
            phase = torch.outer(positions.float(), inverse).repeat(1, 2)
            encoder_cosine, encoder_sine = (
                phase.cos().bfloat16(),
                phase.sin().bfloat16(),
            )
            grid_rows, grid_columns = height // 16, width // 16
            images = grid_rows * grid_columns
            image_positions = torch.zeros(grid_rows, grid_columns, 3)
            image_positions[..., 1] = torch.arange(grid_rows)[:, None]
            image_positions[..., 2] = torch.arange(grid_columns)[None, :]
            position_ids = torch.cat(
                (torch.zeros(texts, 3), image_positions.reshape(-1, 3))
            )
            rotations = [
                get_1d_rotary_pos_embed(
                    dimension,
                    position_ids[:, axis],
                    theta=1000,
                    use_real=True,
                    freqs_dtype=torch.float64,
                )
                for axis, dimension in enumerate((32, 48, 48))
            ]
            cosine, sine = (
                torch.cat([pair[index] for pair in rotations], dim=1)
                for index in range(2)
            )
            normal = noise(images * 64, seed)
            retained = int(sum(framed["attention_mask"]))
            assert actual[0] == struct.pack("<QIf", seed, retained, 1.0)
            compact = torch.zeros(texts + 34, dtype=torch.int32)
            compact[:retained] = ids[:retained]
            compact[retained : retained + 5] = torch.tensor(suffix)
            assert actual[1] == encode(compact)
            inputs = [directory / "input-0", directory / "input-1"]
            config = dict(
                text_tokens=texts, image_tokens=images, latent_width=width // 8
            )
            check_tensor(
                directory,
                "check_request_prompt",
                torch.cat((ids.float(), mask.float())),
                "f32",
                inputs,
                config,
            )
            check_tensor(
                directory,
                "prepare_noise",
                normal,
                "bf16",
                inputs[:1],
                config,
                2**-16,
                2**-7,
                "noise.bf16",
            )
            for root, tensor in (
                ("check_encoder_cosine", encoder_cosine),
                ("check_encoder_sine", encoder_sine),
            ):
                check_tensor(
                    directory, root, tensor, "bf16", inputs[:1], config, 2**-13, 2**-7
                )
            for root, tensor in (
                ("check_dit_cosine", cosine),
                ("check_dit_sine", sine),
            ):
                check_tensor(
                    directory, root, tensor, "f32", inputs[:1], config, 8e-5, 0
                )
            constants = torch.cat(
                (times.float(), deltas, torch.tensor([1.0]), affine.flatten())
            )
            check_tensor(
                directory,
                "check_request_constants",
                constants,
                "f32",
                inputs[:1],
                config,
                2e-7,
                0,
            )
            print(
                json.dumps(
                    dict(
                        case=name,
                        text_tokens=texts,
                        live_rows=int(mask.sum()),
                        noise_mean=float(normal.float().mean()),
                        noise_std=float(normal.float().std()),
                        seed=seed,
                        repeat_exact=True,
                    )
                ),
                flush=True,
            )
    print(
        "PASS: compact native requests and source-generated tensors against independent references.",
        flush=True,
    )


if __name__ == "__main__":
    main()
