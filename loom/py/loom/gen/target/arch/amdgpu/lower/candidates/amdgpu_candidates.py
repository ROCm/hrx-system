# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low descriptor candidate tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[7]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.target.arch.amdgpu.lower.candidates import (  # noqa: E402
    amdgpu_arithmetic_candidates,
    amdgpu_async_gather_candidates,
    amdgpu_atomic_candidates,
    amdgpu_compare_candidates,
    amdgpu_memory_candidates,
)

_ARITHMETIC_PUBLIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/arithmetic_candidates.h"
_ASYNC_GATHER_PUBLIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/async_gather_candidates.h"
_ATOMIC_PUBLIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/atomic_candidates.h"
_COMPARE_PUBLIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low descriptor candidate tables.")
    parser.add_argument(
        "--arithmetic-source",
        required=True,
        type=Path,
        help="Generated arithmetic candidate source path.",
    )
    parser.add_argument(
        "--async-gather-source",
        required=True,
        type=Path,
        help="Generated async gather candidate source path.",
    )
    parser.add_argument(
        "--atomic-source",
        required=True,
        type=Path,
        help="Generated atomic candidate source path.",
    )
    parser.add_argument(
        "--compare-source",
        required=True,
        type=Path,
        help="Generated compare candidate source path.",
    )
    parser.add_argument(
        "--memory-candidate-ranges",
        required=True,
        type=Path,
        help="Generated memory descriptor candidate range fragment path.",
    )
    parser.add_argument(
        "--memory-candidate-rows",
        required=True,
        type=Path,
        help="Generated memory descriptor candidate row fragment path.",
    )
    args = parser.parse_args(argv)

    write_text_file(
        args.arithmetic_source,
        amdgpu_arithmetic_candidates._emit_source(public_header=_ARITHMETIC_PUBLIC_HEADER),
    )
    write_text_file(
        args.async_gather_source,
        amdgpu_async_gather_candidates._emit_source(public_header=_ASYNC_GATHER_PUBLIC_HEADER),
    )
    write_text_file(
        args.atomic_source,
        amdgpu_atomic_candidates._emit_source(public_header=_ATOMIC_PUBLIC_HEADER),
    )
    write_text_file(
        args.compare_source,
        amdgpu_compare_candidates._emit_source(public_header=_COMPARE_PUBLIC_HEADER),
    )
    memory_candidate_ranges, memory_candidate_rows = amdgpu_memory_candidates._emit_tables()
    write_text_file(
        args.memory_candidate_ranges,
        memory_candidate_ranges,
    )
    write_text_file(
        args.memory_candidate_rows,
        memory_candidate_rows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
