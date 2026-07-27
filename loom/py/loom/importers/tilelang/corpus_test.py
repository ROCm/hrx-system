# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest import SkipTest

from loom.importers.check.tilelang.backend import TileLangBackend
from loom.importers.check.tilelang.harness import TileLangHarness
from loom.importers.core import kernel_module_ops, print_loom_module
from loom.importers.tilelang.importer import TileLangImportOptions, import_tilelang
from loom.importers.tilelang.model import TileLangImportInput
from loom.importers.tilelang.testdata.tilekernels import (
    engram,
    moe_expand,
    moe_gate,
    moe_histogram,
    moe_normalize,
    moe_reduce,
    transpose,
)

_TARGET_PRESETS = ("hip -mcpu=gfx942", "hip -mcpu=gfx1100")
_TARGET_DECL_RE = re.compile(
    r"amdgpu\.target<gfx(?:942|1100)> @hip_mcpu_gfx(?:942|1100)"
)
_TARGET_SYMBOL_RE = re.compile(r"@hip_mcpu_gfx(?:942|1100)")


@dataclass(frozen=True, slots=True)
class _CorpusCase:
    name: str
    fixture: Callable[[Any, Any], TileLangImportInput]
    required_fragments: tuple[str, ...]


_CORPUS_CASES = (
    _CorpusCase(
        name="batched_transpose",
        fixture=transpose.tilekernels_batched_transpose_gfx942,
        required_fragments=(
            "kernel.barrier<workgroup>",
            "buffer.alloca",
            "view.store",
        ),
    ),
    _CorpusCase(
        name="engram_hash",
        fixture=engram.tilekernels_engram_hash_gfx1100,
        required_fragments=(
            "kernel.exit",
            "scalar.remsi",
            "scf.select",
            "scf.for",
        ),
    ),
    _CorpusCase(
        name="group_count",
        fixture=moe_histogram.tilekernels_group_count_gfx1100,
        required_fragments=(
            "view.atomic.reduce<addi>",
            "kernel.barrier<workgroup>",
            "scalar.assume",
        ),
    ),
    _CorpusCase(
        name="expand_to_fused",
        fixture=moe_expand.tilekernels_expand_to_fused_gfx1100,
        required_fragments=(
            "index.max",
            "kernel.exit",
            "vector.store",
        ),
    ),
    _CorpusCase(
        name="topk_gate",
        fixture=moe_gate.tilekernels_topk_gate_gfx1100,
        required_fragments=(
            "vector.reduce<maxnumf>",
            "kernel.workgroup.reduce<minsi>",
            "kernel.barrier<workgroup>",
        ),
    ),
    _CorpusCase(
        name="reduce_fused",
        fixture=moe_reduce.tilekernels_reduce_fused_gfx1100,
        required_fragments=(
            "scalar.assume",
            "scalar.extf",
            "vector.insert",
        ),
    ),
    _CorpusCase(
        name="normalize_weight",
        fixture=moe_normalize.tilekernels_normalize_weight_gfx1100,
        required_fragments=(
            "scalar.addf",
            "scalar.divf",
            "view.store",
        ),
    ),
)


def test_tilekernels_corpus_retargets_between_cdna_and_rdna() -> None:
    _require_tilelang()
    harness = TileLangHarness()
    for corpus_case in _CORPUS_CASES:
        outputs = {
            target_preset: _import_corpus_case(
                corpus_case,
                harness=harness,
                target_preset=target_preset,
            )
            for target_preset in _TARGET_PRESETS
        }
        assert _erase_target_identity(outputs["hip -mcpu=gfx942"]) == (
            _erase_target_identity(outputs["hip -mcpu=gfx1100"])
        ), corpus_case.name


def test_cdna_fp8_high_level_gemm_imports_to_mfma_fragments() -> None:
    _require_tilelang()
    harness = TileLangHarness()
    import_input = _build_cdna_fp8_gemm_input(harness.T)
    result = import_tilelang(
        import_input,
        options=TileLangImportOptions(target_preset="hip -mcpu=gfx942"),
    )
    output = print_loom_module(
        result.module,
        ops=kernel_module_ops("hip -mcpu=gfx942"),
    )

    for fragment in (
        'matrix_operand<element_format="f8e4m3fnuz"',
        "kernel.barrier<workgroup>",
        "vector.bitcast",
        "vector.fragment<lhs>",
        "vector.fragment<rhs>",
        "vector.mma",
        "vector.store",
    ):
        assert fragment in output, fragment


def _import_corpus_case(
    corpus_case: _CorpusCase,
    *,
    harness: TileLangHarness,
    target_preset: str,
) -> str:
    import_input = corpus_case.fixture(harness.tilelang, harness.T)
    result = import_tilelang(
        import_input,
        options=TileLangImportOptions(target_preset=target_preset),
    )
    output = print_loom_module(
        result.module,
        ops=kernel_module_ops(target_preset),
    )
    arch = _target_arch(target_preset)
    target_symbol = f"hip_mcpu_{arch}"
    assert f"amdgpu.target<{arch}> @{target_symbol}" in output
    assert f"kernel.def target(@{target_symbol})" in output
    for fragment in corpus_case.required_fragments:
        assert fragment in output, f"{corpus_case.name}: {fragment}"
    return output


def _build_cdna_fp8_gemm_input(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def cdna_fp8_gemm_kernel(
        a: T.Tensor[(64, 64), T.float8_e4m3fnuz],
        b: T.Tensor[(64, 64), T.float8_e4m3fnuz],
        c: T.Tensor[(64, 64), T.float32],
    ) -> None:
        with T.Kernel(1, threads=256):
            a_shared = T.alloc_shared((64, 64), T.float8_e4m3fnuz)
            b_shared = T.alloc_shared((64, 64), T.float8_e4m3fnuz)
            c_local = T.alloc_fragment((64, 64), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.clear(c_local)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                transpose_B=True,
                k_pack=2,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(c_local, c)

    return TileLangImportInput(
        source=cdna_fp8_gemm_kernel,
        target="hip -mcpu=gfx942",
        name="cdna_fp8_gemm_kernel",
    )


def _erase_target_identity(output: str) -> str:
    output = _TARGET_DECL_RE.sub("amdgpu.target<gfx> @hip_mcpu_gfx", output)
    return _TARGET_SYMBOL_RE.sub("@hip_mcpu_gfx", output)


def _target_arch(target_preset: str) -> str:
    match = re.search(r"\bgfx[0-9]+\b", target_preset)
    if match is None:
        raise AssertionError(f"target preset has no AMDGPU arch: {target_preset}")
    return match.group(0)


def _require_tilelang() -> None:
    backend = TileLangBackend()
    availability = backend.probe()
    if availability.available:
        availability = backend.prepare(Namespace())
    if not availability.available:
        raise SkipTest(availability.message())
