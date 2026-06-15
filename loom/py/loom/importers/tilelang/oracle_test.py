# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import ClassVar

from loom.importers.tilelang.model import (
    TileLangImportInput,
    resolve_tilelang_input,
)
from loom.importers.tilelang.oracle import (
    TileLangOracleError,
    summarize_disassembly,
    summarize_source,
    target_arch,
)


class _JitSource:
    pass_configs: ClassVar[dict[str, object]] = {"tl.disable_warp_specialized": True}

    def get_tir(self, hidden: int, *, num_topk: int) -> str:
        return f"tir:{hidden}:{num_topk}"


def test_resolve_tilelang_input_elaborates_get_tir_source() -> None:
    resolved = resolve_tilelang_input(
        TileLangImportInput(
            source=_JitSource(),
            args=(128,),
            kwargs={"num_topk": 2},
            target="hip -mcpu=gfx1100",
            name="expand_to_fused_kernel",
        )
    )

    assert resolved.source == "tir:128:2"
    assert resolved.target == "hip -mcpu=gfx1100"
    assert resolved.name == "expand_to_fused_kernel"


def test_target_arch_requires_explicit_mcpu() -> None:
    assert target_arch("hip -mcpu=gfx1100") == "gfx1100"
    assert target_arch("hip --offload-arch=gfx942") == "gfx942"
    try:
        target_arch("hip")
    except TileLangOracleError:
        pass
    else:
        raise AssertionError("expected missing -mcpu target to fail")


def test_summarize_source_reports_kernel_markers() -> None:
    summary = summarize_source(
        'extern "C" __global__ void kernel() {\n'
        "  __builtin_amdgcn_s_waitcnt(0);\n"
        "  v_mfma_f32_16x16x16f16();\n"
        "}\n"
    )

    assert "source_lines=4" in summary
    assert "has_global=True" in summary
    assert "has_mfma=True" in summary
    assert "has_waitcnt=True" in summary
    assert summary[-1].startswith('kernel_decl=extern "C" __global__ void kernel')


def test_summarize_disassembly_counts_instruction_families() -> None:
    counts = summarize_disassembly(
        "global_load_b128 v[0:3], v0\n"
        "global_store_b128 v0, v[0:3]\n"
        "s_waitcnt vmcnt(0)\n"
        "s_waitcnt lgkmcnt(0)\n"
    )

    assert counts == {
        "global_load": 1,
        "global_store": 1,
        "s_waitcnt": 2,
    }
