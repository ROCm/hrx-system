# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from loom.importers.check.tilelang import TileLangHarness
from loom.importers.tilelang.analysis import (
    collect_address_layout_preferences,
)


def test_marks_gfx1100_dense_gemm_rhs_shared_layout() -> None:
    harness = TileLangHarness()
    T: Any = harness.T
    gemm_kernel = _build_gemm_kernel(T)

    preferences = collect_address_layout_preferences(
        gemm_kernel,
        target_preset="hip -mcpu=gfx1100",
    )

    unique_preferences = {
        (preference.strides, preference.name) for preference in preferences.values()
    }
    assert unique_preferences == {((1, 16), "b_shared_layout")}


def test_ignores_gemm_variants_outside_current_bridge() -> None:
    harness = TileLangHarness()
    T: Any = harness.T
    gemm_kernel = _build_gemm_kernel(T)
    transposed_kernel = _build_gemm_kernel(T, transpose_b=True)

    assert (
        collect_address_layout_preferences(
            gemm_kernel,
            target_preset="hip -mcpu=gfx950",
        )
        == {}
    )
    assert (
        collect_address_layout_preferences(
            transposed_kernel,
            target_preset="hip -mcpu=gfx1100",
        )
        == {}
    )


def _build_gemm_kernel(T: Any, *, transpose_b: bool = False) -> Any:
    @T.prim_func  # type: ignore[untyped-decorator]
    def gemm_kernel(
        a: T.Tensor[(16, 16), T.float16],
        b: T.Tensor[(16, 16), T.float16],
        c: T.Tensor[(16, 16), T.float32],
    ) -> None:
        with T.Kernel(1, threads=32):
            a_shared = T.alloc_shared((16, 16), T.float16)
            b_shared = T.alloc_shared((16, 16), T.float16)
            c_local = T.alloc_fragment((16, 16), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.fill(c_local, 0.0)
            T.gemm(a_shared, b_shared, c_local, transpose_B=transpose_b)
            T.copy(c_local, c)

    return gemm_kernel
