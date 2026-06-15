# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


@tilelang_case(name="tileop_copy_1d", category="op", tags=("tileop", "copy"))
def tileop_copy_1d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_kernel(
            src: T.Tensor[(4,), T.float32],
            dst: T.Tensor[(4,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst)

        return tileop_copy_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_copy_kernel") @tileop_copy_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %src[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_copy_disable_tma_1d",
    category="op",
    tags=("tileop", "copy", "annotation"),
)
def tileop_copy_disable_tma_1d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_disable_tma_kernel(
            src: T.Tensor[(4,), T.float32],
            dst: T.Tensor[(4,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst, disable_tma=True)

        return tileop_copy_disable_tma_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_disable_tma_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_copy_disable_tma_kernel") @tileop_copy_disable_tma_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    %copy = view.load %src[%i0] : view<4xf32, %layout> -> f32
    view.store %copy, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_fill_1d", category="op", tags=("tileop", "fill"))
def tileop_fill_1d(tilelang: Any, T: Any, tir: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_fill_kernel(dst: T.Tensor[(4,), T.float32]) -> None:
            with T.Kernel(1, threads=1):
                T.fill(dst, tir.FloatImm("float32", 3.0))

        return tileop_fill_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_fill_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_fill_kernel") @tileop_fill_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %const = scalar.constant 3.0 : f32
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    view.store %const, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_fill_integer_zero_to_float",
    category="op",
    tags=("tileop", "fill", "constant"),
)
def tileop_fill_integer_zero_to_float(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_fill_integer_zero_kernel(dst: T.Tensor[(4,), T.float32]) -> None:
            with T.Kernel(1, threads=1):
                T.fill(dst, 0)

        return tileop_fill_integer_zero_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_fill_integer_zero_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_fill_integer_zero_kernel") @tileop_fill_integer_zero_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %const = scalar.constant 0.0 : f32
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c4 step %c1] {
    view.store %const, %dst[%i0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_reduce_sum_1d", category="op", tags=("tileop", "reduce"))
def tileop_reduce_sum_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_sum_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float32)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_sum(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_sum_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_sum_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_sum_kernel") @tileop_reduce_sum_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %local_state = vector.splat %f32_zero : vector<4xf32>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %out_state = vector.splat %f32_zero : vector<1xf32>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %copy = vector.load %src[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %reduce = vector.reduce<addf> %copy, %f32_zero : vector<4xf32>, f32
  %store = vector.insert %reduce into %out_state[%c0] : f32, vector<1xf32>
  %c1 = index.constant 1 : index
  %copy_2 = vector.extract %store[%c0] : vector<1xf32> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_sum_fastmath_1d",
    category="op",
    tags=("tileop", "reduce", "fastmath"),
)
def tileop_reduce_sum_fastmath_1d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_reduce_sum_fastmath_kernel(
            src: T.Tensor[(4,), T.float32],
            dst: T.Tensor[(1,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                local = T.alloc_fragment((4,), T.float32)
                out = T.alloc_fragment((1,), T.float32)
                T.copy(src, local)
                T.reduce_sum(local, out, dim=0)
                T.copy(out, dst)

        return tileop_reduce_sum_fastmath_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_sum_fastmath_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_sum_fastmath_kernel") @tileop_reduce_sum_fastmath_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %local_state = vector.splat %f32_zero : vector<4xf32>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %out_state = vector.splat %f32_zero : vector<1xf32>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %copy = vector.load %src[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %reduce = vector.reduce<addf, reassoc|nnan|ninf|nsz|arcp|contract|afn> %copy, %f32_zero : vector<4xf32>, f32
  %store = vector.insert %reduce into %out_state[%c0] : f32, vector<1xf32>
  %c1 = index.constant 1 : index
  %copy_2 = vector.extract %store[%c0] : vector<1xf32> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_abssum_1d",
    category="op",
    tags=("tileop", "reduce"),
)
def tileop_reduce_abssum_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_abssum_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float32)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_abssum(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_abssum_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_abssum_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_abssum_kernel") @tileop_reduce_abssum_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %local_state = vector.splat %f32_zero : vector<4xf32>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %out_state = vector.splat %f32_zero : vector<1xf32>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %copy = vector.load %src[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %abs = vector.absf %copy : vector<4xf32>
  %reduce = vector.reduce<addf> %abs, %f32_zero : vector<4xf32>, f32
  %store = vector.insert %reduce into %out_state[%c0] : f32, vector<1xf32>
  %c1 = index.constant 1 : index
  %copy_2 = vector.extract %store[%c0] : vector<1xf32> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_absmax_1d",
    category="op",
    tags=("tileop", "reduce"),
)
def tileop_reduce_absmax_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_absmax_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float32)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_absmax(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_absmax_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_absmax_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_absmax_kernel") @tileop_reduce_absmax_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 16 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %local_state = vector.splat %f32_zero : vector<4xf32>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %out_state = vector.splat %f32_zero : vector<1xf32>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %copy = vector.load %src[%c0] : view<4xf32, %layout> -> vector<4xf32>
  %abs = vector.absf %copy : vector<4xf32>
  %reduce = vector.reduce<maxnumf> %abs, %f32_zero : vector<4xf32>, f32
  %store = vector.insert %reduce into %out_state[%c0] : f32, vector<1xf32>
  %c1 = index.constant 1 : index
  %copy_2 = vector.extract %store[%c0] : vector<1xf32> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_absmax_widen_1d",
    category="op",
    tags=("tileop", "reduce", "cast"),
)
def tileop_reduce_absmax_widen_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_absmax_widen_kernel(
        src: T.Tensor[(4,), T.float16],
        dst: T.Tensor[(1,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((4,), T.float16)
            out = T.alloc_fragment((1,), T.float32)
            T.copy(src, local)
            T.reduce_absmax(local, out, dim=0)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_absmax_widen_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_absmax_widen_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_absmax_widen_kernel") @tileop_reduce_absmax_widen_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf16, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 8 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 2, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<4xf16, %layout>
  %f16_zero = scalar.constant 0.0 : f16
  %local_state = vector.splat %f16_zero : vector<4xf16>
  %out_bytes = index.constant 4 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %out_state = vector.splat %f32_zero : vector<1xf32>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %copy = vector.load %src[%c0] : view<4xf16, %layout> -> vector<4xf16>
  %reduce_cast = vector.extf %copy : vector<4xf16> to vector<4xf32>
  %abs = vector.absf %reduce_cast : vector<4xf32>
  %reduce = vector.reduce<maxnumf> %abs, %f32_zero : vector<4xf32>, f32
  %store = vector.insert %reduce into %out_state[%c0] : f32, vector<1xf32>
  %c1 = index.constant 1 : index
  %copy_2 = vector.extract %store[%c0] : vector<1xf32> -> f32
  view.store %copy_2, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_reduce_sum_2d", category="op", tags=("tileop", "reduce"))
def tileop_reduce_sum_2d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_sum_2d_kernel(
        src: T.Tensor[(2, 4), T.float32],
        dst: T.Tensor[(2,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((2, 4), T.float32)
            out = T.alloc_fragment((2,), T.float32)
            T.copy(src, local)
            T.reduce_sum(local, out, dim=1)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_sum_2d_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_sum_2d_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_sum_2d_kernel") @tileop_reduce_sum_2d_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 32 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %out_bytes = index.constant 8 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<2xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %out_state = vector.splat %f32_zero : vector<2xf32>
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c4 step %c1] {
      %copy = view.load %src[%i0, %i1] : view<2x4xf32, %layout> -> f32
      view.store %copy, %local[%i0, %i1] : f32, view<2x4xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    %reduce = scf.for %r = [%c0 to %c4 step %c1](%acc = %f32_zero : f32) -> (f32) {
      %reduce_value = view.load %local[%i0, %r] : view<2x4xf32, %layout> -> f32
      %addf = scalar.addf %acc, %reduce_value : f32
      scf.yield %addf : f32
    }
    %store = vector.insert %reduce into %out_state[%i0] : f32, vector<2xf32>
  }
  %i0_base = index.mul %tx, %c2 : index
  %copy_2 = vector.extract %out_state[%i0_base] : vector<2xf32> -> f32
  view.store %copy_2, %dst[%i0_base] : f32, view<2xf32, %layout>
  %i0 = index.add %i0_base, %c1 : index
  %copy_3 = vector.extract %out_state[%i0] : vector<2xf32> -> f32
  view.store %copy_3, %dst[%i0] : f32, view<2xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_finalize_reducer_none",
    category="op",
    tags=("tileop", "finalize_reducer"),
)
def tileop_finalize_reducer_none(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_finalize_reducer_none_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(4,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            reducer = T.alloc_reducer(
                (4,),
                T.float32,
                "sum",
                replication="none",
            )
            T.fill(reducer, 0.0)
            for i in T.serial(0, 4):
                reducer[i] = src[i]
            T.finalize_reducer(reducer)
            T.copy(reducer, dst)

    return TileLangImportInput(
        source=tileop_finalize_reducer_none_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_finalize_reducer_none_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_finalize_reducer_none_kernel") @tileop_finalize_reducer_none_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %reducer_bytes = index.constant 16 : offset
  %reducer_buffer = buffer.alloca %reducer_bytes {base_alignment = 4, memory_space = private} : buffer
  %reducer = buffer.view %reducer_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %reducer_state = vector.splat %f32_zero : vector<4xf32>
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %fill = vector.splat %f32_zero : vector<4xf32>
  %c1 = index.constant 1 : index
  %reducer_state_next = scf.for %i = [%c0 to %c4 step %c1](%reducer_state_iter = %fill : vector<4xf32>) -> (vector<4xf32>) {
    %load = view.load %src[%i] : view<4xf32, %layout> -> f32
    %store = vector.insert %load into %reducer_state_iter[%i] : f32, vector<4xf32>
    scf.yield %store : vector<4xf32>
  }
  %i0_base = index.mul %tx, %c4 : index
  %copy = vector.extract %reducer_state_next[%i0_base] : vector<4xf32> -> f32
  view.store %copy, %dst[%i0_base] : f32, view<4xf32, %layout>
  %i0 = index.add %i0_base, %c1 : index
  %copy_2 = vector.extract %reducer_state_next[%i0] : vector<4xf32> -> f32
  view.store %copy_2, %dst[%i0] : f32, view<4xf32, %layout>
  %c2 = index.constant 2 : index
  %i0_2 = index.add %i0_base, %c2 : index
  %copy_3 = vector.extract %reducer_state_next[%i0_2] : vector<4xf32> -> f32
  view.store %copy_3, %dst[%i0_2] : f32, view<4xf32, %layout>
  %c3 = index.constant 3 : index
  %i0_3 = index.add %i0_base, %c3 : index
  %copy_4 = vector.extract %reducer_state_next[%i0_3] : vector<4xf32> -> f32
  view.store %copy_4, %dst[%i0_3] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_cumsum_shared_1d",
    category="op",
    tags=("tileop", "cumsum", "workgroup"),
)
def tileop_cumsum_shared_1d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_cumsum_shared_1d_kernel(
        src: T.Tensor[(4,), T.int32],
        dst: T.Tensor[(4,), T.int32],
    ) -> None:
        with T.Kernel(1, threads=4):
            thread_index = T.get_thread_binding()
            shared = T.alloc_shared((4,), T.int32)
            shared[thread_index] = src[thread_index]
            T.sync_threads()
            T.cumsum(shared)
            T.sync_threads()
            dst[thread_index] = shared[thread_index]

    return TileLangImportInput(
        source=tileop_cumsum_shared_1d_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_cumsum_shared_1d_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_cumsum_shared_1d_kernel") @tileop_cumsum_shared_1d_kernel() {
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c4, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %shared_bytes = index.constant 16 : offset
  %shared_buffer = buffer.alloca %shared_bytes {base_alignment = 4, memory_space = workgroup} : buffer
  %shared = buffer.view %shared_buffer[%c0_bytes] : buffer -> view<4xi32, %layout>
  %load = view.load %src[%thread_index] : view<4xi32, %layout> -> i32
  view.store %load, %shared[%thread_index] : i32, view<4xi32, %layout>
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c1 = index.constant 1 : index
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %cumsum_value = view.load %shared[%thread_index] : view<4xi32, %layout> -> i32
  %cumsum = kernel.workgroup.scan<addi> %cumsum_value {direction = forward, mode = inclusive} : i32
  view.store %cumsum, %shared[%thread_index] : i32, view<4xi32, %layout>
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %load_2 = view.load %shared[%thread_index] : view<4xi32, %layout> -> i32
  view.store %load_2, %dst[%thread_index] : i32, view<4xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_finalize_reducer_all",
    category="op",
    tags=("tileop", "finalize_reducer", "workgroup"),
)
def tileop_finalize_reducer_all(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_finalize_reducer_all_kernel(
        src: T.Tensor[(4,), T.float32],
        dst: T.Tensor[(4,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=4):
            thread_index = T.get_thread_binding()
            reducer = T.alloc_reducer(
                (1,),
                T.float32,
                "sum",
                replication="all",
            )
            reducer[0] = src[thread_index]
            T.finalize_reducer(reducer)
            dst[thread_index] = reducer[0]

    return TileLangImportInput(
        source=tileop_finalize_reducer_all_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_finalize_reducer_all_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_finalize_reducer_all_kernel") @tileop_finalize_reducer_all_kernel() {
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c4, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %reducer_bytes = index.constant 4 : offset
  %reducer_buffer = buffer.alloca %reducer_bytes {base_alignment = 4, memory_space = private} : buffer
  %reducer = buffer.view %reducer_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %reducer_state = vector.splat %f32_zero : vector<1xf32>
  %load = view.load %src[%thread_index] : view<4xf32, %layout> -> f32
  %c0 = index.constant 0 : index
  %store = vector.insert %load into %reducer_state[%c0] : f32, vector<1xf32>
  %c1 = index.constant 1 : index
  %reducer_state_next = scf.for %i0 = [%c0 to %c1 step %c1](%reducer_state_iter = %store : vector<1xf32>) -> (vector<1xf32>) {
    %reducer_value = vector.extract %reducer_state_iter[%i0] : vector<1xf32> -> f32
    %reducer_all = kernel.workgroup.reduce<addf> %reducer_value : f32
    %store_2 = vector.insert %reducer_all into %reducer_state_iter[%i0] : f32, vector<1xf32>
    scf.yield %store_2 : vector<1xf32>
  }
  %load_2 = vector.extract %reducer_state_next[%c0] : vector<1xf32> -> f32
  view.store %load_2, %dst[%thread_index] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_reduce_absmax_reshape_2d_to_3d",
    category="op",
    tags=("tileop", "reduce", "reshape"),
)
def tileop_reduce_absmax_reshape_2d_to_3d(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_reduce_absmax_reshape_kernel(
        src: T.Tensor[(2, 4), T.float32],
        dst: T.Tensor[(2, 2), T.float32],
    ) -> None:
        with T.Kernel(1, threads=1):
            local = T.alloc_fragment((2, 4), T.float32)
            reshaped = T.reshape(local, (2, 2, 2))
            out = T.alloc_fragment((2, 2), T.float32)
            T.copy(src, local)
            T.reduce_absmax(reshaped, out, dim=2)
            T.copy(out, dst)

    return TileLangImportInput(
        source=tileop_reduce_absmax_reshape_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_reduce_absmax_reshape_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_reduce_absmax_reshape_kernel") @tileop_reduce_absmax_reshape_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<2x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 32 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 4, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %out_bytes = index.constant 16 : offset
  %out_buffer = buffer.alloca %out_bytes {base_alignment = 4, memory_space = private} : buffer
  %out = buffer.view %out_buffer[%c0_bytes] : buffer -> view<2x2xf32, %layout>
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c4 step %c1] {
      %copy = view.load %src[%i0, %i1] : view<2x4xf32, %layout> -> f32
      view.store %copy, %local[%i0, %i1] : f32, view<2x4xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity = scalar.constant 0.0 : f32
      %reduce = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity : f32) -> (f32) {
        %mul = index.mul %i0, %c2 : index
        %add = index.add %mul, %i1 : index
        %mul_2 = index.mul %add, %c2 : index
        %add_2 = index.add %mul_2, %r : index
        %rem = index.rem %add_2, %c4 : index
        %div = index.div %add_2, %c4 : index
        %reduce_value = view.load %local[%div, %rem] : view<2x4xf32, %layout> -> f32
        %abs = scalar.absf %reduce_value : f32
        %maxnumf = scalar.maxnumf %acc, %abs : f32
        scf.yield %maxnumf : f32
      }
      view.store %reduce, %out[%i0, %i1] : f32, view<2x2xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %copy_2 = view.load %out[%i0, %i1] : view<2x2xf32, %layout> -> f32
      view.store %copy_2, %dst[%i0, %i1] : f32, view<2x2xf32, %layout>
    }
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="tileop_copy_2d", category="op", tags=("tileop", "copy"))
def tileop_copy_2d(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def tileop_copy_2d_kernel(
            src: T.Tensor[(2, 3), T.float16],
            dst: T.Tensor[(2, 3), T.float16],
        ) -> None:
            with T.Kernel(1, threads=1):
                T.copy(src, dst)

        return tileop_copy_2d_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_copy_2d_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_copy_2d_kernel") @tileop_copy_2d_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<2x3xf16, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<2x3xf16, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c3 step %c1] {
      %copy = view.load %src[%i0, %i1] : view<2x3xf16, %layout> -> f16
      view.store %copy, %dst[%i0, %i1] : f16, view<2x3xf16, %layout>
    }
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tileop_gemm_dense_16x16x16",
    category="op",
    tags=("tileop", "gemm", "matrix"),
)
def tileop_gemm_dense_16x16x16(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def tileop_gemm_dense_16x16x16_kernel(
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
            T.gemm(a_shared, b_shared, c_local)
            T.copy(c_local, c)

    return TileLangImportInput(
        source=tileop_gemm_dense_16x16x16_kernel,
        target="hip -mcpu=gfx1100",
        name="tileop_gemm_dense_16x16x16_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tileop_gemm_dense_16x16x16_kernel") @tileop_gemm_dense_16x16x16_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%a_handle: buffer, %b_handle: buffer, %c_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %a_noalias = buffer.assume.noalias %a_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %a = buffer.view %a_noalias[%c0_bytes] : buffer -> view<16x16xf16, %layout>
  %b_noalias = buffer.assume.noalias %b_handle : buffer
  %b = buffer.view %b_noalias[%c0_bytes] : buffer -> view<16x16xf16, %layout>
  %c_noalias = buffer.assume.noalias %c_handle : buffer
  %c = buffer.view %c_noalias[%c0_bytes] : buffer -> view<16x16xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %a_shared_bytes = index.constant 512 : offset
  %a_shared_buffer = buffer.alloca %a_shared_bytes {base_alignment = 2, memory_space = workgroup} : buffer
  %a_shared = buffer.view %a_shared_buffer[%c0_bytes] : buffer -> view<16x16xf16, %layout>
  %b_shared_buffer = buffer.alloca %a_shared_bytes {base_alignment = 2, memory_space = workgroup} : buffer
  %b_shared_layout = encoding.layout.strided [1, 16] : encoding<layout>
  %b_shared = buffer.view %b_shared_buffer[%c0_bytes] : buffer -> view<16x16xf16, %b_shared_layout>
  %c_local_bytes = index.constant 1024 : offset
  %c_local_buffer = buffer.alloca %c_local_bytes {base_alignment = 4, memory_space = private} : buffer
  %c_local = buffer.view %c_local_buffer[%c0_bytes] : buffer -> view<16x16xf32, %layout>
  %c0 = index.constant 0 : index
  %c16 = index.constant 16 : index
  %c2 = index.constant 2 : index
  %i_rem = index.rem %tx, %c2 : index
  %i_div = index.div %tx, %c2 : index
  %c8 = index.constant 8 : index
  %i_offset = index.mul %i_rem, %c8 : index
  %copy = vector.load %a[%i_div, %i_offset] : view<16x16xf16, %layout> -> vector<8xf16>
  vector.store %copy, %a_shared[%i_div, %i_offset] : vector<8xf16>, view<16x16xf16, %layout>
  %i_rem_2 = index.rem %tx, %c2 : index
  %i_div_2 = index.div %tx, %c2 : index
  %i_offset_2 = index.mul %i_rem_2, %c8 : index
  %copy_2 = vector.load %b[%i_div_2, %i_offset_2] : view<16x16xf16, %layout> -> vector<8xf16>
  %copy_3 = vector.extract %copy_2[0] : vector<8xf16> -> f16
  view.store %copy_3, %b_shared[%i_div_2, %i_offset_2] : f16, view<16x16xf16, %b_shared_layout>
  %c1 = index.constant 1 : index
  %idx = index.add %i_offset_2, %c1 : index
  %copy_4 = vector.extract %copy_2[1] : vector<8xf16> -> f16
  view.store %copy_4, %b_shared[%i_div_2, %idx] : f16, view<16x16xf16, %b_shared_layout>
  %idx_2 = index.add %i_offset_2, %c2 : index
  %copy_5 = vector.extract %copy_2[2] : vector<8xf16> -> f16
  view.store %copy_5, %b_shared[%i_div_2, %idx_2] : f16, view<16x16xf16, %b_shared_layout>
  %c3 = index.constant 3 : index
  %idx_3 = index.add %i_offset_2, %c3 : index
  %copy_6 = vector.extract %copy_2[3] : vector<8xf16> -> f16
  view.store %copy_6, %b_shared[%i_div_2, %idx_3] : f16, view<16x16xf16, %b_shared_layout>
  %c4 = index.constant 4 : index
  %idx_4 = index.add %i_offset_2, %c4 : index
  %copy_7 = vector.extract %copy_2[4] : vector<8xf16> -> f16
  view.store %copy_7, %b_shared[%i_div_2, %idx_4] : f16, view<16x16xf16, %b_shared_layout>
  %c5 = index.constant 5 : index
  %idx_5 = index.add %i_offset_2, %c5 : index
  %copy_8 = vector.extract %copy_2[5] : vector<8xf16> -> f16
  view.store %copy_8, %b_shared[%i_div_2, %idx_5] : f16, view<16x16xf16, %b_shared_layout>
  %c6 = index.constant 6 : index
  %idx_6 = index.add %i_offset_2, %c6 : index
  %copy_9 = vector.extract %copy_2[6] : vector<8xf16> -> f16
  view.store %copy_9, %b_shared[%i_div_2, %idx_6] : f16, view<16x16xf16, %b_shared_layout>
  %c7 = index.constant 7 : index
  %idx_7 = index.add %i_offset_2, %c7 : index
  %copy_10 = vector.extract %copy_2[7] : vector<8xf16> -> f16
  view.store %copy_10, %b_shared[%i_div_2, %idx_7] : f16, view<16x16xf16, %b_shared_layout>
  %const = scalar.constant 0.0 : f32
  %fill = vector.splat %const : vector<8xf32>
  %init = vector.fragment<init> %fill shape [%c16, %c16] : vector<8xf32>
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %lhs = vector.fragment.load<lhs> %a_shared[%c0, %c0] shape [%c16, %c16] : view<16x16xf16, %layout> -> vector<16xf16>
  %rhs = vector.fragment.load<rhs> %b_shared[%c0, %c0] shape [%c16, %c16] : view<16x16xf16, %b_shared_layout> -> vector<16xf16>
  %gemm = vector.mma %lhs, %rhs, %init : vector<16xf16>, vector<16xf16>, vector<8xf32>
  vector.fragment.store<result> %gemm, %c[%c0, %c0] shape [%c16, %c16] : vector<8xf32>, view<16x16xf32, %layout>
  kernel.return
}
"""
