# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _buffer_pair(tir: Any) -> tuple[Any, Any, Any, Any]:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "float32", name="src")
    dst_buffer = tir.decl_buffer((16,), "float32", name="dst")
    return src, dst, src_buffer, dst_buffer


def _ramp4(tir: Any) -> Any:
    return tir.Ramp(tir.IntImm("int32", 0), tir.IntImm("int32", 1), 4)


# ====
@tilelang_case(name="vector_add_store", category="op", tags=("vector",))
def vector_add_store(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.BufferLoad(src_buffer, [ramp])
        + tir.Broadcast(tir.FloatImm("float32", 2.0), 4),
        [ramp],
    )
    prim_func = tir.PrimFunc(
        [src, dst],
        body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    ).with_attr("global_symbol", "vector_add_store")
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="vector_add_store"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("vector_add_store") @vector_add_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %c0 = index.constant 0 : index
  %load = vector.load %src_view[%c0] : view<16xf32, %layout> -> vector<4xf32>
  %const = scalar.constant 2.0 : f32
  %splat = vector.splat %const : vector<4xf32>
  %addf = vector.addf %load, %splat : vector<4xf32>
  vector.store %addf, %dst_view[%c0] : vector<4xf32>, view<16xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="vector_add_fastmath_store", category="op", tags=("vector",))
def vector_add_fastmath_store(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.BufferLoad(src_buffer, [ramp])
        + tir.Broadcast(tir.FloatImm("float32", 2.0), 4),
        [ramp],
    )
    prim_func = (
        tir.PrimFunc(
            [src, dst],
            body,
            buffer_map={src: src_buffer, dst: dst_buffer},
        )
        .with_attr("global_symbol", "vector_add_fastmath_store")
        .with_attr("tilelang_pass_configs", {"tl.enable_fast_math": True})
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="vector_add_fastmath_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("vector_add_fastmath_store") @vector_add_fastmath_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %c0 = index.constant 0 : index
  %load = vector.load %src_view[%c0] : view<16xf32, %layout> -> vector<4xf32>
  %const = scalar.constant 2.0 : f32
  %splat = vector.splat %const : vector<4xf32>
  %addf = vector.addf<reassoc|nnan|ninf|nsz|arcp|contract|afn> %load, %splat : vector<4xf32>
  vector.store %addf, %dst_view[%c0] : vector<4xf32>, view<16xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="vector_select_store", category="op", tags=("vector",))
def vector_select_store(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    ramp = _ramp4(tir)
    load = tir.BufferLoad(src_buffer, [ramp])
    threshold = tir.Broadcast(tir.FloatImm("float32", 0.0), 4)
    body = tir.BufferStore(
        dst_buffer,
        tir.Select(
            load < threshold,
            tir.Broadcast(tir.FloatImm("float32", 1.0), 4),
            tir.Broadcast(tir.FloatImm("float32", -1.0), 4),
        ),
        [ramp],
    )
    prim_func = tir.PrimFunc(
        [src, dst],
        body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    ).with_attr("global_symbol", "vector_select_store")
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="vector_select_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("vector_select_store") @vector_select_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %c0 = index.constant 0 : index
  %load = vector.load %src_view[%c0] : view<16xf32, %layout> -> vector<4xf32>
  %const = scalar.constant 0.0 : f32
  %splat = vector.splat %const : vector<4xf32>
  %cmp = vector.cmpf olt, %load, %splat : vector<4xf32> -> vector<4xi1>
  %const_2 = scalar.constant 1.0 : f32
  %splat_2 = vector.splat %const_2 : vector<4xf32>
  %const_3 = scalar.constant -1.0 : f32
  %splat_3 = vector.splat %const_3 : vector<4xf32>
  %select = vector.select %cmp, %splat_2, %splat_3 : vector<4xf32>
  vector.store %select, %dst_view[%c0] : vector<4xf32>, view<16xf32, %layout>
  kernel.return
}
"""
