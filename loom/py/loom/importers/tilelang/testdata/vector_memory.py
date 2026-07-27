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


def _prim_func(
    tir: Any,
    *,
    name: str,
    params: list[Any],
    body: Any,
    buffer_map: dict[Any, Any],
) -> Any:
    return tir.PrimFunc(params, body, buffer_map=buffer_map).with_attr(
        "global_symbol", name
    )


def _ramp4(tir: Any, base: int) -> Any:
    return tir.Ramp(tir.IntImm("int32", base), tir.IntImm("int32", 1), 4)


# ====
@tilelang_case(name="broadcast_vector_store", category="op", tags=("vector", "memory"))
def broadcast_vector_store(tir: Any) -> TileLangImportInput:
    _src, dst, _src_buffer, dst_buffer = _buffer_pair(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.Broadcast(tir.FloatImm("float32", 0.0), 4),
        [_ramp4(tir, 4)],
    )
    prim_func = _prim_func(
        tir,
        name="broadcast_vector_store",
        params=[dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="broadcast_vector_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("broadcast_vector_store") @broadcast_vector_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %const = scalar.constant 0.0 : f32
  %splat = vector.splat %const : vector<4xf32>
  %c4 = index.constant 4 : index
  vector.store %splat, %dst_view[%c4] : vector<4xf32>, view<16xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="ramp_vector_load", category="op", tags=("vector", "memory"))
def ramp_vector_load(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.BufferLoad(src_buffer, [_ramp4(tir, 8)]),
        [_ramp4(tir, 0)],
    )
    prim_func = _prim_func(
        tir,
        name="ramp_vector_load",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="ramp_vector_load"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("ramp_vector_load") @ramp_vector_load() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %c8 = index.constant 8 : index
  %load = vector.load %src_view[%c8] : view<16xf32, %layout> -> vector<4xf32>
  %c0 = index.constant 0 : index
  vector.store %load, %dst_view[%c0] : vector<4xf32>, view<16xf32, %layout>
  kernel.return
}
"""
