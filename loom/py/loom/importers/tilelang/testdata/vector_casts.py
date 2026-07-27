# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


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


def _ramp4(tir: Any) -> Any:
    return tir.Ramp(tir.IntImm("int32", 0), tir.IntImm("int32", 1), 4)


# ====
@tilelang_case(name="vector_float_trunc_store", category="op", tags=("vector", "cast"))
def vector_float_trunc_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "float32", name="src")
    dst_buffer = tir.decl_buffer((16,), "float16", name="dst")
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.Cast("float16x4", tir.BufferLoad(src_buffer, [ramp])),
        [ramp],
    )
    prim_func = _prim_func(
        tir,
        name="vector_float_trunc_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="vector_float_trunc_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("vector_float_trunc_store") @vector_float_trunc_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf16, %layout>
  %c0 = index.constant 0 : index
  %load = vector.load %src_view[%c0] : view<16xf32, %layout> -> vector<4xf32>
  %fptrunc = vector.fptrunc %load : vector<4xf32> to vector<4xf16>
  vector.store %fptrunc, %dst_view[%c0] : vector<4xf16>, view<16xf16, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="scalar_reinterpret_store", category="op", tags=("cast",))
def scalar_reinterpret_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((4,), "float32", name="src")
    dst_buffer = tir.decl_buffer((4,), "uint32", name="dst")
    body = tir.BufferStore(
        dst_buffer,
        tir.reinterpret(
            "uint32",
            tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)]),
        ),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="scalar_reinterpret_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="scalar_reinterpret_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scalar_reinterpret_store") @scalar_reinterpret_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  %bitcast = scalar.bitcast %load : f32 to i32
  view.store %bitcast, %dst_view[%c0] : i32, view<4xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="vector_reinterpret_store", category="op", tags=("vector", "cast"))
def vector_reinterpret_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "float32", name="src")
    dst_buffer = tir.decl_buffer((16,), "uint32", name="dst")
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.reinterpret("uint32x4", tir.BufferLoad(src_buffer, [ramp])),
        [ramp],
    )
    prim_func = _prim_func(
        tir,
        name="vector_reinterpret_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="vector_reinterpret_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("vector_reinterpret_store") @vector_reinterpret_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xi32, %layout>
  %c0 = index.constant 0 : index
  %load = vector.load %src_view[%c0] : view<16xf32, %layout> -> vector<4xf32>
  %bitcast = vector.bitcast %load : vector<4xf32> to vector<4xi32>
  vector.store %bitcast, %dst_view[%c0] : vector<4xi32>, view<16xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="vector_int_to_float_store", category="op", tags=("vector", "cast"))
def vector_int_to_float_store(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "int32", name="src")
    dst_buffer = tir.decl_buffer((16,), "float32", name="dst")
    ramp = _ramp4(tir)
    body = tir.BufferStore(
        dst_buffer,
        tir.Cast("float32x4", tir.BufferLoad(src_buffer, [ramp])),
        [ramp],
    )
    prim_func = _prim_func(
        tir,
        name="vector_int_to_float_store",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="vector_int_to_float_store",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("vector_int_to_float_store") @vector_int_to_float_store() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xi32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %c0 = index.constant 0 : index
  %load = vector.load %src_view[%c0] : view<16xi32, %layout> -> vector<4xi32>
  %sitofp = vector.sitofp %load : vector<4xi32> to vector<4xf32>
  vector.store %sitofp, %dst_view[%c0] : vector<4xf32>, view<16xf32, %layout>
  kernel.return
}
"""
