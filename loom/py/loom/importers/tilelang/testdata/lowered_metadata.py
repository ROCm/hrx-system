# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _call(tir: Any, dtype: str, name: str, *args: Any) -> Any:
    return tir.Call(dtype, tir.op.Op.get(name), list(args))


def _buffer_pair(
    tir: Any,
    *,
    dtype: str = "float32",
    shape: tuple[int, ...] = (4,),
) -> tuple[Any, Any, Any, Any]:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer(shape, dtype, name="src")
    dst_buffer = tir.decl_buffer(shape, dtype, name="dst")
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


# ====
@tilelang_case(name="lazy_if_then_else", category="op", tags=("lowered", "control"))
def lazy_if_then_else(tir: Any, T: Any) -> TileLangImportInput:
    n = tir.Var("n", "int32")
    i = tir.Var("i", "int32")
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(16,))
    body = tir.BufferStore(
        dst_buffer,
        T.if_then_else(
            i < n,
            tir.BufferLoad(src_buffer, [i]),
            tir.FloatImm("float32", 0.0),
        ),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="lazy_if_then_else",
        params=[n, i, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="lazy_if_then_else",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("lazy_if_then_else") @lazy_if_then_else() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%n: i32, %i: i32, %src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %cmp = scalar.cmpi slt, %i, %n : i32
  %if = scf.if %cmp -> (f32) {
    %i_idx = index.cast %i : i32 to index
    %load = view.load %src_view[%i_idx] : view<16xf32, %layout> -> f32
    scf.yield %load : f32
  } else {
    %const = scalar.constant 0.0 : f32
    scf.yield %const : f32
  }
  %c0 = index.constant 0 : index
  view.store %if, %dst_view[%c0] : f32, view<16xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="tvm_access_ptr_atomic", category="op", tags=("lowered", "memory"))
def tvm_access_ptr_atomic(tir: Any, T: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "int32", name="dst")
    access = T.tvm_access_ptr(
        T.type_annotation("int32"),
        dst_buffer.data,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 1),
        tir.IntImm("int32", 3),
    )
    body = tir.Evaluate(
        _call(
            tir,
            "handle",
            "tl.atomic_add_elem_op",
            access,
            tir.IntImm("int32", 1),
        )
    )
    prim_func = _prim_func(
        tir,
        name="tvm_access_ptr_atomic",
        params=[dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="tvm_access_ptr_atomic",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tvm_access_ptr_atomic") @tvm_access_ptr_atomic() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %const = scalar.constant 1 : i32
  %c0 = index.constant 0 : index
  view.atomic.reduce<addi> %const, %dst_view[%c0] {ordering = relaxed, scope = device} : i32, view<4xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="tl_infinity_constant", category="op", tags=("lowered", "scalar"))
def tl_infinity_constant(tir: Any, T: Any) -> TileLangImportInput:
    _src, dst, _src_buffer, dst_buffer = _buffer_pair(tir)
    body = tir.BufferStore(
        dst_buffer,
        T.infinity("float32"),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="tl_infinity_constant",
        params=[dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="tl_infinity_constant",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tl_infinity_constant") @tl_infinity_constant() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %inf = scalar.constant inf : f32
  %c0 = index.constant 0 : index
  view.store %inf, %dst_view[%c0] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
# ERROR@+1: "call `tir.call_extern` to extern callee `debug_device` is not imported"
@tilelang_case(name="call_extern_diagnostic", category="diagnostic", tags=("lowered",))
def call_extern_diagnostic(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((1,), "float32", name="dst")
    body = tir.Evaluate(
        tir.call_extern(
            "void",
            "debug_device",
            tir.IntImm("int32", 1),
        )
    )
    prim_func = _prim_func(
        tir,
        name="call_extern_diagnostic",
        params=[dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="call_extern_diagnostic",
    )
