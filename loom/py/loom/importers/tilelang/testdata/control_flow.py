# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _call(tir: Any, dtype: str, name: str, *args: Any) -> Any:
    return tir.call_intrin(dtype, name, *args)


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


def _kernel_args(tir: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    n = tir.Var("n", "int32")
    i = tir.Var("i", "int32")
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((16,), "float32", name="src")
    dst_buffer = tir.decl_buffer((16,), "float32", name="dst")
    return n, i, src, dst, src_buffer, dst_buffer


# ====
@tilelang_case(name="thread_return_prefix_guard", category="op", tags=("control",))
def thread_return_prefix_guard(tir: Any) -> TileLangImportInput:
    n, i, src, dst, src_buffer, dst_buffer = _kernel_args(tir)
    body = tir.SeqStmt(
        [
            tir.IfThenElse(
                i >= n,
                tir.Evaluate(_call(tir, "void", "tir.thread_return")),
                None,
            ),
            tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [i]), [i]),
        ]
    )
    prim_func = _prim_func(
        tir,
        name="thread_return_prefix_guard",
        params=[n, i, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="thread_return_prefix_guard",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("thread_return_prefix_guard") @thread_return_prefix_guard() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%n: i32, %i: i32, %src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %cmp = scalar.cmpi sge, %i, %n : i32
  kernel.exit %cmp : i1
  %i_idx = index.cast %i : i32 to index
  %load = view.load %src_view[%i_idx] : view<16xf32, %layout> -> f32
  view.store %load, %dst_view[%i_idx] : f32, view<16xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="thread_return_prefix_guard_with_effects",
    category="op",
    tags=("control",),
)
def thread_return_prefix_guard_with_effects(tir: Any) -> TileLangImportInput:
    n, i, src, dst, src_buffer, dst_buffer = _kernel_args(tir)
    body = tir.SeqStmt(
        [
            tir.IfThenElse(
                i >= n,
                tir.SeqStmt(
                    [
                        tir.BufferStore(
                            dst_buffer,
                            tir.FloatImm("float32", 0.0),
                            [tir.IntImm("int32", 0)],
                        ),
                        tir.Evaluate(_call(tir, "void", "tir.thread_return")),
                    ]
                ),
                None,
            ),
            tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [i]), [i]),
        ]
    )
    prim_func = _prim_func(
        tir,
        name="thread_return_prefix_guard_with_effects",
        params=[n, i, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="thread_return_prefix_guard_with_effects",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("thread_return_prefix_guard_with_effects") @thread_return_prefix_guard_with_effects() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%n: i32, %i: i32, %src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<16xf32, %layout>
  %cmp = scalar.cmpi sge, %i, %n : i32
  kernel.exit %cmp : i1 {
    %const = scalar.constant 0.0 : f32
    %c0 = index.constant 0 : index
    view.store %const, %dst_view[%c0] : f32, view<16xf32, %layout>
  }
  %i_idx = index.cast %i : i32 to index
  %load = view.load %src_view[%i_idx] : view<16xf32, %layout> -> f32
  view.store %load, %dst_view[%i_idx] : f32, view<16xf32, %layout>
  kernel.return
}
"""
