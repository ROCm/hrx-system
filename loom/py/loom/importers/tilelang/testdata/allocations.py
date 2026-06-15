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


# ====
@tilelang_case(name="shared_block_alloc", category="op", tags=("alloc", "memory"))
def shared_block_alloc(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "float32", name="dst")
    scratch = tir.decl_buffer((4,), "float32", name="scratch", scope="shared.dyn")
    body = tir.Block(
        [],
        [],
        [],
        "root",
        tir.SeqStmt(
            [
                tir.BufferStore(
                    scratch,
                    tir.FloatImm("float32", 1.0),
                    [tir.IntImm("int32", 0)],
                ),
                tir.BufferStore(
                    dst_buffer,
                    tir.BufferLoad(scratch, [tir.IntImm("int32", 0)]),
                    [tir.IntImm("int32", 0)],
                ),
            ]
        ),
        alloc_buffers=[scratch],
    )
    prim_func = _prim_func(
        tir,
        name="shared_block_alloc",
        params=[dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="shared_block_alloc"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("shared_block_alloc") @shared_block_alloc() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %scratch_bytes = index.constant 16 : offset
  %scratch_buffer = buffer.alloca %scratch_bytes {base_alignment = 4, memory_space = workgroup} : buffer
  %scratch = buffer.view %scratch_buffer[%c0_bytes] : buffer -> view<4xf32, %layout>
  %const = scalar.constant 1.0 : f32
  %c0 = index.constant 0 : index
  view.store %const, %scratch[%c0] : f32, view<4xf32, %layout>
  %load = view.load %scratch[%c0] : view<4xf32, %layout> -> f32
  view.store %load, %dst_view[%c0] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="private_block_alloc", category="op", tags=("alloc", "memory"))
def private_block_alloc(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "int32", name="dst")
    scratch = tir.decl_buffer((4,), "int32", name="scratch", scope="local")
    body = tir.Block(
        [],
        [],
        [],
        "root",
        tir.SeqStmt(
            [
                tir.BufferStore(
                    scratch,
                    tir.IntImm("int32", 7),
                    [tir.IntImm("int32", 0)],
                ),
                tir.BufferStore(
                    dst_buffer,
                    tir.BufferLoad(scratch, [tir.IntImm("int32", 0)]),
                    [tir.IntImm("int32", 0)],
                ),
            ]
        ),
        alloc_buffers=[scratch],
    )
    prim_func = _prim_func(
        tir,
        name="private_block_alloc",
        params=[dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="private_block_alloc"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("private_block_alloc") @private_block_alloc() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %scratch_bytes = index.constant 16 : offset
  %scratch_buffer = buffer.alloca %scratch_bytes {base_alignment = 4, memory_space = private} : buffer
  %scratch = buffer.view %scratch_buffer[%c0_bytes] : buffer -> view<4xi32, %layout>
  %const = scalar.constant 7 : i32
  %c0 = index.constant 0 : index
  view.store %const, %scratch[%c0] : i32, view<4xi32, %layout>
  view.store %const, %dst_view[%c0] : i32, view<4xi32, %layout>
  kernel.return
}
"""
