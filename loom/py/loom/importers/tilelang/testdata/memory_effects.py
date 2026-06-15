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


def _access_ptr(tir: Any, dst_buffer: Any) -> Any:
    return _call(
        tir,
        "handle",
        "tl.access_ptr",
        tir.BufferLoad(dst_buffer, [tir.IntImm("int32", 0)]),
        tir.IntImm("int32", 1),
        tir.IntImm("int32", 3),
    )


def _prim_func(tir: Any, *, name: str, body: Any, dst: Any, dst_buffer: Any) -> Any:
    return tir.PrimFunc([dst], body, buffer_map={dst: dst_buffer}).with_attr(
        "global_symbol", name
    )


# ====
@tilelang_case(name="scalar_atomic_add", category="op", tags=("memory",))
def scalar_atomic_add(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "int32", name="dst")
    body = tir.Evaluate(
        _call(
            tir,
            "handle",
            "tl.atomic_add_elem_op",
            _access_ptr(tir, dst_buffer),
            tir.IntImm("int32", 1),
        )
    )
    prim_func = _prim_func(
        tir,
        name="scalar_atomic_add",
        body=body,
        dst=dst,
        dst_buffer=dst_buffer,
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="scalar_atomic_add"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scalar_atomic_add") @scalar_atomic_add() {
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
@tilelang_case(name="scalar_atomic_add_return", category="op", tags=("memory",))
def scalar_atomic_add_return(tir: Any) -> TileLangImportInput:
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((4,), "int32", name="dst")
    body = tir.BufferStore(
        dst_buffer,
        _call(
            tir,
            "int32",
            "tl.atomic_add_ret_elem_op",
            _access_ptr(tir, dst_buffer),
            tir.IntImm("int32", 1),
            tir.IntImm("int32", 2),
        ),
        [tir.IntImm("int32", 1)],
    )
    prim_func = _prim_func(
        tir,
        name="scalar_atomic_add_return",
        body=body,
        dst=dst,
        dst_buffer=dst_buffer,
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="scalar_atomic_add_return",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scalar_atomic_add_return") @scalar_atomic_add_return() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %const = scalar.constant 1 : i32
  %c0 = index.constant 0 : index
  %atomic_add = view.atomic.rmw<addi> %const, %dst_view[%c0] {ordering = acquire, scope = device} : i32, view<4xi32, %layout> -> i32
  %c1 = index.constant 1 : index
  view.store %atomic_add, %dst_view[%c1] : i32, view<4xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="device_asserts", category="op", tags=("assert", "kernel"))
def device_asserts(tilelang: Any, T: Any) -> TileLangImportInput:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_kernel() -> Any:
        @T.prim_func  # type: ignore[untyped-decorator]
        def device_asserts_kernel(
            src: T.Tensor[(1,), T.float32],
            dst: T.Tensor[(1,), T.float32],
        ) -> None:
            with T.Kernel(1, threads=1):
                value = src[0]
                T.device_assert(T.isfinite(value))
                T.device_assert(T.isfinite(value), msg="finite input required")
                dst[0] = value

        return device_asserts_kernel

    return TileLangImportInput(
        source=get_kernel,
        target="hip -mcpu=gfx1100",
        name="device_asserts_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("device_asserts_kernel") @device_asserts_kernel() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %load = view.load %src[%c0] : view<1xf32, %layout> -> f32
  %isfinitef = scalar.isfinitef %load : f32
  kernel.assert %isfinitef "\n  at memory_effects.py:149 in device_asserts_kernel\n" : i1
  %isfinitef_2 = scalar.isfinitef %load : f32
  kernel.assert %isfinitef_2 "finite input required\n  at memory_effects.py:150 in device_asserts_kernel\n" : i1
  view.store %load, %dst[%c0] : f32, view<1xf32, %layout>
  kernel.return
}
"""
