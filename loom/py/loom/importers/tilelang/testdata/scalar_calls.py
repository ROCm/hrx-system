# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


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
@tilelang_case(name="scalar_calls", category="op", tags=("call", "scalar"))
def scalar_calls(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    load = tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])
    body = tir.IfThenElse(
        tir.isfinite(load),
        tir.BufferStore(
            dst_buffer,
            tir.sqrt(tir.abs(load)) + tir.sigmoid(load),
            [tir.IntImm("int32", 0)],
        ),
        tir.BufferStore(dst_buffer, tir.exp(load), [tir.IntImm("int32", 0)]),
    )
    prim_func = _prim_func(
        tir,
        name="scalar_calls",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="scalar_calls"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scalar_calls") @scalar_calls() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  %isfinitef = scalar.isfinitef %load : f32
  scf.if %isfinitef {
    %load_2 = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
    %absf = scalar.absf %load_2 : f32
    %sqrtf = scalar.sqrtf %absf : f32
    %load_3 = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
    %logisticf = scalar.logisticf %load_3 : f32
    %addf = scalar.addf %sqrtf, %logisticf : f32
    view.store %addf, %dst_view[%c0] : f32, view<4xf32, %layout>
  } else {
    %load_4 = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
    %expf = scalar.expf %load_4 : f32
    view.store %expf, %dst_view[%c0] : f32, view<4xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="scalar_rounding_calls", category="op", tags=("call", "scalar"))
def scalar_rounding_calls(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    load = tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])
    body = tir.SeqStmt(
        [
            tir.BufferStore(
                dst_buffer,
                tir.call_intrin("float32", "tir.round", load),
                [tir.IntImm("int32", 0)],
            ),
            tir.BufferStore(
                dst_buffer,
                tir.call_intrin("float32", "tir.nearbyint", load),
                [tir.IntImm("int32", 1)],
            ),
        ]
    )
    prim_func = _prim_func(
        tir,
        name="scalar_rounding_calls",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="scalar_rounding_calls",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scalar_rounding_calls") @scalar_rounding_calls() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  %roundf = scalar.roundf %load : f32
  view.store %roundf, %dst_view[%c0] : f32, view<4xf32, %layout>
  %load_2 = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  %roundevenf = scalar.roundevenf %load_2 : f32
  %c1 = index.constant 1 : index
  view.store %roundevenf, %dst_view[%c1] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="scalar_fastmath_quantizer_math",
    category="op",
    tags=("call", "scalar", "fastmath"),
)
def scalar_fastmath_quantizer_math(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir)
    value = tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])
    divisor = tir.BufferLoad(src_buffer, [tir.IntImm("int32", 1)])
    rounded = tir.call_intrin("float32", "tir.round", value / divisor)
    body = tir.BufferStore(dst_buffer, rounded, [tir.IntImm("int32", 0)])
    prim_func = _prim_func(
        tir,
        name="scalar_fastmath_quantizer_math",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    ).with_attr("tilelang_pass_configs", {"tl.enable_fast_math": True})
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="scalar_fastmath_quantizer_math",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("scalar_fastmath_quantizer_math") @scalar_fastmath_quantizer_math() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  %c1 = index.constant 1 : index
  %load_2 = view.load %src_view[%c1] : view<4xf32, %layout> -> f32
  %divf = scalar.divf<reassoc|nnan|ninf|nsz|arcp|contract|afn> %load, %load_2 : f32
  %roundf = scalar.roundf<reassoc|nnan|ninf|nsz|arcp|contract|afn> %divf : f32
  view.store %roundf, %dst_view[%c0] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="bitwise_not", category="op", tags=("call", "integer"))
def bitwise_not(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, dtype="int32")
    body = tir.BufferStore(
        dst_buffer,
        tir.bitwise_not(tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="bitwise_not",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="bitwise_not"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("bitwise_not") @bitwise_not() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi32, %layout>
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xi32, %layout> -> i32
  %all_ones = scalar.constant -1 : i32
  %noti = scalar.xori %load, %all_ones : i32
  view.store %noti, %dst_view[%c0] : i32, view<4xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="signed_floormod_power_of_two",
    category="op",
    tags=("call", "integer"),
)
def signed_floormod_power_of_two(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, dtype="int64")
    load = tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)])
    body = tir.BufferStore(
        dst_buffer,
        load % tir.IntImm("int64", 64),
        [tir.IntImm("int32", 0)],
    )
    prim_func = _prim_func(
        tir,
        name="signed_floormod_power_of_two",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="signed_floormod_power_of_two",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("signed_floormod_power_of_two") @signed_floormod_power_of_two() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xi64, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xi64, %layout>
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xi64, %layout> -> i64
  %floor_mod_mask = scalar.constant 63 : i64
  %floor_mod = scalar.andi %load, %floor_mod_mask : i64
  view.store %floor_mod, %dst_view[%c0] : i64, view<4xi64, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="dynamic_loop_bound", category="op", tags=("index", "loop"))
def dynamic_loop_bound(tir: Any) -> TileLangImportInput:
    n = tir.Var("n", "int32")
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(8,))
    index = tir.Var("i", "int32")
    body = tir.For(
        index,
        tir.IntImm("int32", 0),
        n,
        tir.ForKind.SERIAL,
        tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [index]), [index]),
    )
    prim_func = _prim_func(
        tir,
        name="dynamic_loop_bound",
        params=[n, src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="dynamic_loop_bound",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("dynamic_loop_bound") @dynamic_loop_bound() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%n: i32, %src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<8xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<8xf32, %layout>
  %c0 = index.constant 0 : index
  %n_idx = index.cast %n : i32 to index
  %c1 = index.constant 1 : index
  scf.for %i = [%c0 to %n_idx step %c1] {
    %load = view.load %src_view[%i] : view<8xf32, %layout> -> f32
    view.store %load, %dst_view[%i] : f32, view<8xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(name="ceildiv_loop", category="op", tags=("call", "index"))
def ceildiv_loop(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(8,))
    index = tir.Var("i", "int32")
    body = tir.For(
        index,
        tir.IntImm("int32", 0),
        tir.ceildiv(tir.IntImm("int32", 7), tir.IntImm("int32", 4)),
        tir.ForKind.SERIAL,
        tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [index]), [index]),
    )
    prim_func = _prim_func(
        tir,
        name="ceildiv_loop",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="ceildiv_loop"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("ceildiv_loop") @ceildiv_loop() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<8xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<8xf32, %layout>
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c1 = index.constant 1 : index
  scf.for %i = [%c0 to %c2 step %c1] {
    %load = view.load %src_view[%i] : view<8xf32, %layout> -> f32
    view.store %load, %dst_view[%i] : f32, view<8xf32, %layout>
  }
  kernel.return
}
"""
