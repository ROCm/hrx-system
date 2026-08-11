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
@tilelang_case(name="fp8_fn_buffer_views", category="op", tags=("fp8", "abi"))
def fp8_fn_buffer_views(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((8,), "float8_e4m3fn", name="src")
    dst_buffer = tir.decl_buffer((8,), "bfloat16", name="dst")
    prim_func = _prim_func(
        tir,
        name="fp8_fn_buffer_views",
        params=[src, dst],
        body=tir.Evaluate(tir.IntImm("int32", 0)),
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx9-4-generic",
        name="fp8_fn_buffer_views",
    )


# ----
r"""
amdgpu.target<gfx9-4-generic> @hip_mcpu_gfx9_4_generic

kernel.def target(@hip_mcpu_gfx9_4_generic) export("fp8_fn_buffer_views") @fp8_fn_buffer_views() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %f8e4m3fn_schema = encoding.define #encoding.f8e4m3fn : encoding<schema>
  %src_storage = encoding.define #encoding.storage {layout = %layout : encoding<layout>, schema = %f8e4m3fn_schema : encoding<schema>} : encoding<storage>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<8xf8E4M3, %src_storage>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<8xbf16, %layout>
  %const = scalar.constant 0 : i32
  kernel.return
}
"""


# ====
@tilelang_case(name="fp8_fnuz_buffer_views", category="op", tags=("fp8", "abi"))
def fp8_fnuz_buffer_views(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer((8,), "float8_e4m3fnuz", name="src")
    dst_buffer = tir.decl_buffer((8,), "float8_e5m2fnuz", name="dst")
    prim_func = _prim_func(
        tir,
        name="fp8_fnuz_buffer_views",
        params=[src, dst],
        body=tir.Evaluate(tir.IntImm("int32", 0)),
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx9-4-generic",
        name="fp8_fnuz_buffer_views",
    )


# ----
r"""
amdgpu.target<gfx9-4-generic> @hip_mcpu_gfx9_4_generic

kernel.def target(@hip_mcpu_gfx9_4_generic) export("fp8_fnuz_buffer_views") @fp8_fnuz_buffer_views() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %f8e4m3fnuz_schema = encoding.define #encoding.f8e4m3fnuz : encoding<schema>
  %src_storage = encoding.define #encoding.storage {layout = %layout : encoding<layout>, schema = %f8e4m3fnuz_schema : encoding<schema>} : encoding<storage>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<8xf8E4M3, %src_storage>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %f8e5m2fnuz_schema = encoding.define #encoding.f8e5m2fnuz : encoding<schema>
  %dst_storage = encoding.define #encoding.storage {layout = %layout : encoding<layout>, schema = %f8e5m2fnuz_schema : encoding<schema>} : encoding<storage>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<8xf8E5M2, %dst_storage>
  %const = scalar.constant 0 : i32
  kernel.return
}
"""


# ====
# ERROR@+1: "TileLang dtype `float8_e4m3fnuz` carries numeric-format semantics"
@tilelang_case(name="fp8_fnuz_scalar_load", category="diagnostic", tags=("fp8",))
def fp8_fnuz_scalar_load(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    src_buffer = tir.decl_buffer((8,), "float8_e4m3fnuz", name="src")
    body = tir.Evaluate(
        tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)]),
    )
    prim_func = _prim_func(
        tir,
        name="fp8_fnuz_scalar_load",
        params=[src],
        body=body,
        buffer_map={src: src_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx9-4-generic",
        name="fp8_fnuz_scalar_load",
    )


# ====
# ERROR@+1: "TileLang dtype `float8_e4m3fnuzx4` carries numeric-format semantics"
@tilelang_case(
    name="fp8_fnuz_vector_load",
    category="diagnostic",
    tags=("fp8", "vector"),
)
def fp8_fnuz_vector_load(tir: Any) -> TileLangImportInput:
    src = tir.Var("src", "handle")
    src_buffer = tir.decl_buffer((8,), "float8_e4m3fnuz", name="src")
    body = tir.Evaluate(tir.BufferLoad(src_buffer, [_ramp4(tir)]))
    prim_func = _prim_func(
        tir,
        name="fp8_fnuz_vector_load",
        params=[src],
        body=body,
        buffer_map={src: src_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx9-4-generic",
        name="fp8_fnuz_vector_load",
    )
