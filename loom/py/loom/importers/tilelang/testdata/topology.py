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
    shape: tuple[int, ...],
) -> tuple[Any, Any, Any, Any]:
    src = tir.Var("src", "handle")
    dst = tir.Var("dst", "handle")
    src_buffer = tir.decl_buffer(shape, "float32", name="src")
    dst_buffer = tir.decl_buffer(shape, "float32", name="dst")
    return src, dst, src_buffer, dst_buffer


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


# ====
@tilelang_case(
    name="call_extern_match_any_sync",
    category="op",
    tags=("topology", "subgroup", "call"),
)
def call_extern_match_any_sync(tir: Any, tvm: Any) -> TileLangImportInput:
    value = tir.Var("value", "int32")
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((32,), "uint32", name="dst")
    thread_index = tvm.te.thread_axis("threadIdx.x")
    mask = tir.call_extern(
        "uint32",
        "__match_any_sync",
        tir.IntImm("uint32", 0xFFFFFFFF),
        value,
    )
    count = tir.call_intrin("uint32", "tir.popcount", mask)
    body = tir.AttrStmt(
        thread_index,
        "thread_extent",
        tir.IntImm("int32", 32),
        tir.BufferStore(dst_buffer, count, [thread_index.var]),
    )
    prim_func = _prim_func(
        tir,
        name="call_extern_match_any_sync",
        params=[value, dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="call_extern_match_any_sync",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("call_extern_match_any_sync") @call_extern_match_any_sync() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%value: i32, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<32xi32, %layout>
  %threadIdx_x = kernel.workitem.id<x> : index
  %match_any = kernel.subgroup.match.any %value : i32 -> i32
  %ctpopi = scalar.ctpopi %match_any : i32
  view.store %ctpopi, %dst_view[%threadIdx_x] : i32, view<32xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tilelang_match_any_sync",
    category="op",
    tags=("topology", "subgroup", "call"),
)
def tilelang_match_any_sync(tir: Any, tvm: Any) -> TileLangImportInput:
    value = tir.Var("value", "int32")
    dst = tir.Var("dst", "handle")
    dst_buffer = tir.decl_buffer((32,), "uint32", name="dst")
    thread_index = tvm.te.thread_axis("threadIdx.x")
    mask = tir.call_intrin(
        "uint32",
        tir.op.Op.get("tl.match_any_sync"),
        tir.IntImm("uint32", 0xFFFFFFFF),
        value,
    )
    count = tir.call_intrin("uint32", "tir.popcount", mask)
    body = tir.AttrStmt(
        thread_index,
        "thread_extent",
        tir.IntImm("int32", 32),
        tir.BufferStore(dst_buffer, count, [thread_index.var]),
    )
    prim_func = _prim_func(
        tir,
        name="tilelang_match_any_sync",
        params=[value, dst],
        body=body,
        buffer_map={dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func,
        target="hip -mcpu=gfx1100",
        name="tilelang_match_any_sync",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("tilelang_match_any_sync") @tilelang_match_any_sync() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%value: i32, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<32xi32, %layout>
  %threadIdx_x = kernel.workitem.id<x> : index
  %match_any = kernel.subgroup.match.any %value : i32 -> i32
  %ctpopi = scalar.ctpopi %match_any : i32
  view.store %ctpopi, %dst_view[%threadIdx_x] : i32, view<32xi32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="launch_thread_attrs", category="op", tags=("topology",))
def launch_thread_attrs(tir: Any, tvm: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(1024,))
    bx = tvm.te.thread_axis("blockIdx.x")
    tx = tvm.te.thread_axis("threadIdx.x")
    index = bx.var + bx.var + tx.var + tx.var
    body = tir.AttrStmt(
        bx,
        "thread_extent",
        tir.IntImm("int32", 8),
        tir.AttrStmt(
            tx,
            "thread_extent",
            tir.IntImm("int32", 64),
            tir.BufferStore(
                dst_buffer,
                tir.BufferLoad(src_buffer, [index]),
                [index],
            ),
        ),
    )
    prim_func = _prim_func(
        tir,
        name="launch_thread_attrs",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="launch_thread_attrs"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("launch_thread_attrs") @launch_thread_attrs() {
  %c8 = index.constant 8 : index
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%c8, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<1024xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<1024xf32, %layout>
  %blockIdx_x = kernel.workgroup.id<x> : index
  %threadIdx_x = kernel.workitem.id<x> : index
  %add = index.add %blockIdx_x, %blockIdx_x : index
  %add_2 = index.add %add, %threadIdx_x : index
  %add_3 = index.add %add_2, %threadIdx_x : index
  %load = view.load %src_view[%add_3] : view<1024xf32, %layout> -> f32
  %add_4 = index.add %blockIdx_x, %blockIdx_x : index
  %add_5 = index.add %add_4, %threadIdx_x : index
  %add_6 = index.add %add_5, %threadIdx_x : index
  view.store %load, %dst_view[%add_6] : f32, view<1024xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="shared_storage_sync", category="op", tags=("topology", "memory"))
def shared_storage_sync(tir: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(4,))
    body = tir.SeqStmt(
        [
            tir.Evaluate(_call(tir, "int32", "tir.tvm_storage_sync", "shared")),
            tir.BufferStore(
                dst_buffer,
                tir.BufferLoad(src_buffer, [tir.IntImm("int32", 0)]),
                [tir.IntImm("int32", 0)],
            ),
        ]
    )
    prim_func = _prim_func(
        tir,
        name="shared_storage_sync",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="shared_storage_sync"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("shared_storage_sync") @shared_storage_sync() {
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<4xf32, %layout>
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c0 = index.constant 0 : index
  %load = view.load %src_view[%c0] : view<4xf32, %layout> -> f32
  view.store %load, %dst_view[%c0] : f32, view<4xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="warp_sync",
    category="op",
    tags=("topology", "subgroup", "barrier"),
)
def warp_sync(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def warp_sync_kernel(
        src: T.Tensor[(32,), T.float32],
        dst: T.Tensor[(32,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=32):
            thread_index = T.get_thread_binding()
            dst[thread_index] = src[thread_index]
            T.sync_warp()

    return TileLangImportInput(
        source=warp_sync_kernel,
        target="hip -mcpu=gfx1100",
        name="warp_sync_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("warp_sync_kernel") @warp_sync_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<32xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<32xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %load = view.load %src[%thread_index] : view<32xf32, %layout> -> f32
  view.store %load, %dst[%thread_index] : f32, view<32xf32, %layout>
  kernel.barrier<workgroup> {ordering = acq_rel, scope = subgroup}
  kernel.return
}
"""


# ====
@tilelang_case(name="thread_binding_loop", category="op", tags=("topology",))
def thread_binding_loop(tir: Any, tvm: Any) -> TileLangImportInput:
    src, dst, src_buffer, dst_buffer = _buffer_pair(tir, shape=(128,))
    tx = tir.Var("tx", "int32")
    body = tir.For(
        tx,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 128),
        tir.ForKind.THREAD_BINDING,
        tir.BufferStore(dst_buffer, tir.BufferLoad(src_buffer, [tx]), [tx]),
        thread_binding=tvm.te.thread_axis("threadIdx.x"),
    )
    prim_func = _prim_func(
        tir,
        name="thread_binding_loop",
        params=[src, dst],
        body=body,
        buffer_map={src: src_buffer, dst: dst_buffer},
    )
    return TileLangImportInput(
        source=prim_func, target="hip -mcpu=gfx1100", name="thread_binding_loop"
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("thread_binding_loop") @thread_binding_loop() {
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%src: buffer, %dst: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src_view = buffer.view %src_noalias[%c0_bytes] : buffer -> view<128xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst : buffer
  %dst_view = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<128xf32, %layout>
  %tx = kernel.workitem.id<x> : index
  %load = view.load %src_view[%tx] : view<128xf32, %layout> -> f32
  view.store %load, %dst_view[%tx] : f32, view<128xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(name="thread_index_to_i64", category="op", tags=("topology", "cast"))
def thread_index_to_i64(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def thread_index_to_i64_kernel(dst: T.Tensor[(32,), T.int64]) -> None:
        with T.Kernel(1, threads=32):
            thread_index = T.get_thread_binding()
            dst[thread_index] = T.cast(thread_index, T.int64)

    return TileLangImportInput(
        source=thread_index_to_i64_kernel,
        target="hip -mcpu=gfx1100",
        name="thread_index_to_i64_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("thread_index_to_i64_kernel") @thread_index_to_i64_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<32xi64, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %cast = index.cast %thread_index : index to i64
  view.store %cast, %dst[%thread_index] : i64, view<32xi64, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="warp_shuffle",
    category="op",
    tags=("topology", "subgroup", "shuffle"),
)
def warp_shuffle(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def warp_shuffle_kernel(
        src: T.Tensor[(32,), T.float32],
        dst: T.Tensor[(128,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=32):
            thread_index = T.get_thread_binding()
            value = src[thread_index]
            dst[thread_index] = T.shfl_xor(value, 1)
            dst[thread_index + 32] = T.shfl_down(value, 1)
            dst[thread_index + 64] = T.shfl_up(value, 1)
            dst[thread_index + 96] = T.shfl_sync(value, 0)

    return TileLangImportInput(
        source=warp_shuffle_kernel,
        target="hip -mcpu=gfx1100",
        name="warp_shuffle_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("warp_shuffle_kernel") @warp_shuffle_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<32xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<128xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %load = view.load %src[%thread_index] : view<32xf32, %layout> -> f32
  %const = scalar.constant 1 : i32
  %const_2 = scalar.constant 32 : i32
  %shuffle, %shuffle_valid = kernel.subgroup.shuffle<xor> %load, %const, %const_2 : f32, i32, i32
  view.store %shuffle, %dst[%thread_index] : f32, view<128xf32, %layout>
  %shuffle_2, %shuffle_valid_2 = kernel.subgroup.shuffle<down> %load, %const, %const_2 : f32, i32, i32
  %c32 = index.constant 32 : index
  %add = index.add %thread_index, %c32 : index
  view.store %shuffle_2, %dst[%add] : f32, view<128xf32, %layout>
  %shuffle_3, %shuffle_valid_3 = kernel.subgroup.shuffle<up> %load, %const, %const_2 : f32, i32, i32
  %c64 = index.constant 64 : index
  %add_2 = index.add %thread_index, %c64 : index
  view.store %shuffle_3, %dst[%add_2] : f32, view<128xf32, %layout>
  %const_3 = scalar.constant 0 : i32
  %shuffle_4, %shuffle_valid_4 = kernel.subgroup.shuffle<index> %load, %const_3, %const_2 : f32, i32, i32
  %c96 = index.constant 96 : index
  %add_3 = index.add %thread_index, %c96 : index
  view.store %shuffle_4, %dst[%add_3] : f32, view<128xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="warp_shuffle_loop_offsets",
    category="op",
    tags=("topology", "subgroup", "shuffle", "cast"),
)
def warp_shuffle_loop_offsets(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def warp_shuffle_loop_offsets_kernel(
        src: T.Tensor[(32,), T.float32],
        dst: T.Tensor[(128,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=32):
            thread_index = T.get_thread_binding()
            value = src[thread_index]
            for i in T.unroll(2):
                dst[thread_index + i * 32] = T.shfl_sync(value, i)
                dst[thread_index + 64 + i * 32] = T.shfl_xor(value, 1 << (4 - i))

    return TileLangImportInput(
        source=warp_shuffle_loop_offsets_kernel,
        target="hip -mcpu=gfx1100",
        name="warp_shuffle_loop_offsets_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("warp_shuffle_loop_offsets_kernel") @warp_shuffle_loop_offsets_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<32xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<128xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %load = view.load %src[%thread_index] : view<32xf32, %layout> -> f32
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c1 = index.constant 1 : index
  scf.for %i = [%c0 to %c2 step %c1] unroll {
    %const = scalar.constant 32 : i32
    %shuffle_offset = index.cast %i : index to i32
    %shuffle, %shuffle_valid = kernel.subgroup.shuffle<index> %load, %shuffle_offset, %const : f32, i32, i32
    %c32 = index.constant 32 : index
    %madd = index.madd %i, %c32, %thread_index : index
    view.store %shuffle, %dst[%madd] : f32, view<128xf32, %layout>
    %const_2 = scalar.constant 1 : i32
    %c4 = index.constant 4 : index
    %sub = index.sub %c4, %i : index
    %rhs_cast = index.cast %sub : index to i32
    %shli = scalar.shli %const_2, %rhs_cast : i32
    %shuffle_2, %shuffle_valid_2 = kernel.subgroup.shuffle<xor> %load, %shli, %const : f32, i32, i32
    %c64 = index.constant 64 : index
    %add = index.add %thread_index, %c64 : index
    %madd_2 = index.madd %i, %c32, %add : index
    view.store %shuffle_2, %dst[%madd_2] : f32, view<128xf32, %layout>
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="warp_reduce_float",
    category="op",
    tags=("topology", "subgroup", "reduce"),
)
def warp_reduce_float(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def warp_reduce_float_kernel(
        src: T.Tensor[(32,), T.float32],
        dst: T.Tensor[(96,), T.float32],
    ) -> None:
        with T.Kernel(1, threads=32):
            thread_index = T.get_thread_binding()
            value = src[thread_index]
            dst[thread_index] = T.warp_reduce_sum(value)
            dst[thread_index + 32] = T.warp_reduce_max(value)
            dst[thread_index + 64] = T.warp_reduce_min(value)

    return TileLangImportInput(
        source=warp_reduce_float_kernel,
        target="hip -mcpu=gfx1100",
        name="warp_reduce_float_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("warp_reduce_float_kernel") @warp_reduce_float_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<32xf32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<96xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %load = view.load %src[%thread_index] : view<32xf32, %layout> -> f32
  %warp_reduce = kernel.subgroup.reduce<addf> %load : f32
  view.store %warp_reduce, %dst[%thread_index] : f32, view<96xf32, %layout>
  %warp_reduce_2 = kernel.subgroup.reduce<maxnumf> %load : f32
  %c32 = index.constant 32 : index
  %add = index.add %thread_index, %c32 : index
  view.store %warp_reduce_2, %dst[%add] : f32, view<96xf32, %layout>
  %warp_reduce_3 = kernel.subgroup.reduce<minnumf> %load : f32
  %c64 = index.constant 64 : index
  %add_2 = index.add %thread_index, %c64 : index
  view.store %warp_reduce_3, %dst[%add_2] : f32, view<96xf32, %layout>
  kernel.return
}
"""


# ====
@tilelang_case(
    name="warp_reduce_integer",
    category="op",
    tags=("topology", "subgroup", "reduce"),
)
def warp_reduce_integer(T: Any) -> TileLangImportInput:
    @T.prim_func  # type: ignore[untyped-decorator]
    def warp_reduce_integer_kernel(
        src: T.Tensor[(32,), T.int32],
        dst: T.Tensor[(160,), T.int32],
        unsigned_src: T.Tensor[(32,), T.uint32],
        unsigned_dst: T.Tensor[(64,), T.uint32],
    ) -> None:
        with T.Kernel(1, threads=32):
            thread_index = T.get_thread_binding()
            value = src[thread_index]
            dst[thread_index] = T.warp_reduce_sum(value)
            dst[thread_index + 32] = T.warp_reduce_max(value)
            dst[thread_index + 64] = T.warp_reduce_min(value)
            dst[thread_index + 96] = T.warp_reduce_bitand(value)
            dst[thread_index + 128] = T.warp_reduce_bitor(value)
            unsigned_value = unsigned_src[thread_index]
            unsigned_dst[thread_index] = T.warp_reduce_max(unsigned_value)
            unsigned_dst[thread_index + 32] = T.warp_reduce_min(unsigned_value)

    return TileLangImportInput(
        source=warp_reduce_integer_kernel,
        target="hip -mcpu=gfx1100",
        name="warp_reduce_integer_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("warp_reduce_integer_kernel") @warp_reduce_integer_kernel() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%src_handle: buffer, %dst_handle: buffer, %unsigned_src_handle: buffer, %unsigned_dst_handle: buffer) {
  %c0_bytes = index.constant 0 : offset
  %src_noalias = buffer.assume.noalias %src_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %src = buffer.view %src_noalias[%c0_bytes] : buffer -> view<32xi32, %layout>
  %dst_noalias = buffer.assume.noalias %dst_handle : buffer
  %dst = buffer.view %dst_noalias[%c0_bytes] : buffer -> view<160xi32, %layout>
  %unsigned_src_noalias = buffer.assume.noalias %unsigned_src_handle : buffer
  %unsigned_src = buffer.view %unsigned_src_noalias[%c0_bytes] : buffer -> view<32xi32, %layout>
  %unsigned_dst_noalias = buffer.assume.noalias %unsigned_dst_handle : buffer
  %unsigned_dst = buffer.view %unsigned_dst_noalias[%c0_bytes] : buffer -> view<64xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_index = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %load = view.load %src[%thread_index] : view<32xi32, %layout> -> i32
  %warp_reduce = kernel.subgroup.reduce<addi> %load : i32
  view.store %warp_reduce, %dst[%thread_index] : i32, view<160xi32, %layout>
  %warp_reduce_2 = kernel.subgroup.reduce<maxsi> %load : i32
  %c32 = index.constant 32 : index
  %add = index.add %thread_index, %c32 : index
  view.store %warp_reduce_2, %dst[%add] : i32, view<160xi32, %layout>
  %warp_reduce_3 = kernel.subgroup.reduce<minsi> %load : i32
  %c64 = index.constant 64 : index
  %add_2 = index.add %thread_index, %c64 : index
  view.store %warp_reduce_3, %dst[%add_2] : i32, view<160xi32, %layout>
  %warp_reduce_4 = kernel.subgroup.reduce<andi> %load : i32
  %c96 = index.constant 96 : index
  %add_3 = index.add %thread_index, %c96 : index
  view.store %warp_reduce_4, %dst[%add_3] : i32, view<160xi32, %layout>
  %warp_reduce_5 = kernel.subgroup.reduce<ori> %load : i32
  %c128 = index.constant 128 : index
  %add_4 = index.add %thread_index, %c128 : index
  view.store %warp_reduce_5, %dst[%add_4] : i32, view<160xi32, %layout>
  %load_2 = view.load %unsigned_src[%thread_index] : view<32xi32, %layout> -> i32
  %warp_reduce_6 = kernel.subgroup.reduce<maxui> %load_2 : i32
  view.store %warp_reduce_6, %unsigned_dst[%thread_index] : i32, view<64xi32, %layout>
  %warp_reduce_7 = kernel.subgroup.reduce<minui> %load_2 : i32
  %add_5 = index.add %thread_index, %c32 : index
  view.store %warp_reduce_7, %unsigned_dst[%add_5] : i32, view<64xi32, %layout>
  kernel.return
}
"""
