# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


# ====
def _make_group_count_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_group_count_kernel(
        num_topk: int,
        num_groups: int,
        num_sms: int,
    ) -> Any:
        num_threads = 128
        num_blocks = num_sms * 2
        num_tokens = T.dynamic("num_tokens")
        aligned_groups = _align_to(num_groups, num_threads)

        @T.prim_func  # type: ignore[untyped-decorator]
        def group_count_kernel(
            group_idx: T.Tensor[(num_tokens, num_topk), T.int64],
            out: T.Tensor[(num_groups,), T.int32],
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid * num_threads + thread_idx

                out_shared = T.alloc_shared((aligned_groups,), T.int32)
                for i in T.serial(thread_idx, aligned_groups, num_threads):
                    out_shared[i] = 0
                T.sync_threads()

                for i in T.serial(
                    global_thread_idx,
                    num_tokens,
                    num_blocks * num_threads,
                ):
                    for j in T.unroll(num_topk):
                        group = T.int32(group_idx[i, j])
                        T.assume(group < num_groups)
                        if group >= 0:
                            T.atomic_add(out_shared[group], 1)

                T.sync_threads()
                for i in T.serial(thread_idx, num_groups, num_threads):
                    if out_shared[i] > 0:
                        T.atomic_add(out[i], out_shared[i])

        return group_count_kernel

    return get_group_count_kernel


def _group_count_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_group_count_kernel(tilelang, T),
        args=(2, 8, 4),
        target=target,
        name="group_count_kernel",
    )


@tilelang_case(
    name="tilekernels_group_count_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "histogram", "amdgpu"),
)
def tilekernels_group_count_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _group_count_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("group_count_kernel") @group_count_kernel() {
  %c8 = index.constant 8 : index
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%c8, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%group_idx_handle: buffer, %out_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %group_idx_noalias = buffer.assume.noalias %group_idx_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %group_idx = buffer.view %group_idx_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi64, %layout>
  %out_noalias = buffer.assume.noalias %out_handle : buffer
  %out = buffer.view %out_noalias[%c0_bytes] : buffer -> view<8xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %out_shared_bytes = index.constant 512 : offset
  %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 4, memory_space = workgroup} : buffer
  %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128xi32, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %c0 = index.constant 0 : index
  %c255 = index.constant 255 : index
  %sub = index.sub %c255, %thread_idx : index
  %div = index.div %sub, %c128 : index
  %c1 = index.constant 1 : index
  scf.for %tmp = [%c0 to %div step %c1] {
    %madd_2 = index.madd %tmp, %c128, %thread_idx : index
    %const = scalar.constant 0 : i32
    view.store %const, %out_shared[%madd_2] : i32, view<128xi32, %layout>
  }
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c1023 = index.constant 1023 : index
  %add = index.add %num_tokens_idx, %c1023 : index
  %sub_2 = index.sub %add, %madd : index
  %c1024 = index.constant 1024 : index
  %div_2 = index.div %sub_2, %c1024 : index
  scf.for %tmp = [%c0 to %div_2 step %c1] {
    %madd_3 = index.madd %tmp, %c1024, %madd : index
    %c2 = index.constant 2 : index
    scf.for %j = [%c0 to %c2 step %c1] unroll {
      %load = view.load %group_idx[%madd_3, %j] : view<[%num_tokens_idx]x2xi64, %layout> -> i64
      %trunci = scalar.trunci %load : i64 to i32
      %group_assumed = scalar.assume %trunci [lt(%trunci, 8)] : i32
      %const_2 = scalar.constant 0 : i32
      %cmp = scalar.cmpi sge, %group_assumed, %const_2 : i32
      scf.if %cmp {
        %const_3 = scalar.constant 1 : i32
        %group_idx_2 = index.cast %group_assumed : i32 to index
        view.atomic.reduce<addi> %const_3, %out_shared[%group_idx_2] {ordering = relaxed, scope = workgroup} : i32, view<128xi32, %layout>
      }
    }
  }
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c135 = index.constant 135 : index
  %sub_3 = index.sub %c135, %thread_idx : index
  %div_3 = index.div %sub_3, %c128 : index
  scf.for %tmp = [%c0 to %div_3 step %c1] {
    %madd_4 = index.madd %tmp, %c128, %thread_idx : index
    %load_2 = view.load %out_shared[%madd_4] : view<128xi32, %layout> -> i32
    %const_4 = scalar.constant 0 : i32
    %cmp_2 = scalar.cmpi sgt, %load_2, %const_4 : i32
    scf.if %cmp_2 {
      %load_3 = view.load %out_shared[%madd_4] : view<128xi32, %layout> -> i32
      view.atomic.reduce<addi> %load_3, %out[%madd_4] {ordering = relaxed, scope = device} : i32, view<8xi32, %layout>
    }
  }
  kernel.return
}
"""


# ====
def _make_aux_fi_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_aux_fi_kernel(
        num_topk: int,
        num_experts: int,
        num_sms: int,
    ) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_blocks = num_sms * 2

        @T.prim_func  # type: ignore[untyped-decorator]
        def aux_fi_kernel(
            topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
            out: T.Tensor[(num_experts,), T.float32],
            num_aux_topk: T.int32,
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid * num_threads + thread_idx
                out_shared = T.alloc_shared(
                    (_align_to(num_experts, num_threads),), T.int32
                )
                for i in T.serial(
                    thread_idx,
                    _align_to(num_experts, num_threads),
                    num_threads,
                ):
                    out_shared[i] = 0
                T.sync_threads()

                for i in T.serial(
                    global_thread_idx,
                    num_tokens,
                    num_blocks * num_threads,
                ):
                    for j in T.unroll(num_topk):
                        expert_idx = T.int32(topk_idx[i, j])
                        T.assume(expert_idx < num_experts)
                        if expert_idx >= 0:
                            T.atomic_add(out_shared[expert_idx], 1)

                T.sync_threads()
                for i in T.serial(thread_idx, num_experts, num_threads):
                    if out_shared[i] > 0:
                        numerator = T.cast(out_shared[i] * num_experts, T.float32)
                        denominator = T.cast(num_tokens * num_aux_topk, T.float32)
                        T.atomic_add(out[i], numerator / denominator)

        return aux_fi_kernel

    return get_aux_fi_kernel


def _aux_fi_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_aux_fi_kernel(tilelang, T),
        args=(2, 8, 4),
        target=target,
        name="aux_fi_kernel",
    )


@tilelang_case(
    name="tilekernels_aux_fi_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "atomic", "amdgpu"),
)
def tilekernels_aux_fi_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _aux_fi_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("aux_fi_kernel") @aux_fi_kernel() {
  %c8 = index.constant 8 : index
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%c8, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%topk_idx_handle: buffer, %out_handle: buffer, %num_aux_topk: i32, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %topk_idx_noalias = buffer.assume.noalias %topk_idx_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %topk_idx = buffer.view %topk_idx_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi64, %layout>
  %out_noalias = buffer.assume.noalias %out_handle : buffer
  %out = buffer.view %out_noalias[%c0_bytes] : buffer -> view<8xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %out_shared_bytes = index.constant 512 : offset
  %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 4, memory_space = workgroup} : buffer
  %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128xi32, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %c0 = index.constant 0 : index
  %c255 = index.constant 255 : index
  %sub = index.sub %c255, %thread_idx : index
  %div = index.div %sub, %c128 : index
  %c1 = index.constant 1 : index
  scf.for %tmp = [%c0 to %div step %c1] {
    %madd_2 = index.madd %tmp, %c128, %thread_idx : index
    %const = scalar.constant 0 : i32
    view.store %const, %out_shared[%madd_2] : i32, view<128xi32, %layout>
  }
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c1023 = index.constant 1023 : index
  %add = index.add %num_tokens_idx, %c1023 : index
  %sub_2 = index.sub %add, %madd : index
  %c1024 = index.constant 1024 : index
  %div_2 = index.div %sub_2, %c1024 : index
  scf.for %tmp = [%c0 to %div_2 step %c1] {
    %madd_3 = index.madd %tmp, %c1024, %madd : index
    %c2 = index.constant 2 : index
    scf.for %j = [%c0 to %c2 step %c1] unroll {
      %load = view.load %topk_idx[%madd_3, %j] : view<[%num_tokens_idx]x2xi64, %layout> -> i64
      %trunci = scalar.trunci %load : i64 to i32
      %expert_idx_assumed = scalar.assume %trunci [lt(%trunci, 8)] : i32
      %const_2 = scalar.constant 0 : i32
      %cmp = scalar.cmpi sge, %expert_idx_assumed, %const_2 : i32
      scf.if %cmp {
        %const_3 = scalar.constant 1 : i32
        %expert_idx_idx = index.cast %expert_idx_assumed : i32 to index
        view.atomic.reduce<addi> %const_3, %out_shared[%expert_idx_idx] {ordering = relaxed, scope = workgroup} : i32, view<128xi32, %layout>
      }
    }
  }
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c135 = index.constant 135 : index
  %sub_3 = index.sub %c135, %thread_idx : index
  %div_3 = index.div %sub_3, %c128 : index
  scf.for %tmp = [%c0 to %div_3 step %c1] {
    %madd_4 = index.madd %tmp, %c128, %thread_idx : index
    %load_2 = view.load %out_shared[%madd_4] : view<128xi32, %layout> -> i32
    %const_4 = scalar.constant 0 : i32
    %cmp_2 = scalar.cmpi sgt, %load_2, %const_4 : i32
    scf.if %cmp_2 {
      %load_3 = view.load %out_shared[%madd_4] : view<128xi32, %layout> -> i32
      %const_5 = scalar.constant 8 : i32
      %muli = scalar.muli %load_3, %const_5 : i32
      %sitofp = scalar.sitofp %muli : i32 to f32
      %muli_2 = scalar.muli %num_tokens, %num_aux_topk : i32
      %sitofp_2 = scalar.sitofp %muli_2 : i32 to f32
      %divf = scalar.divf %sitofp, %sitofp_2 : f32
      view.atomic.reduce<addf> %divf, %out[%madd_4] {ordering = relaxed, scope = device} : f32, view<8xf32, %layout>
    }
  }
  kernel.return
}
"""
