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
def _make_mask_indices_by_tp_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_mask_indices_by_tp_kernel(num_topk: int, dtype: T.dtype) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_blocks = T.ceildiv(num_tokens * num_topk, num_threads)

        @T.prim_func  # type: ignore[untyped-decorator]
        def mask_indices_by_tp_kernel(
            indices: T.Tensor[(num_tokens, num_topk), dtype],
            masked_indices: T.Tensor[(num_tokens, num_topk), dtype],
            per_gpu: T.int32,
            per_dp: T.int32,
            num_tp_ranks: T.int32,
            tp_rank: T.int32,
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                indices_1d = T.reshape(indices, (num_tokens * num_topk,))
                masked_indices_1d = T.reshape(masked_indices, (num_tokens * num_topk,))
                thread_idx = T.get_thread_binding()
                index = pid * num_threads + thread_idx
                value = T.alloc_var(dtype)
                if index < num_tokens * num_topk:
                    value = indices_1d[index]
                    if (
                        value < 0
                        or T.truncmod(T.truncdiv(value, per_gpu), num_tp_ranks)
                        != tp_rank
                    ):
                        masked_indices_1d[index] = -1
                    else:
                        value -= tp_rank * per_gpu
                        dp_rank = T.truncdiv(value, per_dp)
                        value -= dp_rank * (per_dp - per_gpu)
                        masked_indices_1d[index] = T.Select(
                            value < 0,
                            T.int64(-1),
                            value,
                        )

        return mask_indices_by_tp_kernel

    return get_mask_indices_by_tp_kernel


def _mask_indices_by_tp_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_mask_indices_by_tp_kernel(tilelang, T),
        args=(2, T.int64),
        target=target,
        name="mask_indices_by_tp_kernel",
    )


@tilelang_case(
    name="tilekernels_mask_indices_by_tp_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "routing", "amdgpu"),
)
def tilekernels_mask_indices_by_tp_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _mask_indices_by_tp_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mask_indices_by_tp_kernel") @mask_indices_by_tp_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c2 = index.constant 2 : index
  %c128 = index.constant 128 : index
  %madd = index.madd %num_tokens_idx, %c2, %c128 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %madd, %c1 : index
  %div = index.div %sub, %c128 : index
  kernel.launch.config workgroups(%div, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%indices_handle: buffer, %masked_indices_handle: buffer, %per_gpu: i32, %per_dp: i32, %num_tp_ranks: i32, %tp_rank: i32, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %indices_noalias = buffer.assume.noalias %indices_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %indices = buffer.view %indices_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi64, %layout>
  %masked_indices_noalias = buffer.assume.noalias %masked_indices_handle : buffer
  %masked_indices = buffer.view %masked_indices_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi64, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %value_bytes = index.constant 8 : offset
  %value_buffer = buffer.alloca %value_bytes {base_alignment = 8, memory_space = private} : buffer
  %value = buffer.view %value_buffer[%c0_bytes] : buffer -> view<1xi64, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %c2 = index.constant 2 : index
  %mul = index.mul %num_tokens_idx, %c2 : index
  %cmp = index.cmp slt, %madd, %mul : index
  scf.if %cmp {
    %rem = index.rem %madd, %c2 : index
    %div = index.div %madd, %c2 : index
    %load = view.load %indices[%div, %rem] : view<[%num_tokens_idx]x2xi64, %layout> -> i64
    %c0 = index.constant 0 : index
    %const = scalar.constant 0 : i64
    %cmp_2 = scalar.cmpi slt, %load, %const : i64
    %extsi = scalar.extsi %per_gpu : i32 to i64
    %divsi = scalar.divsi %load, %extsi : i64
    %extsi_2 = scalar.extsi %num_tp_ranks : i32 to i64
    %remsi = scalar.remsi %divsi, %extsi_2 : i64
    %extsi_3 = scalar.extsi %tp_rank : i32 to i64
    %cmp_3 = scalar.cmpi ne, %remsi, %extsi_3 : i64
    %ori = scalar.ori %cmp_2, %cmp_3 : i1
    %value_state_if = scf.if %ori -> (i64) {
      %const_2 = scalar.constant -1 : i64
      %rem_2 = index.rem %madd, %c2 : index
      %div_2 = index.div %madd, %c2 : index
      view.store %const_2, %masked_indices[%div_2, %rem_2] : i64, view<[%num_tokens_idx]x2xi64, %layout>
      scf.yield %load : i64
    } else {
      %muli = scalar.muli %tp_rank, %per_gpu : i32
      %extsi_4 = scalar.extsi %muli : i32 to i64
      %subi = scalar.subi %load, %extsi_4 : i64
      %extsi_5 = scalar.extsi %per_dp : i32 to i64
      %divsi_2 = scalar.divsi %subi, %extsi_5 : i64
      %subi_2 = scalar.subi %per_dp, %per_gpu : i32
      %extsi_6 = scalar.extsi %subi_2 : i32 to i64
      %muli_2 = scalar.muli %divsi_2, %extsi_6 : i64
      %subi_3 = scalar.subi %subi, %muli_2 : i64
      %cmp_4 = scalar.cmpi slt, %subi_3, %const : i64
      %const_3 = scalar.constant -1 : i64
      %select = scf.select %cmp_4, %const_3, %subi_3 : i64
      %rem_3 = index.rem %madd, %c2 : index
      %div_3 = index.div %madd, %c2 : index
      view.store %select, %masked_indices[%div_3, %rem_3] : i64, view<[%num_tokens_idx]x2xi64, %layout>
      scf.yield %subi_3 : i64
    }
  }
  kernel.return
}
"""


# ====
def _make_inplace_unique_group_indices_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
        },
    )
    def get_inplace_unique_group_indices_kernel(
        num_topk: int,
        num_groups_aligned: int,
        num_sms: int,
    ) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        grid_x = num_sms * 2

        @T.prim_func  # type: ignore[untyped-decorator]
        def inplace_unique_group_indices_kernel(
            group_indices: T.Tensor[(num_tokens, num_topk), T.int64],
        ) -> None:
            with T.Kernel(grid_x, threads=num_threads) as (pid_token,):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid_token * num_threads + thread_idx
                group_sel = T.alloc_local((2,), T.uint64)

                for i in T.serial(
                    global_thread_idx,
                    num_tokens,
                    grid_x * num_threads,
                ):
                    for j in T.unroll(num_groups_aligned // 64):
                        group_sel[j] = 0
                    for j in T.unroll(num_topk):
                        group_idx = group_indices[i, j]
                        T.assume(group_idx < num_groups_aligned)
                        mask = T.Select(
                            group_idx >= 0,
                            T.uint64(1) << (group_idx % 64),
                            T.uint64(0),
                        )
                        lo_mask = T.Select(group_idx < 64, mask, T.uint64(0))
                        hi_mask = T.Select(group_idx >= 64, mask, T.uint64(0))
                        found = (lo_mask & group_sel[0]) | (hi_mask & group_sel[1])
                        group_sel[0] |= lo_mask
                        group_sel[1] |= hi_mask
                        if found:
                            group_indices[i, j] = -1

        return inplace_unique_group_indices_kernel

    return get_inplace_unique_group_indices_kernel


def _inplace_unique_group_indices_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_inplace_unique_group_indices_kernel(tilelang, T),
        args=(4, _align_to(8, 64), 4),
        target=target,
        name="inplace_unique_group_indices_kernel",
    )


@tilelang_case(
    name="tilekernels_inplace_unique_group_indices_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "dedupe", "amdgpu"),
)
def tilekernels_inplace_unique_group_indices_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _inplace_unique_group_indices_input(
        tilelang,
        T,
        target="hip -mcpu=gfx1100",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("inplace_unique_group_indices_kernel") @inplace_unique_group_indices_kernel() {
  %c8 = index.constant 8 : index
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%c8, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%group_indices_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %group_indices_noalias = buffer.assume.noalias %group_indices_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %group_indices = buffer.view %group_indices_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x4xi64, %layout>
  %bx = kernel.workgroup.id<x> : index
  %thread_idx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %group_sel_bytes = index.constant 16 : offset
  %group_sel_buffer = buffer.alloca %group_sel_bytes {base_alignment = 8, memory_space = private} : buffer
  %group_sel = buffer.view %group_sel_buffer[%c0_bytes] : buffer -> view<2xi64, %layout>
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %thread_idx : index
  %c0 = index.constant 0 : index
  %c1023 = index.constant 1023 : index
  %add = index.add %num_tokens_idx, %c1023 : index
  %sub = index.sub %add, %madd : index
  %c1024 = index.constant 1024 : index
  %div = index.div %sub, %c1024 : index
  %c1 = index.constant 1 : index
  scf.for %tmp = [%c0 to %div step %c1] {
    %madd_2 = index.madd %tmp, %c1024, %madd : index
    scf.for %j = [%c0 to %c1 step %c1] unroll {
      %const = scalar.constant 0 : i64
      view.store %const, %group_sel[%j] : i64, view<2xi64, %layout>
    }
    %c4 = index.constant 4 : index
    scf.for %j = [%c0 to %c4 step %c1] unroll {
      %load = view.load %group_indices[%madd_2, %j] : view<[%num_tokens_idx]x4xi64, %layout> -> i64
      %group_idx_assumed = scalar.assume %load [lt(%load, 64)] : i64
      %const_2 = scalar.constant 0 : i64
      %cmp = scalar.cmpi sge, %group_idx_assumed, %const_2 : i64
      %const_3 = scalar.constant 1 : i64
      %const_4 = scalar.constant 64 : i64
      %remsi = scalar.remsi %group_idx_assumed, %const_4 : i64
      %shli = scalar.shli %const_3, %remsi : i64
      %select = scf.select %cmp, %shli, %const_2 : i64
      %cmp_2 = scalar.cmpi slt, %group_idx_assumed, %const_4 : i64
      %select_2 = scf.select %cmp_2, %select, %const_2 : i64
      %cmp_3 = scalar.cmpi sge, %group_idx_assumed, %const_4 : i64
      %select_3 = scf.select %cmp_3, %select, %const_2 : i64
      %load_2 = view.load %group_sel[%c0] : view<2xi64, %layout> -> i64
      %andi = scalar.andi %select_2, %load_2 : i64
      %load_3 = view.load %group_sel[%c1] : view<2xi64, %layout> -> i64
      %andi_2 = scalar.andi %select_3, %load_3 : i64
      %ori = scalar.ori %andi, %andi_2 : i64
      %ori_2 = scalar.ori %load_2, %select_2 : i64
      view.store %ori_2, %group_sel[%c0] : i64, view<2xi64, %layout>
      %load_4 = view.load %group_sel[%c1] : view<2xi64, %layout> -> i64
      %ori_3 = scalar.ori %load_4, %select_3 : i64
      view.store %ori_3, %group_sel[%c1] : i64, view<2xi64, %layout>
      %cmp_4 = scalar.cmpi ne, %ori, %const_2 : i64
      scf.if %cmp_4 {
        %const_5 = scalar.constant -1 : i64
        view.store %const_5, %group_indices[%madd_2, %j] : i64, view<[%num_tokens_idx]x4xi64, %layout>
      }
    }
  }
  kernel.return
}
"""
