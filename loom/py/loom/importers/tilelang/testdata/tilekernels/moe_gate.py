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


def _make_topk_gate_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_topk_gate_kernel(num_experts: int, num_topk: int) -> Any:
        num_tokens = T.dynamic("num_tokens")
        num_threads = 32
        num_aligned_experts = _align_to(num_experts, num_threads)

        @T.prim_func  # type: ignore[untyped-decorator]
        def topk_gate_kernel(
            scores: T.Tensor[(num_tokens, num_experts), T.float32],
            topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        ) -> None:
            with T.Kernel(num_tokens, threads=num_threads) as pid:
                scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
                amax_fragment = T.alloc_fragment((1,), T.float32)
                idx_fragment = T.alloc_fragment((num_aligned_experts,), T.int32)
                idx_reducer = T.alloc_reducer(
                    (1,),
                    T.int32,
                    "min",
                    replication="all",
                )
                topk_idx_shared = T.alloc_shared((num_topk,), T.int32)

                for i in T.Parallel(num_aligned_experts):
                    if i < num_experts:
                        scores_fragment[i] = scores[pid, i]
                    else:
                        scores_fragment[i] = -T.infinity(T.float32)
                for i in T.Parallel(num_aligned_experts):
                    idx_fragment[i] = i

                for k in T.unroll(num_topk):
                    T.reduce_max(scores_fragment, amax_fragment)
                    T.fill(idx_reducer, T.max_value(T.int32))
                    for i in T.Parallel(num_aligned_experts):
                        if scores_fragment[i] == amax_fragment[0]:
                            idx_reducer[0] = T.min(idx_reducer[0], idx_fragment[i])
                    T.finalize_reducer(idx_reducer)
                    topk_idx_shared[k] = idx_reducer[0]
                    for i in T.Parallel(num_aligned_experts):
                        if idx_fragment[i] == idx_reducer[0]:
                            scores_fragment[i] = -T.infinity(T.float32)

                T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)

        return topk_gate_kernel

    return get_topk_gate_kernel


def _topk_gate_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_topk_gate_kernel(tilelang, T),
        args=(8, 2),
        target=target,
        name="topk_gate_kernel",
    )


# ====
@tilelang_case(
    name="tilekernels_topk_gate_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "topk", "amdgpu"),
)
def tilekernels_topk_gate_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _topk_gate_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("topk_gate_kernel") @topk_gate_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%num_tokens_idx, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%scores_handle: buffer, %topk_idx_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %scores_noalias = buffer.assume.noalias %scores_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %scores = buffer.view %scores_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x8xf32, %layout>
  %topk_idx_noalias = buffer.assume.noalias %topk_idx_handle : buffer
  %topk_idx = buffer.view %topk_idx_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi64, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %scores_fragment_bytes = index.constant 128 : offset
  %scores_fragment_buffer = buffer.alloca %scores_fragment_bytes {base_alignment = 4, memory_space = private} : buffer
  %scores_fragment = buffer.view %scores_fragment_buffer[%c0_bytes] : buffer -> view<32xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %scores_fragment_state = vector.splat %f32_zero : vector<32xf32>
  %amax_fragment_bytes = index.constant 4 : offset
  %amax_fragment_buffer = buffer.alloca %amax_fragment_bytes {base_alignment = 4, memory_space = private} : buffer
  %amax_fragment = buffer.view %amax_fragment_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %amax_fragment_state = vector.splat %f32_zero : vector<1xf32>
  %idx_fragment_buffer = buffer.alloca %scores_fragment_bytes {base_alignment = 4, memory_space = private} : buffer
  %idx_fragment = buffer.view %idx_fragment_buffer[%c0_bytes] : buffer -> view<32xi32, %layout>
  %i32_zero = scalar.constant 0 : i32
  %idx_fragment_state = vector.splat %i32_zero : vector<32xi32>
  %idx_reducer_buffer = buffer.alloca %amax_fragment_bytes {base_alignment = 4, memory_space = private} : buffer
  %idx_reducer = buffer.view %idx_reducer_buffer[%c0_bytes] : buffer -> view<1xi32, %layout>
  %idx_reducer_state = vector.splat %i32_zero : vector<1xi32>
  %topk_idx_shared_bytes = index.constant 8 : offset
  %topk_idx_shared_buffer = buffer.alloca %topk_idx_shared_bytes {base_alignment = 4, memory_space = workgroup} : buffer
  %topk_idx_shared = buffer.view %topk_idx_shared_buffer[%c0_bytes] : buffer -> view<2xi32, %layout>
  %c0 = index.constant 0 : index
  %c32 = index.constant 32 : index
  %c1 = index.constant 1 : index
  %scores_fragment_state_next = scf.for %i = [%c0 to %c32 step %c1](%scores_fragment_state_iter = %scores_fragment_state : vector<32xf32>) -> (vector<32xf32>) {
    %c8 = index.constant 8 : index
    %cmp = index.cmp slt, %i, %c8 : index
    %scores_fragment_state_if = scf.if %cmp -> (vector<32xf32>) {
      %load = view.load %scores[%bx, %i] : view<[%num_tokens_idx]x8xf32, %layout> -> f32
      %store = vector.insert %load into %scores_fragment_state_iter[%i] : f32, vector<32xf32>
      scf.yield %store : vector<32xf32>
    } else {
      %inf = scalar.constant inf : f32
      %const = scalar.constant -1.0 : f32
      %mulf = scalar.mulf %inf, %const : f32
      %store_2 = vector.insert %mulf into %scores_fragment_state_iter[%i] : f32, vector<32xf32>
      scf.yield %store_2 : vector<32xf32>
    }
    scf.yield %scores_fragment_state_if : vector<32xf32>
  }
  %idx_fragment_state_next = scf.for %i = [%c0 to %c32 step %c1](%idx_fragment_state_iter = %idx_fragment_state : vector<32xi32>) -> (vector<32xi32>) {
    %store_cast = index.cast %i : index to i32
    %store_3 = vector.insert %store_cast into %idx_fragment_state_iter[%i] : i32, vector<32xi32>
    scf.yield %store_3 : vector<32xi32>
  }
  %c2 = index.constant 2 : index
  %scores_fragment_state_next_2, %idx_reducer_state_next = scf.for %k = [%c0 to %c2 step %c1](%scores_fragment_state_iter = %scores_fragment_state_next : vector<32xf32>, %idx_reducer_state_iter = %idx_reducer_state : vector<1xi32>) -> (vector<32xf32>, vector<1xi32>) unroll {
    %identity = scalar.constant -inf : f32
    %reduce = vector.reduce<maxnumf> %scores_fragment_state_iter, %identity : vector<32xf32>, f32
    %store_4 = vector.insert %reduce into %amax_fragment_state[%c0] : f32, vector<1xf32>
    %const_2 = scalar.constant 2147483647 : i32
    %fill = vector.splat %const_2 : vector<1xi32>
    %idx_reducer_state_next_2 = scf.for %i = [%c0 to %c32 step %c1](%idx_reducer_state_iter = %fill : vector<1xi32>) -> (vector<1xi32>) {
      %load_2 = vector.extract %scores_fragment_state_iter[%i] : vector<32xf32> -> f32
      %load_3 = vector.extract %store_4[%c0] : vector<1xf32> -> f32
      %cmp_2 = scalar.cmpf oeq, %load_2, %load_3 : f32
      %idx_reducer_state_if = scf.if %cmp_2 -> (vector<1xi32>) {
        %load_4 = vector.extract %idx_reducer_state_iter[%c0] : vector<1xi32> -> i32
        %load_5 = vector.extract %idx_fragment_state_next[%i] : vector<32xi32> -> i32
        %minsi = scalar.minsi %load_4, %load_5 : i32
        %store_5 = vector.insert %minsi into %idx_reducer_state_iter[%c0] : i32, vector<1xi32>
        scf.yield %store_5 : vector<1xi32>
      } else {
        scf.yield %idx_reducer_state_iter : vector<1xi32>
      }
      scf.yield %idx_reducer_state_if : vector<1xi32>
    }
    %idx_reducer_state_next_3 = scf.for %i0 = [%c0 to %c1 step %c1](%idx_reducer_state_iter = %idx_reducer_state_next_2 : vector<1xi32>) -> (vector<1xi32>) {
      %reducer_value = vector.extract %idx_reducer_state_iter[%i0] : vector<1xi32> -> i32
      %reducer_all = kernel.workgroup.reduce<minsi> %reducer_value : i32
      %store_6 = vector.insert %reducer_all into %idx_reducer_state_iter[%i0] : i32, vector<1xi32>
      scf.yield %store_6 : vector<1xi32>
    }
    %load_6 = vector.extract %idx_reducer_state_next_3[%c0] : vector<1xi32> -> i32
    view.store %load_6, %topk_idx_shared[%k] : i32, view<2xi32, %layout>
    %scores_fragment_state_next_3 = scf.for %i = [%c0 to %c32 step %c1](%scores_fragment_state_iter = %scores_fragment_state_iter : vector<32xf32>) -> (vector<32xf32>) {
      %load_7 = vector.extract %idx_fragment_state_next[%i] : vector<32xi32> -> i32
      %load_8 = vector.extract %idx_reducer_state_next_3[%c0] : vector<1xi32> -> i32
      %cmp_3 = scalar.cmpi eq, %load_7, %load_8 : i32
      %scores_fragment_state_if_2 = scf.if %cmp_3 -> (vector<32xf32>) {
        %inf_2 = scalar.constant inf : f32
        %const_3 = scalar.constant -1.0 : f32
        %mulf_2 = scalar.mulf %inf_2, %const_3 : f32
        %store_7 = vector.insert %mulf_2 into %scores_fragment_state_iter[%i] : f32, vector<32xf32>
        scf.yield %store_7 : vector<32xf32>
      } else {
        scf.yield %scores_fragment_state_iter : vector<32xf32>
      }
      scf.yield %scores_fragment_state_if_2 : vector<32xf32>
    }
    scf.yield %scores_fragment_state_next_3, %idx_reducer_state_next_3 : vector<32xf32>, vector<1xi32>
  }
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %i0_active = index.cmp slt, %tx, %c2 : index
  scf.if %i0_active {
    %copy = view.load %topk_idx_shared[%tx] : view<2xi32, %layout> -> i32
    %copy_ext = scalar.extsi %copy : i32 to i64
    view.store %copy_ext, %topk_idx[%bx, %tx] : i64, view<[%num_tokens_idx]x2xi64, %layout>
  }
  kernel.return
}
"""
