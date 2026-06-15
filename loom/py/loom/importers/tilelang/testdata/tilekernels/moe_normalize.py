# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


# ====
def _make_normalize_weight_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_normalize_weight_kernel(num_topk: int) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_blocks = T.ceildiv(num_tokens, 128)

        @T.prim_func  # type: ignore[untyped-decorator]
        def normalize_weight_kernel(
            topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
            denominator: T.Tensor[(num_tokens,), T.float32],
            normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid,):
                tid = T.get_thread_binding()
                weights_local = T.alloc_local((num_topk,), T.float32)
                row = pid * num_threads + tid
                if row < num_tokens:
                    total = T.alloc_var(T.float32, init=1e-20)
                    for i in T.vectorized(num_topk):
                        weights_local[i] = topk_weights[row, i]
                    for i in T.unroll(num_topk):
                        total += weights_local[i]
                    denominator[row] = total
                    for i in T.vectorized(num_topk):
                        normalized_weights[row, i] = weights_local[i] / total

        return normalize_weight_kernel

    return get_normalize_weight_kernel


def _normalize_weight_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_normalize_weight_kernel(tilelang, T),
        args=(2,),
        target=target,
        name="normalize_weight_kernel",
    )


@tilelang_case(
    name="tilekernels_normalize_weight_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "normalize", "amdgpu"),
)
def tilekernels_normalize_weight_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _normalize_weight_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("normalize_weight_kernel") @normalize_weight_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c128 = index.constant 128 : index
  %add = index.add %num_tokens_idx, %c128 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %add, %c1 : index
  %div = index.div %sub, %c128 : index
  kernel.launch.config workgroups(%div, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%topk_weights_handle: buffer, %denominator_handle: buffer, %normalized_weights_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %topk_weights_noalias = buffer.assume.noalias %topk_weights_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %topk_weights = buffer.view %topk_weights_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %denominator_noalias = buffer.assume.noalias %denominator_handle : buffer
  %denominator = buffer.view %denominator_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]xf32, %layout>
  %normalized_weights_noalias = buffer.assume.noalias %normalized_weights_handle : buffer
  %normalized_weights = buffer.view %normalized_weights_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tid = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %weights_local_bytes = index.constant 8 : offset
  %weights_local_buffer = buffer.alloca %weights_local_bytes {base_alignment = 4, memory_space = private} : buffer
  %weights_local = buffer.view %weights_local_buffer[%c0_bytes] : buffer -> view<2xf32, %layout>
  %total_bytes = index.constant 4 : offset
  %total_buffer = buffer.alloca %total_bytes {base_alignment = 4, memory_space = private} : buffer
  %total = buffer.view %total_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %c0 = index.constant 0 : index
  %const = scalar.constant 9.9999999999999995e-21 : f32
  %c128 = index.constant 128 : index
  %madd = index.madd %bx, %c128, %tid : index
  %cmp = index.cmp slt, %madd, %num_tokens_idx : index
  %total_state_if = scf.if %cmp -> (f32) {
    %c2 = index.constant 2 : index
    %c1 = index.constant 1 : index
    scf.for %i = [%c0 to %c2 step %c1] {
      %load = view.load %topk_weights[%madd, %i] : view<[%num_tokens_idx]x2xf32, %layout> -> f32
      view.store %load, %weights_local[%i] : f32, view<2xf32, %layout>
    }
    %total_state_next = scf.for %i = [%c0 to %c2 step %c1](%total_state_iter = %const : f32) -> (f32) unroll {
      %load_2 = view.load %weights_local[%i] : view<2xf32, %layout> -> f32
      %addf = scalar.addf %total_state_iter, %load_2 : f32
      scf.yield %addf : f32
    }
    view.store %total_state_next, %denominator[%madd] : f32, view<[%num_tokens_idx]xf32, %layout>
    scf.for %i = [%c0 to %c2 step %c1] {
      %load_3 = view.load %weights_local[%i] : view<2xf32, %layout> -> f32
      %divf = scalar.divf %load_3, %total_state_next : f32
      view.store %divf, %normalized_weights[%madd, %i] : f32, view<[%num_tokens_idx]x2xf32, %layout>
    }
    scf.yield %total_state_next : f32
  } else {
    scf.yield %const : f32
  }
  kernel.return
}
"""
