# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _make_reduce_fused_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_reduce_fused_kernel(hidden: int, num_topk: int) -> Any:
        num_threads = 128
        num_tokens = T.dynamic("num_tokens")
        num_expanded_tokens = T.dynamic("num_expanded_tokens")

        @T.prim_func  # type: ignore[untyped-decorator]
        def reduce_fused_kernel(
            x: T.Tensor[(num_expanded_tokens, hidden), T.float16],
            topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
            token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
            out: T.Tensor[(num_tokens, hidden), T.float16],
        ) -> None:
            with T.Kernel(num_tokens, threads=num_threads) as (pid_token,):
                reduced_fragment = T.alloc_fragment((hidden,), T.float32)
                topk_weights_local = T.alloc_fragment((num_topk,), T.float32)
                topk_to_pos_local = T.alloc_fragment((num_topk,), T.int32)
                scale = T.alloc_var(T.float32)

                T.clear(reduced_fragment)
                T.copy(topk_weights[pid_token, :], topk_weights_local)
                T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)

                for k in T.unroll(num_topk):
                    pos = topk_to_pos_local[k]
                    T.assume(pos < num_expanded_tokens)
                    if pos >= 0:
                        scale = topk_weights_local[k]
                        for i in T.Parallel(hidden):
                            reduced_fragment[i] += x[pos, i] * scale

                for i in T.Parallel(hidden):
                    out[pid_token, i] = reduced_fragment[i]

        return reduce_fused_kernel

    return get_reduce_fused_kernel


def _reduce_fused_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_reduce_fused_kernel(tilelang, T),
        args=(8, 2),
        target=target,
        name="reduce_fused_kernel",
    )


@tilelang_case(
    name="tilekernels_reduce_fused_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "reduce", "amdgpu"),
)
def tilekernels_reduce_fused_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _reduce_fused_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("reduce_fused_kernel") @reduce_fused_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%num_tokens_idx, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%x_handle: buffer, %topk_weights_handle: buffer, %token_topk_to_pos_handle: buffer, %out_handle: buffer, %num_expanded_tokens: i32, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %x_noalias = buffer.assume.noalias %x_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_expanded_tokens_idx = index.cast %num_expanded_tokens : i32 to index
  %x = buffer.view %x_noalias[%c0_bytes] : buffer -> view<[%num_expanded_tokens_idx]x8xf16, %layout>
  %topk_weights_noalias = buffer.assume.noalias %topk_weights_handle : buffer
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %topk_weights = buffer.view %topk_weights_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %token_topk_to_pos_noalias = buffer.assume.noalias %token_topk_to_pos_handle : buffer
  %token_topk_to_pos = buffer.view %token_topk_to_pos_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi32, %layout>
  %out_noalias = buffer.assume.noalias %out_handle : buffer
  %out = buffer.view %out_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x8xf16, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %reduced_fragment_bytes = index.constant 32 : offset
  %reduced_fragment_buffer = buffer.alloca %reduced_fragment_bytes {base_alignment = 4, memory_space = private} : buffer
  %reduced_fragment = buffer.view %reduced_fragment_buffer[%c0_bytes] : buffer -> view<8xf32, %layout>
  %f32_zero = scalar.constant 0.0 : f32
  %reduced_fragment_state = vector.splat %f32_zero : vector<8xf32>
  %topk_weights_local_bytes = index.constant 8 : offset
  %topk_weights_local_buffer = buffer.alloca %topk_weights_local_bytes {base_alignment = 4, memory_space = private} : buffer
  %topk_weights_local = buffer.view %topk_weights_local_buffer[%c0_bytes] : buffer -> view<2xf32, %layout>
  %topk_weights_local_state = vector.splat %f32_zero : vector<2xf32>
  %topk_to_pos_local_buffer = buffer.alloca %topk_weights_local_bytes {base_alignment = 4, memory_space = private} : buffer
  %topk_to_pos_local = buffer.view %topk_to_pos_local_buffer[%c0_bytes] : buffer -> view<2xi32, %layout>
  %i32_zero = scalar.constant 0 : i32
  %topk_to_pos_local_state = vector.splat %i32_zero : vector<2xi32>
  %scale_bytes = index.constant 4 : offset
  %scale_buffer = buffer.alloca %scale_bytes {base_alignment = 4, memory_space = private} : buffer
  %scale = buffer.view %scale_buffer[%c0_bytes] : buffer -> view<1xf32, %layout>
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %fill = vector.splat %f32_zero : vector<8xf32>
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %copy = vector.load %topk_weights[%bx, %c0] : view<[%num_tokens_idx]x2xf32, %layout> -> vector<2xf32>
  %copy_2 = vector.load %token_topk_to_pos[%bx, %c0] : view<[%num_tokens_idx]x2xi32, %layout> -> vector<2xi32>
  %reduced_fragment_state_next = scf.for %k = [%c0 to %c2 step %c1](%reduced_fragment_state_iter = %fill : vector<8xf32>) -> (vector<8xf32>) unroll {
    %load = vector.extract %copy_2[%k] : vector<2xi32> -> i32
    %pos_assumed, %num_expanded_tokens_assumed = scalar.assume %load, %num_expanded_tokens [lt(%load, %num_expanded_tokens)] : i32, i32
    %cmp = scalar.cmpi sge, %pos_assumed, %i32_zero : i32
    %reduced_fragment_state_if = scf.if %cmp -> (vector<8xf32>) {
      %load_2 = vector.extract %copy[%k] : vector<2xf32> -> f32
      %reduced_fragment_state_next_2 = scf.for %i = [%c0 to %c8 step %c1](%reduced_fragment_state_iter = %reduced_fragment_state_iter : vector<8xf32>) -> (vector<8xf32>) {
        %load_3 = vector.extract %reduced_fragment_state_iter[%i] : vector<8xf32> -> f32
        %pos_idx = index.cast %pos_assumed : i32 to index
        %load_4 = view.load %x[%pos_idx, %i] : view<[%num_expanded_tokens_idx]x8xf16, %layout> -> f16
        %extf = scalar.extf %load_4 : f16 to f32
        %mulf = scalar.mulf %extf, %load_2 : f32
        %addf = scalar.addf %load_3, %mulf : f32
        %store = vector.insert %addf into %reduced_fragment_state_iter[%i] : f32, vector<8xf32>
        scf.yield %store : vector<8xf32>
      }
      scf.yield %reduced_fragment_state_next_2 : vector<8xf32>
    } else {
      scf.yield %reduced_fragment_state_iter : vector<8xf32>
    }
    scf.yield %reduced_fragment_state_if : vector<8xf32>
  }
  scf.for %i = [%c0 to %c8 step %c1] {
    %load_5 = vector.extract %reduced_fragment_state_next[%i] : vector<8xf32> -> f32
    %fptrunc = scalar.fptrunc %load_5 : f32 to f16
    view.store %fptrunc, %out[%bx, %i] : f16, view<[%num_tokens_idx]x8xf16, %layout>
  }
  kernel.return
}
"""
