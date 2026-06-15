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
def _make_expand_to_fused_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_expand_to_fused_kernel(
        hidden: int,
        num_topk: int,
    ) -> Any:
        num_threads = 64
        hidden_aligned = _align_to(hidden, num_threads)
        num_tokens = T.dynamic("num_tokens")
        num_expanded_tokens = T.dynamic("num_expanded_tokens")
        num_blocks = T.max(num_tokens, num_expanded_tokens)

        @T.prim_func  # type: ignore[untyped-decorator]
        def expand_to_fused_kernel(
            x: T.Tensor[(num_tokens, hidden), T.float16],
            expanded_x: T.Tensor[(num_expanded_tokens, hidden), T.float16],
            token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
            pos_to_expert: T.Tensor[(num_expanded_tokens,), T.int32],
        ) -> None:
            with T.Kernel(num_blocks, threads=num_threads) as (pid_token,):
                pos_local = T.alloc_local((num_topk,), T.int32)
                if pid_token < num_expanded_tokens:  # noqa: SIM102
                    if pos_to_expert[pid_token] < 0:
                        for i in T.Parallel(hidden_aligned):
                            expanded_x[pid_token, i] = 0

                if pid_token >= num_tokens:
                    T.thread_return()
                T.assume(pid_token < num_tokens)

                x_fragment = T.alloc_fragment((hidden_aligned,), T.float16)

                T.copy(token_topk_to_pos[pid_token, 0], pos_local)
                T.copy(x[pid_token, :], x_fragment[0:hidden])

                for k in T.serial(num_topk):
                    T.assume(pos_local[k] < num_expanded_tokens)
                    if pos_local[k] >= 0:
                        for i in T.Parallel(hidden_aligned):
                            expanded_x[pos_local[k], i] = x_fragment[i]

        return expand_to_fused_kernel

    return get_expand_to_fused_kernel


def _expand_to_fused_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_expand_to_fused_kernel(tilelang, T),
        args=(128, 2),
        target=target,
        name="expand_to_fused_kernel",
    )


@tilelang_case(
    name="tilekernels_expand_to_fused_gfx1100",
    category="kernel",
    tags=("tilekernels", "moe", "expand", "amdgpu"),
)
def tilekernels_expand_to_fused_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _expand_to_fused_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("expand_to_fused_kernel") @expand_to_fused_kernel(%num_tokens: i32, %num_expanded_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %num_expanded_tokens_idx = index.cast %num_expanded_tokens : i32 to index
  %max = index.max %num_tokens_idx, %num_expanded_tokens_idx : index
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%max, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%x_handle: buffer, %expanded_x_handle: buffer, %token_topk_to_pos_handle: buffer, %pos_to_expert_handle: buffer, %num_tokens: i32, %num_expanded_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %x_noalias = buffer.assume.noalias %x_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %x = buffer.view %x_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x128xf16, %layout>
  %expanded_x_noalias = buffer.assume.noalias %expanded_x_handle : buffer
  %num_expanded_tokens_idx = index.cast %num_expanded_tokens : i32 to index
  %expanded_x = buffer.view %expanded_x_noalias[%c0_bytes] : buffer -> view<[%num_expanded_tokens_idx]x128xf16, %layout>
  %token_topk_to_pos_noalias = buffer.assume.noalias %token_topk_to_pos_handle : buffer
  %token_topk_to_pos = buffer.view %token_topk_to_pos_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xi32, %layout>
  %pos_to_expert_noalias = buffer.assume.noalias %pos_to_expert_handle : buffer
  %pos_to_expert = buffer.view %pos_to_expert_noalias[%c0_bytes] : buffer -> view<[%num_expanded_tokens_idx]xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %pos_local_bytes = index.constant 8 : offset
  %pos_local_buffer = buffer.alloca %pos_local_bytes {base_alignment = 4, memory_space = private} : buffer
  %pos_local = buffer.view %pos_local_buffer[%c0_bytes] : buffer -> view<2xi32, %layout>
  %x_fragment_bytes = index.constant 256 : offset
  %x_fragment_buffer = buffer.alloca %x_fragment_bytes {base_alignment = 2, memory_space = private} : buffer
  %x_fragment = buffer.view %x_fragment_buffer[%c0_bytes] : buffer -> view<128xf16, %layout>
  %f16_zero = scalar.constant 0.0 : f16
  %x_fragment_state = vector.splat %f16_zero : vector<128xf16>
  %cmp = index.cmp slt, %bx, %num_expanded_tokens_idx : index
  scf.if %cmp {
    %load = view.load %pos_to_expert[%bx] : view<[%num_expanded_tokens_idx]xi32, %layout> -> i32
    %const = scalar.constant 0 : i32
    %cmp_2 = scalar.cmpi slt, %load, %const : i32
    scf.if %cmp_2 {
      %c2 = index.constant 2 : index
      %i_base = index.mul %tx, %c2 : index
      %store_splat = vector.splat %f16_zero : vector<2xf16>
      vector.store %store_splat, %expanded_x[%bx, %i_base] : vector<2xf16>, view<[%num_expanded_tokens_idx]x128xf16, %layout>
    }
  }
  %cmp_3 = index.cmp sge, %bx, %num_tokens_idx : index
  kernel.exit %cmp_3 : i1
  %bx_assumed, %num_tokens_assumed = index.assume %bx, %num_tokens_idx [lt(%bx, %num_tokens_idx)] : index, index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2_2 = index.constant 2 : index
  scf.for %i0 = [%c0 to %c2_2 step %c1] {
    %copy = view.load %token_topk_to_pos[%bx_assumed, %i0] : view<[%num_tokens_idx]x2xi32, %layout> -> i32
    view.store %copy, %pos_local[%i0] : i32, view<2xi32, %layout>
  }
  %c128 = index.constant 128 : index
  %i0_base = index.mul %tx, %c2_2 : index
  %copy_2 = vector.load %x[%bx_assumed, %i0_base] : view<[%num_tokens_idx]x128xf16, %layout> -> vector<2xf16>
  scf.for %k = [%c0 to %c2_2 step %c1] {
    %load_2 = view.load %pos_local[%k] : view<2xi32, %layout> -> i32
    %value_assumed, %num_expanded_tokens_assumed = scalar.assume %load_2, %num_expanded_tokens [lt(%load_2, %num_expanded_tokens)] : i32, i32
    %const_2 = scalar.constant 0 : i32
    %cmp_4 = scalar.cmpi sge, %value_assumed, %const_2 : i32
    scf.if %cmp_4 {
      %i_base_2 = index.mul %tx, %c2_2 : index
      %load_idx = index.cast %value_assumed : i32 to index
      vector.store %copy_2, %expanded_x[%load_idx, %i_base_2] : vector<2xf16>, view<[%num_expanded_tokens_idx]x128xf16, %layout>
    }
  }
  kernel.return
}
"""
