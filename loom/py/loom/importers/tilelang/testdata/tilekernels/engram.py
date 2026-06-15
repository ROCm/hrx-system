# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


# ====
def _make_engram_hash_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
        },
    )
    def get_engram_hash_kernel(
        max_ngram_size: int,
        num_ngram_layers: int,
        num_embed_table_per_ngram: int,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")
        threads = 32

        @T.prim_func  # type: ignore[untyped-decorator]
        def engram_hash_kernel(
            ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
            multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
            vocab_sizes: T.Tensor[
                (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
                T.int32,
            ],
            offsets: T.Tensor[
                (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram),
                T.int32,
            ],
            output: T.Tensor[
                (
                    num_ngram_layers,
                    num_tokens,
                    (max_ngram_size - 1) * num_embed_table_per_ngram,
                ),
                T.int32,
            ],
        ) -> None:
            with T.Kernel(
                num_ngram_layers,
                T.ceildiv(num_tokens, threads),
                threads=threads,
            ) as (pid_h, pid_s):
                tid = T.get_thread_binding()
                token_idx = pid_s * threads + tid
                if token_idx >= num_tokens:
                    T.thread_return()

                hash_local = T.alloc_var(T.int64, init=0)
                for ngram_idx in T.unroll(max_ngram_size):
                    token = T.cast(ngram_token_ids[token_idx, ngram_idx], T.int64)
                    hash_local = T.bitwise_xor(
                        hash_local,
                        token * multipliers[pid_h, ngram_idx],
                    )
                    if ngram_idx > 0:
                        for j in T.unroll(num_embed_table_per_ngram):
                            col = (ngram_idx - 1) * num_embed_table_per_ngram + j
                            vocab = T.cast(
                                vocab_sizes[pid_h, ngram_idx - 1, j],
                                T.int64,
                            )
                            output[pid_h, token_idx, col] = (
                                T.cast(
                                    hash_local % vocab,
                                    T.int32,
                                )
                                + offsets[pid_h, col]
                            )

        return engram_hash_kernel

    return get_engram_hash_kernel


def _engram_hash_input(tilelang: Any, T: Any, *, target: str) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_engram_hash_kernel(tilelang, T),
        args=(3, 2, 4),
        target=target,
        name="engram_hash_kernel",
    )


@tilelang_case(
    name="tilekernels_engram_hash_gfx1100",
    category="kernel",
    tags=("tilekernels", "engram", "hash", "amdgpu"),
)
def tilekernels_engram_hash_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _engram_hash_input(tilelang, T, target="hip -mcpu=gfx1100")


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("engram_hash_kernel") @engram_hash_kernel(%num_tokens: i32) {
  %c2 = index.constant 2 : index
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c32 = index.constant 32 : index
  %add = index.add %num_tokens_idx, %c32 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %add, %c1 : index
  %div = index.div %sub, %c32 : index
  kernel.launch.config workgroups(%c2, %div, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%ngram_token_ids_handle: buffer, %multipliers_handle: buffer, %vocab_sizes_handle: buffer, %offsets_handle: buffer, %output_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %ngram_token_ids_noalias = buffer.assume.noalias %ngram_token_ids_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %ngram_token_ids = buffer.view %ngram_token_ids_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x3xi32, %layout>
  %multipliers_noalias = buffer.assume.noalias %multipliers_handle : buffer
  %multipliers = buffer.view %multipliers_noalias[%c0_bytes] : buffer -> view<2x3xi64, %layout>
  %vocab_sizes_noalias = buffer.assume.noalias %vocab_sizes_handle : buffer
  %vocab_sizes = buffer.view %vocab_sizes_noalias[%c0_bytes] : buffer -> view<2x2x4xi32, %layout>
  %offsets_noalias = buffer.assume.noalias %offsets_handle : buffer
  %offsets = buffer.view %offsets_noalias[%c0_bytes] : buffer -> view<2x8xi32, %layout>
  %output_noalias = buffer.assume.noalias %output_handle : buffer
  %output = buffer.view %output_noalias[%c0_bytes] : buffer -> view<2x[%num_tokens_idx]x8xi32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %by = kernel.workgroup.id<y> : index
  %tid = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %hash_local_bytes = index.constant 8 : offset
  %hash_local_buffer = buffer.alloca %hash_local_bytes {base_alignment = 8, memory_space = private} : buffer
  %hash_local = buffer.view %hash_local_buffer[%c0_bytes] : buffer -> view<1xi64, %layout>
  %c0 = index.constant 0 : index
  %const = scalar.constant 0 : i64
  %c32 = index.constant 32 : index
  %madd = index.madd %by, %c32, %tid : index
  %cmp = index.cmp sge, %madd, %num_tokens_idx : index
  kernel.exit %cmp : i1
  %c3 = index.constant 3 : index
  %c1 = index.constant 1 : index
  %hash_local_state_next = scf.for %ngram_idx = [%c0 to %c3 step %c1](%hash_local_state_iter = %const : i64) -> (i64) unroll {
    %load = view.load %ngram_token_ids[%madd, %ngram_idx] : view<[%num_tokens_idx]x3xi32, %layout> -> i32
    %extsi = scalar.extsi %load : i32 to i64
    %load_2 = view.load %multipliers[%bx, %ngram_idx] : view<2x3xi64, %layout> -> i64
    %muli = scalar.muli %extsi, %load_2 : i64
    %xori = scalar.xori %hash_local_state_iter, %muli : i64
    %cmp_2 = index.cmp sgt, %ngram_idx, %c0 : index
    scf.if %cmp_2 {
      %c4 = index.constant 4 : index
      scf.for %j = [%c0 to %c4 step %c1] unroll {
        %sub = index.sub %ngram_idx, %c1 : index
        %madd_2 = index.madd %sub, %c4, %j : index
        %sub_2 = index.sub %ngram_idx, %c1 : index
        %load_3 = view.load %vocab_sizes[%bx, %sub_2, %j] : view<2x2x4xi32, %layout> -> i32
        %extsi_2 = scalar.extsi %load_3 : i32 to i64
        %remsi = scalar.remsi %xori, %extsi_2 : i64
        %trunci = scalar.trunci %remsi : i64 to i32
        %load_4 = view.load %offsets[%bx, %madd_2] : view<2x8xi32, %layout> -> i32
        %addi = scalar.addi %trunci, %load_4 : i32
        view.store %addi, %output[%bx, %madd, %madd_2] : i32, view<2x[%num_tokens_idx]x8xi32, %layout>
      }
    }
    scf.yield %xori : i64
  }
  kernel.return
}
"""
