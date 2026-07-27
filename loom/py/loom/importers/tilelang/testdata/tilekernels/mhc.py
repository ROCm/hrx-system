# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _make_expand_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_expand_kernel(hidden: int, mhc_mult: int) -> Any:
        num_tokens = T.dynamic("num_tokens")
        token_block_size = 32
        hidden_block_size = 128

        @T.prim_func  # type: ignore[untyped-decorator]
        def expand_to_mhc_fwd_kernel(
            x: T.Tensor[(num_tokens, hidden), T.bfloat16],
            output: T.Tensor[(num_tokens, mhc_mult, hidden), T.bfloat16],
        ) -> None:
            with T.Kernel(
                T.ceildiv(num_tokens, token_block_size),
                T.ceildiv(hidden, hidden_block_size),
            ) as (pid_token, pid_hidden):
                if num_tokens > 0:
                    local = T.alloc_fragment(
                        (token_block_size, hidden_block_size),
                        T.bfloat16,
                    )
                    T.copy(
                        x[
                            pid_token * token_block_size,
                            pid_hidden * hidden_block_size,
                        ],
                        local,
                    )
                    for m in T.serial(mhc_mult):
                        for ti, tj in T.Parallel(
                            token_block_size,
                            hidden_block_size,
                        ):
                            i = pid_token * token_block_size + ti
                            j = pid_hidden * hidden_block_size + tj
                            if i < num_tokens and j < hidden:
                                output[i, m, j] = local[ti, tj]

        return expand_to_mhc_fwd_kernel

    return get_expand_kernel


def _make_head_compute_mix_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_head_compute_mix_kernel(
        mhc_mult: int,
        mhc_pre_eps: float,
        token_block_size: int,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func  # type: ignore[untyped-decorator]
        def mhc_head_compute_mix_fwd_kernel(
            input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
            mhc_scale: T.Tensor[(1,), T.float32],
            mhc_base: T.Tensor[(mhc_mult,), T.float32],
            output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        ) -> None:
            with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as (pid,):
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        output_mix[i, j] = (
                            T.sigmoid(
                                input_mix[i, j] * mhc_scale[0] + mhc_base[j],
                            )
                            + mhc_pre_eps
                        )

        return mhc_head_compute_mix_fwd_kernel

    return get_head_compute_mix_kernel


def _make_pre_split_mixes_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_pre_split_mixes_kernel(
        mhc_mult: int,
        mhc_post_mult_value: float,
        mhc_pre_eps: float,
        token_block_size: int,
        dtype: Any = T.float32,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")
        mhc_mult2 = mhc_mult * mhc_mult
        mhc_mult3 = mhc_mult * 2 + mhc_mult2

        @T.prim_func  # type: ignore[untyped-decorator]
        def mhc_pre_split_mixes_fwd_kernel(
            input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype],
            mhc_scale: T.Tensor[(3,), dtype],
            mhc_base: T.Tensor[(mhc_mult3,), dtype],
            pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
            post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
            comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), dtype],
        ) -> None:
            with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as (pid,):
                input_mixes_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult3),
                    dtype,
                )
                pre_layer_mix_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult),
                    dtype,
                )
                post_layer_mix_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult),
                    dtype,
                )
                comb_res_mix_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult2),
                    dtype,
                )

                T.annotate_layout(
                    {
                        input_mixes_frag: T.Fragment(
                            (token_block_size, mhc_mult3),
                            lambda i, j: (i % 32, i // 32 * mhc_mult3 + j),
                        ),
                    },
                )

                T.copy(input_mixes[pid * token_block_size, 0], input_mixes_frag)

                for i, j in T.Parallel(token_block_size, mhc_mult):
                    pre_layer_mix_frag[i, j] = (
                        T.sigmoid(
                            input_mixes_frag[i, j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )
                for i, j in T.Parallel(token_block_size, mhc_mult):
                    post_layer_mix_frag[i, j] = (
                        T.sigmoid(
                            input_mixes_frag[i, j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult],
                        )
                        * mhc_post_mult_value
                    )
                for i, j in T.Parallel(token_block_size, mhc_mult2):
                    comb_res_mix_frag[i, j] = (
                        input_mixes_frag[i, j + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j + mhc_mult * 2]
                    )

                T.copy(pre_layer_mix_frag, pre_layer_mix[pid * token_block_size, 0])
                T.copy(post_layer_mix_frag, post_layer_mix[pid * token_block_size, 0])
                T.copy(comb_res_mix_frag, comb_res_mix[pid * token_block_size, 0])

        return mhc_pre_split_mixes_fwd_kernel

    return get_pre_split_mixes_kernel


def _make_sinkhorn_kernel(tilelang: Any, T: Any) -> Any:
    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_sinkhorn_kernel(
        hidden_size: int,
        token_block_size: int,
        repeat: int,
        eps: float,
    ) -> Any:
        num_tokens = T.dynamic("num_tokens")

        @T.prim_func  # type: ignore[untyped-decorator]
        def mhc_sinkhorn_kernel(
            comb_res_mix: T.Tensor[
                (num_tokens, hidden_size, hidden_size),
                T.float32,
            ],
            comb_res_mix_out: T.Tensor[
                (num_tokens, hidden_size, hidden_size),
                T.float32,
            ],
        ) -> None:
            with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as (pid_x,):
                comb_frag = T.alloc_fragment(
                    (token_block_size, hidden_size, hidden_size),
                    T.float32,
                )
                row_sum = T.alloc_fragment(
                    (token_block_size, hidden_size),
                    T.float32,
                )
                col_sum = T.alloc_fragment(
                    (token_block_size, hidden_size),
                    T.float32,
                )

                T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_frag)

                row_max = T.alloc_fragment(
                    (token_block_size, hidden_size),
                    T.float32,
                )
                T.reduce_max(comb_frag, row_max, dim=2)
                for i, j, k in T.Parallel(
                    token_block_size,
                    hidden_size,
                    hidden_size,
                ):
                    comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(
                    token_block_size,
                    hidden_size,
                    hidden_size,
                ):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(
                    token_block_size,
                    hidden_size,
                    hidden_size,
                ):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

                for _ in T.serial(repeat - 1):
                    T.reduce_sum(comb_frag, row_sum, dim=2)
                    for i, j, k in T.Parallel(
                        token_block_size,
                        hidden_size,
                        hidden_size,
                    ):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

                    T.reduce_sum(comb_frag, col_sum, dim=1)
                    for i, j, k in T.Parallel(
                        token_block_size,
                        hidden_size,
                        hidden_size,
                    ):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

                T.copy(
                    comb_frag,
                    comb_res_mix_out[pid_x * token_block_size, 0, 0],
                )

        return mhc_sinkhorn_kernel

    return get_sinkhorn_kernel


# ====
@tilelang_case(
    name="tilekernels_mhc_expand_fwd_gfx1100",
    category="kernel",
    tags=("tilekernels", "mhc", "expand", "amdgpu"),
)
def tilekernels_mhc_expand_fwd_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_expand_kernel(tilelang, T),
        args=(128, 2),
        target="hip -mcpu=gfx1100",
        name="expand_to_mhc_fwd_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("expand_to_mhc_fwd_kernel") @expand_to_mhc_fwd_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c32 = index.constant 32 : index
  %add = index.add %num_tokens_idx, %c32 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %add, %c1 : index
  %div = index.div %sub, %c32 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%div, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%x_handle: buffer, %output_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %x_noalias = buffer.assume.noalias %x_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %x = buffer.view %x_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x128xbf16, %layout>
  %output_noalias = buffer.assume.noalias %output_handle : buffer
  %output = buffer.view %output_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2x128xbf16, %layout>
  %bx = kernel.workgroup.id<x> : index
  %by = kernel.workgroup.id<y> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %local_bytes = index.constant 8192 : offset
  %local_buffer = buffer.alloca %local_bytes {base_alignment = 2, memory_space = private} : buffer
  %local = buffer.view %local_buffer[%c0_bytes] : buffer -> view<32x128xbf16, %layout>
  %const = scalar.constant 0 : i32
  %cmp = scalar.cmpi sgt, %num_tokens, %const : i32
  scf.if %cmp {
    %c32 = index.constant 32 : index
    %mul = index.mul %bx, %c32 : index
    %c128 = index.constant 128 : index
    %mul_2 = index.mul %by, %c128 : index
    %c0 = index.constant 0 : index
    %c1 = index.constant 1 : index
    scf.for %i0 = [%c0 to %c32 step %c1] {
      scf.for %i1 = [%c0 to %c128 step %c1] {
        %idx = index.add %mul, %i0 : index
        %idx_2 = index.add %mul_2, %i1 : index
        %copy = view.load %x[%idx, %idx_2] : view<[%num_tokens_idx]x128xbf16, %layout> -> bf16
        view.store %copy, %local[%i0, %i1] : bf16, view<32x128xbf16, %layout>
      }
    }
    %c2 = index.constant 2 : index
    scf.for %m = [%c0 to %c2 step %c1] {
      scf.for %ti = [%c0 to %c32 step %c1] {
        scf.for %tj = [%c0 to %c128 step %c1] {
          %madd = index.madd %bx, %c32, %ti : index
          %madd_2 = index.madd %by, %c128, %tj : index
          %cmp_2 = index.cmp slt, %madd, %num_tokens_idx : index
          %cmp_3 = index.cmp slt, %madd_2, %c128 : index
          %andi = scalar.andi %cmp_2, %cmp_3 : i1
          scf.if %andi {
            %load = view.load %local[%ti, %tj] : view<32x128xbf16, %layout> -> bf16
            view.store %load, %output[%madd, %m, %madd_2] : bf16, view<[%num_tokens_idx]x2x128xbf16, %layout>
          }
        }
      }
    }
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tilekernels_mhc_head_compute_mix_fwd_gfx1100",
    category="kernel",
    tags=("tilekernels", "mhc", "sigmoid", "amdgpu"),
)
def tilekernels_mhc_head_compute_mix_fwd_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_head_compute_mix_kernel(tilelang, T),
        args=(2, 1e-5, 2),
        target="hip -mcpu=gfx1100",
        name="mhc_head_compute_mix_fwd_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mhc_head_compute_mix_fwd_kernel") @mhc_head_compute_mix_fwd_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c2 = index.constant 2 : index
  %add = index.add %num_tokens_idx, %c2 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %add, %c1 : index
  %div = index.div %sub, %c2 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%div, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%input_mix_handle: buffer, %mhc_scale_handle: buffer, %mhc_base_handle: buffer, %output_mix_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %input_mix_noalias = buffer.assume.noalias %input_mix_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %input_mix = buffer.view %input_mix_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %mhc_scale_noalias = buffer.assume.noalias %mhc_scale_handle : buffer
  %mhc_scale = buffer.view %mhc_scale_noalias[%c0_bytes] : buffer -> view<1xf32, %layout>
  %mhc_base_noalias = buffer.assume.noalias %mhc_base_handle : buffer
  %mhc_base = buffer.view %mhc_base_noalias[%c0_bytes] : buffer -> view<2xf32, %layout>
  %output_mix_noalias = buffer.assume.noalias %output_mix_handle : buffer
  %output_mix = buffer.view %output_mix_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c1 = index.constant 1 : index
  scf.for %i1 = [%c0 to %c2 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      %madd = index.madd %bx, %c2, %i1 : index
      %cmp = index.cmp slt, %madd, %num_tokens_idx : index
      scf.if %cmp {
        %load = view.load %input_mix[%madd, %j] : view<[%num_tokens_idx]x2xf32, %layout> -> f32
        %load_2 = view.load %mhc_scale[%c0] : view<1xf32, %layout> -> f32
        %mulf = scalar.mulf %load, %load_2 : f32
        %load_3 = view.load %mhc_base[%j] : view<2xf32, %layout> -> f32
        %addf = scalar.addf %mulf, %load_3 : f32
        %logisticf = scalar.logisticf %addf : f32
        %const = scalar.constant 1.0000000000000001e-05 : f32
        %addf_2 = scalar.addf %logisticf, %const : f32
        view.store %addf_2, %output_mix[%madd, %j] : f32, view<[%num_tokens_idx]x2xf32, %layout>
      }
    }
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tilekernels_mhc_pre_split_mixes_fwd_gfx1100",
    category="kernel",
    tags=("tilekernels", "mhc", "split", "amdgpu"),
)
def tilekernels_mhc_pre_split_mixes_fwd_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_pre_split_mixes_kernel(tilelang, T),
        args=(2, 0.5, 1e-5, 2),
        target="hip -mcpu=gfx1100",
        name="mhc_pre_split_mixes_fwd_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mhc_pre_split_mixes_fwd_kernel") @mhc_pre_split_mixes_fwd_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c2 = index.constant 2 : index
  %add = index.add %num_tokens_idx, %c2 : index
  %c1 = index.constant 1 : index
  %sub = index.sub %add, %c1 : index
  %div = index.div %sub, %c2 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%div, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%input_mixes_handle: buffer, %mhc_scale_handle: buffer, %mhc_base_handle: buffer, %pre_layer_mix_handle: buffer, %post_layer_mix_handle: buffer, %comb_res_mix_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %input_mixes_noalias = buffer.assume.noalias %input_mixes_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %input_mixes = buffer.view %input_mixes_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x8xf32, %layout>
  %mhc_scale_noalias = buffer.assume.noalias %mhc_scale_handle : buffer
  %mhc_scale = buffer.view %mhc_scale_noalias[%c0_bytes] : buffer -> view<3xf32, %layout>
  %mhc_base_noalias = buffer.assume.noalias %mhc_base_handle : buffer
  %mhc_base = buffer.view %mhc_base_noalias[%c0_bytes] : buffer -> view<8xf32, %layout>
  %pre_layer_mix_noalias = buffer.assume.noalias %pre_layer_mix_handle : buffer
  %pre_layer_mix = buffer.view %pre_layer_mix_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %post_layer_mix_noalias = buffer.assume.noalias %post_layer_mix_handle : buffer
  %post_layer_mix = buffer.view %post_layer_mix_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2xf32, %layout>
  %comb_res_mix_noalias = buffer.assume.noalias %comb_res_mix_handle : buffer
  %comb_res_mix = buffer.view %comb_res_mix_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x4xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %input_mixes_frag_bytes = index.constant 64 : offset
  %input_mixes_frag_buffer = buffer.alloca %input_mixes_frag_bytes {base_alignment = 4, memory_space = private} : buffer
  %input_mixes_frag = buffer.view %input_mixes_frag_buffer[%c0_bytes] : buffer -> view<2x8xf32, %layout>
  %pre_layer_mix_frag_bytes = index.constant 16 : offset
  %pre_layer_mix_frag_buffer = buffer.alloca %pre_layer_mix_frag_bytes {base_alignment = 4, memory_space = private} : buffer
  %pre_layer_mix_frag = buffer.view %pre_layer_mix_frag_buffer[%c0_bytes] : buffer -> view<2x2xf32, %layout>
  %post_layer_mix_frag_buffer = buffer.alloca %pre_layer_mix_frag_bytes {base_alignment = 4, memory_space = private} : buffer
  %post_layer_mix_frag = buffer.view %post_layer_mix_frag_buffer[%c0_bytes] : buffer -> view<2x2xf32, %layout>
  %comb_res_mix_frag_bytes = index.constant 32 : offset
  %comb_res_mix_frag_buffer = buffer.alloca %comb_res_mix_frag_bytes {base_alignment = 4, memory_space = private} : buffer
  %comb_res_mix_frag = buffer.view %comb_res_mix_frag_buffer[%c0_bytes] : buffer -> view<2x4xf32, %layout>
  %c2 = index.constant 2 : index
  %mul = index.mul %bx, %c2 : index
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %c1 = index.constant 1 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c8 step %c1] {
      %idx = index.add %mul, %i0 : index
      %copy = view.load %input_mixes[%idx, %i1] : view<[%num_tokens_idx]x8xf32, %layout> -> f32
      view.store %copy, %input_mixes_frag[%i0, %i1] : f32, view<2x8xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c2 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      %load = view.load %input_mixes_frag[%i, %j] : view<2x8xf32, %layout> -> f32
      %load_2 = view.load %mhc_scale[%c0] : view<3xf32, %layout> -> f32
      %mulf = scalar.mulf %load, %load_2 : f32
      %load_3 = view.load %mhc_base[%j] : view<8xf32, %layout> -> f32
      %addf = scalar.addf %mulf, %load_3 : f32
      %logisticf = scalar.logisticf %addf : f32
      %const = scalar.constant 1.0000000000000001e-05 : f32
      %addf_2 = scalar.addf %logisticf, %const : f32
      view.store %addf_2, %pre_layer_mix_frag[%i, %j] : f32, view<2x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c2 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      %add = index.add %j, %c2 : index
      %load_4 = view.load %input_mixes_frag[%i, %add] : view<2x8xf32, %layout> -> f32
      %load_5 = view.load %mhc_scale[%c1] : view<3xf32, %layout> -> f32
      %mulf_2 = scalar.mulf %load_4, %load_5 : f32
      %add_2 = index.add %j, %c2 : index
      %load_6 = view.load %mhc_base[%add_2] : view<8xf32, %layout> -> f32
      %addf_3 = scalar.addf %mulf_2, %load_6 : f32
      %logisticf_2 = scalar.logisticf %addf_3 : f32
      %const_2 = scalar.constant 0.5 : f32
      %mulf_3 = scalar.mulf %logisticf_2, %const_2 : f32
      view.store %mulf_3, %post_layer_mix_frag[%i, %j] : f32, view<2x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c2 step %c1] {
    %c4 = index.constant 4 : index
    scf.for %j = [%c0 to %c4 step %c1] {
      %add_3 = index.add %j, %c4 : index
      %load_7 = view.load %input_mixes_frag[%i, %add_3] : view<2x8xf32, %layout> -> f32
      %load_8 = view.load %mhc_scale[%c2] : view<3xf32, %layout> -> f32
      %mulf_4 = scalar.mulf %load_7, %load_8 : f32
      %add_4 = index.add %j, %c4 : index
      %load_9 = view.load %mhc_base[%add_4] : view<8xf32, %layout> -> f32
      %addf_4 = scalar.addf %mulf_4, %load_9 : f32
      view.store %addf_4, %comb_res_mix_frag[%i, %j] : f32, view<2x4xf32, %layout>
    }
  }
  %mul_2 = index.mul %bx, %c2 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %idx_2 = index.add %mul_2, %i0 : index
      %copy_2 = view.load %pre_layer_mix_frag[%i0, %i1] : view<2x2xf32, %layout> -> f32
      view.store %copy_2, %pre_layer_mix[%idx_2, %i1] : f32, view<[%num_tokens_idx]x2xf32, %layout>
    }
  }
  %mul_3 = index.mul %bx, %c2 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %idx_3 = index.add %mul_3, %i0 : index
      %copy_3 = view.load %post_layer_mix_frag[%i0, %i1] : view<2x2xf32, %layout> -> f32
      view.store %copy_3, %post_layer_mix[%idx_3, %i1] : f32, view<[%num_tokens_idx]x2xf32, %layout>
    }
  }
  %c4_2 = index.constant 4 : index
  %mul_4 = index.mul %bx, %c2 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c4_2 step %c1] {
      %idx_4 = index.add %mul_4, %i0 : index
      %copy_4 = view.load %comb_res_mix_frag[%i0, %i1] : view<2x4xf32, %layout> -> f32
      view.store %copy_4, %comb_res_mix[%idx_4, %i1] : f32, view<[%num_tokens_idx]x4xf32, %layout>
    }
  }
  kernel.return
}
"""


# ====
@tilelang_case(
    name="tilekernels_mhc_sinkhorn_gfx1100",
    category="kernel",
    tags=("tilekernels", "mhc", "sinkhorn", "amdgpu"),
)
def tilekernels_mhc_sinkhorn_gfx1100(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_sinkhorn_kernel(tilelang, T),
        args=(2, 1, 2, 1e-6),
        target="hip -mcpu=gfx1100",
        name="mhc_sinkhorn_kernel",
    )


# ----
r"""
amdgpu.target<gfx1100> @hip_mcpu_gfx1100

kernel.def target(@hip_mcpu_gfx1100) export("mhc_sinkhorn_kernel") @mhc_sinkhorn_kernel(%num_tokens: i32) {
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %c1 = index.constant 1 : index
  %add = index.add %num_tokens_idx, %c1 : index
  %sub = index.sub %add, %c1 : index
  %c128 = index.constant 128 : index
  kernel.launch.config workgroups(%sub, %c1, %c1) workgroup_size(%c128, %c1, %c1) : index
} launch(%comb_res_mix_handle: buffer, %comb_res_mix_out_handle: buffer, %num_tokens: i32) {
  %c0_bytes = index.constant 0 : offset
  %comb_res_mix_noalias = buffer.assume.noalias %comb_res_mix_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_tokens_idx = index.cast %num_tokens : i32 to index
  %comb_res_mix = buffer.view %comb_res_mix_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2x2xf32, %layout>
  %comb_res_mix_out_noalias = buffer.assume.noalias %comb_res_mix_out_handle : buffer
  %comb_res_mix_out = buffer.view %comb_res_mix_out_noalias[%c0_bytes] : buffer -> view<[%num_tokens_idx]x2x2xf32, %layout>
  %bx = kernel.workgroup.id<x> : index
  %tx = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %comb_frag_bytes = index.constant 16 : offset
  %comb_frag_buffer = buffer.alloca %comb_frag_bytes {base_alignment = 4, memory_space = private} : buffer
  %comb_frag = buffer.view %comb_frag_buffer[%c0_bytes] : buffer -> view<1x2x2xf32, %layout>
  %row_sum_bytes = index.constant 8 : offset
  %row_sum_buffer = buffer.alloca %row_sum_bytes {base_alignment = 4, memory_space = private} : buffer
  %row_sum = buffer.view %row_sum_buffer[%c0_bytes] : buffer -> view<1x2xf32, %layout>
  %col_sum_buffer = buffer.alloca %row_sum_bytes {base_alignment = 4, memory_space = private} : buffer
  %col_sum = buffer.view %col_sum_buffer[%c0_bytes] : buffer -> view<1x2xf32, %layout>
  %row_max_buffer = buffer.alloca %row_sum_bytes {base_alignment = 4, memory_space = private} : buffer
  %row_max = buffer.view %row_max_buffer[%c0_bytes] : buffer -> view<1x2xf32, %layout>
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %copy = view.load %comb_res_mix[%bx, %i0, %i1] : view<[%num_tokens_idx]x2x2xf32, %layout> -> f32
      view.store %copy, %comb_frag[%c0, %i0, %i1] : f32, view<1x2x2xf32, %layout>
    }
  }
  scf.for %i0 = [%c0 to %c1 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity = scalar.constant -inf : f32
      %reduce = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity : f32) -> (f32) {
        %reduce_value = view.load %comb_frag[%i0, %i1, %r] : view<1x2x2xf32, %layout> -> f32
        %maxnumf = scalar.maxnumf %acc, %reduce_value : f32
        scf.yield %maxnumf : f32
      }
      view.store %reduce, %row_max[%i0, %i1] : f32, view<1x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c1 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      scf.for %k = [%c0 to %c2 step %c1] {
        %load = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
        %load_2 = view.load %row_max[%i, %j] : view<1x2xf32, %layout> -> f32
        %subf = scalar.subf %load, %load_2 : f32
        %expf = scalar.expf %subf : f32
        view.store %expf, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
      }
    }
  }
  scf.for %i0 = [%c0 to %c1 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity_2 = scalar.constant 0.0 : f32
      %reduce_2 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_2 : f32) -> (f32) {
        %reduce_value_2 = view.load %comb_frag[%i0, %i1, %r] : view<1x2x2xf32, %layout> -> f32
        %addf = scalar.addf %acc, %reduce_value_2 : f32
        scf.yield %addf : f32
      }
      view.store %reduce_2, %row_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c1 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      scf.for %k = [%c0 to %c2 step %c1] {
        %load_3 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
        %load_4 = view.load %row_sum[%i, %j] : view<1x2xf32, %layout> -> f32
        %divf = scalar.divf %load_3, %load_4 : f32
        %const = scalar.constant 9.9999999999999995e-07 : f32
        %addf_2 = scalar.addf %divf, %const : f32
        view.store %addf_2, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
      }
    }
  }
  scf.for %i0 = [%c0 to %c1 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %identity_3 = scalar.constant 0.0 : f32
      %reduce_3 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_3 : f32) -> (f32) {
        %reduce_value_3 = view.load %comb_frag[%i0, %r, %i1] : view<1x2x2xf32, %layout> -> f32
        %addf_3 = scalar.addf %acc, %reduce_value_3 : f32
        scf.yield %addf_3 : f32
      }
      view.store %reduce_3, %col_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
    }
  }
  scf.for %i = [%c0 to %c1 step %c1] {
    scf.for %j = [%c0 to %c2 step %c1] {
      scf.for %k = [%c0 to %c2 step %c1] {
        %load_5 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
        %load_6 = view.load %col_sum[%i, %k] : view<1x2xf32, %layout> -> f32
        %const_2 = scalar.constant 9.9999999999999995e-07 : f32
        %addf_4 = scalar.addf %load_6, %const_2 : f32
        %divf_2 = scalar.divf %load_5, %addf_4 : f32
        view.store %divf_2, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
      }
    }
  }
  scf.for %v = [%c0 to %c1 step %c1] {
    scf.for %i0 = [%c0 to %c1 step %c1] {
      scf.for %i1 = [%c0 to %c2 step %c1] {
        %identity_4 = scalar.constant 0.0 : f32
        %reduce_4 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_4 : f32) -> (f32) {
          %reduce_value_4 = view.load %comb_frag[%i0, %i1, %r] : view<1x2x2xf32, %layout> -> f32
          %addf_5 = scalar.addf %acc, %reduce_value_4 : f32
          scf.yield %addf_5 : f32
        }
        view.store %reduce_4, %row_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
      }
    }
    scf.for %i = [%c0 to %c1 step %c1] {
      scf.for %j = [%c0 to %c2 step %c1] {
        scf.for %k = [%c0 to %c2 step %c1] {
          %load_7 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
          %load_8 = view.load %row_sum[%i, %j] : view<1x2xf32, %layout> -> f32
          %const_3 = scalar.constant 9.9999999999999995e-07 : f32
          %addf_6 = scalar.addf %load_8, %const_3 : f32
          %divf_3 = scalar.divf %load_7, %addf_6 : f32
          view.store %divf_3, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
        }
      }
    }
    scf.for %i0 = [%c0 to %c1 step %c1] {
      scf.for %i1 = [%c0 to %c2 step %c1] {
        %identity_5 = scalar.constant 0.0 : f32
        %reduce_5 = scf.for %r = [%c0 to %c2 step %c1](%acc = %identity_5 : f32) -> (f32) {
          %reduce_value_5 = view.load %comb_frag[%i0, %r, %i1] : view<1x2x2xf32, %layout> -> f32
          %addf_7 = scalar.addf %acc, %reduce_value_5 : f32
          scf.yield %addf_7 : f32
        }
        view.store %reduce_5, %col_sum[%i0, %i1] : f32, view<1x2xf32, %layout>
      }
    }
    scf.for %i = [%c0 to %c1 step %c1] {
      scf.for %j = [%c0 to %c2 step %c1] {
        scf.for %k = [%c0 to %c2 step %c1] {
          %load_9 = view.load %comb_frag[%i, %j, %k] : view<1x2x2xf32, %layout> -> f32
          %load_10 = view.load %col_sum[%i, %k] : view<1x2xf32, %layout> -> f32
          %const_4 = scalar.constant 9.9999999999999995e-07 : f32
          %addf_8 = scalar.addf %load_10, %const_4 : f32
          %divf_4 = scalar.divf %load_9, %addf_8 : f32
          view.store %divf_4, %comb_frag[%i, %j, %k] : f32, view<1x2x2xf32, %layout>
        }
      }
    }
  }
  scf.for %i0 = [%c0 to %c2 step %c1] {
    scf.for %i1 = [%c0 to %c2 step %c1] {
      %copy_2 = view.load %comb_frag[%c0, %i0, %i1] : view<1x2x2xf32, %layout> -> f32
      view.store %copy_2, %comb_res_mix_out[%bx, %i0, %i1] : f32, view<[%num_tokens_idx]x2x2xf32, %layout>
    }
  }
  kernel.return
}
"""
