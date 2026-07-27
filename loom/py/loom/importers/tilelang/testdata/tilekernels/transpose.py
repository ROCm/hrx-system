# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E501

from typing import Any

from loom.importers.check.tilelang import TileLangImportInput, tilelang_case


def _make_batched_transpose_kernel(tilelang: Any, T: Any) -> Any:
    def create_loop_layout_fn(block_x: int, num_threads: int = 256) -> Any:
        def loop_layout_fn(i: Any, j: Any) -> tuple[Any, Any]:
            elements = i * block_x + j
            forward_thread = (elements // 4) % num_threads
            forward_local = elements % 4 + elements // (num_threads * 4) * 4
            return forward_thread, forward_local

        return loop_layout_fn

    @tilelang.jit(  # type: ignore[untyped-decorator]
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_batched_transpose_kernel(
        shape_x_mod_128: int,
        shape_y_mod_128: int,
        dtype: T.dtype,
    ) -> Any:
        num_batches = T.dynamic("num_batches")
        shape_x = T.dynamic("shape_x")
        shape_y = T.dynamic("shape_y")
        stride_x = T.dynamic("stride_x")

        num_threads = 256
        block_x = 128 if shape_x_mod_128 == 0 else 64
        block_y = 128 if shape_y_mod_128 == 0 else 64
        block_k = 4
        num_threads_per_row = block_y // block_k

        loop_layout = T.Fragment(
            (block_y, block_x),
            forward_fn=create_loop_layout_fn(block_x, num_threads),
        )

        @T.prim_func  # type: ignore[untyped-decorator]
        def batched_transpose_kernel(
            x: T.StridedTensor[
                (num_batches, shape_x, shape_y),
                (shape_x * stride_x, stride_x, 1),
                dtype,
            ],
            out: T.Tensor[(num_batches, shape_y, shape_x), dtype],
        ) -> None:
            with T.Kernel(
                shape_y // block_y,
                shape_x // block_x,
                num_batches,
                threads=num_threads,
            ) as (pid_y, pid_x, pid_batch):
                out_shared = T.alloc_shared((block_y, block_x + block_k), dtype)
                thread_id = T.get_thread_binding()
                row = thread_id // num_threads_per_row
                col = thread_id % num_threads_per_row

                T.assume(shape_x % block_x == 0)
                T.assume(shape_y % block_y == 0)
                T.assume(stride_x % block_k == 0)

                tmp = T.alloc_local((block_k, block_k), dtype)
                tmp_row = T.alloc_local((block_k,), dtype)
                for i_outer in T.unroll(
                    block_x // block_k // (num_threads // num_threads_per_row)
                ):
                    i = i_outer * (num_threads // num_threads_per_row) + row
                    for j in T.unroll(block_k):
                        for k in T.vectorized(block_k):
                            tmp_row[k] = x[
                                pid_batch,
                                pid_x * block_x + i * block_k + j,
                                pid_y * block_y + col * block_k + k,
                            ]
                        for k in T.unroll(block_k):
                            tmp[k, j] = tmp_row[k]

                    for j in T.unroll(block_k):
                        swizzle_j = (j + thread_id // (8 // dtype.bytes)) % block_k
                        for k in T.vectorized(block_k):
                            out_shared[col * block_k + swizzle_j, i * block_k + k] = (
                                tmp[swizzle_j, k]
                            )

                T.sync_threads()
                for i, j in T.Parallel(block_y, block_x, loop_layout=loop_layout):
                    out[pid_batch, pid_y * block_y + i, pid_x * block_x + j] = (
                        out_shared[i, j]
                    )

        return batched_transpose_kernel

    return get_batched_transpose_kernel


def _batched_transpose_input(
    tilelang: Any, T: Any, *, target: str
) -> TileLangImportInput:
    return TileLangImportInput(
        source=_make_batched_transpose_kernel(tilelang, T),
        args=(0, 0, T.float16),
        target=target,
        name="batched_transpose_kernel",
    )


# ====
@tilelang_case(
    name="tilekernels_batched_transpose_gfx942",
    category="kernel",
    tags=("tilekernels", "transpose", "amdgpu"),
)
def tilekernels_batched_transpose_gfx942(
    tilelang: Any,
    T: Any,
) -> TileLangImportInput:
    return _batched_transpose_input(tilelang, T, target="hip -mcpu=gfx942")


# ----
r"""
amdgpu.target<gfx942> @hip_mcpu_gfx942

kernel.def target(@hip_mcpu_gfx942) export("batched_transpose_kernel") @batched_transpose_kernel(%num_batches: i32, %shape_x: i32, %shape_y: i32) {
  %shape_y_idx = index.cast %shape_y : i32 to index
  %c128 = index.constant 128 : index
  %div = index.div %shape_y_idx, %c128 : index
  %shape_x_idx = index.cast %shape_x : i32 to index
  %div_2 = index.div %shape_x_idx, %c128 : index
  %num_batches_idx = index.cast %num_batches : i32 to index
  %c256 = index.constant 256 : index
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%div, %div_2, %num_batches_idx) workgroup_size(%c256, %c1, %c1) : index
} launch(%x_handle: buffer, %out_handle: buffer, %num_batches: i32, %shape_x: i32, %shape_y: i32, %stride_x: i32) {
  %c0_bytes = index.constant 0 : offset
  %x_noalias = buffer.assume.noalias %x_handle : buffer
  %layout = encoding.layout.dense : encoding<layout>
  %num_batches_idx = index.cast %num_batches : i32 to index
  %shape_x_idx = index.cast %shape_x : i32 to index
  %shape_y_idx = index.cast %shape_y : i32 to index
  %x = buffer.view %x_noalias[%c0_bytes] : buffer -> view<[%num_batches_idx]x[%shape_x_idx]x[%shape_y_idx]xf16, %layout>
  %out_noalias = buffer.assume.noalias %out_handle : buffer
  %out = buffer.view %out_noalias[%c0_bytes] : buffer -> view<[%num_batches_idx]x[%shape_y_idx]x[%shape_x_idx]xf16, %layout>
  %bx = kernel.workgroup.id<x> : index
  %by = kernel.workgroup.id<y> : index
  %bz = kernel.workgroup.id<z> : index
  %thread_id = kernel.workitem.id<x> : index
  %ty = kernel.workitem.id<y> : index
  %tz = kernel.workitem.id<z> : index
  %out_shared_bytes = index.constant 33792 : offset
  %out_shared_buffer = buffer.alloca %out_shared_bytes {base_alignment = 2, memory_space = workgroup} : buffer
  %out_shared = buffer.view %out_shared_buffer[%c0_bytes] : buffer -> view<128x132xf16, %layout>
  %tmp_bytes = index.constant 32 : offset
  %tmp_buffer = buffer.alloca %tmp_bytes {base_alignment = 2, memory_space = private} : buffer
  %tmp = buffer.view %tmp_buffer[%c0_bytes] : buffer -> view<4x4xf16, %layout>
  %tmp_row_bytes = index.constant 8 : offset
  %tmp_row_buffer = buffer.alloca %tmp_row_bytes {base_alignment = 2, memory_space = private} : buffer
  %tmp_row = buffer.view %tmp_row_buffer[%c0_bytes] : buffer -> view<4xf16, %layout>
  %c32 = index.constant 32 : index
  %div = index.div %thread_id, %c32 : index
  %rem = index.rem %thread_id, %c32 : index
  %shape_x_assumed = scalar.assume %shape_x [mul(%shape_x, 128)] : i32
  %shape_y_assumed = scalar.assume %shape_y [mul(%shape_y, 128)] : i32
  %stride_x_assumed = scalar.assume %stride_x [mul(%stride_x, 4)] : i32
  %c0 = index.constant 0 : index
  %c4 = index.constant 4 : index
  %c1 = index.constant 1 : index
  scf.for %i_outer = [%c0 to %c4 step %c1] unroll {
    %c8 = index.constant 8 : index
    %madd = index.madd %i_outer, %c8, %div : index
    scf.for %j = [%c0 to %c4 step %c1] unroll {
      scf.for %k = [%c0 to %c4 step %c1] {
        %c128 = index.constant 128 : index
        %mul = index.mul %madd, %c4 : index
        %madd_2 = index.madd %by, %c128, %mul : index
        %add = index.add %madd_2, %j : index
        %mul_2 = index.mul %rem, %c4 : index
        %madd_3 = index.madd %bx, %c128, %mul_2 : index
        %add_2 = index.add %madd_3, %k : index
        %load = view.load %x[%bz, %add, %add_2] : view<[%num_batches_idx]x[%shape_x_idx]x[%shape_y_idx]xf16, %layout> -> f16
        view.store %load, %tmp_row[%k] : f16, view<4xf16, %layout>
      }
      scf.for %k = [%c0 to %c4 step %c1] unroll {
        %load_2 = view.load %tmp_row[%k] : view<4xf16, %layout> -> f16
        view.store %load_2, %tmp[%k, %j] : f16, view<4x4xf16, %layout>
      }
    }
    scf.for %j = [%c0 to %c4 step %c1] unroll {
      %div_2 = index.div %thread_id, %c4 : index
      %add_3 = index.add %j, %div_2 : index
      %rem_2 = index.rem %add_3, %c4 : index
      scf.for %k = [%c0 to %c4 step %c1] {
        %load_3 = view.load %tmp[%rem_2, %k] : view<4x4xf16, %layout> -> f16
        %madd_4 = index.madd %rem, %c4, %rem_2 : index
        %madd_5 = index.madd %madd, %c4, %k : index
        view.store %load_3, %out_shared[%madd_4, %madd_5] : f16, view<128x132xf16, %layout>
      }
    }
  }
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  %c128_2 = index.constant 128 : index
  scf.for %i = [%c0 to %c128_2 step %c1] {
    scf.for %j = [%c0 to %c128_2 step %c1] {
      %load_4 = view.load %out_shared[%i, %j] : view<128x132xf16, %layout> -> f16
      %madd_6 = index.madd %bx, %c128_2, %i : index
      %madd_7 = index.madd %by, %c128_2, %j : index
      view.store %load_4, %out[%bz, %madd_6, %madd_7] : f16, view<[%num_batches_idx]x[%shape_y_idx]x[%shape_x_idx]xf16, %layout>
    }
  }
  kernel.return
}
"""
