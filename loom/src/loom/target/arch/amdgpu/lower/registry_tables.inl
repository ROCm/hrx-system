// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low callback dispatch rows. This file is included by
// registry.c after the target-specific callback shims are declared.

#define LOOM_AMDGPU_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))

#define LOOM_AMDGPU_DIRECT_ROW(op_kind, select_fn, emit_fn, verify_fn) \
  {                                                                    \
      /*.source_op_kind=*/(op_kind), /*.plan_data_size=*/0,            \
      /*.select=*/(select_fn),       /*.emit=*/(emit_fn),              \
      /*.verify=*/(verify_fn),                                         \
  }

#define LOOM_AMDGPU_DATA_ROW(op_kind, plan_type, select_fn, emit_fn, \
                             verify_fn)                              \
  {                                                                  \
      /*.source_op_kind=*/(op_kind),                                 \
      /*.plan_data_size=*/sizeof(plan_type),                         \
      /*.select=*/(select_fn),                                       \
      /*.emit=*/(emit_fn),                                           \
      /*.verify=*/(verify_fn),                                       \
  }

#define LOOM_AMDGPU_LEGALITY_ROW(op_kind, verify_fn) \
  {                                                  \
      /*.source_op_kind=*/(op_kind),                 \
      /*.plan_data_size=*/0,                         \
      /*.select=*/NULL,                              \
      /*.emit=*/NULL,                                \
      /*.verify=*/(verify_fn),                       \
  }

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuIndexDispatchRows[LOOM_OP_INDEX_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_CONSTANT)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_INDEX_CONSTANT, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_CAST)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_INDEX_CAST, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_ADD)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_INDEX_ADD, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch,
            loom_amdgpu_low_legality_verify_offset_add),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_CMP)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_INDEX_CMP, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch,
            loom_amdgpu_low_legality_verify_offset_compare),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuScalarDispatchRows[LOOM_OP_SCALAR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CONSTANT)] =
            LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_SCALAR_CONSTANT,
                                   loom_amdgpu_select_value_dispatch,
                                   loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_TRUNCI)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_SCALAR_TRUNCI, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_EXTSI)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_SCALAR_EXTSI, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CMPF)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_SCALAR_CMPF, loom_amdgpu_vector_compare_plan_t,
            loom_amdgpu_select_scalar_cmpf_dispatch,
            loom_amdgpu_emit_scalar_cmpf_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_CLAMPF)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_SCALAR_CLAMPF, loom_amdgpu_clampf_plan_t,
            loom_amdgpu_select_scalar_clampf_dispatch,
            loom_amdgpu_emit_scalar_clampf_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuScfDispatchRows[LOOM_OP_SCF_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCF_SELECT)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_SCF_SELECT, loom_amdgpu_vector_select_plan_t,
            loom_amdgpu_select_scf_select_dispatch,
            loom_amdgpu_emit_scf_select_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuBufferDispatchRows[LOOM_OP_BUFFER_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_BUFFER_ALLOCA)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_BUFFER_ALLOCA, loom_amdgpu_select_buffer_dispatch,
            loom_amdgpu_emit_buffer_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuViewDispatchRows[LOOM_OP_VIEW_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_LOAD)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VIEW_LOAD, loom_amdgpu_memory_access_plan_t,
            loom_amdgpu_select_memory_load_dispatch,
            loom_amdgpu_emit_memory_load_dispatch,
            loom_amdgpu_low_legality_verify_memory),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_STORE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VIEW_STORE, loom_amdgpu_memory_access_plan_t,
            loom_amdgpu_select_memory_store_dispatch,
            loom_amdgpu_emit_memory_store_dispatch,
            loom_amdgpu_low_legality_verify_memory),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_REDUCE)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_REDUCE,
                                 loom_amdgpu_atomic_plan_t,
                                 loom_amdgpu_select_view_atomic_dispatch,
                                 loom_amdgpu_emit_view_atomic_dispatch,
                                 loom_amdgpu_low_legality_verify_view_atomic),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_RMW)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VIEW_ATOMIC_RMW, loom_amdgpu_atomic_plan_t,
            loom_amdgpu_select_view_atomic_dispatch,
            loom_amdgpu_emit_view_atomic_dispatch,
            loom_amdgpu_low_legality_verify_view_atomic),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_ATOMIC_CMPXCHG)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VIEW_ATOMIC_CMPXCHG,
                                 loom_amdgpu_atomic_plan_t,
                                 loom_amdgpu_select_view_atomic_dispatch,
                                 loom_amdgpu_emit_view_atomic_dispatch,
                                 loom_amdgpu_low_legality_verify_view_atomic),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VIEW_PREFETCH)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VIEW_PREFETCH, loom_amdgpu_prefetch_plan_t,
            loom_amdgpu_select_view_prefetch_dispatch,
            loom_amdgpu_emit_view_prefetch_dispatch, NULL),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuVectorDispatchRows[LOOM_OP_VECTOR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CONSTANT)] =
            LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_CONSTANT,
                                   loom_amdgpu_select_value_dispatch,
                                   loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_IOTA)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_IOTA, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch,
            loom_amdgpu_low_legality_verify_vector_iota),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SPLAT)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_SPLAT, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_DOTF)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_DOTF, loom_amdgpu_dotf_plan_t,
                                 loom_amdgpu_select_vector_dotf_dispatch,
                                 loom_amdgpu_emit_vector_dotf_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CMPI)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_CMPI, loom_amdgpu_vector_compare_plan_t,
            loom_amdgpu_select_vector_cmpi_dispatch,
            loom_amdgpu_emit_vector_cmpi_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CMPF)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_CMPF, loom_amdgpu_vector_compare_plan_t,
            loom_amdgpu_select_vector_cmpf_dispatch,
            loom_amdgpu_emit_vector_cmpf_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FRAGMENT_LOAD)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_FRAGMENT_LOAD,
                loom_amdgpu_fragment_memory_plan_t,
                loom_amdgpu_select_vector_fragment_load_dispatch,
                loom_amdgpu_emit_vector_fragment_load_dispatch,
                loom_amdgpu_low_legality_verify_vector_fragment_memory),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FRAGMENT_STORE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_FRAGMENT_STORE,
                loom_amdgpu_fragment_memory_plan_t,
                loom_amdgpu_select_vector_fragment_store_dispatch,
                loom_amdgpu_emit_vector_fragment_store_dispatch,
                loom_amdgpu_low_legality_verify_vector_fragment_memory),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SELECT)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_SELECT, loom_amdgpu_vector_select_plan_t,
            loom_amdgpu_select_vector_select_dispatch,
            loom_amdgpu_emit_vector_select_dispatch,
            loom_amdgpu_low_legality_verify_vector_select),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CLAMPF)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_CLAMPF, loom_amdgpu_clampf_plan_t,
            loom_amdgpu_select_vector_clampf_dispatch,
            loom_amdgpu_emit_vector_clampf_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_TABLE_LOOKUP)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_TABLE_LOOKUP, loom_amdgpu_table_lookup_plan_t,
                loom_amdgpu_select_vector_table_lookup_dispatch,
                loom_amdgpu_emit_vector_table_lookup_dispatch,
                loom_amdgpu_low_legality_verify_vector_table),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITFIELD_EXTRACTU)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
                loom_amdgpu_bitfield_extract_plan_t,
                loom_amdgpu_select_vector_bitfield_extract_dispatch,
                loom_amdgpu_emit_vector_bitfield_extract_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITFIELD_EXTRACTS)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
                loom_amdgpu_bitfield_extract_plan_t,
                loom_amdgpu_select_vector_bitfield_extract_dispatch,
                loom_amdgpu_emit_vector_bitfield_extract_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITFIELD_INSERT)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITFIELD_INSERT,
                loom_amdgpu_bitfield_insert_plan_t,
                loom_amdgpu_select_vector_bitfield_insert_dispatch,
                loom_amdgpu_emit_vector_bitfield_insert_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITPACK)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_BITPACK, loom_amdgpu_bitpack_plan_t,
            loom_amdgpu_select_vector_bitpack_dispatch,
            loom_amdgpu_emit_vector_bitpack_dispatch,
            loom_amdgpu_low_legality_verify_vector_bitstream),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITUNPACKS)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITUNPACKS, loom_amdgpu_bitunpack_plan_t,
                loom_amdgpu_select_vector_bitunpack_dispatch,
                loom_amdgpu_emit_vector_bitunpack_dispatch,
                loom_amdgpu_low_legality_verify_vector_bitstream),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITUNPACKU)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_BITUNPACKU, loom_amdgpu_bitunpack_plan_t,
                loom_amdgpu_select_vector_bitunpack_dispatch,
                loom_amdgpu_emit_vector_bitunpack_dispatch,
                loom_amdgpu_low_legality_verify_vector_bitstream),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITCAST)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_BITCAST, loom_amdgpu_vector_bitcast_plan_t,
            loom_amdgpu_select_vector_bitcast_dispatch,
            loom_amdgpu_emit_vector_bitcast_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_CONCAT)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_CONCAT, loom_amdgpu_vector_concat_plan_t,
            loom_amdgpu_select_vector_concat_dispatch,
            loom_amdgpu_emit_vector_concat_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_DEINTERLEAVE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_DEINTERLEAVE,
                loom_amdgpu_vector_deinterleave_plan_t,
                loom_amdgpu_select_vector_deinterleave_dispatch,
                loom_amdgpu_emit_vector_deinterleave_dispatch,
                loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_INTERLEAVE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_VECTOR_INTERLEAVE, loom_amdgpu_vector_interleave_plan_t,
                loom_amdgpu_select_vector_interleave_dispatch,
                loom_amdgpu_emit_vector_interleave_dispatch,
                loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SHUFFLE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_SHUFFLE, loom_amdgpu_vector_shuffle_plan_t,
            loom_amdgpu_select_vector_shuffle_dispatch,
            loom_amdgpu_emit_vector_shuffle_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_TRANSPOSE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_TRANSPOSE, loom_amdgpu_vector_transpose_plan_t,
            loom_amdgpu_select_vector_transpose_dispatch,
            loom_amdgpu_emit_vector_transpose_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_SLICE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_SLICE, loom_amdgpu_vector_slice_plan_t,
            loom_amdgpu_select_vector_slice_dispatch,
            loom_amdgpu_emit_vector_slice_dispatch,
            loom_amdgpu_low_legality_verify_vector_structural),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_EXTRACT)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_EXTRACT, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FROM_ELEMENTS)] =
            LOOM_AMDGPU_DIRECT_ROW(LOOM_OP_VECTOR_FROM_ELEMENTS,
                                   loom_amdgpu_select_value_dispatch,
                                   loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_INSERT)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_INSERT, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_EXTF)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_EXTF, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FPTRUNC)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_VECTOR_FPTRUNC, loom_amdgpu_select_value_dispatch,
            loom_amdgpu_emit_value_dispatch, NULL),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_LOAD)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_LOAD, loom_amdgpu_memory_access_plan_t,
            loom_amdgpu_select_memory_load_dispatch,
            loom_amdgpu_emit_memory_load_dispatch,
            loom_amdgpu_low_legality_verify_memory),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_STORE)] = LOOM_AMDGPU_DATA_ROW(
            LOOM_OP_VECTOR_STORE, loom_amdgpu_memory_access_plan_t,
            loom_amdgpu_select_memory_store_dispatch,
            loom_amdgpu_emit_memory_store_dispatch,
            loom_amdgpu_low_legality_verify_memory),
};

static const loom_amdgpu_lower_dispatch_row_t
    kAmdgpuKernelDispatchRows[LOOM_OP_KERNEL_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_BARRIER)] = LOOM_AMDGPU_DIRECT_ROW(
            LOOM_OP_KERNEL_BARRIER, loom_amdgpu_select_kernel_barrier_dispatch,
            loom_amdgpu_emit_kernel_barrier_dispatch,
            loom_amdgpu_low_legality_verify_kernel_barrier),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKITEM_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKGROUP_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_SIZE)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKGROUP_SIZE,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_COUNT)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKGROUP_COUNT,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_WORKITEM_DISPATCH_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_COUNT)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_COUNT,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SIZE)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_SIZE,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_LANE_ID)] =
            LOOM_AMDGPU_DIRECT_ROW(
                LOOM_OP_KERNEL_SUBGROUP_LANE_ID,
                loom_amdgpu_select_preamble_dispatch,
                loom_amdgpu_emit_preamble_dispatch,
                loom_amdgpu_low_legality_verify_kernel_preamble),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SHUFFLE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_SHUFFLE,
                loom_amdgpu_subgroup_shuffle_plan_t,
                loom_amdgpu_select_kernel_subgroup_shuffle_dispatch,
                loom_amdgpu_emit_kernel_subgroup_shuffle_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_shuffle),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_BROADCAST)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_BROADCAST,
                loom_amdgpu_subgroup_broadcast_plan_t,
                loom_amdgpu_select_kernel_subgroup_broadcast_dispatch,
                loom_amdgpu_emit_kernel_subgroup_broadcast_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_BROADCAST_FIRST,
                loom_amdgpu_subgroup_broadcast_first_plan_t,
                loom_amdgpu_select_kernel_subgroup_broadcast_first_dispatch,
                loom_amdgpu_emit_kernel_subgroup_broadcast_first_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_broadcast_first),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_REDUCE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_REDUCE,
                loom_amdgpu_subgroup_reduce_plan_t,
                loom_amdgpu_select_kernel_subgroup_reduce_dispatch,
                loom_amdgpu_emit_kernel_subgroup_reduce_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_reduce),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_SCAN)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_SCAN, loom_amdgpu_subgroup_scan_plan_t,
                loom_amdgpu_select_kernel_subgroup_scan_dispatch,
                loom_amdgpu_emit_kernel_subgroup_scan_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_scan),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_VOTE_ANY,
                loom_amdgpu_subgroup_vote_any_plan_t,
                loom_amdgpu_select_kernel_subgroup_vote_any_dispatch,
                loom_amdgpu_emit_kernel_subgroup_vote_any_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_vote_any),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_VOTE_ALL,
                loom_amdgpu_subgroup_vote_all_plan_t,
                loom_amdgpu_select_kernel_subgroup_vote_all_dispatch,
                loom_amdgpu_emit_kernel_subgroup_vote_all_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_vote_all),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_VOTE_BALLOT,
                loom_amdgpu_subgroup_ballot_plan_t,
                loom_amdgpu_select_kernel_subgroup_ballot_dispatch,
                loom_amdgpu_emit_kernel_subgroup_ballot_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_ballot),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_SUBGROUP_ACTIVE_MASK,
                loom_amdgpu_subgroup_active_mask_plan_t,
                loom_amdgpu_select_kernel_subgroup_active_mask_dispatch,
                loom_amdgpu_emit_kernel_subgroup_active_mask_dispatch,
                loom_amdgpu_low_legality_verify_kernel_subgroup_active_mask),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_SUBGROUP_MATCH_ANY,
                loom_amdgpu_low_legality_verify_kernel_subgroup_match),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_SUBGROUP_MATCH_ALL,
                loom_amdgpu_low_legality_verify_kernel_subgroup_match),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_REDUCE)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_WORKGROUP_REDUCE,
                loom_amdgpu_workgroup_reduce_plan_t,
                loom_amdgpu_select_kernel_workgroup_reduce_dispatch,
                loom_amdgpu_emit_kernel_workgroup_reduce_dispatch,
                loom_amdgpu_low_legality_verify_kernel_workgroup_reduce),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_SCAN)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_WORKGROUP_SCAN,
                loom_amdgpu_workgroup_scan_plan_t,
                loom_amdgpu_select_kernel_workgroup_scan_dispatch,
                loom_amdgpu_emit_kernel_workgroup_scan_dispatch,
                loom_amdgpu_low_legality_verify_kernel_workgroup_scan),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_VOTE_ANY,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_VOTE_ALL,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_WORKGROUP_VOTE_COUNT,
                loom_amdgpu_low_legality_verify_kernel_collective),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_CLUSTER_GATHER_MASK,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_COPY)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_COPY,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_COPY_MASK)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_COPY_MASK,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GATHER)] =
            LOOM_AMDGPU_DATA_ROW(
                LOOM_OP_KERNEL_ASYNC_GATHER, loom_amdgpu_async_gather_plan_t,
                loom_amdgpu_select_kernel_async_gather_dispatch,
                loom_amdgpu_emit_kernel_async_gather_dispatch,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GATHER_MASK)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_GATHER_MASK,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_GROUP)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_GROUP,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_TENSOR_LOAD_TO_LDS,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS)] =
            LOOM_AMDGPU_LEGALITY_ROW(
                LOOM_OP_KERNEL_ASYNC_TENSOR_STORE_FROM_LDS,
                loom_amdgpu_low_legality_verify_kernel_async),
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_KERNEL_ASYNC_WAIT)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_KERNEL_ASYNC_WAIT,
                                 loom_amdgpu_async_wait_plan_t,
                                 loom_amdgpu_select_kernel_async_wait_dispatch,
                                 loom_amdgpu_emit_kernel_async_wait_dispatch,
                                 loom_amdgpu_low_legality_verify_kernel_async),
};

#undef LOOM_AMDGPU_DIRECT_ROW
#undef LOOM_AMDGPU_DATA_ROW
#undef LOOM_AMDGPU_LEGALITY_ROW
#undef LOOM_AMDGPU_OP_INDEX
