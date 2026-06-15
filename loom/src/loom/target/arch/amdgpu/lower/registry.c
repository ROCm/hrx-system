// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic.h"
#include "loom/target/arch/amdgpu/contracts/arithmetic_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/async.h"
#include "loom/target/arch/amdgpu/contracts/async_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/buffer.h"
#include "loom/target/arch/amdgpu/contracts/buffer_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/compare.h"
#include "loom/target/arch/amdgpu/contracts/compare_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/config.h"
#include "loom/target/arch/amdgpu/contracts/config_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/dot.h"
#include "loom/target/arch/amdgpu/contracts/dot_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/integer.h"
#include "loom/target/arch/amdgpu/contracts/integer_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/matrix.h"
#include "loom/target/arch/amdgpu/contracts/reduce.h"
#include "loom/target/arch/amdgpu/contracts/reduce_lower_rules.h"
#include "loom/target/arch/amdgpu/contracts/view.h"
#include "loom/target/arch/amdgpu/contracts/view_lower_rules.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/abi.h"
#include "loom/target/arch/amdgpu/lower/arithmetic.h"
#include "loom/target/arch/amdgpu/lower/async.h"
#include "loom/target/arch/amdgpu/lower/bitfield.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/buffer.h"
#include "loom/target/arch/amdgpu/lower/control.h"
#include "loom/target/arch/amdgpu/lower/dot.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/mask.h"
#include "loom/target/arch/amdgpu/lower/matrix.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/preamble.h"
#include "loom/target/arch/amdgpu/lower/structural.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/table.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/values.h"
#include "loom/target/arch/amdgpu/lower/workgroup.h"

typedef struct loom_amdgpu_lower_dispatch_row_t
    loom_amdgpu_lower_dispatch_row_t;

typedef iree_status_t (*loom_amdgpu_lower_select_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan);

typedef iree_status_t (*loom_amdgpu_lower_emit_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan);

typedef iree_status_t (*loom_amdgpu_lower_verify_fn_t)(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled);

typedef struct loom_amdgpu_lower_dispatch_row_t {
  // Source op kind covered by this AMDGPU lowering row.
  loom_op_kind_t source_op_kind;
  // Bytes of plan data allocated by typed data selectors. Direct selector
  // hooks keep this zero and own any mixed plan allocation themselves.
  iree_host_size_t plan_data_size;
  // Optional source-to-low plan selection hook.
  loom_amdgpu_lower_select_fn_t select;
  // Optional source-to-low plan emission hook.
  loom_amdgpu_lower_emit_fn_t emit;
  // Optional target-low legality verifier hook.
  loom_amdgpu_lower_verify_fn_t verify;
} loom_amdgpu_lower_dispatch_row_t;

#define LOOM_AMDGPU_DEFINE_DATA_SELECT(name, plan_type, select_fn)             \
  static iree_status_t name(loom_low_lower_context_t* context,                 \
                            const loom_op_t* source_op,                        \
                            const loom_amdgpu_lower_dispatch_row_t* row,       \
                            loom_low_lower_plan_t* out_plan) {                 \
    plan_type* plan_data = NULL;                                               \
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(                    \
        context, row->plan_data_size, (void**)&plan_data));                    \
    bool selected = false;                                                     \
    IREE_RETURN_IF_ERROR(select_fn(context, source_op, plan_data, &selected)); \
    if (selected) {                                                            \
      *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);        \
    }                                                                          \
    return iree_ok_status();                                                   \
  }

#define LOOM_AMDGPU_DEFINE_DATA_EMIT(name, plan_type, emit_fn)              \
  static iree_status_t name(loom_low_lower_context_t* context,              \
                            const loom_op_t* source_op,                     \
                            const loom_amdgpu_lower_dispatch_row_t* row,    \
                            loom_low_lower_plan_t plan) {                   \
    (void)row;                                                              \
    return emit_fn(context, source_op, (const plan_type*)plan.target_data); \
  }

static iree_status_t loom_amdgpu_select_value_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_value_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_value_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  return loom_amdgpu_lower_value_op(context, source_op, plan);
}

static iree_status_t loom_amdgpu_select_buffer_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_buffer_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_buffer_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  return loom_amdgpu_lower_buffer_op(context, source_op, plan);
}

static iree_status_t loom_amdgpu_select_preamble_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_preamble_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_preamble_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  (void)plan;
  return loom_amdgpu_lower_preamble_op(context, source_op);
}

static iree_status_t loom_amdgpu_select_kernel_barrier_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row,
    loom_low_lower_plan_t* out_plan) {
  (void)row;
  return loom_amdgpu_select_kernel_barrier_plan(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_emit_kernel_barrier_dispatch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_lower_dispatch_row_t* row, loom_low_lower_plan_t plan) {
  (void)row;
  return loom_amdgpu_lower_kernel_barrier(
      context, source_op,
      (const loom_amdgpu_kernel_barrier_plan_t*)plan.target_data);
}

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_broadcast_dispatch,
    loom_amdgpu_subgroup_broadcast_plan_t,
    loom_amdgpu_select_kernel_subgroup_broadcast_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(
    loom_amdgpu_emit_kernel_subgroup_broadcast_dispatch,
    loom_amdgpu_subgroup_broadcast_plan_t,
    loom_amdgpu_lower_kernel_subgroup_broadcast)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_broadcast_first_dispatch,
    loom_amdgpu_subgroup_broadcast_first_plan_t,
    loom_amdgpu_select_kernel_subgroup_broadcast_first_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(
    loom_amdgpu_emit_kernel_subgroup_broadcast_first_dispatch,
    loom_amdgpu_subgroup_broadcast_first_plan_t,
    loom_amdgpu_lower_kernel_subgroup_broadcast_first)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_shuffle_dispatch,
    loom_amdgpu_subgroup_shuffle_plan_t,
    loom_amdgpu_select_kernel_subgroup_shuffle_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_shuffle_dispatch,
                             loom_amdgpu_subgroup_shuffle_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_shuffle)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_reduce_dispatch,
    loom_amdgpu_subgroup_reduce_plan_t,
    loom_amdgpu_select_kernel_subgroup_reduce_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_reduce_dispatch,
                             loom_amdgpu_subgroup_reduce_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_reduce)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_workgroup_reduce_dispatch,
    loom_amdgpu_workgroup_reduce_plan_t,
    loom_amdgpu_select_kernel_workgroup_reduce_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_workgroup_reduce_dispatch,
                             loom_amdgpu_workgroup_reduce_plan_t,
                             loom_amdgpu_lower_kernel_workgroup_reduce)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_subgroup_scan_dispatch,
                               loom_amdgpu_subgroup_scan_plan_t,
                               loom_amdgpu_select_kernel_subgroup_scan_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_scan_dispatch,
                             loom_amdgpu_subgroup_scan_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_scan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_workgroup_scan_dispatch,
    loom_amdgpu_workgroup_scan_plan_t,
    loom_amdgpu_select_kernel_workgroup_scan_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_workgroup_scan_dispatch,
                             loom_amdgpu_workgroup_scan_plan_t,
                             loom_amdgpu_lower_kernel_workgroup_scan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_active_mask_dispatch,
    loom_amdgpu_subgroup_active_mask_plan_t,
    loom_amdgpu_select_kernel_subgroup_active_mask_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(
    loom_amdgpu_emit_kernel_subgroup_active_mask_dispatch,
    loom_amdgpu_subgroup_active_mask_plan_t,
    loom_amdgpu_lower_kernel_subgroup_active_mask)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_ballot_dispatch,
    loom_amdgpu_subgroup_ballot_plan_t,
    loom_amdgpu_select_kernel_subgroup_ballot_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_ballot_dispatch,
                             loom_amdgpu_subgroup_ballot_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_ballot)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_vote_any_dispatch,
    loom_amdgpu_subgroup_vote_any_plan_t,
    loom_amdgpu_select_kernel_subgroup_vote_any_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_vote_any_dispatch,
                             loom_amdgpu_subgroup_vote_any_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_vote_any)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_kernel_subgroup_vote_all_dispatch,
    loom_amdgpu_subgroup_vote_all_plan_t,
    loom_amdgpu_select_kernel_subgroup_vote_all_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_subgroup_vote_all_dispatch,
                             loom_amdgpu_subgroup_vote_all_plan_t,
                             loom_amdgpu_lower_kernel_subgroup_vote_all)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_async_gather_dispatch,
                               loom_amdgpu_async_gather_plan_t,
                               loom_amdgpu_select_kernel_async_gather_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_async_gather_dispatch,
                             loom_amdgpu_async_gather_plan_t,
                             loom_amdgpu_lower_kernel_async_gather)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_kernel_async_wait_dispatch,
                               loom_amdgpu_async_wait_plan_t,
                               loom_amdgpu_select_kernel_async_wait_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_kernel_async_wait_dispatch,
                             loom_amdgpu_async_wait_plan_t,
                             loom_amdgpu_lower_kernel_async_wait)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_cmpi_dispatch,
                               loom_amdgpu_vector_compare_plan_t,
                               loom_amdgpu_select_vector_cmpi_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_cmpi_dispatch,
                             loom_amdgpu_vector_compare_plan_t,
                             loom_amdgpu_lower_vector_cmpi)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_cmpf_dispatch,
                               loom_amdgpu_vector_compare_plan_t,
                               loom_amdgpu_select_vector_cmpf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_cmpf_dispatch,
                             loom_amdgpu_vector_compare_plan_t,
                             loom_amdgpu_lower_vector_cmpf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_cmpf_dispatch,
                               loom_amdgpu_vector_compare_plan_t,
                               loom_amdgpu_select_scalar_cmpf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_cmpf_dispatch,
                             loom_amdgpu_vector_compare_plan_t,
                             loom_amdgpu_lower_scalar_cmpf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_dotf_dispatch,
                               loom_amdgpu_dotf_plan_t,
                               loom_amdgpu_select_vector_dotf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_dotf_dispatch,
                             loom_amdgpu_dotf_plan_t,
                             loom_amdgpu_lower_vector_dotf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_clampf_dispatch,
                               loom_amdgpu_clampf_plan_t,
                               loom_amdgpu_select_scalar_clampf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_clampf_dispatch,
                             loom_amdgpu_clampf_plan_t,
                             loom_amdgpu_lower_clampf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_fragment_load_dispatch,
                               loom_amdgpu_fragment_memory_plan_t,
                               loom_amdgpu_select_vector_fragment_load_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_fragment_load_dispatch,
                             loom_amdgpu_fragment_memory_plan_t,
                             loom_amdgpu_lower_vector_fragment_load)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_vector_fragment_store_dispatch,
    loom_amdgpu_fragment_memory_plan_t,
    loom_amdgpu_select_vector_fragment_store_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_fragment_store_dispatch,
                             loom_amdgpu_fragment_memory_plan_t,
                             loom_amdgpu_lower_vector_fragment_store)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_select_dispatch,
                               loom_amdgpu_vector_select_plan_t,
                               loom_amdgpu_select_vector_select_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_select_dispatch,
                             loom_amdgpu_vector_select_plan_t,
                             loom_amdgpu_lower_select)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scf_select_dispatch,
                               loom_amdgpu_vector_select_plan_t,
                               loom_amdgpu_select_scf_select_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scf_select_dispatch,
                             loom_amdgpu_vector_select_plan_t,
                             loom_amdgpu_lower_select)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_clampf_dispatch,
                               loom_amdgpu_clampf_plan_t,
                               loom_amdgpu_select_vector_clampf_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_clampf_dispatch,
                             loom_amdgpu_clampf_plan_t,
                             loom_amdgpu_lower_clampf)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_table_lookup_dispatch,
                               loom_amdgpu_table_lookup_plan_t,
                               loom_amdgpu_select_vector_table_lookup_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_table_lookup_dispatch,
                             loom_amdgpu_table_lookup_plan_t,
                             loom_amdgpu_lower_vector_table_lookup)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_vector_bitfield_extract_dispatch,
    loom_amdgpu_bitfield_extract_plan_t,
    loom_amdgpu_select_vector_bitfield_extract_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitfield_extract_dispatch,
                             loom_amdgpu_bitfield_extract_plan_t,
                             loom_amdgpu_lower_vector_bitfield_extract)

LOOM_AMDGPU_DEFINE_DATA_SELECT(
    loom_amdgpu_select_vector_bitfield_insert_dispatch,
    loom_amdgpu_bitfield_insert_plan_t,
    loom_amdgpu_select_vector_bitfield_insert_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitfield_insert_dispatch,
                             loom_amdgpu_bitfield_insert_plan_t,
                             loom_amdgpu_lower_vector_bitfield_insert)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_fmaf_mix_dispatch,
                               loom_amdgpu_fma_mix_plan_t,
                               loom_amdgpu_select_scalar_fmaf_mix_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_fmaf_mix_dispatch,
                             loom_amdgpu_fma_mix_plan_t,
                             loom_amdgpu_lower_scalar_fmaf_mix)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_packed_fmaf_dispatch,
                               loom_amdgpu_packed_ternary_plan_t,
                               loom_amdgpu_select_vector_packed_fmaf_plan)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_packed_fmai_dispatch,
                               loom_amdgpu_packed_ternary_plan_t,
                               loom_amdgpu_select_vector_packed_fmai_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_packed_ternary_dispatch,
                             loom_amdgpu_packed_ternary_plan_t,
                             loom_amdgpu_lower_vector_packed_ternary)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_scalar_mulf_mix_dispatch,
                               loom_amdgpu_mulf_mix_plan_t,
                               loom_amdgpu_select_scalar_mulf_mix_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_scalar_mulf_mix_dispatch,
                             loom_amdgpu_mulf_mix_plan_t,
                             loom_amdgpu_lower_mulf_mix)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_mulf_mix_dispatch,
                               loom_amdgpu_mulf_mix_plan_t,
                               loom_amdgpu_select_vector_mulf_mix_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_mulf_mix_dispatch,
                             loom_amdgpu_mulf_mix_plan_t,
                             loom_amdgpu_lower_mulf_mix)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_bitpack_dispatch,
                               loom_amdgpu_bitpack_plan_t,
                               loom_amdgpu_select_vector_bitpack_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitpack_dispatch,
                             loom_amdgpu_bitpack_plan_t,
                             loom_amdgpu_lower_vector_bitpack)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_bitunpack_dispatch,
                               loom_amdgpu_bitunpack_plan_t,
                               loom_amdgpu_select_vector_bitunpack_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitunpack_dispatch,
                             loom_amdgpu_bitunpack_plan_t,
                             loom_amdgpu_lower_vector_bitunpack)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_bitcast_dispatch,
                               loom_amdgpu_vector_bitcast_plan_t,
                               loom_amdgpu_select_vector_bitcast_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_bitcast_dispatch,
                             loom_amdgpu_vector_bitcast_plan_t,
                             loom_amdgpu_lower_vector_bitcast)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_concat_dispatch,
                               loom_amdgpu_vector_concat_plan_t,
                               loom_amdgpu_select_vector_concat_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_concat_dispatch,
                             loom_amdgpu_vector_concat_plan_t,
                             loom_amdgpu_lower_vector_concat)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_deinterleave_dispatch,
                               loom_amdgpu_vector_deinterleave_plan_t,
                               loom_amdgpu_select_vector_deinterleave_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_deinterleave_dispatch,
                             loom_amdgpu_vector_deinterleave_plan_t,
                             loom_amdgpu_lower_vector_deinterleave)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_interleave_dispatch,
                               loom_amdgpu_vector_interleave_plan_t,
                               loom_amdgpu_select_vector_interleave_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_interleave_dispatch,
                             loom_amdgpu_vector_interleave_plan_t,
                             loom_amdgpu_lower_vector_interleave)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_shuffle_dispatch,
                               loom_amdgpu_vector_shuffle_plan_t,
                               loom_amdgpu_select_vector_shuffle_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_shuffle_dispatch,
                             loom_amdgpu_vector_shuffle_plan_t,
                             loom_amdgpu_lower_vector_shuffle)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_transpose_dispatch,
                               loom_amdgpu_vector_transpose_plan_t,
                               loom_amdgpu_select_vector_transpose_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_transpose_dispatch,
                             loom_amdgpu_vector_transpose_plan_t,
                             loom_amdgpu_lower_vector_transpose)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_vector_slice_dispatch,
                               loom_amdgpu_vector_slice_plan_t,
                               loom_amdgpu_select_vector_slice_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_vector_slice_dispatch,
                             loom_amdgpu_vector_slice_plan_t,
                             loom_amdgpu_lower_vector_slice)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_memory_load_dispatch,
                               loom_amdgpu_memory_access_plan_t,
                               loom_amdgpu_select_memory_load_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_memory_load_dispatch,
                             loom_amdgpu_memory_access_plan_t,
                             loom_amdgpu_lower_memory_load)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_memory_store_dispatch,
                               loom_amdgpu_memory_access_plan_t,
                               loom_amdgpu_select_memory_store_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_memory_store_dispatch,
                             loom_amdgpu_memory_access_plan_t,
                             loom_amdgpu_lower_memory_store)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_view_atomic_dispatch,
                               loom_amdgpu_atomic_plan_t,
                               loom_amdgpu_select_view_atomic_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_view_atomic_dispatch,
                             loom_amdgpu_atomic_plan_t,
                             loom_amdgpu_lower_view_atomic)

LOOM_AMDGPU_DEFINE_DATA_SELECT(loom_amdgpu_select_view_prefetch_dispatch,
                               loom_amdgpu_prefetch_plan_t,
                               loom_amdgpu_select_view_prefetch_plan)

LOOM_AMDGPU_DEFINE_DATA_EMIT(loom_amdgpu_emit_view_prefetch_dispatch,
                             loom_amdgpu_prefetch_plan_t,
                             loom_amdgpu_lower_view_prefetch)

#undef LOOM_AMDGPU_DEFINE_DATA_SELECT
#undef LOOM_AMDGPU_DEFINE_DATA_EMIT

#include "loom/target/arch/amdgpu/lower/registry_tables.inl"  // IWYU pragma: keep

static const loom_amdgpu_lower_dispatch_row_t*
loom_amdgpu_lower_dispatch_row_from_table(
    loom_op_kind_t op_kind, const loom_amdgpu_lower_dispatch_row_t* rows,
    iree_host_size_t row_count) {
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >= row_count) {
    return NULL;
  }
  const loom_amdgpu_lower_dispatch_row_t* row = &rows[op_index];
  return row->source_op_kind == op_kind ? row : NULL;
}

static const loom_amdgpu_lower_dispatch_row_t*
loom_amdgpu_find_lower_dispatch_row(loom_low_lower_plan_id_t plan_id) {
  if (plan_id > UINT16_MAX) {
    return NULL;
  }
  const loom_op_kind_t op_kind = (loom_op_kind_t)plan_id;
  switch (loom_op_dialect_id(op_kind)) {
    case LOOM_DIALECT_INDEX:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuIndexDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuIndexDispatchRows));
    case LOOM_DIALECT_SCALAR:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuScalarDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuScalarDispatchRows));
    case LOOM_DIALECT_SCF:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuScfDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuScfDispatchRows));
    case LOOM_DIALECT_BUFFER:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuBufferDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuBufferDispatchRows));
    case LOOM_DIALECT_VIEW:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuViewDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuViewDispatchRows));
    case LOOM_DIALECT_VECTOR:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuVectorDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuVectorDispatchRows));
    case LOOM_DIALECT_KERNEL:
      return loom_amdgpu_lower_dispatch_row_from_table(
          op_kind, kAmdgpuKernelDispatchRows,
          IREE_ARRAYSIZE(kAmdgpuKernelDispatchRows));
    default:
      return NULL;
  }
}

static iree_status_t loom_amdgpu_select_plan_id(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(source_op->kind);
  if (row == NULL || row->select == NULL) {
    return iree_ok_status();
  }
  return row->select(context, source_op, row, out_plan);
}

static iree_status_t loom_amdgpu_select_op(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  return loom_amdgpu_select_plan_id(context, source_op, out_plan);
}

static iree_status_t loom_amdgpu_preselect_op(void* user_data,
                                              loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  if (loom_amdgpu_value_plan_needs_preselection(source_op)) {
    return loom_amdgpu_preselect_value_plan(context, source_op, out_plan);
  }
  if (!loom_vector_dotf_isa(source_op) && !loom_index_add_isa(source_op) &&
      !loom_index_cmp_isa(source_op) && !loom_scalar_fmaf_isa(source_op) &&
      !loom_vector_fmaf_isa(source_op) && !loom_vector_fmai_isa(source_op) &&
      !loom_scalar_mulf_isa(source_op) && !loom_vector_mulf_isa(source_op)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_plan_id(context, source_op, out_plan));
  if (loom_low_lower_plan_is_empty(*out_plan) &&
      (loom_scalar_fmaf_isa(source_op) || loom_vector_fmaf_isa(source_op))) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fmaf_literal_operand_form_diagnostic(
        context, source_op));
  }
  return iree_ok_status();
}

static void loom_amdgpu_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  if (plan.id == LOOM_OP_SCALAR_FMAF) {
    loom_amdgpu_mark_fma_mix_plan_storage_demands(
        context, source_op,
        (const loom_amdgpu_fma_mix_plan_t*)plan.target_data);
    return;
  }
  if (plan.id == LOOM_OP_VECTOR_FMAF || plan.id == LOOM_OP_VECTOR_FMAI) {
    loom_amdgpu_mark_packed_ternary_plan_storage_demands(
        context, source_op,
        (const loom_amdgpu_packed_ternary_plan_t*)plan.target_data);
    return;
  }
  if (plan.id == LOOM_OP_SCALAR_MULF || plan.id == LOOM_OP_VECTOR_MULF) {
    loom_amdgpu_mark_mulf_mix_plan_storage_demands(
        context, source_op,
        (const loom_amdgpu_mulf_mix_plan_t*)plan.target_data);
    return;
  }
  if (plan.id == LOOM_OP_VECTOR_FROM_ELEMENTS ||
      plan.id == LOOM_OP_VECTOR_SPLAT || plan.id == LOOM_OP_VECTOR_INSERT) {
    loom_amdgpu_mark_value_plan_storage_demands(context, source_op, plan);
    return;
  }
  loom_low_lower_require_source_operands_storage(context, source_op);
}

static iree_string_view_t loom_amdgpu_workgroup_reduce_plan_detail(
    const loom_amdgpu_workgroup_reduce_plan_t* plan) {
  switch (plan->publication_kind) {
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LDS:
      return IREE_SV("amdgpu.workgroup_reduce.publication.lds");
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP:
      return IREE_SV("amdgpu.workgroup_reduce.publication.redundant_subgroup");
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_LEADER_WORKITEM:
      return IREE_SV("amdgpu.workgroup_reduce.publication.leader_workitem");
    case LOOM_AMDGPU_WORKGROUP_REDUCE_PUBLICATION_REDUNDANT_SUBGROUP_LEADER_LANE:
      return IREE_SV(
          "amdgpu.workgroup_reduce.publication."
          "redundant_subgroup_leader_lane");
    default:
      return iree_string_view_empty();
  }
}

static iree_string_view_t loom_amdgpu_describe_plan(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  (void)context;
  (void)source_op;
  if (plan.id == LOOM_OP_KERNEL_WORKGROUP_REDUCE && plan.target_data != NULL) {
    return loom_amdgpu_workgroup_reduce_plan_detail(
        (const loom_amdgpu_workgroup_reduce_plan_t*)plan.target_data);
  }
  return iree_string_view_empty();
}

static iree_status_t loom_amdgpu_emit_op(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_low_lower_plan_t plan) {
  (void)user_data;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(plan.id);
  return row->emit(context, source_op, row, plan);
}

static iree_status_t loom_amdgpu_low_legality_try_verify_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  const loom_amdgpu_lower_dispatch_row_t* row =
      loom_amdgpu_find_lower_dispatch_row(op->kind);
  if (row == NULL || row->verify == NULL) {
    return iree_ok_status();
  }
  return row->verify(provider, context, op, out_handled);
}

#define LOOM_AMDGPU_RULE_SET_FRAGMENTS(F)                                   \
  F(ARITHMETIC, loom_amdgpu_arithmetic_contract_fragment,                   \
    loom_amdgpu_arithmetic_lower_rule_set)                                  \
  F(BUFFER, loom_amdgpu_buffer_contract_fragment,                           \
    loom_amdgpu_buffer_lower_rule_set)                                      \
  F(CONFIG, loom_amdgpu_config_contract_fragment,                           \
    loom_amdgpu_config_lower_rule_set)                                      \
  F(INTEGER, loom_amdgpu_integer_contract_fragment,                         \
    loom_amdgpu_integer_lower_rule_set)                                     \
  F(COMPARE, loom_amdgpu_compare_contract_fragment,                         \
    loom_amdgpu_compare_lower_rule_set)                                     \
  F(DOT, loom_amdgpu_dot_contract_fragment, loom_amdgpu_dot_lower_rule_set) \
  F(REDUCE, loom_amdgpu_reduce_contract_fragment,                           \
    loom_amdgpu_reduce_lower_rule_set)                                      \
  F(ASYNC, loom_amdgpu_async_contract_fragment,                             \
    loom_amdgpu_async_lower_rule_set)                                       \
  F(VIEW, loom_amdgpu_view_contract_fragment, loom_amdgpu_view_lower_rule_set)

// clang-format off
typedef enum loom_amdgpu_rule_set_index_e {
#define LOOM_AMDGPU_RULE_SET_ENUM(name, fragment, rule_set) \
  LOOM_AMDGPU_RULE_SET_INDEX_##name,
  LOOM_AMDGPU_RULE_SET_FRAGMENTS(LOOM_AMDGPU_RULE_SET_ENUM)
#undef LOOM_AMDGPU_RULE_SET_ENUM
  LOOM_AMDGPU_RULE_SET_INDEX_COUNT_,
} loom_amdgpu_rule_set_index_t;

static const loom_low_lower_rule_set_t* const kAmdgpuRuleSets[] = {
#define LOOM_AMDGPU_RULE_SET_ROW(name, fragment, rule_set) &rule_set,
  LOOM_AMDGPU_RULE_SET_FRAGMENTS(LOOM_AMDGPU_RULE_SET_ROW)
#undef LOOM_AMDGPU_RULE_SET_ROW
};

static const loom_target_contract_binding_t kAmdgpuContractBindings[] = {
#define LOOM_AMDGPU_CONTRACT_BINDING_ROW(name, fragment, rule_set) \
  {&fragment, LOOM_AMDGPU_RULE_SET_INDEX_##name},
  LOOM_AMDGPU_RULE_SET_FRAGMENTS(LOOM_AMDGPU_CONTRACT_BINDING_ROW)
#undef LOOM_AMDGPU_CONTRACT_BINDING_ROW
  {&loom_amdgpu_matrix_contract_fragment, UINT16_MAX},
};
// clang-format on

static_assert(IREE_ARRAYSIZE(kAmdgpuRuleSets) ==
                  LOOM_AMDGPU_RULE_SET_INDEX_COUNT_,
              "AMDGPU rule-set index enum must match the rule-set table");

static const loom_low_lower_policy_t kAmdgpuLowLowerPolicy = {
    .name = IREE_SVL("amdgpu-register-lower"),
    .error_catalog = &loom_amdgpu_error_catalog,
    .map_type = {.fn = loom_amdgpu_map_type, .user_data = NULL},
    .map_value = {.fn = loom_amdgpu_map_value, .user_data = NULL},
    .map_contract_value = {.fn = loom_amdgpu_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_amdgpu_map_argument, .user_data = NULL},
    .emit_preamble = {.fn = loom_amdgpu_emit_preamble, .user_data = NULL},
    .emit_entry_setup = {.fn = loom_amdgpu_emit_entry_setup, .user_data = NULL},
    .prepare_branch = {.fn = loom_amdgpu_prepare_branch, .user_data = NULL},
    .materialize_branch_arg = {.fn = loom_amdgpu_materialize_branch_arg,
                               .user_data = NULL},
    .emit_cond_branch = {.fn = loom_amdgpu_emit_cond_branch, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kAmdgpuRuleSets),
            .values = kAmdgpuRuleSets,
        },
    .contract_bindings = kAmdgpuContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kAmdgpuContractBindings),
    .descriptor_matrix =
        {
            .options = loom_amdgpu_descriptor_matrix_options,
            .query = loom_amdgpu_descriptor_matrix_query,
            .attrs = loom_amdgpu_descriptor_matrix_attrs,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_amdgpu_preselect_op, .user_data = NULL},
    .select_op = {.fn = loom_amdgpu_select_op, .user_data = NULL},
    .mark_plan_storage_demands = {.fn = loom_amdgpu_mark_plan_storage_demands,
                                  .user_data = NULL},
    .describe_plan = {.fn = loom_amdgpu_describe_plan, .user_data = NULL},
    .emit_op = {.fn = loom_amdgpu_emit_op, .user_data = NULL},
};

const loom_target_low_legality_provider_t
    loom_amdgpu_low_legality_provider_storage = {
        .name = IREE_SVL("amdgpu"),
        .builtin_dialect_bits =
            (1u << LOOM_DIALECT_INDEX) | (1u << LOOM_DIALECT_BUFFER) |
            (1u << LOOM_DIALECT_SCALAR) | (1u << LOOM_DIALECT_VIEW) |
            (1u << LOOM_DIALECT_VECTOR) | (1u << LOOM_DIALECT_KERNEL),
        .try_verify_op = loom_amdgpu_low_legality_try_verify_op,
};

const loom_low_lower_policy_t* loom_amdgpu_low_lower_policy(void) {
  return &kAmdgpuLowLowerPolicy;
}

const loom_target_low_legality_provider_t* loom_amdgpu_low_legality_provider(
    void) {
  return &loom_amdgpu_low_legality_provider_storage;
}

void loom_amdgpu_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  // clang-format off
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
    {.contract_set_key = IREE_SVL("amdgpu.cdna3.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.cdna4.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.rdna3.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.rdna4.core"), .policy = &kAmdgpuLowLowerPolicy},
    {.contract_set_key = IREE_SVL("amdgpu.rdna4.gfx125x.core"), .policy = &kAmdgpuLowLowerPolicy},
  };
  // clang-format on
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
