// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/storage.h"

#include <stdint.h>

#include "loom/ir/facts.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/util/fact_table.h"

typedef enum loom_aie2p_storage_plan_kind_e {
  LOOM_AIE2P_STORAGE_PLAN_ALLOCA = 1,
} loom_aie2p_storage_plan_kind_t;

typedef struct loom_aie2p_storage_alloca_plan_t {
  // Target-Low function storage space reserved for the allocation.
  loom_storage_space_t storage_space;
  // Proven finite maximum byte length reserved for every execution.
  int64_t byte_length;
  // Source-required base alignment of the reservation.
  int64_t byte_alignment;
} loom_aie2p_storage_alloca_plan_t;

bool loom_aie2p_storage_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AIE2P_STORAGE_PLAN_ALLOCA;
}

static iree_status_t loom_aie2p_select_storage_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  loom_storage_space_t storage_space;
  switch (loom_buffer_alloca_memory_space(source_op)) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      storage_space = LOOM_STORAGE_SPACE_PRIVATE;
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      storage_space = LOOM_STORAGE_SPACE_WORKGROUP;
      break;
    default:
      return iree_ok_status();
  }

  int64_t byte_length = 0;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (!loom_value_facts_as_non_negative_i64_maximum(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &byte_length) ||
      byte_length <= 0) {
    // Diagnostic emission records an error and returns success. Claim the
    // operation so selection does not also report a missing target contract.
    *out_plan = loom_low_lower_plan_make(LOOM_AIE2P_STORAGE_PLAN_ALLOCA, NULL);
    return loom_low_lower_emit_function_storage_extent_unsupported(
        context, source_op, storage_space,
        loom_buffer_alloca_byte_length(source_op));
  }

  loom_aie2p_storage_alloca_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_allocate_plan_data(context, sizeof(*plan), (void**)&plan));
  *plan = (loom_aie2p_storage_alloca_plan_t){
      .storage_space = storage_space,
      .byte_length = byte_length,
      .byte_alignment = loom_buffer_alloca_base_alignment(source_op),
  };
  *out_plan = loom_low_lower_plan_make(LOOM_AIE2P_STORAGE_PLAN_ALLOCA, plan);
  return iree_ok_status();
}

iree_status_t loom_aie2p_select_storage_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  if (loom_buffer_alloca_isa(source_op)) {
    return loom_aie2p_select_storage_alloca(context, source_op, out_plan);
  }
  return iree_ok_status();
}

void loom_aie2p_mark_storage_plan_demands(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  (void)context;
  (void)source_op;
  switch ((loom_aie2p_storage_plan_kind_t)plan.id) {
    case LOOM_AIE2P_STORAGE_PLAN_ALLOCA:
      // Selection retains the finite byte bound, so emission has no additional
      // source-value demand.
      return;
  }
  IREE_ASSERT_UNREACHABLE("AIE2P storage demand has unknown plan kind");
}

void loom_aie2p_describe_storage_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  *out_report = (loom_low_lower_plan_report_t){0};
  switch ((loom_aie2p_storage_plan_kind_t)plan.id) {
    case LOOM_AIE2P_STORAGE_PLAN_ALLOCA:
      out_report->plan_key = IREE_SV("function-storage.alloca");
      return;
  }
  IREE_ASSERT_UNREACHABLE("AIE2P storage report has unknown plan kind");
}

static iree_status_t loom_aie2p_emit_storage_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_aie2p_storage_alloca_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, plan->byte_length, plan->byte_alignment,
      loom_type_storage(plan->storage_space), source_op->location,
      &storage_op));

  loom_type_t address_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, &address_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op), /*offset=*/0,
      address_type, source_op->location, &address_op));
  return loom_low_lower_bind_value(context,
                                   loom_buffer_alloca_result(source_op),
                                   loom_low_storage_address_result(address_op));
}

iree_status_t loom_aie2p_emit_storage_plan(loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_low_lower_plan_t plan) {
  switch ((loom_aie2p_storage_plan_kind_t)plan.id) {
    case LOOM_AIE2P_STORAGE_PLAN_ALLOCA:
      return loom_aie2p_emit_storage_alloca(
          context, source_op,
          (const loom_aie2p_storage_alloca_plan_t*)plan.target_data);
  }
  IREE_ASSERT_UNREACHABLE("AIE2P storage emission has unknown plan kind");
  IREE_BUILTIN_UNREACHABLE();
}
