// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/buffer.h"

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_select_buffer_alloca_plan_from_facts(
    const loom_value_fact_table_t* fact_table, const loom_op_t* source_op,
    loom_amdgpu_buffer_alloca_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_buffer_alloca_plan_t){0};
  const loom_value_fact_memory_space_t memory_space =
      loom_buffer_alloca_memory_space(source_op);
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      out_plan->storage_space = LOOM_STORAGE_SPACE_WORKGROUP;
      break;
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      out_plan->storage_space = LOOM_STORAGE_SPACE_PRIVATE;
      break;
    default:
      return false;
  }
  const int64_t base_alignment = loom_buffer_alloca_base_alignment(source_op);
  if (base_alignment <= 0 || base_alignment > UINT32_MAX ||
      !loom_amdgpu_u32_is_power_of_two((uint32_t)base_alignment)) {
    return false;
  }
  int64_t byte_length = 0;
  if (fact_table == NULL ||
      !loom_value_facts_as_non_negative_i64_maximum(
          loom_value_fact_table_lookup(
              fact_table, loom_buffer_alloca_byte_length(source_op)),
          &byte_length) ||
      byte_length <= 0) {
    return false;
  }
  out_plan->byte_length = byte_length;
  out_plan->base_alignment = base_alignment;
  return true;
}

static bool loom_amdgpu_select_buffer_alloca_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_buffer_alloca_plan_t* out_plan) {
  return loom_amdgpu_select_buffer_alloca_plan_from_facts(
      loom_low_lower_context_fact_table(context), source_op, out_plan);
}

iree_status_t loom_amdgpu_select_buffer_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA: {
      loom_amdgpu_buffer_alloca_plan_t local_plan = {0};
      if (!loom_amdgpu_select_buffer_alloca_plan(context, source_op,
                                                 &local_plan)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_record_lower_alloca(
          context, source_op, (uint64_t)local_plan.byte_length));
      loom_amdgpu_buffer_alloca_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      *plan_data = local_plan;
      *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

iree_status_t loom_amdgpu_low_legality_record_buffer_op(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (op->kind != LOOM_OP_BUFFER_ALLOCA) {
    return iree_ok_status();
  }

  loom_amdgpu_buffer_alloca_plan_t plan = {0};
  if (!loom_amdgpu_select_buffer_alloca_plan_from_facts(
          loom_target_low_legality_fact_table(context), op, &plan)) {
    return iree_ok_status();
  }
  return loom_amdgpu_source_alloca_layout_record_low_legality_alloca(
      context, op, (uint64_t)plan.byte_length);
}

static iree_status_t loom_amdgpu_lower_buffer_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_buffer_alloca_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_amdgpu_source_alloca_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_source_alloca_layout_for_lower_context(context, &layout));
  loom_value_id_t storage_root = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_source_alloca_layout_lookup_low_storage(
      layout, loom_buffer_alloca_memory_space(source_op),
      loom_buffer_alloca_result(source_op), &storage_root);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, storage_root, /*offset=*/0, vgpr_type, source_op->location,
      &address_op));
  return loom_low_lower_bind_value(context,
                                   loom_buffer_alloca_result(source_op),
                                   loom_low_storage_address_result(address_op));
}

iree_status_t loom_amdgpu_lower_buffer_op(loom_low_lower_context_t* context,
                                          const loom_op_t* source_op,
                                          loom_low_lower_plan_t plan) {
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA:
      return loom_amdgpu_lower_buffer_alloca(
          context, source_op,
          (const loom_amdgpu_buffer_alloca_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE("AMDGPU buffer plan selected unknown op kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}
