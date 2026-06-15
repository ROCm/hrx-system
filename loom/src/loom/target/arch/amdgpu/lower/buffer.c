// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/buffer.h"

#include <stdint.h>

#include "loom/ops/buffer/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_select_buffer_alloca_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_buffer_alloca_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_buffer_alloca_plan_t){0};
  if (loom_buffer_alloca_memory_space(source_op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }
  const int64_t base_alignment = loom_buffer_alloca_base_alignment(source_op);
  if (base_alignment <= 0 || base_alignment > UINT32_MAX ||
      !loom_amdgpu_u32_is_power_of_two((uint32_t)base_alignment)) {
    return false;
  }
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  int64_t byte_length = 0;
  if (!loom_amdgpu_value_facts_as_exact_non_negative_i64(
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

iree_status_t loom_amdgpu_select_buffer_plan(loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  switch (source_op->kind) {
    case LOOM_OP_BUFFER_ALLOCA: {
      loom_amdgpu_buffer_alloca_plan_t* plan_data = NULL;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
          context, sizeof(*plan_data), (void**)&plan_data));
      if (loom_amdgpu_select_buffer_alloca_plan(context, source_op,
                                                plan_data)) {
        *out_plan = loom_low_lower_plan_make(source_op->kind, plan_data);
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_amdgpu_lower_buffer_alloca(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_buffer_alloca_plan_t* plan) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_op_t* storage_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
      builder, plan->byte_length, plan->base_alignment,
      loom_type_storage(LOOM_STORAGE_SPACE_WORKGROUP), source_op->location,
      &storage_op));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, loom_low_storage_reserve_storage(storage_op),
      /*offset=*/0, vgpr_type, source_op->location, &address_op));
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
