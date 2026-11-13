// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/workgroup_storage.h"

#include "loom/analysis/storage_layout.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/plan.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"

iree_status_t loom_amdgpu_validate_workgroup_storage(
    loom_low_lower_context_t* context) {
  uint64_t byte_extent = 0;

  // Entry setup emits the packed source-allocation arena before any source-op
  // reservations, so its retained physical requirement begins the layout.
  const loom_amdgpu_source_alloca_layout_t* source_alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &source_alloca_layout));
  loom_amdgpu_source_alloca_storage_requirement_t source_requirement = {0};
  if (loom_amdgpu_source_alloca_layout_storage_requirement(
          source_alloca_layout, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
          &source_requirement)) {
    byte_extent = source_requirement.byte_length;
  }

  const iree_host_size_t selected_plan_count =
      loom_low_lower_context_selected_plan_count(context);
  // Selected plans are stored in source emission order. Append only the exact
  // reservations retained by plans that will reach body emission.
  const loom_op_t* overflow_op = NULL;
  for (iree_host_size_t i = 0; i < selected_plan_count && !overflow_op; ++i) {
    const loom_low_lower_selected_plan_view_t selected_plan =
        loom_low_lower_context_selected_plan_view(context, i);
    if (selected_plan.elided) {
      continue;
    }
    uint32_t scratch_byte_length = 0;
    switch (selected_plan.plan.id) {
      case LOOM_OP_KERNEL_WORKGROUP_REDUCE:
        scratch_byte_length = ((const loom_amdgpu_workgroup_reduce_plan_t*)
                                   selected_plan.plan.target_data)
                                  ->scratch_byte_length;
        break;
      case LOOM_OP_KERNEL_WORKGROUP_SCAN:
        scratch_byte_length = ((const loom_amdgpu_workgroup_scan_plan_t*)
                                   selected_plan.plan.target_data)
                                  ->scratch_byte_length;
        break;
      default:
        continue;
    }
    if (scratch_byte_length == 0) {
      continue;
    }
    if (!loom_storage_layout_append(scratch_byte_length, /*byte_alignment=*/4,
                                    &byte_extent, /*out_byte_offset=*/NULL)) {
      overflow_op = selected_plan.source_op;
    }
  }
  if (overflow_op) {
    const loom_diagnostic_param_t params[] = {
        loom_param_string(loom_low_lower_context_function_name(context)),
        loom_param_string(IREE_SV("workgroup")),
    };
    return loom_low_lower_emit_error_ref(context, overflow_op,
                                         LOOM_ERR_TARGET_096_REF, params,
                                         IREE_ARRAYSIZE(params));
  }
  const uint64_t byte_limit = loom_low_lower_context_bundle(context)
                                  ->snapshot->max_workgroup_storage_bytes;
  if (byte_limit == 0 || byte_extent <= byte_limit) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_u64(byte_extent),
      loom_param_u64(byte_limit),
  };
  return loom_low_lower_emit_error_ref(
      context, loom_low_lower_context_source_function(context).op,
      LOOM_ERR_TARGET_051_REF, params, IREE_ARRAYSIZE(params));
}
