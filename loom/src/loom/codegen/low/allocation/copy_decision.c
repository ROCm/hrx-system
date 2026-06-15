// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/copy_decision.h"

#include <string.h>

#include "loom/codegen/low/allocation/move_topology.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_allocation_copy_decision_plan_record(
    const loom_low_allocation_copy_decision_context_t* context,
    loom_low_allocation_copy_decision_plan_t* plan, const loom_op_t* op) {
  uint32_t source_assignment_index = 0;
  const loom_low_allocation_assignment_t* source_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_copy_source(op),
          &source_assignment_index, &source_assignment));
  uint32_t result_assignment_index = 0;
  const loom_low_allocation_assignment_t* result_assignment = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, loom_low_copy_result(op),
          &result_assignment_index, &result_assignment));
  const bool coalesced =
      loom_low_allocation_assignment_is_register_like(source_assignment) &&
      loom_low_allocation_assignment_is_register_like(result_assignment) &&
      loom_low_allocation_storage_assignment_locations_share(
          context->descriptor_set, source_assignment, result_assignment);
  plan->decisions[plan->decision_count++] =
      (loom_low_allocation_copy_decision_t){
          .source_value_id = loom_low_copy_source(op),
          .result_value_id = loom_low_copy_result(op),
          .source_assignment_index = source_assignment_index,
          .result_assignment_index = result_assignment_index,
          .kind = coalesced ? LOOM_LOW_ALLOCATION_COPY_COALESCED
                            : LOOM_LOW_ALLOCATION_COPY_MATERIALIZED,
      };
  if (coalesced) {
    ++plan->coalesced_count;
  } else {
    ++plan->materialized_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_copy_decision_count_region_copies(
    const loom_liveness_analysis_t* liveness, const loom_region_t* region,
    iree_host_size_t* inout_copy_count) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_copy_isa(op)) {
        if (*inout_copy_count == IREE_HOST_SIZE_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "low.copy decision count exceeds host size");
        }
        ++*inout_copy_count;
      }
      if (!loom_liveness_analysis_includes_region_tree(liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_copy_decision_count_region_copies(
                liveness, regions[i], inout_copy_count));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_copy_decision_plan_record_region(
    const loom_low_allocation_copy_decision_context_t* context,
    loom_low_allocation_copy_decision_plan_t* plan,
    const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_copy_isa(op)) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_copy_decision_plan_record(context, plan, op));
      }
      if (!loom_liveness_analysis_includes_region_tree(
              context->assignment_map.liveness)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_copy_decision_plan_record_region(context, plan,
                                                                 regions[i]));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_copy_decision_plan_build(
    const loom_low_allocation_copy_decision_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_copy_decision_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_low_allocation_copy_decision_plan_t){0};

  iree_host_size_t copy_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_copy_decision_count_region_copies(
      context->assignment_map.liveness, context->body, &copy_count));
  if (copy_count == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_copy_decision_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, copy_count, sizeof(*plan.decisions), (void**)&plan.decisions));
  memset(plan.decisions, 0, copy_count * sizeof(*plan.decisions));

  IREE_RETURN_IF_ERROR(loom_low_allocation_copy_decision_plan_record_region(
      context, &plan, context->body));
  *out_plan = plan;
  return iree_ok_status();
}
