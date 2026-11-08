// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/execution.h"

#include "loom/codegen/low/lower/context.h"
#include "loom/util/cfg_execution.h"
#include "loom/util/cfg_loop_nest.h"
#include "loom/util/fact_cfg.h"

static iree_status_t loom_low_lower_calculate_block_execution_counts(
    loom_low_lower_context_t* context, loom_region_t* body,
    iree_arena_allocator_t* scratch_arena,
    loom_low_lower_execution_counts_t* analysis, uint64_t* multipliers) {
  if (!iree_any_bit_set(body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    multipliers[0] = 1;
    analysis->blocks.values = multipliers;
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* region =
      loom_value_fact_table_lookup_cfg_region(context->fact_table, body);
  const loom_cfg_loop_nest_t* loops = &region->loops;
  uint64_t* trip_counts = NULL;
  if (loops->loop_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, loops->loop_count,
                                  sizeof(*trip_counts), (void**)&trip_counts));
  }
  for (iree_host_size_t i = 0; i < loops->loop_count; ++i) {
    const loom_loop_recurrence_facts_t recurrence =
        loom_value_fact_induction_facts(context->fact_table, context->module,
                                        &region->inductions[i]);
    if (recurrence.trip_count_kind != LOOM_LOOP_TRIP_COUNT_EXACT) {
      return iree_ok_status();
    }
    trip_counts[i] = recurrence.trip_count;
  }
  // The visibility heuristic consumes only whole-function exact counts.
  // Path-local uncertainty exists solely to enrich requested report rows.
  return loom_cfg_execution_counts_calculate(
      loops, trip_counts,
      loom_low_lower_context_wants_report_rows(context)
          ? &region->control_structure
          : NULL,
      &context->analysis_arena, multipliers, &analysis->blocks);
}

static iree_status_t loom_low_lower_initialize_block_execution_counts(
    loom_low_lower_context_t* context) {
  loom_low_lower_execution_counts_t* analysis =
      &context->function_analysis.execution_counts;
  if (analysis->initialized) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(context->source_function);
  if (body == NULL || body->block_count == 0) {
    analysis->initialized = true;
    return iree_ok_status();
  }
  uint64_t* multipliers = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&context->analysis_arena, body->block_count,
                                sizeof(*multipliers), (void**)&multipliers));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(context->module->arena.block_pool, &scratch_arena);
  iree_status_t status = loom_low_lower_calculate_block_execution_counts(
      context, body, &scratch_arena, analysis, multipliers);
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status)) {
    analysis->initialized = true;
  }
  return status;
}

iree_status_t loom_low_lower_source_block_execution_counts(
    loom_low_lower_context_t* context, const uint64_t** out_counts) {
  *out_counts = NULL;
  loom_low_lower_execution_counts_t* analysis =
      &context->function_analysis.execution_counts;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_initialize_block_execution_counts(context));
  if (analysis->blocks.values != NULL &&
      (analysis->blocks.unmodeled_blocks.bit_count == 0 ||
       iree_bitmap_none_set(analysis->blocks.unmodeled_blocks))) {
    *out_counts = analysis->blocks.values;
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_source_block_execution_count(
    loom_low_lower_context_t* context, uint16_t block_index,
    uint64_t* out_count, bool* out_exact) {
  *out_count = 0;
  *out_exact = false;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_initialize_block_execution_counts(context));
  const loom_low_lower_execution_counts_t* analysis =
      &context->function_analysis.execution_counts;
  if (!loom_cfg_execution_count_is_exact(&analysis->blocks, block_index)) {
    return iree_ok_status();
  }
  *out_count = analysis->blocks.values[block_index];
  *out_exact = true;
  return iree_ok_status();
}
