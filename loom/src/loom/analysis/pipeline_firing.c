// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_firing.h"

#include <string.h>

enum {
  LOOM_PIPELINE_FIRING_RECORD = 0,
  LOOM_PIPELINE_FIRING_COMPLETION = 1,
};

static bool loom_pipeline_firing_shapes_equal(
    loom_pipeline_plan_record_shape_t lhs,
    loom_pipeline_plan_record_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint8_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dimensions[i] != rhs.dimensions[i]) return false;
  }
  return true;
}

static const loom_pipeline_plan_flow_t* loom_pipeline_firing_stage_flow(
    const loom_pipeline_plan_t* plan, const loom_pipeline_plan_stage_t* stage) {
  if (stage->input_count == 0 && stage->output_count == 0) return NULL;
  return &plan->flows[plan->stage_ports[stage->port_start].flow_index];
}

static iree_status_t loom_pipeline_firing_define_groups(
    const loom_pipeline_plan_t* plan, loom_pipeline_firing_group_t* groups) {
  for (uint32_t i = 0; i < plan->stage_count; ++i) {
    const loom_pipeline_plan_stage_t* stage = &plan->stages[i];
    loom_pipeline_firing_group_t* group = &groups[stage->group_index];
    const loom_pipeline_plan_flow_t* flow =
        loom_pipeline_firing_stage_flow(plan, stage);
    const loom_pipeline_plan_record_shape_t shape =
        flow != NULL ? flow->record_shape
                     : (loom_pipeline_plan_record_shape_t){0};
    const uint32_t record_count = flow != NULL ? flow->record_count : 1;
    if (stage->fold_record_count != 0) {
      if (group->fold_count != 0 &&
          !loom_pipeline_firing_shapes_equal(group->record_shape, shape)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "pipeline group %u folds require one record shape; stage %u "
            "requires a separate or nested frame",
            stage->group_index, i);
      }
      group->record_shape = shape;
      group->records_per_frame = stage->fold_record_count;
      group->frame_count = record_count / stage->fold_record_count;
      ++group->fold_count;
    } else if (group->records_per_frame == 0) {
      group->record_shape = shape;
      group->records_per_frame = 1;
      group->frame_count = record_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_firing_classify_stages(
    const loom_pipeline_plan_t* plan, loom_pipeline_firing_group_t* groups,
    uint8_t* stage_phases) {
  for (uint32_t i = 0; i < plan->stage_count; ++i) {
    const loom_pipeline_plan_stage_t* stage = &plan->stages[i];
    loom_pipeline_firing_group_t* group = &groups[stage->group_index];
    const loom_pipeline_plan_flow_t* flow =
        loom_pipeline_firing_stage_flow(plan, stage);
    const loom_pipeline_plan_record_shape_t shape =
        flow != NULL ? flow->record_shape
                     : (loom_pipeline_plan_record_shape_t){0};
    bool is_completion = false;
    for (uint16_t port = 0; port < stage->input_count; ++port) {
      const loom_pipeline_plan_flow_t* input =
          &plan->flows[plan->stage_ports[stage->port_start + port].flow_index];
      if (input->producer_kind != LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE ||
          input->group_index != stage->group_index) {
        continue;
      }
      const uint32_t producer_index = input->producer_stage_index;
      if (stage_phases[producer_index] == LOOM_PIPELINE_FIRING_COMPLETION ||
          plan->stages[producer_index].fold_record_count != 0) {
        is_completion = true;
      }
    }
    loom_pipeline_plan_record_shape_t frame_shape = group->record_shape;
    if (group->fold_count != 0 && frame_shape.rank != 0) --frame_shape.rank;
    if (!loom_pipeline_firing_shapes_equal(shape, group->record_shape)) {
      if (group->fold_count == 0 ||
          !loom_pipeline_firing_shapes_equal(shape, frame_shape)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "pipeline group %u stage %u record shape is neither the shared "
            "record cadence nor its completed frame",
            stage->group_index, i);
      }
      is_completion = true;
    }
    if (is_completion &&
        (stage->fold_record_count != 0 ||
         !loom_pipeline_firing_shapes_equal(shape, frame_shape))) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "pipeline group %u stage %u requires nested frame execution",
          stage->group_index, i);
    }
    stage_phases[i] = is_completion ? LOOM_PIPELINE_FIRING_COMPLETION
                                    : LOOM_PIPELINE_FIRING_RECORD;
    if (is_completion) {
      ++group->completion_stage_count;
    } else {
      ++group->record_stage_count;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_firing_plan_build(
    const loom_pipeline_plan_t* plan, iree_arena_allocator_t* arena,
    loom_pipeline_firing_plan_t* out_firing) {
  *out_firing = (loom_pipeline_firing_plan_t){0};
  loom_pipeline_firing_group_t* groups = NULL;
  uint32_t* stage_indices = NULL;
  uint8_t* stage_phases = NULL;
  uint32_t* positions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->group_count, sizeof(*groups), (void**)&groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->stage_count,
                                                 sizeof(*stage_indices),
                                                 (void**)&stage_indices));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->stage_count, sizeof(*stage_phases), (void**)&stage_phases));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan->group_count, 2 * sizeof(*positions), (void**)&positions));
  memset(groups, 0, plan->group_count * sizeof(*groups));
  IREE_RETURN_IF_ERROR(loom_pipeline_firing_define_groups(plan, groups));
  IREE_RETURN_IF_ERROR(
      loom_pipeline_firing_classify_stages(plan, groups, stage_phases));
  uint32_t stage_start = 0;
  for (uint32_t i = 0; i < plan->group_count; ++i) {
    groups[i].stage_start = stage_start;
    positions[2 * i] = stage_start;
    positions[2 * i + 1] = stage_start + groups[i].record_stage_count;
    stage_start += plan->groups[i].stage_count;
  }
  for (uint32_t i = 0; i < plan->stage_count; ++i) {
    const uint32_t group_index = plan->stages[i].group_index;
    stage_indices[positions[2 * group_index + stage_phases[i]]++] = i;
  }
  *out_firing = (loom_pipeline_firing_plan_t){
      .groups = groups,
      .stage_indices = stage_indices,
  };
  return iree_ok_status();
}
