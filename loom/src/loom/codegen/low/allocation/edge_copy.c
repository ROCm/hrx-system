// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/edge_copy.h"

#include <string.h>

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/unit_location.h"
#include "loom/ops/low/ops.h"

static iree_status_t loom_low_allocation_edge_copy_record_segment(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_low_allocation_edge_copy_plan_t* plan, uint16_t payload_index,
    loom_value_id_t source_value_id, loom_value_id_t destination_value_id,
    uint32_t source_unit_offset, uint32_t destination_unit_offset,
    uint32_t unit_count) {
  uint32_t source_assignment_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, source_value_id, &source_assignment_index,
          NULL));
  uint32_t destination_assignment_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_assignment_map_require_assignment_for_value(
          &context->assignment_map, destination_value_id,
          &destination_assignment_index, NULL));
  plan->copies[plan->copy_count++] = (loom_low_allocation_edge_copy_t){
      .payload_index = payload_index,
      .source_value_id = source_value_id,
      .destination_value_id = destination_value_id,
      .source_assignment_index = source_assignment_index,
      .destination_assignment_index = destination_assignment_index,
      .source_unit_offset = source_unit_offset,
      .destination_unit_offset = destination_unit_offset,
      .unit_count = unit_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_value_ordinal(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_value_id_t value_id, loom_value_ordinal_t* out_value_ordinal) {
  if (!loom_low_allocation_assignment_map_value_ordinal_for_value(
          &context->assignment_map, value_id, out_value_ordinal)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.br edge-copy value %u is outside the allocation value domain",
        (unsigned)value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_branch_relation(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_value_id_t payload_value_id, loom_value_id_t destination_value_id,
    const loom_low_placement_relation_t** out_relation) {
  *out_relation = NULL;
  loom_value_ordinal_t payload_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_value_ordinal(
      context, payload_value_id, &payload_ordinal));
  loom_value_ordinal_t destination_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_value_ordinal(
      context, destination_value_id, &destination_ordinal));

  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(context->placement,
                                                          destination_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_LOW_BRANCH &&
        relation->source_ordinal == payload_ordinal) {
      *out_relation = relation;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "low.br edge-copy payload is missing a branch placement relation");
}

static bool loom_low_allocation_edge_copy_concat_relation_covers_branch_source(
    const loom_low_placement_relation_t* concat_relation,
    const loom_low_placement_relation_t* branch_relation) {
  if (concat_relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
      concat_relation->result_ordinal != branch_relation->source_ordinal) {
    return false;
  }
  if (concat_relation->result_unit_offset <
      branch_relation->source_unit_offset) {
    return false;
  }
  if (concat_relation->unit_count >
          UINT32_MAX - concat_relation->result_unit_offset ||
      branch_relation->unit_count >
          UINT32_MAX - branch_relation->source_unit_offset) {
    return false;
  }
  const uint32_t concat_source_end =
      concat_relation->result_unit_offset + concat_relation->unit_count;
  const uint32_t branch_source_end =
      branch_relation->source_unit_offset + branch_relation->unit_count;
  return concat_source_end <= branch_source_end;
}

static iree_status_t loom_low_allocation_edge_copy_count_concat_segments(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_placement_relation_t* branch_relation,
    iree_host_size_t* out_segment_count) {
  *out_segment_count = 0;
  uint32_t covered_unit_count = 0;
  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(
          context->placement, branch_relation->source_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (!loom_low_allocation_edge_copy_concat_relation_covers_branch_source(
            relation, branch_relation)) {
      continue;
    }
    if (*out_segment_count == IREE_HOST_SIZE_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.br edge-copy count exceeds host size");
    }
    if (relation->unit_count > UINT32_MAX - covered_unit_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.br edge-copy concat units exceed u32 "
                              "range");
    }
    ++*out_segment_count;
    covered_unit_count += relation->unit_count;
  }
  if (*out_segment_count != 0 &&
      covered_unit_count != branch_relation->unit_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.br edge-copy concat placement does not cover the branch payload");
  }
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_edge_copy_count_branch_payload_segments(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_value_id_t payload_value_id, loom_value_id_t destination_value_id,
    iree_host_size_t* inout_copy_count) {
  const loom_low_placement_relation_t* branch_relation = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_branch_relation(
      context, payload_value_id, destination_value_id, &branch_relation));

  iree_host_size_t payload_copy_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_count_concat_segments(
      context, branch_relation, &payload_copy_count));
  if (payload_copy_count == 0) {
    payload_copy_count = 1;
  }
  if (payload_copy_count > IREE_HOST_SIZE_MAX - *inout_copy_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.br edge-copy count exceeds host size");
  }
  *inout_copy_count += payload_copy_count;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_edge_copy_record_branch_payload_segments(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_low_allocation_edge_copy_plan_t* plan, uint16_t payload_index,
    loom_value_id_t payload_value_id, loom_value_id_t destination_value_id) {
  const loom_low_placement_relation_t* branch_relation = NULL;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_branch_relation(
      context, payload_value_id, destination_value_id, &branch_relation));

  iree_host_size_t concat_segment_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_count_concat_segments(
      context, branch_relation, &concat_segment_count));
  if (concat_segment_count == 0) {
    return loom_low_allocation_edge_copy_record_segment(
        context, plan, payload_index, payload_value_id, destination_value_id,
        branch_relation->source_unit_offset,
        branch_relation->result_unit_offset, branch_relation->unit_count);
  }

  const loom_low_placement_relation_range_t range =
      loom_low_placement_relation_range_for_value_ordinal(
          context->placement, branch_relation->source_ordinal);
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_placement_relation_t* relation =
        &context->placement->relations[range.start + i];
    if (!loom_low_allocation_edge_copy_concat_relation_covers_branch_source(
            relation, branch_relation)) {
      continue;
    }
    const uint32_t branch_source_delta =
        relation->result_unit_offset - branch_relation->source_unit_offset;
    if (branch_relation->result_unit_offset >
        UINT32_MAX - branch_source_delta) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low.br edge-copy destination offset exceeds "
                              "u32 range");
    }
    const loom_value_id_t source_value_id = loom_low_placement_value_id(
        context->placement, relation->source_ordinal);
    IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_record_segment(
        context, plan, payload_index, source_value_id, destination_value_id,
        relation->source_unit_offset,
        branch_relation->result_unit_offset + branch_source_delta,
        relation->unit_count));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_op_program_point(
    const loom_low_allocation_edge_copy_context_t* context, const loom_op_t* op,
    uint32_t* out_program_point) {
  return loom_low_allocation_live_range_ordered_op_program_point(
      context->assignment_map.liveness, context->body, context->liveness_order,
      op, out_program_point);
}

static iree_status_t loom_low_allocation_edge_copy_record_group(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_low_allocation_edge_copy_plan_t* plan, const loom_op_t* op,
    uint32_t source_ordinal) {
  loom_value_slice_t args = loom_low_br_args(op);
  if (args.count == 0) {
    return iree_ok_status();
  }
  const loom_block_t* dest = loom_low_br_dest(op);
  if (args.count != dest->arg_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.br edge-copy payload count does not match destination block args");
  }
  if (plan->copy_count > UINT32_MAX ||
      args.count > UINT32_MAX - plan->copy_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.br edge-copy group exceeds u32 range");
  }
  uint32_t program_point = UINT32_MAX;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_op_program_point(
      context, op, &program_point));
  loom_low_allocation_edge_copy_group_t* group =
      &plan->groups[plan->group_count++];
  *group = (loom_low_allocation_edge_copy_group_t){
      .terminator_op = op,
      .source_ordinal = source_ordinal,
      .program_point = program_point,
      .copy_start = (uint32_t)plan->copy_count,
      .copy_count = 0,
      .temporary_start = 0,
      .temporary_count = 0,
  };
  const uint32_t copy_start = (uint32_t)plan->copy_count;
  for (uint16_t i = 0; i < dest->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_edge_copy_record_branch_payload_segments(
            context, plan, i, args.values[i], dest->arg_ids[i]));
  }
  if (plan->copy_count < copy_start ||
      plan->copy_count - copy_start > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low.br edge-copy group exceeds u32 range");
  }
  group->copy_count = (uint32_t)(plan->copy_count - copy_start);
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_record_groups(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_low_allocation_edge_copy_plan_t* plan) {
  uint32_t source_ordinal = 0;
  loom_block_t* block = NULL;
  loom_region_for_each_block(context->body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_br_isa(op)) {
        ++source_ordinal;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_record_group(
          context, plan, op, source_ordinal));
      ++source_ordinal;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_count_groups(
    const loom_low_allocation_edge_copy_context_t* context,
    iree_host_size_t* out_group_count, iree_host_size_t* out_copy_count) {
  *out_group_count = 0;
  *out_copy_count = 0;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(context->body, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_br_isa(op)) {
        continue;
      }
      loom_value_slice_t args = loom_low_br_args(op);
      if (args.count == 0) {
        continue;
      }
      const loom_block_t* dest = loom_low_br_dest(op);
      if (args.count != dest->arg_count) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "low.br edge-copy payload count does not "
                                "match destination block args");
      }
      if (*out_group_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy count exceeds host size");
      }
      ++*out_group_count;
      for (uint16_t i = 0; i < args.count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_edge_copy_count_branch_payload_segments(
                context, args.values[i], dest->arg_ids[i], out_copy_count));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_unit_locations(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_t* edge_copy, uint32_t unit_index,
    loom_low_allocation_unit_location_t* out_source,
    loom_low_allocation_unit_location_t* out_destination) {
  const loom_low_allocation_assignment_t* source_assignment =
      &context->assignment_map.assignments[edge_copy->source_assignment_index];
  const loom_low_allocation_assignment_t* destination_assignment =
      &context->assignment_map
           .assignments[edge_copy->destination_assignment_index];
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
      source_assignment, edge_copy->source_unit_offset + unit_index,
      out_source));
  IREE_RETURN_IF_ERROR(loom_low_allocation_assignment_unit_location(
      destination_assignment, edge_copy->destination_unit_offset + unit_index,
      out_destination));
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_group_unit_count(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    uint32_t* out_unit_count) {
  *out_unit_count = 0;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      if (*out_unit_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy unit count exceeds u32");
      }
      ++*out_unit_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_group_find_destination(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* destination, bool* out_found,
    loom_low_allocation_unit_location_t* out_source) {
  *out_found = false;
  *out_source = (loom_low_allocation_unit_location_t){0};
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t candidate_destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &candidate_destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &candidate_destination)) {
        continue;
      }
      if (!loom_low_allocation_unit_storage_classes_equal(
              destination, &candidate_destination)) {
        continue;
      }
      if (!loom_low_allocation_unit_locations_equal(destination,
                                                    &candidate_destination)) {
        continue;
      }
      if (*out_found) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation edge-copy group has duplicate destinations");
      }
      *out_found = true;
      *out_source = source;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_unit_starts_cycle(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* destination,
    const loom_low_allocation_unit_location_t* source, uint32_t max_unit_count,
    bool* out_has_cycle) {
  *out_has_cycle = false;
  loom_low_allocation_unit_location_t next_destination = *source;
  for (uint32_t step = 0; step < max_unit_count; ++step) {
    if (loom_low_allocation_unit_locations_equal(&next_destination,
                                                 destination)) {
      *out_has_cycle = true;
      return iree_ok_status();
    }
    bool found = false;
    loom_low_allocation_unit_location_t next_source = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_group_find_destination(
        context, plan, group, &next_destination, &found, &next_source));
    if (!found) {
      return iree_ok_status();
    }
    next_destination = next_source;
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "low allocation edge-copy cycle is malformed");
}

static iree_status_t loom_low_allocation_edge_copy_class_seen_before(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* storage_class,
    uint32_t stop_copy_index, uint32_t stop_unit_index, bool* out_seen) {
  *out_seen = false;
  for (uint32_t copy_index = 0; copy_index <= stop_copy_index; ++copy_index) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + copy_index];
    uint32_t unit_limit = edge_copy->unit_count;
    if (copy_index == stop_copy_index) {
      unit_limit = stop_unit_index;
    }
    for (uint32_t unit_index = 0; unit_index < unit_limit; ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      if (loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                         &destination)) {
        *out_seen = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_class_has_cycle(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* storage_class,
    bool* out_has_cycle) {
  *out_has_cycle = false;
  uint32_t unit_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_group_unit_count(
      context, plan, group, &unit_count));
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_storage_classes_equal(storage_class,
                                                          &destination) ||
          !loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_starts_cycle(
          context, plan, group, &destination, &source, unit_count,
          out_has_cycle));
      if (*out_has_cycle) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_count_temporaries_for_group(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    iree_host_size_t* inout_temporary_count) {
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      bool seen_class = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_seen_before(
          context, plan, group, &destination, i, unit_index, &seen_class));
      if (seen_class) {
        continue;
      }
      bool has_cycle = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_has_cycle(
          context, plan, group, &destination, &has_cycle));
      if (!has_cycle) {
        continue;
      }
      if (*inout_temporary_count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low.br edge-copy temporary count overflow");
      }
      ++*inout_temporary_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_group_uses_location(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* location, bool* out_uses) {
  *out_uses = false;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      if (loom_low_allocation_unit_locations_equal(location, &source) ||
          loom_low_allocation_unit_locations_equal(location, &destination)) {
        *out_uses = true;
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_find_temporary(
    const loom_low_allocation_edge_copy_context_t* context,
    const loom_low_allocation_edge_copy_plan_t* plan,
    const loom_low_allocation_edge_copy_group_t* group,
    const loom_low_allocation_unit_location_t* storage_class,
    loom_low_allocation_unit_location_t* out_temporary) {
  *out_temporary = (loom_low_allocation_unit_location_t){0};
  if (!loom_low_allocation_location_kind_is_register_like(
          storage_class->location_kind)) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
        context->target_constraints, group->terminator_op,
        storage_class->value_class, 0, 1,
        IREE_SV("edge-copy-non-register-storage")));
    return iree_ok_status();
  }

  loom_low_allocation_class_capacity_t capacity = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_class_capacity(
      context->target_constraints, storage_class->value_class, &capacity));
  if (capacity.location_kind != storage_class->location_kind) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
        context->target_constraints, group->terminator_op,
        storage_class->value_class,
        capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
        IREE_SV("edge-copy-storage-kind-mismatch")));
    return iree_ok_status();
  }

  uint32_t last_location = 0;
  if (capacity.is_bounded) {
    if (capacity.max_units == 0) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints, group->terminator_op,
          storage_class->value_class, capacity.max_units, 1,
          IREE_SV("edge-copy-empty-budget")));
      return iree_ok_status();
    }
    last_location = capacity.max_units - 1u;
  } else {
    last_location =
        loom_low_allocation_target_constraints_assigned_location_search_limit(
            context->target_constraints, storage_class->descriptor_reg_class_id,
            storage_class->location_kind);
    if (last_location == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
          context->target_constraints, group->terminator_op,
          storage_class->value_class, UINT32_MAX, 1,
          IREE_SV("edge-copy-location-range-overflow")));
      return iree_ok_status();
    }
  }

  for (uint32_t location = 0; location <= last_location; ++location) {
    loom_low_allocation_unit_location_t temporary = {
        .location_kind = storage_class->location_kind,
        .value_class = storage_class->value_class,
        .descriptor_reg_class_id = storage_class->descriptor_reg_class_id,
        .location = location,
    };
    bool group_uses_location = false;
    IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_group_uses_location(
        context, plan, group, &temporary, &group_uses_location));
    if (loom_low_allocation_target_constraints_reserved_range_conflicts(
            context->target_constraints, temporary.descriptor_reg_class_id,
            temporary.location_kind, temporary.location, 1) ||
        loom_low_allocation_unit_location_is_live_at_point(
            context->descriptor_set, context->assignment_map.assignments,
            context->assignment_map.assignment_count,
            context->unit_liveness->end_points,
            context->unit_liveness->end_point_count, &temporary,
            group->program_point) ||
        group_uses_location) {
      if (location == UINT32_MAX) {
        break;
      }
      continue;
    }
    *out_temporary = temporary;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_target_constraints_emit_failure(
      context->target_constraints, group->terminator_op,
      storage_class->value_class,
      capacity.is_bounded ? capacity.max_units : UINT32_MAX, 1,
      IREE_SV("edge-copy-no-scratch-unit")));
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_edge_copy_plan_record_temporaries_for_group(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_low_allocation_edge_copy_plan_t* plan,
    loom_low_allocation_edge_copy_group_t* group) {
  if (plan->temporary_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low.br edge-copy temporary group exceeds u32 range");
  }
  group->temporary_start = (uint32_t)plan->temporary_count;
  for (uint32_t i = 0; i < group->copy_count; ++i) {
    const loom_low_allocation_edge_copy_t* edge_copy =
        &plan->copies[group->copy_start + i];
    for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
         ++unit_index) {
      loom_low_allocation_unit_location_t source = {0};
      loom_low_allocation_unit_location_t destination = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_unit_locations(
          context, edge_copy, unit_index, &source, &destination));
      if (!loom_low_allocation_unit_locations_form_register_move(
              &source, &destination)) {
        continue;
      }
      bool seen_class = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_seen_before(
          context, plan, group, &destination, i, unit_index, &seen_class));
      if (seen_class) {
        continue;
      }
      bool has_cycle = false;
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_class_has_cycle(
          context, plan, group, &destination, &has_cycle));
      if (!has_cycle) {
        continue;
      }
      loom_low_allocation_unit_location_t temporary = {0};
      IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_find_temporary(
          context, plan, group, &destination, &temporary));
      if (context->target_constraints->error_count != 0) {
        return iree_ok_status();
      }
      if (plan->temporary_count > UINT32_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low.br edge-copy temporary group exceeds u32 range");
      }
      plan->temporaries[plan->temporary_count++] =
          (loom_low_allocation_edge_copy_temporary_t){
              .value_class = temporary.value_class,
              .descriptor_reg_class_id = temporary.descriptor_reg_class_id,
              .location_kind = temporary.location_kind,
              .location = temporary.location,
          };
      ++group->temporary_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_edge_copy_record_temporaries(
    const loom_low_allocation_edge_copy_context_t* context,
    loom_low_allocation_edge_copy_plan_t* plan, iree_arena_allocator_t* arena) {
  iree_host_size_t temporary_count = 0;
  for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_edge_copy_count_temporaries_for_group(
            context, plan, &plan->groups[i], &temporary_count));
  }
  if (temporary_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, temporary_count,
                                                 sizeof(*plan->temporaries),
                                                 (void**)&plan->temporaries));
  memset(plan->temporaries, 0, temporary_count * sizeof(*plan->temporaries));
  for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_edge_copy_plan_record_temporaries_for_group(
            context, plan, &plan->groups[i]));
    if (context->target_constraints->error_count != 0) {
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_edge_copy_plan_build(
    const loom_low_allocation_edge_copy_context_t* context,
    iree_arena_allocator_t* arena,
    loom_low_allocation_edge_copy_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_low_allocation_edge_copy_plan_t){0};

  iree_host_size_t group_count = 0;
  iree_host_size_t copy_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_edge_copy_count_groups(
      context, &group_count, &copy_count));
  if (copy_count == 0) {
    return iree_ok_status();
  }

  loom_low_allocation_edge_copy_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, copy_count, sizeof(*plan.copies), (void**)&plan.copies));
  memset(plan.copies, 0, copy_count * sizeof(*plan.copies));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*plan.groups), (void**)&plan.groups));
  memset(plan.groups, 0, group_count * sizeof(*plan.groups));

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_edge_copy_record_groups(context, &plan));
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_edge_copy_record_temporaries(context, &plan, arena));
  *out_plan = plan;
  return iree_ok_status();
}
