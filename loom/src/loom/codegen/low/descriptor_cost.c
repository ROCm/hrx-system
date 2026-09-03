// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_cost.h"

#include <string.h>

static iree_status_t loom_low_descriptor_cost_overflow(const char* quantity) {
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "low descriptor cost %s overflows", quantity);
}

static iree_status_t loom_low_descriptor_cost_add_u64(uint64_t value,
                                                      const char* quantity,
                                                      uint64_t* inout_total) {
  if (!iree_checked_add_u64(*inout_total, value, inout_total)) {
    return loom_low_descriptor_cost_overflow(quantity);
  }
  return iree_ok_status();
}

void loom_low_descriptor_resource_cost_initialize(
    const loom_low_descriptor_set_t* descriptor_set, uint16_t resource_id,
    loom_low_descriptor_resource_cost_t* out_cost) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(out_cost);
  IREE_ASSERT_LT(resource_id, descriptor_set->resource_count);
  const loom_low_resource_t* resource = &descriptor_set->resources[resource_id];
  IREE_ASSERT_NE(resource->capacity_per_cycle, 0);
  *out_cost = (loom_low_descriptor_resource_cost_t){
      .resource_id = resource_id,
      .resource_name = loom_low_descriptor_set_string(
          descriptor_set, resource->name_string_offset),
      .resource_kind = resource->kind,
      .resource_flags = resource->flags,
      .capacity_per_cycle = resource->capacity_per_cycle,
      .contention_group_id = resource->contention_group_id,
  };
}

iree_status_t loom_low_descriptor_resource_cost_accumulate(
    const loom_low_issue_use_t* issue_use, uint32_t occurrence_count,
    loom_low_descriptor_resource_cost_t* inout_cost) {
  IREE_ASSERT_ARGUMENT(issue_use);
  IREE_ASSERT_ARGUMENT(inout_cost);
  if (occurrence_count == 0 || issue_use->cycles == 0 ||
      issue_use->units == 0 || inout_cost->capacity_per_cycle == 0 ||
      issue_use->units > inout_cost->capacity_per_cycle ||
      issue_use->resource_id != inout_cost->resource_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid low descriptor resource cost row");
  }
  if (occurrence_count > UINT32_MAX - inout_cost->use_count) {
    return loom_low_descriptor_cost_overflow("resource use count");
  }
  uint64_t occupied_cycles = 0;
  uint64_t unit_cycles = 0;
  if (!iree_checked_mul_u64(issue_use->cycles, occurrence_count,
                            &occupied_cycles) ||
      !iree_checked_mul_u64(occupied_cycles, issue_use->units, &unit_cycles)) {
    return loom_low_descriptor_cost_overflow("resource cycle count");
  }
  IREE_RETURN_IF_ERROR(
      loom_low_descriptor_cost_add_u64(occupied_cycles, "occupied cycle count",
                                       &inout_cost->total_occupied_cycles));
  IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_add_u64(
      unit_cycles, "unit cycle count", &inout_cost->total_unit_cycles));
  inout_cost->use_count += occurrence_count;
  inout_cost->estimated_min_cycles =
      inout_cost->total_unit_cycles / inout_cost->capacity_per_cycle +
      (inout_cost->total_unit_cycles % inout_cost->capacity_per_cycle != 0);
  inout_cost->peak_units_per_cycle =
      iree_max(inout_cost->peak_units_per_cycle, issue_use->units);
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_cost_note_model_quality(
    loom_low_model_quality_t model_quality,
    loom_low_descriptor_cost_t* inout_cost) {
  if (model_quality > LOOM_LOW_MODEL_QUALITY_FALLBACK) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "descriptor recipe has invalid model quality");
  }
  if (model_quality == LOOM_LOW_MODEL_QUALITY_UNKNOWN ||
      inout_cost->model_quality == LOOM_LOW_MODEL_QUALITY_UNKNOWN) {
    inout_cost->model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN;
  } else if (model_quality > inout_cost->model_quality) {
    inout_cost->model_quality = model_quality;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_descriptor_cost_note_memory_effect(
    const loom_low_effect_t* effect, uint32_t occurrence_count,
    loom_low_descriptor_memory_cost_t* inout_cost) {
  uint64_t* operation_count = NULL;
  uint64_t* byte_count = NULL;
  uint64_t* unknown_width_count = NULL;
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
      operation_count = &inout_cost->read_operation_count;
      byte_count = &inout_cost->read_byte_count;
      unknown_width_count = &inout_cost->read_unknown_width_count;
      break;
    case LOOM_LOW_EFFECT_KIND_WRITE:
      operation_count = &inout_cost->write_operation_count;
      byte_count = &inout_cost->write_byte_count;
      unknown_width_count = &inout_cost->write_unknown_width_count;
      break;
    default:
      return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_add_u64(
      occurrence_count, "memory operation count", operation_count));
  if (effect->width_bits == 0 || effect->width_bits % 8 != 0) {
    return loom_low_descriptor_cost_add_u64(
        occurrence_count, "unknown memory width count", unknown_width_count);
  }
  uint64_t total_byte_count = 0;
  if (!iree_checked_mul_u64(effect->width_bits / 8, occurrence_count,
                            &total_byte_count)) {
    return loom_low_descriptor_cost_overflow("memory byte count");
  }
  return loom_low_descriptor_cost_add_u64(total_byte_count, "memory byte count",
                                          byte_count);
}

static iree_status_t loom_low_descriptor_cost_note_pressure_delta(
    const loom_low_pressure_delta_t* pressure_delta, uint32_t occurrence_count,
    iree_host_size_t pressure_cost_count,
    loom_low_descriptor_pressure_cost_t* pressure_costs) {
  if (pressure_delta->reg_class_id >= pressure_cost_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "descriptor recipe pressure delta references "
                            "invalid register class");
  }
  int64_t scaled_delta = 0;
  int64_t total_delta = 0;
  if (!iree_checked_mul_i64(pressure_delta->delta, occurrence_count,
                            &scaled_delta) ||
      !iree_checked_add_i64(
          pressure_costs[pressure_delta->reg_class_id].total_unit_delta,
          scaled_delta, &total_delta)) {
    return loom_low_descriptor_cost_overflow("register pressure delta");
  }
  pressure_costs[pressure_delta->reg_class_id].total_unit_delta = total_delta;
  return iree_ok_status();
}

static bool loom_low_descriptor_cost_span_is_valid(uint32_t start,
                                                   uint32_t count,
                                                   uint32_t capacity) {
  return start <= capacity && count <= capacity - start;
}

iree_status_t loom_low_descriptor_cost_compute(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_recipe_t* recipe, iree_arena_allocator_t* arena,
    loom_low_descriptor_cost_t* out_cost) {
  IREE_ASSERT_ARGUMENT(descriptor_set);
  IREE_ASSERT_ARGUMENT(recipe);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_cost);
  *out_cost = (loom_low_descriptor_cost_t){
      .descriptor_set = descriptor_set,
      .model_quality = LOOM_LOW_MODEL_QUALITY_EXACT,
  };
  if ((recipe->entry_count != 0) != (recipe->entries != NULL) ||
      (recipe->dependency_count != 0) != (recipe->dependencies != NULL) ||
      (recipe->durable_pressure_delta_count != 0) !=
          (recipe->durable_pressure_deltas != NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "descriptor recipe has inconsistent row storage");
  }
  if (descriptor_set->resource_count > UINT16_MAX ||
      descriptor_set->reg_class_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "verified descriptor set exceeds cost index "
                            "capacity");
  }

  loom_low_descriptor_resource_cost_t* resource_costs = NULL;
  if (descriptor_set->resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, descriptor_set->resource_count, sizeof(*resource_costs),
        (void**)&resource_costs));
    for (uint16_t i = 0; i < descriptor_set->resource_count; ++i) {
      loom_low_descriptor_resource_cost_initialize(descriptor_set, i,
                                                   &resource_costs[i]);
    }
  }

  loom_low_descriptor_pressure_cost_t* pressure_costs = NULL;
  if (descriptor_set->reg_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, descriptor_set->reg_class_count, sizeof(*pressure_costs),
        (void**)&pressure_costs));
    memset(pressure_costs, 0,
           descriptor_set->reg_class_count * sizeof(*pressure_costs));
    for (uint16_t i = 0; i < descriptor_set->reg_class_count; ++i) {
      pressure_costs[i].reg_class_id = i;
    }
  }

  uint64_t* entry_critical_path_cycles = NULL;
  if (recipe->entry_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, recipe->entry_count, sizeof(*entry_critical_path_cycles),
        (void**)&entry_critical_path_cycles));
  }

  for (uint16_t i = 0; i < recipe->entry_count; ++i) {
    const loom_low_descriptor_recipe_entry_t* entry = &recipe->entries[i];
    if (entry->occurrence_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descriptor recipe entry has zero occurrences");
    }
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              entry->descriptor_ordinal);
    if (descriptor == NULL || descriptor_set->descriptor_views == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descriptor recipe references invalid "
                              "descriptor ordinal");
    }
    const uint16_t schedule_class_id =
        descriptor_set->descriptor_views[entry->descriptor_ordinal]
            .schedule_class_id;
    if (schedule_class_id >= descriptor_set->schedule_class_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "descriptor recipe references invalid schedule "
                              "class");
    }
    const loom_low_schedule_class_t* schedule_class =
        &descriptor_set->schedule_classes[schedule_class_id];
    if (!loom_low_descriptor_cost_span_is_valid(
            schedule_class->issue_use_start, schedule_class->issue_use_count,
            descriptor_set->issue_use_count) ||
        !loom_low_descriptor_cost_span_is_valid(
            schedule_class->pressure_delta_start,
            schedule_class->pressure_delta_count,
            descriptor_set->pressure_delta_count) ||
        !loom_low_descriptor_cost_span_is_valid(descriptor->effect_start,
                                                descriptor->effect_count,
                                                descriptor_set->effect_count)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "descriptor recipe references malformed "
                              "descriptor spans");
    }
    IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_add_u64(
        entry->occurrence_count, "instruction count",
        &out_cost->instruction_count));
    IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_note_model_quality(
        schedule_class->model_quality, out_cost));
    entry_critical_path_cycles[i] = schedule_class->latency_cycles;
    out_cost->critical_path_cycles =
        iree_max(out_cost->critical_path_cycles, entry_critical_path_cycles[i]);

    for (uint16_t j = 0; j < schedule_class->issue_use_count; ++j) {
      const loom_low_issue_use_t* issue_use =
          &descriptor_set->issue_uses[schedule_class->issue_use_start + j];
      if (issue_use->resource_id >= descriptor_set->resource_count) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "descriptor recipe issue use references "
                                "invalid resource");
      }
      IREE_RETURN_IF_ERROR(loom_low_descriptor_resource_cost_accumulate(
          issue_use, entry->occurrence_count,
          &resource_costs[issue_use->resource_id]));
    }
    for (uint16_t j = 0; j < schedule_class->pressure_delta_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_note_pressure_delta(
          &descriptor_set
               ->pressure_deltas[schedule_class->pressure_delta_start + j],
          entry->occurrence_count, descriptor_set->reg_class_count,
          pressure_costs));
    }
    for (uint16_t j = 0; j < descriptor->effect_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_note_memory_effect(
          &descriptor_set->effects[descriptor->effect_start + j],
          entry->occurrence_count, &out_cost->memory));
    }
  }

  uint32_t previous_dependency_key = 0;
  bool has_previous_dependency = false;
  for (uint16_t i = 0; i < recipe->dependency_count; ++i) {
    const loom_low_descriptor_recipe_dependency_t dependency =
        recipe->dependencies[i];
    const uint32_t dependency_key =
        ((uint32_t)dependency.target_entry << 16) | dependency.source_entry;
    if (dependency.source_entry >= dependency.target_entry ||
        dependency.target_entry >= recipe->entry_count ||
        (has_previous_dependency &&
         dependency_key <= previous_dependency_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "descriptor recipe dependencies are not unique "
                              "topological rows");
    }
    const loom_low_descriptor_recipe_entry_t* target_entry =
        &recipe->entries[dependency.target_entry];
    const loom_low_descriptor_view_t* target_view =
        &descriptor_set->descriptor_views[target_entry->descriptor_ordinal];
    const uint16_t target_latency_cycles =
        descriptor_set->schedule_classes[target_view->schedule_class_id]
            .latency_cycles;
    uint64_t dependent_path_cycles = 0;
    if (!iree_checked_add_u64(
            entry_critical_path_cycles[dependency.source_entry],
            target_latency_cycles, &dependent_path_cycles)) {
      return loom_low_descriptor_cost_overflow("critical path");
    }
    entry_critical_path_cycles[dependency.target_entry] =
        iree_max(entry_critical_path_cycles[dependency.target_entry],
                 dependent_path_cycles);
    out_cost->critical_path_cycles =
        iree_max(out_cost->critical_path_cycles, dependent_path_cycles);
    previous_dependency_key = dependency_key;
    has_previous_dependency = true;
  }

  for (uint16_t i = 0; i < recipe->durable_pressure_delta_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_note_pressure_delta(
        &recipe->durable_pressure_deltas[i], /*occurrence_count=*/1,
        descriptor_set->reg_class_count, pressure_costs));
  }

  iree_host_size_t resource_write_index = 0;
  for (uint16_t i = 0; i < descriptor_set->resource_count; ++i) {
    if (resource_costs[i].use_count == 0) continue;
    IREE_RETURN_IF_ERROR(loom_low_descriptor_cost_add_u64(
        resource_costs[i].estimated_min_cycles, "total resource cycles",
        &out_cost->total_resource_cycles));
    out_cost->maximum_resource_cycles =
        iree_max(out_cost->maximum_resource_cycles,
                 resource_costs[i].estimated_min_cycles);
    resource_costs[resource_write_index++] = resource_costs[i];
  }
  out_cost->resource_costs = resource_costs;
  out_cost->resource_cost_count = resource_write_index;

  iree_host_size_t pressure_write_index = 0;
  for (uint16_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    if (pressure_costs[i].total_unit_delta == 0) continue;
    pressure_costs[pressure_write_index++] = pressure_costs[i];
  }
  out_cost->pressure_costs = pressure_costs;
  out_cost->pressure_cost_count = pressure_write_index;
  return iree_ok_status();
}

static bool loom_low_descriptor_cost_evidence_is_compatible(
    const loom_low_descriptor_cost_t* left,
    const loom_low_descriptor_cost_t* right) {
  return left->descriptor_set != NULL &&
         left->descriptor_set == right->descriptor_set &&
         left->model_quality != LOOM_LOW_MODEL_QUALITY_UNKNOWN &&
         left->model_quality == right->model_quality &&
         left->memory.read_unknown_width_count == 0 &&
         left->memory.write_unknown_width_count == 0 &&
         right->memory.read_unknown_width_count == 0 &&
         right->memory.write_unknown_width_count == 0;
}

static bool loom_low_descriptor_cost_no_worse_u64(uint64_t left, uint64_t right,
                                                  bool* inout_strict) {
  if (left > right) return false;
  *inout_strict |= left < right;
  return true;
}

static bool loom_low_descriptor_cost_resources_dominate(
    const loom_low_descriptor_cost_t* left,
    const loom_low_descriptor_cost_t* right, bool* inout_strict) {
  iree_host_size_t left_index = 0;
  iree_host_size_t right_index = 0;
  while (left_index < left->resource_cost_count ||
         right_index < right->resource_cost_count) {
    const uint16_t left_id = left_index < left->resource_cost_count
                                 ? left->resource_costs[left_index].resource_id
                                 : UINT16_MAX;
    const uint16_t right_id =
        right_index < right->resource_cost_count
            ? right->resource_costs[right_index].resource_id
            : UINT16_MAX;
    const loom_low_descriptor_resource_cost_t* left_cost =
        left_id <= right_id ? &left->resource_costs[left_index] : NULL;
    const loom_low_descriptor_resource_cost_t* right_cost =
        right_id <= left_id ? &right->resource_costs[right_index] : NULL;
    if (!loom_low_descriptor_cost_no_worse_u64(
            left_cost != NULL ? left_cost->total_unit_cycles : 0,
            right_cost != NULL ? right_cost->total_unit_cycles : 0,
            inout_strict) ||
        !loom_low_descriptor_cost_no_worse_u64(
            left_cost != NULL ? left_cost->estimated_min_cycles : 0,
            right_cost != NULL ? right_cost->estimated_min_cycles : 0,
            inout_strict)) {
      return false;
    }
    left_index += left_cost != NULL;
    right_index += right_cost != NULL;
  }
  return true;
}

static bool loom_low_descriptor_cost_pressure_dominates(
    const loom_low_descriptor_cost_t* left,
    const loom_low_descriptor_cost_t* right, bool* inout_strict) {
  iree_host_size_t left_index = 0;
  iree_host_size_t right_index = 0;
  while (left_index < left->pressure_cost_count ||
         right_index < right->pressure_cost_count) {
    const uint16_t left_id = left_index < left->pressure_cost_count
                                 ? left->pressure_costs[left_index].reg_class_id
                                 : UINT16_MAX;
    const uint16_t right_id =
        right_index < right->pressure_cost_count
            ? right->pressure_costs[right_index].reg_class_id
            : UINT16_MAX;
    const loom_low_descriptor_pressure_cost_t* left_cost =
        left_id <= right_id ? &left->pressure_costs[left_index] : NULL;
    const loom_low_descriptor_pressure_cost_t* right_cost =
        right_id <= left_id ? &right->pressure_costs[right_index] : NULL;
    const int64_t left_delta =
        left_cost != NULL ? left_cost->total_unit_delta : 0;
    const int64_t right_delta =
        right_cost != NULL ? right_cost->total_unit_delta : 0;
    if (left_delta > right_delta) return false;
    *inout_strict |= left_delta < right_delta;
    left_index += left_cost != NULL;
    right_index += right_cost != NULL;
  }
  return true;
}

bool loom_low_descriptor_cost_dominates(
    const loom_low_descriptor_cost_t* left,
    const loom_low_descriptor_cost_t* right) {
  IREE_ASSERT_ARGUMENT(left);
  IREE_ASSERT_ARGUMENT(right);
  if (!loom_low_descriptor_cost_evidence_is_compatible(left, right)) {
    return false;
  }
  bool strict = false;
  return loom_low_descriptor_cost_resources_dominate(left, right, &strict) &&
         loom_low_descriptor_cost_pressure_dominates(left, right, &strict) &&
         loom_low_descriptor_cost_no_worse_u64(left->maximum_resource_cycles,
                                               right->maximum_resource_cycles,
                                               &strict) &&
         loom_low_descriptor_cost_no_worse_u64(left->total_resource_cycles,
                                               right->total_resource_cycles,
                                               &strict) &&
         loom_low_descriptor_cost_no_worse_u64(left->critical_path_cycles,
                                               right->critical_path_cycles,
                                               &strict) &&
         loom_low_descriptor_cost_no_worse_u64(
             left->instruction_count, right->instruction_count, &strict) &&
         loom_low_descriptor_cost_no_worse_u64(
             left->memory.read_operation_count,
             right->memory.read_operation_count, &strict) &&
         loom_low_descriptor_cost_no_worse_u64(
             left->memory.write_operation_count,
             right->memory.write_operation_count, &strict) &&
         loom_low_descriptor_cost_no_worse_u64(left->memory.read_byte_count,
                                               right->memory.read_byte_count,
                                               &strict) &&
         loom_low_descriptor_cost_no_worse_u64(left->memory.write_byte_count,
                                               right->memory.write_byte_count,
                                               &strict) &&
         strict;
}

static loom_low_descriptor_cost_order_t
loom_low_descriptor_cost_compare_lower_u64(uint64_t left, uint64_t right) {
  if (left < right) return LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT;
  if (left > right) return LOOM_LOW_DESCRIPTOR_COST_ORDER_RIGHT;
  return LOOM_LOW_DESCRIPTOR_COST_ORDER_EQUIVALENT;
}

static loom_low_descriptor_cost_order_t
loom_low_descriptor_cost_compare_identity(
    const loom_low_descriptor_cost_candidate_t* left,
    const loom_low_descriptor_cost_candidate_t* right) {
  if (left->is_canonical != right->is_canonical) {
    return left->is_canonical ? LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT
                              : LOOM_LOW_DESCRIPTOR_COST_ORDER_RIGHT;
  }
  return loom_low_descriptor_cost_compare_lower_u64(left->stable_key,
                                                    right->stable_key);
}

loom_low_descriptor_cost_order_t loom_low_descriptor_cost_compare(
    const loom_low_descriptor_cost_candidate_t* left,
    const loom_low_descriptor_cost_candidate_t* right) {
  IREE_ASSERT_ARGUMENT(left);
  IREE_ASSERT_ARGUMENT(left->cost);
  IREE_ASSERT_ARGUMENT(right);
  IREE_ASSERT_ARGUMENT(right->cost);
  if (!loom_low_descriptor_cost_evidence_is_compatible(left->cost,
                                                       right->cost)) {
    return loom_low_descriptor_cost_compare_identity(left, right);
  }
  if (loom_low_descriptor_cost_dominates(left->cost, right->cost)) {
    return LOOM_LOW_DESCRIPTOR_COST_ORDER_LEFT;
  }
  if (loom_low_descriptor_cost_dominates(right->cost, left->cost)) {
    return LOOM_LOW_DESCRIPTOR_COST_ORDER_RIGHT;
  }
  loom_low_descriptor_cost_order_t order =
      loom_low_descriptor_cost_compare_lower_u64(
          left->cost->maximum_resource_cycles,
          right->cost->maximum_resource_cycles);
  if (order != LOOM_LOW_DESCRIPTOR_COST_ORDER_EQUIVALENT) return order;
  order = loom_low_descriptor_cost_compare_lower_u64(
      left->cost->total_resource_cycles, right->cost->total_resource_cycles);
  if (order != LOOM_LOW_DESCRIPTOR_COST_ORDER_EQUIVALENT) return order;
  order = loom_low_descriptor_cost_compare_lower_u64(
      left->cost->critical_path_cycles, right->cost->critical_path_cycles);
  if (order != LOOM_LOW_DESCRIPTOR_COST_ORDER_EQUIVALENT) return order;
  order = loom_low_descriptor_cost_compare_lower_u64(
      left->cost->instruction_count, right->cost->instruction_count);
  if (order != LOOM_LOW_DESCRIPTOR_COST_ORDER_EQUIVALENT) return order;
  return loom_low_descriptor_cost_compare_identity(left, right);
}
