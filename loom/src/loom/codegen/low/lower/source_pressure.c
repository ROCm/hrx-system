// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_pressure.h"

#include <string.h>

#include "loom/analysis/liveness.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_low_source_pressure_query_state_record_t {
  // Target-owned stable state key.
  const void* key;
  // Byte length of |data|.
  iree_host_size_t data_length;
  // Zero-initialized target-owned state payload.
  void* data;
} loom_low_source_pressure_query_state_record_t;

typedef struct loom_low_source_pressure_query_state_t {
  // Arena owning records and target state payloads.
  iree_arena_allocator_t* arena;
  // Target-owned query state records.
  loom_low_source_pressure_query_state_record_t* records;
  // Number of populated entries in |records|.
  iree_host_size_t record_count;
  // Allocated capacity of |records|.
  iree_host_size_t record_capacity;
} loom_low_source_pressure_query_state_t;

static iree_status_t loom_low_source_pressure_get_or_allocate_query_state(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data) {
  loom_low_source_pressure_query_state_t* state =
      (loom_low_source_pressure_query_state_t*)user_data;
  *out_data = NULL;
  if (key == NULL || data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target query state requires a key and storage");
  }
  for (iree_host_size_t i = 0; i < state->record_count; ++i) {
    loom_low_source_pressure_query_state_record_t* record = &state->records[i];
    if (record->key != key) continue;
    if (record->data_length != data_length) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target query state key requested with inconsistent storage size");
    }
    *out_data = record->data;
    return iree_ok_status();
  }

  if (state->record_count == state->record_capacity) {
    iree_host_size_t minimum_capacity = 0;
    if (!iree_host_size_checked_add(state->record_count, 1,
                                    &minimum_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "target query state capacity overflow");
    }
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->record_count, minimum_capacity,
        sizeof(*state->records), &state->record_capacity,
        (void**)&state->records));
  }

  void* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(state->arena, data_length, &data));
  memset(data, 0, data_length);
  loom_low_source_pressure_query_state_record_t* record =
      &state->records[state->record_count++];
  *record = (loom_low_source_pressure_query_state_record_t){
      .key = key,
      .data_length = data_length,
      .data = data,
  };
  *out_data = data;
  return iree_ok_status();
}

typedef struct loom_low_source_pressure_event_t {
  // Structured liveness program point where pressure changes.
  uint32_t point;
  // Residency direct resource receiving the unit delta.
  uint16_t direct_resource_id;
  // Target allocation units entering or leaving the live set.
  uint32_t units;
  // Negative for a segment end and positive for a segment start.
  int8_t direction;
  // Source value used for deterministic event ordering.
  loom_value_id_t value_id;
} loom_low_source_pressure_event_t;

static bool loom_low_source_pressure_event_less(
    const loom_low_source_pressure_event_t* lhs,
    const loom_low_source_pressure_event_t* rhs) {
  if (lhs->point != rhs->point) return lhs->point < rhs->point;
  if (lhs->direction != rhs->direction) {
    return lhs->direction < rhs->direction;
  }
  if (lhs->direct_resource_id != rhs->direct_resource_id) {
    return lhs->direct_resource_id < rhs->direct_resource_id;
  }
  return lhs->value_id < rhs->value_id;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_source_pressure_event_sort,
                          loom_low_source_pressure_event_t,
                          loom_low_source_pressure_event_less)

static iree_status_t loom_low_source_pressure_copy_reserves(
    loom_low_lower_pressure_reserve_list_t policy_reserves,
    const loom_low_source_pressure_options_t* options,
    iree_host_size_t direct_resource_count, iree_arena_allocator_t* arena,
    loom_low_lower_pressure_reserve_t** out_reserves,
    iree_host_size_t* out_reserve_count, uint64_t** out_reserved_units) {
  *out_reserves = NULL;
  *out_reserve_count = 0;
  *out_reserved_units = NULL;
  if (direct_resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, direct_resource_count,
                                                   sizeof(**out_reserved_units),
                                                   (void**)out_reserved_units));
    memset(*out_reserved_units, 0,
           direct_resource_count * sizeof(**out_reserved_units));
  }
  if (policy_reserves.count > IREE_HOST_SIZE_MAX - options->reserve_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source pressure reserve count exceeds host size");
  }
  const iree_host_size_t reserve_count =
      policy_reserves.count + options->reserve_count;
  if (reserve_count == 0) return iree_ok_status();
  if (policy_reserves.count != 0 && policy_reserves.values == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "lowering policy has a null non-empty pressure reserve list");
  }
  if (options->reserves == NULL) {
    if (options->reserve_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source pressure has a null non-empty reserve contributor list");
    }
  }

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, reserve_count, sizeof(**out_reserves), (void**)out_reserves));
  for (iree_host_size_t i = 0; i < reserve_count; ++i) {
    const loom_low_lower_pressure_reserve_t* source =
        i < policy_reserves.count
            ? &policy_reserves.values[i]
            : &options->reserves[i - policy_reserves.count];
    if (iree_string_view_is_empty(source->name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source pressure reserve %zu has no name", i);
    }
    if (source->direct_resource_count != direct_resource_count ||
        (direct_resource_count != 0 && source->direct_resource_units == NULL)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source pressure reserve '%.*s' has %zu resource values; model "
          "requires %zu",
          (int)source->name.size, source->name.data,
          source->direct_resource_count, direct_resource_count);
    }
    uint64_t* copied_units = NULL;
    if (direct_resource_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, direct_resource_count, sizeof(*copied_units),
          (void**)&copied_units));
      memcpy(copied_units, source->direct_resource_units,
             direct_resource_count * sizeof(*copied_units));
    }
    (*out_reserves)[i] = (loom_low_lower_pressure_reserve_t){
        .name = source->name,
        .direct_resource_units = copied_units,
        .direct_resource_count = direct_resource_count,
    };
    for (iree_host_size_t resource_id = 0; resource_id < direct_resource_count;
         ++resource_id) {
      uint64_t* total = &(*out_reserved_units)[resource_id];
      if (*total > UINT64_MAX - copied_units[resource_id]) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "source pressure reserves overflow direct resource %zu",
            resource_id);
      }
      *total += copied_units[resource_id];
    }
  }
  *out_reserve_count = reserve_count;
  return iree_ok_status();
}

static iree_status_t loom_low_source_pressure_collect_events(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_policy_t* policy,
    const loom_target_residency_model_t* model,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_source_pressure_event_t** out_events,
    iree_host_size_t* out_event_count, iree_host_size_t* out_mapped_value_count,
    iree_host_size_t* out_non_register_value_count,
    iree_host_size_t* out_live_segment_count,
    iree_host_size_t* out_transient_segment_count) {
  *out_events = NULL;
  *out_event_count = 0;
  *out_mapped_value_count = 0;
  *out_non_register_value_count = 0;
  *out_live_segment_count = 0;
  *out_transient_segment_count = 0;
  iree_host_size_t segment_capacity = 0;
  if (!iree_host_size_checked_add(liveness->segment_count,
                                  liveness->value_count, &segment_capacity) ||
      segment_capacity > IREE_HOST_SIZE_MAX / 2u) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source pressure event count exceeds host size");
  }
  const iree_host_size_t event_capacity = segment_capacity * 2u;
  loom_low_source_pressure_event_t* events = NULL;
  if (event_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, event_capacity, sizeof(*events), (void**)&events));
  }

  iree_host_size_t event_count = 0;
  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < liveness->value_count; ++value_ordinal) {
    const loom_liveness_segment_range_t segment_range =
        loom_liveness_segment_range_for_value_ordinal(liveness, value_ordinal);
    const loom_value_id_t value_id = liveness->value_ids[value_ordinal];
    const loom_value_t* value =
        loom_module_value(environment->module, value_id);
    const bool is_block_argument = loom_value_is_block_arg(value);
    const loom_op_t* source_op =
        is_block_argument ? environment->function.op : loom_value_def_op(value);
    const bool has_leaf_result_issue =
        !is_block_argument && source_op != NULL && source_op->region_count == 0;
    if (segment_range.count == 0 && !has_leaf_result_issue) continue;
    loom_low_lower_rule_mapped_value_t mapped_value =
        loom_low_lower_rule_mapped_value_none();
    IREE_RETURN_IF_ERROR(policy->map_contract_value.fn(
        policy->map_contract_value.user_data, environment, source_op, value_id,
        &mapped_value));
    if (!mapped_value.is_register) {
      ++*out_non_register_value_count;
      continue;
    }
    if (mapped_value.register_unit_count == 0 ||
        mapped_value.descriptor_register_class_id >=
            model->direct_resources.resource_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "source value %u maps to invalid residency resource %u with %u "
          "unit(s)",
          value_id, (unsigned)mapped_value.descriptor_register_class_id,
          (unsigned)mapped_value.register_unit_count);
    }
    ++*out_mapped_value_count;
    for (uint32_t i = 0; i < segment_range.count; ++i) {
      const loom_liveness_segment_t segment =
          liveness->segments[segment_range.start + i];
      if (segment.start_point >= segment.end_point) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "source value %u has an empty pressure liveness segment", value_id);
      }
      events[event_count++] = (loom_low_source_pressure_event_t){
          .point = segment.end_point,
          .direct_resource_id = mapped_value.descriptor_register_class_id,
          .units = mapped_value.register_unit_count,
          .direction = -1,
          .value_id = value_id,
      };
      events[event_count++] = (loom_low_source_pressure_event_t){
          .point = segment.start_point,
          .direct_resource_id = mapped_value.descriptor_register_class_id,
          .units = mapped_value.register_unit_count,
          .direction = 1,
          .value_id = value_id,
      };
      ++*out_live_segment_count;
    }
    if (has_leaf_result_issue) {
      uint32_t issue_point = 0;
      IREE_RETURN_IF_ERROR(loom_liveness_op_program_point(
          liveness, loom_liveness_order_empty(), source_op, &issue_point));
      bool live_at_issue = false;
      for (uint32_t i = 0; i < segment_range.count; ++i) {
        const loom_liveness_segment_t segment =
            liveness->segments[segment_range.start + i];
        if (segment.start_point <= issue_point &&
            issue_point < segment.end_point) {
          live_at_issue = true;
          break;
        }
      }
      if (!live_at_issue) {
        if (issue_point == UINT32_MAX) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "source pressure issue segment exceeds program-point range");
        }
        events[event_count++] = (loom_low_source_pressure_event_t){
            .point = issue_point + 1u,
            .direct_resource_id = mapped_value.descriptor_register_class_id,
            .units = mapped_value.register_unit_count,
            .direction = -1,
            .value_id = value_id,
        };
        events[event_count++] = (loom_low_source_pressure_event_t){
            .point = issue_point,
            .direct_resource_id = mapped_value.descriptor_register_class_id,
            .units = mapped_value.register_unit_count,
            .direction = 1,
            .value_id = value_id,
        };
        ++*out_live_segment_count;
        ++*out_transient_segment_count;
      }
    }
  }
  IREE_ASSERT_LE(event_count, event_capacity);
  loom_low_source_pressure_event_sort(events, event_count);
  *out_events = events;
  *out_event_count = event_count;
  return iree_ok_status();
}

static iree_status_t loom_low_source_pressure_apply_event(
    const loom_low_source_pressure_event_t* event, uint64_t* current_units) {
  uint64_t* units = &current_units[event->direct_resource_id];
  if (event->direction < 0) {
    if (*units < event->units) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source pressure liveness sweep underflow");
    }
    *units -= event->units;
    return iree_ok_status();
  }
  if (*units > UINT64_MAX - event->units) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source pressure liveness sweep overflow");
  }
  *units += event->units;
  return iree_ok_status();
}

static iree_status_t loom_low_source_pressure_sweep(
    const loom_target_residency_model_t* model,
    const loom_target_residency_evaluator_t* evaluator,
    const loom_low_source_pressure_event_t* events,
    iree_host_size_t event_count, const uint64_t* reserved_units,
    loom_liveness_region_point_range_t point_range,
    iree_arena_allocator_t* arena, uint32_t* out_minimum_tier,
    uint32_t* out_minimum_tier_point, uint64_t** out_peak_units,
    uint64_t** out_minimum_tier_units) {
  const iree_host_size_t direct_resource_count =
      model->direct_resources.resource_count;
  uint64_t* current_units = NULL;
  uint64_t* peak_units = NULL;
  uint64_t* minimum_tier_units = NULL;
  if (direct_resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, direct_resource_count,
                                                   sizeof(*current_units),
                                                   (void**)&current_units));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, direct_resource_count,
                                                   sizeof(*peak_units),
                                                   (void**)&peak_units));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, direct_resource_count, sizeof(*minimum_tier_units),
        (void**)&minimum_tier_units));
    memcpy(current_units, reserved_units,
           direct_resource_count * sizeof(*current_units));
    memset(peak_units, 0, direct_resource_count * sizeof(*peak_units));
    memset(minimum_tier_units, 0,
           direct_resource_count * sizeof(*minimum_tier_units));
  }

  iree_host_size_t event_index = 0;
  while (event_index < event_count &&
         events[event_index].point < point_range.start_point) {
    IREE_RETURN_IF_ERROR(loom_low_source_pressure_apply_event(
        &events[event_index++], current_units));
  }
  while (event_index < event_count &&
         events[event_index].point == point_range.start_point &&
         point_range.start_point < point_range.end_point) {
    IREE_RETURN_IF_ERROR(loom_low_source_pressure_apply_event(
        &events[event_index++], current_units));
  }

  uint32_t minimum_tier = 0;
  IREE_RETURN_IF_ERROR(loom_target_residency_evaluator_evaluate_tier(
      evaluator, current_units, direct_resource_count, &minimum_tier));
  uint32_t minimum_tier_point = point_range.start_point;
  if (direct_resource_count != 0) {
    memcpy(peak_units, current_units,
           direct_resource_count * sizeof(*peak_units));
    memcpy(minimum_tier_units, current_units,
           direct_resource_count * sizeof(*minimum_tier_units));
  }
  while (event_index < event_count) {
    const uint32_t point = events[event_index].point;
    if (point >= point_range.end_point) break;
    do {
      IREE_RETURN_IF_ERROR(loom_low_source_pressure_apply_event(
          &events[event_index], current_units));
      ++event_index;
    } while (event_index < event_count && events[event_index].point == point);
    for (iree_host_size_t resource_id = 0; resource_id < direct_resource_count;
         ++resource_id) {
      peak_units[resource_id] =
          iree_max(peak_units[resource_id], current_units[resource_id]);
    }
    uint32_t tier = 0;
    IREE_RETURN_IF_ERROR(loom_target_residency_evaluator_evaluate_tier(
        evaluator, current_units, direct_resource_count, &tier));
    if (tier < minimum_tier) {
      minimum_tier = tier;
      minimum_tier_point = point;
      if (direct_resource_count != 0) {
        memcpy(minimum_tier_units, current_units,
               direct_resource_count * sizeof(*minimum_tier_units));
      }
    }
  }

  *out_minimum_tier = minimum_tier;
  *out_minimum_tier_point = minimum_tier_point;
  *out_peak_units = peak_units;
  *out_minimum_tier_units = minimum_tier_units;
  return iree_ok_status();
}

iree_status_t loom_low_source_pressure_analyze_regions(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_policy_t* policy,
    const loom_low_source_pressure_options_t* options,
    const loom_region_t* const* regions, iree_host_size_t region_count,
    iree_arena_allocator_t* arena, loom_low_source_pressure_t* out_pressures) {
  if (environment == NULL || policy == NULL || arena == NULL ||
      (region_count != 0 && (regions == NULL || out_pressures == NULL))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source pressure requires an environment, policy, region list, arena, "
        "and output list");
  }
  if (region_count == 0) return iree_ok_status();
  memset(out_pressures, 0, region_count * sizeof(*out_pressures));
  if (environment->module == NULL || environment->function.op == NULL ||
      environment->descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source pressure environment requires a function and descriptor set");
  }
  const loom_low_source_pressure_options_t empty_options =
      loom_low_source_pressure_options_empty();
  if (options == NULL) options = &empty_options;

  const loom_target_residency_model_t* model = NULL;
  if (policy->residency_model.fn != NULL) {
    model = policy->residency_model.fn(policy->residency_model.user_data,
                                       environment);
  }
  if (model == NULL) {
    if (options->reserve_count != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source pressure reserves require an available residency model");
    }
    return iree_ok_status();
  }
  if (policy->map_contract_value.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "lowering policy '%.*s' has a residency model but no source value "
        "mapping",
        (int)policy->name.size, policy->name.data);
  }
  if (model->direct_resources.resource_count !=
      environment->descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "lowering policy '%.*s' residency model has %u direct resources; "
        "descriptor set has %u register classes",
        (int)policy->name.size, policy->name.data,
        (unsigned)model->direct_resources.resource_count,
        (unsigned)environment->descriptor_set->reg_class_count);
  }

  loom_target_residency_evaluator_t residency_evaluator;
  IREE_RETURN_IF_ERROR(
      loom_target_residency_evaluator_initialize(model, &residency_evaluator));

  loom_local_value_domain_t owned_value_domain = {0};
  const loom_local_value_domain_t* value_domain = environment->value_domain;
  bool owns_value_domain = false;
  iree_status_t status = iree_ok_status();
  const loom_region_t* body = loom_func_like_body(environment->function);
  if (value_domain == NULL) {
    status = loom_local_value_domain_acquire_for_region_tree(
        (loom_module_t*)environment->module, body, arena, &owned_value_domain);
    if (iree_status_is_ok(status)) {
      value_domain = &owned_value_domain;
      owns_value_domain = true;
    }
  } else if (!loom_local_value_domain_is_acquired(value_domain) ||
             value_domain->region != body ||
             !iree_any_bit_set(value_domain->flags,
                               LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source pressure requires a region-tree value domain for the function");
  }

  loom_target_contract_query_environment_t query_environment = *environment;
  loom_view_region_table_t owned_view_regions = {0};
  loom_low_source_pressure_query_state_t query_state = {
      .arena = arena,
  };
  if (iree_status_is_ok(status)) {
    query_environment.value_domain = value_domain;
    query_environment.arena = arena;
    if (query_environment.view_regions == NULL) {
      status = loom_view_region_table_initialize(
          (loom_value_fact_table_t*)query_environment.fact_table, value_domain,
          arena, &owned_view_regions);
      if (iree_status_is_ok(status)) {
        status = loom_view_region_table_analyze(&owned_view_regions);
      }
      if (iree_status_is_ok(status)) {
        query_environment.view_regions = &owned_view_regions;
      }
    }
    if (query_environment.target_state_allocator.fn == NULL) {
      query_environment.target_state_allocator =
          (loom_target_contract_query_state_allocator_t){
              .fn = loom_low_source_pressure_get_or_allocate_query_state,
              .user_data = &query_state,
          };
    }
  }

  loom_low_lower_pressure_reserve_list_t policy_reserves =
      loom_low_lower_pressure_reserve_list_empty();
  if (iree_status_is_ok(status) && policy->pressure_reserves.fn != NULL) {
    status = policy->pressure_reserves.fn(policy->pressure_reserves.user_data,
                                          &query_environment, &policy_reserves);
  }
  loom_low_lower_pressure_reserve_t* reserves = NULL;
  iree_host_size_t reserve_count = 0;
  uint64_t* reserved_units = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_source_pressure_copy_reserves(
        policy_reserves, options, model->direct_resources.resource_count, arena,
        &reserves, &reserve_count, &reserved_units);
  }

  loom_liveness_analysis_t liveness = {0};
  if (iree_status_is_ok(status)) {
    status = loom_liveness_analyze_local_value_domain(
        value_domain, loom_liveness_order_empty(), arena, &liveness);
  }
  loom_low_source_pressure_event_t* events = NULL;
  iree_host_size_t event_count = 0;
  iree_host_size_t mapped_value_count = 0;
  iree_host_size_t non_register_value_count = 0;
  iree_host_size_t live_segment_count = 0;
  iree_host_size_t transient_segment_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_low_source_pressure_collect_events(
        &query_environment, policy, model, &liveness, arena, &events,
        &event_count, &mapped_value_count, &non_register_value_count,
        &live_segment_count, &transient_segment_count);
  }

  loom_low_source_pressure_flags_t flags =
      LOOM_LOW_SOURCE_PRESSURE_FLAG_MODEL_AVAILABLE;
  if (iree_any_bit_set(policy_reserves.flags,
                       LOOM_LOW_LOWER_PRESSURE_RESERVE_FLAG_COMPLETE) ||
      iree_any_bit_set(
          options->flags,
          LOOM_LOW_SOURCE_PRESSURE_OPTION_FLAG_RESERVES_COMPLETE)) {
    flags |= LOOM_LOW_SOURCE_PRESSURE_FLAG_PROJECTION_COMPLETE;
  }
  for (iree_host_size_t i = 0; i < region_count && iree_status_is_ok(status);
       ++i) {
    const loom_region_t* pressure_region =
        regions[i] != NULL ? regions[i] : body;
    loom_liveness_region_point_range_t point_range = {0};
    status = loom_liveness_region_point_range(&liveness, pressure_region,
                                              &point_range);
    uint32_t minimum_tier = 0;
    uint32_t minimum_tier_point = 0;
    uint64_t* peak_units = NULL;
    uint64_t* minimum_tier_units = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_low_source_pressure_sweep(
          model, &residency_evaluator, events, event_count, reserved_units,
          point_range, arena, &minimum_tier, &minimum_tier_point, &peak_units,
          &minimum_tier_units);
    }
    if (!iree_status_is_ok(status)) break;
    out_pressures[i] = (loom_low_source_pressure_t){
        .flags = flags,
        .residency_model = model,
        .minimum_tier = minimum_tier,
        .minimum_tier_point = minimum_tier_point,
        .peak_direct_resource_units = peak_units,
        .minimum_tier_direct_resource_units = minimum_tier_units,
        .reserved_direct_resource_units = reserved_units,
        .direct_resource_count = model->direct_resources.resource_count,
        .reserves = reserves,
        .reserve_count = reserve_count,
        .mapped_value_count = mapped_value_count,
        .non_register_value_count = non_register_value_count,
        .live_segment_count = live_segment_count,
        .transient_segment_count = transient_segment_count,
    };
  }
  if (owns_value_domain) {
    loom_local_value_domain_release(&owned_value_domain);
  }
  return status;
}

iree_status_t loom_low_source_pressure_analyze(
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_policy_t* policy,
    const loom_low_source_pressure_options_t* options,
    iree_arena_allocator_t* arena, loom_low_source_pressure_t* out_pressure) {
  const loom_low_source_pressure_options_t empty_options =
      loom_low_source_pressure_options_empty();
  if (options == NULL) options = &empty_options;
  const loom_region_t* regions[] = {options->region};
  return loom_low_source_pressure_analyze_regions(
      environment, policy, options, regions, IREE_ARRAYSIZE(regions), arena,
      out_pressure);
}
