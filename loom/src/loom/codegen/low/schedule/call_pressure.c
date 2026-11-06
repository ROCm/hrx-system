// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/call_pressure.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/call.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/schedule/target_pressure.h"
#include "loom/ops/low/ops.h"

typedef struct loom_low_schedule_call_pressure_record_t {
  // Sparse capacities reduced by this call, owned by the scheduling arena.
  const loom_low_schedule_call_pressure_limit_t* limits;
  // Number of boundary limits for this call.
  uint32_t limit_count;
} loom_low_schedule_call_pressure_record_t;

struct loom_low_schedule_call_pressure_table_t {
  // Records aligned with loom_low_schedule_build_state_t::call_node_indices.
  loom_low_schedule_call_pressure_record_t* records;
};

static const loom_low_call_contract_t* loom_low_schedule_call_pressure_contract(
    const loom_low_schedule_build_state_t* state, iree_host_size_t call_index) {
  const loom_low_call_contract_provider_t* provider =
      state->options->call_contracts;
  const loom_op_t* call = state->nodes[state->call_node_indices[call_index]].op;
  const loom_low_call_contract_t* contract =
      provider->query(provider->user_data, loom_low_func_call_callee(call));
  IREE_ASSERT_NE(contract, NULL);
  return contract;
}

static bool loom_low_schedule_call_pressure_class_limit(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id,
    uint16_t* out_pressure_set_id, uint32_t* out_limit) {
  const loom_low_reg_class_t* reg_class =
      &state->target.descriptor_set->reg_classes[reg_class_id];
  const uint16_t descriptor_pressure_set_id =
      loom_low_reg_class_pressure_set_id(reg_class);
  if (descriptor_pressure_set_id != 0) {
    const loom_low_schedule_pressure_set_limit_t* pressure_set_limit =
        &state->pressure_limits.pressure_sets[descriptor_pressure_set_id];
    if (pressure_set_limit->representative_reg_class_id != reg_class_id) {
      return false;
    }
    if (pressure_set_limit->has_multiple_active_reg_classes) {
      *out_pressure_set_id = descriptor_pressure_set_id;
      *out_limit = pressure_set_limit->live_unit_limit;
    } else {
      *out_pressure_set_id = 0;
      *out_limit = state->pressure_limits.by_reg_class[reg_class_id];
    }
  } else {
    *out_pressure_set_id = 0;
    *out_limit = state->pressure_limits.by_reg_class[reg_class_id];
  }
  return *out_limit != UINT32_MAX;
}

iree_host_size_t loom_low_schedule_call_pressure_build_limits(
    const loom_low_schedule_build_state_t* state,
    const loom_low_call_contract_t* contract,
    loom_low_call_clobber_list_t common_clobbers,
    loom_low_schedule_call_pressure_limit_t* limits) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  iree_host_size_t limit_count = 0;
  for (uint16_t reg_class_id = 0;
       reg_class_id < descriptor_set->reg_class_count; ++reg_class_id) {
    uint16_t pressure_set_id = 0;
    uint32_t ordinary_capacity = UINT32_MAX;
    if (!loom_low_schedule_call_pressure_class_limit(
            state, reg_class_id, &pressure_set_id, &ordinary_capacity)) {
      continue;
    }
    const uint32_t preserved_capacity =
        loom_low_allocation_target_constraints_reg_class_call_preserved_unit_capacity(
            state->allocation_constraints, reg_class_id, common_clobbers,
            contract->clobbers);
    if (preserved_capacity >= ordinary_capacity) {
      continue;
    }
    limits[limit_count] = (loom_low_schedule_call_pressure_limit_t){
        .source_id = reg_class_id,
        .pressure_set_id = pressure_set_id,
        .preserved_unit_capacity = preserved_capacity,
        .source_kind = LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS,
    };
    ++limit_count;
  }
  for (uint16_t resource_id = 0;
       state->pressure_limits.register_packing_resource_capacities != NULL &&
       resource_id < descriptor_set->register_packing_resource_count;
       ++resource_id) {
    const uint32_t ordinary_capacity =
        loom_low_schedule_register_packing_resource_capacity(state,
                                                             resource_id);
    if (ordinary_capacity == UINT32_MAX) {
      continue;
    }
    const loom_low_register_packing_resource_t* resource =
        &descriptor_set->register_packing_resources[resource_id];
    const loom_low_register_packing_resource_member_t* capacity_member =
        &descriptor_set->register_packing_resource_members
             [resource->capacity_member_index];
    const uint32_t preserved_member_capacity =
        loom_low_allocation_target_constraints_reg_class_call_preserved_unit_capacity(
            state->allocation_constraints, capacity_member->reg_class_id,
            common_clobbers, contract->clobbers);
    const uint32_t preserved_capacity =
        loom_low_schedule_register_packing_member_capacity(
            preserved_member_capacity, resource, capacity_member);
    if (preserved_capacity >= ordinary_capacity) {
      continue;
    }
    limits[limit_count] = (loom_low_schedule_call_pressure_limit_t){
        .source_id = resource_id,
        .preserved_unit_capacity = preserved_capacity,
        .source_kind =
            LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE,
        .requires_hard_capacity = iree_all_bits_set(
            resource->flags,
            LOOM_LOW_REGISTER_PACKING_RESOURCE_FLAG_UNSPILLABLE),
    };
    ++limit_count;
  }
  return limit_count;
}

iree_status_t loom_low_schedule_call_pressure_initialize(
    loom_low_schedule_build_state_t* state) {
  const loom_low_call_contract_provider_t* provider =
      state->options->call_contracts;
  if (state->call_node_count == 0 || provider == NULL ||
      provider->query == NULL) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < state->call_node_count; ++i) {
    const loom_op_t* call = state->nodes[state->call_node_indices[i]].op;
    const loom_symbol_ref_t callee = loom_low_func_call_callee(call);
    if (provider->validate != NULL) {
      IREE_RETURN_IF_ERROR(provider->validate(provider->user_data, callee));
    }
  }
  if (state->pressure_limits.by_reg_class == NULL ||
      state->allocation_constraints == NULL) {
    return iree_ok_status();
  }

  const loom_low_call_clobber_list_t common_clobbers =
      provider->query_common_clobbers != NULL
          ? provider->query_common_clobbers(provider->user_data,
                                            state->target.descriptor_set)
          : (loom_low_call_clobber_list_t){0};
  // One reusable row buffer bounds construction by target classes/resources,
  // without repeating clobber analysis to discover each call's sparse size.
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const iree_host_size_t limit_capacity =
      (iree_host_size_t)descriptor_set->reg_class_count +
      descriptor_set->register_packing_resource_count;
  loom_low_schedule_call_pressure_limit_t* limits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, limit_capacity, sizeof(*limits), (void**)&limits));
  loom_low_schedule_call_pressure_table_t* table = NULL;
  for (iree_host_size_t i = 0; i < state->call_node_count; ++i) {
    const iree_host_size_t limit_count =
        loom_low_schedule_call_pressure_build_limits(
            state, loom_low_schedule_call_pressure_contract(state, i),
            common_clobbers, limits);
    if (limit_count == 0) {
      continue;
    }
    if (table == NULL) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->scratch_arena, 1, sizeof(*table), (void**)&table));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->scratch_arena, state->call_node_count, sizeof(*table->records),
          (void**)&table->records));
      memset(table->records, 0,
             state->call_node_count * sizeof(*table->records));
    }
    loom_low_schedule_call_pressure_limit_t* retained_limits = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, limit_count, sizeof(*retained_limits),
        (void**)&retained_limits));
    memcpy(retained_limits, limits, limit_count * sizeof(*retained_limits));
    table->records[i] = (loom_low_schedule_call_pressure_record_t){
        .limits = retained_limits,
        .limit_count = (uint32_t)limit_count,
    };
  }
  state->call_pressure = table;
  return iree_ok_status();
}

static iree_host_size_t loom_low_schedule_call_pressure_find_call(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = state->call_node_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (state->call_node_indices[middle] < node_index) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  IREE_ASSERT_LT(begin, state->call_node_count);
  IREE_ASSERT_EQ(state->call_node_indices[begin], node_index);
  return begin;
}

static uint64_t loom_low_schedule_call_pressure_crossing_packing_units(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint16_t resource_id) {
  IREE_ASSERT_NE(pressure_state->current_register_packing_units, NULL);
  const loom_low_register_packing_resource_t* resource =
      &descriptor_set->register_packing_resources[resource_id];
  uint64_t crossing_units =
      pressure_state->current_register_packing_units[resource_id];
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    const loom_low_register_packing_resource_member_t* member =
        &descriptor_set->register_packing_resource_members[member_index];
    const uint16_t reg_class_id = member->reg_class_id;
    const uint64_t current_live_units =
        pressure_state->current_live_units_by_reg_class[reg_class_id];
    const uint64_t killed_live_units =
        pressure_state->candidate_killed_units_by_reg_class[reg_class_id];
    IREE_ASSERT_LE(killed_live_units, current_live_units);
    const uint64_t current_contribution =
        loom_low_schedule_register_packing_contribution(current_live_units,
                                                        member);
    const uint64_t crossing_contribution =
        loom_low_schedule_register_packing_contribution(
            current_live_units - killed_live_units, member);
    IREE_ASSERT_LE(current_contribution, crossing_units);
    crossing_units = iree_math_saturating_add_u64(
        crossing_units - current_contribution, crossing_contribution);
  }
  return crossing_units;
}

void loom_low_schedule_call_pressure_score_limits(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_call_pressure_limit_t* limits, uint32_t limit_count,
    loom_low_schedule_candidate_score_t* score) {
  for (uint32_t i = 0; i < limit_count; ++i) {
    const loom_low_schedule_call_pressure_limit_t* limit = &limits[i];
    uint64_t current_live_units = 0;
    uint64_t killed_live_units = 0;
    if (limit->source_kind ==
        LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_PACKING_RESOURCE) {
      current_live_units =
          loom_low_schedule_call_pressure_crossing_packing_units(
              descriptor_set, pressure_state, limit->source_id);
    } else if (limit->pressure_set_id != 0) {
      const loom_low_schedule_pressure_set_record_t* pressure_set =
          &pressure_state->pressure_sets.records[limit->pressure_set_id];
      current_live_units = pressure_set->current_live_units;
      killed_live_units = pressure_set->candidate_killed_units;
    } else {
      current_live_units =
          pressure_state->current_live_units_by_reg_class[limit->source_id];
      killed_live_units =
          pressure_state->candidate_killed_units_by_reg_class[limit->source_id];
    }
    IREE_ASSERT_LE(killed_live_units, current_live_units);
    const uint64_t crossing_live_units = current_live_units - killed_live_units;
    if (crossing_live_units <= limit->preserved_unit_capacity) {
      continue;
    }

    score->flags |=
        limit->requires_hard_capacity
            ? LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_EXCEEDS_UNSPILLABLE_CAPACITY
            : LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_OPENS_REGISTER_DEBT;
    const uint64_t debt = crossing_live_units - limit->preserved_unit_capacity;
    score->pressure_cliff_penalty = iree_math_saturating_add_u32(
        score->pressure_cliff_penalty,
        debt > UINT32_MAX ? UINT32_MAX : (uint32_t)debt);
    if (score->pressure_cliff_units == LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE) {
      score->pressure_cliff_source_kind = limit->source_kind;
      score->pressure_cliff_source_id = limit->source_id;
      score->pressure_cliff_units = limit->preserved_unit_capacity;
      score->units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE;
    }
  }
}

void loom_low_schedule_call_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, loom_low_schedule_candidate_score_t* score) {
  if (state->call_pressure == NULL ||
      candidate_node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      !loom_low_func_call_isa(state->nodes[candidate_node_index].op)) {
    return;
  }
  const iree_host_size_t call_index =
      loom_low_schedule_call_pressure_find_call(state, candidate_node_index);
  const loom_low_schedule_call_pressure_record_t* record =
      &state->call_pressure->records[call_index];
  loom_low_schedule_call_pressure_score_limits(state->target.descriptor_set,
                                               pressure_state, record->limits,
                                               record->limit_count, score);
}
