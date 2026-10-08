// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-order and ready-frontier priority analysis for low scheduling.

#include "iree/base/internal/math.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/schedule/completion_wait.h"
#include "loom/codegen/low/schedule/pressure.h"
#include "loom/codegen/low/schedule/target_pressure.h"

static uint32_t loom_low_schedule_saturate_u64_to_u32(uint64_t value) {
  return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

// Returns a static nomination tier for materializations that may open live
// storage before it becomes actionable. Descriptor-bound slice/concat aliases
// retain ordinary critical-path priority so they can expose consumers outside
// the source-order window. Rematerializable leaves follow ordinary work, then
// other storage setup. Exact candidate scoring still defers non-actionable
// storage growth and closes an opened rematerialization chain before selecting
// another leaf.
static uint64_t loom_low_schedule_node_materialization_key(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (iree_any_bit_set(node->flags,
                       LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP) &&
      !iree_all_bits_set(node->flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_TRANSPARENT |
                             LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP)) {
    return UINT64_C(1) << 63;
  }
  if (node->descriptor == NULL || node->operand_count != 0 ||
      node->result_count == 0 ||
      iree_any_bit_set(node->traits, LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return 0;
  }
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    if (!loom_low_descriptor_result_can_rematerialize(
            state->target.descriptor_set, node->descriptor, result_index)) {
      return 0;
    }
  }
  return UINT64_C(1) << 62;
}

uint64_t loom_low_schedule_pressure_ready_key(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t killed_units = 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t i = 0; i < node->operand_count; ++i) {
    loom_low_schedule_note_candidate_operand_use(pressure_state,
                                                 operand_ordinals[i]);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) &&
        value->remaining_use_count ==
            pressure_state->candidate_operand_use_counts[value_ordinal]) {
      killed_units += value->live_unit_count;
    }
  }
  loom_low_schedule_reset_candidate_operand_uses(state, pressure_state);

  uint64_t produced_units = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[i]];
    if (value->remaining_use_count != 0 &&
        !iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      produced_units += value->unit_count;
    }
  }
  const uint32_t growth = loom_low_schedule_saturate_u64_to_u32(
      produced_units > killed_units ? produced_units - killed_units : 0);
  const uint32_t relief = loom_low_schedule_saturate_u64_to_u32(
      killed_units > produced_units ? killed_units - produced_units : 0);
  return ((uint64_t)growth << 32) | (uint64_t)(UINT32_MAX - relief);
}

static uint64_t loom_low_schedule_ready_schedule_key(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  // The strategy keys occupy at most the low 48 bits. Keeping potential
  // materializations in the upper tiers lets bounded recovery find ordinary
  // work and actionable rematerialization even when the source window consists
  // entirely of deferred storage setup.
  const uint64_t materialization_key =
      loom_low_schedule_node_materialization_key(state, node);
  switch (state->options->strategy) {
    case LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL: {
      const uint32_t critical_path =
          state->node_critical_path_cycles != NULL
              ? state->node_critical_path_cycles[node_index]
              : 0;
      return materialization_key | (UINT32_MAX - critical_path);
    }
    case LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING: {
      const uint16_t dependency_latency =
          state->node_dependency_latency_cycles != NULL
              ? state->node_dependency_latency_cycles[node_index]
              : 0;
      const uint16_t latency = loom_low_schedule_class_schedule_distance_cycles(
          node->schedule_class);
      return materialization_key | ((uint64_t)dependency_latency << 32) |
             (uint64_t)(UINT16_MAX - latency);
    }
    default:
      return materialization_key | node->source_ordinal;
  }
}

static uint64_t loom_low_schedule_ready_storage_key(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    const uint32_t producer_node =
        state->value_producer_nodes[operand_ordinals[operand_index]];
    if (producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        iree_any_bit_set(state->nodes[producer_node].flags,
                         LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP)) {
      // Complete opened storage before considering more setup. This bounds
      // live ranges in scarce destination register classes without requiring
      // target-specific fixed-register policy.
      return 0;
    }
  }
  const uint16_t relation_count = node->storage_relation_count;
  return relation_count == 0 ? UINT64_MAX
                             : 1u + (uint64_t)(UINT16_MAX - relation_count);
}

loom_low_schedule_ready_keys_t loom_low_schedule_pressure_ready_keys(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  return (loom_low_schedule_ready_keys_t){
      .values =
          {
              [LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE] = node->source_ordinal,
              [LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE] =
                  loom_low_schedule_pressure_ready_key(state, pressure_state,
                                                       node_index),
              [LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE] =
                  loom_low_schedule_ready_schedule_key(state, node_index),
              [LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE] =
                  loom_low_schedule_ready_storage_key(state, node_index),
          },
  };
}

static uint64_t loom_low_schedule_compute_register_packing_result_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    const loom_low_register_packing_resource_t* resource) {
  uint64_t resource_units = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    const loom_low_register_packing_resource_member_t* member =
        &state->target.descriptor_set
             ->register_packing_resource_members[member_index];
    uint64_t register_units = 0;
    for (uint16_t result_index = 0; result_index < node->result_count;
         ++result_index) {
      const loom_low_schedule_value_record_t* value =
          &state->values[result_ordinals[result_index]];
      if (value->register_class_id == member->reg_class_id) {
        register_units =
            iree_math_saturating_add_u64(register_units, value->unit_count);
      }
    }
    resource_units = iree_math_saturating_add_u64(
        resource_units, loom_low_schedule_register_packing_contribution(
                            register_units, member));
  }
  return resource_units;
}

// Returns true when the operand value is first produced at or after
// |source_range_start|. Values produced before the independently reorderable
// source range, including block live-ins, are already represented by current
// pressure when a node in the range is scored.
static bool loom_low_schedule_operand_has_future_activation(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    const loom_value_ordinal_t* operand_ordinals, uint16_t operand_index,
    uint32_t source_range_start) {
  const loom_value_ordinal_t operand_ordinal = operand_ordinals[operand_index];
  for (uint16_t previous_index = 0; previous_index < operand_index;
       ++previous_index) {
    if (operand_ordinals[previous_index] == operand_ordinal) {
      return false;
    }
  }
  const uint32_t producer_node = state->value_producer_nodes[operand_ordinal];
  return producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
         producer_node >= source_range_start &&
         state->nodes[producer_node].block_index == node->block_index;
}

static uint64_t loom_low_schedule_compute_register_packing_operand_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    const loom_low_register_packing_resource_t* resource,
    uint32_t source_range_start, bool* out_reads_resource) {
  *out_reads_resource = false;
  uint64_t resource_units = 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  const uint16_t member_end = resource->member_start + resource->member_count;
  for (uint16_t member_index = resource->member_start;
       member_index < member_end; ++member_index) {
    const loom_low_register_packing_resource_member_t* member =
        &state->target.descriptor_set
             ->register_packing_resource_members[member_index];
    uint64_t register_units = 0;
    for (uint16_t operand_index = 0; operand_index < node->operand_count;
         ++operand_index) {
      const loom_value_ordinal_t operand_ordinal =
          operand_ordinals[operand_index];
      if (state->values[operand_ordinal].register_class_id !=
          member->reg_class_id) {
        continue;
      }
      // Completion identity includes every resource operand. Only operands
      // first produced after this source range begins add future activation.
      *out_reads_resource = true;
      if (loom_low_schedule_operand_has_future_activation(
              state, node, operand_ordinals, operand_index,
              source_range_start)) {
        register_units = iree_math_saturating_add_u64(
            register_units, state->values[operand_ordinal].unit_count);
      }
    }
    resource_units = iree_math_saturating_add_u64(
        resource_units, loom_low_schedule_register_packing_contribution(
                            register_units, member));
  }
  return resource_units;
}

static uint64_t loom_low_schedule_compute_unspillable_result_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, uint16_t completion_domain_id) {
  uint64_t result_units = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
    if (loom_low_schedule_unspillable_completion_domain_id(
            state, value->register_class_id) == completion_domain_id) {
      result_units =
          iree_math_saturating_add_u64(result_units, value->unit_count);
    }
  }
  return result_units;
}

// Returns units of |operand_index| handed back to an already-live argument of
// the consumer's block. A loop backedge replaces that entry storage; it does
// not activate another copy when the terminator becomes ready.
static uint64_t loom_low_schedule_unspillable_handoff_replacement_units(
    const loom_low_schedule_build_state_t* state, uint32_t consumer_node,
    uint16_t operand_index, uint16_t completion_domain_id) {
  const loom_low_schedule_node_t* consumer = &state->nodes[consumer_node];
  const loom_block_t* consumer_block =
      state->body->blocks[consumer->block_index];
  uint64_t replacement_units = 0;
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     consumer_node);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, consumer_node);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (relation->source_operand_index != operand_index ||
        (relation->cause != LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH &&
         relation->cause != LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD)) {
      continue;
    }
    const loom_low_schedule_value_record_t* destination =
        &state->values[relation->destination_ordinal];
    const loom_value_t* destination_value = loom_module_value(
        state->module,
        state->value_domain->value_ids[relation->destination_ordinal]);
    if (!loom_value_is_block_arg(destination_value) ||
        loom_value_def_block(destination_value) != consumer_block ||
        loom_low_schedule_unspillable_completion_domain_id(
            state, destination->register_class_id) != completion_domain_id) {
      continue;
    }
    replacement_units =
        iree_math_saturating_add_u64(replacement_units, relation->unit_count);
  }
  return replacement_units;
}

static uint64_t loom_low_schedule_compute_unspillable_operand_units(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t completion_domain_id, uint32_t source_range_start) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t operand_units = 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    const loom_value_ordinal_t operand_ordinal =
        operand_ordinals[operand_index];
    const loom_low_schedule_value_record_t* value =
        &state->values[operand_ordinal];
    if (loom_low_schedule_operand_has_future_activation(
            state, node, operand_ordinals, operand_index, source_range_start) &&
        loom_low_schedule_unspillable_completion_domain_id(
            state, value->register_class_id) == completion_domain_id) {
      const uint64_t replacement_units =
          loom_low_schedule_unspillable_handoff_replacement_units(
              state, node_index, operand_index, completion_domain_id);
      const uint64_t activation_units =
          value->unit_count > replacement_units
              ? value->unit_count - replacement_units
              : 0;
      operand_units =
          iree_math_saturating_add_u64(operand_units, activation_units);
    }
  }
  return operand_units;
}

static uint32_t loom_low_schedule_compute_downstream_activation_units(
    uint64_t producer_result_units, uint64_t consumer_operand_units,
    uint64_t consumer_result_units, uint32_t consumer_activation_units,
    bool is_early_clobber) {
  const uint64_t consumer_required_units = iree_math_saturating_add_u64(
      consumer_result_units, consumer_activation_units);
  uint64_t required_units =
      iree_max(consumer_operand_units, consumer_required_units);
  if (is_early_clobber) {
    required_units = iree_max(
        required_units, iree_math_saturating_add_u64(consumer_operand_units,
                                                     consumer_result_units));
  }
  const uint64_t activation_units = required_units > producer_result_units
                                        ? required_units - producer_result_units
                                        : 0;
  return loom_low_schedule_saturate_u64_to_u32(activation_units);
}

// Retains the least expensive concrete exit for a value. Maximum downstream
// activation remains node-owned; this pair identifies one transaction that can
// release the value with the least additional storage. Node order breaks an
// otherwise identical pressure tie.
static void loom_low_schedule_update_unspillable_completion_path(
    uint32_t activation_units, uint32_t sink,
    loom_low_schedule_unspillable_completion_path_t* inout_path) {
  if (sink == LOOM_LOW_SCHEDULE_NODE_NONE ||
      activation_units > inout_path->activation_units ||
      (activation_units == inout_path->activation_units &&
       inout_path->sink != LOOM_LOW_SCHEDULE_NODE_NONE &&
       sink >= inout_path->sink)) {
    return;
  }
  inout_path->activation_units = activation_units;
  inout_path->sink = sink;
}

// Selects the least expensive retained completion among a node's result
// values. Reverse analysis has already populated every same-block result path
// before visiting its producers.
static loom_low_schedule_unspillable_completion_path_t
loom_low_schedule_select_unspillable_result_completion(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, uint16_t completion_domain_id) {
  loom_low_schedule_unspillable_completion_path_t selected = {
      .activation_units = UINT32_MAX,
      .sink = LOOM_LOW_SCHEDULE_NODE_NONE,
  };
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
    if (loom_low_schedule_unspillable_completion_domain_id(
            state, value->register_class_id) != completion_domain_id) {
      continue;
    }
    loom_low_schedule_update_unspillable_completion_path(
        value->unspillable_completion.activation_units,
        value->unspillable_completion.sink, &selected);
  }
  return selected;
}

// Returns the first node in the independently reorderable source range ending
// at |range_last_node|. Reverse priority analysis enters each range at its last
// node, so every node is visited at most twice without retaining another table.
static uint32_t loom_low_schedule_source_range_start(
    const loom_low_schedule_build_state_t* state, uint32_t range_last_node) {
  uint32_t range_start = range_last_node;
  if (iree_any_bit_set(state->nodes[range_start].flags,
                       LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY)) {
    return range_start;
  }
  const uint32_t block_start =
      state->blocks[state->nodes[range_start].block_index].node_start;
  while (range_start > block_start &&
         !iree_any_bit_set(state->nodes[range_start - 1].flags,
                           LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY)) {
    --range_start;
  }
  return range_start;
}

void loom_low_schedule_pressure_compute_node_priorities(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const loom_low_schedule_dependency_detail_index_t* dependency_details,
    loom_low_schedule_pressure_state_t* pressure_state) {
  if (state->node_critical_path_cycles == NULL &&
      state->node_dependency_latency_cycles == NULL &&
      state->node_opened_completion_latency_cycles == NULL &&
      state->node_pressure_demand_units == NULL &&
      state->node_pressure_activation_units == NULL &&
      state->node_unspillable_activation_units == NULL &&
      state->node_register_packing.activation_units == NULL &&
      pressure_state->first_actionable_pressure_cliff_indices == NULL) {
    return;
  }
  uint32_t source_range_start = (uint32_t)node_count;
  for (iree_host_size_t i = node_count; i > 0; --i) {
    const uint32_t node_index = (uint32_t)(i - 1);
    if (node_index < source_range_start) {
      source_range_start =
          loom_low_schedule_source_range_start(state, node_index);
    }
    loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_low_schedule_block_t* block_record =
        &state->blocks[node->block_index];
    if (node_index == block_record->node_start + block_record->node_count - 1) {
      loom_low_schedule_storage_lifetimes_set_forwarded_values(
          state, node->block_index, true);
      if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
          state->pressure_resources != NULL) {
        loom_low_schedule_initialize_source_pressure_block(
            state, pressure_state, node->block_index);
      }
    }
    const bool is_storage_setup = iree_any_bit_set(
        node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
    // Same-block SSA consumers follow their producers in source order. Their
    // result footprints are already retained when this reverse sweep reaches
    // the producer and propagates downstream demand through its dependencies.
    if (state->node_register_packing.result_units != NULL) {
      const uint16_t resource_count =
          state->target.descriptor_set->register_packing_resource_count;
      uint64_t* result_units = state->node_register_packing.result_units +
                               (iree_host_size_t)node_index * resource_count;
      for (uint16_t resource_id = 0; resource_id < resource_count;
           ++resource_id) {
        result_units[resource_id] =
            loom_low_schedule_compute_register_packing_result_units(
                state, node,
                &state->target.descriptor_set
                     ->register_packing_resources[resource_id]);
      }
    }
    uint16_t dependency_latency_cycles = 0;
    if (state->node_dependency_latency_cycles != NULL) {
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const uint32_t producer_node =
            state->value_producer_nodes[operand_ordinals[operand_index]];
        if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
            state->nodes[producer_node].block_index != node->block_index) {
          continue;
        }
        const loom_low_schedule_class_t* producer_schedule_class =
            state->nodes[producer_node].schedule_class;
        const uint16_t producer_latency =
            loom_low_schedule_class_schedule_distance_cycles(
                producer_schedule_class);
        dependency_latency_cycles =
            iree_max(dependency_latency_cycles, producer_latency);
      }
      state->node_dependency_latency_cycles[node_index] =
          dependency_latency_cycles;
    }
    uint32_t successor_path_cycles = 0;
    uint32_t pressure_demand_units = 0;
    uint32_t pressure_activation_units = 0;
    bool has_effect_consumer = false;
    uint32_t* register_packing_activation_units =
        state->node_register_packing.activation_units != NULL
            ? loom_low_schedule_register_packing_row(
                  state, state->node_register_packing.activation_units,
                  node_index)
            : NULL;
    uint32_t* register_packing_completion_sinks =
        state->node_register_packing.completion_sinks != NULL
            ? loom_low_schedule_register_packing_row(
                  state, state->node_register_packing.completion_sinks,
                  node_index)
            : NULL;
    uint32_t* unspillable_activation_units =
        state->node_unspillable_activation_units != NULL
            ? loom_low_schedule_unspillable_pressure_row(
                  state, state->node_unspillable_activation_units, node_index)
            : NULL;
    if (dependency_details->dependency_count != 0) {
      const uint32_t dependency_begin =
          dependency_details->producer_dependency_starts[node_index];
      const uint32_t dependency_end =
          dependency_details->producer_dependency_starts[node_index + 1];
      for (uint32_t i = dependency_begin; i < dependency_end; ++i) {
        const uint32_t dependency_index =
            loom_low_schedule_dependency_detail_index_at(dependency_details, i);
        const loom_low_schedule_dependency_t* dependency =
            loom_low_schedule_dependency_graph_at(&state->dependencies,
                                                  dependency_index);
        if (dependency->producer_node != node_index ||
            dependency->consumer_node >= node_count) {
          continue;
        }
        const loom_low_schedule_node_t* consumer =
            &state->nodes[dependency->consumer_node];
        if (consumer->block_index != node->block_index) {
          continue;
        }
        has_effect_consumer |=
            dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT;
        if (state->node_critical_path_cycles != NULL) {
          successor_path_cycles = iree_max(
              successor_path_cycles,
              state->node_critical_path_cycles[dependency->consumer_node]);
        }
        if (state->node_pressure_demand_units != NULL &&
            state->node_pressure_activation_units != NULL &&
            dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
          if (is_storage_setup &&
              (consumer->kind == LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
               iree_any_bit_set(
                   consumer->flags,
                   LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP))) {
            node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_DESCRIPTOR_SETUP;
          }
          uint32_t consumer_demand =
              consumer->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL
                  ? state->node_pressure_demand_units[dependency->consumer_node]
                  : 0;
          if (consumer_demand == 0 &&
              dependency->value_operand_index < consumer->operand_count) {
            const loom_value_ordinal_t operand_ordinal =
                loom_low_schedule_node_const_operand_ordinals(
                    consumer)[dependency->value_operand_index];
            consumer_demand = state->values[operand_ordinal].unit_count;
          }
          if (consumer_demand == 0) {
            consumer_demand = 1;
          }
          pressure_demand_units = iree_math_saturating_add_u32(
              pressure_demand_units, consumer_demand);
          const uint32_t consumer_activation =
              consumer->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL
                  ? state->node_pressure_activation_units[dependency
                                                              ->consumer_node]
                  : consumer_demand;
          pressure_activation_units =
              iree_max(pressure_activation_units, consumer_activation);
          if (unspillable_activation_units != NULL) {
            const uint32_t* consumer_activation_units =
                loom_low_schedule_const_unspillable_pressure_row(
                    state, state->node_unspillable_activation_units,
                    dependency->consumer_node);
            const uint16_t completion_domain_count =
                state->pressure_limits.unspillable_completion_domain_count;
            for (uint16_t completion_domain_id = 0;
                 completion_domain_id < completion_domain_count;
                 ++completion_domain_id) {
              const uint64_t node_result_units =
                  loom_low_schedule_compute_unspillable_result_units(
                      state, node, completion_domain_id);
              const uint64_t consumer_operand_units =
                  loom_low_schedule_compute_unspillable_operand_units(
                      state, dependency->consumer_node, completion_domain_id,
                      source_range_start);
              const uint64_t consumer_result_units =
                  loom_low_schedule_compute_unspillable_result_units(
                      state, consumer, completion_domain_id);
              const uint32_t activation_units =
                  loom_low_schedule_compute_downstream_activation_units(
                      node_result_units, consumer_operand_units,
                      consumer_result_units,
                      consumer_activation_units[completion_domain_id],
                      iree_any_bit_set(
                          consumer->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER));
              unspillable_activation_units[completion_domain_id] =
                  iree_max(unspillable_activation_units[completion_domain_id],
                           activation_units);
              if (dependency->value_operand_index >= consumer->operand_count) {
                continue;
              }
              const loom_value_ordinal_t value_ordinal =
                  loom_low_schedule_node_const_operand_ordinals(
                      consumer)[dependency->value_operand_index];
              loom_low_schedule_value_record_t* value =
                  &state->values[value_ordinal];
              if (state->value_producer_nodes[value_ordinal] != node_index ||
                  iree_any_bit_set(value->flags,
                                   LOOM_LOW_SCHEDULE_VALUE_FLAG_FORWARDED) ||
                  loom_low_schedule_unspillable_completion_domain_id(
                      state, value->register_class_id) !=
                      completion_domain_id) {
                continue;
              }
              loom_low_schedule_unspillable_completion_path_t consumer_path =
                  loom_low_schedule_select_unspillable_result_completion(
                      state, consumer, completion_domain_id);
              if (consumer_result_units == 0 &&
                  !loom_low_schedule_node_retains_aggregate_packing_from_class(
                      state, dependency->consumer_node,
                      value->register_class_id)) {
                consumer_path =
                    (loom_low_schedule_unspillable_completion_path_t){
                        .activation_units = 0,
                        .sink = dependency->consumer_node,
                    };
              }
              if (consumer_path.sink != LOOM_LOW_SCHEDULE_NODE_NONE) {
                loom_low_schedule_update_unspillable_completion_path(
                    loom_low_schedule_compute_downstream_activation_units(
                        node_result_units, consumer_operand_units,
                        consumer_result_units, consumer_path.activation_units,
                        iree_any_bit_set(
                            consumer->flags,
                            LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER)),
                    consumer_path.sink, &value->unspillable_completion);
              }
            }
          }
          if (register_packing_activation_units != NULL) {
            const uint32_t* consumer_activation_units =
                loom_low_schedule_const_register_packing_row(
                    state, state->node_register_packing.activation_units,
                    dependency->consumer_node);
            const uint32_t* consumer_completion_sinks =
                loom_low_schedule_const_register_packing_row(
                    state, state->node_register_packing.completion_sinks,
                    dependency->consumer_node);
            const uint16_t resource_count =
                state->target.descriptor_set->register_packing_resource_count;
            const loom_value_ordinal_t source_ordinal =
                loom_low_schedule_node_const_operand_ordinals(
                    consumer)[dependency->value_operand_index];
            const bool forwards_source =
                iree_any_bit_set(state->values[source_ordinal].flags,
                                 LOOM_LOW_SCHEDULE_VALUE_FLAG_FORWARDED);
            for (uint16_t resource_id = 0; resource_id < resource_count;
                 ++resource_id) {
              const loom_low_register_packing_resource_t* resource =
                  &state->target.descriptor_set
                       ->register_packing_resources[resource_id];
              const uint64_t node_result_units =
                  loom_low_schedule_node_register_packing_result_units(
                      state, node_index, resource_id);
              bool consumer_reads_resource = false;
              const uint64_t consumer_operand_units =
                  loom_low_schedule_compute_register_packing_operand_units(
                      state, consumer, resource, source_range_start,
                      &consumer_reads_resource);
              const uint64_t consumer_result_units =
                  loom_low_schedule_node_register_packing_result_units(
                      state, dependency->consumer_node, resource_id);
              register_packing_activation_units[resource_id] = iree_max(
                  register_packing_activation_units[resource_id],
                  loom_low_schedule_compute_downstream_activation_units(
                      node_result_units, consumer_operand_units,
                      consumer_result_units,
                      consumer_activation_units[resource_id],
                      iree_any_bit_set(
                          consumer->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER)));

              if (forwards_source) {
                continue;
              }
              const bool exits_resource =
                  consumer_reads_resource && consumer_result_units == 0;
              const uint32_t completion_sink =
                  exits_resource ? dependency->consumer_node
                                 : consumer_completion_sinks[resource_id];
              if (completion_sink != LOOM_LOW_SCHEDULE_NODE_NONE &&
                  (register_packing_completion_sinks[resource_id] ==
                       LOOM_LOW_SCHEDULE_NODE_NONE ||
                   completion_sink <
                       register_packing_completion_sinks[resource_id])) {
                register_packing_completion_sinks[resource_id] =
                    completion_sink;
              }
            }
          }
        }
      }
    }
    if (state->node_critical_path_cycles != NULL) {
      const uint16_t latency_cycles =
          loom_low_schedule_class_schedule_distance_cycles(
              node->schedule_class);
      state->node_critical_path_cycles[node_index] =
          iree_math_saturating_add_u32(latency_cycles, successor_path_cycles);
    }
    if (state->node_opened_completion_latency_cycles != NULL &&
        has_effect_consumer) {
      uint16_t completion_wait_cycles = 0;
      if (loom_low_schedule_class_query_completion_wait(
              state->target.descriptor_set, node->schedule_class,
              &completion_wait_cycles)) {
        state->node_opened_completion_latency_cycles[node_index] =
            node->schedule_class->latency_cycles;
      }
    }
    if (state->node_pressure_demand_units != NULL) {
      state->node_pressure_demand_units[node_index] =
          pressure_demand_units != 0 ? pressure_demand_units : 1;
    }
    if (state->node_pressure_activation_units != NULL) {
      state->node_pressure_activation_units[node_index] =
          pressure_activation_units != 0 ? pressure_activation_units : 1;
    }
    if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
        state->pressure_resources != NULL) {
      loom_low_schedule_reverse_source_pressure_node(state, pressure_state,
                                                     node);
    }
    if (node_index == block_record->node_start) {
      loom_low_schedule_storage_lifetimes_set_forwarded_values(
          state, node->block_index, false);
    }
  }
  if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
      state->pressure_resources != NULL) {
    // The reverse source sweep shares the existing priority traversal and
    // leaves only its immutable per-class cliff floors behind.
    loom_low_schedule_reset_source_pressure_sweep(state, pressure_state);
  }
}
