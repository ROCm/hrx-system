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

// Selects the earliest exact downstream completion identity. Completion
// anchors use the same order, keeping each hard domain on one coherent path.
static uint32_t loom_low_schedule_merge_completion_sink(uint32_t lhs,
                                                        uint32_t rhs) {
  if (lhs == LOOM_LOW_SCHEDULE_NODE_NONE) return rhs;
  if (rhs == LOOM_LOW_SCHEDULE_NODE_NONE) return lhs;
  return iree_min(lhs, rhs);
}

// Returns a static nomination tier for materializations that may open live
// storage before it becomes actionable. Ordinary work comes first, followed by
// rematerializable leaves that can unlock ordinary consumers, and finally
// storage setup. Exact candidate scoring still recognizes setup that
// immediately advances storage and closes an opened rematerialization chain
// before selecting another leaf.
static uint64_t loom_low_schedule_node_materialization_key(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (iree_any_bit_set(node->flags,
                       LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP)) {
    return UINT64_C(1) << 63;
  }
  if (node->descriptor == NULL || node->operand_count != 0 ||
      node->result_count == 0) {
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
        state->values[operand_ordinals[operand_index]].producer_node;
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

void loom_low_schedule_pressure_compute_node_priorities(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const loom_low_schedule_dependency_detail_index_t* dependency_details,
    loom_low_schedule_pressure_state_t* pressure_state) {
  if (state->node_critical_path_cycles == NULL &&
      state->node_dependency_latency_cycles == NULL &&
      state->node_opened_completion_latency_cycles == NULL &&
      state->node_pressure_demand_units == NULL &&
      state->node_pressure_activation_units == NULL &&
      state->node_register_packing_activation_units == NULL &&
      state->node_unspillable_completion_signatures == NULL &&
      pressure_state->first_actionable_pressure_cliff_indices == NULL) {
    return;
  }
  for (iree_host_size_t i = node_count; i > 0; --i) {
    const uint32_t node_index = (uint32_t)(i - 1);
    loom_low_schedule_node_t* node = &state->nodes[node_index];
    const bool is_storage_setup = iree_any_bit_set(
        node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
    uint16_t dependency_latency_cycles = 0;
    if (state->node_dependency_latency_cycles != NULL) {
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const uint32_t producer_node =
            state->values[operand_ordinals[operand_index]].producer_node;
        if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
            state->nodes[producer_node].block != node->block) {
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
    uint32_t* unspillable_completion_signatures =
        state->node_unspillable_completion_signatures != NULL
            ? loom_low_schedule_unspillable_completion_signature_row(
                  state, state->node_unspillable_completion_signatures,
                  node_index)
            : NULL;
    if (unspillable_completion_signatures != NULL) {
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const uint16_t reg_class_id =
            state->values[operand_ordinals[operand_index]].register_class_id;
        const uint16_t completion_domain_id =
            loom_low_schedule_unspillable_completion_domain_id(state,
                                                               reg_class_id);
        if (completion_domain_id != UINT16_MAX) {
          unspillable_completion_signatures[completion_domain_id] =
              loom_low_schedule_merge_completion_sink(
                  unspillable_completion_signatures[completion_domain_id],
                  node_index);
        }
      }
    }
    uint32_t* register_packing_activation_units =
        state->node_register_packing_activation_units != NULL
            ? loom_low_schedule_register_packing_row(
                  state, state->node_register_packing_activation_units,
                  node_index)
            : NULL;
    uint32_t* register_packing_completion_sinks =
        state->node_register_packing_completion_sinks != NULL
            ? loom_low_schedule_register_packing_row(
                  state, state->node_register_packing_completion_sinks,
                  node_index)
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
        if (unspillable_completion_signatures != NULL &&
            dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
          const uint32_t* consumer_completion_signatures =
              loom_low_schedule_const_unspillable_completion_signature_row(
                  state, state->node_unspillable_completion_signatures,
                  dependency->consumer_node);
          const uint16_t completion_domain_count =
              state->pressure_limits.unspillable_completion_domain_count;
          for (uint16_t completion_domain_id = 0;
               completion_domain_id < completion_domain_count;
               ++completion_domain_id) {
            unspillable_completion_signatures[completion_domain_id] =
                loom_low_schedule_merge_completion_sink(
                    unspillable_completion_signatures[completion_domain_id],
                    consumer_completion_signatures[completion_domain_id]);
          }
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
          if (register_packing_activation_units != NULL) {
            const uint32_t* consumer_activation_units =
                loom_low_schedule_const_register_packing_row(
                    state, state->node_register_packing_activation_units,
                    dependency->consumer_node);
            const uint32_t* consumer_completion_sinks =
                loom_low_schedule_const_register_packing_row(
                    state, state->node_register_packing_completion_sinks,
                    dependency->consumer_node);
            const uint16_t resource_count =
                state->target.descriptor_set->register_packing_resource_count;
            for (uint16_t resource_id = 0; resource_id < resource_count;
                 ++resource_id) {
              const loom_low_register_packing_resource_t* resource =
                  &state->target.descriptor_set
                       ->register_packing_resources[resource_id];
              const uint64_t node_result_units =
                  loom_low_schedule_node_register_packing_result_units(
                      state, node, resource);
              const uint64_t consumer_operand_units =
                  loom_low_schedule_node_register_packing_operand_units(
                      state, consumer, resource);
              const uint64_t consumer_result_units =
                  loom_low_schedule_node_register_packing_result_units(
                      state, consumer, resource);
              const uint64_t consumer_required_units =
                  iree_math_saturating_add_u64(
                      consumer_result_units,
                      consumer_activation_units[resource_id]);
              uint64_t required_units =
                  iree_max(consumer_operand_units, consumer_required_units);
              if (iree_any_bit_set(consumer->flags,
                                   LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER)) {
                required_units = iree_max(
                    required_units,
                    iree_math_saturating_add_u64(consumer_operand_units,
                                                 consumer_result_units));
              }
              const uint64_t activation_units =
                  required_units > node_result_units
                      ? required_units - node_result_units
                      : 0;
              register_packing_activation_units[resource_id] = iree_max(
                  register_packing_activation_units[resource_id],
                  loom_low_schedule_saturate_u64_to_u32(activation_units));

              const bool exits_resource =
                  consumer_operand_units != 0 && consumer_result_units == 0;
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
      const loom_low_schedule_block_t* block_record =
          &state->blocks[node->block_index];
      if (node_index == block_record->node_start) {
        loom_low_schedule_remove_source_pressure_block_arguments(
            state, pressure_state, node->block);
      }
    }
  }
  if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
      state->pressure_resources != NULL) {
    // The reverse source sweep shares the existing priority traversal and
    // leaves only its immutable per-class cliff floors behind.
    loom_low_schedule_reset_source_pressure_sweep(state, pressure_state);
  }
}
