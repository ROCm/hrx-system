// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/graph.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/math.h"

#define LOOM_LOW_SCHEDULE_DEPENDENCY_SET_LINEAR_LIMIT 32
#define LOOM_LOW_SCHEDULE_DEPENDENCY_SET_MIN_CAPACITY 64

typedef struct loom_low_schedule_effect_frontier_t {
  // Latest ordered effect node that every later dependency effect must follow.
  uint32_t ordered_node;
  // Outstanding read nodes not yet subsumed by a later write or ordered effect.
  uint32_t* read_nodes;
  // Access summary for each outstanding read node.
  loom_low_memory_access_summary_t* read_summaries;
  // Number of outstanding read entries.
  iree_host_size_t read_count;
  // Outstanding write nodes not yet subsumed by a later write or ordered
  // effect.
  uint32_t* write_nodes;
  // Access summary for each outstanding write node.
  loom_low_memory_access_summary_t* write_summaries;
  // Number of outstanding write entries.
  iree_host_size_t write_count;
} loom_low_schedule_effect_frontier_t;

typedef struct loom_low_schedule_memory_effect_t {
  // Memory access summary used for alias checks.
  loom_low_memory_access_summary_t summary;
  // True when the node reads memory under summary.
  bool reads;
  // True when the node writes memory under summary.
  bool writes;
  // True when the node has an ordered effect.
  bool ordered;
} loom_low_schedule_memory_effect_t;

static bool loom_low_schedule_memory_summary_equal(
    const loom_low_memory_access_summary_t* lhs,
    const loom_low_memory_access_summary_t* rhs) {
  return lhs->memory_space == rhs->memory_space &&
         lhs->alias_root_id == rhs->alias_root_id &&
         lhs->alias_group_id == rhs->alias_group_id &&
         lhs->precision_flags == rhs->precision_flags &&
         lhs->byte_interval == rhs->byte_interval;
}

static void loom_low_schedule_memory_effect_merge_summary(
    loom_low_schedule_memory_effect_t* effect,
    const loom_low_memory_access_summary_t* summary) {
  if (!effect->reads && !effect->writes && !effect->ordered) {
    effect->summary = *summary;
    return;
  }
  if (loom_low_schedule_memory_summary_equal(&effect->summary, summary)) {
    return;
  }
  effect->summary =
      loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
}

static bool loom_low_schedule_op_is_descriptor_packet(const loom_op_t* op) {
  return loom_low_op_isa(op) || loom_low_const_isa(op);
}

// Structural low operations that may emit target register packets after
// scheduling must carry any target state dependencies their eventual packets
// require.
static bool loom_low_schedule_op_is_structural_materialization(
    const loom_op_t* op) {
  return loom_low_copy_isa(op) || loom_low_slice_isa(op) ||
         loom_low_concat_isa(op) || loom_low_storage_address_isa(op);
}

static bool loom_low_schedule_op_is_terminator(const loom_module_t* module,
                                               const loom_op_t* op) {
  return iree_any_bit_set(loom_op_effective_traits(module, op),
                          LOOM_TRAIT_TERMINATOR);
}

static bool loom_low_schedule_node_has_effects(
    const loom_low_schedule_node_t* node,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor) {
    return descriptor->effect_count != 0 ||
           iree_any_bit_set(descriptor->flags,
                            LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                                LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR);
  }
  return iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY |
                                            LOOM_TRAIT_WRITES_MEMORY |
                                            LOOM_TRAIT_NON_DETERMINISTIC |
                                            LOOM_TRAIT_UNKNOWN_EFFECTS);
}

static iree_status_t loom_low_schedule_resolve_descriptor(
    loom_low_schedule_build_state_t* state, const loom_op_t* op,
    loom_low_schedule_node_t* node,
    const loom_low_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (!loom_low_schedule_op_is_descriptor_packet(op)) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, &state->target, op, &packet));
  if (packet.descriptor == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_missing_descriptor(state, op, packet.key));
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule descriptor '%.*s' is not available",
                            (int)packet.key.size, packet.key.data);
  }

  node->descriptor = packet.descriptor;
  node->effect_count = packet.descriptor->effect_count;
  node->schedule_class_id = packet.descriptor->schedule_class_id;
  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set
           ->schedule_classes[packet.descriptor->schedule_class_id];
  node->latency_cycles = schedule_class->latency_cycles;
  node->latency_kind = schedule_class->latency_kind;
  node->model_quality = schedule_class->model_quality;
  node->issue_use_count = schedule_class->issue_use_count;
  node->hazard_count = schedule_class->hazard_count;
  node->schedule_class_name = loom_low_descriptor_set_string(
      state->target.descriptor_set, schedule_class->name_string_offset);
  *out_descriptor = packet.descriptor;
  return iree_ok_status();
}

static int loom_low_schedule_compare_memory_access_position(
    const loom_low_memory_access_position_t* position, uint16_t block_index,
    uint64_t block_ordinal) {
  const loom_low_memory_access_position_t key = {
      .block_index = block_index,
      .block_ordinal = block_ordinal,
  };
  return loom_low_memory_access_position_compare_order(position, &key);
}

static void loom_low_schedule_bind_memory_access_record(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t block_index, const loom_op_t* op) {
  while (state->memory_access_record_bind_index <
         state->memory_access_record_count) {
    const loom_low_memory_access_record_t* record =
        &state->memory_access_records[state->memory_access_record_bind_index];
    if (record->op != NULL) {
      if (iree_any_bit_set(record->op->flags, LOOM_OP_FLAG_DEAD) ||
          record->op->block_ordinal == 0) {
        ++state->memory_access_record_bind_index;
        continue;
      }
      if (record->op == op) {
        state->nodes[node_index].memory_access_record_index =
            (uint32_t)state->memory_access_record_bind_index++;
      }
      return;
    }
    const int compare = loom_low_schedule_compare_memory_access_position(
        &record->position, block_index, op->block_ordinal);
    if (compare > 0) {
      return;
    }
    if (compare == 0) {
      state->nodes[node_index].memory_access_record_index =
          (uint32_t)state->memory_access_record_bind_index++;
      return;
    }
    ++state->memory_access_record_bind_index;
  }
}

static bool loom_low_schedule_dependency_equal(
    const loom_low_schedule_dependency_t* dependency, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  return dependency->producer_node == producer_node &&
         dependency->consumer_node == consumer_node &&
         dependency->kind == kind && dependency->operand_index == operand_index;
}

static uint64_t loom_low_schedule_dependency_hash_mix(uint64_t value) {
  value ^= value >> 30;
  value *= UINT64_C(0xBF58476D1CE4E5B9);
  value ^= value >> 27;
  value *= UINT64_C(0x94D049BB133111EB);
  value ^= value >> 31;
  return value;
}

static uint64_t loom_low_schedule_dependency_hash(
    uint32_t producer_node, uint32_t consumer_node,
    loom_low_schedule_dependency_kind_t kind, uint32_t operand_index) {
  uint64_t value = (uint64_t)producer_node << 32 | consumer_node;
  value ^= ((uint64_t)kind << 48) ^ operand_index;
  return loom_low_schedule_dependency_hash_mix(value);
}

static bool loom_low_schedule_dependency_set_has_room(iree_host_size_t capacity,
                                                      iree_host_size_t count) {
  return count <= capacity - capacity / 4;
}

static iree_status_t loom_low_schedule_dependency_set_insert_reserved(
    loom_low_schedule_build_state_t* state, uint32_t dependency_index) {
  if (dependency_index == UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency set exceeds uint32_t index capacity");
  }
  const loom_low_schedule_dependency_t* dependency =
      &state->dependencies[dependency_index];
  const iree_host_size_t mask = state->dependency_set_capacity - 1;
  iree_host_size_t slot =
      (iree_host_size_t)loom_low_schedule_dependency_hash(
          dependency->producer_node, dependency->consumer_node,
          dependency->kind, dependency->operand_index) &
      mask;
  while (state->dependency_set_indices[slot] != 0) {
    slot = (slot + 1) & mask;
  }
  state->dependency_set_indices[slot] = dependency_index + 1;
  return iree_ok_status();
}

static bool loom_low_schedule_dependency_set_contains(
    const loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  const iree_host_size_t mask = state->dependency_set_capacity - 1;
  iree_host_size_t slot =
      (iree_host_size_t)loom_low_schedule_dependency_hash(
          producer_node, consumer_node, kind, operand_index) &
      mask;
  while (true) {
    const uint32_t dependency_index_plus_one =
        state->dependency_set_indices[slot];
    if (dependency_index_plus_one == 0) {
      return false;
    }
    const loom_low_schedule_dependency_t* dependency =
        &state->dependencies[dependency_index_plus_one - 1];
    if (loom_low_schedule_dependency_equal(
            dependency, producer_node, consumer_node, kind, operand_index)) {
      return true;
    }
    slot = (slot + 1) & mask;
  }
}

static iree_status_t loom_low_schedule_dependency_set_reserve(
    loom_low_schedule_build_state_t* state, iree_host_size_t required_count) {
  if (required_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency set exceeds uint32_t index capacity");
  }
  if (state->dependency_set_capacity != 0 &&
      loom_low_schedule_dependency_set_has_room(state->dependency_set_capacity,
                                                required_count)) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = LOOM_LOW_SCHEDULE_DEPENDENCY_SET_MIN_CAPACITY;
  if (state->dependency_set_capacity != 0) {
    if (state->dependency_set_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "low schedule dependency set capacity exceeds host size");
    }
    new_capacity = state->dependency_set_capacity * 2;
  }
  while (!loom_low_schedule_dependency_set_has_room(new_capacity,
                                                    required_count)) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "low schedule dependency set capacity exceeds host size");
    }
    new_capacity *= 2;
  }

  uint32_t* new_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, new_capacity, sizeof(*new_indices), (void**)&new_indices));
  memset(new_indices, 0, new_capacity * sizeof(*new_indices));
  state->dependency_set_indices = new_indices;
  state->dependency_set_capacity = new_capacity;
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_dependency_set_insert_reserved(state, (uint32_t)i));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  if (state->dependency_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t index capacity");
  }
  if (state->dependency_count >= state->dependency_capacity) {
    iree_host_size_t new_capacity =
        state->dependency_capacity == 0 ? 16 : state->dependency_capacity * 2;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, state->dependency_count,
                              new_capacity, sizeof(*state->dependencies),
                              &new_capacity, (void**)&state->dependencies));
    state->dependency_capacity = new_capacity;
  }
  state->dependencies[state->dependency_count++] =
      (loom_low_schedule_dependency_t){
          .producer_node = producer_node,
          .consumer_node = consumer_node,
          .kind = kind,
          .operand_index = operand_index,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_append_visibility_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind) {
  if (producer_node == consumer_node) {
    return iree_ok_status();
  }
  if (state->visibility_dependency_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule visibility dependency count exceeds uint32_t index "
        "capacity");
  }
  if (state->visibility_dependency_count >=
      state->visibility_dependency_capacity) {
    iree_host_size_t new_capacity =
        state->visibility_dependency_capacity == 0
            ? 16
            : state->visibility_dependency_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->visibility_dependency_count, new_capacity,
        sizeof(*state->visibility_dependencies), &new_capacity,
        (void**)&state->visibility_dependencies));
    state->visibility_dependency_capacity = new_capacity;
  }
  state->visibility_dependencies[state->visibility_dependency_count++] =
      (loom_low_schedule_visibility_dependency_t){
          .producer_node = producer_node,
          .consumer_node = consumer_node,
          .kind = kind,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint32_t operand_index) {
  if (producer_node == consumer_node) {
    return iree_ok_status();
  }
  if (state->dependency_set_capacity == 0 &&
      state->dependency_count < LOOM_LOW_SCHEDULE_DEPENDENCY_SET_LINEAR_LIMIT) {
    for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
      if (loom_low_schedule_dependency_equal(&state->dependencies[i],
                                             producer_node, consumer_node, kind,
                                             operand_index)) {
        return iree_ok_status();
      }
    }
    if (state->dependency_count + 1 ==
        LOOM_LOW_SCHEDULE_DEPENDENCY_SET_LINEAR_LIMIT) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_set_reserve(
          state, state->dependency_count + 1));
      IREE_RETURN_IF_ERROR(loom_low_schedule_append_dependency(
          state, producer_node, consumer_node, kind, operand_index));
      IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_set_insert_reserved(
          state, (uint32_t)(state->dependency_count - 1)));
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_append_dependency(
        state, producer_node, consumer_node, kind, operand_index));
    return iree_ok_status();
  }

  if (state->dependency_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t index capacity");
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_set_reserve(
      state, state->dependency_count + 1));
  if (loom_low_schedule_dependency_set_contains(
          state, producer_node, consumer_node, kind, operand_index)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_append_dependency(
      state, producer_node, consumer_node, kind, operand_index));
  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_set_insert_reserved(
      state, (uint32_t)(state->dependency_count - 1)));
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_descriptor_operand_reg_class_id(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    uint16_t* out_reg_class_id) {
  *out_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  const uint32_t operand_row =
      descriptor->operand_start + descriptor_operand_index;
  if (operand_row >= descriptor_set->operand_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low schedule descriptor state operand row is out of range");
  }
  const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
  if (operand->reg_class_alt_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule state operand must have one register-class alternative");
  }
  const uint32_t alt_index = operand->reg_class_alt_start;
  if (alt_index >= descriptor_set->reg_class_alt_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low schedule state operand register-class alternative is out of "
        "range");
  }
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[alt_index];
  if (iree_any_bit_set(alt->flags, LOOM_LOW_REG_CLASS_ALT_FLAG_IMMEDIATE) ||
      alt->reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      alt->reg_class_id >= descriptor_set->reg_class_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule state operand must name a concrete register class");
  }
  *out_reg_class_id = alt->reg_class_id;
  return iree_ok_status();
}

static bool loom_low_schedule_reg_class_is_state(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id) {
  return reg_class_id < state->target.descriptor_set->reg_class_count &&
         state->reg_class_state_flags != NULL &&
         state->reg_class_state_flags[reg_class_id] != 0;
}

static iree_status_t loom_low_schedule_lookup_descriptor_packet_operand(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, uint16_t packet_operand_index,
    uint16_t* out_descriptor_operand_index,
    const loom_low_operand_t** out_operand) {
  const uint16_t descriptor_operand_index =
      loom_low_descriptor_packet_operand_descriptor_index(
          descriptor_set, descriptor, packet_operand_index);
  if (descriptor_operand_index == LOOM_LOW_ID_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule descriptor operand count is less than packet operand "
        "count");
  }
  if (out_descriptor_operand_index != NULL) {
    *out_descriptor_operand_index = descriptor_operand_index;
  }
  if (out_operand != NULL) {
    *out_operand =
        &descriptor_set
             ->operands[descriptor->operand_start + descriptor_operand_index];
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id) {
  if (state->state_read_record_count >= state->state_read_record_capacity) {
    iree_host_size_t new_capacity = state->state_read_record_capacity == 0
                                        ? 16
                                        : state->state_read_record_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->state_read_record_count, new_capacity,
        sizeof(*state->state_read_records), &new_capacity,
        (void**)&state->state_read_records));
    state->state_read_record_capacity = new_capacity;
  }
  state->state_read_records[state->state_read_record_count] =
      (loom_low_schedule_state_read_record_t){
          .node_index = node_index,
          .next_record = state->state_read_heads[reg_class_id],
      };
  state->state_read_heads[reg_class_id] =
      (uint32_t)state->state_read_record_count++;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_chain_read(
    loom_low_schedule_build_state_t* state, uint32_t producer_node_index,
    uint32_t reader_node_index) {
  if (producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      producer_node_index == reader_node_index ||
      state->state_chain_read_heads == NULL) {
    return iree_ok_status();
  }
  if (state->nodes[producer_node_index].block !=
      state->nodes[reader_node_index].block) {
    return iree_ok_status();
  }
  if (state->state_chain_read_record_count >=
      state->state_chain_read_record_capacity) {
    iree_host_size_t new_capacity =
        state->state_chain_read_record_capacity == 0
            ? 16
            : state->state_chain_read_record_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->state_chain_read_record_count, new_capacity,
        sizeof(*state->state_chain_read_records), &new_capacity,
        (void**)&state->state_chain_read_records));
    state->state_chain_read_record_capacity = new_capacity;
  }
  state->state_chain_read_records[state->state_chain_read_record_count] =
      (loom_low_schedule_state_chain_read_record_t){
          .reader_node = reader_node_index,
          .next_record = state->state_chain_read_heads[producer_node_index],
      };
  state->state_chain_read_heads[producer_node_index] =
      (uint32_t)state->state_chain_read_record_count++;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_chain_read_dependencies(
    loom_low_schedule_build_state_t* state, uint32_t producer_node_index,
    uint32_t consumer_node_index) {
  if (producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      producer_node_index == consumer_node_index ||
      state->state_chain_read_heads == NULL) {
    return iree_ok_status();
  }
  if (state->nodes[producer_node_index].block !=
      state->nodes[consumer_node_index].block) {
    return iree_ok_status();
  }
  uint32_t read_record_index =
      state->state_chain_read_heads[producer_node_index];
  while (read_record_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    const loom_low_schedule_state_chain_read_record_t* read_record =
        &state->state_chain_read_records[read_record_index];
    if (read_record->reader_node != consumer_node_index &&
        state->nodes[read_record->reader_node].source_ordinal <
            state->nodes[consumer_node_index].source_ordinal) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
          state, read_record->reader_node, consumer_node_index,
          LOOM_LOW_SCHEDULE_DEPENDENCY_STATE, UINT32_MAX));
    }
    read_record_index = read_record->next_record;
  }
  return iree_ok_status();
}

static void loom_low_schedule_reset_storage_reads(
    loom_low_schedule_build_state_t* state) {
  if (state->storage_reads.heads == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < state->storage_reads.touched_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        state->storage_reads.touched_ordinals[i];
    state->storage_reads.heads[value_ordinal] = LOOM_LOW_SCHEDULE_NODE_NONE;
    state->values[value_ordinal].flags &=
        ~LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TOUCHED;
  }
  state->storage_reads.touched_count = 0;
  state->storage_reads.record_count = 0;
}

static loom_low_register_part_mask_t loom_low_schedule_value_full_storage_mask(
    const loom_low_schedule_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  const uint16_t reg_class_id = state->values[value_ordinal].register_class_id;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      reg_class_id >= state->target.descriptor_set->reg_class_count) {
    return UINT32_MAX;
  }
  return state->target.descriptor_set->reg_classes[reg_class_id]
      .full_register_part_mask;
}

static loom_low_register_part_mask_t
loom_low_schedule_descriptor_operand_storage_mask(
    const loom_low_schedule_build_state_t* state,
    const loom_low_descriptor_t* descriptor, uint16_t descriptor_operand_index,
    loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_LT(descriptor_operand_index, descriptor->operand_count);
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const loom_low_operand_t* operand =
      &descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  if (operand->register_part_id == LOOM_LOW_REGISTER_PART_NONE) {
    return loom_low_schedule_value_full_storage_mask(state, value_ordinal);
  }
  IREE_ASSERT_LT(operand->register_part_id,
                 descriptor_set->register_part_count);
  return descriptor_set->register_parts[operand->register_part_id].mask;
}

static void loom_low_schedule_touch_storage_read_value(
    loom_low_schedule_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  if (!iree_any_bit_set(state->values[value_ordinal].flags,
                        LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TOUCHED)) {
    state->storage_reads
        .touched_ordinals[state->storage_reads.touched_count++] = value_ordinal;
    state->values[value_ordinal].flags |=
        LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TOUCHED;
  }
}

static iree_status_t loom_low_schedule_add_storage_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    loom_value_ordinal_t value_ordinal,
    loom_low_register_part_mask_t read_mask) {
  if (state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(read_mask, 0u);
  loom_low_schedule_touch_storage_read_value(state, value_ordinal);
  if (state->storage_reads.record_count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule storage-read record count exceeds uint32_t");
  }
  if (state->storage_reads.record_count >=
      state->storage_reads.record_capacity) {
    iree_host_size_t new_capacity =
        state->storage_reads.record_capacity == 0
            ? 16
            : state->storage_reads.record_capacity * 2;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->storage_reads.record_count, new_capacity,
        sizeof(*state->storage_reads.records), &new_capacity,
        (void**)&state->storage_reads.records));
    state->storage_reads.record_capacity = new_capacity;
  }
  state->storage_reads.records[state->storage_reads.record_count] =
      (loom_low_schedule_storage_read_record_t){
          .reader_node = node_index,
          .read_mask = read_mask,
          .next_record = state->storage_reads.heads[value_ordinal],
      };
  state->storage_reads.heads[value_ordinal] =
      (uint32_t)state->storage_reads.record_count++;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_storage_write_dependencies(
    loom_low_schedule_build_state_t* state, uint32_t writer_node_index,
    uint16_t tied_operand_index, loom_value_ordinal_t value_ordinal,
    loom_value_ordinal_t result_value_ordinal,
    loom_low_register_part_mask_t write_mask) {
  if (state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(write_mask, 0u);
  uint32_t read_record_index = state->storage_reads.heads[value_ordinal];
  uint32_t retained_head = LOOM_LOW_SCHEDULE_NODE_NONE;
  uint32_t retained_tail = LOOM_LOW_SCHEDULE_NODE_NONE;
  while (read_record_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    loom_low_schedule_storage_read_record_t* read_record =
        &state->storage_reads.records[read_record_index];
    const uint32_t next_record_index = read_record->next_record;
    read_record->next_record = LOOM_LOW_SCHEDULE_NODE_NONE;
    if (iree_any_bit_set(read_record->read_mask, write_mask)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
          state, read_record->reader_node, writer_node_index,
          LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, tied_operand_index));
    } else if (retained_tail == LOOM_LOW_SCHEDULE_NODE_NONE) {
      retained_head = read_record_index;
      retained_tail = read_record_index;
    } else {
      state->storage_reads.records[retained_tail].next_record =
          read_record_index;
      retained_tail = read_record_index;
    }
    read_record_index = next_record_index;
  }
  state->storage_reads.heads[value_ordinal] = LOOM_LOW_SCHEDULE_NODE_NONE;
  if (retained_head == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  if (value_ordinal == result_value_ordinal) {
    state->storage_reads.heads[value_ordinal] = retained_head;
    return iree_ok_status();
  }
  loom_low_schedule_touch_storage_read_value(state, result_value_ordinal);
  state->storage_reads.records[retained_tail].next_record =
      state->storage_reads.heads[result_value_ordinal];
  state->storage_reads.heads[result_value_ordinal] = retained_head;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_tied_storage_writes(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const loom_op_t* op = node->op;
  if (op->tied_result_count == 0 || state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    if (tied.result_index >= node->result_count ||
        tied.operand_index >= node->operand_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "low schedule saw malformed tied result "
                              "metadata");
    }
    loom_low_register_part_mask_t write_mask =
        loom_low_schedule_value_full_storage_mask(
            state, result_ordinals[tied.result_index]);
    if (node->descriptor != NULL &&
        tied.result_index < node->descriptor->result_count) {
      write_mask = loom_low_schedule_descriptor_operand_storage_mask(
          state, node->descriptor, tied.result_index,
          result_ordinals[tied.result_index]);
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_write_dependencies(
        state, node_index, tied.operand_index,
        operand_ordinals[tied.operand_index],
        result_ordinals[tied.result_index], write_mask));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_storage_reads(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  if (state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  if (descriptor == NULL) {
    for (uint16_t operand_index = 0; operand_index < node->operand_count;
         ++operand_index) {
      const loom_value_ordinal_t value_ordinal =
          operand_ordinals[operand_index];
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_read(
          state, node_index, value_ordinal,
          loom_low_schedule_value_full_storage_mask(state, value_ordinal)));
    }
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t packet_operand_index = 0;
       packet_operand_index < node->operand_count; ++packet_operand_index) {
    uint16_t descriptor_operand_index = LOOM_LOW_ID_NONE;
    IREE_RETURN_IF_ERROR(loom_low_schedule_lookup_descriptor_packet_operand(
        descriptor_set, descriptor, packet_operand_index,
        &descriptor_operand_index, NULL));
    const loom_value_ordinal_t value_ordinal =
        operand_ordinals[packet_operand_index];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_read(
        state, node_index, value_ordinal,
        loom_low_schedule_descriptor_operand_storage_mask(
            state, descriptor, descriptor_operand_index, value_ordinal)));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_read_dependencies(
    loom_low_schedule_build_state_t* state, uint32_t writer_node_index,
    uint16_t reg_class_id) {
  uint32_t read_record_index = state->state_read_heads[reg_class_id];
  while (read_record_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    const loom_low_schedule_state_read_record_t* read_record =
        &state->state_read_records[read_record_index];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read_record->node_index, writer_node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_STATE, UINT32_MAX));
    read_record_index = read_record->next_record;
  }
  state->state_read_heads[reg_class_id] = LOOM_LOW_SCHEDULE_NODE_NONE;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_state_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_write, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_add_state_chain_read(state, last_write, node_index));
  }
  return loom_low_schedule_add_state_read(state, node_index, reg_class_id);
}

static iree_status_t loom_low_schedule_note_state_write(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_write, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_read_dependencies(
      state, node_index, reg_class_id));
  state->state_last_write_nodes[reg_class_id] = node_index;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_state_fence(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, last_write, node_index, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_read_dependencies(
      state, node_index, reg_class_id));
  state->state_last_write_nodes[reg_class_id] = node_index;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_explicit_state_value_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    loom_value_ordinal_t value_ordinal, uint16_t reg_class_id) {
  const loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  const uint32_t producer_node = value->producer_node;
  const bool has_same_block_producer =
      producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
      state->nodes[producer_node].block == state->nodes[node_index].block;
  const uint32_t last_write = state->state_last_write_nodes[reg_class_id];
  if (last_write != LOOM_LOW_SCHEDULE_NODE_NONE &&
      (!has_same_block_producer || last_write != producer_node)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, node_index, last_write, LOOM_LOW_SCHEDULE_DEPENDENCY_STATE,
        UINT32_MAX));
  }
  if (has_same_block_producer) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_chain_read(
        state, producer_node, node_index));
  }
  return loom_low_schedule_add_state_read(state, node_index, reg_class_id);
}

static iree_status_t loom_low_schedule_note_state_value_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    loom_value_ordinal_t value_ordinal) {
  const uint16_t reg_class_id = state->values[value_ordinal].register_class_id;
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  return loom_low_schedule_note_explicit_state_value_read(
      state, node_index, value_ordinal, reg_class_id);
}

static bool loom_low_schedule_effect_is_ordered(
    const loom_low_effect_t* effect) {
  if (iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_ORDERED)) {
    return true;
  }
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return false;
    case LOOM_LOW_EFFECT_KIND_UNKNOWN:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
    default:
      return true;
  }
}

static bool loom_low_schedule_effect_orders_memory(
    const loom_low_effect_t* effect) {
  if (!loom_low_schedule_effect_is_ordered(effect)) {
    return false;
  }
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
    case LOOM_LOW_EFFECT_KIND_WRITE:
    case LOOM_LOW_EFFECT_KIND_UNKNOWN:
    case LOOM_LOW_EFFECT_KIND_CALL:
    case LOOM_LOW_EFFECT_KIND_BARRIER:
    case LOOM_LOW_EFFECT_KIND_COUNTER:
    case LOOM_LOW_EFFECT_KIND_CONTROL:
      return true;
    case LOOM_LOW_EFFECT_KIND_CONVERGENT:
    default:
      return effect->memory_space != LOOM_LOW_MEMORY_SPACE_NONE;
  }
}

static bool loom_low_schedule_descriptor_has_ordered_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_is_ordered(effect)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_schedule_descriptor_state_read_has_explicit_value(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  if (!loom_low_descriptor_operand_maps_to_packet_operand(
          descriptor_set, descriptor, descriptor_operand_index)) {
    return false;
  }
  return (uint32_t)loom_low_descriptor_operand_packet_index(
             descriptor_set, descriptor, descriptor_operand_index) <
         node->operand_count;
}

static iree_status_t loom_low_schedule_note_descriptor_state_accesses(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const bool has_ordered_effect =
      loom_low_schedule_descriptor_has_ordered_effect(descriptor_set,
                                                      descriptor);
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_row = descriptor->operand_start + i;
    const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
    const uint16_t state_flags =
        operand->flags &
        (LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    if (state_flags == 0) {
      continue;
    }
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_low_schedule_descriptor_operand_reg_class_id(
        descriptor_set, descriptor, i, &reg_class_id));
    if (!iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      continue;
    }
    if (!loom_low_schedule_descriptor_state_read_has_explicit_value(
            descriptor_set, &state->nodes[node_index], descriptor, i)) {
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_state_read(state, node_index, reg_class_id));
    }
    if (has_ordered_effect) {
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_state_fence(state, node_index, reg_class_id));
    }
  }
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_row = descriptor->operand_start + i;
    const loom_low_operand_t* operand = &descriptor_set->operands[operand_row];
    const uint16_t state_flags =
        operand->flags &
        (LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    if (!iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      continue;
    }
    uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    IREE_RETURN_IF_ERROR(loom_low_schedule_descriptor_operand_reg_class_id(
        descriptor_set, descriptor, i, &reg_class_id));
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_note_state_write(state, node_index, reg_class_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_structural_state_reads(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_structural_state_read_list_t state_reads =
      state->options->structural_state_reads;
  if (loom_low_schedule_structural_state_read_list_is_empty(state_reads)) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (node->descriptor != NULL ||
      !loom_low_schedule_op_is_structural_materialization(node->op)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (iree_host_size_t i = 0; i < state_reads.count; ++i) {
    const loom_low_schedule_structural_state_read_t* row =
        &state_reads.values[i];
    bool has_matching_result = false;
    for (uint16_t result_index = 0; result_index < node->result_count;
         ++result_index) {
      const loom_value_ordinal_t result_ordinal = result_ordinals[result_index];
      if (state->values[result_ordinal].register_class_id ==
          row->result_reg_class_id) {
        has_matching_result = true;
        break;
      }
    }
    if (has_matching_result) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_read(
          state, node_index, row->state_reg_class_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_node_value_ordinals(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node) {
  const loom_op_t* op = node->op;
  node->operand_count = op->operand_count;
  node->result_count = op->result_count;
  const uint32_t total_value_count =
      (uint32_t)op->operand_count + (uint32_t)op->result_count;
  if (total_value_count >
      LOOM_LOW_SCHEDULE_NODE_INLINE_VALUE_ORDINAL_CAPACITY) {
    node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_VALUE_ORDINALS_OVERFLOW;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, total_value_count,
        sizeof(*node->value_ordinals.overflow_value_ordinals),
        (void**)&node->value_ordinals.overflow_value_ordinals));
  }
  loom_value_ordinal_t* value_ordinals =
      loom_low_schedule_node_value_ordinals(node);
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    value_ordinals[i] =
        loom_local_value_domain_ordinal(state->value_domain, operands[i]);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  loom_value_ordinal_t* result_ordinals = value_ordinals + op->operand_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    result_ordinals[i] =
        loom_local_value_domain_ordinal(state->value_domain, results[i]);
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_fill_nodes(
    loom_low_schedule_build_state_t* state) {
  uint32_t next_node_index = 0;
  for (uint16_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_block_t* block = state->body->blocks[block_index];
    if (!block) {
      continue;
    }
    state->blocks[block_index] = (loom_low_schedule_block_t){
        .block = block,
        .node_start = next_node_index,
        .node_count = block->op_count,
        .scheduled_node_start = next_node_index,
        .scheduled_node_count = block->op_count,
    };

    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_schedule_node_t* node = &state->nodes[next_node_index];
      *node = (loom_low_schedule_node_t){
          .op = op,
          .block = block,
          .block_index = block_index,
          .source_ordinal = next_node_index,
          .scheduled_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
          .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
          .traits = loom_op_effective_traits(state->module, op),
          .descriptor = NULL,
          .memory_access_record_index =
              LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE,
          .schedule_class_id = LOOM_LOW_SCHEDULE_CLASS_NONE,
      };
      if (loom_low_schedule_op_is_terminator(state->module, op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
      } else if (loom_low_schedule_op_is_descriptor_packet(op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
      }

      const loom_low_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_resolve_descriptor(state, op, node, &descriptor));
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_initialize_node_value_ordinals(state, node));
      loom_low_schedule_bind_memory_access_record(state, next_node_index,
                                                  block_index, op);

      const loom_value_ordinal_t* result_ordinals =
          loom_low_schedule_node_const_result_ordinals(node);
      for (uint16_t result_index = 0; result_index < node->result_count;
           ++result_index) {
        state->values[result_ordinals[result_index]].producer_node =
            next_node_index;
      }
      ++next_node_index;
    }
  }
  return iree_ok_status();
}

static void loom_low_schedule_effect_frontier_reset(
    loom_low_schedule_effect_frontier_t* frontier) {
  frontier->ordered_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  frontier->read_count = 0;
  frontier->write_count = 0;
}

static void loom_low_schedule_effect_frontier_initialize(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* out_frontier) {
  *out_frontier = (loom_low_schedule_effect_frontier_t){
      .read_nodes = state->effect_read_nodes,
      .read_summaries = state->effect_read_summaries,
      .write_nodes = state->effect_write_nodes,
      .write_summaries = state->effect_write_summaries,
  };
  loom_low_schedule_effect_frontier_reset(out_frontier);
}

static iree_status_t loom_low_schedule_effect_frontier_depend_on_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index) {
  if (frontier->ordered_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  return loom_low_schedule_add_dependency(
      state, frontier->ordered_node, node_index,
      LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX);
}

static iree_status_t loom_low_schedule_effect_frontier_note_read(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    if (!loom_low_memory_access_summaries_may_alias(
            summary, &frontier->write_summaries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->write_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  IREE_ASSERT(frontier->read_count < state->effect_read_capacity,
              "precomputed effect-frontier read capacity must cover all rows");
  frontier->read_nodes[frontier->read_count] = node_index;
  frontier->read_summaries[frontier->read_count] = *summary;
  ++frontier->read_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_write_complete(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary) {
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < frontier->read_count;
       ++read_index) {
    const loom_low_memory_access_summary_t* read_summary =
        &frontier->read_summaries[read_index];
    if (loom_low_memory_access_write_subsumes_read(summary, read_summary)) {
      continue;
    }
    frontier->read_nodes[write_index] = frontier->read_nodes[read_index];
    frontier->read_summaries[write_index] = *read_summary;
    ++write_index;
  }
  frontier->read_count = write_index;
  write_index = 0;
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    if (loom_low_memory_access_write_subsumes_access(
            summary, &frontier->write_summaries[i])) {
      continue;
    }
    frontier->write_nodes[write_index] = frontier->write_nodes[i];
    frontier->write_summaries[write_index] = frontier->write_summaries[i];
    ++write_index;
  }
  frontier->write_count = write_index;
  IREE_ASSERT(frontier->write_count < state->effect_write_capacity,
              "precomputed effect-frontier write capacity must cover all rows");
  frontier->write_nodes[frontier->write_count] = node_index;
  frontier->write_summaries[frontier->write_count] = *summary;
  ++frontier->write_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_write(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    if (!loom_low_memory_access_summaries_may_alias(
            summary, &frontier->write_summaries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->write_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    if (!loom_low_memory_access_summaries_may_alias(
            summary, &frontier->read_summaries[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->read_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  return loom_low_schedule_effect_frontier_note_write_complete(
      state, frontier, node_index, summary);
}

static iree_status_t loom_low_schedule_effect_frontier_note_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->write_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, frontier->read_nodes[i], node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, UINT32_MAX));
  }
  loom_low_schedule_effect_frontier_reset(frontier);
  frontier->ordered_node = node_index;
  return iree_ok_status();
}

static uint16_t loom_low_schedule_descriptor_dependency_memory_effect_count(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  uint16_t count = 0;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    if (effect->kind == LOOM_LOW_EFFECT_KIND_READ ||
        effect->kind == LOOM_LOW_EFFECT_KIND_WRITE) {
      ++count;
    }
  }
  return count;
}

static const loom_low_memory_access_summary_t*
loom_low_schedule_lookup_memory_access_summary(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  const uint32_t record_index =
      state->nodes[node_index].memory_access_record_index;
  if (record_index == LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE) {
    return NULL;
  }
  if (record_index >= state->memory_access_record_count) {
    return NULL;
  }
  if (loom_low_schedule_descriptor_dependency_memory_effect_count(
          state->target.descriptor_set, descriptor) != 1) {
    return NULL;
  }
  const loom_low_memory_access_record_t* record =
      &state->memory_access_records[record_index];
  return &record->summary;
}

static iree_status_t loom_low_schedule_resolve_descriptor_memory_effect(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor,
    loom_low_schedule_memory_effect_t* out_effect) {
  *out_effect = (loom_low_schedule_memory_effect_t){0};
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }

  const loom_low_memory_access_summary_t* source_summary =
      loom_low_schedule_lookup_memory_access_summary(state, node_index,
                                                     descriptor);
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_orders_memory(effect)) {
      loom_low_memory_access_summary_t summary =
          loom_low_memory_access_summary_synthetic(
              LOOM_LOW_MEMORY_SPACE_GENERIC);
      loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
      out_effect->ordered = true;
      return iree_ok_status();
    }
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_from_effect(effect);
    if (source_summary != NULL) {
      summary = *source_summary;
    }
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ:
        loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
        out_effect->reads = true;
        break;
      case LOOM_LOW_EFFECT_KIND_WRITE:
        loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
        out_effect->writes = true;
        break;
      default: {
        if (loom_low_schedule_effect_orders_memory(effect)) {
          loom_low_memory_access_summary_t generic_summary =
              loom_low_memory_access_summary_synthetic(
                  LOOM_LOW_MEMORY_SPACE_GENERIC);
          loom_low_schedule_memory_effect_merge_summary(out_effect,
                                                        &generic_summary);
          out_effect->ordered = true;
          return iree_ok_status();
        }
        break;
      }
    }
  }
  return iree_ok_status();
}

static void loom_low_schedule_resolve_structural_memory_effect(
    const loom_low_schedule_node_t* node,
    loom_low_schedule_memory_effect_t* out_effect) {
  *out_effect = (loom_low_schedule_memory_effect_t){0};
  const loom_low_memory_access_summary_t summary =
      loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                         LOOM_TRAIT_UNKNOWN_EFFECTS |
                                         LOOM_TRAIT_CONVERGENT)) {
    loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
    out_effect->ordered = true;
    return;
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
    loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
    out_effect->reads = true;
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
    loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
    out_effect->writes = true;
  }
}

static bool loom_low_schedule_node_observes_program_exit_memory(
    const loom_low_schedule_node_t* node) {
  return node->op != NULL && loom_low_return_isa(node->op);
}

static void loom_low_schedule_resolve_program_exit_memory_effect(
    loom_low_schedule_memory_effect_t* out_effect) {
  *out_effect = (loom_low_schedule_memory_effect_t){0};
  loom_low_memory_access_summary_t summary =
      loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
  loom_low_schedule_memory_effect_merge_summary(out_effect, &summary);
  out_effect->reads = true;
}

static iree_status_t loom_low_schedule_resolve_memory_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_memory_effect_t* effects, uint32_t node_count) {
  for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_low_descriptor_t* descriptor = node->descriptor;
    if (descriptor != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_resolve_descriptor_memory_effect(
          state, node_index, descriptor, &effects[node_index]));
      continue;
    }
    if (loom_low_schedule_node_observes_program_exit_memory(node)) {
      loom_low_schedule_resolve_program_exit_memory_effect(
          &effects[node_index]);
      continue;
    }
    if (loom_low_schedule_node_has_effects(node, NULL)) {
      loom_low_schedule_resolve_structural_memory_effect(node,
                                                         &effects[node_index]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_descriptor_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  const loom_low_memory_access_summary_t* source_summary =
      loom_low_schedule_lookup_memory_access_summary(state, node_index,
                                                     descriptor);
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_orders_memory(effect)) {
      return loom_low_schedule_effect_frontier_note_ordered(state, frontier,
                                                            node_index);
    }
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        loom_low_memory_access_summary_t summary =
            loom_low_memory_access_summary_from_effect(effect);
        if (source_summary != NULL) {
          summary = *source_summary;
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_read(
            state, frontier, node_index, &summary));
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        loom_low_memory_access_summary_t summary =
            loom_low_memory_access_summary_from_effect(effect);
        if (source_summary != NULL) {
          summary = *source_summary;
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_write(
            state, frontier, node_index, &summary));
        break;
      }
      default:
        if (loom_low_schedule_effect_orders_memory(effect)) {
          return loom_low_schedule_effect_frontier_note_ordered(state, frontier,
                                                                node_index);
        }
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_structural_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                         LOOM_TRAIT_UNKNOWN_EFFECTS |
                                         LOOM_TRAIT_CONVERGENT)) {
    return loom_low_schedule_effect_frontier_note_ordered(state, frontier,
                                                          node_index);
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_write(state, frontier,
                                                        node_index, &summary);
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_read(state, frontier,
                                                       node_index, &summary);
  }
  return iree_ok_status();
}

static iree_host_size_t loom_low_schedule_bitset_word_count(
    iree_host_size_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static bool loom_low_schedule_bitset_test(const uint64_t* bits,
                                          uint32_t bit_index) {
  return iree_any_bit_set(bits[bit_index / 64u], UINT64_C(1)
                                                     << (bit_index % 64u));
}

static void loom_low_schedule_bitset_set(uint64_t* bits, uint32_t bit_index) {
  bits[bit_index / 64u] |= UINT64_C(1) << (bit_index % 64u);
}

static bool loom_low_schedule_bitset_copy_changed(uint64_t* target,
                                                  const uint64_t* source,
                                                  iree_host_size_t word_count) {
  bool changed = false;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    changed = changed || target[i] != source[i];
    target[i] = source[i];
  }
  return changed;
}

static bool loom_low_schedule_bitset_union_changed(
    uint64_t* target, const uint64_t* source, iree_host_size_t word_count) {
  bool changed = false;
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    const uint64_t next = target[i] | source[i];
    changed = changed || next != target[i];
    target[i] = next;
  }
  return changed;
}

static uint64_t* loom_low_schedule_block_bitset(uint64_t* bitsets,
                                                iree_host_size_t word_count,
                                                uint16_t block_index) {
  return bitsets + (iree_host_size_t)block_index * word_count;
}

static bool loom_low_schedule_effect_observes_writes(
    const loom_low_schedule_memory_effect_t* effect) {
  return effect->reads || effect->writes || effect->ordered;
}

static iree_status_t loom_low_schedule_add_cross_block_visibility_dependency(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_memory_effect_t* effects, uint32_t producer_node,
    uint32_t consumer_node) {
  if (state->nodes[producer_node].block_index ==
      state->nodes[consumer_node].block_index) {
    return iree_ok_status();
  }
  if (!loom_low_memory_access_summaries_may_alias(
          &effects[producer_node].summary, &effects[consumer_node].summary)) {
    return iree_ok_status();
  }
  return loom_low_schedule_append_visibility_dependency(
      state, producer_node, consumer_node, LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT);
}

static iree_status_t loom_low_schedule_add_visibility_dependencies_for_block(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_memory_effect_t* effects, uint32_t node_count,
    const uint64_t* incoming_writes, uint64_t* active_writes,
    const loom_low_schedule_block_t* block_record,
    iree_host_size_t word_count) {
  memcpy(active_writes, incoming_writes, word_count * sizeof(*active_writes));
  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t node_index = block_record->node_start;
       node_index < block_node_end; ++node_index) {
    const loom_low_schedule_memory_effect_t* effect = &effects[node_index];
    if (loom_low_schedule_effect_observes_writes(effect)) {
      for (iree_host_size_t word_index = 0; word_index < word_count;
           ++word_index) {
        uint64_t active_word = active_writes[word_index];
        while (active_word != 0) {
          const uint32_t producer_node =
              (uint32_t)(word_index * 64u) +
              (uint32_t)iree_math_count_trailing_zeros_u64(active_word);
          if (producer_node < node_count) {
            IREE_RETURN_IF_ERROR(
                loom_low_schedule_add_cross_block_visibility_dependency(
                    state, effects, producer_node, node_index));
          }
          active_word &= active_word - 1u;
        }
      }
    }
    if (effect->writes || effect->ordered) {
      loom_low_schedule_bitset_set(active_writes, node_index);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_build_visibility_dependencies(
    loom_low_schedule_build_state_t* state, uint32_t node_count) {
  if (state->body->block_count <= 1 || node_count == 0) {
    return iree_ok_status();
  }

  loom_low_schedule_memory_effect_t* effects = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*effects), (void**)&effects));
  memset(effects, 0, node_count * sizeof(*effects));
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_resolve_memory_effects(state, effects, node_count));

  const iree_host_size_t word_count =
      loom_low_schedule_bitset_word_count(node_count);
  const iree_host_size_t block_count = state->body->block_count;
  uint64_t* block_write_bits = NULL;
  uint64_t* incoming_bits = NULL;
  uint64_t* outgoing_bits = NULL;
  uint64_t* next_incoming_bits = NULL;
  uint64_t* active_bits = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, block_count * word_count, sizeof(*block_write_bits),
      (void**)&block_write_bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, block_count * word_count, sizeof(*incoming_bits),
      (void**)&incoming_bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, block_count * word_count, sizeof(*outgoing_bits),
      (void**)&outgoing_bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, word_count,
                                                 sizeof(*next_incoming_bits),
                                                 (void**)&next_incoming_bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, word_count, sizeof(*active_bits), (void**)&active_bits));
  memset(block_write_bits, 0,
         block_count * word_count * sizeof(*block_write_bits));
  memset(incoming_bits, 0, block_count * word_count * sizeof(*incoming_bits));
  memset(outgoing_bits, 0, block_count * word_count * sizeof(*outgoing_bits));

  for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
    uint64_t* block_writes = loom_low_schedule_block_bitset(
        block_write_bits, word_count, block_index);
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      if (effects[node_index].writes || effects[node_index].ordered) {
        loom_low_schedule_bitset_set(block_writes, node_index);
      }
    }
    uint64_t* block_outgoing =
        loom_low_schedule_block_bitset(outgoing_bits, word_count, block_index);
    memcpy(block_outgoing, block_writes, word_count * sizeof(*block_outgoing));
  }

  loom_cfg_graph_t graph = {0};
  IREE_RETURN_IF_ERROR(
      loom_cfg_graph_build(state->module, state->body, state->arena, &graph));
  if (graph.malformed) {
    return iree_ok_status();
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
      memset(next_incoming_bits, 0, word_count * sizeof(*next_incoming_bits));
      loom_cfg_block_index_span_t predecessors =
          loom_cfg_graph_predecessors(&graph, block_index);
      for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
        const uint16_t predecessor_index = predecessors.values[i];
        if (!loom_cfg_graph_block_is_reachable(&graph, predecessor_index)) {
          continue;
        }
        const uint64_t* predecessor_outgoing = loom_low_schedule_block_bitset(
            outgoing_bits, word_count, predecessor_index);
        loom_low_schedule_bitset_union_changed(
            next_incoming_bits, predecessor_outgoing, word_count);
      }

      uint64_t* block_incoming = loom_low_schedule_block_bitset(
          incoming_bits, word_count, block_index);
      if (loom_low_schedule_bitset_copy_changed(
              block_incoming, next_incoming_bits, word_count)) {
        changed = true;
      }

      uint64_t* block_outgoing = loom_low_schedule_block_bitset(
          outgoing_bits, word_count, block_index);
      memcpy(next_incoming_bits, block_incoming,
             word_count * sizeof(*next_incoming_bits));
      const uint64_t* block_writes = loom_low_schedule_block_bitset(
          block_write_bits, word_count, block_index);
      loom_low_schedule_bitset_union_changed(next_incoming_bits, block_writes,
                                             word_count);
      if (loom_low_schedule_bitset_copy_changed(
              block_outgoing, next_incoming_bits, word_count)) {
        changed = true;
      }
    }
  }

  for (uint16_t block_index = 0; block_index < block_count; ++block_index) {
    if (!loom_cfg_graph_block_is_reachable(&graph, block_index)) {
      continue;
    }
    const uint64_t* block_incoming =
        loom_low_schedule_block_bitset(incoming_bits, word_count, block_index);
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_add_visibility_dependencies_for_block(
            state, effects, node_count, block_incoming, active_bits,
            &state->blocks[block_index], word_count));
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_build_dependencies(
    loom_low_schedule_build_state_t* state) {
  uint32_t node_count = 0;
  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    node_count += block_record->node_count;
    loom_low_schedule_effect_frontier_t effect_frontier;
    loom_low_schedule_effect_frontier_initialize(state, &effect_frontier);
    uint32_t last_live_in_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    bool live_in_preamble_open = true;
    if (state->state_chain_read_heads != NULL) {
      memset(&state->state_chain_read_heads[block_record->node_start], 0xFF,
             block_record->node_count * sizeof(*state->state_chain_read_heads));
      state->state_chain_read_record_count = 0;
    }
    loom_low_schedule_reset_storage_reads(state);
    if (state->target.descriptor_set->reg_class_count != 0) {
      memset(state->state_last_write_nodes, 0xFF,
             state->target.descriptor_set->reg_class_count *
                 sizeof(*state->state_last_write_nodes));
      memset(state->state_read_heads, 0xFF,
             state->target.descriptor_set->reg_class_count *
                 sizeof(*state->state_read_heads));
      state->state_read_record_count = 0;
    }
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      const loom_op_t* op = node->op;
      if (loom_low_live_in_isa(op)) {
        if (block_index != 0 || !live_in_preamble_open) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "low schedule requires low.live_in packets in the entry "
              "preamble");
        }
        if (last_live_in_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_live_in_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR, UINT32_MAX));
        }
        last_live_in_node = node_index;
      } else {
        live_in_preamble_open = false;
        if (last_live_in_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, last_live_in_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_ANCHOR, UINT32_MAX));
        }
      }

      // Process tied writes before recording this node's operand reads. This
      // keeps overlapping tied operands from depending on their own writer,
      // while retained disjoint reads transfer to the tied result value.
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_tied_storage_writes(state, node_index));

      const loom_low_descriptor_t* descriptor = node->descriptor;
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const loom_value_ordinal_t operand_ordinal =
            operand_ordinals[operand_index];
        const uint32_t producer_node =
            state->values[operand_ordinal].producer_node;
        if (producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
            state->nodes[producer_node].block == node->block) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, producer_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, operand_index));
          IREE_RETURN_IF_ERROR(
              loom_low_schedule_add_state_chain_read_dependencies(
                  state, producer_node, node_index));
        }
        bool reads_descriptor_state = false;
        if (descriptor != NULL) {
          const loom_low_operand_t* operand = NULL;
          IREE_RETURN_IF_ERROR(
              loom_low_schedule_lookup_descriptor_packet_operand(
                  state->target.descriptor_set, descriptor, operand_index, NULL,
                  &operand));
          reads_descriptor_state = iree_any_bit_set(
              operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_READ);
        }
        if (descriptor == NULL || reads_descriptor_state) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_value_read(
              state, node_index, operand_ordinal));
        }
      }

      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_storage_reads(state, node_index, descriptor));
      IREE_RETURN_IF_ERROR(loom_low_schedule_note_descriptor_state_accesses(
          state, node_index, descriptor));
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_structural_state_reads(state, node_index));
      if (descriptor != NULL) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_descriptor_effects(
            state, &effect_frontier, node_index, descriptor));
      } else if (loom_low_schedule_node_has_effects(node, NULL)) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_structural_effects(
            state, &effect_frontier, node_index));
      }

      if (node->kind == LOOM_LOW_SCHEDULE_NODE_TERMINATOR) {
        for (uint32_t predecessor_node = block_record->node_start;
             predecessor_node < node_index; ++predecessor_node) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, predecessor_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_CONTROL, UINT32_MAX));
        }
      }
    }
    loom_low_schedule_reset_storage_reads(state);
  }
  return loom_low_schedule_build_visibility_dependencies(state, node_count);
}
