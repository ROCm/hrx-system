// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/graph.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_low_schedule_dependency_timing_t {
  // Signed minimum consumer issue cycle relative to producer issue.
  int32_t minimum_issue_separation_cycles;
  // Target model quality for the selected timing rule.
  loom_low_model_quality_t model_quality;
  // Origin of the selected timing rule.
  loom_low_schedule_separation_source_t separation_source;
} loom_low_schedule_dependency_timing_t;

typedef struct loom_low_schedule_effect_frontier_t {
  // Latest ordered effect node that every later dependency effect must follow.
  uint32_t ordered_node;
  // Descriptor attachment for ordered_node, or an empty endpoint.
  loom_low_schedule_dependency_endpoint_t ordered_endpoint;
  // Outstanding reads not yet subsumed by a later write or ordered effect.
  loom_low_schedule_effect_frontier_entry_t* reads;
  // Number of outstanding read entries.
  iree_host_size_t read_count;
  // Outstanding writes not yet subsumed by a later write or ordered effect.
  loom_low_schedule_effect_frontier_entry_t* writes;
  // Number of outstanding write entries.
  iree_host_size_t write_count;
} loom_low_schedule_effect_frontier_t;

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_dependency_endpoint_none(void) {
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = LOOM_LOW_ID_NONE,
      .timing_event_id = LOOM_LOW_TIMING_EVENT_NONE,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_NONE,
  };
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_dependency_operand_endpoint(uint16_t operand_index,
                                              uint16_t timing_event_id) {
  if (operand_index == LOOM_LOW_ID_NONE) {
    return loom_low_schedule_dependency_endpoint_none();
  }
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = operand_index,
      .timing_event_id = timing_event_id,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_OPERAND,
  };
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_dependency_effect_endpoint(uint16_t effect_ordinal,
                                             uint16_t timing_event_id) {
  if (effect_ordinal == LOOM_LOW_ID_NONE) {
    return loom_low_schedule_dependency_endpoint_none();
  }
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = effect_ordinal,
      .timing_event_id = timing_event_id,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT,
  };
}

static loom_low_schedule_state_access_t loom_low_schedule_state_access_none(
    void) {
  return (loom_low_schedule_state_access_t){
      .node_index = LOOM_LOW_SCHEDULE_NODE_NONE,
      .endpoint = loom_low_schedule_dependency_endpoint_none(),
  };
}

static loom_low_schedule_state_access_t loom_low_schedule_state_access(
    uint32_t node_index, loom_low_schedule_dependency_endpoint_t endpoint) {
  return (loom_low_schedule_state_access_t){
      .node_index = node_index,
      .endpoint = endpoint,
  };
}

static void loom_low_schedule_reset_state_accesses(
    loom_low_schedule_state_access_t* accesses, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    accesses[i] = loom_low_schedule_state_access_none();
  }
}

static bool loom_low_schedule_op_is_descriptor_packet(const loom_op_t* op) {
  return loom_low_op_isa(op) || loom_low_const_isa(op);
}

// Structural low operations that may emit target register packets after
// scheduling must carry any target state dependencies their eventual packets
// require.
static bool loom_low_schedule_op_is_structural_materialization(
    const loom_op_t* op) {
  return loom_low_copy_isa(op) || loom_low_move_isa(op) ||
         loom_low_slice_isa(op) || loom_low_concat_isa(op) ||
         loom_low_storage_address_isa(op);
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

  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(state->target.descriptor_set, op,
                                        &packet);

  node->descriptor = packet.descriptor;
  node->source_descriptor = packet.descriptor;
  if (iree_any_bit_set(packet.descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_EARLY_CLOBBER)) {
    node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_EARLY_CLOBBER;
  }
  if (iree_any_bit_set(packet.descriptor->flags,
                       LOOM_LOW_DESCRIPTOR_FLAG_BARRIER)) {
    node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
  }
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view_at(state->target.descriptor_set,
                                                 packet.descriptor_ordinal);
  node->schedule_class =
      &state->target.descriptor_set
           ->schedule_classes[descriptor_view->schedule_class_id];
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
    uint16_t value_operand_index,
    loom_low_schedule_dependency_endpoint_t producer_endpoint,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint) {
  return dependency->producer_node == producer_node &&
         dependency->consumer_node == consumer_node &&
         dependency->kind == kind &&
         dependency->value_operand_index == value_operand_index &&
         dependency->producer_attachment_index ==
             producer_endpoint.attachment_index &&
         dependency->consumer_attachment_index ==
             consumer_endpoint.attachment_index &&
         dependency->producer_attachment_kind ==
             producer_endpoint.attachment_kind &&
         dependency->consumer_attachment_kind ==
             consumer_endpoint.attachment_kind;
}

static loom_low_schedule_dependency_timing_t
loom_low_schedule_resolve_dependency_timing(
    const loom_low_schedule_build_state_t* state, uint32_t producer_node,
    loom_low_schedule_dependency_endpoint_t producer_endpoint,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint) {
  const loom_low_event_separation_t* event_separation =
      loom_low_descriptor_set_lookup_event_separation(
          state->target.descriptor_set, producer_endpoint.timing_event_id,
          consumer_endpoint.timing_event_id);
  if (event_separation != NULL) {
    return (loom_low_schedule_dependency_timing_t){
        .minimum_issue_separation_cycles =
            event_separation->minimum_issue_separation_cycles,
        .model_quality = event_separation->model_quality,
        .separation_source = LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR,
    };
  }
  const loom_low_schedule_class_t* schedule_class =
      state->nodes[producer_node].schedule_class;
  if (schedule_class != NULL) {
    return (loom_low_schedule_dependency_timing_t){
        .minimum_issue_separation_cycles =
            schedule_class->minimum_issue_separation_cycles,
        .model_quality = schedule_class->model_quality,
        .separation_source = LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS,
    };
  }
  return (loom_low_schedule_dependency_timing_t){
      .minimum_issue_separation_cycles = 0,
      .model_quality = LOOM_LOW_MODEL_QUALITY_UNKNOWN,
      .separation_source = LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_STRUCTURAL,
  };
}

static iree_status_t loom_low_schedule_append_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint16_t value_operand_index,
    loom_low_schedule_dependency_endpoint_t producer_endpoint,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint) {
  if (state->dependencies.count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t index capacity");
  }
  const loom_low_schedule_dependency_timing_t timing =
      loom_low_schedule_resolve_dependency_timing(
          state, producer_node, producer_endpoint, consumer_endpoint);
  return loom_low_schedule_dependency_graph_append(
      &state->dependencies,
      (loom_low_schedule_dependency_t){
          .producer_node = producer_node,
          .consumer_node = consumer_node,
          .minimum_issue_separation_cycles =
              timing.minimum_issue_separation_cycles,
          .producer_attachment_index = producer_endpoint.attachment_index,
          .consumer_attachment_index = consumer_endpoint.attachment_index,
          .producer_event_id = producer_endpoint.timing_event_id,
          .consumer_event_id = consumer_endpoint.timing_event_id,
          .value_operand_index = value_operand_index,
          .producer_attachment_kind = producer_endpoint.attachment_kind,
          .consumer_attachment_kind = consumer_endpoint.attachment_kind,
          .kind = kind,
          .separation_source = timing.separation_source,
          .model_quality = (uint8_t)timing.model_quality,
      },
      state->arena);
}

static iree_status_t loom_low_schedule_add_dependency(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node, loom_low_schedule_dependency_kind_t kind,
    uint16_t value_operand_index,
    loom_low_schedule_dependency_endpoint_t producer_endpoint,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint) {
  if (producer_node == consumer_node) {
    return iree_ok_status();
  }
  const iree_host_size_t dependency_count = state->dependencies.count;
  if (dependency_count != 0 &&
      loom_low_schedule_dependency_equal(
          loom_low_schedule_dependency_graph_at(
              &state->dependencies, (uint32_t)(dependency_count - 1)),
          producer_node, consumer_node, kind, value_operand_index,
          producer_endpoint, consumer_endpoint)) {
    return iree_ok_status();
  }
  return loom_low_schedule_append_dependency(
      state, producer_node, consumer_node, kind, value_operand_index,
      producer_endpoint, consumer_endpoint);
}

static iree_status_t loom_low_schedule_add_state_dependency(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_state_access_t producer,
    loom_low_schedule_state_access_t consumer, uint16_t value_operand_index) {
  return loom_low_schedule_add_dependency(
      state, producer.node_index, consumer.node_index,
      LOOM_LOW_SCHEDULE_DEPENDENCY_STATE, value_operand_index,
      producer.endpoint, consumer.endpoint);
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

static const uint16_t* loom_low_schedule_index_descriptor_operands(
    loom_low_schedule_build_state_t* state,
    const loom_low_descriptor_t* descriptor, uint16_t operand_count) {
  if (descriptor == NULL) return NULL;
  IREE_ASSERT_LE(operand_count, state->descriptor_operands.capacity);
  const bool has_variadic_operands =
      loom_low_descriptor_has_variadic_operands(descriptor);
  const uint16_t fixed_descriptor_operand_end =
      descriptor->operand_count - (has_variadic_operands ? 1 : 0);
  uint16_t indexed_operand_count = 0;
  for (uint16_t descriptor_operand_index = descriptor->result_count;
       descriptor_operand_index < fixed_descriptor_operand_end;
       ++descriptor_operand_index) {
    const loom_low_operand_t* operand =
        &state->target.descriptor_set
             ->operands[descriptor->operand_start + descriptor_operand_index];
    if (!loom_low_operand_role_is_packet_operand(operand->role)) continue;
    IREE_ASSERT_LT(operand->source_value_index, operand_count);
    state->descriptor_operands.indices[operand->source_value_index] =
        descriptor_operand_index;
    ++indexed_operand_count;
  }
  if (has_variadic_operands) {
    IREE_ASSERT_LE(descriptor->minimum_packet_operand_count, operand_count);
    for (uint16_t packet_operand_index =
             descriptor->minimum_packet_operand_count;
         packet_operand_index < operand_count; ++packet_operand_index) {
      state->descriptor_operands.indices[packet_operand_index] =
          fixed_descriptor_operand_end;
      ++indexed_operand_count;
    }
  }
  IREE_ASSERT_EQ(indexed_operand_count, operand_count);
  return state->descriptor_operands.indices;
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_value_write_endpoint(
    const loom_low_schedule_build_state_t* state,
    loom_value_ordinal_t value_ordinal) {
  const uint32_t producer_node = state->values[value_ordinal].producer_node;
  if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return loom_low_schedule_dependency_endpoint_none();
  }
  const loom_low_schedule_node_t* node = &state->nodes[producer_node];
  if (node->descriptor == NULL) {
    return loom_low_schedule_dependency_endpoint_none();
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    if (result_ordinals[result_index] != value_ordinal) {
      continue;
    }
    if (result_index >= node->descriptor->result_count) {
      return loom_low_schedule_dependency_endpoint_none();
    }
    const loom_low_operand_t* result =
        &state->target.descriptor_set
             ->operands[node->descriptor->operand_start + result_index];
    return loom_low_schedule_dependency_operand_endpoint(
        result_index, result->write_event_id);
  }
  IREE_ASSERT(false, "producer node must list the value as a result");
  return loom_low_schedule_dependency_endpoint_none();
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_descriptor_operand_read_endpoint(
    const loom_low_schedule_build_state_t* state,
    const loom_low_descriptor_t* descriptor,
    uint16_t descriptor_operand_index) {
  if (descriptor == NULL) {
    return loom_low_schedule_dependency_endpoint_none();
  }
  const loom_low_operand_t* operand =
      &state->target.descriptor_set
           ->operands[descriptor->operand_start + descriptor_operand_index];
  return loom_low_schedule_dependency_operand_endpoint(descriptor_operand_index,
                                                       operand->read_event_id);
}

static iree_status_t loom_low_schedule_add_state_read(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_state_access_t access, uint16_t reg_class_id) {
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
          .access = access,
          .next_record = state->state_read_heads[reg_class_id],
      };
  state->state_read_heads[reg_class_id] =
      (uint32_t)state->state_read_record_count++;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_chain_read(
    loom_low_schedule_build_state_t* state, uint32_t producer_node_index,
    loom_low_schedule_state_access_t reader) {
  if (producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      producer_node_index == reader.node_index ||
      state->state_chain_read_heads == NULL) {
    return iree_ok_status();
  }
  if (state->nodes[producer_node_index].block !=
      state->nodes[reader.node_index].block) {
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
          .access = reader,
          .next_record = state->state_chain_read_heads[producer_node_index],
      };
  state->state_chain_read_heads[producer_node_index] =
      (uint32_t)state->state_chain_read_record_count++;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_chain_read_dependencies(
    loom_low_schedule_build_state_t* state, uint32_t producer_node_index,
    loom_low_schedule_state_access_t consumer) {
  if (producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE ||
      producer_node_index == consumer.node_index ||
      state->state_chain_read_heads == NULL) {
    return iree_ok_status();
  }
  if (state->nodes[producer_node_index].block !=
      state->nodes[consumer.node_index].block) {
    return iree_ok_status();
  }
  uint32_t read_record_index =
      state->state_chain_read_heads[producer_node_index];
  while (read_record_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    const loom_low_schedule_state_chain_read_record_t* read_record =
        &state->state_chain_read_records[read_record_index];
    if (read_record->access.node_index != consumer.node_index &&
        state->nodes[read_record->access.node_index].source_ordinal <
            state->nodes[consumer.node_index].source_ordinal) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
          state, read_record->access, consumer, LOOM_LOW_ID_NONE));
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

static bool loom_low_schedule_unit_ranges_overlap(uint32_t lhs_offset,
                                                  uint32_t lhs_count,
                                                  uint32_t rhs_offset,
                                                  uint32_t rhs_count) {
  const uint64_t lhs_end = (uint64_t)lhs_offset + lhs_count;
  const uint64_t rhs_end = (uint64_t)rhs_offset + rhs_count;
  return lhs_offset < rhs_end && rhs_offset < lhs_end;
}

static loom_low_register_part_mask_t
loom_low_schedule_descriptor_operand_storage_mask(
    const loom_low_schedule_build_state_t* state,
    const loom_low_operand_t* operand, loom_value_ordinal_t value_ordinal) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
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
    loom_value_ordinal_t value_ordinal, uint32_t unit_offset,
    uint32_t unit_count, loom_low_register_part_mask_t read_mask,
    uint16_t descriptor_operand_index, uint16_t timing_event_id) {
  if (state->storage_reads.heads == NULL ||
      !iree_any_bit_set(state->values[value_ordinal].flags,
                        LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED)) {
    return iree_ok_status();
  }
  IREE_ASSERT(
      unit_offset <= state->values[value_ordinal].unit_count &&
          unit_count <= state->values[value_ordinal].unit_count - unit_offset,
      "verified low storage read range must fit value units");
  if (unit_count == 0) {
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
          .unit_offset = unit_offset,
          .unit_count = unit_count,
          .read_mask = read_mask,
          .descriptor_operand_index = descriptor_operand_index,
          .timing_event_id = timing_event_id,
          .next_record = state->storage_reads.heads[value_ordinal],
      };
  state->storage_reads.heads[value_ordinal] =
      (uint32_t)state->storage_reads.record_count++;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_storage_write_dependencies(
    loom_low_schedule_build_state_t* state, uint32_t writer_node_index,
    uint16_t value_operand_index,
    loom_low_schedule_dependency_endpoint_t writer_endpoint,
    loom_value_ordinal_t value_ordinal,
    loom_value_ordinal_t result_value_ordinal, uint32_t write_unit_offset,
    uint32_t write_unit_count, loom_low_register_part_mask_t write_mask) {
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
    const loom_low_schedule_node_t* reader =
        &state->nodes[read_record->reader_node];
    const loom_low_schedule_node_t* writer = &state->nodes[writer_node_index];
    if (read_record->reader_node != writer_node_index &&
        reader->source_ordinal < writer->source_ordinal &&
        loom_low_schedule_unit_ranges_overlap(
            read_record->unit_offset, read_record->unit_count,
            write_unit_offset, write_unit_count) &&
        iree_any_bit_set(read_record->read_mask, write_mask)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
          state, read_record->reader_node, writer_node_index,
          LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, value_operand_index,
          loom_low_schedule_dependency_operand_endpoint(
              read_record->descriptor_operand_index,
              read_record->timing_event_id),
          writer_endpoint));
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

static iree_status_t loom_low_schedule_add_storage_antidependencies(
    loom_low_schedule_build_state_t* state, uint32_t writer_node_index,
    loom_low_schedule_dependency_endpoint_t writer_endpoint,
    loom_value_ordinal_t value_ordinal, uint32_t write_unit_offset,
    uint32_t write_unit_count, loom_low_register_part_mask_t write_mask) {
  if (state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT_NE(write_mask, 0u);
  uint32_t read_record_index = state->storage_reads.heads[value_ordinal];
  while (read_record_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    const loom_low_schedule_storage_read_record_t* read_record =
        &state->storage_reads.records[read_record_index];
    const loom_low_schedule_node_t* reader =
        &state->nodes[read_record->reader_node];
    const loom_low_schedule_node_t* writer = &state->nodes[writer_node_index];
    if (read_record->reader_node != writer_node_index &&
        reader->source_ordinal < writer->source_ordinal &&
        loom_low_schedule_unit_ranges_overlap(
            read_record->unit_offset, read_record->unit_count,
            write_unit_offset, write_unit_count) &&
        iree_any_bit_set(read_record->read_mask, write_mask)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
          state, read_record->reader_node, writer_node_index,
          LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE, LOOM_LOW_ID_NONE,
          loom_low_schedule_dependency_operand_endpoint(
              read_record->descriptor_operand_index,
              read_record->timing_event_id),
          writer_endpoint));
    }
    read_record_index = read_record->next_record;
  }
  return iree_ok_status();
}

static bool loom_low_schedule_relation_is_edge_handoff(
    const loom_low_schedule_storage_relation_t* relation) {
  return relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH ||
         relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD;
}

static bool loom_low_schedule_relation_is_structural_alias(
    const loom_low_schedule_storage_relation_t* relation) {
  return relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY ||
         relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_MOVE ||
         relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SLICE ||
         relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_CONCAT;
}

static iree_status_t loom_low_schedule_push_edge_source(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_edge_source_record_t record,
    iree_host_size_t* inout_worklist_count) {
  IREE_ASSERT(
      record.value_unit_offset <=
              state->values[record.value_ordinal].unit_count &&
          record.unit_count <= state->values[record.value_ordinal].unit_count -
                                   record.value_unit_offset,
      "verified low edge source range must fit value units");
  if (record.unit_count == 0) {
    return iree_ok_status();
  }
  if (*inout_worklist_count >=
      state->storage_reads.edge_source_worklist_capacity) {
    const iree_host_size_t minimum_capacity = *inout_worklist_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, *inout_worklist_count, minimum_capacity, sizeof(record),
        &state->storage_reads.edge_source_worklist_capacity,
        (void**)&state->storage_reads.edge_source_worklist));
  }
  state->storage_reads.edge_source_worklist[(*inout_worklist_count)++] = record;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_edge_source_writes(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* edge_node,
    const loom_low_schedule_storage_relation_t* edge_relation) {
  const loom_value_ordinal_t destination_ordinal =
      edge_relation->destination_ordinal;
  const loom_value_ordinal_t source_ordinal = edge_relation->source_ordinal;
  iree_host_size_t worklist_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_schedule_push_edge_source(
      state,
      (loom_low_schedule_edge_source_record_t){
          .value_ordinal = source_ordinal,
          .value_unit_offset = edge_relation->source_unit_offset,
          .destination_unit_offset = edge_relation->destination_unit_offset,
          .unit_count = edge_relation->unit_count,
      },
      &worklist_count));
  for (iree_host_size_t i = 0; i < worklist_count; ++i) {
    const loom_low_schedule_edge_source_record_t current =
        state->storage_reads.edge_source_worklist[i];
    const loom_value_ordinal_t current_ordinal = current.value_ordinal;
    const uint32_t writer_node = state->values[current_ordinal].producer_node;
    if (writer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        state->nodes[writer_node].block != edge_node->block) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_antidependencies(
        state, writer_node,
        loom_low_schedule_value_write_endpoint(state, current_ordinal),
        destination_ordinal, current.destination_unit_offset,
        current.unit_count,
        loom_low_schedule_value_full_storage_mask(state, destination_ordinal)));

    const uint32_t relation_begin =
        loom_low_schedule_storage_relation_index_begin(
            &state->storage_relations, writer_node);
    const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
        &state->storage_relations, writer_node);
    for (uint32_t relation_index = relation_begin;
         relation_index < relation_end; ++relation_index) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      relation_index);
      if (relation->destination_ordinal != current_ordinal ||
          !loom_low_schedule_relation_is_structural_alias(relation)) {
        continue;
      }
      const uint32_t intersection_offset =
          current.value_unit_offset > relation->destination_unit_offset
              ? current.value_unit_offset
              : relation->destination_unit_offset;
      const uint64_t current_end =
          (uint64_t)current.value_unit_offset + current.unit_count;
      const uint64_t relation_unit_end =
          (uint64_t)relation->destination_unit_offset + relation->unit_count;
      const uint64_t intersection_end =
          current_end < relation_unit_end ? current_end : relation_unit_end;
      if ((uint64_t)intersection_offset >= intersection_end) {
        continue;
      }
      const loom_value_ordinal_t alias_source_ordinal =
          relation->source_ordinal;
      const uint32_t intersection_count =
          (uint32_t)(intersection_end - intersection_offset);
      IREE_RETURN_IF_ERROR(loom_low_schedule_push_edge_source(
          state,
          (loom_low_schedule_edge_source_record_t){
              .value_ordinal = alias_source_ordinal,
              .value_unit_offset =
                  relation->source_unit_offset +
                  (intersection_offset - relation->destination_unit_offset),
              .destination_unit_offset =
                  current.destination_unit_offset +
                  (intersection_offset - current.value_unit_offset),
              .unit_count = intersection_count,
          },
          &worklist_count));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_edge_storage_writes(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  if (state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     node_index);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, node_index);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    if (!loom_low_schedule_relation_is_edge_handoff(relation)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_note_edge_source_writes(state, node, relation));
  }
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
    IREE_ASSERT(tied.result_index < node->result_count &&
                    tied.operand_index < node->operand_count,
                "verified tied result metadata must reference node values");
    loom_low_register_part_mask_t write_mask =
        loom_low_schedule_value_full_storage_mask(
            state, result_ordinals[tied.result_index]);
    if (node->descriptor != NULL &&
        tied.result_index < node->descriptor->result_count) {
      write_mask = loom_low_schedule_descriptor_operand_storage_mask(
          state,
          &state->target.descriptor_set
               ->operands[node->descriptor->operand_start + tied.result_index],
          result_ordinals[tied.result_index]);
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_write_dependencies(
        state, node_index, tied.operand_index,
        loom_low_schedule_value_write_endpoint(
            state, result_ordinals[tied.result_index]),
        operand_ordinals[tied.operand_index],
        result_ordinals[tied.result_index], /*write_unit_offset=*/0,
        state->values[result_ordinals[tied.result_index]].unit_count,
        write_mask));
  }
  return iree_ok_status();
}

static loom_low_register_part_mask_t
loom_low_schedule_operand_storage_read_mask(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_operand_t* descriptor_operand, uint16_t operand_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  IREE_ASSERT_LT(operand_index, node->operand_count);
  const loom_value_ordinal_t value_ordinal =
      loom_low_schedule_node_const_operand_ordinals(node)[operand_index];
  if (descriptor_operand == NULL) {
    return loom_low_schedule_value_full_storage_mask(state, value_ordinal);
  }
  return loom_low_schedule_descriptor_operand_storage_mask(
      state, descriptor_operand, value_ordinal);
}

static iree_status_t loom_low_schedule_note_storage_reads(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor,
    const uint16_t* descriptor_operand_indices) {
  if (state->storage_reads.heads == NULL) {
    return iree_ok_status();
  }
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  IREE_ASSERT_LE(node->operand_count,
                 state->storage_reads.operand_relation_flag_capacity);
  if (node->operand_count != 0) {
    memset(state->storage_reads.operand_relation_flags, 0,
           node->operand_count *
               sizeof(*state->storage_reads.operand_relation_flags));
  }

  const uint32_t relation_begin =
      loom_low_schedule_storage_relation_index_begin(&state->storage_relations,
                                                     node_index);
  const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, node_index);
  for (uint32_t relation_index = relation_begin; relation_index < relation_end;
       ++relation_index) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    relation_index);
    const uint16_t operand_index = relation->source_operand_index;
    IREE_ASSERT_LT(operand_index, node->operand_count);
    const loom_value_ordinal_t value_ordinal = operand_ordinals[operand_index];
    IREE_ASSERT_EQ(value_ordinal, relation->source_ordinal);
    const loom_low_operand_t* descriptor_operand = NULL;
    if (descriptor != NULL) {
      descriptor_operand =
          &state->target.descriptor_set
               ->operands[descriptor->operand_start +
                          descriptor_operand_indices[operand_index]];
    }
    const loom_low_register_part_mask_t read_mask =
        loom_low_schedule_operand_storage_read_mask(
            state, node_index, descriptor_operand, operand_index);
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_read(
        state, node_index, value_ordinal, relation->source_unit_offset,
        relation->unit_count, read_mask,
        descriptor != NULL ? descriptor_operand_indices[operand_index]
                           : LOOM_LOW_ID_NONE,
        descriptor_operand != NULL ? descriptor_operand->read_event_id
                                   : LOOM_LOW_TIMING_EVENT_NONE));
    state->storage_reads.operand_relation_flags[operand_index] = 1;
  }

  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    if (state->storage_reads.operand_relation_flags[operand_index]) {
      continue;
    }
    const loom_value_ordinal_t value_ordinal = operand_ordinals[operand_index];
    const loom_low_operand_t* descriptor_operand = NULL;
    if (descriptor != NULL) {
      descriptor_operand =
          &state->target.descriptor_set
               ->operands[descriptor->operand_start +
                          descriptor_operand_indices[operand_index]];
    }
    const loom_low_register_part_mask_t read_mask =
        loom_low_schedule_operand_storage_read_mask(
            state, node_index, descriptor_operand, operand_index);
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_storage_read(
        state, node_index, value_ordinal, /*unit_offset=*/0,
        state->values[value_ordinal].unit_count, read_mask,
        descriptor != NULL ? descriptor_operand_indices[operand_index]
                           : LOOM_LOW_ID_NONE,
        descriptor_operand != NULL ? descriptor_operand->read_event_id
                                   : LOOM_LOW_TIMING_EVENT_NONE));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_state_read_dependencies(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_state_access_t writer, uint16_t reg_class_id) {
  uint32_t read_record_index = state->state_read_heads[reg_class_id];
  while (read_record_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    const loom_low_schedule_state_read_record_t* read_record =
        &state->state_read_records[read_record_index];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, read_record->access, writer, LOOM_LOW_ID_NONE));
    read_record_index = read_record->next_record;
  }
  state->state_read_heads[reg_class_id] = LOOM_LOW_SCHEDULE_NODE_NONE;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_state_read(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_state_access_t reader, uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const loom_low_schedule_state_access_t last_write =
      state->state_last_writes[reg_class_id];
  if (last_write.node_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, last_write, reader, LOOM_LOW_ID_NONE));
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_chain_read(
        state, last_write.node_index, reader));
  }
  const loom_low_schedule_state_access_t ordering_frontier =
      state->state_ordering_frontiers[reg_class_id];
  if (ordering_frontier.node_index != LOOM_LOW_SCHEDULE_NODE_NONE &&
      ordering_frontier.node_index != last_write.node_index) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, ordering_frontier, reader, LOOM_LOW_ID_NONE));
  }
  return loom_low_schedule_add_state_read(state, reader, reg_class_id);
}

static iree_status_t loom_low_schedule_note_state_write(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_state_access_t writer, uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const loom_low_schedule_state_access_t last_write =
      state->state_last_writes[reg_class_id];
  if (last_write.node_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, last_write, writer, LOOM_LOW_ID_NONE));
  }
  const loom_low_schedule_state_access_t ordering_frontier =
      state->state_ordering_frontiers[reg_class_id];
  if (ordering_frontier.node_index != LOOM_LOW_SCHEDULE_NODE_NONE &&
      ordering_frontier.node_index != last_write.node_index) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, ordering_frontier, writer, LOOM_LOW_ID_NONE));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_read_dependencies(
      state, writer, reg_class_id));
  state->state_last_writes[reg_class_id] = writer;
  state->state_ordering_frontiers[reg_class_id] =
      loom_low_schedule_state_access_none();
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_state_fence(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_state_access_t fence, uint16_t reg_class_id) {
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  const loom_low_schedule_state_access_t last_write =
      state->state_last_writes[reg_class_id];
  if (last_write.node_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, last_write, fence, LOOM_LOW_ID_NONE));
  }
  const loom_low_schedule_state_access_t ordering_frontier =
      state->state_ordering_frontiers[reg_class_id];
  if (ordering_frontier.node_index != LOOM_LOW_SCHEDULE_NODE_NONE &&
      ordering_frontier.node_index != last_write.node_index) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, ordering_frontier, fence, LOOM_LOW_ID_NONE));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_read_dependencies(
      state, fence, reg_class_id));
  state->state_ordering_frontiers[reg_class_id] = fence;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_explicit_state_value_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t operand_index, loom_value_ordinal_t value_ordinal,
    uint16_t reg_class_id,
    loom_low_schedule_dependency_endpoint_t read_endpoint) {
  const loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  const uint32_t producer_node = value->producer_node;
  const bool has_same_block_producer =
      producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
      state->nodes[producer_node].block == state->nodes[node_index].block;
  const loom_low_schedule_state_access_t reader =
      loom_low_schedule_state_access(node_index, read_endpoint);
  const loom_low_schedule_state_access_t first_clobber =
      has_same_block_producer ? value->state_next_write
                              : state->state_first_writes[reg_class_id];
  if (first_clobber.node_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, reader, first_clobber, operand_index));
  }
  const loom_low_schedule_state_access_t ordering_frontier =
      state->state_ordering_frontiers[reg_class_id];
  if (ordering_frontier.node_index != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_state_dependency(
        state, ordering_frontier, reader, LOOM_LOW_ID_NONE));
  }
  if (has_same_block_producer) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_add_state_chain_read(state, producer_node, reader));
  }
  return loom_low_schedule_add_state_read(state, reader, reg_class_id);
}

static iree_status_t loom_low_schedule_note_state_value_read(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t operand_index, loom_value_ordinal_t value_ordinal,
    loom_low_schedule_dependency_endpoint_t read_endpoint) {
  const uint16_t reg_class_id = state->values[value_ordinal].register_class_id;
  if (!loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
    return iree_ok_status();
  }
  return loom_low_schedule_note_explicit_state_value_read(
      state, node_index, operand_index, value_ordinal, reg_class_id,
      read_endpoint);
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
      IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_read(
          state,
          loom_low_schedule_state_access(
              node_index, loom_low_schedule_dependency_operand_endpoint(
                              i, operand->read_event_id)),
          reg_class_id));
    }
    if (has_ordered_effect) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_fence(
          state,
          loom_low_schedule_state_access(
              node_index, loom_low_schedule_dependency_operand_endpoint(
                              i, operand->read_event_id)),
          reg_class_id));
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
    IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_write(
        state,
        loom_low_schedule_state_access(
            node_index, loom_low_schedule_dependency_operand_endpoint(
                            i, operand->write_event_id)),
        reg_class_id));
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
          state,
          loom_low_schedule_state_access(
              node_index, loom_low_schedule_dependency_endpoint_none()),
          row->state_reg_class_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_index_state_value_clobbers(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  if (descriptor_set->reg_class_count == 0) {
    return iree_ok_status();
  }
  loom_low_schedule_reset_state_accesses(state->state_first_writes,
                                         descriptor_set->reg_class_count);

  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t node_index = block_node_end;
       node_index-- > block_record->node_start;) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t result_index = 0; result_index < node->result_count;
         ++result_index) {
      loom_low_schedule_value_record_t* value =
          &state->values[result_ordinals[result_index]];
      const uint16_t reg_class_id = value->register_class_id;
      if (loom_low_schedule_reg_class_is_state(state, reg_class_id)) {
        value->state_next_write = state->state_first_writes[reg_class_id];
      }
    }

    const loom_low_descriptor_t* descriptor = node->descriptor;
    if (descriptor == NULL) {
      continue;
    }
    for (uint16_t operand_index = 0; operand_index < descriptor->operand_count;
         ++operand_index) {
      const loom_low_operand_t* operand =
          &descriptor_set->operands[descriptor->operand_start + operand_index];
      if (!iree_any_bit_set(operand->flags,
                            LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
        continue;
      }
      uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
      IREE_RETURN_IF_ERROR(loom_low_schedule_descriptor_operand_reg_class_id(
          descriptor_set, descriptor, operand_index, &reg_class_id));
      state->state_first_writes[reg_class_id] = loom_low_schedule_state_access(
          node_index, loom_low_schedule_dependency_operand_endpoint(
                          operand_index, operand->write_event_id));
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
          .issue_cycle = LOOM_LOW_SCHEDULE_NODE_NONE,
          .issue_group_ordinal = LOOM_LOW_SCHEDULE_NODE_NONE,
          .kind = LOOM_LOW_SCHEDULE_NODE_STRUCTURAL,
          .traits = loom_op_effective_traits(state->module, op),
          .descriptor = NULL,
          .schedule_class = NULL,
          .memory_access_record_index =
              LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE,
      };
      if (loom_low_schedule_op_is_terminator(state->module, op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_TERMINATOR;
        node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
      } else if (loom_low_schedule_op_is_descriptor_packet(op)) {
        node->kind = LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR;
      } else if (op->region_count == 0 &&
                 iree_any_bit_set(node->traits, LOOM_TRAIT_STORAGE_RELATION)) {
        node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP;
        if (op->kind == LOOM_OP_LOW_SLICE || op->kind == LOOM_OP_LOW_CONCAT) {
          node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_TRANSPARENT;
        }
      }
      if (loom_low_live_in_isa(op) || loom_low_resource_isa(op)) {
        node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY |
                       LOOM_LOW_SCHEDULE_NODE_FLAG_ZERO_ISSUE_WIDTH;
      }
      if (loom_low_move_isa(op)) {
        node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
      }
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_HINT)) {
        node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY;
      }
      if (loom_low_return_isa(op)) {
        node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_PROGRAM_EXIT_MEMORY;
      }
      if ((loom_low_copy_isa(op) && loom_low_copy_detached(op)) ||
          (loom_low_move_isa(op) && loom_low_move_detached(op))) {
        ++state->detached_transfer_node_count;
      }
      if (loom_low_storage_reserve_isa(op)) {
        IREE_RETURN_IF_ERROR(loom_low_storage_layout_builder_append(
            state->module, op, state->arena, &state->storage_layout_builder));
      }

      const loom_low_descriptor_t* descriptor = NULL;
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_resolve_descriptor(state, op, node, &descriptor));
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_initialize_node_value_ordinals(state, node));
      node->storage_relation_count =
          loom_low_storage_relation_count(state->module, op);
      state->storage_relation_count += node->storage_relation_count;
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
  frontier->ordered_endpoint = loom_low_schedule_dependency_endpoint_none();
  frontier->read_count = 0;
  frontier->write_count = 0;
}

static void loom_low_schedule_effect_frontier_initialize(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* out_frontier) {
  *out_frontier = (loom_low_schedule_effect_frontier_t){
      .reads = state->effect_read_entries,
      .writes = state->effect_write_entries,
  };
  loom_low_schedule_effect_frontier_reset(out_frontier);
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_effect_frontier_entry_endpoint(
    const loom_low_schedule_effect_frontier_entry_t* entry) {
  return loom_low_schedule_dependency_effect_endpoint(entry->effect_ordinal,
                                                      entry->timing_event_id);
}

static iree_status_t loom_low_schedule_effect_frontier_depend_on_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint) {
  if (frontier->ordered_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return iree_ok_status();
  }
  return loom_low_schedule_add_dependency(
      state, frontier->ordered_node, node_index,
      LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
      frontier->ordered_endpoint, consumer_endpoint);
}

static iree_status_t loom_low_schedule_effect_frontier_note_read(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary,
    loom_low_schedule_dependency_endpoint_t endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, endpoint));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    if (!loom_low_memory_access_summaries_may_alias(summary, &write->summary)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write), endpoint));
  }
  IREE_ASSERT(frontier->read_count < state->effect_read_capacity,
              "precomputed effect-frontier read capacity must cover all rows");
  frontier->reads[frontier->read_count] =
      (loom_low_schedule_effect_frontier_entry_t){
          .node_index = node_index,
          .effect_ordinal = endpoint.attachment_index,
          .timing_event_id = endpoint.timing_event_id,
          .summary = *summary,
      };
  ++frontier->read_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_write_complete(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary,
    loom_low_schedule_dependency_endpoint_t endpoint) {
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < frontier->read_count;
       ++read_index) {
    const loom_low_schedule_effect_frontier_entry_t* read =
        &frontier->reads[read_index];
    if (loom_low_memory_access_write_subsumes_read(summary, &read->summary)) {
      continue;
    }
    frontier->reads[write_index] = *read;
    ++write_index;
  }
  frontier->read_count = write_index;
  write_index = 0;
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    if (loom_low_memory_access_write_subsumes_access(summary,
                                                     &write->summary)) {
      continue;
    }
    frontier->writes[write_index] = *write;
    ++write_index;
  }
  frontier->write_count = write_index;
  IREE_ASSERT(frontier->write_count < state->effect_write_capacity,
              "precomputed effect-frontier write capacity must cover all rows");
  frontier->writes[frontier->write_count] =
      (loom_low_schedule_effect_frontier_entry_t){
          .node_index = node_index,
          .effect_ordinal = endpoint.attachment_index,
          .timing_event_id = endpoint.timing_event_id,
          .summary = *summary,
      };
  ++frontier->write_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_write(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary,
    loom_low_schedule_dependency_endpoint_t endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, endpoint));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    if (!loom_low_memory_access_summaries_may_alias(summary, &write->summary)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write), endpoint));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* read = &frontier->reads[i];
    if (!loom_low_memory_access_summaries_may_alias(summary, &read->summary)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(read), endpoint));
  }
  return loom_low_schedule_effect_frontier_note_write_complete(
      state, frontier, node_index, summary, endpoint);
}

static iree_status_t loom_low_schedule_effect_frontier_note_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    loom_low_schedule_dependency_endpoint_t endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, endpoint));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write), endpoint));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* read = &frontier->reads[i];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(read), endpoint));
  }
  loom_low_schedule_effect_frontier_reset(frontier);
  frontier->ordered_node = node_index;
  frontier->ordered_endpoint = endpoint;
  return iree_ok_status();
}

static const loom_low_memory_access_summary_t*
loom_low_schedule_lookup_memory_access_summary(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor,
    const loom_low_effect_t* selected_effect) {
  const uint32_t record_index =
      state->nodes[node_index].memory_access_record_index;
  if (record_index == LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE) {
    return NULL;
  }
  if (record_index >= state->memory_access_record_count) {
    return NULL;
  }
  const loom_low_memory_access_record_t* record =
      &state->memory_access_records[record_index];
  const loom_low_memory_access_summary_t* summary = &record->summary;

  // A single dependency memory effect is unambiguous regardless of whether
  // its descriptor space is generic. Multi-effect descriptors may refine the
  // unique effect in the same concrete memory space as the source record.
  // Ambiguous same-space effects remain conservative because a record does not
  // yet identify an individual descriptor effect ordinal.
  uint16_t dependency_memory_effect_count = 0;
  uint16_t matching_memory_space_effect_count = 0;
  const loom_low_memory_space_t summary_space =
      loom_low_memory_access_normalize_space(summary->memory_space);
  const loom_low_memory_space_t selected_space =
      loom_low_memory_access_normalize_space(selected_effect->memory_space);
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY) ||
        (effect->kind != LOOM_LOW_EFFECT_KIND_READ &&
         effect->kind != LOOM_LOW_EFFECT_KIND_WRITE)) {
      continue;
    }
    ++dependency_memory_effect_count;
    if (summary_space != LOOM_LOW_MEMORY_SPACE_GENERIC &&
        loom_low_memory_access_normalize_space(effect->memory_space) ==
            summary_space) {
      ++matching_memory_space_effect_count;
    }
  }
  if (dependency_memory_effect_count == 1) {
    return summary;
  }
  return summary_space != LOOM_LOW_MEMORY_SPACE_GENERIC &&
                 selected_space == summary_space &&
                 matching_memory_space_effect_count == 1
             ? summary
             : NULL;
}

static iree_status_t loom_low_schedule_note_descriptor_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor->effect_count == 0) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_orders_memory(effect)) {
      return loom_low_schedule_effect_frontier_note_ordered(
          state, frontier, node_index,
          loom_low_schedule_dependency_effect_endpoint(
              i, effect->timing_event_id));
    }
  }
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    const loom_low_memory_access_summary_t* source_summary =
        loom_low_schedule_lookup_memory_access_summary(state, node_index,
                                                       descriptor, effect);
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        loom_low_memory_access_summary_t summary =
            loom_low_memory_access_summary_from_effect(effect);
        if (source_summary != NULL) {
          summary = *source_summary;
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_read(
            state, frontier, node_index, &summary,
            loom_low_schedule_dependency_effect_endpoint(
                i, effect->timing_event_id)));
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        loom_low_memory_access_summary_t summary =
            loom_low_memory_access_summary_from_effect(effect);
        if (source_summary != NULL) {
          summary = *source_summary;
        }
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_write(
            state, frontier, node_index, &summary,
            loom_low_schedule_dependency_effect_endpoint(
                i, effect->timing_event_id)));
        break;
      }
      default:
        if (loom_low_schedule_effect_orders_memory(effect)) {
          return loom_low_schedule_effect_frontier_note_ordered(
              state, frontier, node_index,
              loom_low_schedule_dependency_effect_endpoint(
                  i, effect->timing_event_id));
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
    return loom_low_schedule_effect_frontier_note_ordered(
        state, frontier, node_index,
        loom_low_schedule_dependency_endpoint_none());
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_write(
        state, frontier, node_index, &summary,
        loom_low_schedule_dependency_endpoint_none());
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_read(
        state, frontier, node_index, &summary,
        loom_low_schedule_dependency_endpoint_none());
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_build_dependencies(
    loom_low_schedule_build_state_t* state) {
  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    loom_low_schedule_effect_frontier_t effect_frontier;
    loom_low_schedule_effect_frontier_initialize(state, &effect_frontier);
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_index_state_value_clobbers(state, block_record));
    if (state->state_chain_read_heads != NULL) {
      memset(&state->state_chain_read_heads[block_record->node_start], 0xFF,
             block_record->node_count * sizeof(*state->state_chain_read_heads));
      state->state_chain_read_record_count = 0;
    }
    loom_low_schedule_reset_storage_reads(state);
    if (state->target.descriptor_set->reg_class_count != 0) {
      loom_low_schedule_reset_state_accesses(
          state->state_last_writes,
          state->target.descriptor_set->reg_class_count);
      loom_low_schedule_reset_state_accesses(
          state->state_ordering_frontiers,
          state->target.descriptor_set->reg_class_count);
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

      // Process tied writes before recording this node's operand reads. This
      // keeps overlapping tied operands from depending on their own writer,
      // while retained disjoint reads transfer to the tied result value.
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_tied_storage_writes(state, node_index));
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_edge_storage_writes(state, node_index));

      const loom_low_descriptor_t* descriptor = node->descriptor;
      const uint16_t* descriptor_operand_indices =
          loom_low_schedule_index_descriptor_operands(state, descriptor,
                                                      node->operand_count);
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
          const uint16_t descriptor_operand_index =
              descriptor != NULL ? descriptor_operand_indices[operand_index]
                                 : LOOM_LOW_ID_NONE;
          const loom_low_schedule_dependency_endpoint_t read_endpoint =
              loom_low_schedule_descriptor_operand_read_endpoint(
                  state, descriptor, descriptor_operand_index);
          IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
              state, producer_node, node_index,
              LOOM_LOW_SCHEDULE_DEPENDENCY_SSA, operand_index,
              loom_low_schedule_value_write_endpoint(state, operand_ordinal),
              read_endpoint));
          IREE_RETURN_IF_ERROR(
              loom_low_schedule_add_state_chain_read_dependencies(
                  state, producer_node,
                  loom_low_schedule_state_access(node_index, read_endpoint)));
        }
        bool reads_descriptor_state = false;
        if (descriptor != NULL) {
          const loom_low_operand_t* operand =
              &state->target.descriptor_set
                   ->operands[descriptor->operand_start +
                              descriptor_operand_indices[operand_index]];
          reads_descriptor_state = iree_any_bit_set(
              operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_READ);
        }
        if (descriptor == NULL || reads_descriptor_state) {
          IREE_RETURN_IF_ERROR(loom_low_schedule_note_state_value_read(
              state, node_index, operand_index, operand_ordinal,
              loom_low_schedule_descriptor_operand_read_endpoint(
                  state, descriptor,
                  descriptor != NULL ? descriptor_operand_indices[operand_index]
                                     : LOOM_LOW_ID_NONE)));
        }
      }

      IREE_RETURN_IF_ERROR(loom_low_schedule_note_storage_reads(
          state, node_index, descriptor, descriptor_operand_indices));
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
    }
    loom_low_schedule_reset_storage_reads(state);
  }
  return iree_ok_status();
}
