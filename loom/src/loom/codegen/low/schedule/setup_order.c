// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/setup_order.h"

#include <string.h>

#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/pressure.h"

iree_status_t loom_low_schedule_setup_order_initialize(
    uint32_t node_count, iree_arena_allocator_t* arena,
    loom_low_schedule_setup_order_t* out_order) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*out_order->completion_nodes),
      (void**)&out_order->completion_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*out_order->entry_nodes),
      (void**)&out_order->entry_nodes));
  memset(out_order->completion_nodes, 0xFF,
         node_count * sizeof(*out_order->completion_nodes));
  memset(out_order->entry_nodes, 0xFF,
         node_count * sizeof(*out_order->entry_nodes));
  return iree_ok_status();
}

static bool loom_low_schedule_setup_order_has_per_user_placement(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, bool* out_requires_register) {
  *out_requires_register = false;
  const iree_bitmap_t per_user_placement_values =
      state->options->per_user_placement_values;
  if (node->result_count != 1 || per_user_placement_values.bit_count == 0) {
    return false;
  }
  const loom_value_id_t value_id =
      state->value_domain
          ->value_ids[loom_low_schedule_node_const_result_ordinals(node)[0]];
  if (value_id >= per_user_placement_values.bit_count ||
      !iree_bitmap_test(per_user_placement_values, value_id)) {
    return false;
  }
  *out_requires_register =
      value_id < state->options->required_register_values.bit_count &&
      iree_bitmap_test(state->options->required_register_values, value_id);
  return true;
}

static bool loom_low_schedule_setup_order_is_member(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node, bool order_structural_setup) {
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL) {
    if (order_structural_setup) {
      return true;
    }
    bool requires_register = false;
    if (loom_low_schedule_setup_order_has_per_user_placement(
            state, node, &requires_register) &&
        requires_register) {
      return true;
    }
    if (!iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP) ||
        node->result_count != 1) {
      return false;
    }
    // A singleton cannot hold a later setup while its consumer's prerequisites
    // still need that location. Other storage remains available for pressure
    // and issue-slot scheduling instead of acquiring these ordering edges.
    const loom_low_schedule_value_record_t* result =
        &state->values[loom_low_schedule_node_const_result_ordinals(node)[0]];
    if (!loom_low_schedule_value_requires_register(result)) {
      return false;
    }
    const uint16_t domain = loom_low_schedule_unspillable_completion_domain_id(
        state, result->register_class_id);
    return domain != UINT16_MAX && result->unit_count == 1 &&
           state->pressure_limits.unspillable_completion_domains[domain]
                   .capacity == 1;
  }
  if (node->result_count != 1) {
    return false;
  }
  // Input-free rematerializable values remain pressure-scheduled. A required
  // private value cannot be cloned and instead retains the complete target
  // setup chain that produced it.
  bool requires_register = false;
  const bool has_per_user_placement =
      loom_low_schedule_setup_order_has_per_user_placement(state, node,
                                                           &requires_register);
  return has_per_user_placement &&
         (node->operand_count != 0 || requires_register);
}

void loom_low_schedule_setup_order_classify_node(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node,
    bool order_structural_setup) {
  if (loom_low_schedule_setup_order_is_member(state, node,
                                              order_structural_setup)) {
    node->flags |= LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP;
    state->setup_order.has_members = true;
  }
}

static iree_status_t loom_low_schedule_setup_order_append(
    loom_low_schedule_build_state_t* state, uint32_t producer,
    uint32_t consumer) {
  if (state->dependencies.count >= UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t index capacity");
  }
  // The semantic graph is complete: these edges only close its retained setup
  // chains and do not contribute another successor observation.
  return loom_low_schedule_dependency_graph_append(
      &state->dependencies,
      (loom_low_schedule_dependency_t){
          .producer_node = producer,
          .consumer_node = consumer,
          .producer_attachment_index = LOOM_LOW_ID_NONE,
          .consumer_attachment_index = LOOM_LOW_ID_NONE,
          .producer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
          .consumer_event_id = LOOM_LOW_TIMING_EVENT_NONE,
          .value_operand_index = LOOM_LOW_ID_NONE,
          .kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER,
      },
      state->arena);
}

iree_status_t loom_low_schedule_setup_order_finish(
    loom_low_schedule_build_state_t* state) {
  loom_low_schedule_setup_order_t* order = &state->setup_order;
  if (order->completion_nodes == NULL) {
    return iree_ok_status();
  }
  const uint32_t dependency_count = (uint32_t)state->dependencies.count;
  iree_status_t status = iree_ok_status();
  for (uint16_t block_index = 0;
       block_index < state->body->block_count && iree_status_is_ok(status);
       ++block_index) {
    const loom_low_schedule_block_t* block = &state->blocks[block_index];
    uint32_t range_end = block->node_start + block->node_count;
    for (uint32_t i = range_end;
         i > block->node_start && iree_status_is_ok(status); --i) {
      const uint32_t node_index = i - 1;
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      const uint32_t successor = order->completion_nodes[node_index];
      const uint32_t value_successor = order->entry_nodes[node_index];
      order->completion_nodes[node_index] = node_index;
      order->entry_nodes[node_index] = node_index;
      if (iree_any_bit_set(node->flags,
                           LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY)) {
        range_end = node_index;
        continue;
      }
      // An order-only successor does not consume the produced value. Grouping
      // unrelated readers around their next state writer would retain all of
      // their inputs together instead of completing each private setup.
      if (successor != value_successor || successor <= node_index ||
          successor >= range_end ||
          !iree_any_bit_set(node->flags,
                            LOOM_LOW_SCHEDULE_NODE_FLAG_ORDERED_SETUP)) {
        continue;
      }
      const uint32_t completion = order->completion_nodes[successor];
      const uint32_t next_member = order->entry_nodes[completion];
      order->completion_nodes[node_index] = completion;
      order->entry_nodes[completion] = node_index;
      if (next_member != successor) {
        status = loom_low_schedule_setup_order_append(state, node_index,
                                                      next_member);
      }
    }
  }
  // Emit prerequisites from retained membership. Each original edge adds at
  // most one order edge in this linear pass. The ordinary dependency index
  // owns deduplication and all subsequent scheduling.
  for (uint32_t i = 0; i < dependency_count && iree_status_is_ok(status); ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&state->dependencies, i);
    const uint32_t producer = dependency->producer_node;
    const uint32_t consumer = dependency->consumer_node;
    const uint32_t completion = order->completion_nodes[consumer];
    const uint32_t entry = order->entry_nodes[completion];
    if (entry != consumer && order->completion_nodes[producer] != completion) {
      status = loom_low_schedule_setup_order_append(state, producer, entry);
    }
  }
  return status;
}
