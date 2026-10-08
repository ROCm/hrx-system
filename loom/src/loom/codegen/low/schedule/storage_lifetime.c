// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/storage_lifetime.h"

#include <string.h>

#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/pressure.h"

static bool loom_low_schedule_storage_relation_carries_lifetime(
    const loom_low_schedule_storage_relation_t* relation) {
  return relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT ||
         (relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_COPY &&
          relation->kind == LOOM_LOW_STORAGE_RELATION_SAME_STORAGE);
}

static bool loom_low_schedule_storage_relation_is_handoff(
    const loom_low_schedule_storage_relation_t* relation) {
  return relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH ||
         relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD;
}

static void loom_low_schedule_storage_lifetimes_populate(
    loom_low_schedule_build_state_t* state) {
  loom_low_schedule_storage_lifetimes_t* lifetimes = &state->storage_lifetimes;
  uint32_t handoff_count = 0;
  for (uint32_t node_index = 0;
       node_index < state->storage_relations.node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const uint32_t begin = loom_low_schedule_storage_relation_index_begin(
        &state->storage_relations, node_index);
    const uint32_t end = loom_low_schedule_storage_relation_index_end(
        &state->storage_relations, node_index);
    for (uint32_t i = begin; i < end; ++i) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      i);
      if (lifetimes->roots != NULL &&
          loom_low_schedule_storage_relation_carries_lifetime(relation)) {
        const loom_value_ordinal_t root =
            lifetimes->roots[relation->source_ordinal];
        const loom_low_schedule_value_record_t* source = &state->values[root];
        const loom_low_schedule_value_record_t* destination =
            &state->values[relation->destination_ordinal];
        const loom_value_t* value = loom_module_value(
            state->module, state->value_domain->value_ids[root]);
        if (loom_value_is_block_arg(value) &&
            loom_value_def_block(value) ==
                state->body->blocks[node->block_index] &&
            iree_any_bit_set(
                source->flags,
                LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED) &&
            source->register_class_id == destination->register_class_id &&
            source->unit_count == destination->unit_count &&
            relation->source_unit_offset == 0 &&
            relation->destination_unit_offset == 0 &&
            relation->unit_count == source->unit_count) {
          lifetimes->roots[relation->destination_ordinal] = root;
        }
      }
      if (lifetimes->block_handoffs != NULL &&
          loom_low_schedule_storage_relation_is_handoff(relation)) {
        const uint32_t producer =
            state->value_producer_nodes[relation->source_ordinal];
        if (producer != LOOM_LOW_SCHEDULE_NODE_NONE &&
            state->nodes[producer].block_index != node->block_index) {
          const uint32_t block_index = state->nodes[producer].block_index;
          lifetimes->handoffs[handoff_count] =
              (loom_low_schedule_storage_handoff_t){
                  .relation_index = i,
                  .next_handoff = lifetimes->block_handoffs[block_index],
              };
          lifetimes->block_handoffs[block_index] = handoff_count++;
        }
      }
    }
  }
}

iree_status_t loom_low_schedule_storage_lifetimes_initialize(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  IREE_ASSERT_LE(node_count, UINT32_MAX);
  IREE_RETURN_IF_ERROR(loom_low_schedule_storage_relation_index_initialize(
      state->module, state->value_domain, state->nodes, (uint32_t)node_count,
      state->storage_relation_count, state->scratch_arena,
      &state->storage_relations));
  bool needs_storage_read_tracking = false;
  bool needs_edge_source_worklist = false;
  bool needs_lifetime_roots = false;
  uint32_t handoff_count = 0;
  iree_host_size_t max_operand_count = 0;
  // Mark all tracked destinations before propagating header lifetimes: a
  // backedge establishing the placement opportunity occurs after its reads.
  for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    max_operand_count = iree_max(max_operand_count, node->operand_count);
    const uint32_t begin = loom_low_schedule_storage_relation_index_begin(
        &state->storage_relations, node_index);
    const uint32_t end = loom_low_schedule_storage_relation_index_end(
        &state->storage_relations, node_index);
    for (uint32_t i = begin; i < end; ++i) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      i);
      needs_lifetime_roots |=
          loom_low_schedule_storage_relation_carries_lifetime(relation);
      if (relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT) {
        state->values[relation->source_ordinal].flags |=
            LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED;
        needs_storage_read_tracking = true;
      }
      if (loom_low_schedule_storage_relation_is_handoff(relation)) {
        state->values[relation->destination_ordinal].flags |=
            LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED;
        needs_storage_read_tracking = true;
        needs_edge_source_worklist = true;
        const uint32_t producer =
            state->value_producer_nodes[relation->source_ordinal];
        if (producer != LOOM_LOW_SCHEDULE_NODE_NONE &&
            state->nodes[producer].block_index != node->block_index) {
          ++handoff_count;
        }
      }
    }
  }
  if (!needs_storage_read_tracking || state->value_domain->value_count == 0) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, value_count, sizeof(*state->storage_reads.heads),
      (void**)&state->storage_reads.heads));
  memset(state->storage_reads.heads, 0xFF,
         value_count * sizeof(*state->storage_reads.heads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, value_count,
      sizeof(*state->storage_reads.touched_ordinals),
      (void**)&state->storage_reads.touched_ordinals));
  if (max_operand_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, max_operand_count,
        sizeof(*state->storage_reads.operand_relation_flags),
        (void**)&state->storage_reads.operand_relation_flags));
    state->storage_reads.operand_relation_flag_capacity = max_operand_count;
  }
  if (needs_edge_source_worklist) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, value_count,
        sizeof(*state->storage_reads.edge_source_worklist),
        (void**)&state->storage_reads.edge_source_worklist));
    state->storage_reads.edge_source_worklist_capacity = value_count;
  }
  loom_low_schedule_storage_lifetimes_t* lifetimes = &state->storage_lifetimes;
  if (needs_lifetime_roots) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, value_count, sizeof(*lifetimes->roots),
        (void**)&lifetimes->roots));
    for (loom_value_ordinal_t i = 0; i < value_count; ++i) {
      lifetimes->roots[i] = i;
    }
  }
  if (handoff_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, state->body->block_count,
        sizeof(*lifetimes->block_handoffs),
        (void**)&lifetimes->block_handoffs));
    memset(lifetimes->block_handoffs, 0xFF,
           state->body->block_count * sizeof(*lifetimes->block_handoffs));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, handoff_count, sizeof(*lifetimes->handoffs),
        (void**)&lifetimes->handoffs));
  }
  if (needs_lifetime_roots || handoff_count != 0) {
    loom_low_schedule_storage_lifetimes_populate(state);
  }
  if (state->storage_origins != NULL) {
    for (loom_value_ordinal_t value = 0; value < value_count; ++value) {
      state->values[state->storage_origins[value]].flags |=
          state->values[value].flags &
          LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED;
    }
  }
  return iree_ok_status();
}

void loom_low_schedule_storage_lifetimes_set_forwarded_values(
    loom_low_schedule_build_state_t* state, uint32_t block_index,
    bool is_forwarded) {
  const loom_low_schedule_block_t* block = &state->blocks[block_index];
  if (block->node_count == 0 ||
      !loom_low_schedule_strategy_uses_pressure(state->options->strategy) ||
      (state->pressure_limits.unspillable_completion_domain_count == 0 &&
       state->target.descriptor_set->register_packing_resource_count == 0)) {
    return;
  }
  const loom_low_schedule_value_flags_t forwarding_flags =
      is_forwarded ? LOOM_LOW_SCHEDULE_VALUE_FLAG_FORWARDED : 0;
  const loom_liveness_block_relation_t* liveness =
      &state->liveness_dataflow->blocks[block_index];
  for (iree_host_size_t i = 0; i < liveness->live_out_count; ++i) {
    const loom_value_ordinal_t ordinal = loom_local_value_domain_ordinal(
        state->value_domain, liveness->live_out_values[i]);
    loom_low_schedule_value_record_t* value = &state->values[ordinal];
    value->flags = (value->flags & ~LOOM_LOW_SCHEDULE_VALUE_FLAG_FORWARDED) |
                   forwarding_flags;
  }
  const uint32_t endpoint = block->node_start + block->node_count - 1;
  const uint32_t begin = loom_low_schedule_storage_relation_index_begin(
      &state->storage_relations, endpoint);
  const uint32_t end = loom_low_schedule_storage_relation_index_end(
      &state->storage_relations, endpoint);
  for (uint32_t i = begin; i < end; ++i) {
    const loom_low_schedule_storage_relation_t* relation =
        loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                    i);
    if (relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH) {
      loom_low_schedule_value_record_t* value =
          &state->values[relation->source_ordinal];
      value->flags = (value->flags & ~LOOM_LOW_SCHEDULE_VALUE_FLAG_FORWARDED) |
                     forwarding_flags;
    }
  }
}
