// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/effect_dependencies.h"

#include <string.h>

#include "loom/codegen/low/schedule/graph.h"

typedef struct loom_low_schedule_effect_frontier_t {
  // Latest ordered effect node that every later dependency effect must follow.
  uint32_t ordered_node;
  // Descriptor attachment for ordered_node, or an empty endpoint.
  loom_low_schedule_dependency_endpoint_t ordered_endpoint;
  // Reads not yet completed through an equivalent later read.
  loom_low_schedule_effect_frontier_entry_t* reads;
  // Number of outstanding read entries.
  iree_host_size_t read_count;
  // Writes not yet completed through an equivalent later write.
  loom_low_schedule_effect_frontier_entry_t* writes;
  // Number of outstanding write entries.
  iree_host_size_t write_count;
  // Last write cutoff used to retire canonical descriptor-only reads in each
  // normalized memory space. Zero cannot precede an earlier node.
  uint32_t read_retirement_nodes[LOOM_LOW_MEMORY_SPACE_WASM_MEMORY + 1];
} loom_low_schedule_effect_frontier_t;

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_effect_endpoint(uint16_t effect_ordinal,
                                  uint16_t timing_event_id) {
  if (effect_ordinal == LOOM_LOW_ID_NONE) {
    return loom_low_schedule_dependency_endpoint_none();
  }
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = effect_ordinal,
      .timing_event_id = timing_event_id,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT,
      .is_commutative_state_update = false,
  };
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_boundary_event_endpoint(uint16_t timing_event_id) {
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = LOOM_LOW_ID_NONE,
      .timing_event_id = timing_event_id,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_NONE,
      .is_commutative_state_update = false,
  };
}

static bool loom_low_schedule_effect_is_ordered(
    const loom_low_effect_t* effect, loom_memory_access_flags_t access_flags) {
  if (iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_ORDERED)) {
    return true;
  }
  switch (effect->kind) {
    case LOOM_LOW_EFFECT_KIND_READ:
    case LOOM_LOW_EFFECT_KIND_WRITE:
      return iree_any_bit_set(access_flags, LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
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
    const loom_low_effect_t* effect, loom_memory_access_flags_t access_flags) {
  if (!loom_low_schedule_effect_is_ordered(effect, access_flags)) {
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

static bool loom_low_schedule_node_has_structural_effects(
    const loom_low_schedule_node_t* node) {
  return loom_traits_may_read(node->traits) ||
         loom_traits_may_write(node->traits) ||
         iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                            LOOM_TRAIT_CONVERGENT |
                                            LOOM_TRAIT_OBSERVABLE_EFFECT);
}

bool loom_low_schedule_node_has_ordered_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  const loom_low_descriptor_t* descriptor = node->descriptor;
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  for (uint16_t i = 0; i < descriptor_layout->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor_layout->effect_start + i];
    if (loom_low_schedule_effect_is_ordered(effect, node->op->instance_flags)) {
      return true;
    }
  }
  return false;
}

static void loom_low_schedule_effect_frontier_reset(
    loom_low_schedule_effect_frontier_t* frontier) {
  frontier->ordered_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  frontier->ordered_endpoint = loom_low_schedule_dependency_endpoint_none();
  frontier->read_count = 0;
  frontier->write_count = 0;
  memset(frontier->read_retirement_nodes, 0,
         sizeof(frontier->read_retirement_nodes));
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
  return loom_low_schedule_effect_endpoint(entry->effect_ordinal,
                                           entry->producer_event_id);
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

// An equal-footprint access after an opposite-kind predecessor carries the
// earlier access's completion transitively: R -> W -> R or W -> R -> W.
// W -> W alone is insufficient because it does not require completion.
static iree_host_size_t loom_low_schedule_effect_frontier_retire_completed(
    loom_low_schedule_effect_frontier_entry_t* entries, iree_host_size_t count,
    uint32_t predecessor_node,
    const loom_low_memory_access_summary_t* summary) {
  iree_host_size_t retained_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (entries[i].node_index < predecessor_node &&
        loom_low_memory_access_summaries_equal(summary, entries[i].summary)) {
      continue;
    }
    entries[retained_count] = entries[i];
    ++retained_count;
  }
  return retained_count;
}

static iree_status_t loom_low_schedule_effect_frontier_note_read(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint,
    loom_low_schedule_dependency_endpoint_t producer_endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, consumer_endpoint));
  uint32_t predecessor_node = 0;
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    if (!loom_low_memory_access_summaries_may_alias(
            summary, write->summary,
            LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write),
        consumer_endpoint));
    predecessor_node = iree_max(predecessor_node, write->node_index);
  }
  const loom_low_memory_space_t memory_space =
      loom_low_memory_access_normalize_space(summary->memory_space);
  if (summary == loom_low_memory_access_summary_for_space(memory_space) &&
      predecessor_node > frontier->read_retirement_nodes[memory_space]) {
    // Canonical descriptor summaries have a bounded key domain. Each new
    // opposite-kind cutoff retires once, so a read-only run never rescans its
    // growing read frontier after the first read.
    frontier->read_count = loom_low_schedule_effect_frontier_retire_completed(
        frontier->reads, frontier->read_count, predecessor_node, summary);
    frontier->read_retirement_nodes[memory_space] = predecessor_node;
  }
  IREE_ASSERT(frontier->read_count < state->effect_read_capacity,
              "precomputed effect-frontier read capacity must cover all rows");
  frontier->reads[frontier->read_count] =
      (loom_low_schedule_effect_frontier_entry_t){
          .node_index = node_index,
          .effect_ordinal = producer_endpoint.attachment_index,
          .producer_event_id = producer_endpoint.timing_event_id,
          .summary = summary,
      };
  ++frontier->read_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_write(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint,
    loom_low_schedule_dependency_endpoint_t producer_endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, consumer_endpoint));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    if (!loom_low_memory_access_summaries_may_alias(
            summary, write->summary,
            LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write),
        consumer_endpoint));
  }
  uint32_t predecessor_node = 0;
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* read = &frontier->reads[i];
    if (!loom_low_memory_access_summaries_may_alias(
            summary, read->summary,
            LOOM_LOW_MEMORY_COMPARISON_SAME_EVALUATION)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(read),
        consumer_endpoint));
    predecessor_node = iree_max(predecessor_node, read->node_index);
  }
  if (predecessor_node != 0) {
    frontier->write_count = loom_low_schedule_effect_frontier_retire_completed(
        frontier->writes, frontier->write_count, predecessor_node, summary);
  }
  IREE_ASSERT(frontier->write_count < state->effect_write_capacity,
              "precomputed effect-frontier write capacity must cover all rows");
  frontier->writes[frontier->write_count] =
      (loom_low_schedule_effect_frontier_entry_t){
          .node_index = node_index,
          .effect_ordinal = producer_endpoint.attachment_index,
          .producer_event_id = producer_endpoint.timing_event_id,
          .summary = summary,
      };
  ++frontier->write_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint,
    loom_low_schedule_dependency_endpoint_t producer_endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, consumer_endpoint));
  // Earlier ordered nodes already carry source order for their predecessors.
  // Keep the memory frontier separately: ordering a counter or barrier does
  // not establish completion of every memory access that precedes it.
  for (iree_host_size_t i = frontier->write_count; i > 0; --i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i - 1];
    if (frontier->ordered_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        write->node_index <= frontier->ordered_node) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write),
        consumer_endpoint));
  }
  for (iree_host_size_t i = frontier->read_count; i > 0; --i) {
    const loom_low_schedule_effect_frontier_entry_t* read =
        &frontier->reads[i - 1];
    if (frontier->ordered_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        read->node_index <= frontier->ordered_node) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(read),
        consumer_endpoint));
  }
  frontier->ordered_node = node_index;
  frontier->ordered_endpoint = producer_endpoint;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_note_descriptor_effects(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  if (descriptor_layout->effect_count == 0) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < descriptor_layout->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor_layout->effect_start + i];
    if (loom_low_schedule_effect_orders_memory(
            effect, state->nodes[node_index].op->instance_flags)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_ordered(
          state, frontier, node_index,
          loom_low_schedule_effect_endpoint(i, effect->consumer_event_id),
          loom_low_schedule_effect_endpoint(i, effect->producer_event_id)));
      break;
    }
  }
  for (uint16_t i = 0; i < descriptor_layout->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor_layout->effect_start + i];
    if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY)) {
      continue;
    }
    const loom_low_memory_access_summary_t* summary =
        loom_low_memory_access_map_lookup(state->memory_accesses,
                                          state->nodes[node_index].op, i);
    if (summary == NULL) {
      summary = loom_low_memory_access_summary_for_space(effect->memory_space);
    }
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_read(
            state, frontier, node_index, summary,
            loom_low_schedule_effect_endpoint(i, effect->consumer_event_id),
            loom_low_schedule_effect_endpoint(i, effect->producer_event_id)));
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_write(
            state, frontier, node_index, summary,
            loom_low_schedule_effect_endpoint(i, effect->consumer_event_id),
            loom_low_schedule_effect_endpoint(i, effect->producer_event_id)));
        break;
      }
      default:
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
                                         LOOM_TRAIT_CONVERGENT |
                                         LOOM_TRAIT_OBSERVABLE_EFFECT)) {
    return loom_low_schedule_effect_frontier_note_ordered(
        state, frontier, node_index,
        loom_low_schedule_dependency_endpoint_none(),
        loom_low_schedule_dependency_endpoint_none());
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
    const loom_low_memory_access_summary_t* summary =
        loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_write(
        state, frontier, node_index, summary,
        loom_low_schedule_dependency_endpoint_none(),
        loom_low_schedule_dependency_endpoint_none());
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
    const loom_low_memory_access_summary_t* summary =
        loom_low_memory_access_summary_for_space(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_read(
        state, frontier, node_index, summary,
        loom_low_schedule_dependency_endpoint_none(),
        loom_low_schedule_dependency_endpoint_none());
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_boundary_event_requirement(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    loom_low_schedule_dependency_endpoint_t producer_endpoint,
    uint32_t terminator_node, uint16_t consumer_event_id) {
  const loom_low_event_separation_t* separation =
      loom_low_schedule_model_lookup_event_separation(
          state->target.schedule_model, producer_endpoint.timing_event_id,
          consumer_event_id);
  if (separation == NULL || separation->minimum_issue_separation_cycles <= 0) {
    return iree_ok_status();
  }
  if (producer_node == terminator_node) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot preserve a positive timing-event separation across a CFG "
        "edge when the producer is the source terminator");
  }
  // The virtual consumer endpoint carries the future block's timing event
  // without pretending that its descriptor attachment belongs to the source
  // terminator.
  return loom_low_schedule_add_dependency(
      state, producer_node, terminator_node,
      LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE, producer_endpoint,
      loom_low_schedule_boundary_event_endpoint(consumer_event_id));
}

static iree_status_t loom_low_schedule_add_boundary_entry_requirement(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_effect_frontier_entry_t* entry,
    uint32_t terminator_node, uint16_t consumer_event_id) {
  return loom_low_schedule_add_boundary_event_requirement(
      state, entry->node_index,
      loom_low_schedule_effect_frontier_entry_endpoint(entry), terminator_node,
      consumer_event_id);
}

static iree_status_t loom_low_schedule_add_boundary_ordered_requirement(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_effect_frontier_t* frontier,
    uint32_t terminator_node, uint16_t consumer_event_id) {
  if (frontier->ordered_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_event_requirement(
        state, frontier->ordered_node, frontier->ordered_endpoint,
        terminator_node, consumer_event_id));
  }
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_entry_requirement(
        state, &frontier->writes[i], terminator_node, consumer_event_id));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_entry_requirement(
        state, &frontier->reads[i], terminator_node, consumer_event_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_boundary_memory_requirement(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_effect_frontier_t* frontier,
    uint32_t terminator_node, const loom_low_memory_access_summary_t* summary,
    loom_low_effect_kind_t consumer_kind, uint16_t consumer_event_id) {
  if (frontier->ordered_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_event_requirement(
        state, frontier->ordered_node, frontier->ordered_endpoint,
        terminator_node, consumer_event_id));
  }
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    if (loom_low_memory_access_summaries_may_alias(
            summary, write->summary, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_entry_requirement(
          state, write, terminator_node, consumer_event_id));
    }
  }
  if (consumer_kind == LOOM_LOW_EFFECT_KIND_WRITE) {
    for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
      const loom_low_schedule_effect_frontier_entry_t* read =
          &frontier->reads[i];
      if (loom_low_memory_access_summaries_may_alias(
              summary, read->summary, LOOM_LOW_MEMORY_COMPARISON_INDEPENDENT)) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_entry_requirement(
            state, read, terminator_node, consumer_event_id));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_scan_boundary_consumer_block(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_effect_frontier_t* frontier,
    uint32_t terminator_node, uint16_t block_index,
    bool* out_frontier_consumed) {
  *out_frontier_consumed = false;
  const loom_low_schedule_block_t* block = &state->blocks[block_index];
  const uint32_t node_end = block->node_start + block->node_count;
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint32_t node_index = block->node_start; node_index < node_end;
       ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_low_descriptor_t* descriptor = node->descriptor;
    if (descriptor == NULL) {
      if (iree_any_bit_set(node->traits, LOOM_TRAIT_NON_DETERMINISTIC |
                                             LOOM_TRAIT_UNKNOWN_EFFECTS |
                                             LOOM_TRAIT_CONVERGENT |
                                             LOOM_TRAIT_OBSERVABLE_EFFECT)) {
        *out_frontier_consumed = true;
        return iree_ok_status();
      }
      continue;
    }
    const loom_low_descriptor_layout_t* descriptor_layout =
        loom_low_descriptor_layout(descriptor_set, descriptor);
    for (uint16_t i = 0; i < descriptor_layout->effect_count; ++i) {
      const loom_low_effect_t* effect =
          &descriptor_set->effects[descriptor_layout->effect_start + i];
      if (!loom_low_schedule_effect_orders_memory(effect,
                                                  node->op->instance_flags)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_ordered_requirement(
          state, frontier, terminator_node, effect->consumer_event_id));
      *out_frontier_consumed = true;
      return iree_ok_status();
    }
    for (uint16_t i = 0; i < descriptor_layout->effect_count; ++i) {
      const loom_low_effect_t* effect =
          &descriptor_set->effects[descriptor_layout->effect_start + i];
      if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY) ||
          (effect->kind != LOOM_LOW_EFFECT_KIND_READ &&
           effect->kind != LOOM_LOW_EFFECT_KIND_WRITE)) {
        continue;
      }
      const loom_low_memory_access_summary_t* summary =
          loom_low_memory_access_map_lookup(state->memory_accesses,
                                            state->nodes[node_index].op, i);
      if (summary == NULL) {
        summary =
            loom_low_memory_access_summary_for_space(effect->memory_space);
      }
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_memory_requirement(
          state, frontier, terminator_node, summary, effect->kind,
          effect->consumer_event_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_cfg_boundary_dependencies(
    loom_low_schedule_build_state_t* state, uint16_t source_block_index,
    const loom_low_schedule_effect_frontier_t* frontier,
    uint16_t* traversal_stack, uint32_t* visit_epochs, uint32_t visit_epoch) {
  if (frontier->ordered_node == LOOM_LOW_SCHEDULE_NODE_NONE &&
      frontier->read_count == 0 && frontier->write_count == 0) {
    return iree_ok_status();
  }
  const loom_low_schedule_block_t* source_block =
      &state->blocks[source_block_index];
  if (source_block->node_count == 0) {
    return iree_ok_status();
  }
  const uint32_t terminator_node =
      source_block->node_start + source_block->node_count - 1;
  IREE_ASSERT(state->nodes[terminator_node].kind ==
              LOOM_LOW_SCHEDULE_NODE_TERMINATOR);

  // Each block has its own issue-cycle origin and the dependency DAG remains
  // block-local so loop backedges cannot make it cyclic. Search every CFG path
  // for future consumers and constrain the source terminator instead: padding
  // the source through the required event separation makes successor cycle 0
  // safe while naturally admitting any later issue cycle in that successor.
  // The source block is intentionally not pre-visited so a backedge can impose
  // the next iteration's first-consumer requirement.
  iree_host_size_t stack_count = 0;
  const loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(state->cfg_graph, source_block_index);
  for (iree_host_size_t i = 0; i < successors.count; ++i) {
    const uint16_t successor = successors.values[i];
    if (visit_epochs[successor] == visit_epoch) {
      continue;
    }
    visit_epochs[successor] = visit_epoch;
    traversal_stack[stack_count++] = successor;
  }
  while (stack_count != 0) {
    const uint16_t block_index = traversal_stack[--stack_count];
    bool frontier_consumed = false;
    IREE_RETURN_IF_ERROR(loom_low_schedule_scan_boundary_consumer_block(
        state, frontier, terminator_node, block_index, &frontier_consumed));
    if (frontier_consumed) {
      continue;
    }
    const loom_cfg_block_index_span_t block_successors =
        loom_cfg_graph_successors(state->cfg_graph, block_index);
    for (iree_host_size_t i = 0; i < block_successors.count; ++i) {
      const uint16_t successor = block_successors.values[i];
      if (visit_epochs[successor] == visit_epoch) {
        continue;
      }
      visit_epochs[successor] = visit_epoch;
      traversal_stack[stack_count++] = successor;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_build_effect_dependencies(
    loom_low_schedule_build_state_t* state) {
  uint16_t* traversal_stack = NULL;
  uint32_t* visit_epochs = NULL;
  // Only positive effect pairs can constrain a source terminator. Operand
  // timing and block-local effect ordering need no successor traversal.
  const bool has_boundary_timing =
      iree_any_bit_set(
          state->target.schedule_model->flags,
          LOOM_LOW_SCHEDULE_MODEL_FLAG_POSITIVE_EFFECT_SEPARATIONS) &&
      state->cfg_graph->edge_count != 0;
  if (has_boundary_timing) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, state->body->block_count,
        sizeof(*traversal_stack), (void**)&traversal_stack));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, state->body->block_count, sizeof(*visit_epochs),
        (void**)&visit_epochs));
    memset(visit_epochs, 0, state->body->block_count * sizeof(*visit_epochs));
  }

  uint32_t visit_epoch = 0;
  for (uint16_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_low_schedule_effect_frontier_t frontier;
    loom_low_schedule_effect_frontier_initialize(state, &frontier);
    const loom_low_schedule_block_t* block = &state->blocks[block_index];
    const uint32_t node_end = block->node_start + block->node_count;
    for (uint32_t node_index = block->node_start; node_index < node_end;
         ++node_index) {
      const loom_low_schedule_node_t* node = &state->nodes[node_index];
      if (node->descriptor != NULL) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_descriptor_effects(
            state, &frontier, node_index, node->descriptor));
      } else if (loom_low_schedule_node_has_structural_effects(node)) {
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_structural_effects(
            state, &frontier, node_index));
      }
    }

    if (has_boundary_timing) {
      ++visit_epoch;
      if (visit_epoch == 0) {
        memset(visit_epochs, 0,
               state->body->block_count * sizeof(*visit_epochs));
        visit_epoch = 1;
      }
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_cfg_boundary_dependencies(
          state, block_index, &frontier, traversal_stack, visit_epochs,
          visit_epoch));
    }
  }
  return iree_ok_status();
}
