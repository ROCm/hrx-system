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
loom_low_schedule_effect_endpoint_none(void) {
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = LOOM_LOW_ID_NONE,
      .timing_event_id = LOOM_LOW_TIMING_EVENT_NONE,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_NONE,
  };
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_effect_endpoint(uint16_t effect_ordinal,
                                  uint16_t timing_event_id) {
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = effect_ordinal,
      .timing_event_id = timing_event_id,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_EFFECT,
  };
}

static loom_low_schedule_dependency_endpoint_t
loom_low_schedule_boundary_event_endpoint(uint16_t timing_event_id) {
  return (loom_low_schedule_dependency_endpoint_t){
      .attachment_index = LOOM_LOW_ID_NONE,
      .timing_event_id = timing_event_id,
      .attachment_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_NONE,
  };
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
  if (!loom_low_schedule_effect_is_ordered(effect)) return false;
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

bool loom_low_schedule_descriptor_has_ordered_effect(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_is_ordered(effect)) return true;
  }
  return false;
}

static void loom_low_schedule_effect_frontier_reset(
    loom_low_schedule_effect_frontier_t* frontier) {
  frontier->ordered_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  frontier->ordered_endpoint = loom_low_schedule_effect_endpoint_none();
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

static iree_status_t loom_low_schedule_effect_frontier_note_read(
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
    if (!loom_low_memory_access_summaries_may_alias(summary, &write->summary)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write),
        consumer_endpoint));
  }
  IREE_ASSERT(frontier->read_count < state->effect_read_capacity,
              "precomputed effect-frontier read capacity must cover all rows");
  frontier->reads[frontier->read_count] =
      (loom_low_schedule_effect_frontier_entry_t){
          .node_index = node_index,
          .effect_ordinal = producer_endpoint.attachment_index,
          .producer_event_id = producer_endpoint.timing_event_id,
          .summary = *summary,
      };
  ++frontier->read_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_effect_frontier_note_write_complete(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    const loom_low_memory_access_summary_t* summary,
    loom_low_schedule_dependency_endpoint_t producer_endpoint) {
  iree_host_size_t write_index = 0;
  for (iree_host_size_t read_index = 0; read_index < frontier->read_count;
       ++read_index) {
    const loom_low_schedule_effect_frontier_entry_t* read =
        &frontier->reads[read_index];
    if (loom_low_memory_access_write_subsumes_read(summary, &read->summary)) {
      continue;
    }
    frontier->reads[write_index++] = *read;
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
    frontier->writes[write_index++] = *write;
  }
  frontier->write_count = write_index;
  IREE_ASSERT(frontier->write_count < state->effect_write_capacity,
              "precomputed effect-frontier write capacity must cover all rows");
  frontier->writes[frontier->write_count] =
      (loom_low_schedule_effect_frontier_entry_t){
          .node_index = node_index,
          .effect_ordinal = producer_endpoint.attachment_index,
          .producer_event_id = producer_endpoint.timing_event_id,
          .summary = *summary,
      };
  ++frontier->write_count;
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
    if (!loom_low_memory_access_summaries_may_alias(summary, &write->summary)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write),
        consumer_endpoint));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* read = &frontier->reads[i];
    if (!loom_low_memory_access_summaries_may_alias(summary, &read->summary)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(read),
        consumer_endpoint));
  }
  return loom_low_schedule_effect_frontier_note_write_complete(
      state, frontier, node_index, summary, producer_endpoint);
}

static iree_status_t loom_low_schedule_effect_frontier_note_ordered(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_effect_frontier_t* frontier, uint32_t node_index,
    loom_low_schedule_dependency_endpoint_t consumer_endpoint,
    loom_low_schedule_dependency_endpoint_t producer_endpoint) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_depend_on_ordered(
      state, frontier, node_index, consumer_endpoint));
  for (iree_host_size_t i = 0; i < frontier->write_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* write =
        &frontier->writes[i];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, write->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(write),
        consumer_endpoint));
  }
  for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
    const loom_low_schedule_effect_frontier_entry_t* read = &frontier->reads[i];
    IREE_RETURN_IF_ERROR(loom_low_schedule_add_dependency(
        state, read->node_index, node_index,
        LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT, LOOM_LOW_ID_NONE,
        loom_low_schedule_effect_frontier_entry_endpoint(read),
        consumer_endpoint));
  }
  loom_low_schedule_effect_frontier_reset(frontier);
  frontier->ordered_node = node_index;
  frontier->ordered_endpoint = producer_endpoint;
  return iree_ok_status();
}

static const loom_low_memory_access_summary_t*
loom_low_schedule_lookup_memory_access_summary(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    const loom_low_descriptor_t* descriptor,
    const loom_low_effect_t* selected_effect) {
  const uint32_t record_index =
      state->nodes[node_index].memory_access_record_index;
  if (record_index == LOOM_LOW_SCHEDULE_MEMORY_ACCESS_RECORD_NONE ||
      record_index >= state->memory_access_record_count) {
    return NULL;
  }
  const loom_low_memory_access_record_t* record =
      &state->memory_access_records[record_index];
  const loom_low_memory_access_summary_t* summary = &record->summary;

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
  if (dependency_memory_effect_count == 1) return summary;
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
  if (descriptor->effect_count == 0) return iree_ok_status();
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
    const loom_low_effect_t* effect =
        &descriptor_set->effects[descriptor->effect_start + i];
    if (loom_low_schedule_effect_orders_memory(effect)) {
      return loom_low_schedule_effect_frontier_note_ordered(
          state, frontier, node_index,
          loom_low_schedule_effect_endpoint(i, effect->consumer_event_id),
          loom_low_schedule_effect_endpoint(i, effect->producer_event_id));
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
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_from_effect(effect);
    if (source_summary != NULL) summary = *source_summary;
    switch (effect->kind) {
      case LOOM_LOW_EFFECT_KIND_READ: {
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_read(
            state, frontier, node_index, &summary,
            loom_low_schedule_effect_endpoint(i, effect->consumer_event_id),
            loom_low_schedule_effect_endpoint(i, effect->producer_event_id)));
        break;
      }
      case LOOM_LOW_EFFECT_KIND_WRITE: {
        IREE_RETURN_IF_ERROR(loom_low_schedule_effect_frontier_note_write(
            state, frontier, node_index, &summary,
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
        state, frontier, node_index, loom_low_schedule_effect_endpoint_none(),
        loom_low_schedule_effect_endpoint_none());
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_WRITES_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_write(
        state, frontier, node_index, &summary,
        loom_low_schedule_effect_endpoint_none(),
        loom_low_schedule_effect_endpoint_none());
  }
  if (iree_any_bit_set(node->traits, LOOM_TRAIT_READS_MEMORY)) {
    loom_low_memory_access_summary_t summary =
        loom_low_memory_access_summary_synthetic(LOOM_LOW_MEMORY_SPACE_GENERIC);
    return loom_low_schedule_effect_frontier_note_read(
        state, frontier, node_index, &summary,
        loom_low_schedule_effect_endpoint_none(),
        loom_low_schedule_effect_endpoint_none());
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_add_boundary_event_requirement(
    loom_low_schedule_build_state_t* state, uint32_t producer_node,
    loom_low_schedule_dependency_endpoint_t producer_endpoint,
    uint32_t terminator_node, uint16_t consumer_event_id) {
  const loom_low_event_separation_t* separation =
      loom_low_descriptor_set_lookup_event_separation(
          state->target.descriptor_set, producer_endpoint.timing_event_id,
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
    if (loom_low_memory_access_summaries_may_alias(summary, &write->summary)) {
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_entry_requirement(
          state, write, terminator_node, consumer_event_id));
    }
  }
  if (consumer_kind == LOOM_LOW_EFFECT_KIND_WRITE) {
    for (iree_host_size_t i = 0; i < frontier->read_count; ++i) {
      const loom_low_schedule_effect_frontier_entry_t* read =
          &frontier->reads[i];
      if (loom_low_memory_access_summaries_may_alias(summary, &read->summary)) {
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
    for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
      const loom_low_effect_t* effect =
          &descriptor_set->effects[descriptor->effect_start + i];
      if (!loom_low_schedule_effect_orders_memory(effect)) continue;
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_ordered_requirement(
          state, frontier, terminator_node, effect->consumer_event_id));
      *out_frontier_consumed = true;
      return iree_ok_status();
    }
    for (uint16_t i = 0; i < descriptor->effect_count; ++i) {
      const loom_low_effect_t* effect =
          &descriptor_set->effects[descriptor->effect_start + i];
      if (!iree_any_bit_set(effect->flags, LOOM_LOW_EFFECT_FLAG_DEPENDENCY) ||
          (effect->kind != LOOM_LOW_EFFECT_KIND_READ &&
           effect->kind != LOOM_LOW_EFFECT_KIND_WRITE)) {
        continue;
      }
      const loom_low_memory_access_summary_t* source_summary =
          loom_low_schedule_lookup_memory_access_summary(state, node_index,
                                                         descriptor, effect);
      loom_low_memory_access_summary_t summary =
          loom_low_memory_access_summary_from_effect(effect);
      if (source_summary != NULL) summary = *source_summary;
      IREE_RETURN_IF_ERROR(loom_low_schedule_add_boundary_memory_requirement(
          state, frontier, terminator_node, &summary, effect->kind,
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
  if (source_block->node_count == 0) return iree_ok_status();
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
    if (visit_epochs[successor] == visit_epoch) continue;
    visit_epochs[successor] = visit_epoch;
    traversal_stack[stack_count++] = successor;
  }
  while (stack_count != 0) {
    const uint16_t block_index = traversal_stack[--stack_count];
    bool frontier_consumed = false;
    IREE_RETURN_IF_ERROR(loom_low_schedule_scan_boundary_consumer_block(
        state, frontier, terminator_node, block_index, &frontier_consumed));
    if (frontier_consumed) continue;
    const loom_cfg_block_index_span_t block_successors =
        loom_cfg_graph_successors(state->cfg_graph, block_index);
    for (iree_host_size_t i = 0; i < block_successors.count; ++i) {
      const uint16_t successor = block_successors.values[i];
      if (visit_epochs[successor] == visit_epoch) continue;
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
  const bool has_cfg_edges = state->cfg_graph->edge_count != 0;
  if (has_cfg_edges) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->body->block_count, sizeof(*traversal_stack),
        (void**)&traversal_stack));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->body->block_count, sizeof(*visit_epochs),
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

    if (has_cfg_edges) {
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
