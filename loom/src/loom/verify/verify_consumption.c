// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify_consumption.h"

#include <string.h>

#include "loom/analysis/value_relation.h"
#include "loom/error/error_catalog.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/segmented_storage.h"
#include "loom/verify/verify_diagnostics.h"
#include "loom/verify/verify_state.h"

// All payloads fit in a normal 4KiB arena block. Indexes address fixed-size
// segments; growth never copies payloads or requests a module-sized allocation.
#define LOOM_VERIFY_CONSUMPTION_SEGMENT_COUNT 64u

typedef struct loom_verify_storage_identity_t {
  // Canonical nonwriting owner plus one, or zero for an independent owner.
  uint32_t owner;
  // Next alias plus one; the owner's entry heads its complete member list.
  uint32_t next;
} loom_verify_storage_identity_t;

typedef struct loom_verify_consumption_region_t {
  // Region visited by the structural verifier.
  const loom_region_t* region;
  // Shared verifier CFG, or NULL until a single-block CFG needs extraction.
  loom_cfg_graph_t* graph;
  // Operation whose execution contains this region, or NULL at module scope.
  const loom_op_t* owner;
  // Enclosing retained region, or UINT32_MAX at the verification root.
  uint32_t parent;
  // First block's index in reusable block summaries.
  uint32_t block_base;
  // Canonical owner plus one currently using this region's scratch.
  uint32_t active_owner;
  // First touched block in the active owner's sparse list, or UINT16_MAX.
  uint16_t first_block;
  // Declared recurrence and continuation at the owning operation.
  loom_region_execution_t execution;
} loom_verify_consumption_region_t;

typedef struct loom_verify_consumption_event_t {
  // Original consuming operand, retained for diagnostics.
  loom_use_t use;
  // Canonical owner, independent of the operand's alias spelling.
  loom_value_id_t owner;
  // Retained region containing the consumer.
  uint32_t region_index;
} loom_verify_consumption_event_t;

typedef struct loom_verify_consumption_block_t {
  // Observation proving a local conflict, or the last observation so far.
  loom_use_t observation;
  // Representative observation reached when entry is already consumed.
  loom_use_t entry_observation;
  // Earliest local consumer, then a witness for consumed entry reachability.
  loom_use_t consumption;
  // Block-local operation anchoring observation.
  const loom_op_t* observation_anchor;
  // Block-local operation anchoring the earliest local consumption.
  const loom_op_t* consumption_anchor;
  // Canonical owner plus one currently using this block's scratch.
  uint32_t active_owner;
  // Next touched block in this region, or UINT16_MAX.
  uint16_t next_block;
  // True once union reachability has queued this block for the active owner.
  bool queued;
} loom_verify_consumption_block_t;

static_assert(sizeof(loom_verify_storage_identity_t) == 8,
              "storage identity entries must remain two value indexes");
static_assert(sizeof(loom_verify_consumption_event_t) == 16,
              "consumers must retain compact use and owner indexes");
static_assert(sizeof(loom_verify_consumption_region_t) <= 40,
              "retained regions must remain compact");
static_assert(sizeof(loom_verify_consumption_block_t) <= 48,
              "reusable block summaries must remain compact");

struct loom_verify_consumption_t {
  // Lazily initialized value-indexed required storage identities.
  loom_segmented_storage_t identities;
  // Region facts produced by the verifier's existing structural walk.
  loom_segmented_storage_t regions;
  // Number of retained region facts.
  uint32_t region_count;
  // Total number of blocks covered by retained regions.
  uint32_t block_count;
  // Ownership-transfer occurrences, sorted by canonical owner.
  loom_segmented_storage_t events;
  // Number of recorded occurrences.
  iree_host_size_t event_count;
  // Region indexes sorted by region pointer for direct use-site lookup.
  loom_segmented_storage_t region_index;
  // Reusable sparse block summaries for one ownership family.
  loom_segmented_storage_t blocks;
  // Region indexes touched by the active family, sorted child before parent.
  loom_segmented_storage_t active_regions;
  // Number of active region indexes.
  iree_host_size_t active_region_count;
  // Reusable block-index FIFO for one region's union reachability.
  loom_segmented_storage_t frontier;
  // Number of queued blocks.
  iree_host_size_t frontier_count;
};

static void* loom_verify_consumption_at(loom_segmented_storage_t* storage,
                                        iree_host_size_t index,
                                        iree_host_size_t element_size) {
  return (uint8_t*)loom_segmented_storage_segment(
             storage,
             (uint32_t)(index / LOOM_VERIFY_CONSUMPTION_SEGMENT_COUNT)) +
         (index % LOOM_VERIFY_CONSUMPTION_SEGMENT_COUNT) * element_size;
}

static iree_status_t loom_verify_consumption_grow(
    loom_verify_state_t* state, loom_segmented_storage_t* storage,
    iree_host_size_t index, iree_host_size_t element_size) {
  if (!storage->segment_size) {
    loom_segmented_storage_initialize(
        element_size * LOOM_VERIFY_CONSUMPTION_SEGMENT_COUNT, iree_max_align_t,
        storage);
  }
  while (index / LOOM_VERIFY_CONSUMPTION_SEGMENT_COUNT >=
         storage->segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(storage, &state->arena, &segment));
    memset(segment, 0, storage->segment_size);
  }
  return iree_ok_status();
}

static iree_status_t loom_verify_consumption_initialize(
    loom_verify_state_t* state) {
  if (!state->consumption) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(&state->arena,
                                             sizeof(*state->consumption),
                                             (void**)&state->consumption));
    memset(state->consumption, 0, sizeof(*state->consumption));
  }
  return iree_ok_status();
}

static loom_verify_consumption_region_t* loom_verify_consumption_region(
    loom_verify_consumption_t* consumption, uint32_t index) {
  return loom_verify_consumption_at(&consumption->regions, index,
                                    sizeof(loom_verify_consumption_region_t));
}

static loom_verify_consumption_event_t* loom_verify_consumption_event(
    loom_segmented_storage_t* events, iree_host_size_t index) {
  return loom_verify_consumption_at(events, index,
                                    sizeof(loom_verify_consumption_event_t));
}

static uint32_t* loom_verify_consumption_index(
    loom_segmented_storage_t* indexes, iree_host_size_t index) {
  return loom_verify_consumption_at(indexes, index, sizeof(uint32_t));
}

static loom_verify_storage_identity_t* loom_verify_storage_identity(
    loom_verify_consumption_t* consumption, loom_value_id_t value) {
  if (value / LOOM_VERIFY_CONSUMPTION_SEGMENT_COUNT >=
      consumption->identities.segment_count) {
    return NULL;
  }
  return loom_verify_consumption_at(&consumption->identities, value,
                                    sizeof(loom_verify_storage_identity_t));
}

static loom_value_id_t loom_verify_storage_owner(
    loom_verify_consumption_t* consumption, loom_value_id_t value) {
  const loom_verify_storage_identity_t* identity =
      loom_verify_storage_identity(consumption, value);
  return identity && identity->owner ? identity->owner - 1 : value;
}

iree_status_t loom_verify_consumption_record_region(
    loom_verify_state_t* state, const loom_region_t* region,
    const loom_op_t* owner, loom_region_execution_t execution,
    const loom_cfg_graph_t* graph, uint32_t parent_index, uint32_t* out_index) {
  IREE_RETURN_IF_ERROR(loom_verify_consumption_initialize(state));
  loom_verify_consumption_t* consumption = state->consumption;
  if (consumption->region_count == UINT32_MAX ||
      region->block_count > UINT32_MAX - consumption->block_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "verification region/block index exceeds u32");
  }
  const uint32_t index = consumption->region_count;
  IREE_RETURN_IF_ERROR(
      loom_verify_consumption_grow(state, &consumption->regions, index,
                                   sizeof(loom_verify_consumption_region_t)));
  loom_cfg_graph_t* retained_graph = NULL;
  if (graph->region) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &state->arena, sizeof(*retained_graph), (void**)&retained_graph));
    *retained_graph = *graph;
  }
  *loom_verify_consumption_region(consumption, index) =
      (loom_verify_consumption_region_t){
          .region = region,
          .graph = retained_graph,
          .owner = owner,
          .parent = parent_index,
          .block_base = consumption->block_count,
          .execution = execution,
      };
  ++consumption->region_count;
  consumption->block_count += region->block_count;
  *out_index = index;
  return iree_ok_status();
}

iree_status_t loom_verify_consumption_record_aliases(loom_verify_state_t* state,
                                                     const loom_op_t* op) {
  if (!loom_traits_have_storage_relation(op->traits) ||
      !iree_any_bit_set(op->traits,
                        LOOM_TRAIT_FACT_IDENTITY | LOOM_TRAIT_VALUE_ALIAS)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_verify_consumption_initialize(state));
  loom_verify_consumption_t* consumption = state->consumption;
  loom_value_relation_iterator_t iterator;
  loom_value_relation_iterator_initialize(
      state->module, op,
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
          LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS),
      &iterator);
  loom_value_relation_t relation;
  while (loom_value_relation_iterator_next(&iterator, &relation)) {
    const loom_value_id_t owner =
        loom_verify_storage_owner(consumption, relation.source_value_id);
    IREE_RETURN_IF_ERROR(loom_verify_consumption_grow(
        state, &consumption->identities,
        iree_max(owner, relation.destination_value_id),
        sizeof(loom_verify_storage_identity_t)));
    loom_verify_storage_identity_t* member = loom_verify_storage_identity(
        consumption, relation.destination_value_id);
    // A function list can contain the same immutable function more than once.
    if (member->owner) {
      continue;
    }
    loom_verify_storage_identity_t* root =
        loom_verify_storage_identity(consumption, owner);
    *member = (loom_verify_storage_identity_t){
        .owner = owner + 1,
        .next = root->next,
    };
    root->next = relation.destination_value_id + 1;
  }
  return iree_ok_status();
}

iree_status_t loom_verify_consumption_record(loom_verify_state_t* state,
                                             const loom_op_t* op,
                                             uint16_t operand_index) {
  const loom_value_id_t value = loom_op_const_operands(op)[operand_index];
  if (value == LOOM_VALUE_ID_INVALID || value >= state->module->values.count ||
      !loom_verify_value_is_visible(state, value)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_verify_consumption_initialize(state));
  loom_verify_consumption_t* consumption = state->consumption;
  IREE_RETURN_IF_ERROR(loom_verify_consumption_grow(
      state, &consumption->events, consumption->event_count,
      sizeof(loom_verify_consumption_event_t)));
  *loom_verify_consumption_event(&consumption->events,
                                 consumption->event_count++) =
      (loom_verify_consumption_event_t){
          .use = loom_use_make((loom_op_t*)op, operand_index, 0),
          .owner = loom_verify_storage_owner(consumption, value),
          .region_index = state->region_scope.index,
      };
  return iree_ok_status();
}

static bool loom_verify_consumption_event_less(
    void* context, const loom_verify_consumption_event_t* lhs,
    const loom_verify_consumption_event_t* rhs) {
  (void)context;
  return lhs->owner < rhs->owner;
}
LOOM_DEFINE_ADAPTIVE_SORT_WITH_ACCESSOR(loom_verify_consumption_sort_events,
                                        loom_verify_consumption_event_t,
                                        loom_segmented_storage_t*,
                                        loom_verify_consumption_event, void*,
                                        loom_verify_consumption_event_less)

static bool loom_verify_consumption_region_less(
    loom_verify_consumption_t* consumption, const uint32_t* lhs,
    const uint32_t* rhs) {
  return (uintptr_t)loom_verify_consumption_region(consumption, *lhs)->region <
         (uintptr_t)loom_verify_consumption_region(consumption, *rhs)->region;
}
LOOM_DEFINE_ADAPTIVE_SORT_WITH_ACCESSOR(
    loom_verify_consumption_sort_region_index, uint32_t,
    loom_segmented_storage_t*, loom_verify_consumption_index,
    loom_verify_consumption_t*, loom_verify_consumption_region_less)

static bool loom_verify_consumption_region_later(void* context,
                                                 const uint32_t* lhs,
                                                 const uint32_t* rhs) {
  (void)context;
  return *lhs > *rhs;
}
LOOM_DEFINE_ADAPTIVE_SORT_WITH_ACCESSOR(
    loom_verify_consumption_sort_active_regions, uint32_t,
    loom_segmented_storage_t*, loom_verify_consumption_index, void*,
    loom_verify_consumption_region_later)

static uint32_t loom_verify_consumption_find_region(
    loom_verify_consumption_t* consumption, const loom_region_t* region) {
  uint32_t begin = 0;
  uint32_t end = consumption->region_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const uint32_t index =
        *loom_verify_consumption_index(&consumption->region_index, middle);
    const loom_region_t* candidate =
        loom_verify_consumption_region(consumption, index)->region;
    if (candidate == region) {
      return index;
    }
    if ((uintptr_t)candidate < (uintptr_t)region) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  // Function-scoped verification intentionally omits other module regions.
  return UINT32_MAX;
}

static iree_status_t loom_verify_consumption_activate(
    loom_verify_state_t* state, uint32_t index, loom_value_id_t owner,
    uint32_t definition_region) {
  loom_verify_consumption_t* consumption = state->consumption;
  while (index != UINT32_MAX) {
    loom_verify_consumption_region_t* region =
        loom_verify_consumption_region(consumption, index);
    if (region->active_owner == owner + 1) {
      break;
    }
    IREE_RETURN_IF_ERROR(loom_verify_consumption_grow(
        state, &consumption->active_regions, consumption->active_region_count,
        sizeof(uint32_t)));
    *loom_verify_consumption_index(&consumption->active_regions,
                                   consumption->active_region_count++) = index;
    region->active_owner = owner + 1;
    region->first_block = UINT16_MAX;
    if (index == definition_region) {
      break;
    }
    index = region->parent;
  }
  return iree_ok_status();
}

static loom_verify_consumption_block_t* loom_verify_consumption_block(
    loom_verify_consumption_t* consumption,
    loom_verify_consumption_region_t* region, uint16_t block_index,
    loom_value_id_t owner) {
  loom_verify_consumption_block_t* block = loom_verify_consumption_at(
      &consumption->blocks, region->block_base + block_index,
      sizeof(loom_verify_consumption_block_t));
  if (block->active_owner != owner + 1) {
    *block = (loom_verify_consumption_block_t){
        .active_owner = owner + 1,
        .next_block = region->first_block,
    };
    region->first_block = block_index;
  }
  return block;
}

static void loom_verify_consumption_observe(
    loom_verify_consumption_block_t* block, const loom_op_t* anchor,
    loom_use_t use) {
  if (!block->observation_anchor) {
    block->entry_observation = use;
  }
  const bool already_after = block->consumption_anchor &&
                             block->observation_anchor &&
                             block->observation_anchor->block_ordinal >
                                 block->consumption_anchor->block_ordinal;
  const bool newly_after =
      block->consumption_anchor &&
      anchor->block_ordinal > block->consumption_anchor->block_ordinal;
  if (!block->observation_anchor ||
      (already_after
           ? newly_after && anchor->block_ordinal <
                                block->observation_anchor->block_ordinal
           : anchor->block_ordinal >
                 block->observation_anchor->block_ordinal)) {
    block->observation = use;
    block->observation_anchor = anchor;
  }
}

static void loom_verify_consumption_seed(loom_verify_consumption_block_t* block,
                                         const loom_op_t* anchor,
                                         loom_use_t use) {
  if (!block->consumption_anchor ||
      anchor->block_ordinal < block->consumption_anchor->block_ordinal) {
    block->consumption = use;
    block->consumption_anchor = anchor;
  }
}

static void loom_verify_consumption_emit(loom_verify_state_t* state,
                                         loom_use_t use,
                                         loom_use_t consumption) {
  const loom_op_t* use_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  const loom_op_t* consuming_op = loom_use_user_op(consumption);
  const loom_op_vtable_t* consuming_vtable =
      loom_verify_lookup_vtable(state, consuming_op->kind);
  loom_diagnostic_field_ref_t operand_ref =
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, operand_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_verify_value_name(
              state, loom_op_const_operands(use_op)[operand_index])),
          operand_ref),
      loom_param_string(loom_op_vtable_name(consuming_vtable)),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("consumed here"),
      .op = consuming_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = use_op,
      .error = LOOM_ERR_DOMINANCE_002,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  loom_verify_emit_diagnostic(state, &emission);
}

// Necessary reachability bounds over the active owner's observations.
typedef struct loom_verify_consumption_bounds_t {
  // Lowest successor-before-predecessor component containing an observation.
  uint16_t minimum_component;
  // Latest DFS preorder containing an observation.
  uint16_t maximum_preorder;
} loom_verify_consumption_bounds_t;

static iree_status_t loom_verify_consumption_push_successors(
    loom_verify_state_t* state, loom_verify_consumption_region_t* region,
    uint16_t block_index, loom_value_id_t owner, const loom_block_t* definition,
    loom_verify_consumption_bounds_t bounds, loom_use_t witness) {
  loom_verify_consumption_t* consumption = state->consumption;
  const loom_cfg_block_index_span_t successors =
      loom_cfg_graph_successors(region->graph, block_index);
  for (iree_host_size_t i = 0; i < successors.count; ++i) {
    const uint16_t successor = successors.values[i];
    const loom_cfg_block_info_t* info = &region->graph->blocks[successor];
    if (info->block == definition || !info->reachable ||
        info->component < bounds.minimum_component ||
        region->graph->blocks[info->reachability_root].preorder >
            bounds.maximum_preorder) {
      continue;
    }
    loom_verify_consumption_block_t* block =
        loom_verify_consumption_block(consumption, region, successor, owner);
    if (block->queued) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_verify_consumption_grow(
        state, &consumption->frontier, consumption->frontier_count,
        sizeof(uint16_t)));
    *(uint16_t*)loom_verify_consumption_at(&consumption->frontier,
                                           consumption->frontier_count++,
                                           sizeof(uint16_t)) = successor;
    block->queued = true;
    block->consumption = witness;
  }
  return iree_ok_status();
}

static iree_status_t loom_verify_consumption_check_region(
    loom_verify_state_t* state, uint32_t index, loom_value_id_t owner,
    const loom_block_t* definition) {
  loom_verify_consumption_t* consumption = state->consumption;
  loom_verify_consumption_region_t* region =
      loom_verify_consumption_region(consumption, index);
  if (!region->graph &&
      iree_any_bit_set(region->region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &state->arena, sizeof(*region->graph), (void**)&region->graph));
    IREE_RETURN_IF_ERROR(loom_cfg_graph_build(state->module, region->region,
                                              &state->arena, region->graph));
  }
  const loom_cfg_graph_t* graph = region->graph;
  loom_use_t observation = {0};
  loom_use_t escaping_consumption = {0};
  loom_verify_consumption_bounds_t bounds = {
      .minimum_component = UINT16_MAX,
  };
  consumption->frontier_count = 0;
  const uint16_t first_block = region->first_block;
  for (uint16_t b = first_block; b != UINT16_MAX;) {
    loom_verify_consumption_block_t* block =
        loom_verify_consumption_block(consumption, region, b, owner);
    if (block->consumption_anchor && block->observation_anchor &&
        block->observation_anchor->block_ordinal >
            block->consumption_anchor->block_ordinal) {
      loom_verify_consumption_emit(state, block->observation,
                                   block->consumption);
      return iree_ok_status();
    }
    if (!graph || graph->blocks[b].reachable) {
      if (block->observation_anchor) {
        observation = block->entry_observation;
        if (graph) {
          bounds.minimum_component =
              iree_min(bounds.minimum_component, graph->blocks[b].component);
          bounds.maximum_preorder =
              iree_max(bounds.maximum_preorder, graph->blocks[b].preorder);
        }
      }
      if (block->consumption_anchor &&
          (!graph || graph->blocks[b].can_reach_exit)) {
        escaping_consumption = block->consumption;
      }
    }
    b = block->next_block;
  }
  if (graph && !graph->malformed) {
    // Seed every source before expanding the union. A block/edge is processed
    // at most once for this owner, regardless of its number of consumers.
    // Retained component order and earliest reachable DFS preorder bound the
    // observed frontier. Independent local owners and disjoint branch uses
    // never force traversal of an unobserved suffix, in either DFS order.
    for (uint16_t b = first_block; b != UINT16_MAX;) {
      loom_verify_consumption_block_t* block =
          loom_verify_consumption_block(consumption, region, b, owner);
      if (block->consumption_anchor && graph->blocks[b].reachable) {
        IREE_RETURN_IF_ERROR(loom_verify_consumption_push_successors(
            state, region, b, owner, definition, bounds, block->consumption));
      }
      b = block->next_block;
    }
    for (iree_host_size_t i = 0; i < consumption->frontier_count; ++i) {
      const uint16_t b = *(uint16_t*)loom_verify_consumption_at(
          &consumption->frontier, i, sizeof(uint16_t));
      loom_verify_consumption_block_t* block =
          loom_verify_consumption_block(consumption, region, b, owner);
      if (block->observation_anchor) {
        loom_verify_consumption_emit(state, block->entry_observation,
                                     block->consumption);
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_verify_consumption_push_successors(
          state, region, b, owner, definition, bounds, block->consumption));
    }
  }
  if (region->region == definition->parent_region ||
      region->parent == UINT32_MAX) {
    return iree_ok_status();
  }
  if (loom_use_user_op(escaping_consumption) &&
      region->execution == LOOM_REGION_EXECUTION_REPEATED) {
    loom_verify_consumption_emit(state, escaping_consumption,
                                 escaping_consumption);
    return iree_ok_status();
  }
  loom_verify_consumption_region_t* parent =
      loom_verify_consumption_region(consumption, region->parent);
  loom_verify_consumption_block_t* parent_block = loom_verify_consumption_block(
      consumption, parent, region->owner->parent_block->region_index, owner);
  if (loom_use_user_op(observation)) {
    loom_verify_consumption_observe(parent_block, region->owner, observation);
  }
  if (loom_use_user_op(escaping_consumption) &&
      region->execution != LOOM_REGION_EXECUTION_EXIT) {
    loom_verify_consumption_seed(parent_block, region->owner,
                                 escaping_consumption);
  }
  return iree_ok_status();
}

static iree_status_t loom_verify_consumption_check_owner(
    loom_verify_state_t* state, iree_host_size_t begin, iree_host_size_t end) {
  loom_verify_consumption_t* consumption = state->consumption;
  const loom_value_id_t owner =
      loom_verify_consumption_event(&consumption->events, begin)->owner;
  const loom_value_t* value = loom_module_value(state->module, owner);
  const loom_block_t* definition = loom_value_is_block_arg(value)
                                       ? loom_value_def_block(value)
                                       : loom_value_def_op(value)->parent_block;
  const uint32_t definition_region = loom_verify_consumption_find_region(
      consumption, definition->parent_region);
  consumption->active_region_count = 0;
  for (iree_host_size_t i = begin; i < end; ++i) {
    const loom_verify_consumption_event_t* event =
        loom_verify_consumption_event(&consumption->events, i);
    IREE_RETURN_IF_ERROR(loom_verify_consumption_activate(
        state, event->region_index, owner, definition_region));
    loom_verify_consumption_region_t* region =
        loom_verify_consumption_region(consumption, event->region_index);
    const loom_op_t* op = loom_use_user_op(event->use);
    loom_verify_consumption_block_t* block = loom_verify_consumption_block(
        consumption, region, op->parent_block->region_index, owner);
    if (block->consumption_anchor == op &&
        loom_use_operand_index(block->consumption) !=
            loom_use_operand_index(event->use)) {
      const uint16_t first_index = loom_use_operand_index(block->consumption);
      const uint16_t second_index = loom_use_operand_index(event->use);
      loom_diagnostic_param_t params[] = {
          loom_param_with_field_ref(
              loom_param_u32(first_index),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                        first_index)),
          loom_param_with_field_ref(
              loom_param_u32(second_index),
              loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                        second_index)),
          loom_param_string(
              loom_op_vtable_name(loom_verify_lookup_vtable(state, op->kind))),
          loom_param_string(loom_verify_value_name(state, owner)),
      };
      loom_verify_emit_structured(state, op, LOOM_ERR_DOMINANCE_019, params,
                                  IREE_ARRAYSIZE(params));
      return iree_ok_status();
    }
    loom_verify_consumption_seed(block, op, event->use);
  }
  uint32_t member = owner + 1;
  while (member) {
    const loom_value_t* member_value =
        loom_module_value(state->module, member - 1);
    const loom_use_t* use = NULL;
    loom_value_for_each_use(member_value, use) {
      const loom_op_t* op = loom_use_user_op(*use);
      if (!op->parent_block) {
        continue;
      }
      const uint32_t region_index = loom_verify_consumption_find_region(
          consumption, op->parent_block->parent_region);
      if (region_index == UINT32_MAX) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_verify_consumption_activate(
          state, region_index, owner, definition_region));
      loom_verify_consumption_region_t* region =
          loom_verify_consumption_region(consumption, region_index);
      loom_verify_consumption_observe(
          loom_verify_consumption_block(consumption, region,
                                        op->parent_block->region_index, owner),
          op, *use);
    }
    const loom_verify_storage_identity_t* identity =
        loom_verify_storage_identity(consumption, member - 1);
    member = identity ? identity->next : 0;
  }
  loom_verify_consumption_sort_active_regions(
      NULL, &consumption->active_regions, consumption->active_region_count);
  const uint32_t initial_errors = state->result->error_count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < consumption->active_region_count &&
       state->result->error_count == initial_errors;
       ++i) {
    const uint32_t index =
        *loom_verify_consumption_index(&consumption->active_regions, i);
    status =
        loom_verify_consumption_check_region(state, index, owner, definition);
  }
  return status;
}

iree_status_t loom_verify_consumption_check(loom_verify_state_t* state) {
  loom_verify_consumption_t* consumption = state->consumption;
  if (!consumption || !consumption->event_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_verify_consumption_grow(
      state, &consumption->region_index, consumption->region_count - 1,
      sizeof(uint32_t)));
  IREE_RETURN_IF_ERROR(loom_verify_consumption_grow(
      state, &consumption->blocks, consumption->block_count - 1,
      sizeof(loom_verify_consumption_block_t)));
  for (uint32_t i = 0; i < consumption->region_count; ++i) {
    *loom_verify_consumption_index(&consumption->region_index, i) = i;
  }
  loom_verify_consumption_sort_region_index(
      consumption, &consumption->region_index, consumption->region_count);
  loom_verify_consumption_sort_events(NULL, &consumption->events,
                                      consumption->event_count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t begin = 0; iree_status_is_ok(status) &&
                                   begin < consumption->event_count &&
                                   !loom_verify_at_error_limit(state);) {
    iree_host_size_t end = begin + 1;
    const loom_value_id_t owner =
        loom_verify_consumption_event(&consumption->events, begin)->owner;
    while (end < consumption->event_count &&
           loom_verify_consumption_event(&consumption->events, end)->owner ==
               owner) {
      ++end;
    }
    status = loom_verify_consumption_check_owner(state, begin, end);
    begin = end;
  }
  if (iree_status_is_ok(status)) {
    status = loom_verify_pending_diagnostic_status(state);
  }
  return status;
}
