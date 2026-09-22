// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_facts.h"

#include <string.h>

#include "loom/analysis/cfg_condition_operand_domain.h"
#include "loom/analysis/cfg_condition_relation_table.h"
#include "loom/ops/cfg/ops.h"
#include "loom/util/adaptive_sort.h"

//===----------------------------------------------------------------------===//
// Complete finite relation propagation
//===----------------------------------------------------------------------===//

#define LOOM_CFG_CONDITION_VIEW_INVALID UINT32_MAX

typedef struct loom_cfg_condition_mapping_pair_t {
  // Source operand or local-value ordinal.
  uint32_t source;

  // Target operand or local-value ordinal.
  uint32_t target;
} loom_cfg_condition_mapping_pair_t;

typedef struct loom_cfg_condition_mapping_t {
  // Unique pairs sorted by source and then target.
  const loom_cfg_condition_mapping_pair_t* pairs;

  // Number of entries in pairs.
  uint32_t pair_count;
} loom_cfg_condition_mapping_t;

typedef struct loom_cfg_condition_truth_t {
  // Values known false and true, respectively.
  loom_condition_relation_set_id_t values[2];
} loom_cfg_condition_truth_t;

typedef struct loom_cfg_condition_matrix_state_t {
  // Candidate rows, possibly shared with another propagation state.
  loom_condition_relation_matrix_t matrix;

  // True when another state may observe the matrix storage. The rows must be
  // copied before modification.
  bool shared;
} loom_cfg_condition_matrix_state_t;

typedef struct loom_cfg_condition_edge_state_t {
  // Borrowed stable CFG edge metadata.
  const loom_cfg_edge_info_t* cfg_edge;

  // Borrowed target block.
  const loom_block_t* target_block;

  // Complete raw relations derived from the edge selector.
  const loom_condition_integer_relation_t* raw_relations;

  // Complete raw Boolean truths derived from the edge selector.
  const loom_condition_boolean_fact_t* raw_boolean_facts;

  // Explicit source-to-target payload images.
  loom_cfg_condition_mapping_t payload_mapping;

  // Edge-local assertions after target-entry projection.
  loom_condition_relation_matrix_t assertions;

  // Edge-local exact truth after target-entry projection.
  loom_cfg_condition_truth_t truth_assertions;

  // Current projected contribution for a join predecessor.
  loom_cfg_condition_matrix_state_t contribution;

  // Current exact-truth contribution for a join predecessor.
  loom_cfg_condition_truth_t truth_contribution;

  // Number of entries in raw_relations.
  uint32_t raw_relation_count;

  // Number of entries in raw_boolean_facts.
  uint32_t raw_boolean_fact_count;

  // Stable graph edge ordinal.
  loom_cfg_edge_index_t edge_index;

  // Dense source block index.
  uint16_t source;

  // Dense target block index.
  uint16_t target;

  // True for a reachable in-region edge.
  bool active;

  // True when this contribution seeded the reverse-postorder meet.
  bool included_in_initial_meet;

  // True when the edge contribution aliases its single-predecessor target.
  bool shares_target_facts;
} loom_cfg_condition_edge_state_t;

typedef struct loom_cfg_condition_block_state_t {
  // Current monotone integer-relation facts.
  loom_cfg_condition_matrix_state_t integer_relations;

  // Current monotone exact-Boolean facts.
  loom_cfg_condition_truth_t truth;
} loom_cfg_condition_block_state_t;

typedef struct loom_cfg_condition_projection_cache_slot_t {
  // Stable edge ordinal, or UINT32_MAX for an empty slot.
  uint32_t edge;

  // Source construction-set ID.
  loom_condition_relation_set_id_t source;

  // Projected construction-set ID.
  loom_condition_relation_set_id_t target;
} loom_cfg_condition_projection_cache_slot_t;

typedef struct loom_cfg_condition_projection_cache_t {
  // Scratch arena owning every slot-table generation.
  iree_arena_allocator_t* arena;

  // Open-addressed cache slots.
  loom_cfg_condition_projection_cache_slot_t* slots;

  // Power-of-two number of entries in slots.
  uint32_t capacity;

  // Number of populated entries.
  uint32_t count;
} loom_cfg_condition_projection_cache_t;

typedef uint8_t loom_cfg_condition_event_kind_t;
enum loom_cfg_condition_event_kind_e {
  LOOM_CFG_CONDITION_EVENT_INTEGER = 0,
  LOOM_CFG_CONDITION_EVENT_BOOLEAN = 1,
};

typedef struct loom_cfg_condition_event_t {
  // Row ordinal within the changed block.
  uint32_t row;

  // Dense changed block index.
  uint16_t block;

  // Integer outcome or Boolean result ordinal.
  uint8_t outcome;

  // Integer-row or Boolean-set event category.
  loom_cfg_condition_event_kind_t kind;
} loom_cfg_condition_event_t;

typedef struct loom_cfg_condition_event_queue_t {
  // Scratch arena owning dynamic event storage.
  iree_arena_allocator_t* arena;

  // Prefix sum mapping block-local rows to global row ordinals.
  uint32_t* block_row_offsets;

  // Per-global-row masks of queued integer outcomes.
  uint8_t* integer_queued_masks;

  // Per-block masks of queued Boolean outcomes.
  uint8_t* boolean_queued_masks;

  // Contiguous storage for currently queued events.
  loom_cfg_condition_event_t* events;

  // Number of entries in block_row_offsets minus one.
  uint16_t block_count;

  // Allocated entry count in events.
  iree_host_size_t capacity;

  // First queued event in events.
  iree_host_size_t head;

  // Number of queued events.
  iree_host_size_t count;
} loom_cfg_condition_event_queue_t;

typedef struct loom_cfg_condition_relation_solver_t {
  // Borrowed immutable compiler inputs.
  const loom_module_t* module;
  const loom_cfg_graph_t* graph;
  const loom_value_fact_table_t* fact_table;
  const loom_dominance_info_t* dominance;
  loom_local_value_domain_t* value_domain;
  const loom_cfg_value_identity_table_t* identities;

  // Arena retained by the caller after solve completion.
  iree_arena_allocator_t* arena;

  // Child arena released immediately after publication.
  iree_arena_allocator_t* scratch_arena;

  // Compact scratch operand domain used by set and matrix construction.
  loom_cfg_condition_operand_domain_t operand_domain;

  // Direct local-value-ordinal to relation-operand mapping.
  loom_cfg_condition_operand_t* relation_operands;

  // Interned construction sets.
  loom_condition_relation_set_builder_t* set_builder;

  // Reusable high-water matrix candidate storage.
  loom_condition_relation_matrix_builder_t matrix_builder;

  // Per-edge construction and propagation state.
  loom_cfg_condition_edge_state_t* edges;

  // Per-block monotone propagation state.
  loom_cfg_condition_block_state_t* blocks;

  // Cache of immutable edge/set projections.
  loom_cfg_condition_projection_cache_t projection_cache;

  // Last queried target plus one in bits 1..31 and its availability in bit 0
  // for each value operand. Zero denotes an empty entry.
  uint32_t* value_availability_cache;

  // Reusable set-projection member storage.
  uint32_t* projection_values;

  // Number of populated entries in projection_values.
  iree_host_size_t projection_value_count;

  // Allocated entry count in projection_values.
  iree_host_size_t projection_value_capacity;

  // Finite coalescing propagation queue.
  loom_cfg_condition_event_queue_t pending;

  // True when edge derivation produced any path-sensitive facts.
  bool has_derived_facts;
} loom_cfg_condition_relation_solver_t;

static bool loom_cfg_condition_mapping_pair_less(
    const loom_cfg_condition_mapping_pair_t* left,
    const loom_cfg_condition_mapping_pair_t* right) {
  return left->source < right->source ||
         (left->source == right->source && left->target < right->target);
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_cfg_condition_sort_mapping_pairs,
                          loom_cfg_condition_mapping_pair_t,
                          loom_cfg_condition_mapping_pair_less)

static bool loom_cfg_condition_i64_less(const int64_t* left,
                                        const int64_t* right) {
  return *left < *right;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_cfg_condition_sort_i64, int64_t,
                          loom_cfg_condition_i64_less)

static uint32_t loom_cfg_condition_projection_hash(
    loom_cfg_edge_index_t edge, loom_condition_relation_set_id_t source) {
  uint64_t value = ((uint64_t)edge << 32) | source;
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  return (uint32_t)value;
}

static bool loom_cfg_condition_projection_cache_lookup(
    const loom_cfg_condition_projection_cache_t* cache,
    loom_cfg_edge_index_t edge, loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t* out_target) {
  if (cache->capacity == 0) {
    return false;
  }
  uint32_t slot =
      loom_cfg_condition_projection_hash(edge, source) & (cache->capacity - 1);
  while (cache->slots[slot].edge != UINT32_MAX) {
    if (cache->slots[slot].edge == edge &&
        cache->slots[slot].source == source) {
      *out_target = cache->slots[slot].target;
      return true;
    }
    slot = (slot + 1) & (cache->capacity - 1);
  }
  return false;
}

static iree_status_t loom_cfg_condition_projection_cache_reserve(
    loom_cfg_condition_projection_cache_t* cache) {
  uint32_t capacity = cache->capacity == 0 ? 64 : cache->capacity;
  while ((uint64_t)(cache->count + 1) * 4 > (uint64_t)capacity * 3) {
    if (capacity > UINT32_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition projection cache exceeds uint32_t");
    }
    capacity *= 2;
  }
  if (capacity == cache->capacity) {
    return iree_ok_status();
  }

  loom_cfg_condition_projection_cache_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      cache->arena, capacity, sizeof(*slots), (void**)&slots));
  memset(slots, 0xFF, capacity * sizeof(*slots));
  for (uint32_t i = 0; i < cache->capacity; ++i) {
    const loom_cfg_condition_projection_cache_slot_t entry = cache->slots[i];
    if (entry.edge == UINT32_MAX) {
      continue;
    }
    uint32_t slot =
        loom_cfg_condition_projection_hash(entry.edge, entry.source) &
        (capacity - 1);
    while (slots[slot].edge != UINT32_MAX) {
      slot = (slot + 1) & (capacity - 1);
    }
    slots[slot] = entry;
  }
  cache->slots = slots;
  cache->capacity = capacity;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_projection_cache_insert(
    loom_cfg_condition_projection_cache_t* cache, loom_cfg_edge_index_t edge,
    loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t target) {
  IREE_RETURN_IF_ERROR(loom_cfg_condition_projection_cache_reserve(cache));
  uint32_t slot =
      loom_cfg_condition_projection_hash(edge, source) & (cache->capacity - 1);
  while (cache->slots[slot].edge != UINT32_MAX) {
    slot = (slot + 1) & (cache->capacity - 1);
  }
  cache->slots[slot] = (loom_cfg_condition_projection_cache_slot_t){
      .edge = edge,
      .source = source,
      .target = target,
  };
  ++cache->count;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_event_queue_initialize(
    uint16_t block_count, const loom_cfg_condition_block_state_t* blocks,
    iree_arena_allocator_t* arena,
    loom_cfg_condition_event_queue_t* out_queue) {
  *out_queue = (loom_cfg_condition_event_queue_t){
      .arena = arena,
      .block_count = block_count,
  };
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, (iree_host_size_t)block_count + 1,
                                sizeof(*out_queue->block_row_offsets),
                                (void**)&out_queue->block_row_offsets));
  uint64_t total_row_count = 0;
  out_queue->block_row_offsets[0] = 0;
  for (uint16_t block = 0; block < block_count; ++block) {
    total_row_count += blocks[block].integer_relations.matrix.row_count;
    if (total_row_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition event rows exceed uint32_t");
    }
    out_queue->block_row_offsets[block + 1] = (uint32_t)total_row_count;
  }
  if (total_row_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, (iree_host_size_t)total_row_count,
                                  sizeof(*out_queue->integer_queued_masks),
                                  (void**)&out_queue->integer_queued_masks));
    memset(out_queue->integer_queued_masks, 0,
           (iree_host_size_t)total_row_count);
  }
  if (block_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, block_count, sizeof(*out_queue->boolean_queued_masks),
        (void**)&out_queue->boolean_queued_masks));
    memset(out_queue->boolean_queued_masks, 0, block_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_event_queue_append(
    loom_cfg_condition_event_queue_t* queue, loom_cfg_condition_event_t event) {
  if (queue->head + queue->count == queue->capacity && queue->head != 0) {
    memmove(queue->events, &queue->events[queue->head],
            queue->count * sizeof(*queue->events));
    queue->head = 0;
  }
  if (queue->count == queue->capacity) {
    const iree_host_size_t minimum_capacity =
        iree_max((iree_host_size_t)64, queue->count + 1);
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        queue->arena, queue->count, minimum_capacity, sizeof(*queue->events),
        &queue->capacity, (void**)&queue->events));
  }
  queue->events[queue->head + queue->count] = event;
  ++queue->count;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_event_queue_enqueue_integer(
    loom_cfg_condition_event_queue_t* queue, uint16_t block, uint32_t row,
    loom_condition_relation_outcome_t outcome) {
  const uint32_t global_row = queue->block_row_offsets[block] + row;
  const uint8_t bit = (uint8_t)(1u << outcome);
  if ((queue->integer_queued_masks[global_row] & bit) != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_append(
      queue, (loom_cfg_condition_event_t){
                 .row = row,
                 .block = block,
                 .outcome = outcome,
                 .kind = LOOM_CFG_CONDITION_EVENT_INTEGER,
             }));
  queue->integer_queued_masks[global_row] |= bit;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_event_queue_enqueue_boolean(
    loom_cfg_condition_event_queue_t* queue, uint16_t block, bool value) {
  const uint8_t outcome = value ? 1 : 0;
  const uint8_t bit = (uint8_t)(1u << outcome);
  if ((queue->boolean_queued_masks[block] & bit) != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_append(
      queue, (loom_cfg_condition_event_t){
                 .block = block,
                 .outcome = outcome,
                 .kind = LOOM_CFG_CONDITION_EVENT_BOOLEAN,
             }));
  queue->boolean_queued_masks[block] |= bit;
  return iree_ok_status();
}

static bool loom_cfg_condition_event_queue_pop(
    loom_cfg_condition_event_queue_t* queue,
    loom_cfg_condition_event_t* out_event) {
  if (queue->count == 0) {
    return false;
  }
  const loom_cfg_condition_event_t event = queue->events[queue->head];
  ++queue->head;
  --queue->count;
  if (queue->count == 0) {
    queue->head = 0;
  }
  const uint8_t bit = (uint8_t)(1u << event.outcome);
  if (event.kind == LOOM_CFG_CONDITION_EVENT_INTEGER) {
    const uint32_t global_row =
        queue->block_row_offsets[event.block] + event.row;
    queue->integer_queued_masks[global_row] &= (uint8_t)~bit;
  } else {
    queue->boolean_queued_masks[event.block] &= (uint8_t)~bit;
  }
  *out_event = event;
  return true;
}

static void loom_cfg_condition_mapping_initialize(
    loom_cfg_condition_mapping_pair_t* pairs, iree_host_size_t pair_count,
    loom_cfg_condition_mapping_t* out_mapping) {
  loom_cfg_condition_sort_mapping_pairs(pairs, pair_count);
  iree_host_size_t unique_count = 0;
  for (iree_host_size_t i = 0; i < pair_count; ++i) {
    if (unique_count == 0 ||
        pairs[unique_count - 1].source != pairs[i].source ||
        pairs[unique_count - 1].target != pairs[i].target) {
      pairs[unique_count++] = pairs[i];
    }
  }
  *out_mapping = (loom_cfg_condition_mapping_t){
      .pairs = pairs,
      .pair_count = (uint32_t)unique_count,
  };
}

static const loom_cfg_condition_mapping_pair_t*
loom_cfg_condition_mapping_lookup(const loom_cfg_condition_mapping_t* mapping,
                                  uint32_t source, uint32_t* out_count) {
  uint32_t begin = 0;
  uint32_t end = mapping->pair_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    if (mapping->pairs[middle].source < source) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  const uint32_t first = begin;
  end = mapping->pair_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    if (mapping->pairs[middle].source <= source) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  *out_count = begin - first;
  return mapping->pairs != NULL ? mapping->pairs + first : NULL;
}

static loom_condition_relation_outcome_bits_t
loom_cfg_condition_relation_exclusions(
    loom_symbolic_integer_relation_t relation) {
  switch (relation) {
    case LOOM_SYMBOLIC_INTEGER_RELATION_EQ:
      return LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS |
             LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
    case LOOM_SYMBOLIC_INTEGER_RELATION_NE:
      return LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LT:
      return LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL |
             LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
    case LOOM_SYMBOLIC_INTEGER_RELATION_LE:
      return LOOM_CONDITION_RELATION_OUTCOME_BIT_GREATER;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GT:
      return LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS |
             LOOM_CONDITION_RELATION_OUTCOME_BIT_EQUAL;
    case LOOM_SYMBOLIC_INTEGER_RELATION_GE:
      return LOOM_CONDITION_RELATION_OUTCOME_BIT_LESS;
    default:
      return 0;
  }
}

static loom_condition_relation_outcome_t loom_cfg_condition_swap_outcome(
    loom_condition_relation_outcome_t outcome) {
  if (outcome == LOOM_CONDITION_RELATION_OUTCOME_LESS) {
    return LOOM_CONDITION_RELATION_OUTCOME_GREATER;
  }
  if (outcome == LOOM_CONDITION_RELATION_OUTCOME_GREATER) {
    return LOOM_CONDITION_RELATION_OUTCOME_LESS;
  }
  return outcome;
}

static bool loom_cfg_condition_relation_is_boolean_truth(
    const loom_module_t* module,
    const loom_condition_integer_relation_t* relation) {
  if (relation->relation != LOOM_SYMBOLIC_INTEGER_RELATION_EQ) {
    return false;
  }
  loom_condition_integer_operand_t value = relation->left;
  loom_condition_integer_operand_t constant = relation->right;
  if (value.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
    const loom_condition_integer_operand_t temporary = value;
    value = constant;
    constant = temporary;
  }
  if (value.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      constant.kind != LOOM_CONDITION_INTEGER_OPERAND_CONSTANT ||
      (constant.constant != 0 && constant.constant != 1) ||
      value.value_id >= module->values.count) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(module, value.value_id);
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static iree_status_t loom_cfg_condition_relation_derive_edges(
    loom_cfg_condition_relation_solver_t* solver) {
  const loom_cfg_graph_t* graph = solver->graph;
  if (graph->edge_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, graph->edge_count, sizeof(*solver->edges),
        (void**)&solver->edges));
    memset(solver->edges, 0, graph->edge_count * sizeof(*solver->edges));
  }

  loom_condition_query_t query;
  loom_condition_query_initialize(solver->module, solver->value_domain,
                                  solver->scratch_arena, &query);
  loom_condition_derivation_t derivation;
  loom_condition_derivation_initialize(solver->scratch_arena, &derivation);
  for (loom_cfg_edge_index_t edge_index = 0; edge_index < graph->edge_count;
       ++edge_index) {
    const loom_cfg_edge_info_t* cfg_edge = &graph->edges[edge_index];
    if (!loom_cfg_graph_block_is_reachable(graph,
                                           cfg_edge->source_block_index) ||
        !loom_cfg_graph_block_is_reachable(graph,
                                           cfg_edge->target_block_index)) {
      continue;
    }
    loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    *edge = (loom_cfg_condition_edge_state_t){
        .cfg_edge = cfg_edge,
        .target_block = graph->blocks[cfg_edge->target_block_index].block,
        .edge_index = edge_index,
        .source = cfg_edge->source_block_index,
        .target = cfg_edge->target_block_index,
        .active = true,
    };
    if (!loom_cfg_cond_br_isa(cfg_edge->terminator) ||
        cfg_edge->successor_index >= 2) {
      continue;
    }
    const loom_value_id_t condition =
        loom_cfg_cond_br_condition(cfg_edge->terminator);
    const bool assumed_truth = cfg_edge->successor_index == 0;
    IREE_RETURN_IF_ERROR(loom_condition_facts_query_complete(
        &query, solver->fact_table, condition, assumed_truth, &derivation));
    if (derivation.integer_facts.integer_relation_count > UINT32_MAX ||
        derivation.boolean_fact_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition derivation exceeds uint32_t");
    }
    if (derivation.integer_facts.integer_relation_count != 0) {
      loom_condition_integer_relation_t* relations = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          solver->scratch_arena,
          derivation.integer_facts.integer_relation_count, sizeof(*relations),
          (void**)&relations));
      memcpy(
          relations, derivation.integer_facts.integer_relations,
          derivation.integer_facts.integer_relation_count * sizeof(*relations));
      edge->raw_relations = relations;
      edge->raw_relation_count =
          (uint32_t)derivation.integer_facts.integer_relation_count;
    }
    if (derivation.boolean_fact_count != 0) {
      loom_condition_boolean_fact_t* boolean_facts = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          solver->scratch_arena, derivation.boolean_fact_count,
          sizeof(*boolean_facts), (void**)&boolean_facts));
      memcpy(boolean_facts, derivation.boolean_facts,
             derivation.boolean_fact_count * sizeof(*boolean_facts));
      edge->raw_boolean_facts = boolean_facts;
      edge->raw_boolean_fact_count = (uint32_t)derivation.boolean_fact_count;
    }
    solver->has_derived_facts |=
        edge->raw_relation_count != 0 || edge->raw_boolean_fact_count != 0;
  }
  return iree_ok_status();
}

typedef struct loom_cfg_condition_operand_builder_t {
  // Solver receiving the completed compact domain.
  loom_cfg_condition_relation_solver_t* solver;

  // Mark bit indexed by local value ordinal.
  uint8_t* marked_values;

  // Unsorted literal constants.
  int64_t* constants;

  // Number of populated entries in constants.
  iree_host_size_t constant_count;

  // Allocated entry count in constants.
  iree_host_size_t constant_capacity;
} loom_cfg_condition_operand_builder_t;

static loom_value_ordinal_t loom_cfg_condition_operand_builder_value_ordinal(
    const loom_cfg_condition_operand_builder_t* builder,
    loom_value_id_t value_id) {
  const loom_value_id_t canonical = loom_cfg_value_identity_table_lookup(
      builder->solver->identities, value_id);
  return loom_local_value_domain_ordinal(builder->solver->value_domain,
                                         canonical);
}

static loom_value_ordinal_t loom_cfg_condition_operand_builder_mark_value(
    loom_cfg_condition_operand_builder_t* builder, loom_value_id_t value_id) {
  const loom_value_ordinal_t ordinal =
      loom_cfg_condition_operand_builder_value_ordinal(builder, value_id);
  builder->marked_values[ordinal] = 1;
  return ordinal;
}

static iree_status_t loom_cfg_condition_operand_builder_add_constant(
    loom_cfg_condition_operand_builder_t* builder, int64_t constant) {
  if (builder->constant_count >= builder->constant_capacity) {
    const iree_host_size_t minimum_capacity =
        builder->constant_capacity == 0 ? 16 : builder->constant_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->solver->scratch_arena, builder->constant_count,
        minimum_capacity, sizeof(*builder->constants),
        &builder->constant_capacity, (void**)&builder->constants));
  }
  builder->constants[builder->constant_count++] = constant;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_operand_builder_add_operand(
    loom_cfg_condition_operand_builder_t* builder,
    loom_condition_integer_operand_t operand) {
  if (operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_cfg_condition_operand_builder_mark_value(builder, operand.value_id);
    return iree_ok_status();
  }
  return loom_cfg_condition_operand_builder_add_constant(builder,
                                                         operand.constant);
}

static iree_status_t loom_cfg_condition_collect_forwarding_pairs(
    loom_cfg_condition_operand_builder_t* builder,
    loom_cfg_condition_mapping_pair_t** out_pairs,
    iree_host_size_t* out_pair_count) {
  *out_pairs = NULL;
  *out_pair_count = 0;
  const loom_cfg_condition_relation_solver_t* solver = builder->solver;
  iree_host_size_t pair_capacity = 0;
  for (loom_cfg_edge_index_t edge_index = 0;
       edge_index < solver->graph->edge_count; ++edge_index) {
    const loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    if (!edge->active) {
      continue;
    }
    const loom_value_id_t* payload = NULL;
    uint16_t payload_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(edge->cfg_edge->terminator,
                                                   edge->target_block, &payload,
                                                   &payload_count)) {
      continue;
    }
    const uint16_t count =
        iree_min(payload_count, edge->target_block->arg_count);
    if (*out_pair_count + count > pair_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          solver->scratch_arena, *out_pair_count, *out_pair_count + count,
          sizeof(**out_pairs), &pair_capacity, (void**)out_pairs));
    }
    for (uint16_t i = 0; i < count; ++i) {
      const loom_value_ordinal_t source =
          loom_cfg_condition_operand_builder_value_ordinal(builder, payload[i]);
      const loom_value_id_t target_value =
          loom_block_arg_id(edge->target_block, i);
      const loom_value_ordinal_t target =
          loom_cfg_condition_operand_builder_value_ordinal(builder,
                                                           target_value);
      (*out_pairs)[(*out_pair_count)++] = (loom_cfg_condition_mapping_pair_t){
          .source = source,
          .target = target,
      };
    }
  }
  loom_cfg_condition_sort_mapping_pairs(*out_pairs, *out_pair_count);
  iree_host_size_t unique_count = 0;
  for (iree_host_size_t i = 0; i < *out_pair_count; ++i) {
    if (unique_count == 0 ||
        (*out_pairs)[unique_count - 1].source != (*out_pairs)[i].source ||
        (*out_pairs)[unique_count - 1].target != (*out_pairs)[i].target) {
      (*out_pairs)[unique_count++] = (*out_pairs)[i];
    }
  }
  *out_pair_count = unique_count;
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_cfg_condition_relation_build_operand_domain(
    loom_cfg_condition_relation_solver_t* solver) {
  const loom_value_ordinal_t local_value_count =
      solver->value_domain->value_count;
  loom_cfg_condition_operand_builder_t builder = {
      .solver = solver,
  };
  if (local_value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, local_value_count,
        sizeof(*builder.marked_values), (void**)&builder.marked_values));
    memset(
        builder.marked_values, 0,
        (iree_host_size_t)local_value_count * sizeof(*builder.marked_values));
  }

  for (loom_cfg_edge_index_t edge_index = 0;
       edge_index < solver->graph->edge_count; ++edge_index) {
    const loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    if (!edge->active) {
      continue;
    }
    for (uint32_t i = 0; i < edge->raw_relation_count; ++i) {
      const loom_condition_integer_relation_t* relation =
          &edge->raw_relations[i];
      if (loom_cfg_condition_relation_is_boolean_truth(solver->module,
                                                       relation)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_cfg_condition_operand_builder_add_operand(
          &builder, relation->left));
      IREE_RETURN_IF_ERROR(loom_cfg_condition_operand_builder_add_operand(
          &builder, relation->right));
    }
    for (uint32_t i = 0; i < edge->raw_boolean_fact_count; ++i) {
      loom_cfg_condition_operand_builder_mark_value(
          &builder, edge->raw_boolean_facts[i].value_id);
    }
  }

  loom_cfg_condition_mapping_pair_t* forwarding_pairs = NULL;
  iree_host_size_t forwarding_pair_count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_condition_collect_forwarding_pairs(
      &builder, &forwarding_pairs, &forwarding_pair_count));
  uint32_t* forwarding_offsets = NULL;
  if (local_value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, (iree_host_size_t)local_value_count + 1,
        sizeof(*forwarding_offsets), (void**)&forwarding_offsets));
  }
  iree_host_size_t pair_position = 0;
  for (loom_value_ordinal_t source = 0; source < local_value_count; ++source) {
    forwarding_offsets[source] = (uint32_t)pair_position;
    while (pair_position < forwarding_pair_count &&
           forwarding_pairs[pair_position].source == source) {
      ++pair_position;
    }
  }
  if (forwarding_offsets != NULL) {
    forwarding_offsets[local_value_count] = (uint32_t)pair_position;
  }

  loom_value_ordinal_t* pending_values = NULL;
  if (local_value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, local_value_count, sizeof(*pending_values),
        (void**)&pending_values));
  }
  loom_value_ordinal_t pending_count = 0;
  for (loom_value_ordinal_t value = 0; value < local_value_count; ++value) {
    if (builder.marked_values[value]) {
      pending_values[pending_count++] = value;
    }
  }
  for (loom_value_ordinal_t pending_position = 0;
       pending_position < pending_count; ++pending_position) {
    const loom_value_ordinal_t source = pending_values[pending_position];
    for (uint32_t i = forwarding_offsets[source];
         i < forwarding_offsets[source + 1]; ++i) {
      const loom_value_ordinal_t target = forwarding_pairs[i].target;
      if (!builder.marked_values[target]) {
        builder.marked_values[target] = 1;
        pending_values[pending_count++] = target;
      }
    }
  }

  uint32_t relation_value_count = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < local_value_count;
       ++ordinal) {
    if (!builder.marked_values[ordinal]) {
      continue;
    }
    ++relation_value_count;
    int64_t exact_value = 0;
    if (solver->fact_table != NULL &&
        loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(
                solver->fact_table, solver->value_domain->value_ids[ordinal]),
            &exact_value)) {
      IREE_RETURN_IF_ERROR(loom_cfg_condition_operand_builder_add_constant(
          &builder, exact_value));
    }
  }

  loom_cfg_condition_sort_i64(builder.constants, builder.constant_count);
  iree_host_size_t unique_constant_count = 0;
  for (iree_host_size_t i = 0; i < builder.constant_count; ++i) {
    if (unique_constant_count == 0 ||
        builder.constants[unique_constant_count - 1] != builder.constants[i]) {
      builder.constants[unique_constant_count++] = builder.constants[i];
    }
  }
  if (unique_constant_count > UINT32_MAX - relation_value_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition operand domain exceeds uint32_t");
  }

  loom_value_id_t* relation_values = NULL;
  if (relation_value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, relation_value_count, sizeof(*relation_values),
        (void**)&relation_values));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(solver->scratch_arena, relation_value_count,
                                  sizeof(*solver->value_availability_cache),
                                  (void**)&solver->value_availability_cache));
    memset(solver->value_availability_cache, 0,
           (iree_host_size_t)relation_value_count *
               sizeof(*solver->value_availability_cache));
  }
  if (local_value_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(solver->scratch_arena, local_value_count,
                                  sizeof(*solver->relation_operands),
                                  (void**)&solver->relation_operands));
    memset(solver->relation_operands, 0xFF,
           (iree_host_size_t)local_value_count *
               sizeof(*solver->relation_operands));
  }
  uint32_t relation_value = 0;
  for (loom_value_ordinal_t ordinal = 0; ordinal < local_value_count;
       ++ordinal) {
    if (!builder.marked_values[ordinal]) {
      continue;
    }
    relation_values[relation_value] = solver->value_domain->value_ids[ordinal];
    solver->relation_operands[ordinal] = relation_value++;
  }
  solver->operand_domain = (loom_cfg_condition_operand_domain_t){
      .value_domain = solver->value_domain,
      .identities = solver->identities,
      .values = relation_values,
      .constants = builder.constants,
      .value_count = relation_value_count,
      .constant_count = (uint32_t)unique_constant_count,
  };
  return loom_condition_relation_set_builder_allocate(
      loom_cfg_condition_operand_domain_size(&solver->operand_domain),
      solver->scratch_arena, &solver->set_builder);
}

static loom_cfg_condition_operand_t
loom_cfg_condition_relation_solver_value_operand(
    const loom_cfg_condition_relation_solver_t* solver,
    loom_value_id_t value_id) {
  const loom_value_id_t canonical =
      loom_cfg_value_identity_table_lookup(solver->identities, value_id);
  const loom_value_ordinal_t local_ordinal =
      loom_local_value_domain_ordinal(solver->value_domain, canonical);
  return solver->relation_operands[local_ordinal];
}

static loom_cfg_condition_operand_t loom_cfg_condition_relation_solver_operand(
    const loom_cfg_condition_relation_solver_t* solver,
    loom_condition_integer_operand_t operand) {
  if (operand.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return loom_cfg_condition_relation_solver_value_operand(solver,
                                                            operand.value_id);
  }
  return loom_cfg_condition_operand_domain_constant(&solver->operand_domain,
                                                    operand.constant);
}

static void loom_cfg_condition_relation_solver_operand_variants(
    const loom_cfg_condition_relation_solver_t* solver,
    loom_condition_integer_operand_t operand,
    loom_cfg_condition_operand_t out_operands[2]) {
  out_operands[0] = loom_cfg_condition_relation_solver_operand(solver, operand);
  out_operands[1] = LOOM_CFG_CONDITION_OPERAND_INVALID;
  if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      solver->fact_table == NULL) {
    return;
  }
  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(solver->fact_table, operand.value_id),
          &exact_value)) {
    return;
  }
  const loom_cfg_condition_operand_t constant =
      loom_cfg_condition_operand_domain_constant(&solver->operand_domain,
                                                 exact_value);
  if (constant != out_operands[0]) {
    out_operands[1] = constant;
  }
}

static loom_condition_relation_set_id_t loom_cfg_condition_relation_singleton(
    loom_cfg_condition_operand_t operand) {
  IREE_ASSERT_NE(operand, LOOM_CFG_CONDITION_OPERAND_INVALID);
  return (loom_condition_relation_set_id_t)(operand + 1);
}

static iree_status_t loom_cfg_condition_relation_add_symmetric(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_operand_t left, loom_cfg_condition_operand_t right,
    loom_condition_relation_outcome_bits_t exclusions) {
  const loom_condition_relation_set_id_t left_set =
      loom_cfg_condition_relation_singleton(left);
  const loom_condition_relation_set_id_t right_set =
      loom_cfg_condition_relation_singleton(right);
  for (loom_condition_relation_outcome_t outcome = 0;
       outcome < LOOM_CONDITION_RELATION_OUTCOME_COUNT; ++outcome) {
    if ((exclusions & (1u << outcome)) == 0) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_builder_add(
        &solver->matrix_builder, outcome, left, right_set));
    IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_builder_add(
        &solver->matrix_builder, loom_cfg_condition_swap_outcome(outcome),
        right, left_set));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_build_matrix(
    loom_cfg_condition_relation_solver_t* solver,
    loom_condition_relation_matrix_t* out_matrix) {
  iree_status_t status = loom_condition_relation_matrix_builder_build(
      &solver->matrix_builder, solver->set_builder, solver->scratch_arena,
      out_matrix);
  loom_condition_relation_matrix_builder_reset(&solver->matrix_builder);
  return status;
}

static iree_status_t loom_cfg_condition_relation_drop_contradictions(
    loom_cfg_condition_relation_solver_t* solver,
    loom_condition_relation_matrix_t* matrix) {
  for (uint32_t row_index = 0; row_index < matrix->row_count; ++row_index) {
    loom_condition_relation_matrix_row_t* row = &matrix->rows[row_index];
    loom_condition_relation_set_id_t contradiction =
        LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
        solver->set_builder, row->excluded[0], row->excluded[1],
        &contradiction));
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
        solver->set_builder, contradiction, row->excluded[2], &contradiction));
    if (contradiction == LOOM_CONDITION_RELATION_SET_EMPTY) {
      continue;
    }
    for (loom_condition_relation_outcome_t outcome = 0;
         outcome < LOOM_CONDITION_RELATION_OUTCOME_COUNT; ++outcome) {
      IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_difference(
          solver->set_builder, row->excluded[outcome], contradiction,
          &row->excluded[outcome]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_matrix_make_writable(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_matrix_state_t* state) {
  if (!state->shared) {
    return iree_ok_status();
  }
  loom_condition_relation_matrix_t matrix = {0};
  IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_clone(
      &state->matrix, solver->scratch_arena, &matrix));
  state->matrix = matrix;
  state->shared = false;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_truth_normalize(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_truth_t* truth) {
  loom_condition_relation_set_id_t contradiction =
      LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
      solver->set_builder, truth->values[0], truth->values[1], &contradiction));
  if (contradiction == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return iree_ok_status();
  }
  for (uint8_t value = 0; value < 2; ++value) {
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_difference(
        solver->set_builder, truth->values[value], contradiction,
        &truth->values[value]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_truth_union_into(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_truth_t* destination,
    const loom_cfg_condition_truth_t* source) {
  for (uint8_t value = 0; value < 2; ++value) {
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_union(
        solver->set_builder, destination->values[value], source->values[value],
        &destination->values[value]));
  }
  return loom_cfg_condition_truth_normalize(solver, destination);
}

static iree_status_t loom_cfg_condition_truth_intersect_into(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_truth_t* destination,
    const loom_cfg_condition_truth_t* source) {
  for (uint8_t value = 0; value < 2; ++value) {
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
        solver->set_builder, destination->values[value], source->values[value],
        &destination->values[value]));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_initialize_payload_mapping(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_edge_state_t* edge) {
  const loom_value_id_t* payload = NULL;
  uint16_t payload_count = 0;
  if (!loom_cfg_terminator_payload_for_successor(edge->cfg_edge->terminator,
                                                 edge->target_block, &payload,
                                                 &payload_count)) {
    return iree_ok_status();
  }
  const uint16_t count = iree_min(payload_count, edge->target_block->arg_count);
  loom_cfg_condition_mapping_pair_t* pairs = NULL;
  if (count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, count, sizeof(*pairs), (void**)&pairs));
  }
  iree_host_size_t pair_count = 0;
  for (uint16_t i = 0; i < count; ++i) {
    const loom_cfg_condition_operand_t source =
        loom_cfg_condition_relation_solver_value_operand(solver, payload[i]);
    const loom_cfg_condition_operand_t target =
        loom_cfg_condition_relation_solver_value_operand(
            solver, loom_block_arg_id(edge->target_block, i));
    if (source == LOOM_CFG_CONDITION_OPERAND_INVALID ||
        target == LOOM_CFG_CONDITION_OPERAND_INVALID) {
      continue;
    }
    pairs[pair_count++] = (loom_cfg_condition_mapping_pair_t){
        .source = source,
        .target = target,
    };
  }
  loom_cfg_condition_mapping_initialize(pairs, pair_count,
                                        &edge->payload_mapping);
  return iree_ok_status();
}

typedef struct loom_cfg_condition_image_span_t {
  // Explicit payload images.
  const loom_cfg_condition_mapping_pair_t* pairs;

  // Number of entries in pairs.
  uint32_t pair_count;

  // Preserved identity image, or UINT32_MAX.
  loom_cfg_condition_operand_t identity;
} loom_cfg_condition_image_span_t;

static loom_cfg_condition_image_span_t loom_cfg_condition_relation_images(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    loom_cfg_condition_operand_t source) {
  loom_cfg_condition_image_span_t images = {
      .identity = LOOM_CFG_CONDITION_OPERAND_INVALID,
  };
  images.pairs = loom_cfg_condition_mapping_lookup(&edge->payload_mapping,
                                                   source, &images.pair_count);

  bool has_identity = source >= solver->operand_domain.value_count;
  if (!has_identity) {
    const uint32_t target_key = (uint32_t)edge->target + 1;
    const uint32_t cached = solver->value_availability_cache[source];
    if ((cached >> 1) == target_key) {
      has_identity = (cached & 1) != 0;
    } else {
      const loom_value_id_t value_id = solver->operand_domain.values[source];
      const loom_value_t* value = loom_module_value(solver->module, value_id);
      const bool rebound = loom_value_is_block_arg(value) &&
                           loom_value_def_block(value) == edge->target_block;
      has_identity =
          !rebound && edge->target_block->first_op &&
          loom_value_is_available_before_op(solver->dominance, value_id,
                                            edge->target_block->first_op);
      solver->value_availability_cache[source] =
          (target_key << 1) | has_identity;
    }
  }
  if (!has_identity) {
    return images;
  }
  for (uint32_t i = 0; i < images.pair_count; ++i) {
    if (images.pairs[i].target == source) {
      return images;
    }
  }
  images.identity = source;
  return images;
}

static bool loom_cfg_condition_image_span_is_identity(
    loom_cfg_condition_image_span_t images,
    loom_cfg_condition_operand_t operand) {
  if (images.pair_count == 0) {
    return images.identity == operand;
  }
  return images.pair_count == 1 && images.pairs[0].target == operand &&
         images.identity == LOOM_CFG_CONDITION_OPERAND_INVALID;
}

static iree_status_t loom_cfg_condition_projection_values_append(
    loom_cfg_condition_relation_solver_t* solver, uint32_t value) {
  if (solver->projection_value_count >= solver->projection_value_capacity) {
    const iree_host_size_t minimum_capacity =
        solver->projection_value_capacity == 0
            ? 16
            : solver->projection_value_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        solver->scratch_arena, solver->projection_value_count, minimum_capacity,
        sizeof(*solver->projection_values), &solver->projection_value_capacity,
        (void**)&solver->projection_values));
  }
  solver->projection_values[solver->projection_value_count++] = value;
  return iree_ok_status();
}

typedef struct loom_cfg_condition_projection_visit_t {
  // Active solver.
  loom_cfg_condition_relation_solver_t* solver;

  // Edge defining the projection.
  const loom_cfg_condition_edge_state_t* edge;

  // Terminal append failure.
  iree_status_t status;

  // True while every visited operand has exactly one identity image.
  bool is_identity;
} loom_cfg_condition_projection_visit_t;

static bool loom_cfg_condition_relation_project_member(
    void* user_data, uint32_t source_operand) {
  loom_cfg_condition_projection_visit_t* visit =
      (loom_cfg_condition_projection_visit_t*)user_data;
  const loom_cfg_condition_image_span_t images =
      loom_cfg_condition_relation_images(visit->solver, visit->edge,
                                         source_operand);
  visit->is_identity &=
      loom_cfg_condition_image_span_is_identity(images, source_operand);
  for (uint32_t i = 0; i < images.pair_count; ++i) {
    visit->status = loom_cfg_condition_projection_values_append(
        visit->solver, images.pairs[i].target);
    if (!iree_status_is_ok(visit->status)) {
      return false;
    }
  }
  if (images.identity != LOOM_CFG_CONDITION_OPERAND_INVALID) {
    visit->status = loom_cfg_condition_projection_values_append(
        visit->solver, images.identity);
  }
  return iree_status_is_ok(visit->status);
}

static iree_status_t loom_cfg_condition_relation_project_set(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    loom_condition_relation_set_id_t source,
    loom_condition_relation_set_id_t* out_target) {
  if (source == LOOM_CONDITION_RELATION_SET_EMPTY) {
    *out_target = LOOM_CONDITION_RELATION_SET_EMPTY;
    return iree_ok_status();
  }
  if (loom_cfg_condition_projection_cache_lookup(
          &solver->projection_cache, edge->edge_index, source, out_target)) {
    return iree_ok_status();
  }
  solver->projection_value_count = 0;
  loom_cfg_condition_projection_visit_t visit = {
      .solver = solver,
      .edge = edge,
      .status = iree_ok_status(),
      .is_identity = true,
  };
  loom_condition_relation_set_builder_for_each_while(
      solver->set_builder, source, loom_cfg_condition_relation_project_member,
      &visit);
  IREE_RETURN_IF_ERROR(visit.status);
  if (visit.is_identity) {
    *out_target = source;
  } else {
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intern(
        solver->set_builder, solver->projection_values,
        solver->projection_value_count, out_target));
  }
  return loom_cfg_condition_projection_cache_insert(
      &solver->projection_cache, edge->edge_index, source, *out_target);
}

static iree_status_t loom_cfg_condition_relation_project_matrix_into_builder(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    const loom_condition_relation_matrix_t* source) {
  for (uint32_t row_index = 0; row_index < source->row_count; ++row_index) {
    const loom_condition_relation_matrix_row_t* row = &source->rows[row_index];
    const loom_cfg_condition_image_span_t left_images =
        loom_cfg_condition_relation_images(solver, edge, row->left);
    for (loom_condition_relation_outcome_t outcome = 0;
         outcome < LOOM_CONDITION_RELATION_OUTCOME_COUNT; ++outcome) {
      if (row->excluded[outcome] == LOOM_CONDITION_RELATION_SET_EMPTY) {
        continue;
      }
      loom_condition_relation_set_id_t projected_rights =
          LOOM_CONDITION_RELATION_SET_EMPTY;
      IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_set(
          solver, edge, row->excluded[outcome], &projected_rights));
      if (projected_rights == LOOM_CONDITION_RELATION_SET_EMPTY) {
        continue;
      }
      for (uint32_t i = 0; i < left_images.pair_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_builder_add(
            &solver->matrix_builder, outcome, left_images.pairs[i].target,
            projected_rights));
      }
      if (left_images.identity != LOOM_CFG_CONDITION_OPERAND_INVALID) {
        IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_builder_add(
            &solver->matrix_builder, outcome, left_images.identity,
            projected_rights));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_matrix_projection_is_identity(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    const loom_condition_relation_matrix_t* source, bool* out_is_identity) {
  *out_is_identity = false;
  for (uint32_t row_index = 0; row_index < source->row_count; ++row_index) {
    const loom_condition_relation_matrix_row_t* row = &source->rows[row_index];
    if (loom_condition_relation_matrix_row_is_empty(row)) {
      // The general projection compacts empty propagation tombstones.
      return iree_ok_status();
    }
    const loom_cfg_condition_image_span_t left_images =
        loom_cfg_condition_relation_images(solver, edge, row->left);
    if (!loom_cfg_condition_image_span_is_identity(left_images, row->left)) {
      return iree_ok_status();
    }
    for (loom_condition_relation_outcome_t outcome = 0;
         outcome < LOOM_CONDITION_RELATION_OUTCOME_COUNT; ++outcome) {
      loom_condition_relation_set_id_t projected =
          LOOM_CONDITION_RELATION_SET_EMPTY;
      IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_set(
          solver, edge, row->excluded[outcome], &projected));
      if (projected != row->excluded[outcome]) {
        return iree_ok_status();
      }
    }
  }
  *out_is_identity = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_project_matrix(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    loom_cfg_condition_matrix_state_t* source,
    loom_cfg_condition_matrix_state_t* out_state) {
  *out_state = (loom_cfg_condition_matrix_state_t){0};
  bool is_identity = false;
  IREE_RETURN_IF_ERROR(
      loom_cfg_condition_relation_matrix_projection_is_identity(
          solver, edge, &source->matrix, &is_identity));
  if (is_identity) {
    out_state->matrix = source->matrix;
    source->shared = true;
    out_state->shared = true;
    return iree_ok_status();
  }
  loom_condition_relation_matrix_builder_reset(&solver->matrix_builder);
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_matrix_into_builder(
      solver, edge, &source->matrix));
  IREE_RETURN_IF_ERROR(
      loom_cfg_condition_relation_build_matrix(solver, &out_state->matrix));
  return loom_cfg_condition_relation_drop_contradictions(solver,
                                                         &out_state->matrix);
}

static iree_status_t loom_cfg_condition_relation_project_truth(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    const loom_cfg_condition_truth_t* source,
    loom_cfg_condition_truth_t* out_truth) {
  *out_truth = (loom_cfg_condition_truth_t){0};
  for (uint8_t value = 0; value < 2; ++value) {
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_set(
        solver, edge, source->values[value], &out_truth->values[value]));
  }
  return loom_cfg_condition_truth_normalize(solver, out_truth);
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_cfg_condition_relation_build_edges(
    loom_cfg_condition_relation_solver_t* solver) {
  solver->projection_cache.arena = solver->scratch_arena;
  for (loom_cfg_edge_index_t edge_index = 0;
       edge_index < solver->graph->edge_count; ++edge_index) {
    loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    if (!edge->active) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_condition_relation_initialize_payload_mapping(solver, edge));

    loom_condition_relation_matrix_builder_reset(&solver->matrix_builder);
    for (uint32_t relation_index = 0; relation_index < edge->raw_relation_count;
         ++relation_index) {
      const loom_condition_integer_relation_t* relation =
          &edge->raw_relations[relation_index];
      if (loom_cfg_condition_relation_is_boolean_truth(solver->module,
                                                       relation)) {
        continue;
      }
      loom_cfg_condition_operand_t left_variants[2];
      loom_cfg_condition_operand_t right_variants[2];
      loom_cfg_condition_relation_solver_operand_variants(
          solver, relation->left, left_variants);
      loom_cfg_condition_relation_solver_operand_variants(
          solver, relation->right, right_variants);
      const loom_condition_relation_outcome_bits_t exclusions =
          loom_cfg_condition_relation_exclusions(relation->relation);
      for (uint8_t left_index = 0; left_index < 2; ++left_index) {
        if (left_variants[left_index] == LOOM_CFG_CONDITION_OPERAND_INVALID) {
          continue;
        }
        for (uint8_t right_index = 0; right_index < 2; ++right_index) {
          if (right_variants[right_index] ==
              LOOM_CFG_CONDITION_OPERAND_INVALID) {
            continue;
          }
          IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_add_symmetric(
              solver, left_variants[left_index], right_variants[right_index],
              exclusions));
        }
      }
    }
    loom_condition_relation_matrix_t raw_relations = {0};
    IREE_RETURN_IF_ERROR(
        loom_cfg_condition_relation_build_matrix(solver, &raw_relations));
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_drop_contradictions(
        solver, &raw_relations));

    loom_cfg_condition_truth_t raw_truth = {0};
    for (uint32_t truth_index = 0; truth_index < edge->raw_boolean_fact_count;
         ++truth_index) {
      const loom_condition_boolean_fact_t fact =
          edge->raw_boolean_facts[truth_index];
      const loom_cfg_condition_operand_t operand =
          loom_cfg_condition_relation_solver_value_operand(solver,
                                                           fact.value_id);
      if (operand == LOOM_CFG_CONDITION_OPERAND_INVALID) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_union(
          solver->set_builder, raw_truth.values[fact.value],
          loom_cfg_condition_relation_singleton(operand),
          &raw_truth.values[fact.value]));
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_condition_truth_normalize(solver, &raw_truth));
    loom_cfg_condition_matrix_state_t raw_state = {.matrix = raw_relations};
    loom_cfg_condition_matrix_state_t assertions = {0};
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_matrix(
        solver, edge, &raw_state, &assertions));
    edge->assertions = assertions.matrix;
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_truth(
        solver, edge, &raw_truth, &edge->truth_assertions));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_project_contribution(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    loom_cfg_condition_matrix_state_t* source,
    loom_cfg_condition_matrix_state_t* out_contribution) {
  *out_contribution = (loom_cfg_condition_matrix_state_t){0};
  if (edge->assertions.row_count == 0) {
    return loom_cfg_condition_relation_project_matrix(solver, edge, source,
                                                      out_contribution);
  }
  loom_condition_relation_matrix_builder_reset(&solver->matrix_builder);
  IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_builder_add_matrix(
      &solver->matrix_builder, &edge->assertions));
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_matrix_into_builder(
      solver, edge, &source->matrix));
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_build_matrix(
      solver, &out_contribution->matrix));
  return loom_cfg_condition_relation_drop_contradictions(
      solver, &out_contribution->matrix);
}

static iree_status_t loom_cfg_condition_relation_project_truth_contribution(
    loom_cfg_condition_relation_solver_t* solver,
    const loom_cfg_condition_edge_state_t* edge,
    const loom_cfg_condition_truth_t* source,
    loom_cfg_condition_truth_t* out_contribution) {
  if (edge->truth_assertions.values[0] == LOOM_CONDITION_RELATION_SET_EMPTY &&
      edge->truth_assertions.values[1] == LOOM_CONDITION_RELATION_SET_EMPTY) {
    return loom_cfg_condition_relation_project_truth(solver, edge, source,
                                                     out_contribution);
  }
  loom_cfg_condition_truth_t projected = {0};
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_truth(
      solver, edge, source, &projected));
  *out_contribution = edge->truth_assertions;
  return loom_cfg_condition_truth_union_into(solver, out_contribution,
                                             &projected);
}

static iree_status_t loom_cfg_condition_relation_intersect_block(
    loom_cfg_condition_relation_solver_t* solver, uint16_t block,
    const loom_condition_relation_matrix_t* contribution,
    loom_cfg_condition_event_queue_t* event_queue) {
  loom_cfg_condition_matrix_state_t* facts =
      &solver->blocks[block].integer_relations;
  uint32_t contribution_position = 0;
  for (uint32_t row_index = 0; row_index < facts->matrix.row_count;
       ++row_index) {
    const loom_condition_relation_matrix_row_t* row =
        &facts->matrix.rows[row_index];
    while (contribution_position < contribution->row_count &&
           contribution->rows[contribution_position].left < row->left) {
      ++contribution_position;
    }
    const loom_condition_relation_matrix_row_t* contribution_row =
        contribution_position < contribution->row_count &&
                contribution->rows[contribution_position].left == row->left
            ? &contribution->rows[contribution_position]
            : NULL;
    for (loom_condition_relation_outcome_t outcome = 0;
         outcome < LOOM_CONDITION_RELATION_OUTCOME_COUNT; ++outcome) {
      const loom_condition_relation_set_id_t incoming =
          contribution_row ? contribution_row->excluded[outcome]
                           : LOOM_CONDITION_RELATION_SET_EMPTY;
      loom_condition_relation_set_id_t intersection =
          LOOM_CONDITION_RELATION_SET_EMPTY;
      IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
          solver->set_builder, row->excluded[outcome], incoming,
          &intersection));
      if (intersection == row->excluded[outcome]) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_cfg_condition_matrix_make_writable(solver, facts));
      facts->matrix.rows[row_index].excluded[outcome] = intersection;
      if (event_queue != NULL) {
        IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_enqueue_integer(
            event_queue, block, row_index, outcome));
      }
    }
  }
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_cfg_condition_relation_initialize_candidates(
    loom_cfg_condition_relation_solver_t* solver) {
  const loom_cfg_graph_t* graph = solver->graph;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      solver->scratch_arena, graph->block_count, sizeof(*solver->blocks),
      (void**)&solver->blocks));
  memset(solver->blocks, 0, graph->block_count * sizeof(*solver->blocks));
  uint8_t* initialized = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(solver->scratch_arena, graph->block_count,
                                sizeof(*initialized), (void**)&initialized));
  memset(initialized, 0, graph->block_count * sizeof(*initialized));
  initialized[0] = 1;

  for (iree_host_size_t position = 0; position < graph->reverse_postorder.count;
       ++position) {
    const uint16_t block = graph->reverse_postorder.values[position];
    if (block == 0) {
      continue;
    }
    const loom_cfg_edge_index_span_t incoming_edges =
        loom_cfg_graph_predecessor_edges(graph, block);
    iree_host_size_t active_predecessor_count = 0;
    for (iree_host_size_t i = 0; i < incoming_edges.count; ++i) {
      active_predecessor_count +=
          solver->edges[incoming_edges.values[i]].active ? 1 : 0;
    }
    bool first = true;
    for (iree_host_size_t i = 0; i < incoming_edges.count; ++i) {
      loom_cfg_condition_edge_state_t* edge =
          &solver->edges[incoming_edges.values[i]];
      if (!edge->active || !initialized[edge->source]) {
        continue;
      }
      loom_cfg_condition_matrix_state_t contribution = {0};
      loom_cfg_condition_truth_t truth_contribution = {0};
      IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_contribution(
          solver, edge, &solver->blocks[edge->source].integer_relations,
          &contribution));
      IREE_RETURN_IF_ERROR(
          loom_cfg_condition_relation_project_truth_contribution(
              solver, edge, &solver->blocks[edge->source].truth,
              &truth_contribution));
      edge->included_in_initial_meet = true;
      if (active_predecessor_count == 1) {
        edge->shares_target_facts = true;
        solver->blocks[block].integer_relations = contribution;
        solver->blocks[block].truth = truth_contribution;
        first = false;
        continue;
      }
      edge->contribution = contribution;
      edge->truth_contribution = truth_contribution;
      if (first) {
        solver->blocks[block].integer_relations = contribution;
        edge->contribution.shared = true;
        solver->blocks[block].integer_relations.shared = true;
        solver->blocks[block].truth = edge->truth_contribution;
        first = false;
      } else {
        IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_intersect_block(
            solver, block, &edge->contribution.matrix, NULL));
        IREE_RETURN_IF_ERROR(loom_cfg_condition_truth_intersect_into(
            solver, &solver->blocks[block].truth, &edge->truth_contribution));
      }
    }
    if (!first) {
      initialized[block] = 1;
    }
  }

  for (loom_cfg_edge_index_t edge_index = 0; edge_index < graph->edge_count;
       ++edge_index) {
    loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    if (!edge->active || edge->included_in_initial_meet) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_contribution(
        solver, edge, &solver->blocks[edge->source].integer_relations,
        &edge->contribution));
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_truth_contribution(
        solver, edge, &solver->blocks[edge->source].truth,
        &edge->truth_contribution));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_intersect_block_truth(
    loom_cfg_condition_relation_solver_t* solver, uint16_t block,
    const loom_cfg_condition_truth_t* contribution) {
  loom_cfg_condition_truth_t* truth = &solver->blocks[block].truth;
  for (uint8_t value = 0; value < 2; ++value) {
    loom_condition_relation_set_id_t intersection =
        LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
        solver->set_builder, truth->values[value], contribution->values[value],
        &intersection));
    if (intersection == truth->values[value]) {
      continue;
    }
    truth->values[value] = intersection;
    IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_enqueue_boolean(
        &solver->pending, block, value != 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_initialize_meets(
    loom_cfg_condition_relation_solver_t* solver) {
  const loom_cfg_graph_t* graph = solver->graph;
  for (uint16_t block = 1; block < graph->block_count; ++block) {
    const loom_cfg_edge_index_span_t incoming_edges =
        loom_cfg_graph_predecessor_edges(graph, block);
    for (iree_host_size_t i = 0; i < incoming_edges.count; ++i) {
      const loom_cfg_condition_edge_state_t* edge =
          &solver->edges[incoming_edges.values[i]];
      if (!edge->active || edge->included_in_initial_meet) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_intersect_block(
          solver, block, &edge->contribution.matrix, &solver->pending));
      IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_intersect_block_truth(
          solver, block, &edge->truth_contribution));
    }
  }
  return iree_ok_status();
}

static const loom_condition_relation_set_id_t*
loom_cfg_condition_relation_matrix_exclusions(
    const loom_condition_relation_matrix_t* matrix,
    loom_cfg_condition_operand_t left) {
  const loom_condition_relation_matrix_row_t* row =
      loom_condition_relation_matrix_find_const(matrix, left);
  return row != NULL ? row->excluded : NULL;
}

static iree_status_t loom_cfg_condition_relation_update_contribution_row(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_edge_state_t* edge,
    loom_condition_relation_outcome_t outcome,
    loom_cfg_condition_operand_t target_left,
    loom_condition_relation_set_id_t candidate) {
  loom_cfg_condition_matrix_state_t* target_facts =
      &solver->blocks[edge->target].integer_relations;
  const loom_condition_relation_matrix_row_t* target_row =
      loom_condition_relation_matrix_find_const(&target_facts->matrix,
                                                target_left);
  const uint32_t target_row_index =
      target_row != NULL ? (uint32_t)(target_row - target_facts->matrix.rows)
                         : 0;
  if (edge->shares_target_facts) {
    if (target_row == NULL) {
      return iree_ok_status();
    }
    loom_condition_relation_set_id_t intersection =
        LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
        solver->set_builder, target_row->excluded[outcome], candidate,
        &intersection));
    if (intersection == target_row->excluded[outcome]) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(
        loom_cfg_condition_matrix_make_writable(solver, target_facts));
    target_facts->matrix.rows[target_row_index].excluded[outcome] =
        intersection;
    return loom_cfg_condition_event_queue_enqueue_integer(
        &solver->pending, edge->target, target_row_index, outcome);
  }

  const loom_condition_relation_matrix_row_t* contribution_row =
      loom_condition_relation_matrix_find_const(&edge->contribution.matrix,
                                                target_left);
  if (contribution_row == NULL) {
    return iree_ok_status();
  }
  const uint32_t contribution_row_index =
      (uint32_t)(contribution_row - edge->contribution.matrix.rows);
  loom_condition_relation_set_id_t contribution =
      LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
      solver->set_builder, contribution_row->excluded[outcome], candidate,
      &contribution));
  if (contribution == contribution_row->excluded[outcome]) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_condition_matrix_make_writable(solver, &edge->contribution));
  edge->contribution.matrix.rows[contribution_row_index].excluded[outcome] =
      contribution;
  if (target_row == NULL) {
    return iree_ok_status();
  }
  loom_condition_relation_set_id_t target = LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
      solver->set_builder, target_row->excluded[outcome], contribution,
      &target));
  if (target == target_row->excluded[outcome]) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_condition_matrix_make_writable(solver, target_facts));
  target_facts->matrix.rows[target_row_index].excluded[outcome] = target;
  return loom_cfg_condition_event_queue_enqueue_integer(
      &solver->pending, edge->target, target_row_index, outcome);
}

static iree_status_t loom_cfg_condition_relation_propagate_integer(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_edge_state_t* edge,
    const loom_cfg_condition_event_t* event) {
  const loom_condition_relation_matrix_t* source =
      &solver->blocks[event->block].integer_relations.matrix;
  const loom_condition_relation_matrix_row_t* source_row =
      &source->rows[event->row];
  loom_condition_relation_set_id_t projected =
      LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_set(
      solver, edge, source_row->excluded[event->outcome], &projected));
  const loom_cfg_condition_image_span_t left_images =
      loom_cfg_condition_relation_images(solver, edge, source_row->left);
  for (uint32_t i = 0; i <= left_images.pair_count; ++i) {
    const loom_cfg_condition_operand_t target_left =
        i < left_images.pair_count ? left_images.pairs[i].target
                                   : left_images.identity;
    if (target_left == LOOM_CFG_CONDITION_OPERAND_INVALID) {
      continue;
    }
    const loom_condition_relation_set_id_t* assertion_exclusions =
        loom_cfg_condition_relation_matrix_exclusions(&edge->assertions,
                                                      target_left);
    const loom_condition_relation_set_id_t assertion =
        assertion_exclusions ? assertion_exclusions[event->outcome]
                             : LOOM_CONDITION_RELATION_SET_EMPTY;
    loom_condition_relation_set_id_t candidate =
        LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_union(
        solver->set_builder, assertion, projected, &candidate));
    IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_update_contribution_row(
        solver, edge, event->outcome, target_left, candidate));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_propagate_boolean(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_edge_state_t* edge,
    const loom_cfg_condition_event_t* event) {
  loom_condition_relation_set_id_t projected =
      LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_project_set(
      solver, edge, solver->blocks[event->block].truth.values[event->outcome],
      &projected));
  loom_condition_relation_set_id_t candidate =
      LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_union(
      solver->set_builder, edge->truth_assertions.values[event->outcome],
      projected, &candidate));
  loom_cfg_condition_truth_t* target_truth =
      &solver->blocks[edge->target].truth;
  if (edge->shares_target_facts) {
    loom_condition_relation_set_id_t intersection =
        LOOM_CONDITION_RELATION_SET_EMPTY;
    IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
        solver->set_builder, target_truth->values[event->outcome], candidate,
        &intersection));
    if (intersection != target_truth->values[event->outcome]) {
      target_truth->values[event->outcome] = intersection;
      IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_enqueue_boolean(
          &solver->pending, edge->target, event->outcome != 0));
    }
    return iree_ok_status();
  }

  loom_condition_relation_set_id_t contribution =
      LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
      solver->set_builder, edge->truth_contribution.values[event->outcome],
      candidate, &contribution));
  if (contribution == edge->truth_contribution.values[event->outcome]) {
    return iree_ok_status();
  }
  edge->truth_contribution.values[event->outcome] = contribution;
  loom_condition_relation_set_id_t target = LOOM_CONDITION_RELATION_SET_EMPTY;
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_intersection(
      solver->set_builder, target_truth->values[event->outcome], contribution,
      &target));
  if (target != target_truth->values[event->outcome]) {
    target_truth->values[event->outcome] = target;
    IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_enqueue_boolean(
        &solver->pending, edge->target, event->outcome != 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_propagate(
    loom_cfg_condition_relation_solver_t* solver) {
  loom_cfg_condition_event_t event;
  while (loom_cfg_condition_event_queue_pop(&solver->pending, &event)) {
    const loom_cfg_edge_index_span_t outgoing_edges =
        loom_cfg_graph_successor_edges(solver->graph, event.block);
    for (iree_host_size_t i = 0; i < outgoing_edges.count; ++i) {
      loom_cfg_condition_edge_state_t* edge =
          &solver->edges[outgoing_edges.values[i]];
      if (!edge->active) {
        continue;
      }
      if (event.kind == LOOM_CFG_CONDITION_EVENT_INTEGER) {
        IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_propagate_integer(
            solver, edge, &event));
      } else {
        IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_propagate_boolean(
            solver, edge, &event));
      }
    }
  }
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_cfg_condition_relation_publish(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_relation_table_t* out_table) {
  const loom_cfg_graph_t* graph = solver->graph;
  uint64_t view_count = graph->block_count;
  for (loom_cfg_edge_index_t edge_index = 0; edge_index < graph->edge_count;
       ++edge_index) {
    const loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    view_count += edge->active && !edge->shares_target_facts ? 1 : 0;
  }
  if (view_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition relation views exceed uint32_t");
  }

  loom_cfg_condition_relation_table_builder_view_t* views = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      solver->scratch_arena, (iree_host_size_t)view_count, sizeof(*views),
      (void**)&views));
  for (uint32_t block = 0; block < graph->block_count; ++block) {
    views[block] = (loom_cfg_condition_relation_table_builder_view_t){
        .integer_relations = solver->blocks[block].integer_relations.matrix,
        .boolean_values = {solver->blocks[block].truth.values[0],
                           solver->blocks[block].truth.values[1]},
    };
  }
  uint32_t* edge_view_indices = NULL;
  if (graph->edge_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        solver->scratch_arena, graph->edge_count, sizeof(*edge_view_indices),
        (void**)&edge_view_indices));
  }
  uint32_t next_view = (uint32_t)graph->block_count;
  for (loom_cfg_edge_index_t edge_index = 0; edge_index < graph->edge_count;
       ++edge_index) {
    const loom_cfg_condition_edge_state_t* edge = &solver->edges[edge_index];
    if (!edge->active) {
      edge_view_indices[edge_index] = LOOM_CFG_CONDITION_VIEW_INVALID;
    } else if (edge->shares_target_facts) {
      edge_view_indices[edge_index] = edge->target;
    } else {
      edge_view_indices[edge_index] = next_view;
      views[next_view++] = (loom_cfg_condition_relation_table_builder_view_t){
          .integer_relations = edge->contribution.matrix,
          .boolean_values = {edge->truth_contribution.values[0],
                             edge->truth_contribution.values[1]},
      };
    }
  }
  IREE_ASSERT_EQ(next_view, view_count);
  loom_cfg_condition_relation_table_builder_t builder = {
      .operand_domain = &solver->operand_domain,
      .set_builder = solver->set_builder,
      .views = views,
      .edge_view_indices = edge_view_indices,
      .view_count = (uint32_t)view_count,
      .block_count = (uint32_t)graph->block_count,
      .edge_count = (uint32_t)graph->edge_count,
  };
  return loom_cfg_condition_relation_table_publish(
      &builder, out_table, solver->scratch_arena, solver->arena);
}

static iree_status_t loom_cfg_condition_relation_solve(
    loom_cfg_condition_relation_solver_t* solver,
    loom_cfg_condition_relation_table_t* out_table) {
  loom_condition_relation_matrix_builder_initialize(solver->scratch_arena,
                                                    &solver->matrix_builder);
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_derive_edges(solver));
  if (!solver->has_derived_facts) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cfg_condition_relation_build_operand_domain(solver));
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_build_edges(solver));
  IREE_RETURN_IF_ERROR(
      loom_cfg_condition_relation_initialize_candidates(solver));
  IREE_RETURN_IF_ERROR(loom_cfg_condition_event_queue_initialize(
      (uint16_t)solver->graph->block_count, solver->blocks,
      solver->scratch_arena, &solver->pending));
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_initialize_meets(solver));
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_propagate(solver));
  return loom_cfg_condition_relation_publish(solver, out_table);
}

iree_status_t loom_cfg_condition_relation_table_compute(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance,
    loom_local_value_domain_t* value_domain,
    const loom_cfg_value_identity_table_t* identities,
    iree_arena_allocator_t* arena,
    loom_cfg_condition_relation_table_t* out_table) {
  *out_table = (loom_cfg_condition_relation_table_t){0};
  if (graph->malformed || graph->block_count == 0) {
    return iree_ok_status();
  }
  if (graph->block_count > UINT16_MAX || graph->edge_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition CFG exceeds index capacity");
  }
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_cfg_condition_relation_solver_t solver = {
      .module = module,
      .graph = graph,
      .fact_table = fact_table,
      .dominance = dominance,
      .value_domain = value_domain,
      .identities = identities,
      .arena = arena,
      .scratch_arena = &scratch_arena,
  };
  iree_status_t status = loom_cfg_condition_relation_solve(&solver, out_table);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
