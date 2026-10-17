// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_participants.h"

#include <string.h>

#include "loom/util/adaptive_sort.h"

#define LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT 64u

typedef struct loom_control_participant_record_t {
  // Block in the immutable source snapshot.
  const loom_block_t* block;
  // Coordinates guaranteed zero whenever this block executes.
  loom_control_participant_zeros_t zeros;
} loom_control_participant_record_t;

static loom_control_participant_record_t* loom_control_participant_record_at(
    loom_segmented_storage_t* storage, iree_host_size_t index) {
  return (loom_control_participant_record_t*)loom_segmented_storage_segment(
             storage,
             (uint32_t)(index / LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT)) +
         index % LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT;
}

static bool loom_control_participant_record_less(
    void* context, const loom_control_participant_record_t* lhs,
    const loom_control_participant_record_t* rhs) {
  return (uintptr_t)lhs->block < (uintptr_t)rhs->block;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_ACCESSOR(loom_control_participant_sort,
                                        loom_control_participant_record_t,
                                        loom_segmented_storage_t*,
                                        loom_control_participant_record_at,
                                        void*,
                                        loom_control_participant_record_less)

void loom_control_participant_table_initialize(
    iree_arena_allocator_t* arena,
    loom_control_participant_table_t* out_table) {
  *out_table = (loom_control_participant_table_t){.arena = arena};
  loom_segmented_storage_initialize(
      LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT *
          sizeof(loom_control_participant_record_t),
      iree_alignof(loom_control_participant_record_t), &out_table->records);
}

iree_status_t loom_control_participant_table_append(
    loom_control_participant_table_t* table, const loom_block_t* block,
    loom_control_participant_zeros_t zeros) {
  if (zeros == 0) {
    return iree_ok_status();
  }
  if (table->count % LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT == 0) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&table->records, table->arena, &segment));
  }
  *loom_control_participant_record_at(&table->records, table->count++) =
      (loom_control_participant_record_t){.block = block, .zeros = zeros};
  return iree_ok_status();
}

void loom_control_participant_table_publish(
    loom_control_participant_table_t* table) {
  loom_control_participant_sort(NULL, &table->records, table->count);
}

loom_control_participant_zeros_t loom_control_participant_table_lookup(
    const loom_control_participant_table_t* table, const loom_block_t* block) {
  uint32_t begin = 0;
  uint32_t end = table->count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const loom_control_participant_record_t* segment =
        loom_segmented_storage_const_segment(
            &table->records,
            middle / LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT);
    const loom_control_participant_record_t* record =
        &segment[middle % LOOM_CONTROL_PARTICIPANT_RECORDS_PER_SEGMENT];
    if (record->block == block) {
      return record->zeros;
    }
    if ((uintptr_t)record->block < (uintptr_t)block) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return 0;
}

static loom_control_participant_zeros_t loom_control_participant_value_domain(
    const loom_value_fact_table_t* facts, loom_value_id_t value) {
  const loom_value_fact_topology_domain_t* domain =
      loom_value_facts_topology_domain(
          loom_value_fact_table_lookup(facts, value));
  if (!domain ||
      (domain->value_kind != LOOM_VALUE_FACT_TOPOLOGY_VALUE_WORKITEM_ID &&
       domain->value_kind != LOOM_VALUE_FACT_TOPOLOGY_VALUE_SUBGROUP_LANE_ID)) {
    return 0;
  }
  return (loom_control_participant_zeros_t)(1u << domain->axis);
}

static loom_control_participant_zeros_t
loom_control_participant_expression_domain(
    const loom_value_fact_table_t* facts,
    const loom_symbolic_expr_t* expression) {
  if (!loom_symbolic_expr_is_linear(expression) || expression->constant != 0 ||
      expression->term_count != 1 || expression->terms[0].coefficient != 1) {
    return 0;
  }
  return loom_control_participant_value_domain(facts,
                                               expression->terms[0].value_id) |
         loom_control_participant_value_domain(
             facts, expression->terms[0].relation_value_id);
}

iree_status_t loom_control_participant_zeros_from_conditions(
    loom_symbolic_expr_context_t* expressions,
    const loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_count,
    loom_control_participant_zeros_t* out_zeros) {
  *out_zeros = 0;
  for (iree_host_size_t i = 0; i < relation_count; ++i) {
    const loom_condition_integer_relation_t* relation = &relations[i];
    if (relation->relation != LOOM_SYMBOLIC_INTEGER_RELATION_EQ) {
      continue;
    }
    const loom_condition_integer_operand_t operands[] = {relation->left,
                                                         relation->right};
    loom_symbolic_expr_t sides[2] = {0};
    for (uint8_t side = 0; side < 2; ++side) {
      if (operands[side].kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT) {
        loom_symbolic_expr_constant(operands[side].constant, &sides[side]);
      } else {
        IREE_RETURN_IF_ERROR(loom_symbolic_expr_from_value(
            expressions, operands[side].value_id, &sides[side]));
      }
    }
    for (uint8_t side = 0; side < 2; ++side) {
      if (loom_symbolic_expr_is_constant(&sides[side]) &&
          sides[side].constant == 0) {
        *out_zeros |= loom_control_participant_expression_domain(
            expressions->fact_table, &sides[1 - side]);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_control_participant_zeros_for_cfg(
    const loom_cfg_graph_t* graph, const loom_cfg_condition_edge_facts_t* edges,
    loom_symbolic_expr_context_t* expressions, iree_arena_allocator_t* scratch,
    loom_control_participant_zeros_t** out_blocks) {
  *out_blocks = NULL;
  if (graph->block_count == 0) {
    return iree_ok_status();
  }
  loom_control_participant_zeros_t* blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, graph->block_count, sizeof(*blocks), (void**)&blocks));
  memset(blocks, 0, graph->block_count * sizeof(*blocks));
  loom_control_participant_zeros_t* guards = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, graph->edge_count, sizeof(*guards), (void**)&guards));
  for (loom_cfg_edge_index_t i = 0; i < graph->edge_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_control_participant_zeros_from_conditions(
        expressions, edges[i].integer_relations,
        edges[i].integer_relation_count, &guards[i]));
  }

  // Only identities with an unguarded path from entry propagate. Each bit can
  // be added to a block once; queued blocks combine all pending additions.
  enum { kQueued = 1u << 7 };
  enum { kWorklistEntriesPerSegment = 2048 };
  loom_segmented_storage_t worklist;
  loom_segmented_storage_initialize(
      kWorklistEntriesPerSegment * sizeof(uint16_t), iree_alignof(uint16_t),
      &worklist);
  uint16_t* first_segment = NULL;
  IREE_RETURN_IF_ERROR(loom_segmented_storage_append(&worklist, scratch,
                                                     (void**)&first_segment));
  uint32_t count = 1;
  first_segment[0] = 0;
  blocks[0] = LOOM_CONTROL_PARTICIPANT_ZERO_ALL | kQueued;
  while (count != 0) {
    --count;
    const uint16_t* segment = loom_segmented_storage_const_segment(
        &worklist, count / kWorklistEntriesPerSegment);
    const uint16_t source = segment[count % kWorklistEntriesPerSegment];
    blocks[source] &= ~kQueued;
    const loom_control_participant_zeros_t source_bits = blocks[source];
    const loom_cfg_edge_index_span_t successors =
        loom_cfg_graph_successor_edges(graph, source);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      const loom_cfg_edge_index_t index = successors.values[i];
      const uint16_t target = graph->edges[index].target_block_index;
      const loom_control_participant_zeros_t added =
          source_bits & ~guards[index] & ~blocks[target];
      if (added == 0) {
        continue;
      }
      if (!(blocks[target] & kQueued)) {
        const uint32_t segment_index = count / kWorklistEntriesPerSegment;
        uint16_t* target_segment = NULL;
        if (segment_index == worklist.segment_count) {
          IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
              &worklist, scratch, (void**)&target_segment));
        } else {
          target_segment =
              loom_segmented_storage_segment(&worklist, segment_index);
        }
        target_segment[count++ % kWorklistEntriesPerSegment] = target;
      }
      blocks[target] |= added | kQueued;
    }
  }
  for (uint32_t i = 0; i < graph->block_count; ++i) {
    blocks[i] = loom_cfg_graph_block_is_reachable(graph, (uint16_t)i)
                    ? LOOM_CONTROL_PARTICIPANT_ZERO_ALL & ~blocks[i]
                    : 0;
  }
  *out_blocks = blocks;
  return iree_ok_status();
}
