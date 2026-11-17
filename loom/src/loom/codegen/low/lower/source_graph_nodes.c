// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_graph_nodes.h"

#include <string.h>

static iree_host_size_t loom_low_lower_source_graph_node_hash(
    const loom_op_t* source_op) {
  uint64_t value = (uint64_t)(uintptr_t)source_op >> 4;
  value ^= value >> 16;
  return (iree_host_size_t)(value * UINT64_C(0x9e3779b97f4a7c15));
}

static bool loom_low_lower_source_graph_node_value_ordinal(
    const loom_low_lower_source_graph_node_index_t* index,
    const loom_op_t* source_op, loom_value_ordinal_t* out_ordinal) {
  *out_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (source_op->result_count == 0 || index->value_domain == NULL ||
      !loom_local_value_domain_is_acquired(index->value_domain)) {
    return false;
  }
  const loom_value_ordinal_t ordinal = loom_local_value_domain_try_ordinal(
      index->value_domain, loom_op_const_results(source_op)[0]);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return false;
  }
  *out_ordinal = ordinal;
  return true;
}

void loom_low_lower_source_graph_node_index_initialize(
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena,
    loom_low_lower_source_graph_node_index_t* out_index) {
  *out_index = (loom_low_lower_source_graph_node_index_t){
      .value_domain = value_domain,
      .arena = arena,
  };
}

loom_low_lower_source_graph_node_t*
loom_low_lower_source_graph_node_index_lookup(
    const loom_low_lower_source_graph_node_index_t* index,
    const loom_op_t* source_op) {
  loom_value_ordinal_t ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_lower_source_graph_node_value_ordinal(index, source_op,
                                                     &ordinal)) {
    return index->value_nodes != NULL && ordinal < index->value_node_count
               ? index->value_nodes[ordinal]
               : NULL;
  }
  if (index->source_op_entry_capacity == 0) {
    return NULL;
  }
  const iree_host_size_t mask = index->source_op_entry_capacity - 1;
  iree_host_size_t slot =
      loom_low_lower_source_graph_node_hash(source_op) & mask;
  while (index->source_op_entries[slot].source_op != NULL) {
    if (index->source_op_entries[slot].source_op == source_op) {
      return index->source_op_entries[slot].node;
    }
    slot = (slot + 1) & mask;
  }
  return NULL;
}

static void loom_low_lower_source_graph_node_insert_source_op(
    loom_low_lower_source_graph_source_op_entry_t* entries,
    iree_host_size_t capacity, const loom_op_t* source_op,
    loom_low_lower_source_graph_node_t* node) {
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot =
      loom_low_lower_source_graph_node_hash(source_op) & mask;
  while (entries[slot].source_op != NULL &&
         entries[slot].source_op != source_op) {
    slot = (slot + 1) & mask;
  }
  entries[slot] = (loom_low_lower_source_graph_source_op_entry_t){
      .source_op = source_op,
      .node = node,
  };
}

iree_status_t loom_low_lower_source_graph_node_index_get_or_add(
    loom_low_lower_source_graph_node_index_t* index, const loom_op_t* source_op,
    loom_low_lower_source_graph_node_t** out_node) {
  *out_node = loom_low_lower_source_graph_node_index_lookup(index, source_op);
  if (*out_node != NULL) {
    return iree_ok_status();
  }
  if (index->count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "source graph node count overflow");
  }
  loom_low_lower_source_graph_node_t* node = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(index->arena, sizeof(*node), (void**)&node));
  *node = (loom_low_lower_source_graph_node_t){
      .source_op = source_op,
      .index = index->count++,
  };
  if (index->last != NULL) {
    index->last->next = node;
  } else {
    index->first = node;
  }
  index->last = node;

  loom_value_ordinal_t ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_lower_source_graph_node_value_ordinal(index, source_op,
                                                     &ordinal)) {
    if (index->value_nodes == NULL) {
      index->value_node_count = index->value_domain->value_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          index->arena, index->value_node_count, sizeof(*index->value_nodes),
          (void**)&index->value_nodes));
      memset(index->value_nodes, 0,
             (iree_host_size_t)index->value_node_count *
                 sizeof(*index->value_nodes));
    }
    IREE_ASSERT_LT(ordinal, index->value_node_count);
    index->value_nodes[ordinal] = node;
    *out_node = node;
    return iree_ok_status();
  }

  if (index->source_op_entry_capacity == 0 ||
      index->source_op_entry_count >= index->source_op_entry_capacity / 2) {
    const iree_host_size_t new_capacity =
        index->source_op_entry_capacity == 0
            ? 8
            : index->source_op_entry_capacity * 2;
    if (new_capacity < index->source_op_entry_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "source graph node index capacity overflow");
    }
    loom_low_lower_source_graph_source_op_entry_t* new_entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(index->arena, new_capacity,
                                                   sizeof(*new_entries),
                                                   (void**)&new_entries));
    memset(new_entries, 0, new_capacity * sizeof(*new_entries));
    for (iree_host_size_t i = 0; i < index->source_op_entry_capacity; ++i) {
      const loom_low_lower_source_graph_source_op_entry_t entry =
          index->source_op_entries[i];
      if (entry.source_op != NULL) {
        loom_low_lower_source_graph_node_insert_source_op(
            new_entries, new_capacity, entry.source_op, entry.node);
      }
    }
    index->source_op_entries = new_entries;
    index->source_op_entry_capacity = new_capacity;
  }

  loom_low_lower_source_graph_node_insert_source_op(
      index->source_op_entries, index->source_op_entry_capacity, source_op,
      node);
  ++index->source_op_entry_count;
  *out_node = node;
  return iree_ok_status();
}
