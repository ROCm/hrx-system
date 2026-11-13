// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/graph.h"
#include "iree/base/internal/math.h"

// Short predecessor lists use their existing storage directly. An index is
// prepared only when another mutation would repeatedly scan a wide list.
enum { IREE_HAL_STREAMING_GRAPH_DEPENDENCY_INDEX_THRESHOLD = 8 };

static iree_hal_streaming_graph_dependency_index_t*
iree_hal_streaming_graph_dependency_index(
    const iree_hal_streaming_graph_node_t* node) {
  return node->flags & IREE_HAL_STREAMING_GRAPH_NODE_FLAG_INDEXED_DEPENDENCIES
             ? node->dependency_index
             : NULL;
}

static iree_host_size_t iree_hal_streaming_graph_dependency_hash(uint32_t key) {
  // Source identities survive removal/renumbering of other nodes in the graph.
  return (iree_host_size_t)(((uint64_t)key * 0x9e3779b185ebca87ull) >> 32);
}

static iree_host_size_t iree_hal_streaming_graph_dependency_slot(
    const iree_hal_streaming_graph_dependency_index_t* index, uint32_t key) {
  const iree_host_size_t mask = index->capacity - 1;
  iree_host_size_t slot = iree_hal_streaming_graph_dependency_hash(key) & mask;
  while (index->slots[slot] && index->slots[slot] != key) {
    slot = (slot + 1) & mask;
  }
  return slot;
}

static void iree_hal_streaming_graph_dependency_index_insert(
    iree_hal_streaming_graph_dependency_index_t* index, uint32_t key) {
  index->slots[iree_hal_streaming_graph_dependency_slot(index, key)] = key;
  ++index->count;
}

static void iree_hal_streaming_graph_dependency_index_erase(
    iree_hal_streaming_graph_dependency_index_t* index,
    const iree_hal_streaming_graph_node_t* predecessor) {
  if (!index) {
    return;
  }
  const iree_host_size_t mask = index->capacity - 1;
  iree_host_size_t hole = iree_hal_streaming_graph_dependency_slot(
      index, predecessor->clone_source_node_index + 1);
  if (!index->slots[hole]) {
    return;
  }
  --index->count;
  // Close the probe chain so deletion retains no tombstones or rebuild debt.
  for (iree_host_size_t slot = (hole + 1) & mask; index->slots[slot];
       slot = (slot + 1) & mask) {
    const iree_host_size_t home =
        iree_hal_streaming_graph_dependency_hash(index->slots[slot]) & mask;
    if (((hole - home) & mask) < ((slot - home) & mask)) {
      index->slots[hole] = index->slots[slot];
      hole = slot;
    }
  }
  index->slots[hole] = 0;
}

static bool iree_hal_streaming_graph_dependency_exists(
    const iree_hal_streaming_graph_node_t* from_node,
    const iree_hal_streaming_graph_node_t* to_node,
    iree_host_size_t* out_predecessor_count) {
  iree_hal_streaming_graph_dependency_index_t* index =
      iree_hal_streaming_graph_dependency_index(to_node);
  if (index) {
    *out_predecessor_count = index->count;
    return index->slots[iree_hal_streaming_graph_dependency_slot(
               index, from_node->clone_source_node_index + 1)] != 0;
  }
  *out_predecessor_count = to_node->dependency_count;
  for (uint32_t i = 0; i < to_node->dependency_count; ++i) {
    if (iree_hal_streaming_graph_node_dependencies(to_node)[i] == from_node) {
      return true;
    }
  }
  for (const iree_hal_streaming_graph_dependency_t* edge =
           to_node->additional_dependencies;
       edge; edge = edge->next) {
    if (edge->from == from_node) {
      return true;
    }
    ++*out_predecessor_count;
  }
  return false;
}

// Returns the existing index when it has capacity or an unpublished
// replacement. Callers handle short lists directly. Preparation never changes
// the old index.
static iree_status_t iree_hal_streaming_graph_prepare_dependency_index(
    iree_hal_streaming_graph_node_t* node, iree_host_size_t predecessor_count,
    iree_hal_streaming_graph_dependency_index_t** out_index) {
  iree_hal_streaming_graph_dependency_index_t* old_index =
      iree_hal_streaming_graph_dependency_index(node);
  *out_index = old_index;
  if (old_index &&
      predecessor_count < old_index->capacity - old_index->capacity / 4) {
    return iree_ok_status();
  }
  iree_host_size_t capacity = old_index ? old_index->capacity : 16;
  while (predecessor_count >= capacity - capacity / 4) {
    if (!iree_host_size_checked_mul(capacity, 2, &capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "graph dependency index capacity overflow");
    }
  }
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_mul_add(sizeof(**out_index), capacity,
                                      sizeof((*out_index)->slots[0]),
                                      &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependency index allocation size overflow");
  }
  iree_hal_streaming_graph_dependency_index_t* index = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      node->graph->host_allocator, allocation_size, (void**)&index));
  index->head = iree_hal_streaming_graph_additional_dependencies(node);
  index->count = 0;
  index->capacity = capacity;
  memset(index->slots, 0, capacity * sizeof(index->slots[0]));
  if (old_index) {
    for (iree_host_size_t i = 0; i < old_index->capacity; ++i) {
      if (old_index->slots[i]) {
        iree_hal_streaming_graph_dependency_index_insert(index,
                                                         old_index->slots[i]);
      }
    }
  } else {
    for (uint32_t i = 0; i < node->dependency_count; ++i) {
      iree_hal_streaming_graph_dependency_index_insert(
          index, iree_hal_streaming_graph_node_dependencies(node)[i]
                         ->clone_source_node_index +
                     1);
    }
    for (const iree_hal_streaming_graph_dependency_t* edge = index->head; edge;
         edge = edge->next) {
      iree_hal_streaming_graph_dependency_index_insert(
          index, edge->from->clone_source_node_index + 1);
    }
  }
  *out_index = index;
  return iree_ok_status();
}

void iree_hal_streaming_graph_release_dependency_index(
    iree_hal_streaming_graph_node_t* node) {
  iree_hal_streaming_graph_dependency_index_t* index =
      iree_hal_streaming_graph_dependency_index(node);
  if (index) {
    node->additional_dependencies = index->head;
    node->flags &= ~IREE_HAL_STREAMING_GRAPH_NODE_FLAG_INDEXED_DEPENDENCIES;
    iree_allocator_free(node->graph->host_allocator, index);
  }
}

iree_status_t iree_hal_streaming_graph_validate_dependencies(
    const iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** dependencies,
    iree_host_size_t dependency_count) {
  if (dependency_count == 0) {
    return iree_ok_status();
  }
  if (!dependencies) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency array must be provided");
  }
  // Graph mutations are externally serialized. Borrow a flag in each supplied
  // predecessor instead of allocating a set or searching the input prefix.
  // The zero/one-dependency paths never need to write the predecessor.
  const uint8_t mark =
      dependency_count > 1
          ? IREE_HAL_STREAMING_GRAPH_NODE_FLAG_VALIDATING_DEPENDENCY
          : 0;
  iree_status_t status = iree_ok_status();
  iree_host_size_t marked_count = 0;
  for (iree_host_size_t i = 0;
       i < dependency_count && iree_status_is_ok(status); ++i) {
    if (!iree_hal_streaming_graph_node_is_active_in_graph(graph,
                                                          dependencies[i])) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dependency at index %" PRIhsz
                                " does not belong to the target graph",
                                i);
    } else if (mark) {
      if (dependencies[i]->flags & mark) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "duplicate dependency at index %" PRIhsz, i);
      } else {
        dependencies[i]->flags |= mark;
        ++marked_count;
      }
    }
  }
  for (iree_host_size_t i = 0; i < marked_count; ++i) {
    dependencies[i]->flags &= ~mark;
  }
  return status;
}

// Returns the first repeated pair, or |count| if every pair is unique.
// Endpoints have already passed membership validation. The unpublished edge
// allocation supplies two pointer-sized slots per pair, leaving the table half
// full on both 32- and 64-bit hosts. Slots hold request ordinals plus one, so
// zero terminates a collision chain.
static iree_host_size_t iree_hal_streaming_graph_find_duplicate_dependency(
    iree_hal_streaming_graph_node_t* const* from_nodes,
    iree_hal_streaming_graph_node_t* const* to_nodes, iree_host_size_t count,
    iree_host_size_t storage_size, void* storage) {
  static_assert(sizeof(iree_hal_streaming_graph_dependency_t) >=
                    2 * sizeof(iree_host_size_t),
                "edge storage must leave empty duplicate-index slots");
  const iree_host_size_t slot_count = storage_size / sizeof(iree_host_size_t);
  iree_host_size_t* slots = storage;
  memset(slots, 0, slot_count * sizeof(*slots));
  for (iree_host_size_t i = 0; i < count; ++i) {
    // Mix both dense node indices so chains and either fan direction distribute
    // across the table independently of caller insertion order.
    uint64_t hash =
        ((uint64_t)from_nodes[i]->node_index << 32) | to_nodes[i]->node_index;
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33;
    iree_host_size_t slot = (iree_host_size_t)hash % slot_count;
    while (slots[slot]) {
      const iree_host_size_t previous = slots[slot] - 1;
      if (from_nodes[previous] == from_nodes[i] &&
          to_nodes[previous] == to_nodes[i]) {
        return i;
      }
      if (++slot == slot_count) {
        slot = 0;
      }
    }
    slots[slot] = i + 1;
  }
  return count;
}

static iree_status_t iree_hal_streaming_graph_validate_dependency_pair(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* from,
    iree_hal_streaming_graph_node_t* to, iree_host_size_t ordinal) {
  if (!from || !to) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "null node in dependency list at index %" PRIhsz,
                            ordinal);
  }
  if (from == to) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node cannot depend on itself at index %" PRIhsz,
                            ordinal);
  }
  if (!iree_hal_streaming_graph_node_is_active_in_graph(graph, from) ||
      !iree_hal_streaming_graph_node_is_active_in_graph(graph, to)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency node at index %" PRIhsz
                            " does not belong to the target graph",
                            ordinal);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_graph_add_dependency(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* from,
    iree_hal_streaming_graph_node_t* to) {
  iree_host_size_t predecessor_count = 0;
  if (iree_hal_streaming_graph_dependency_exists(from, to,
                                                 &predecessor_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate dependency at index 0");
  }
  iree_hal_streaming_graph_dependency_index_t* old_index =
      iree_hal_streaming_graph_dependency_index(to);
  iree_hal_streaming_graph_dependency_index_t* index = old_index;
  if (old_index || predecessor_count >=
                       IREE_HAL_STREAMING_GRAPH_DEPENDENCY_INDEX_THRESHOLD) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_prepare_dependency_index(
        to, predecessor_count, &index));
  }
  iree_hal_streaming_graph_dependency_t* edge = NULL;
  iree_status_t status =
      iree_arena_allocate(&graph->arena, sizeof(*edge), (void**)&edge);
  if (iree_status_is_ok(status)) {
    if (index) {
      iree_hal_streaming_graph_dependency_index_insert(
          index, from->clone_source_node_index + 1);
      to->dependency_index = index;
      to->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_INDEXED_DEPENDENCIES;
    }
    iree_hal_streaming_graph_dependency_t** head =
        iree_hal_streaming_graph_additional_dependency_head(to);
    *edge =
        (iree_hal_streaming_graph_dependency_t){.next = *head, .from = from};
    *head = edge;
    ++graph->additional_edge_count;
    if (index != old_index) {
      iree_allocator_free(graph->host_allocator, old_index);
    }
  } else if (index != old_index) {
    iree_allocator_free(graph->host_allocator, index);
  }
  return status;
}

// Fits in one unpublished edge slot. An insertion can grow at most one index;
// the edge allocation therefore also bounds every batch's rollback journal.
typedef struct iree_hal_streaming_graph_dependency_change_t {
  // Destination whose index was replaced during preparation.
  iree_hal_streaming_graph_node_t* node;
  // Previous index, or NULL when the destination used its incoming list.
  iree_hal_streaming_graph_dependency_index_t* old_index;
} iree_hal_streaming_graph_dependency_change_t;
static_assert(sizeof(iree_hal_streaming_graph_dependency_change_t) <=
                  sizeof(iree_hal_streaming_graph_dependency_t),
              "unpublished edges must hold their index rollback records");

iree_status_t iree_hal_streaming_graph_add_dependencies(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** from_nodes,
    iree_hal_streaming_graph_node_t** to_nodes, iree_host_size_t count) {
  IREE_ASSERT_ARGUMENT(graph);
  if (count == 0) {
    return iree_ok_status();
  }
  if (!from_nodes || !to_nodes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency arrays must be provided");
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_streaming_graph_validate_dependency_pair(
        graph, from_nodes[i], to_nodes[i], i));
  }
  iree_host_size_t new_edge_count = 0;
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(graph->additional_edge_count, count,
                                  &new_edge_count) ||
      !iree_host_size_checked_mul(count,
                                  sizeof(iree_hal_streaming_graph_dependency_t),
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "graph dependency batch size overflow");
  }
  if (count == 1) {
    return iree_hal_streaming_graph_add_dependency(graph, from_nodes[0],
                                                   to_nodes[0]);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_arena_checkpoint_t checkpoint =
      iree_arena_checkpoint_save(&graph->arena);
  iree_hal_streaming_graph_dependency_t* edges = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&graph->arena, allocation_size, (void**)&edges));
  const iree_host_size_t duplicate =
      iree_hal_streaming_graph_find_duplicate_dependency(
          from_nodes, to_nodes, count, allocation_size, edges);
  if (duplicate != count) {
    iree_arena_checkpoint_restore(&checkpoint);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "duplicate dependency within request at index %" PRIhsz, duplicate);
  }

  iree_hal_streaming_graph_dependency_change_t* changes = (void*)edges;
  iree_host_size_t change_count = 0;
  iree_host_size_t prepared_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_streaming_graph_node_t* to = to_nodes[i];
    // Neither representation can contain predecessors when both fields are
    // zero. Within-request duplicates have already been checked as a batch.
    if (!to->dependency_count && !to->additional_dependencies) {
      ++prepared_count;
      continue;
    }
    iree_host_size_t predecessor_count = 0;
    if (iree_hal_streaming_graph_dependency_exists(from_nodes[i], to,
                                                   &predecessor_count)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate dependency at index %" PRIhsz, i);
    } else {
      iree_hal_streaming_graph_dependency_index_t* old_index =
          iree_hal_streaming_graph_dependency_index(to);
      iree_hal_streaming_graph_dependency_index_t* index = old_index;
      if (old_index ||
          predecessor_count >=
              IREE_HAL_STREAMING_GRAPH_DEPENDENCY_INDEX_THRESHOLD) {
        status = iree_hal_streaming_graph_prepare_dependency_index(
            to, predecessor_count, &index);
      }
      if (iree_status_is_ok(status)) {
        if (index != old_index) {
          changes[change_count++] =
              (iree_hal_streaming_graph_dependency_change_t){
                  .node = to, .old_index = old_index};
          to->dependency_index = index;
          to->flags |= IREE_HAL_STREAMING_GRAPH_NODE_FLAG_INDEXED_DEPENDENCIES;
        }
        if (index) {
          iree_hal_streaming_graph_dependency_index_insert(
              index, from_nodes[i]->clone_source_node_index + 1);
        }
        ++prepared_count;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < change_count; ++i) {
      iree_allocator_free(graph->host_allocator, changes[i].old_index);
    }
    // All fallible work is complete; overwrite the journal with real edges.
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_hal_streaming_graph_dependency_t** head =
          iree_hal_streaming_graph_additional_dependency_head(to_nodes[i]);
      edges[i] = (iree_hal_streaming_graph_dependency_t){.next = *head,
                                                         .from = from_nodes[i]};
      *head = &edges[i];
    }
    graph->additional_edge_count = new_edge_count;
  } else {
    // Restore index ownership before removing provisional entries: an old
    // table may contain entries inserted earlier in this same batch.
    while (change_count) {
      const iree_hal_streaming_graph_dependency_change_t change =
          changes[--change_count];
      iree_hal_streaming_graph_dependency_index_t* replacement =
          change.node->dependency_index;
      if (change.old_index) {
        change.node->dependency_index = change.old_index;
      } else {
        change.node->additional_dependencies = replacement->head;
        change.node->flags &=
            ~IREE_HAL_STREAMING_GRAPH_NODE_FLAG_INDEXED_DEPENDENCIES;
      }
      iree_allocator_free(graph->host_allocator, replacement);
    }
    for (iree_host_size_t i = 0; i < prepared_count; ++i) {
      iree_hal_streaming_graph_dependency_index_erase(
          iree_hal_streaming_graph_dependency_index(to_nodes[i]),
          from_nodes[i]);
    }
    iree_arena_checkpoint_restore(&checkpoint);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_graph_remove_dependency_pair(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* from,
    iree_hal_streaming_graph_node_t* to) {
  iree_hal_streaming_graph_dependency_index_t* index =
      iree_hal_streaming_graph_dependency_index(to);
  iree_hal_streaming_graph_dependency_index_erase(index, from);
  if (index && !index->count) {
    iree_hal_streaming_graph_release_dependency_index(to);
  }
  iree_hal_streaming_graph_node_t** dependencies =
      iree_hal_streaming_graph_node_dependencies(to);
  for (uint32_t i = 0; i < to->dependency_count; ++i) {
    if (dependencies[i] == from) {
      memmove(dependencies + i, dependencies + i + 1,
              (--to->dependency_count - i) * sizeof(dependencies[0]));
      return;
    }
  }
  iree_hal_streaming_graph_dependency_t** edge =
      iree_hal_streaming_graph_additional_dependency_head(to);
  while (*edge) {
    if ((*edge)->from == from) {
      *edge = (*edge)->next;
      --graph->additional_edge_count;
      return;
    }
    edge = &(*edge)->next;
  }
}

iree_status_t iree_hal_streaming_graph_remove_dependencies(
    iree_hal_streaming_graph_t* graph,
    iree_hal_streaming_graph_node_t** from_nodes,
    iree_hal_streaming_graph_node_t** to_nodes, iree_host_size_t count) {
  if (count && (!from_nodes || !to_nodes)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dependency arrays must be provided");
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_host_size_t predecessor_count = 0;
    if (!iree_hal_streaming_graph_node_is_active_in_graph(graph,
                                                          from_nodes[i]) ||
        !iree_hal_streaming_graph_node_is_active_in_graph(graph, to_nodes[i]) ||
        !iree_hal_streaming_graph_dependency_exists(from_nodes[i], to_nodes[i],
                                                    &predecessor_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dependency at index %" PRIhsz " is not in the graph", i);
    }
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_graph_remove_dependency_pair(graph, from_nodes[i],
                                                    to_nodes[i]);
  }
  return iree_ok_status();
}

void iree_hal_streaming_graph_remove_dependency_refs(
    iree_hal_streaming_graph_t* graph, iree_hal_streaming_graph_node_t* node) {
  for (iree_hal_streaming_node_block_t* block = graph->node_blocks; block;
       block = block->next) {
    for (iree_host_size_t i = 0; i < block->count; ++i) {
      iree_hal_streaming_graph_node_t* existing = block->nodes[i];
      iree_hal_streaming_graph_dependency_index_t* index =
          iree_hal_streaming_graph_dependency_index(existing);
      iree_hal_streaming_graph_dependency_index_erase(index, node);
      if (index && !index->count) {
        iree_hal_streaming_graph_release_dependency_index(existing);
      }
      iree_hal_streaming_graph_node_t** dependencies =
          iree_hal_streaming_graph_node_dependencies(existing);
      for (uint32_t j = 0; j < existing->dependency_count; ++j) {
        if (dependencies[j] == node) {
          memmove(dependencies + j, dependencies + j + 1,
                  (--existing->dependency_count - j) * sizeof(dependencies[0]));
          break;
        }
      }
      iree_hal_streaming_graph_dependency_t** edge =
          iree_hal_streaming_graph_additional_dependency_head(existing);
      while (*edge) {
        if (existing == node || (*edge)->from == node) {
          *edge = (*edge)->next;
          --graph->additional_edge_count;
        } else {
          edge = &(*edge)->next;
        }
      }
    }
  }
}
