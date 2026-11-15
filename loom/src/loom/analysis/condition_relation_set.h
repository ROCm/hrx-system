// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact finite sets used by condition-relation propagation.
//
// Domains of at most 32 values encode membership directly in the set ID.
// Larger domains use interned dense bitmaps in caller-owned scratch, then
// publish selected roots into an immutable shared DAG in a longer-lived arena.
// Consumers retain no dense bitmaps or construction hash tables.

#ifndef LOOM_ANALYSIS_CONDITION_RELATION_SET_H_
#define LOOM_ANALYSIS_CONDITION_RELATION_SET_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonical identity of a set within a builder or an immutable index. Zero is
// the empty set; other IDs are representation-owned and domain-dependent.
// Publication translates every supplied root into the index's ID space.
typedef uint32_t loom_condition_relation_set_id_t;

#define LOOM_CONDITION_RELATION_SET_EMPTY ((loom_condition_relation_set_id_t)0)

typedef struct loom_condition_relation_set_builder_t
    loom_condition_relation_set_builder_t;
typedef union loom_condition_relation_set_index_node_t
    loom_condition_relation_set_index_node_t;

// Immutable set index produced from selected builder roots.
typedef struct loom_condition_relation_set_index_t {
  // Compact nodes shared by every published root.
  const loom_condition_relation_set_index_node_t* nodes;

  // Number of values in the finite set domain.
  uint32_t value_count;

  // Number of entries in nodes.
  uint32_t node_count;

  // Root tree level shared by every non-inline set ID.
  uint8_t root_level;
} loom_condition_relation_set_index_t;

// Visits one member ordinal. Returning false stops iteration.
typedef bool (*loom_condition_relation_set_visit_fn_t)(void* user_data,
                                                       uint32_t value);

// Allocates a set builder in |scratch_arena|. The arena owns the builder, every
// interned bitmap, and all publication scratch. |value_count| must leave room
// in the 32-bit ID space for at least one non-singleton node.
iree_status_t loom_condition_relation_set_builder_allocate(
    uint32_t value_count, iree_arena_allocator_t* scratch_arena,
    loom_condition_relation_set_builder_t** out_builder);

// Returns the singleton containing |value| in the builder's finite domain.
loom_condition_relation_set_id_t loom_condition_relation_set_builder_singleton(
    const loom_condition_relation_set_builder_t* builder, uint32_t value);

// Interns the set containing |values|. Values may be unsorted and repeated.
iree_status_t loom_condition_relation_set_builder_intern(
    loom_condition_relation_set_builder_t* builder, const uint32_t* values,
    iree_host_size_t value_count, loom_condition_relation_set_id_t* out_set);

// Interns the union of |left| and |right|.
iree_status_t loom_condition_relation_set_builder_union(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set);

// Interns the intersection of |left| and |right|.
iree_status_t loom_condition_relation_set_builder_intersection(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set);

// Interns the members of |left| that are absent from |right|.
iree_status_t loom_condition_relation_set_builder_difference(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t left,
    loom_condition_relation_set_id_t right,
    loom_condition_relation_set_id_t* out_set);

// Visits the members of one builder set in ascending order. Returns false when
// |visit| stops iteration.
bool loom_condition_relation_set_builder_for_each_while(
    const loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t set,
    loom_condition_relation_set_visit_fn_t visit, void* user_data);

// Publishes only sets reachable from |roots| into |arena| and rewrites each
// root in place into the immutable index's ID space. |arena| must outlive the
// builder's scratch arena and must be a distinct arena instance. Failure leaves
// roots unchanged and produces an empty index.
iree_status_t loom_condition_relation_set_builder_publish(
    loom_condition_relation_set_builder_t* builder,
    loom_condition_relation_set_id_t* roots, iree_host_size_t root_count,
    iree_arena_allocator_t* arena,
    loom_condition_relation_set_index_t* out_index);

// Returns true when |set| contains |value|.
bool loom_condition_relation_set_index_contains(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set, uint32_t value);

// Visits the members of one immutable set in ascending order. Returns false
// when |visit| stops iteration.
bool loom_condition_relation_set_index_for_each_while(
    const loom_condition_relation_set_index_t* index,
    loom_condition_relation_set_id_t set,
    loom_condition_relation_set_visit_fn_t visit, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CONDITION_RELATION_SET_H_
