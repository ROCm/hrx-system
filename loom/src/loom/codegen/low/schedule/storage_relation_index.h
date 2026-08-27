// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact function-local storage relations used by low scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_STORAGE_RELATION_INDEX_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_STORAGE_RELATION_INDEX_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of compact storage-relation rows stored in each stable segment.
#define LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_CAPACITY 2048u

// Shift mapping a storage-relation index to its segment index.
#define LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_SHIFT 11u

// Mask mapping a storage-relation index to its row within a segment.
#define LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_MASK \
  (LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_CAPACITY - 1u)

static_assert((1u << LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_SHIFT) ==
                  LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_CAPACITY,
              "storage-relation segment capacity must match its index shift");

// One verified storage relation remapped into function-local value ordinals.
typedef struct loom_low_schedule_storage_relation_t {
  // Destination value receiving related storage.
  loom_value_ordinal_t destination_ordinal;
  // Source value providing related storage.
  loom_value_ordinal_t source_ordinal;
  // First related unit inside the destination value.
  uint32_t destination_unit_offset;
  // First related unit inside the source value.
  uint32_t source_unit_offset;
  // Number of related storage units.
  uint32_t unit_count;
  // Operand index of source_ordinal on the owning node.
  uint16_t source_operand_index;
  // Hard or preferred relation behavior.
  loom_low_storage_relation_flags_t flags;
  // Structural relation shape.
  loom_low_storage_relation_kind_t kind;
  // IR feature that introduced the relation.
  loom_low_storage_relation_cause_t cause;
} loom_low_schedule_storage_relation_t;

// Stable storage for one segment of compact storage relations.
typedef iree_alignas(64) struct loom_low_schedule_storage_relation_segment_t {
  // Relation rows indexed by the low bits of a relation index.
  loom_low_schedule_storage_relation_t
      rows[LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_CAPACITY];
} loom_low_schedule_storage_relation_segment_t;

static_assert(sizeof(loom_low_schedule_storage_relation_segment_t) <= 64 * 1024,
              "storage-relation segment must fit a compiler workspace block");

// Node-contiguous storage relations for one scheduled function.
typedef struct loom_low_schedule_storage_relation_index_t {
  // Relation range starts indexed by schedule node, with a terminal sentinel.
  uint32_t* node_relation_starts;
  // Arena-owned stable relation segments, or NULL when there are no relations.
  loom_segmented_storage_t* relations;
  // Number of represented schedule nodes.
  uint32_t node_count;
  // Number of populated relation rows.
  uint32_t relation_count;
} loom_low_schedule_storage_relation_index_t;

// Builds an exact compact index from the verified relations cached on |nodes|.
// The zero-relation path performs no allocation.
iree_status_t loom_low_schedule_storage_relation_index_initialize(
    const loom_module_t* module, const loom_local_value_domain_t* value_domain,
    const loom_low_schedule_node_t* nodes, uint32_t node_count,
    iree_host_size_t relation_count, iree_arena_allocator_t* arena,
    loom_low_schedule_storage_relation_index_t* out_index);

// Returns the first relation index owned by |node_index|.
static inline uint32_t loom_low_schedule_storage_relation_index_begin(
    const loom_low_schedule_storage_relation_index_t* index,
    uint32_t node_index) {
  IREE_ASSERT_LT(node_index, index->node_count);
  return index->node_relation_starts ? index->node_relation_starts[node_index]
                                     : 0;
}

// Returns the exclusive relation end owned by |node_index|.
static inline uint32_t loom_low_schedule_storage_relation_index_end(
    const loom_low_schedule_storage_relation_index_t* index,
    uint32_t node_index) {
  IREE_ASSERT_LT(node_index, index->node_count);
  return index->node_relation_starts
             ? index->node_relation_starts[node_index + 1]
             : 0;
}

// Returns a relation by its stable zero-based index.
static inline const loom_low_schedule_storage_relation_t*
loom_low_schedule_storage_relation_index_at(
    const loom_low_schedule_storage_relation_index_t* index,
    uint32_t relation_index) {
  IREE_ASSERT_LT(relation_index, index->relation_count);
  const loom_low_schedule_storage_relation_segment_t* segment =
      (const loom_low_schedule_storage_relation_segment_t*)
          loom_segmented_storage_const_segment(
              index->relations,
              relation_index >>
                  LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_SHIFT);
  return &segment->rows[relation_index &
                        LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_MASK];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_STORAGE_RELATION_INDEX_H_
