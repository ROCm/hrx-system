// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/storage_relation_index.h"

static iree_status_t loom_low_schedule_storage_relation_index_allocate_segments(
    loom_low_schedule_storage_relation_index_t* index,
    iree_host_size_t relation_count, iree_arena_allocator_t* arena) {
  const iree_host_size_t segment_count =
      (relation_count + LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_CAPACITY -
       1) >>
      LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_SHIFT;
  for (iree_host_size_t i = 0; i < segment_count; ++i) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(index->relations, arena, &segment));
  }
  return iree_ok_status();
}

static loom_low_schedule_storage_relation_t*
loom_low_schedule_storage_relation_index_mutable_at(
    loom_low_schedule_storage_relation_index_t* index,
    uint32_t relation_index) {
  IREE_ASSERT_LT(relation_index, index->relation_count);
  loom_low_schedule_storage_relation_segment_t* segment =
      (loom_low_schedule_storage_relation_segment_t*)
          loom_segmented_storage_segment(
              index->relations,
              relation_index >>
                  LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_SHIFT);
  return &segment->rows[relation_index &
                        LOOM_LOW_SCHEDULE_STORAGE_RELATION_SEGMENT_MASK];
}

iree_status_t loom_low_schedule_storage_relation_index_initialize(
    const loom_module_t* module, const loom_local_value_domain_t* value_domain,
    const loom_low_schedule_node_t* nodes, uint32_t node_count,
    iree_host_size_t relation_count, iree_arena_allocator_t* arena,
    loom_low_schedule_storage_relation_index_t* out_index) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(value_domain);
  IREE_ASSERT(node_count == 0 || nodes != NULL);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_index);
  IREE_ASSERT_LE(relation_count, UINT32_MAX);
  *out_index = (loom_low_schedule_storage_relation_index_t){
      .node_count = node_count,
      .relation_count = (uint32_t)relation_count,
  };
  if (relation_count == 0) return iree_ok_status();

  const iree_host_size_t node_sentinel_count = (iree_host_size_t)node_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_sentinel_count, sizeof(*out_index->node_relation_starts),
      (void**)&out_index->node_relation_starts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, sizeof(*out_index->relations),
                                           (void**)&out_index->relations));
  loom_segmented_storage_initialize(
      sizeof(loom_low_schedule_storage_relation_segment_t),
      iree_alignof(loom_low_schedule_storage_relation_segment_t),
      out_index->relations);
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_storage_relation_index_allocate_segments(
          out_index, relation_count, arena));

  uint32_t next_relation_index = 0;
  for (uint32_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &nodes[node_index];
    out_index->node_relation_starts[node_index] = next_relation_index;
    loom_low_storage_relation_iterator_t iterator;
    loom_low_storage_relation_iterator_initialize(module, node->op, &iterator);
    loom_low_storage_relation_t relation;
    while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
      IREE_ASSERT_LT(next_relation_index, out_index->relation_count);
      *loom_low_schedule_storage_relation_index_mutable_at(
          out_index, next_relation_index++) =
          (loom_low_schedule_storage_relation_t){
              .destination_ordinal = loom_local_value_domain_ordinal(
                  value_domain, relation.destination_value_id),
              .source_ordinal = loom_local_value_domain_ordinal(
                  value_domain, relation.source_value_id),
              .destination_unit_offset = relation.destination_unit_offset,
              .source_unit_offset = relation.source_unit_offset,
              .unit_count = relation.unit_count,
              .source_operand_index = relation.source_operand_index,
              .flags = relation.flags,
              .kind = relation.kind,
              .cause = relation.cause,
          };
    }
    IREE_ASSERT_EQ(next_relation_index,
                   out_index->node_relation_starts[node_index] +
                       node->storage_relation_count);
  }
  out_index->node_relation_starts[node_count] = next_relation_index;
  IREE_ASSERT_EQ(next_relation_index, out_index->relation_count);
  return iree_ok_status();
}
