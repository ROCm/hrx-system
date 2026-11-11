// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/class_partition.h"

#include <string.h>

iree_status_t loom_decision_class_partition_initialize(
    iree_host_size_t site_count, loom_decision_class_ordinal_t class_limit,
    uint32_t maximum_outcome_count, iree_arena_allocator_t* arena,
    loom_decision_class_partition_t* out_partition) {
  *out_partition = (loom_decision_class_partition_t){
      .site_count = site_count,
      .class_limit = class_limit,
      .class_count = 1,
  };
  const uint32_t class_capacity = (uint32_t)iree_min(site_count, class_limit);
  uint32_t bucket_count = 2;
  while (bucket_count < class_capacity * 2) {
    bucket_count *= 2;
  }
  // At most one distinct pair per site can become a class. Keep the table at
  // most half full, regardless of the number of unobserved provider outcomes.
  const uint64_t direct_bytes = (uint64_t)class_capacity *
                                maximum_outcome_count *
                                sizeof(loom_decision_class_ordinal_t);
  const uint32_t sparse_bytes = bucket_count * sizeof(uint64_t);
  // Direct lookup remains worthwhile when its full vocabulary costs no more
  // than one packed pair per input site. Both choices stay linear in input
  // storage; small launch batches cannot pay for a large provider catalog.
  const uint32_t site_budget =
      (uint32_t)iree_min(site_count, UINT32_MAX / sizeof(uint64_t)) *
      sizeof(uint64_t);
  out_partition->pair_storage_byte_length =
      direct_bytes <= iree_max(sparse_bytes, site_budget)
          ? (uint32_t)direct_bytes
          : sparse_bytes;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*out_partition->site_classes),
      (void**)&out_partition->site_classes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*out_partition->candidate_site_classes),
      (void**)&out_partition->candidate_site_classes));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, out_partition->pair_storage_byte_length,
                          &out_partition->pair_storage));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_capacity, sizeof(*out_partition->candidate_parent_classes),
      (void**)&out_partition->candidate_parent_classes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_capacity, sizeof(*out_partition->candidate_outcomes),
      (void**)&out_partition->candidate_outcomes));
  memset(out_partition->site_classes, 0,
         site_count * sizeof(*out_partition->site_classes));
  return iree_ok_status();
}

void loom_decision_class_partition_begin(
    loom_decision_class_partition_t* partition, uint32_t outcome_count) {
  partition->candidate_class_count = 0;
  partition->active_outcome_count = outcome_count;
  const uint64_t direct_bytes = (uint64_t)partition->class_count *
                                outcome_count *
                                sizeof(loom_decision_class_ordinal_t);
  uint32_t clear_bytes = partition->pair_storage_byte_length;
  if (direct_bytes <= clear_bytes) {
    partition->pair_bucket_mask = 0;
    clear_bytes = (uint32_t)direct_bytes;
  } else {
    partition->pair_bucket_mask = clear_bytes / sizeof(uint64_t) - 1;
  }
  memset(partition->pair_storage, 0xFF, clear_bytes);
}

void loom_decision_class_partition_commit(
    loom_decision_class_partition_t* partition) {
  loom_decision_class_ordinal_t* previous_classes = partition->site_classes;
  partition->site_classes = partition->candidate_site_classes;
  partition->candidate_site_classes = previous_classes;
  partition->class_count = partition->candidate_class_count;
}
