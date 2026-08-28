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
  iree_host_size_t pair_count = 0;
  if (!iree_host_size_checked_mul(class_limit, maximum_outcome_count,
                                  &pair_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "decision class partition exceeds host range");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*out_partition->site_classes),
      (void**)&out_partition->site_classes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*out_partition->candidate_site_classes),
      (void**)&out_partition->candidate_site_classes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, pair_count, sizeof(*out_partition->pair_classes),
      (void**)&out_partition->pair_classes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_limit, sizeof(*out_partition->candidate_parent_classes),
      (void**)&out_partition->candidate_parent_classes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_limit, sizeof(*out_partition->candidate_outcomes),
      (void**)&out_partition->candidate_outcomes));
  memset(out_partition->site_classes, 0,
         site_count * sizeof(*out_partition->site_classes));
  return iree_ok_status();
}

void loom_decision_class_partition_begin(
    loom_decision_class_partition_t* partition, uint32_t outcome_count) {
  partition->candidate_class_count = 0;
  partition->active_outcome_count = outcome_count;
  memset(partition->pair_classes, 0xFF,
         (iree_host_size_t)partition->class_count * outcome_count *
             sizeof(*partition->pair_classes));
}

void loom_decision_class_partition_commit(
    loom_decision_class_partition_t* partition) {
  loom_decision_class_ordinal_t* previous_classes = partition->site_classes;
  partition->site_classes = partition->candidate_site_classes;
  partition->candidate_site_classes = previous_classes;
  partition->class_count = partition->candidate_class_count;
}
