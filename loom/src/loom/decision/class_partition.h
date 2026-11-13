// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded partition refinement over finite decision outcomes.

#ifndef LOOM_DECISION_CLASS_PARTITION_H_
#define LOOM_DECISION_CLASS_PARTITION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Dense class ordinal local to one partition.
typedef uint16_t loom_decision_class_ordinal_t;

// Invalid class ordinal.
#define LOOM_DECISION_CLASS_ORDINAL_INVALID UINT16_MAX

// Mutable bounded partition over a fixed set of sites.
//
// Refinement maps each (current class, finite outcome) pair to one candidate
// class. Correlated decisions therefore retain only their observed quotient
// instead of forming an eager Cartesian product. All storage is allocated once
// during initialization. Small outcome vocabularies use direct indexing;
// larger ones retain only observed pairs in a bounded table. Refinement
// performs no allocation.
typedef struct loom_decision_class_partition_t {
  // Current dense class ordinal for every site.
  loom_decision_class_ordinal_t* site_classes;

  // Candidate dense class ordinal for every site during one refinement.
  loom_decision_class_ordinal_t* candidate_site_classes;

  // Reusable direct ordinal map or packed observed-pair table, selected by
  // begin(). Both representations map (parent class, outcome) to a new class.
  void* pair_storage;

  // Parent class for each candidate class created by the active refinement.
  loom_decision_class_ordinal_t* candidate_parent_classes;

  // Outcome for each candidate class created by the active refinement.
  uint32_t* candidate_outcomes;

  // Number of sites represented by the partition.
  iree_host_size_t site_count;

  // Maximum number of live classes.
  loom_decision_class_ordinal_t class_limit;

  // Current live class count.
  loom_decision_class_ordinal_t class_count;

  // Candidate class count in the active refinement.
  loom_decision_class_ordinal_t candidate_class_count;

  // Finite outcome count in the active refinement.
  uint32_t active_outcome_count;

  // Allocated pair-map bytes, bounded by sites and possible observed classes.
  uint32_t pair_storage_byte_length;

  // Active observed-pair bucket mask, or zero for direct indexing.
  uint32_t pair_bucket_mask;
} loom_decision_class_partition_t;

// Initializes a partition with all sites in class zero.
//
// |site_count|, |class_limit|, and |maximum_outcome_count| must be positive.
// The class limit must be below LOOM_DECISION_CLASS_ORDINAL_INVALID. Storage is
// owned by |arena| and remains live until the arena is reset.
iree_status_t loom_decision_class_partition_initialize(
    iree_host_size_t site_count, loom_decision_class_ordinal_t class_limit,
    uint32_t maximum_outcome_count, iree_arena_allocator_t* arena,
    loom_decision_class_partition_t* out_partition);

// Begins one refinement with outcomes in [0, |outcome_count|). The count must
// not exceed the maximum supplied during initialization.
void loom_decision_class_partition_begin(
    loom_decision_class_partition_t* partition, uint32_t outcome_count);

// Records one observed pair in the bounded table. The 48-bit parent/outcome
// identity occupies the high bits; the low 16 bits hold its dense class. An
// invalid parent ordinal keeps the all-ones empty word distinct from every key.
static inline bool loom_decision_class_partition_record_sparse(
    loom_decision_class_partition_t* partition, iree_host_size_t site_ordinal,
    uint32_t outcome) {
  const loom_decision_class_ordinal_t parent_class =
      partition->site_classes[site_ordinal];
  const uint64_t key = ((uint64_t)parent_class << 32) | outcome;
  uint32_t bucket = (uint32_t)((key * UINT64_C(11400714819323198485)) >> 32) &
                    partition->pair_bucket_mask;
  uint64_t* entries = (uint64_t*)partition->pair_storage;
  while (entries[bucket] != UINT64_MAX && (entries[bucket] >> 16) != key) {
    bucket = (bucket + 1) & partition->pair_bucket_mask;
  }
  loom_decision_class_ordinal_t candidate_class =
      (loom_decision_class_ordinal_t)entries[bucket];
  if (candidate_class == LOOM_DECISION_CLASS_ORDINAL_INVALID) {
    if (partition->candidate_class_count == partition->class_limit) {
      return false;
    }
    candidate_class = partition->candidate_class_count++;
    entries[bucket] = (key << 16) | candidate_class;
    partition->candidate_parent_classes[candidate_class] = parent_class;
    partition->candidate_outcomes[candidate_class] = outcome;
  }
  partition->candidate_site_classes[site_ordinal] = candidate_class;
  return true;
}

// Records one site's outcome in the active refinement.
//
// Returns false when accepting the outcome would exceed the class limit. The
// caller must abandon the active refinement in that case. Site ordinals must
// be supplied exactly once in ascending order before committing.
static inline bool loom_decision_class_partition_record(
    loom_decision_class_partition_t* partition, iree_host_size_t site_ordinal,
    uint32_t outcome) {
  if (IREE_UNLIKELY(partition->pair_bucket_mask != 0)) {
    return loom_decision_class_partition_record_sparse(partition, site_ordinal,
                                                       outcome);
  }
  const loom_decision_class_ordinal_t parent_class =
      partition->site_classes[site_ordinal];
  const iree_host_size_t pair_ordinal =
      (iree_host_size_t)parent_class * partition->active_outcome_count +
      outcome;
  loom_decision_class_ordinal_t* pair_classes =
      (loom_decision_class_ordinal_t*)partition->pair_storage;
  loom_decision_class_ordinal_t candidate_class = pair_classes[pair_ordinal];
  if (candidate_class == LOOM_DECISION_CLASS_ORDINAL_INVALID) {
    if (partition->candidate_class_count == partition->class_limit) {
      return false;
    }
    candidate_class = partition->candidate_class_count++;
    pair_classes[pair_ordinal] = candidate_class;
    partition->candidate_parent_classes[candidate_class] = parent_class;
    partition->candidate_outcomes[candidate_class] = outcome;
  }
  partition->candidate_site_classes[site_ordinal] = candidate_class;
  return true;
}

// Commits the active refinement and makes its candidate classes current.
void loom_decision_class_partition_commit(
    loom_decision_class_partition_t* partition);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_DECISION_CLASS_PARTITION_H_
