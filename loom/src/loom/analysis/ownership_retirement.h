// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final-use boundaries over canonical carrier identities.
#ifndef LOOM_ANALYSIS_OWNERSHIP_RETIREMENT_H_
#define LOOM_ANALYSIS_OWNERSHIP_RETIREMENT_H_

#include "loom/analysis/liveness.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_ownership_retirement_entry_t {
  // Canonical function-local carrier ordinal to retire.
  loom_value_ordinal_t ordinal;
  // Call or structural-copy transport already retires this final-use input.
  bool transferred;
  // Next retirement at this boundary, or UINT32_MAX.
  uint32_t next;
} loom_ownership_retirement_entry_t;

typedef struct loom_ownership_retirement_plan_t {
  // List heads after each liveness operation, or NULL without selected values.
  uint32_t* operations;
  // List heads for unused arguments at each block entry.
  uint32_t* blocks;
  // Retained retirement entries shared by both boundary indexes.
  loom_ownership_retirement_entry_t* entries;
  // Number of initialized entries.
  iree_host_size_t entry_count;
} loom_ownership_retirement_plan_t;

// Builds last-use and dead-definition boundaries over one immutable flat body.
// |owners| maps value ordinals to canonical nonwriting aliases, or is NULL.
// |selected_words| selects carrier roots and has ceil(value_count / 64) words.
// Terminator payloads transfer at the boundary; edge liveness owns other
// deaths.
iree_status_t loom_ownership_retirement_build(
    const loom_local_value_domain_t* domain, const loom_value_ordinal_t* owners,
    const uint64_t* selected_words, const loom_liveness_analysis_t* liveness,
    iree_arena_allocator_t* arena, loom_ownership_retirement_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_ANALYSIS_OWNERSHIP_RETIREMENT_H_
