// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Storage lifetimes retained for block-local dependency construction.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_STORAGE_LIFETIME_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_STORAGE_LIFETIME_H_

#include "iree/base/api.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// An edge relation whose source is produced before the edge's block.
typedef struct loom_low_schedule_storage_handoff_t {
  // Exact relation in the scheduler's compact storage-relation index.
  uint32_t relation_index;
  // Next handoff owned by the same producer block, or UINT32_MAX.
  uint32_t next_handoff;
} loom_low_schedule_storage_handoff_t;

// Immutable scratch facts consumed while building each block's dependencies.
// These describe placement opportunities, not mandatory allocator aliases.
typedef struct loom_low_schedule_storage_lifetimes_t {
  // Same-block header reached through full-width, same-class copy/tied
  // relations, or the value's own ordinal when no such lifetime is retained.
  // NULL when no tracked storage can have a copied lifetime.
  loom_value_ordinal_t* roots;
  // Handoff list heads indexed by producer block, with UINT32_MAX for empty
  // lists. NULL when all handoffs are local to their edge's block.
  uint32_t* block_handoffs;
  // Exact cross-block handoffs linked through block_handoffs.
  loom_low_schedule_storage_handoff_t* handoffs;
} loom_low_schedule_storage_lifetimes_t;

struct loom_low_schedule_build_state_t;

// Builds the compact storage-relation index, marks tracked values, initializes
// reader scratch, and retains copy/tied lifetimes and producer-owned handoffs.
// All storage belongs to the schedule's scratch arena. Nodes and source
// producers must already be populated; no IR or register placement is changed.
iree_status_t loom_low_schedule_storage_lifetimes_initialize(
    struct loom_low_schedule_build_state_t* state, iree_host_size_t node_count);

// Marks storage retained beyond the block, through canonical live-outs or the
// endpoint's indexed branch payloads. Reverse analysis clears these flags after
// the block; dynamic scheduling clears its touched values before the next
// block. Strategies without a pressure-completion consumer require no
// projection.
void loom_low_schedule_storage_lifetimes_set_forwarded_values(
    struct loom_low_schedule_build_state_t* state, uint32_t block_index,
    bool is_forwarded);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_STORAGE_LIFETIME_H_
