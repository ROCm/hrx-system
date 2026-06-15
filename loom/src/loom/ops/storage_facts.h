// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Type-owned value facts for storage values.

#ifndef LOOM_OPS_STORAGE_FACTS_H_
#define LOOM_OPS_STORAGE_FACTS_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Storage reference facts carried by the storage type domain.
typedef struct loom_storage_reference_facts_t {
  // Byte offset facts relative to backing_value_id.
  loom_value_facts_t byte_offset;

  // Target-assigned byte offset facts, or unknown when layout is unassigned.
  loom_value_facts_t target_byte_offset;

  // Byte length facts for the valid subspan rooted at byte_offset.
  loom_value_facts_t valid_byte_length;

  // Minimum provable byte alignment at byte_offset.
  uint64_t minimum_alignment;

  // SSA value that owns the backing reservation identity.
  loom_value_id_t backing_value_id;

  // Storage space of the backing reference.
  loom_storage_space_t storage_space;
} loom_storage_reference_facts_t;

static_assert(sizeof(loom_storage_reference_facts_t) == 112,
              "storage reference facts must stay compact and padding-free");

// Fact domain attached to the generated storage type descriptor.
extern const loom_value_fact_domain_t loom_storage_fact_domain;

// Creates facts for a storage reference. Invalid or unbounded inputs degrade to
// no extension, which is the storage top/unknown value.
iree_status_t loom_storage_facts_make_reference(
    loom_fact_context_t* context,
    const loom_storage_reference_facts_t* reference,
    loom_value_facts_t* out_facts);

// Returns true and populates |out_reference| when |facts| carries a
// storage reference extension owned by |context|.
bool loom_storage_facts_query_reference(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_storage_reference_facts_t* out_reference);

// Creates reference facts for a static storage reservation root.
iree_status_t loom_storage_facts_make_reserve(
    loom_fact_context_t* context, loom_value_id_t storage_value_id,
    loom_storage_space_t storage_space, int64_t byte_length,
    int64_t byte_alignment, loom_value_facts_t* out_facts);

// Creates reference facts for a static subspan of an existing storage
// reference.
iree_status_t loom_storage_facts_make_static_view(
    loom_fact_context_t* context, loom_value_facts_t source_facts,
    int64_t byte_offset, int64_t byte_length, loom_value_facts_t* out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_STORAGE_FACTS_H_
