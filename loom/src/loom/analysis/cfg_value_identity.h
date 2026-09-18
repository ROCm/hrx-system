// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact SSA identities established by CFG forwarding.
//
// Block arguments may carry the same value through correlated joins and
// cycles even when their numeric facts have widened. This analysis preserves
// those identities as direct representatives in a function-local value domain.
// Consumers perform one indexed lookup and never walk the CFG or reconstruct
// forwarding relationships.

#ifndef LOOM_ANALYSIS_CFG_VALUE_IDENTITY_H_
#define LOOM_ANALYSIS_CFG_VALUE_IDENTITY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_cfg_value_identity_table_t {
  // Caller-owned function-local value domain indexing representatives.
  const loom_local_value_domain_t* value_domain;

  // Direct representative indexed by local value ordinal. Invalid entries
  // represent values for which no CFG identity has been established.
  loom_value_id_t* representatives;

  // Number of entries in representatives.
  loom_value_ordinal_t representative_count;
} loom_cfg_value_identity_table_t;

// Initializes an empty identity table covering the current value domain.
// |value_domain| must remain acquired and unchanged for the table lifetime.
iree_status_t loom_cfg_value_identity_table_initialize(
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_cfg_value_identity_table_t* out_table);

// Adds identities proven from one retained CFG snapshot.
//
// The snapshot and dominance information are borrowed for the call. Temporary
// occurrence, partition, and obligation storage is released before returning;
// only direct representatives remain in |table|. A valid CFG whose branch
// payloads cannot be represented by the generic CFG interface contributes no
// identities.
iree_status_t loom_cfg_value_identity_table_update(
    loom_cfg_value_identity_table_t* table,
    const loom_value_fact_cfg_region_t* region,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena);

// Returns the direct representative for |value_id|, or |value_id| when no
// identity has been established.
static inline loom_value_id_t loom_cfg_value_identity_table_lookup(
    const loom_cfg_value_identity_table_t* table, loom_value_id_t value_id) {
  if (table == NULL || table->value_domain == NULL) {
    return value_id;
  }
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_try_ordinal(table->value_domain, value_id);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      ordinal >= table->representative_count) {
    return value_id;
  }
  const loom_value_id_t representative = table->representatives[ordinal];
  return representative == LOOM_VALUE_ID_INVALID ? value_id : representative;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CFG_VALUE_IDENTITY_H_
