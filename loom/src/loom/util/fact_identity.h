// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_FACT_IDENTITY_H_
#define LOOM_UTIL_FACT_IDENTITY_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// A publication token shared by aliases. Union roots own live representatives;
// tokens remain distinct so deferred inference can observe a replacement even
// after queries have begun returning its new representative.
typedef struct loom_value_fact_identity_t {
  // Larger-rank union parent, or NULL for a root.
  struct loom_value_fact_identity_t* parent;
  // Next scope-owned allocation, including nodes subsequently united.
  struct loom_value_fact_identity_t* next;
  // Live SSA representative at a root, or INVALID after withdrawal. Non-root
  // nodes retain their prior representative ID for scope cleanup.
  loom_value_id_t representative;
  // Union rank, bounding read-only lookup depth logarithmically.
  uint32_t rank;
} loom_value_fact_identity_t;

static inline loom_value_fact_identity_t* loom_value_fact_identity_root(
    loom_value_fact_identity_t* identity) {
  while (identity->parent) {
    identity = identity->parent;
  }
  return identity;
}

// Publishes a declared identity after defining |value_id|'s numeric facts.
// Reports a changed publication without clearing an existing change report.
iree_status_t loom_value_fact_table_define_identity(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t source_value_id, bool* inout_changed);

// Retargets existing aliases before full SSA replacement can retire a value.
// The old SSA value can subsequently acquire new users with its own identity.
iree_status_t loom_value_fact_table_replace_identity_value(
    loom_value_fact_table_t* table, loom_value_id_t old_value,
    loom_value_id_t new_value);

// Withdraws references to a definition erased without a replacement.
void loom_value_fact_table_erase_identity_value(loom_value_fact_table_t* table,
                                                loom_value_id_t value_id);

// Undefines inferred alias membership, retaining a live own representative.
void loom_value_fact_table_undefine_identity(loom_value_fact_table_t* table,
                                             loom_value_id_t value_id);

// Clears publications and source tokens before resetting the transient arena.
void loom_value_fact_table_clear_identities(loom_value_fact_table_t* table);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_IDENTITY_H_
