// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IR_MODULE_RECORD_H_
#define LOOM_IR_MODULE_RECORD_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// One attr-only module metadata operation and its exact canonical key.
typedef struct loom_module_record_t {
  // Module-owned record operation.
  const loom_op_t* op;
  // Generated operation metadata defining the record contract.
  const loom_op_vtable_t* vtable;
  // Exact string key used for ordering and duplicate identity.
  iree_string_view_t key;
  // Physical record ordinal used only to make duplicate diagnostics stable.
  iree_host_size_t physical_ordinal;
} loom_module_record_t;

// Canonically ordered projection of every keyed record in a module.
//
// Records are ordered first by exact operation name and then by exact key
// bytes. Physical module-body order is intentionally ignored. Duplicate
// identities remain adjacent so the verifier can diagnose them without a
// second index.
typedef struct loom_module_record_plan_t {
  // Canonically ordered record rows.
  loom_module_record_t* records;
  // Number of rows in |records|.
  iree_host_size_t record_count;
  // Plan-scoped storage backed by the module's arena block pool.
  iree_arena_allocator_t arena;
} loom_module_record_plan_t;

// Builds the canonical keyed-record projection for |module|.
//
// Generated vtable metadata and operation attributes are trusted compiler IR
// at this boundary. Callers consuming externally supplied IR verify the module
// before constructing the plan.
iree_status_t loom_module_record_plan_initialize(
    const loom_module_t* module, loom_module_record_plan_t* out_plan);

// Releases storage owned by |plan|.
void loom_module_record_plan_deinitialize(loom_module_record_plan_t* plan);

// Returns true when two adjacent canonical rows have the same identity.
static inline bool loom_module_record_identity_equal(
    const loom_module_record_t* lhs, const loom_module_record_t* rhs) {
  return lhs->op->kind == rhs->op->kind &&
         iree_string_view_equal(lhs->key, rhs->key);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_MODULE_RECORD_H_
