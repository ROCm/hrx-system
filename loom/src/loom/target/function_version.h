// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target facts owned by one concrete compiler function version.

#ifndef LOOM_TARGET_FUNCTION_VERSION_H_
#define LOOM_TARGET_FUNCTION_VERSION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/function_version.h"
#include "loom/target/facts.h"
#include "loom/target/resolved_target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_function_version_t {
  // Generic compiler function-version base. Must remain the first field.
  loom_function_version_t base;

  // Authored target witness name used for diagnostics and reports. Empty when
  // the function had no authored target.
  iree_string_view_t authored_target_name;

  // Optional target requirement projected from the authored target witness.
  const loom_target_facts_t* target_requirement_facts;

  // Exact invocation context inherited by retained semantic callees.
  //
  // The provider remains paired with the profile projection and authored
  // target requirement so artifact boundaries can materialize the target
  // without rediscovering its family. Its facts exclude function-local ABI or
  // export overlays.
  loom_resolved_target_t resolved_target;

  // Non-NULL function target facts used to compile this version, including
  // its function-local ABI and export contract.
  const loom_target_facts_t* function_target_facts;
} loom_target_function_version_t;

// Static identity for target-refined function versions.
extern const loom_function_version_type_t loom_target_function_version_type;

// Returns |version| as a target-refined version, or NULL for another type.
static inline loom_target_function_version_t* loom_target_function_version_cast(
    loom_function_version_t* version) {
  return version != NULL && version->type == &loom_target_function_version_type
             ? (loom_target_function_version_t*)version
             : NULL;
}

// Returns |version| as a target-refined version, or NULL for another type.
static inline const loom_target_function_version_t*
loom_target_function_version_const_cast(
    const loom_function_version_t* version) {
  return version != NULL && version->type == &loom_target_function_version_type
             ? (const loom_target_function_version_t*)version
             : NULL;
}

// Returns function target facts carried by |version|, or NULL for another type.
static inline const loom_target_facts_t*
loom_target_function_version_target_facts(
    const loom_function_version_t* version) {
  const loom_target_function_version_t* target_version =
      loom_target_function_version_const_cast(version);
  if (target_version == NULL) return NULL;
  return target_version->function_target_facts;
}

// Finds the target-refined version currently implemented by |function|.
loom_target_function_version_t* loom_target_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function);

// Target-refined versions observed against one module symbol snapshot.
//
// Entries are indexed directly by module-local symbol ID. The snapshot is
// valid only until the module adds, erases, replaces, or compacts symbols.
// Compiler version objects remain the semantic identity; this transient vector
// only reconciles them with one mutable module observation boundary.
typedef struct loom_target_function_version_snapshot_t {
  // Arena-owned pointer vector borrowing stable version handles by module
  // symbol ID.
  loom_function_version_t** version_handles_by_symbol;

  // Number of module symbols represented by |version_handles_by_symbol|.
  iree_host_size_t symbol_count;
} loom_target_function_version_snapshot_t;

// Observes |function_versions| against the current |module| symbol table.
//
// Construction is O(symbol_count + version_count). The returned snapshot is
// allocated from |arena| and must be discarded before the module symbol table
// changes. Each target-refined version must name a distinct live function
// symbol in |module|.
iree_status_t loom_target_function_version_snapshot_build(
    const loom_module_t* module,
    const loom_function_version_list_t* function_versions,
    iree_arena_allocator_t* arena,
    loom_target_function_version_snapshot_t* out_snapshot);

// Returns the stable generic version handle observed at |symbol_id|, or NULL.
//
// The handle may be transferred to a replacement function after the snapshot
// is no longer used. Target-family facts remain immutable.
static inline loom_function_version_t*
loom_target_function_version_snapshot_handle_at(
    const loom_target_function_version_snapshot_t* snapshot,
    loom_symbol_id_t symbol_id) {
  return snapshot->version_handles_by_symbol != NULL
             ? snapshot->version_handles_by_symbol[symbol_id]
             : NULL;
}

// Returns an immutable target-refined view observed at |symbol_id|, or NULL.
static inline const loom_target_function_version_t*
loom_target_function_version_snapshot_at(
    const loom_target_function_version_snapshot_t* snapshot,
    loom_symbol_id_t symbol_id) {
  return loom_target_function_version_const_cast(
      loom_target_function_version_snapshot_handle_at(snapshot, symbol_id));
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_FUNCTION_VERSION_H_
