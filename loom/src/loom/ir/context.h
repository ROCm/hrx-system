// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Global compilation context: dialect registration, op vtable and semantic
// metadata lookup, and context-owned encoding vtables.
//
// Lifecycle:
//   1. loom_context_initialize() — zero-init with allocator.
//   2. loom_context_register_dialect() — register each dialect's vtables.
//   3. loom_context_register_dialect_semantics() — register dialect metadata.
//   4. loom_context_finalize() — build acceleration structures.
//   5. Use: create modules, parse, compile, verify.
//   6. loom_context_deinitialize() — release resources.
//
// After finalization, the dialect/op registries and op-name table are
// immutable and can be read from any thread without synchronization.

#ifndef LOOM_IR_CONTEXT_H_
#define LOOM_IR_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/semantics.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// loom_context_t
//===----------------------------------------------------------------------===//

// A dialect's dense operation metadata. Each dialect registers its static
// vtable array at context creation and may then attach a matching semantic
// metadata array. Both arrays are indexed by op_index within the dialect and
// live for the process lifetime.
typedef struct loom_dialect_vtables_t {
  // Number of operation slots in the dialect arrays.
  uint16_t op_count;
  // Dense generated vtable array, indexed by dialect-local operation index.
  const loom_op_vtable_t* const* entries;
  // Dense generated semantic metadata array, or NULL when none is registered.
  const loom_op_semantics_t* semantics;
} loom_dialect_vtables_t;

// Two-level op metadata registry: dialect table indexed by dialect_id, each
// entry pointing to dense per-dialect arrays indexed by op_index. Lookup is two
// array indexes — same cost as a flat array but uses <2KB instead of 18KB.
typedef struct loom_op_vtable_registry_t {
  // Built-in dialect metadata registrations indexed by dialect ID.
  loom_dialect_vtables_t dialects[LOOM_DIALECT_BUILTIN_COUNT_];
} loom_op_vtable_registry_t;

// Entry in the op name hash table. Maps a dotted op name string
// (e.g., "test.addi") to the op kind and vtable pointer. The name
// field is a view into the vtable's B-string — no allocation.
typedef struct loom_op_name_entry_t {
  // Borrowed dotted op name string from the registered op vtable.
  iree_string_view_t name;
  // Encoded dialect-local op kind resolved from |name|.
  loom_op_kind_t kind;
  // Borrowed vtable pointer for the resolved op kind.
  const loom_op_vtable_t* vtable;
} loom_op_name_entry_t;

// Open-addressed hash table for O(1) op name → vtable resolution.
// Built during loom_context_finalize(). Allocated with the context's
// host allocator. Uses FNV-1a hashing and linear probing.
typedef struct loom_op_name_table_t {
  // Allocated hash table entries; NULL until context finalization.
  loom_op_name_entry_t* entries;
  // Entry capacity. Always a power of two when non-zero.
  uint32_t capacity;
  // Number of occupied entries.
  uint32_t count;
} loom_op_name_table_t;

// The global context: vtables, allocator, and shared language registries.
//
// Created once at startup, shared across all modules and threads.
// Dialect/op lookup state is immutable after finalization.
//
// Lifetime: the context must outlive all modules created from it.
struct loom_context_t {
  iree_allocator_t allocator;

  // Context-owned encoding vtables.
  loom_encoding_vtable_list_t encoding_vtables;

  // Op vtable registry: two-level lookup by dialect_id then op_index.
  loom_op_vtable_registry_t op_vtables;

  // Op name hash table for string → vtable resolution.
  loom_op_name_table_t op_name_table;
};

// Initializes a context with the given host allocator. After initialization,
// register dialects and populate any context-owned encoding tables, then call
// loom_context_finalize().
void loom_context_initialize(iree_allocator_t allocator,
                             loom_context_t* out_context);

// Releases all resources owned by the context. All modules created
// from this context must have been freed first.
void loom_context_deinitialize(loom_context_t* context);

// Registers a dialect's vtable array with the context. The |vtables|
// pointer must remain valid for the lifetime of the context (typically
// a static array generated from the DSL). Returns INVALID_ARGUMENT if
// the dialect ID is out of range, ALREADY_EXISTS if already registered.
// Must be called before loom_context_finalize().
iree_status_t loom_context_register_dialect(
    loom_context_t* context, uint8_t dialect_id,
    const loom_op_vtable_t* const* vtables, uint16_t op_count);

// Registers a dialect's semantic metadata array with the context.
//
// The dialect must already be registered through loom_context_register_dialect,
// and |op_count| must match that dialect's vtable count exactly. The
// |semantics| pointer must remain valid for the lifetime of the context
// (typically a static array generated from the DSL). Returns INVALID_ARGUMENT
// if the dialect ID is out of range or the metadata is missing,
// FAILED_PRECONDITION if the dialect vtables were not registered first or the
// count differs, and ALREADY_EXISTS if semantic metadata was already
// registered.
iree_status_t loom_context_register_dialect_semantics(
    loom_context_t* context, uint8_t dialect_id,
    const loom_op_semantics_t* semantics, uint16_t op_count);

// Registers one encoding family vtable with the context.
//
// `vtable->name` must be non-empty and stable for the context lifetime
// (usually static storage). Family names are unique; duplicate registration
// returns ALREADY_EXISTS. Runtime callbacks may be NULL if the family only
// needs parser/verifier visibility for now.
iree_status_t loom_context_register_encoding_vtable(
    loom_context_t* context, const loom_encoding_vtable_t* vtable);

// Finalizes the context after dialect registration and encoding-table setup.
// Builds acceleration structures for fast op name lookup. Must be called
// before creating modules or parsing.
iree_status_t loom_context_finalize(loom_context_t* context);

// Resolves an op kind to its vtable. Returns NULL if the dialect is
// not registered or the op index is out of range.
const loom_op_vtable_t* loom_context_resolve_op(const loom_context_t* context,
                                                loom_op_kind_t kind);

// Resolves an op kind to its semantic metadata. Returns empty metadata if the
// dialect is not registered, no semantic metadata is registered for that
// dialect, or the op index is out of range.
loom_op_semantics_t loom_context_resolve_op_semantics(
    const loom_context_t* context, loom_op_kind_t kind);

// Looks up an op by its dotted name string (e.g., "test.addi").
// Returns the vtable pointer, or NULL if not found. On success,
// |out_kind| is set to the op kind. The context must be finalized.
const loom_op_vtable_t* loom_context_lookup_op_by_name(
    const loom_context_t* context, iree_string_view_t name,
    loom_op_kind_t* out_kind);

// Looks up an encoding family by its bare name (`q8_0`, `dense`, ...).
// Returns NULL when no matching family has been registered.
const loom_encoding_vtable_t* loom_context_lookup_encoding_vtable(
    const loom_context_t* context, iree_string_view_t name);

//===----------------------------------------------------------------------===//
// Op convenience accessors
//===----------------------------------------------------------------------===//

// Returns the vtable for |op|, or NULL if no vtable is registered.
const loom_op_vtable_t* loom_op_vtable(const loom_module_t* module,
                                       const loom_op_t* op);

// Returns the dotted name of |op| (e.g., "test.addi"). Returns
// "unknown" if no vtable is registered for the op's kind.
iree_string_view_t loom_op_name(const loom_module_t* module,
                                const loom_op_t* op);

// Returns the semantic metadata for |op|, or empty metadata if no metadata is
// registered for the op's kind.
loom_op_semantics_t loom_op_semantics(const loom_module_t* module,
                                      const loom_op_t* op);

// Returns true if |op| has the given trait bit(s) set in its vtable.
// Returns false if no vtable is registered for the op's kind.
bool loom_op_has_trait(const loom_module_t* module, const loom_op_t* op,
                       loom_trait_flags_t trait);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_CONTEXT_H_
