// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-module cloning through caller-prepared symbol identity.

#ifndef LOOM_REWRITE_MODULE_PROJECTION_H_
#define LOOM_REWRITE_MODULE_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/rewrite/remap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepared correspondence used while cloning one complete source module.
//
// The caller owns target symbol creation and supplies one target reference for
// every source symbol. The correspondence remains valid after cloning and lets
// the caller address cloned symbols and live values without name lookup.
//
// The source and target modules and |target_symbols| must outlive the
// projection. The projection owns no allocations and requires no teardown.
typedef struct loom_ir_module_projection_t {
  // Immutable module whose complete operation body will be cloned.
  const loom_module_t* source_module;
  // Module receiving cloned operations and projected values.
  loom_module_t* target_module;
  // Caller-prepared target reference indexed by source symbol ID.
  const loom_symbol_ref_t* target_symbols;
  // Number of entries in |target_symbols|.
  iree_host_size_t target_symbol_count;
  // Source-to-target value correspondence populated while cloning live IR.
  loom_ir_remap_t remap;
  // Optional sparse operation correspondence populated during the same clone.
  struct {
    // Caller-owned entries in clone visitation order.
    loom_ir_remap_op_projection_t* entries;
    // Number of selected source operations.
    iree_host_size_t count;
  } operations;
} loom_ir_module_projection_t;

// Capacity and correspondence options for allocating an exact module clone.
typedef struct loom_ir_module_clone_options_t {
  // Additional string-table capacity reserved for caller materialization.
  iree_host_size_t additional_string_capacity;
  // Additional symbol-table capacity reserved for caller materialization.
  iree_host_size_t additional_symbol_capacity;
  // Optional sparse operation correspondence in clone visitation order.
  struct {
    // Caller-owned projection entries.
    loom_ir_remap_op_projection_t* entries;
    // Number of selected source operations.
    iree_host_size_t count;
  } operations;
} loom_ir_module_clone_options_t;

// Initializes a complete module projection.
//
// |target_symbols| must contain exactly one valid module-local target reference
// for each source symbol. Symbol selection, identity, collision handling, and
// declaration/definition merge policy are established by the caller before
// this operation. Several source symbols may name the same target identity only
// when the caller also ensures that at most one cloned definition owns it.
iree_status_t loom_ir_module_projection_initialize(
    const loom_module_t* source_module, loom_module_t* target_module,
    const loom_symbol_ref_t* target_symbols,
    iree_host_size_t target_symbol_count,
    loom_ir_module_projection_t* out_projection);

// Selects source operations whose cloned target pointers must be retained.
//
// |entries| must follow clone visitation order and remain live through
// loom_ir_module_projection_clone(). Every selected operation must be live in
// the complete source module. The clone fails if any entry is not observed.
iree_status_t loom_ir_module_projection_track_operations(
    loom_ir_module_projection_t* projection,
    loom_ir_remap_op_projection_t* entries, iree_host_size_t entry_count);

// Clones the complete source module body through |projection|.
//
// Temporary remap state is allocated from |scratch_arena|. Operation order,
// nested regions, CFG edges, source metadata, presentation, and all remappable
// payloads are preserved. Live value correspondence is indexed by source value
// ID; erased source rows remain unmapped and do not create target value holes.
// No linker, reachability, or merge policy is applied.
iree_status_t loom_ir_module_projection_clone(
    loom_ir_module_projection_t* projection,
    iree_arena_allocator_t* scratch_arena);

// Allocates and exactly clones |source_module| into a new owned module.
//
// Symbol IDs are preserved one-to-one and |out_projection| retains source to
// target symbol, value, and optionally selected operation correspondence.
// Projection storage belongs to |scratch_arena| and must remain live while the
// caller queries it. The returned module is owned by the caller.
iree_status_t loom_ir_module_clone(
    const loom_module_t* source_module,
    const loom_ir_module_clone_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    iree_allocator_t allocator, loom_ir_module_projection_t* out_projection,
    loom_module_t** out_module);

// Returns the prepared target reference for |source_symbol_id|.
static inline loom_symbol_ref_t loom_ir_module_projection_target_symbol(
    const loom_ir_module_projection_t* projection,
    loom_symbol_id_t source_symbol_id) {
  IREE_ASSERT(source_symbol_id < projection->target_symbol_count);
  return projection->target_symbols[source_symbol_id];
}

// Looks up the cloned target value corresponding to |source_value_id|.
static inline bool loom_ir_module_projection_try_target_value(
    const loom_ir_module_projection_t* projection,
    loom_value_id_t source_value_id, loom_value_id_t* out_target_value_id) {
  return loom_ir_remap_try_lookup_value(&projection->remap, source_value_id,
                                        out_target_value_id);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_REWRITE_MODULE_PROJECTION_H_
