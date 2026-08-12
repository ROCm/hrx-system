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
} loom_ir_module_projection_t;

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

// Clones the complete source module body through |projection|.
//
// Temporary remap state is allocated from |scratch_arena|. Operation order,
// nested regions, CFG edges, source metadata, presentation, and all remappable
// payloads are preserved. No linker, reachability, or merge policy is applied.
iree_status_t loom_ir_module_projection_clone(
    loom_ir_module_projection_t* projection,
    iree_arena_allocator_t* scratch_arena);

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
