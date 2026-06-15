// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic IR clone materialization.
//
// This layer clones operations and regions using an explicit loom_ir_remap_t.
// It does not decide whether cloning is profitable, whether a callable should
// inline, or how symbols should be imported. Callers bind source values in the
// remap and provide symbol policy before asking this helper to materialize IR.

#ifndef LOOM_REWRITE_MATERIALIZE_H_
#define LOOM_REWRITE_MATERIALIZE_H_

#include "iree/base/api.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/remap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options for cloning a source block's operation list into the current builder
// insertion block.
typedef struct loom_ir_clone_block_options_t {
  // Skips source ops whose vtable declares LOOM_TRAIT_TERMINATOR. This is used
  // by callable inlining wrappers that interpret the source terminator
  // separately instead of cloning it into the destination block.
  bool omit_terminators;
} loom_ir_clone_block_options_t;

// Options for moving a source block's operation list into another location in
// the same module.
typedef struct loom_ir_move_block_options_t {
  // Skips source ops whose vtable declares LOOM_TRAIT_TERMINATOR. This is used
  // when consuming a body whose terminator is interpreted by the caller, such
  // as callable inlining from a func.return-like terminator.
  bool omit_terminators;
} loom_ir_move_block_options_t;

// Clones |source_op| at the builder insertion point.
//
// Source operands, successors, result types, attributes, nested region
// payloads, locations, strings, encodings, and symbols are remapped through
// |remap|. Source result values are mapped to their cloned target result values
// before result types are remapped, so co-result dynamic type references are
// preserved. Bodyless func-like signature operands are cloned as fresh value
// definitions, not resolved as ordinary operand uses.
iree_status_t loom_ir_clone_op(loom_builder_t* builder,
                               const loom_op_t* source_op,
                               loom_ir_remap_t* remap,
                               loom_op_t** out_cloned_op);

// Clones every live op in |source_block| into the builder insertion block.
iree_status_t loom_ir_clone_block_ops(
    loom_builder_t* builder, const loom_block_t* source_block,
    loom_ir_remap_t* remap, const loom_ir_clone_block_options_t* options);

// Clones |source_region| and returns a new target-module-owned region.
iree_status_t loom_ir_clone_region(loom_builder_t* builder,
                                   const loom_region_t* source_region,
                                   loom_ir_remap_t* remap,
                                   loom_region_t** out_target_region);

// Rewrites every remappable payload in an existing op subtree through |remap|.
//
// This mutates ordinary operands, explicitly mapped successor targets, result
// types, attributes, nested block argument types, and nested child ops in-place
// through |rewriter| so use lists, type-use lists, effect summaries, and
// worklist state remain coherent. Result values and nested block arguments keep
// their existing IDs; this helper is for same-module move/consume transforms,
// not cross-module cloning.
iree_status_t loom_ir_remap_op_references(loom_rewriter_t* rewriter,
                                          loom_op_t* op,
                                          loom_ir_remap_t* remap);

// Moves live ops from |source_block| before |before_op| in the same module.
//
// Each moved op subtree is first checked after applying |remap|: all remapped
// operands, result-type references, attributes, nested block argument types,
// and nested child captures must be available immediately before |before_op|
// unless they are defined inside the op subtree being moved. The helper then
// remaps the existing op subtree in-place and splices it through the rewriter.
// This is a consuming move: callers that omit terminators are responsible for
// erasing or otherwise repairing the source owner before verification.
//
// The source block must be distinct from |before_op|'s parent block. Moving an
// entire block into itself is ambiguous because preserving order is either a
// no-op or a rotation depending on which prefix the caller intended.
iree_status_t loom_ir_move_block_ops_before(
    loom_rewriter_t* rewriter, loom_block_t* source_block, loom_op_t* before_op,
    loom_ir_remap_t* remap, const loom_ir_move_block_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_MATERIALIZE_H_
