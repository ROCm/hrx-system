// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector-to-scalar reference lowering.
//
// This pass exposes vector lane semantics using scalar ops and scf.for loops
// while preserving function ABI. Vector arguments/results/calls/returns remain
// vector-typed; vector.extract/vector.insert/vector.from_elements are the
// aggregate boundary ops used to move between vector values and scalar lane
// programs.

#ifndef LOOM_TRANSFORMS_VECTOR_TO_SCALAR_H_
#define LOOM_TRANSFORMS_VECTOR_TO_SCALAR_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/vector/to_scalar_options.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_vector_to_scalar_pass_info(void);

iree_status_t loom_vector_to_scalar_run(loom_pass_t* pass,
                                        loom_module_t* module,
                                        loom_func_like_t function);

const loom_pass_info_t* loom_vector_reduce_axes_to_scalar_pass_info(void);

iree_status_t loom_vector_reduce_axes_to_scalar_run(loom_pass_t* pass,
                                                    loom_module_t* module,
                                                    loom_func_like_t function);

// Rewrites one vector.reduce.axes op using the same scalar reference lowering
// as the standalone pass. Statistics for lane and loop materialization are
// recorded in the current pass when provided; the caller owns op-rewrite
// accounting.
iree_status_t loom_vector_reduce_axes_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

// Rewrites one vector.reduce op using scalar reference semantics.
iree_status_t loom_vector_reduce_to_scalar_rewrite_op(loom_pass_t* pass,
                                                      loom_rewriter_t* rewriter,
                                                      loom_op_t* op,
                                                      bool* out_rewritten);

// Rewrites one vector.dotf op using scalar reference semantics.
iree_status_t loom_vector_dotf_to_scalar_rewrite_op(loom_pass_t* pass,
                                                    loom_rewriter_t* rewriter,
                                                    loom_op_t* op,
                                                    bool* out_rewritten);

// Rewrites one descriptor-backed vector op using scalar reference semantics.
// This is the generic lane-by-lane expansion used by the standalone pass for
// arithmetic, bitwise, structural, and packed vector ops represented in the
// vector-to-scalar descriptor table.
iree_status_t loom_vector_descriptor_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

// Rewrites one vector.mma op using scalar reference semantics. Dense logical
// fragments lower directly; target-shaped physical fragments require options
// that provide the selected matrix-fragment layout and permit subgroup
// communication in the current source placement.
iree_status_t loom_vector_mma_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    loom_vector_mma_to_scalar_options_t options, bool* out_rewritten);

// Returns rejection flags explaining why the scalar reference lowering would
// refuse one vector.mma op under |options|. This is a cold reporting predicate
// for target legalization; it does not emit IR or diagnostics.
uint32_t loom_vector_mma_to_scalar_reference_rejection_bits(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    loom_vector_mma_to_scalar_options_t options);

// Returns the first role-local rejection detail explaining why the scalar
// reference lowering would refuse one vector.mma op under |options|.
uint32_t loom_vector_mma_to_scalar_reference_rejection_detail(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    loom_vector_mma_to_scalar_options_t options);

// Rewrites one vector.store into scalar view.store loops. The source vector is
// consumed lane-by-lane so supported producer trees can disappear through DCE
// without first materializing a dynamic vector aggregate.
iree_status_t loom_vector_store_to_scalar_rewrite_op(loom_pass_t* pass,
                                                     loom_rewriter_t* rewriter,
                                                     loom_op_t* op,
                                                     bool* out_rewritten);

// Rewrites one vector.fragment.store into scalar view.store loops over the
// fragment's logical matrix shape. The source fragment is consumed
// lane-by-lane so target-shaped producers can be erased without first
// materializing a dense vector aggregate.
iree_status_t loom_vector_fragment_store_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

// Returns target-independent rejection flags explaining why the scalar
// reference lowering would refuse one vector.fragment.store op.
uint32_t loom_vector_fragment_store_to_scalar_reference_rejection_bits(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op);

// Rewrites one scalar-result vector.extract when its lane can be rematerialized
// from the source producer tree.
iree_status_t loom_vector_extract_to_scalar_rewrite_op(
    loom_pass_t* pass, loom_rewriter_t* rewriter, loom_op_t* op,
    bool* out_rewritten);

const loom_pass_info_t* loom_vector_gather_to_scalar_pass_info(void);

iree_status_t loom_vector_gather_to_scalar_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_TO_SCALAR_H_
