// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_WORKER_MATERIALIZATION_H_
#define LOOM_TRANSFORMS_PIPELINE_WORKER_MATERIALIZATION_H_

#include "loom/analysis/pipeline/workers.h"
#include "loom/transforms/pipeline/channel_materialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consuming instance of an immutable, already selected worker occurrence.
// Source facts, ownership, completion, and transport proofs remain on the
// occurrence. This object translates only the identities needed by emission.
typedef struct loom_pipeline_worker_materialization_t {
  // Independently rewritable body, channel plan, CFG, and value correspondence.
  loom_channel_materialization_instance_t channels;
  // Source-to-instance asynchronous effect sites, indexed by source pointer.
  loom_ir_remap_op_projection_t* operations;
  // Number of selected transfer, group, and wait operations in operations.
  iree_host_size_t operation_count;
} loom_pipeline_worker_materialization_t;

// Clones without repeating channel, movement, CFG, or storage analysis.
// The source domain is acquired. Release it before restoring the instance's
// domain. The immutable source occurrence outlives the consuming rewrite.
iree_status_t loom_pipeline_worker_materialization_clone(
    loom_rewriter_t* rewriter, const loom_pipeline_worker_t* source,
    loom_symbol_ref_t symbol,
    loom_pipeline_worker_materialization_t* out_instance);

// Projects a selected asynchronous effect site. Source sites remain valid
// even after the corresponding instance operations have been consumed.
loom_op_t* loom_pipeline_worker_materialization_op(
    const loom_pipeline_worker_materialization_t* instance,
    const loom_op_t* source);

// Projects a value from the immutable source body. It must be a definition in
// the cloned callable; source facts and protocol identities are not SSA uses.
loom_value_id_t loom_pipeline_worker_materialization_value(
    const loom_pipeline_worker_materialization_t* instance,
    loom_value_id_t source);

// Emits an already analyzed byte-address recipe in the cloned body, preserving
// the source anchor's dominance. This translates emission operands only; it
// creates no second fact table or symbolic analysis for the instance.
iree_status_t loom_pipeline_worker_materialization_offset(
    const loom_pipeline_worker_materialization_t* instance,
    loom_builder_t* builder, const loom_symbolic_expr_t* source,
    loom_value_id_t anchor, loom_value_id_t* out_value);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_WORKER_MATERIALIZATION_H_
