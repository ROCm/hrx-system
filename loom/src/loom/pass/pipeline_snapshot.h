// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Independently owned source for one named pass pipeline.

#ifndef LOOM_PASS_PIPELINE_SNAPSHOT_H_
#define LOOM_PASS_PIPELINE_SNAPSHOT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// A selected pass.pipeline and its transitive pass.call closure.
//
// The snapshot owns |module|. |pipeline_op| points into that module and remains
// valid until the snapshot is deinitialized. Keeping this source independent
// lets a compiler materialize a different root closure for the subject module
// before compiling or executing the selected pipeline.
typedef struct loom_pass_pipeline_snapshot_t {
  // Owned module containing only the selected pipeline closure.
  loom_module_t* module;
  // Selected pass.pipeline operation in |module|.
  const loom_op_t* pipeline_op;
} loom_pass_pipeline_snapshot_t;

// Materializes |pipeline_symbol| and its transitive pass.call closure from
// |source_module|. The symbol may be spelled with or without leading '@'
// sigils. |identifier| names the snapshot module.
//
// The source module and context must remain immutable during this call but may
// be released or mutated after it returns. The caller must deinitialize the
// snapshot on both success and failure. Rejection emits a diagnostic and
// returns an empty snapshot with OK status.
iree_status_t loom_pass_pipeline_snapshot_initialize(
    const loom_module_t* source_module, iree_string_view_t pipeline_symbol,
    iree_string_view_t identifier, iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_pass_pipeline_snapshot_t* out_snapshot);

// Releases snapshot-owned module storage.
void loom_pass_pipeline_snapshot_deinitialize(
    loom_pass_pipeline_snapshot_t* snapshot);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_PIPELINE_SNAPSHOT_H_
