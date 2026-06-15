// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation materialization for target-low functions.
//
// Allocation tables keep physical placement and spill decisions outside the
// SSA program. This layer turns selected table decisions back into explicit
// low IR when later emission needs real storage traffic. The materialized IR
// remains ordinary Loom SSA: spill stores consume virtual register values,
// reloads define new virtual register values, and only the table describes
// final physical placement.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MATERIALIZATION_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MATERIALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_allocation_materialization_options_t {
  // Allows a function body that already contains low.spill or low.reload ops.
  // The default false value rejects existing traffic so repeated or partial
  // materialization fails loud instead of stacking generated reloads.
  bool allow_existing_storage_traffic;
  // True when |supported_storage_spaces| names the complete target lowering
  // capability for newly generated spill storage.
  bool has_supported_storage_spaces;
  // Storage spaces this materialization may create for spill slots.
  loom_low_storage_space_set_t supported_storage_spaces;
  // True when successful spill insertion should emit BACKEND/009 feedback.
  bool emit_spill_diagnostics;
  // Maximum number of current spill plans to materialize. Zero materializes all
  // plans in the table. Passes that reallocate greedily use a small limit so
  // CFG/signature rewrites can invalidate obsolete later plans before they are
  // turned into unnecessary storage traffic.
  iree_host_size_t max_spill_plan_count;
  // Structured diagnostic emitter for materialized allocation feedback.
  iree_diagnostic_emitter_t emitter;
} loom_low_allocation_materialization_options_t;

typedef struct loom_low_allocation_materialization_result_t {
  // Number of error diagnostics emitted before mutating the low function.
  uint32_t error_count;
  // Number of low.storage.reserve ops created from spill plans.
  uint32_t storage_count;
  // Number of low.spill stores inserted.
  uint32_t spill_count;
  // Number of low.reload ops inserted.
  uint32_t reload_count;
} loom_low_allocation_materialization_result_t;

// Materializes spill plans in |table| into low.storage.reserve, low.spill, and
// low.reload ops. The table must describe |module| and a target-low function
// inside it. Storage reservations are inserted in the low entry block after
// ABI live-in/resource imports and existing storage reservations. Stores are
// inserted at the defining point of each spilled op-result value, non-entry
// block argument stores are inserted on incoming edges before the branch, and
// reloads are inserted immediately before each remaining original operand use.
// Materialized non-entry block arguments are removed from the block signature
// and from all predecessor low.br payloads.
iree_status_t loom_low_allocation_materialize_spills(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MATERIALIZATION_H_
