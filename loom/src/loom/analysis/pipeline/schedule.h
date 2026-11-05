// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_SCHEDULE_H_
#define LOOM_ANALYSIS_PIPELINE_SCHEDULE_H_

#include "loom/error/emitter.h"
#include "loom/util/fact_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pipeline_schedule_t loom_pipeline_schedule_t;

typedef enum loom_pipeline_schedule_step_kind_e {
  // Pure source dataflow whose results feed construction or child arguments.
  LOOM_PIPELINE_SCHEDULE_STEP_DATAFLOW,
  // One complete child invocation, with fresh child-local resource lifetime.
  LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION,
  // A nested ordered scope, complete before this block advances.
  LOOM_PIPELINE_SCHEDULE_STEP_SERIAL,
  // A backing selection, resolved in the enclosing invocation's inventory.
  LOOM_PIPELINE_SCHEDULE_STEP_MEMORY,
  // Fresh storage whose execution and lifetime remain in this control scope.
  LOOM_PIPELINE_SCHEDULE_STEP_ALLOCATION,
  // Fresh protocol identity, independent of its backing storage identity.
  LOOM_PIPELINE_SCHEDULE_STEP_CHANNEL,
  // Control transfer whose successors and arguments belong to the retained CFG.
  LOOM_PIPELINE_SCHEDULE_STEP_BRANCH,
  // Completion of this serial scope, including its final child.
  LOOM_PIPELINE_SCHEDULE_STEP_YIELD,
} loom_pipeline_schedule_step_kind_t;

// A static source site. A loop visits the same site repeatedly; it does not
// manufacture another plan row or another copy of the child's program.
typedef struct loom_pipeline_schedule_step_t {
  // Source operation with its typed arguments, results, and source location.
  loom_op_t* operation;
  // Complete nested scope for SERIAL; NULL for every other step kind.
  const loom_pipeline_schedule_t* nested;
  // Semantics classified once by the construction owner.
  loom_pipeline_schedule_step_kind_t kind;
} loom_pipeline_schedule_step_t;

typedef struct loom_pipeline_schedule_block_t {
  // Source steps in block order, including the control-transfer terminator.
  const loom_pipeline_schedule_step_t* steps;
  // Number of retained source sites in this block.
  iree_host_size_t step_count;
} loom_pipeline_schedule_block_t;

// One ordered child scope. Top-level scopes progress independently of outer
// strands and other top-level scopes; nesting orders complete children only.
struct loom_pipeline_schedule_t {
  // Authored serial operation whose region owns this scope's SSA values.
  loom_op_t* declaration;
  // Borrowed canonical CFG, loops, and recurrences from the value-fact owner.
  const loom_value_fact_cfg_region_t* control;
  // Arena-owned classified steps, indexed like control->graph.blocks.
  const loom_pipeline_schedule_block_t* blocks;
  // Acyclic pure control selecting at most one child on every path, without
  // scope-local resources or nested schedules. Start and retirement can be
  // projected separately without moving an intermediate child completion.
  bool single_child_choice;
};

// Retains one serial scope and its nested scopes after structured control has
// been lowered to CFG. Each region is visited once. Child compositions remain
// boundaries; this analysis neither traverses callees nor assigns workers,
// addresses, synchronization, or storage reuse. Those consumers retain the
// scope/occurrence correspondence instead of rediscovering it from source.
//
// Represented source and established facts remain immutable through
// consumption. Source edits invalidate this plan; a consuming rewrite
// translates its retained sites.
// The fact scope outlives the plan, and all new plan storage belongs to arena.
// Unsupported construction emits a diagnostic and leaves out_valid false;
// statuses carry allocation or diagnostic-sink failures.
iree_status_t loom_pipeline_schedule_build(
    loom_module_t* module, loom_op_t* declaration,
    loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_schedule_t** out_schedule, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_SCHEDULE_H_
