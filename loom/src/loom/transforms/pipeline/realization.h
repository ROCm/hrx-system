// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_REALIZATION_H_
#define LOOM_TRANSFORMS_PIPELINE_REALIZATION_H_

#include "loom/analysis/pipeline/invocations.h"
#include "loom/analysis/pipeline/workers.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/transforms/pipeline/worker_materialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resource inventory supplied by a physical execution owner. Storage packing
// and all occurrence analysis remain owned by the common realization lifecycle.
typedef struct loom_pipeline_realization_inventory_t {
  // Backing stores selected for this execution occurrence.
  const loom_pipeline_resource_pool_t* pools;
  // Number of distinct backing stores.
  iree_host_size_t pool_count;
  // Selectable memories in the admitted worker geometry.
  const loom_pipeline_resource_memory_t* memories;
  // Number of selectable memories.
  iree_host_size_t memory_count;
  // Caller-supplied pool arguments, if this execution imports storage.
  const loom_pipeline_resource_pool_binding_t* bindings;
  // Number of imported pool arguments.
  iree_host_size_t binding_count;
} loom_pipeline_realization_inventory_t;

// Complete immutable source occurrence presented to physical selection.
// Facts, channel identity and movement plans are established exactly once.
// Selection observes the source; emission consumes independent instances with
// their retained identities projected by the common cloning machinery.
typedef struct loom_pipeline_realization_t {
  // Construction whose public invocation contract is being implemented.
  loom_func_like_t function;
  // Immutable facts for the construction's captures and allocations.
  const loom_value_fact_table_t* facts;
  // Construction-owned resources, independent of physical storage addresses.
  loom_pipeline_resources_t resources;
  // Complete static inventory, including bound children and retained control.
  loom_pipeline_invocations_t invocations;
  // Bound worker instances, in complete invocations.workers order.
  loom_pipeline_worker_t* workers;
  // Independent consuming bodies, created only after selection succeeds.
  // A worker fully realized by other mechanics has a zero instance here.
  loom_pipeline_worker_materialization_t* instances;
  // Compilation owner receiving workers and their generated helper versions.
  loom_function_version_owner_t* version_owner;
  // Exact execution contexts retained at cloning, in the same worker order.
  // NULL for workers whose selected mechanics require no instruction body.
  loom_target_function_version_t** worker_versions;
  // Fresh context ordinal reserved for a changed entry execution contract.
  loom_target_context_ordinal_t entry_target_context;
} loom_pipeline_realization_t;

typedef struct loom_pipeline_realization_callback_t {
  // Admits target resources and chooses mechanics from canonical source plans.
  // No IR mutation occurs here. Authored support failures are diagnostics and
  // leave out_valid false; status carries allocation and emission failures.
  iree_status_t (*select)(void* user_data, loom_module_t* module,
                          const loom_pipeline_realization_t* realization,
                          bool* out_valid);
  // Queries the retained physical selection by worker ordinal. A descriptor
  // or configuration realization can consume the source plan without creating
  // an instruction body. This indexed fact performs no source analysis.
  bool (*has_worker_body)(const void* user_data, iree_host_size_t worker_index);
  // Consumes one admitted worker instance. Its ordinal domain is acquired;
  // the immutable source worker's domain remains released.
  // The callback consumes channel/movement plans and emits ordinary executable
  // operations; it does not rediscover the worker body or compile it privately.
  // Called only for workers whose selection requires an instruction body.
  // Original entry arguments remain in order until common capture realization
  // and pruning; mechanics may append invocation operands to that signature.
  iree_status_t (*worker)(void* user_data, loom_rewriter_t* rewriter,
                          const loom_pipeline_realization_t* realization,
                          iree_host_size_t worker_index);
  // Emits the execution container referring to the materialized workers. The
  // callback replaces the source definition and updates the active function
  // version when its callable or target contract changes.
  iree_status_t (*entry)(void* user_data, loom_rewriter_t* rewriter,
                         const loom_pipeline_realization_t* realization);
  // Borrowed physical selection and emission state.
  void* user_data;
} loom_pipeline_realization_callback_t;

// Registers an ordinary helper in its calling worker's exact execution context.
// The helper owns its own function contract; it inherits no entry retention,
// source schedules or memory proofs. Normal IR projection materializes the
// shared context at a saved compilation boundary.
iree_status_t loom_pipeline_realization_register_helper(
    loom_module_t* module, const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, loom_func_like_t helper,
    iree_diagnostic_emitter_t diagnostic_emitter);

// Consumes an already analyzed occurrence within its enclosing execution.
// The caller has closed source-stability proofs over all concurrent children.
// Source definitions stay immutable during selection and worker emission;
// entry emission owns replacing the admitted execution boundary. No resource,
// channel, movement, or source-stability analysis is repeated here.
// versions is the source snapshot already used to bind this occurrence's
// execution environment; generated definitions are registered with its owner.
iree_status_t loom_pipeline_realize_invocations(
    loom_pass_t* pass, loom_module_t* module,
    const loom_pipeline_invocations_t* invocations,
    loom_pipeline_worker_t* workers,
    const loom_target_function_version_snapshot_t* versions,
    const loom_pipeline_realization_callback_t* callback);

// Realizes one specialized pipeline after lexical loop preparation/outlining
// and before ordinary target memory legalization. Common code owns resource
// discovery, occurrence cloning, capture binding, analysis lifetime and worker
// version registration. The physical owner supplies an inventory and mechanics.
// Non-pipeline functions never enter this lifecycle. Generated workers resume
// in the existing compiler pipeline; this function starts no nested compiler.
iree_status_t loom_pipeline_realize(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_pipeline_realization_inventory_t* inventory,
    const loom_pipeline_realization_callback_t* callback);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_REALIZATION_H_
