// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_CHANNEL_MATERIALIZATION_H_
#define LOOM_TRANSFORMS_PIPELINE_CHANNEL_MATERIALIZATION_H_

#include "loom/analysis/channel_plan.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/cfg_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// An independently rewritable callable with the selected source plan projected
// into its value and operation identities. Protocol identities stay owned by
// the admitting construction, independently of this function's symbol.
typedef struct loom_channel_materialization_instance_t {
  // Cloned ordinary callable, owned by the source module.
  loom_func_like_t function;
  // Released domain whose ordinals match the source plan exactly.
  loom_local_value_domain_t value_domain;
  // Projected actions and exits, borrowing source membership and admission
  // ordinal tables. Admission values resolve through this instance's domain.
  loom_channel_plan_t plan;
  // Original CFG topology with instance block, terminator, and selector IDs.
  // Adjacency and traversal arrays borrow the immutable source graph.
  loom_cfg_graph_t graph;
  // Clone-owned SSA correspondence, independent of the active ordinal domain.
  loom_ir_remap_t remap;
} loom_channel_materialization_instance_t;

// Clones a selected callable occurrence before consuming channel rewrites.
// The source plan's domain is acquired and describes source_function's flat
// body. target_symbol is an existing same-module symbol with no definition.
// Cloning projects retained action sites and value ordinals without repeating
// channel analysis. It leaves the original definition and domain untouched.
//
// Release the source domain before restoring the instance's domain. The
// source ordinal tables and protocol identities outlive materialization.
// Ordinal-indexed selections keep their correspondence; action and return
// pointers refer only to the clone. The CFG retains its original topology;
// observer translates additional selected effect sites during the same clone.
// Initial runtime state, other selected SSA
// operands and dependencies of physical carrier types must use the instance's
// value_ids at their original ordinals. Cloned source types already refer to
// the corresponding cloned values.
// The enclosing lowering owns callsite/signature realization and admission;
// this helper neither retargets callers nor selects transport mechanics.
iree_status_t loom_channel_materialization_clone(
    loom_rewriter_t* rewriter, loom_func_like_t source_function,
    const loom_channel_plan_t* source_plan,
    const loom_cfg_graph_t* source_graph, loom_symbol_ref_t target_symbol,
    loom_ir_clone_observer_t observer,
    loom_channel_materialization_instance_t* out_instance);

// Emission mechanics for one already selected channel realization. Selection
// proves legality and retains resource bindings before this consuming rewrite.
typedef struct loom_channel_materialization_callback_t {
  // Emits immediately before action->op. Its operands already contain the
  // physical carriers substituted by previous actions. The retained action
  // supplies protocol identities independently of those carriers' addresses.
  // Writes one replacement per materialized result to results (erased result
  // slots are ignored) and updates state with successor cursor/control values.
  // The callback inserts ordinary operations but leaves the source operation
  // and containing CFG intact;
  // this common owner replaces results, erases actions and threads CFG state.
  // Status carries allocation or emission failure, not realization selection.
  iree_status_t (*fn)(void* user_data, loom_rewriter_t* rewriter,
                      const loom_channel_plan_action_t* action,
                      loom_value_id_t* state, iree_host_size_t state_count,
                      loom_value_id_t* results);
  // Optional execution-exit mechanics emitted before an original func.return,
  // after all actions in its block. Receives the path's final protocol state.
  // The realization owner selects this for an execution boundary that owns
  // outstanding transfers; an ordinary helper return does not imply a drain.
  // Leaves the return and CFG intact. NULL emits no completion operation.
  iree_status_t (*exit)(void* user_data, loom_rewriter_t* rewriter,
                        const loom_op_t* terminator,
                        const loom_value_id_t* state,
                        iree_host_size_t state_count);
  // Borrowed selected bindings, helper symbols and target emission context.
  void* user_data;
} loom_channel_materialization_callback_t;

typedef struct loom_channel_materialization_options_t {
  // Carrier type by original value ordinal, or a null type when unchanged.
  // Block arguments change here; operation results are replaced by emission.
  // The boundary owner also realizes function result types and all callsites.
  const loom_type_t* carrier_types;
  // Optional byte per original value ordinal selecting zero-bit carriers.
  // Every use must be a retained action or canonical CFG transport. Selected
  // emission cannot use these values; their ownership effects become code or
  // are discharged by the realization proof, never by a fabricated SSA value.
  const uint8_t* erased_values;
  // Initial private protocol state, available before the region's first action.
  // These are actual cursor/control values, not synthetic dependency tokens.
  const loom_value_id_t* initial_state;
  // Number of independently threaded SSA state values; zero needs no CFG edit.
  iree_host_size_t state_count;
  // Mechanics selected for every retained action, including channel-aware
  // calls.
  loom_channel_materialization_callback_t emit;
} loom_channel_materialization_options_t;

// Consumes a fixed-channel plan into ordinary executable IR. The plan's local
// value domain stays acquired; queries are only for original source values.
// Other source analyses are invalidated before this batch. Intermediate source
// operations can still refer to carriers whose types have already changed.
// The retained CFG describes the original blocks and edges. Reachable blocks
// with one predecessor inherit its SSA state directly; joins carry explicit
// arguments through loops and reconvergence. Only conditional edges entering
// those joins need forwarding blocks. Unreachable blocks retain explicit state
// arguments. Exit mechanics see the same path-specific state. The common owner
// imposes no synchronization or global drain.
//
// Source ownership and realization legality have already been established.
// The caller owns callable signature conversion and dead channel-argument
// pruning after all consumers have been realized. No fabricated channel value
// is introduced to keep an erased protocol argument alive.
iree_status_t loom_channel_materialize(
    loom_rewriter_t* rewriter, const loom_channel_plan_t* plan,
    const loom_cfg_graph_t* graph,
    const loom_channel_materialization_options_t* options);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_CHANNEL_MATERIALIZATION_H_
