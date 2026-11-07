// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-call argument facts retained for source-to-low planning.

#ifndef LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_INVOCATION_FACTS_H_
#define LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_INVOCATION_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"
#include "loom/util/call_graph.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_invocation_facts_t {
  // Entry and private-callee facts indexed by argument value ID.
  loom_value_fact_table_t arguments;
  // Complete source call graph retained for dependent module planning.
  loom_call_graph_t call_graph;
  // One when the corresponding module symbol has a retained direct semantic
  // caller after source inlining. NULL when the module has no semantic calls.
  const uint8_t* direct_call_targets;
  // Number of entries in |direct_call_targets|.
  iree_host_size_t direct_call_target_count;
} loom_low_source_invocation_facts_t;

// Computes entry and private acyclic call facts before source lowering.
// Modules without semantic calls return an empty summary without allocating or
// walking IR; the module maintains that inventory as operations mutate.
//
// Every retained direct semantic caller participates in each callee argument
// meet, independent of whether source policy or profitability retained the
// boundary. Target-required inline edges consume caller facts after inlining.
// HAL launch-only entries contribute their ABI uniformity. Exported and
// recursive callees retain their declared value domains and admit lane-varying
// arguments on targets with more than one lane per subgroup.
// Region-declared uniformity comes from the ordinary fact solve instead of
// being duplicated in this summary.
// Selected functions use their exact target facts while computing caller-local
// facts.
iree_status_t loom_low_source_invocation_facts_compute(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_source_selection_list_t* selections,
    iree_arena_allocator_t* arena,
    loom_low_source_invocation_facts_t* out_facts);

// Returns the argument-fact inputs for one source function solve.
loom_value_fact_table_view_t loom_low_source_invocation_facts_for_function(
    const loom_low_source_invocation_facts_t* facts, loom_func_like_t function);

// Returns whether |function| is the target of a retained direct semantic call.
// This describes the physical source boundary that source-to-Low must plan; it
// is independent of the authored inline policy that produced that boundary.
bool loom_low_source_invocation_facts_has_direct_callers(
    const loom_low_source_invocation_facts_t* facts, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TRANSFORMS_PIPELINE_SOURCE_INVOCATION_FACTS_H_
