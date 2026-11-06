// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Callable ownership summaries and their single structural call-graph walk.
//
// State is scoped to one lifetime-analysis invocation. The graph owns
// structural observations and exact carrier aliases; the CFG solver updates
// argument/result summaries monotonically. Cleanup may insert release
// operations without changing these callable edges or aliases.

#ifndef LOOM_ANALYSIS_OWNERSHIP_CALLABLE_H_
#define LOOM_ANALYSIS_OWNERSHIP_CALLABLE_H_

#include "loom/analysis/ownership_lifetime.h"
#include "loom/analysis/scc.h"
#include "loom/ir/local_value_domain.h"

#ifdef __cplusplus
extern "C" {
#endif

// The all-resource analyzer checks descriptor effects. The selected-family
// modes additionally plan final-use cleanup, either requiring prepared IR or
// materializing releases allowed by the policy's builder.
typedef enum loom_ownership_lifetime_mode_e {
  LOOM_OWNERSHIP_LIFETIME_ANALYZE,
  LOOM_OWNERSHIP_LIFETIME_VERIFY_PREPARED,
  LOOM_OWNERSHIP_LIFETIME_MATERIALIZE,
} loom_ownership_lifetime_mode_t;

typedef enum loom_ownership_lifetime_value_state_e {
  LOOM_OWNERSHIP_LIFETIME_VALUE_UNKNOWN = 0,
  LOOM_OWNERSHIP_LIFETIME_VALUE_BORROWED = 1,
  LOOM_OWNERSHIP_LIFETIME_VALUE_OWNED = 2,
} loom_ownership_lifetime_value_state_t;

typedef struct loom_ownership_lifetime_callee_t
    loom_ownership_lifetime_callee_t;
typedef struct loom_ownership_lifetime_alias_t loom_ownership_lifetime_alias_t;

typedef struct loom_ownership_lifetime_function_summary_t {
  // Function-like op this summary describes.
  loom_func_like_t function;
  // Instruction semantics resolved once from this function's representation.
  loom_ownership_effect_provider_t effects;
  // Body region analyzed for this summary, or NULL for conservative callees.
  loom_region_t* body;
  // Declared body exit kind, or UNKNOWN for bodyless callables.
  loom_op_kind_t body_exit_kind;
  // Number of operands in the callable signature.
  uint16_t arg_count;
  // Number of results in the callable signature.
  uint16_t result_count;
  // Per-argument flag set when the body requires the corresponding operand to
  // arrive owned.
  bool* arg_consumed;
  // Per-result state inferred from returned values.
  loom_ownership_lifetime_value_state_t* result_states;
  // First operation with nested execution regions, or NULL for flat control.
  const loom_op_t* structured_op;
  // True when a local operation creates or transfers an owned obligation.
  bool has_owned_effects;
  // True when any argument consumes ownership or any result returns it.
  bool transfers_ownership;
  // Arena-owned callable edges retained by the structural graph walk.
  loom_ownership_lifetime_callee_t* callees;
  // Required nonwriting aliases retained by the structural graph walk.
  loom_ownership_lifetime_alias_t* aliases;
  // Function-local value domain retained across inference iterations. NULL
  // until the first transfer; only the active function acquires its scratch.
  loom_local_value_domain_t* value_domain;
  // Canonical carrier ordinals in |value_domain|, or NULL without aliases.
  // Flattened once before the first transfer and immutable thereafter.
  loom_value_ordinal_t* alias_owners;
  // True when suppressed analysis saw a possible diagnostic on this summary.
  bool needs_diagnostic_rerun;
} loom_ownership_lifetime_function_summary_t;

struct loom_ownership_lifetime_callee_t {
  // Indexed summary of a callee, including declaration-only call contracts.
  const loom_ownership_lifetime_function_summary_t* summary;
  // Next callable edge observed in the same function, or NULL.
  loom_ownership_lifetime_callee_t* next;
};

typedef struct loom_ownership_lifetime_module_state_t {
  // Module being checked.
  loom_module_t* module;
  // Caller-owned analysis options.
  const loom_ownership_lifetime_options_t* options;
  // Domain and cleanup behavior for this invocation.
  loom_ownership_lifetime_mode_t mode;
  // Result object receiving final diagnostic and traversal counters.
  loom_ownership_lifetime_result_t* result;
  // Dense summaries indexed by module symbol ID.
  loom_ownership_lifetime_function_summary_t* summaries;
  // Number of entries in |summaries|.
  iree_host_size_t summary_count;
} loom_ownership_lifetime_module_state_t;

typedef struct loom_ownership_lifetime_graph_node_t {
  // Summary owned by the module state for this bodyful function.
  loom_ownership_lifetime_function_summary_t* summary;
} loom_ownership_lifetime_graph_node_t;

typedef struct loom_ownership_lifetime_graph_t {
  // Module-level analysis state.
  loom_ownership_lifetime_module_state_t* module_state;
  // Dense bodyful function nodes.
  loom_ownership_lifetime_graph_node_t* nodes;
  // Number of entries in |nodes|.
  iree_host_size_t node_count;
  // Symbol ID to graph node index map, or IREE_HOST_SIZE_MAX.
  iree_host_size_t* symbol_to_node;
  // Number of entries in |symbol_to_node|.
  iree_host_size_t symbol_to_node_count;
} loom_ownership_lifetime_graph_t;

// Initializes callable summaries once, indexed by the module symbol table.
iree_status_t loom_ownership_lifetime_initialize_module_summaries(
    loom_ownership_lifetime_module_state_t* module_state);

// Builds the bodyful call graph with one structural walk per function.
// Summaries and graph data live in |arena|.
iree_status_t loom_ownership_lifetime_build_graph(
    loom_ownership_lifetime_module_state_t* module_state,
    iree_arena_allocator_t* arena, loom_ownership_lifetime_graph_t* out_graph,
    loom_scc_list_t* out_sccs);

// Acquires the summary's function-local value domain, preparing exact carrier
// owners on the first transfer. Later transfers restore the retained domain
// without inspecting source IR. The caller releases the domain after use.
iree_status_t loom_ownership_lifetime_acquire_value_domain(
    loom_ownership_lifetime_module_state_t* module_state,
    loom_ownership_lifetime_function_summary_t* summary);

// Returns the nested execution operation requiring CFG lowering when the
// function may own resources, or NULL when its control boundary is supported.
// Uses retained structural facts and monotone callable summaries without IR
// traversal. A structured function with an ownership-neutral ABI and body is
// safe to leave in source form while its resource-using callees are checked.
const loom_op_t* loom_ownership_lifetime_cfg_boundary(
    const loom_ownership_lifetime_function_summary_t* summary);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_OWNERSHIP_CALLABLE_H_
