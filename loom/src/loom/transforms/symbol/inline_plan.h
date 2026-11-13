// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared compiler-owned plan representation for callable selection and
// execution. This is not a public transform API; inline_callables.h owns the
// producer-facing opaque plan contract.

#ifndef LOOM_TRANSFORMS_SYMBOL_INLINE_PLAN_H_
#define LOOM_TRANSFORMS_SYMBOL_INLINE_PLAN_H_

#include "loom/analysis/scc.h"
#include "loom/transforms/symbol/inline_callables.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_INLINE_CALLABLES_STATISTICS(V, statistics_type)     \
  V(statistics_type, required_edges, "required-edges",           \
    "Number of call edges required to inline by policy.")        \
  V(statistics_type, profitable_edges, "profitable-edges",       \
    "Number of optional call edges selected by profitability.")  \
  V(statistics_type, deferred_edges, "deferred-edges",           \
    "Number of required callable-exit edges deferred.")          \
  V(statistics_type, kept_edges, "kept-edges",                   \
    "Number of call edges left unchanged by policy.")            \
  V(statistics_type, calls_cloned, "calls-cloned",               \
    "Number of call sites inlined by cloning the callee body.")  \
  V(statistics_type, calls_transferred, "calls-transferred",     \
    "Number of call sites inlined by moving the callee body.")   \
  V(statistics_type, symbols_transferred, "symbols-transferred", \
    "Number of private callable symbols erased by transfer inline.")

LOOM_PASS_STATISTICS_DECLARE(loom_inline_callables_statistics,
                             loom_inline_callables_statistics_t,
                             LOOM_INLINE_CALLABLES_STATISTICS)

#define LOOM_INLINE_PLAN_ENTRY_INVALID ((uint32_t)UINT32_MAX)

typedef enum loom_inline_plan_action_e {
  LOOM_INLINE_PLAN_ACTION_KEEP = 0,
  LOOM_INLINE_PLAN_ACTION_REQUIRED = 1,
  LOOM_INLINE_PLAN_ACTION_CANDIDATE = 2,
  LOOM_INLINE_PLAN_ACTION_CLONE = 3,
  LOOM_INLINE_PLAN_ACTION_TRANSFER = 4,
  LOOM_INLINE_PLAN_ACTION_ERROR = 5,
} loom_inline_plan_action_t;

typedef enum loom_inline_blocker_e {
  LOOM_INLINE_BLOCKER_NONE = 0,
  LOOM_INLINE_BLOCKER_POLICY_CONFLICT = 1,
  LOOM_INLINE_BLOCKER_REQUIRED_CYCLE = 2,
  LOOM_INLINE_BLOCKER_CALL_NOT_CALL_LIKE = 3,
  LOOM_INLINE_BLOCKER_CALL_NOT_OWNED_BY_SYMBOL = 4,
  LOOM_INLINE_BLOCKER_INVALID_CALLEE_SYMBOL = 5,
  LOOM_INLINE_BLOCKER_CALLEE_NOT_FUNCTION_LIKE = 6,
  LOOM_INLINE_BLOCKER_FUNC_CALL_TARGET_NOT_FUNCTION_LIKE = 7,
  LOOM_INLINE_BLOCKER_UNSUPPORTED_CALL_KIND = 8,
  LOOM_INLINE_BLOCKER_NON_CALL_SHAPE = 9,
  LOOM_INLINE_BLOCKER_CALLEE_MISSING_BODY = 10,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_EMPTY = 11,
  LOOM_INLINE_BLOCKER_RECURSIVE_BODY = 12,
  LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR = 13,
  LOOM_INLINE_BLOCKER_TARGET_REQUIRES_INLINE = 14,
  LOOM_INLINE_BLOCKER_LOW_CALLEE_KIND = 15,
  LOOM_INLINE_BLOCKER_LOW_ALLOCATION = 16,
  LOOM_INLINE_BLOCKER_LOW_ENTRY_RESOURCE = 17,
  LOOM_INLINE_BLOCKER_LOW_LOCKED_NESTED_REGION = 18,
  LOOM_INLINE_BLOCKER_CALLER_REGION_NOT_CFG_CAPABLE = 19,
  LOOM_INLINE_BLOCKER_TARGET_REQUIRES_LEAF,
  LOOM_INLINE_BLOCKER_NATIVE_CALL_CYCLE,
  LOOM_INLINE_BLOCKER_NATIVE_CALLEE_RESOURCES,
  LOOM_INLINE_BLOCKER_NATIVE_CALLEE_STORAGE,
  LOOM_INLINE_BLOCKER_COUNT,
} loom_inline_blocker_t;

typedef struct loom_inline_symbol_info_t {
  // Borrowed symbol table entry for this symbol id.
  const loom_symbol_t* symbol;
  // Function-like view of symbol->defining_op, or empty when not function-like.
  loom_func_like_t function;
  // Number of incoming direct call edges in the dependency snapshot.
  uint32_t incoming_call_count;
  // Number of incoming non-call symbol references in the dependency snapshot.
  uint32_t incoming_non_call_ref_count;
  // Number of incoming call edges selected for required inlining.
  uint32_t planned_call_removals;
  // Whether the function body remains linear after its required call closure
  // is flattened in callee-before-caller order.
  uint8_t body_will_be_linear : 1;
  // Whether this symbol directly owns a func.exit operation.
  uint8_t body_has_callable_exit : 1;
  // Whether the expanded body still depends on its inherited target context.
  uint8_t body_is_contextual : 1;
  // An explicit retained edge requires this body and its expansions to be
  // leaves.
  uint8_t requires_leaf_body : 1;
  // The selected expansion of this body still contains a native call.
  uint8_t body_has_retained_call : 1;
  // The selected Low body contains an explicit entry resource.
  uint8_t body_has_resources : 1;
  // The selected Low body contains an explicit storage reservation.
  uint8_t body_has_storage : 1;
  // A selected reservation contributes nonempty workgroup storage.
  uint8_t body_has_workgroup_storage : 1;
  // Cached Low-body blocker plus one; zero means the body is not inventoried.
  uint8_t body_preflight;
} loom_inline_symbol_info_t;

typedef struct loom_inline_profitability_symbol_t {
  // Expanded body cost, or UINT32_MAX before the symbol is marked reachable.
  uint32_t expanded_body_cost;
  // Head of the optional-call list targeting this symbol.
  uint32_t first_candidate_by_target;
  // Whether the authored body is linear before any call closure is expanded.
  bool direct_body_is_linear;
  // Whether the authored body directly depends on inherited target context.
  bool direct_body_is_contextual;
} loom_inline_profitability_symbol_t;

typedef struct loom_inline_plan_entry_t {
  // Stable ordinal assigned in reference occurrence order.
  uint32_t ordinal;
  // Next call entry with the same source symbol.
  uint32_t next_graph_from_source;
  // Next required-inline entry executed in the same SCC component.
  uint32_t next_required_in_component;
  // Symbol whose definition owns call_op.
  loom_symbol_id_t source_symbol_id;
  // Symbol referenced by the call-like callee attr.
  loom_symbol_id_t target_symbol_id;
  // Direct call-like operation to rewrite.
  loom_op_t* call_op;
  // Call-like interface for call_op.
  loom_call_like_t call;
  // Function-like interface for the target definition.
  loom_func_like_t callee;
  // Inline policy read from the callee symbol definition.
  uint8_t callee_policy;
  // Inline policy read from the call site.
  uint8_t call_policy;
  // Effective edge inline policy after conflict resolution.
  uint8_t effective_policy;
  // Temperature hint read from the callee symbol definition.
  uint8_t callee_temperature;
  // Temperature hint read from the call site.
  uint8_t call_temperature;
  // Effective edge temperature hint after call-site override.
  uint8_t effective_temperature;
  // Planned loom_inline_plan_action_t, packed with the temperature hints.
  uint8_t action;
  // Resolved loom_target_call_policy_t constraints for this boundary.
  uint8_t target_policy;
  // Stable blocker code for ACTION_ERROR.
  loom_inline_blocker_t blocker;
  // Call-site execution context retained by the reference producer.
  loom_symbol_reference_execution_scope_id_t execution_scope;
} loom_inline_plan_entry_t;

typedef struct loom_inline_component_entries_t {
  // First required-inline entry in stable occurrence order.
  uint32_t first_entry;
  // Last required-inline entry used to append in stable occurrence order.
  uint32_t last_entry;
} loom_inline_component_entries_t;

typedef struct loom_inline_execution_targets_t loom_inline_execution_targets_t;

struct loom_inline_callables_plan_t {
  // Active pass invocation.
  loom_pass_t* pass;
  // Borrowed planning inputs and inline policy.
  loom_inline_callables_plan_options_t options;
  // Typed statistics storage for the current pass invocation.
  loom_inline_callables_statistics_t statistics;
  // Module being transformed.
  loom_module_t* module;
  // Mutable compiler-version owner, or NULL outside target compilation.
  loom_function_version_owner_t* version_owner;
  // Lazily allocated target projections for independent execution scopes.
  loom_inline_execution_targets_t* execution_targets;
  // Dense symbol summaries indexed by module symbol id.
  loom_inline_symbol_info_t* symbols;
  // Dense plan entries collected from direct call edges.
  loom_inline_plan_entry_t* entries;
  // Number of valid entries in entries.
  uint32_t entry_count;
  // Number of unresolved optional call entries.
  uint32_t candidate_count;
  // Whether any retained boundary constrains the callee's expanded body.
  bool has_native_constraints;
  // At least one authored retained boundary fixes a leaf body.
  bool has_required_leaf_body;
  // First call entry for each source symbol.
  uint32_t* first_graph_by_source;
  // Lazily allocated optional-inline summaries indexed by symbol id.
  loom_inline_profitability_symbol_t* profitability_symbols;
  // Lazily allocated optional-call links indexed by plan entry ordinal.
  uint32_t* next_candidate_by_target;
  // Required-inline SCCs in callee-before-caller order.
  loom_scc_list_t sccs;
  // SCC ordinal for each symbol id, or IREE_HOST_SIZE_MAX when absent.
  iree_host_size_t* component_by_symbol;
  // Required-inline entries indexed by their source SCC component.
  loom_inline_component_entries_t* component_entries;
  // Function versions whose definitions were erased during execution.
  iree_host_size_t erased_version_count;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_INLINE_PLAN_H_
