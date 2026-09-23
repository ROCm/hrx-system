// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Composed physical representation projections across callable and CFG
// boundaries.
//
// Planning consumes retained function-local analyses while their scratch
// leases are active. Rules select one final physical component schema for each
// logical slot. The retained plan contains only exact rule recipes and stable
// IR identities needed by the atomic application phase.

#ifndef LOOM_TRANSFORMS_BOUNDARY_PROJECTION_PLAN_H_
#define LOOM_TRANSFORMS_BOUNDARY_PROJECTION_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/types.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/transforms/boundary/projection_rule.h"
#include "loom/util/fact_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_boundary_projection_slot_t {
  // Semantic value replaced by this physical representation.
  loom_value_id_t value_id;
  // Original logical type reconstructed at the definition boundary.
  loom_type_t logical_type;
  // Final component schema selected for this logical definition.
  loom_boundary_projection_schema_t schema;
  // Block owning a block-argument definition, otherwise NULL.
  loom_block_t* block;
  // Original first operation anchoring reconstruction in the owning block.
  loom_op_t* reconstruction_anchor;
  // Call owning a result definition, otherwise NULL.
  loom_op_t* call_op;
  // Physical component identities in schema order.
  loom_value_id_t* component_value_ids;
  // Reconstructed semantic value, populated while applying the plan.
  loom_value_id_t replacement_value_id;
  // Planned incoming source per retained CFG predecessor edge.
  loom_boundary_projection_source_t* incoming_sources;
  // Number of entries in incoming_sources.
  iree_host_size_t incoming_source_count;
  // Physical definition kind.
  loom_boundary_projection_slot_role_t role;
  // First slot whose availability depends on this slot.
  iree_host_size_t first_dependent;
  // Whether the complete projection remains available.
  bool selected;
};

typedef struct loom_boundary_projection_dependency_t {
  // Slot requiring the source slot to remain selected.
  iree_host_size_t target;
  // Next dependent of the same source slot.
  iree_host_size_t next;
  // Whether reconstruction of target must follow reconstruction of source.
  bool orders_reconstruction;
} loom_boundary_projection_dependency_t;

typedef struct loom_boundary_projection_call_t {
  // Original semantic direct call rebuilt once through CallLike.
  loom_call_like_t call;
  // Callee function plan indexed in the module plan.
  iree_host_size_t callee_index;
  // Planned projections parallel to original call operands.
  loom_boundary_projection_source_t* operand_sources;
  // Preallocated result identities in the rebuilt operation's physical order.
  loom_value_id_t* result_ids;
} loom_boundary_projection_call_t;

typedef struct loom_boundary_projection_return_t {
  // Original function-body return terminator rebuilt once.
  loom_op_t* op;
  // Planned projections parallel to original return operands.
  loom_boundary_projection_source_t* sources;
} loom_boundary_projection_return_t;

typedef struct loom_boundary_projection_edge_t {
  // Original single-successor terminator rebuilt once.
  loom_op_t* terminator;
  // Planned projections parallel to the original destination arguments.
  loom_boundary_projection_source_t* sources;
} loom_boundary_projection_edge_t;

typedef struct loom_boundary_projection_block_t {
  // Block whose complete signature is rebuilt.
  loom_block_t* block;
  // Original arguments in signature order.
  loom_value_id_t* original_arguments;
  // Number of entries in original_arguments.
  uint16_t original_argument_count;
  // Final physical argument count.
  uint16_t final_argument_count;
  // Incoming single-successor payload edges, empty for a function entry block.
  loom_boundary_projection_edge_t* edges;
  // Number of incoming payload edges.
  iree_host_size_t edge_count;
} loom_boundary_projection_block_t;

struct loom_boundary_projection_function_t {
  // Original function-like operation.
  loom_func_like_t function;
  // Stable compiler version transferred if the function is replaced.
  loom_function_version_t* version;
  // Original logical arguments.
  const loom_value_id_t* arguments;
  // Original argument count.
  uint16_t argument_count;
  // Flat operand offset of declaration arguments, or UINT16_MAX for a body.
  uint16_t argument_operand_offset;
  // Original function result values.
  const loom_value_id_t* results;
  // Original result count.
  uint16_t result_count;
  // Terminator kind returning from the body, or UNKNOWN when bodyless.
  loom_op_kind_t return_kind;
  // Expanded argument count.
  uint16_t final_argument_count;
  // Expanded result count.
  uint16_t final_result_count;
  // Projection schema parallel to logical arguments.
  loom_boundary_projection_schema_t* argument_schemas;
  // Physical start index for each logical argument.
  uint16_t* argument_indices;
  // Projection schema parallel to logical results.
  loom_boundary_projection_schema_t* result_schemas;
  // Physical start index for each logical result.
  uint16_t* result_indices;
  // Function-local facts retained until the atomic rewrite completes.
  loom_value_fact_table_t* facts;
  // Retained CFG snapshot for the function body, or NULL for declarations.
  const loom_value_fact_cfg_region_t* cfg;
  // Function-local correspondence domain.
  loom_local_value_domain_t domain;
  // Rule-owned function state parallel to the invocation rule list.
  void** rule_states;
  // Projection candidates defined inside this function.
  loom_boundary_projection_slot_t* candidates;
  // Number of projection candidates.
  iree_host_size_t candidate_count;
  // Allocated candidate capacity.
  iree_host_size_t candidate_capacity;
  // Availability and reconstruction dependencies between original slots.
  loom_boundary_projection_dependency_t* dependencies;
  // Number of populated dependencies.
  iree_host_size_t dependency_count;
  // Allocated dependency capacity.
  iree_host_size_t dependency_capacity;
  // Selected block-slot indices in reconstruction order.
  iree_host_size_t* reconstruction_order;
  // Number of entries in reconstruction_order.
  iree_host_size_t reconstruction_count;
  // Calls to expanded callees.
  loom_boundary_projection_call_t* calls;
  // Number of calls.
  iree_host_size_t call_count;
  // Allocated call capacity.
  iree_host_size_t call_capacity;
  // Returns matching an expanded result signature.
  loom_boundary_projection_return_t* returns;
  // Number of returns.
  iree_host_size_t return_count;
  // Allocated return capacity.
  iree_host_size_t return_capacity;
  // Rewritten CFG block signatures.
  loom_boundary_projection_block_t* blocks;
  // Number of rewritten blocks.
  iree_host_size_t block_count;
  // Whether this function participates in the rewrite batch.
  bool selected;
  // Whether at least one signature slot is projected.
  bool signature_changes;
  // Whether at least one function result is projected.
  bool result_signature_changes;
};

struct loom_boundary_projection_plan_t {
  // Active pass instance.
  loom_pass_t* pass;
  // Module being rewritten.
  loom_module_t* module;
  // Pass scratch arena owning the plan.
  iree_arena_allocator_t* arena;
  // Shared rewriter for atomic mutation.
  loom_rewriter_t rewriter;
  // Concrete function-version snapshot by symbol ID.
  loom_target_function_version_snapshot_t versions;
  // Compiler-owned semantic rules composed by this invocation.
  loom_boundary_projection_rule_list_t rules;
  // Rule-owned invocation state parallel to rules.
  void** rule_states;
  // Generic application statistics parallel to rules.
  loom_boundary_projection_rule_statistics_t* rule_statistics;
  // Function plans indexed densely.
  loom_boundary_projection_function_t* functions;
  // Number of function plans.
  iree_host_size_t function_count;
  // Function plan index by symbol ID.
  iree_host_size_t* function_indices;
  // Number of entries in function_indices.
  iree_host_size_t function_index_count;
  // Function signatures replaced by the application phase.
  int64_t functions_rewritten;
  // Semantic calls replaced by the application phase.
  int64_t calls_rewritten;
  // Function returns replaced by the application phase.
  int64_t returns_rewritten;
  // Direct CFG edges replaced by the application phase.
  int64_t cfg_edges_rewritten;
};

// Builds a complete, non-mutating boundary projection plan.
iree_status_t loom_boundary_projection_plan_prepare(
    loom_boundary_projection_plan_t* plan,
    const loom_function_version_list_t* version_list,
    loom_boundary_projection_rule_list_t rules);

// Finds a function-local projection candidate by semantic value identity.
iree_host_size_t loom_boundary_projection_slot_index(
    const loom_boundary_projection_function_t* function,
    loom_value_id_t value_id);

// Adds one retained slot dependency. Rejection propagates from |source| to
// |target|; ordered dependencies additionally constrain reconstruction.
iree_status_t loom_boundary_projection_add_dependency(
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function, iree_host_size_t source,
    iree_host_size_t target, bool orders_reconstruction);

// Returns invocation-wide state owned by |rule|, or NULL when unset.
void* loom_boundary_projection_rule_state(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule);

// Sets invocation-wide state owned by |rule|. May be called once.
void loom_boundary_projection_set_rule_state(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule, void* state);

// Returns function-local state owned by |rule|, or NULL when unset.
void* loom_boundary_projection_function_rule_state(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    const loom_boundary_projection_rule_t* rule);

// Sets function-local state owned by |rule|. May be called once per function.
void loom_boundary_projection_set_function_rule_state(
    const loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_rule_t* rule, void* state);

// Records generic application work performed by |rule|.
void loom_boundary_projection_record(
    loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_rule_t* rule, int64_t projections,
    int64_t components);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BOUNDARY_PROJECTION_PLAN_H_
