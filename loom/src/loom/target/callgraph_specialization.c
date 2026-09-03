// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/callgraph_specialization.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/rewrite/callable.h"
#include "loom/target/facts_builder.h"
#include "loom/target/function_contract.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_TARGET_CALLGRAPH_SPECIALIZATION_STATISTICS(V, statistics_type) \
  V(statistics_type, call_edges_planned, "call-edges-planned",              \
    "Number of concrete target-propagating call edges planned.")            \
  V(statistics_type, contexts_derived, "contexts-derived",                  \
    "Number of distinct inherited target contexts derived.")                \
  V(statistics_type, versions_created, "versions-created",                  \
    "Number of retained callable target versions created.")                 \
  V(statistics_type, functions_cloned, "functions-cloned",                  \
    "Number of private callable definitions cloned for target contexts.")   \
  V(statistics_type, calls_retargeted, "calls-retargeted",                  \
    "Number of target-propagating calls retargeted to concrete callable "   \
    "versions.")

LOOM_PASS_STATISTICS_DEFINE(loom_target_callgraph_specialization_statistics,
                            loom_target_callgraph_specialization_statistics_t,
                            LOOM_TARGET_CALLGRAPH_SPECIALIZATION_STATISTICS)

static const loom_pass_info_t
    loom_target_callgraph_specialization_pass_info_storage = {
        .name = IREE_SVL("specialize-target-callgraph"),
        .description =
            IREE_SVL("Propagate invocation target contexts through retained "
                     "semantic call graphs."),
        .kind = LOOM_PASS_MODULE,
        .statistic_layout =
            &loom_target_callgraph_specialization_statistics_layout,
};

static bool loom_target_callgraph_kind_propagates_target(
    loom_call_like_kind_t kind) {
  return kind == LOOM_CALL_LIKE_KIND_SEMANTIC ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INVOKE ||
         kind == LOOM_CALL_LIKE_KIND_LOW_INTERNAL;
}

const loom_pass_info_t* loom_target_callgraph_specialization_pass_info(void) {
  return &loom_target_callgraph_specialization_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Transient plan
//===----------------------------------------------------------------------===//

typedef uint32_t loom_target_callgraph_row_id_t;
#define LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID \
  ((loom_target_callgraph_row_id_t)UINT32_MAX)

typedef struct loom_target_callgraph_context_t loom_target_callgraph_context_t;

// One constructed target context in the transient specialization plan.
struct loom_target_callgraph_context_t {
  // Stable compiler-owned target resolution represented by this context.
  loom_resolved_target_t resolved_target;

  // Compilation-local identity retained by every version in this context.
  loom_target_context_ordinal_t target_context_ordinal;

  // Requirement applied to construct this context, or NULL when unconstrained.
  const loom_target_facts_t* applied_requirement;

  // Parent construction context, or NULL for an explicit root version.
  loom_target_callgraph_context_t* parent;

  // First context derived directly from this context.
  loom_target_callgraph_context_t* first_child;

  // Next context with the same parent.
  loom_target_callgraph_context_t* next_sibling;

  // Next explicit root context.
  loom_target_callgraph_context_t* next_root;
};

typedef struct loom_target_callgraph_symbol_t {
  // Live function-like definition for this symbol snapshot entry.
  loom_func_like_t function;

  // Function facts projected once before the module mutates.
  const loom_func_symbol_facts_t* function_facts;

  // Stable authored target requirement, or NULL when unconstrained. A
  // target.decl witness names a context without adding a fact requirement.
  const loom_target_facts_t* authored_target_requirement;

  // Borrowed authored target witness name, or empty when targetless.
  iree_string_view_t authored_target_name;

  // First concrete plan row whose source is this symbol.
  loom_target_callgraph_row_id_t first_row_id;

  // Row that keeps the original function definition.
  loom_target_callgraph_row_id_t original_row_id;

  // Next suffix ordinal considered for a cloned definition.
  iree_host_size_t next_clone_ordinal;

  // Last concrete definition inserted after the original function.
  loom_op_t* insertion_anchor;

  // True after function and target facts have been projected.
  bool initialized;

  // True when the callable may be cloned without changing an external
  // artifact contract.
  bool module_internal;
} loom_target_callgraph_symbol_t;

typedef struct loom_target_callgraph_row_t {
  // Original snapshot symbol whose callable body this row versions.
  loom_symbol_id_t source_symbol_id;

  // Inherited target context used by this concrete callable version.
  loom_target_callgraph_context_t* context;

  // Next row with the same source symbol.
  loom_target_callgraph_row_id_t next_symbol_row_id;

  // Existing stable version that seeded this row, or NULL for a new row.
  loom_target_function_version_t* existing_version;

  // Compiler-owned version prepared for a new row, or NULL for a seed row.
  loom_target_function_version_t* pending_version;

  // Concrete function implementing this row after materialization.
  loom_func_like_t concrete_function;

  // Concrete symbol implementing this row after materialization.
  loom_symbol_ref_t concrete_ref;

  // Planned concrete symbol name.
  iree_string_view_t concrete_name;

  // True when materialization must clone the source callable.
  bool clone_required;

  // True when this row is an authored artifact root rather than an inherited
  // module-internal callable.
  bool artifact_root;
} loom_target_callgraph_row_t;

typedef struct loom_target_callgraph_state_t {
  // Pass invocation owning diagnostics, scratch, and statistics.
  loom_pass_t* pass;

  // Mutable module being specialized.
  loom_module_t* module;

  // Mutable owner receiving stable callable versions.
  loom_function_version_owner_t* version_owner;

  // Target-family providers linked into this compile session.
  const loom_target_environment_t* target_environment;

  // Typed pass statistics storage.
  loom_target_callgraph_specialization_statistics_t* statistics;

  // Post-authoring symbol reference snapshot.
  loom_symbol_reference_table_t references;

  // Lazy symbol-fact cache valid until materialization begins.
  loom_symbol_fact_table_t fact_table;

  // Per-snapshot-symbol specialization state.
  loom_target_callgraph_symbol_t* symbols;

  // Stable canonical target requirements indexed by target symbol ID.
  const loom_target_facts_t** requirements_by_target_symbol;

  // Number of symbols in the immutable planning snapshot.
  iree_host_size_t source_symbol_count;

  // Growable concrete version plan indexed by stable row IDs.
  loom_target_callgraph_row_t* rows;

  // Number of live entries in |rows|.
  iree_host_size_t row_count;

  // Number of allocated entries in |rows|.
  iree_host_size_t row_capacity;

  // Explicit root construction contexts.
  loom_target_callgraph_context_t* root_contexts;

  // Concrete row IDs indexed by the final module symbol ID.
  loom_target_callgraph_row_id_t* concrete_rows_by_symbol;

  // Number of entries allocated in |concrete_rows_by_symbol|.
  iree_host_size_t concrete_symbol_capacity;

  // First context ordinal not carried by an existing function version.
  iree_host_size_t next_target_context_ordinal;

  // True while every reachable edge has a valid materialization plan.
  bool plan_valid;
} loom_target_callgraph_state_t;

static iree_status_t loom_target_callgraph_initialize_state(
    loom_target_callgraph_state_t* state) {
  state->source_symbol_count = state->module->symbols.count;
  state->plan_valid = true;
  loom_symbol_fact_table_initialize(&state->fact_table, state->pass->arena);
  if (state->source_symbol_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->pass->arena, state->source_symbol_count, sizeof(*state->symbols),
      (void**)&state->symbols));
  memset(state->symbols, 0,
         state->source_symbol_count * sizeof(*state->symbols));
  for (iree_host_size_t i = 0; i < state->source_symbol_count; ++i) {
    state->symbols[i].first_row_id = LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
    state->symbols[i].original_row_id = LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->pass->arena, state->source_symbol_count,
                                sizeof(*state->requirements_by_target_symbol),
                                (void**)&state->requirements_by_target_symbol));
  memset(state->requirements_by_target_symbol, 0,
         state->source_symbol_count *
             sizeof(*state->requirements_by_target_symbol));
  return iree_ok_status();
}

static iree_status_t loom_target_callgraph_prepare_symbol(
    loom_target_callgraph_state_t* state, loom_symbol_id_t symbol_id) {
  if (symbol_id >= state->source_symbol_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "semantic callee symbol %u is outside the "
                            "callgraph planning snapshot",
                            (unsigned)symbol_id);
  }
  loom_target_callgraph_symbol_t* info = &state->symbols[symbol_id];
  if (info->initialized) return iree_ok_status();

  loom_symbol_t* symbol = &state->module->symbols.entries[symbol_id];
  info->function = loom_func_like_cast(state->module, symbol->defining_op);
  if (!loom_func_like_isa(info->function)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified semantic callee symbol %u is not function-like",
        (unsigned)symbol_id);
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(
      &state->fact_table, state->module, symbol_id, &base_facts));
  info->function_facts = loom_func_symbol_facts_cast(base_facts);
  if (info->function_facts == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified function-like symbol %u has no function facts",
        (unsigned)symbol_id);
  }

  const loom_symbol_ref_t target_ref = info->function_facts->target_symbol;
  if (loom_symbol_ref_is_valid(target_ref)) {
    if (target_ref.module_id != 0 ||
        target_ref.symbol_id >= state->source_symbol_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "verified function target is outside the module symbol snapshot");
    }
    const loom_symbol_facts_base_t* target_base_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
        &state->fact_table, state->module, target_ref, &target_base_facts));
    const loom_symbol_t* target_symbol =
        &state->module->symbols.entries[target_ref.symbol_id];
    info->authored_target_name =
        state->module->strings.entries[target_symbol->name_id];
    const loom_target_symbol_facts_t* target_facts =
        loom_target_symbol_facts_cast(target_base_facts);
    if (target_facts != NULL) {
      const loom_target_facts_t* stable_requirement =
          state->requirements_by_target_symbol[target_ref.symbol_id];
      if (stable_requirement == NULL) {
        loom_target_facts_t* cloned_requirement = NULL;
        IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
            target_facts->projection, state->version_owner->arena,
            &cloned_requirement));
        stable_requirement = cloned_requirement;
        state->requirements_by_target_symbol[target_ref.symbol_id] =
            stable_requirement;
      }
      info->authored_target_requirement = stable_requirement;
    }
  }

  info->module_internal = loom_func_like_is_module_internal(info->function);
  info->insertion_anchor = info->function.op;
  info->initialized = true;
  return iree_ok_status();
}

static loom_target_callgraph_context_t* loom_target_callgraph_find_root_context(
    const loom_target_callgraph_state_t* state,
    loom_resolved_target_t resolved_target,
    const loom_target_facts_t* requirement) {
  for (loom_target_callgraph_context_t* context = state->root_contexts;
       context != NULL; context = context->next_root) {
    if (context->resolved_target.provider == resolved_target.provider &&
        context->resolved_target.facts == resolved_target.facts &&
        context->applied_requirement == requirement) {
      return context;
    }
  }
  return NULL;
}

static iree_status_t loom_target_callgraph_get_root_context(
    loom_target_callgraph_state_t* state,
    loom_resolved_target_t resolved_target,
    loom_target_context_ordinal_t target_context_ordinal,
    const loom_target_facts_t* requirement,
    loom_target_callgraph_context_t** out_context) {
  *out_context = loom_target_callgraph_find_root_context(state, resolved_target,
                                                         requirement);
  if (*out_context != NULL) return iree_ok_status();

  loom_target_callgraph_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(state->pass->arena, sizeof(*context),
                                           (void**)&context));
  *context = (loom_target_callgraph_context_t){
      .resolved_target = resolved_target,
      .target_context_ordinal = target_context_ordinal,
      .applied_requirement = requirement,
      .next_root = state->root_contexts,
  };
  state->root_contexts = context;
  *out_context = context;
  return iree_ok_status();
}

static bool loom_target_callgraph_context_contains_requirement(
    const loom_target_callgraph_context_t* context,
    const loom_target_facts_t* requirement) {
  for (const loom_target_callgraph_context_t* current = context;
       current != NULL; current = current->parent) {
    if (current->applied_requirement == requirement) return true;
  }
  return false;
}

static loom_target_callgraph_context_t*
loom_target_callgraph_find_derived_context(
    loom_target_callgraph_context_t* parent,
    const loom_target_facts_t* requirement) {
  if (requirement == NULL ||
      loom_target_callgraph_context_contains_requirement(parent, requirement)) {
    return parent;
  }
  for (loom_target_callgraph_context_t* child = parent->first_child;
       child != NULL; child = child->next_sibling) {
    if (child->applied_requirement == requirement) return child;
  }
  return NULL;
}

static const loom_target_callgraph_context_t*
loom_target_callgraph_find_existing_equivalent_context(
    const loom_target_callgraph_state_t* state,
    loom_symbol_id_t source_symbol_id,
    const loom_target_facts_t* candidate_facts) {
  loom_target_callgraph_row_id_t row_id =
      state->symbols[source_symbol_id].first_row_id;
  while (row_id != LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID) {
    const loom_target_callgraph_row_t* row = &state->rows[row_id];
    if (row->existing_version != NULL &&
        loom_target_facts_are_equivalent(row->context->resolved_target.facts,
                                         candidate_facts)) {
      return row->context;
    }
    row_id = row->next_symbol_row_id;
  }
  return NULL;
}

static iree_status_t loom_target_callgraph_emit_target_conflict(
    loom_target_callgraph_state_t* state, const loom_op_t* call_op,
    const loom_target_callgraph_symbol_t* callee,
    const loom_target_facts_t* inherited_facts) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee->function_facts->name),
      loom_param_string(callee->authored_target_name),
      loom_param_string(loom_target_facts_identity_name(inherited_facts)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = call_op,
      .error = LOOM_ERR_TARGET_052,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  state->plan_valid = false;
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_target_callgraph_derive_context(
    loom_target_callgraph_state_t* state,
    loom_target_callgraph_context_t* parent, loom_symbol_id_t callee_symbol_id,
    const loom_target_callgraph_symbol_t* callee, const loom_op_t* call_op,
    loom_target_callgraph_context_t** out_context) {
  *out_context = loom_target_callgraph_find_derived_context(
      parent, callee->authored_target_requirement);
  if (*out_context != NULL) return iree_ok_status();

  if (!loom_target_facts_satisfy_specialization_requirement(
          parent->resolved_target.facts, callee->authored_target_requirement)) {
    return loom_target_callgraph_emit_target_conflict(
        state, call_op, callee, parent->resolved_target.facts);
  }

  loom_target_facts_t* candidate_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
      parent->resolved_target.facts, state->pass->arena, &candidate_facts));
  loom_target_facts_builder_apply_requirement(
      callee->authored_target_requirement, candidate_facts);

  const loom_target_callgraph_context_t* existing_context =
      loom_target_callgraph_find_existing_equivalent_context(
          state, callee_symbol_id, candidate_facts);
  const loom_target_facts_t* derived_facts = NULL;
  loom_target_context_ordinal_t target_context_ordinal =
      LOOM_TARGET_CONTEXT_ORDINAL_INVALID;
  if (existing_context != NULL) {
    derived_facts = existing_context->resolved_target.facts;
    target_context_ordinal = existing_context->target_context_ordinal;
  } else {
    loom_target_facts_t* stable_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_target_facts_builder_clone(
        candidate_facts, state->version_owner->arena, &stable_facts));
    derived_facts = stable_facts;
    if (state->next_target_context_ordinal >=
        LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "target callgraph exceeds %u invocation contexts",
                              (unsigned)LOOM_TARGET_CONTEXT_ORDINAL_INVALID);
    }
    target_context_ordinal =
        (loom_target_context_ordinal_t)state->next_target_context_ordinal++;
  }

  loom_target_callgraph_context_t* context = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(state->pass->arena, sizeof(*context),
                                           (void**)&context));
  *context = (loom_target_callgraph_context_t){
      .resolved_target =
          {
              .provider = parent->resolved_target.provider,
              .facts = derived_facts,
          },
      .target_context_ordinal = target_context_ordinal,
      .applied_requirement = callee->authored_target_requirement,
      .parent = parent,
      .next_sibling = parent->first_child,
  };
  parent->first_child = context;
  ++state->statistics->contexts_derived;
  *out_context = context;
  return iree_ok_status();
}

static loom_target_callgraph_row_id_t loom_target_callgraph_find_row(
    const loom_target_callgraph_state_t* state, loom_symbol_id_t source_symbol,
    const loom_target_callgraph_context_t* context) {
  loom_target_callgraph_row_id_t row_id =
      state->symbols[source_symbol].first_row_id;
  while (row_id != LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID) {
    const loom_target_callgraph_row_t* row = &state->rows[row_id];
    if (row->context->resolved_target.provider ==
            context->resolved_target.provider &&
        row->context->resolved_target.facts == context->resolved_target.facts) {
      return row_id;
    }
    row_id = row->next_symbol_row_id;
  }
  return LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
}

static iree_status_t loom_target_callgraph_emit_external_conflict(
    loom_target_callgraph_state_t* state, const loom_op_t* call_op,
    const loom_target_callgraph_symbol_t* callee) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee->function_facts->name),
  };
  const loom_diagnostic_emission_t emission = {
      .op = call_op,
      .error = LOOM_ERR_TARGET_069,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  state->plan_valid = false;
  return iree_diagnostic_emit(state->pass->diagnostic_emitter, &emission);
}

static iree_status_t loom_target_callgraph_append_row(
    loom_target_callgraph_state_t* state, loom_symbol_id_t source_symbol_id,
    loom_target_callgraph_context_t* context,
    loom_target_function_version_t* existing_version, bool artifact_root,
    const loom_op_t* demanding_call,
    loom_target_callgraph_row_id_t* out_row_id) {
  *out_row_id = LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
  loom_target_callgraph_symbol_t* info = &state->symbols[source_symbol_id];
  if (info->first_row_id != LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID &&
      !info->module_internal) {
    return loom_target_callgraph_emit_external_conflict(
        state, demanding_call != NULL ? demanding_call : info->function.op,
        info);
  }
  if (state->row_count >= UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target callgraph exceeds %u concrete versions",
                            (unsigned)(UINT32_MAX - 1));
  }
  if (state->row_count >= state->row_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->pass->arena, state->row_count, state->row_count + 1,
        sizeof(*state->rows), &state->row_capacity, (void**)&state->rows));
  }

  const loom_target_callgraph_row_id_t row_id =
      (loom_target_callgraph_row_id_t)state->row_count++;
  state->rows[row_id] = (loom_target_callgraph_row_t){
      .source_symbol_id = source_symbol_id,
      .context = context,
      .next_symbol_row_id = info->first_row_id,
      .existing_version = existing_version,
      .concrete_ref = loom_symbol_ref_null(),
      .artifact_root = artifact_root,
  };
  info->first_row_id = row_id;
  if (info->original_row_id == LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID) {
    info->original_row_id = row_id;
  }
  *out_row_id = row_id;
  return iree_ok_status();
}

static iree_status_t loom_target_callgraph_seed_versions(
    loom_target_callgraph_state_t* state) {
  loom_target_function_version_snapshot_t snapshot = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      state->module, loom_function_version_owner_list(state->version_owner),
      state->pass->arena, &snapshot));

  for (iree_host_size_t i = 0; i < snapshot.symbol_count; ++i) {
    const loom_symbol_id_t symbol_id = (loom_symbol_id_t)i;
    loom_target_function_version_t* version = loom_target_function_version_cast(
        loom_target_function_version_snapshot_handle_at(&snapshot, symbol_id));
    if (version == NULL) continue;
    const iree_host_size_t next_target_context_ordinal =
        (iree_host_size_t)version->target_context_ordinal + 1;
    if (next_target_context_ordinal > state->next_target_context_ordinal) {
      state->next_target_context_ordinal = next_target_context_ordinal;
    }
    IREE_RETURN_IF_ERROR(
        loom_target_callgraph_prepare_symbol(state, symbol_id));
    loom_target_callgraph_context_t* context = NULL;
    IREE_RETURN_IF_ERROR(loom_target_callgraph_get_root_context(
        state, version->resolved_target, version->target_context_ordinal,
        state->symbols[symbol_id].authored_target_requirement, &context));
    loom_target_callgraph_row_id_t row_id =
        LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_target_callgraph_append_row(
        state, symbol_id, context, version, /*artifact_root=*/true,
        /*demanding_call=*/NULL, &row_id));
    if (!state->plan_valid) return iree_ok_status();
  }
  return iree_ok_status();
}

static bool loom_target_callgraph_has_propagating_incoming_call(
    const loom_target_callgraph_state_t* state,
    loom_symbol_id_t target_symbol_id) {
  loom_symbol_reference_occurrence_id_t occurrence_id =
      state->references.symbols[target_symbol_id].first_incoming_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &state->references.occurrences[occurrence_id];
    occurrence_id = occurrence->next_incoming_occurrence_id;
    if (occurrence->source_symbol_id == target_symbol_id ||
        !loom_symbol_reference_occurrence_is_dependency(occurrence) ||
        occurrence->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
        occurrence->user_op == NULL) {
      continue;
    }
    const loom_call_like_t call =
        loom_call_like_cast(state->module, (loom_op_t*)occurrence->user_op);
    if (loom_call_like_isa(call) &&
        loom_target_callgraph_kind_propagates_target(
            loom_call_like_kind(call))) {
      return true;
    }
  }
  return false;
}

// Seeds authored concrete callgraph roots when the embedding did not supply a
// profile specialization. Externally visible functions are always roots;
// module-internal functions reached by a propagating call instead inherit and
// refine the caller's exact context through the ordinary planning path below.
// This keeps authored-only and profile-driven compilation on the same
// compiler-version representation.
static iree_status_t loom_target_callgraph_seed_authored_roots(
    loom_target_callgraph_state_t* state) {
  if (state->target_environment == NULL) return iree_ok_status();

  for (loom_symbol_id_t symbol_id = 0; symbol_id < state->source_symbol_count;
       ++symbol_id) {
    loom_target_callgraph_symbol_t* info = &state->symbols[symbol_id];
    if (info->first_row_id != LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID) continue;
    const loom_symbol_t* symbol = &state->module->symbols.entries[symbol_id];
    if (!loom_func_like_isa(
            loom_func_like_cast(state->module, symbol->defining_op))) {
      continue;
    }

    IREE_RETURN_IF_ERROR(
        loom_target_callgraph_prepare_symbol(state, symbol_id));
    if (!info->function_facts->has_body ||
        info->authored_target_requirement == NULL ||
        (info->module_internal &&
         loom_target_callgraph_has_propagating_incoming_call(state,
                                                             symbol_id))) {
      continue;
    }

    const loom_target_provider_t* provider =
        loom_target_environment_lookup_fact_provider(
            state->target_environment,
            info->authored_target_requirement->fact_type);
    // Source-only pipelines may carry authored targets whose providers are not
    // part of the active compilation environment. Such functions remain
    // unspecialized just as they did before authored-root discovery; a later
    // target-bound compilation will seed them once the provider is available.
    if (provider == NULL) continue;

    const loom_resolved_target_t resolved_target = {
        .provider = provider,
        .facts = info->authored_target_requirement,
    };
    loom_target_callgraph_context_t* context =
        loom_target_callgraph_find_root_context(
            state, resolved_target, info->authored_target_requirement);
    if (context == NULL) {
      if (state->next_target_context_ordinal >=
          LOOM_TARGET_CONTEXT_ORDINAL_INVALID) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "target callgraph exceeds %u invocation contexts",
            (unsigned)LOOM_TARGET_CONTEXT_ORDINAL_INVALID);
      }
      const loom_target_context_ordinal_t target_context_ordinal =
          (loom_target_context_ordinal_t)state->next_target_context_ordinal++;
      IREE_RETURN_IF_ERROR(loom_target_callgraph_get_root_context(
          state, resolved_target, target_context_ordinal,
          info->authored_target_requirement, &context));
    }

    loom_target_callgraph_row_id_t row_id =
        LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_target_callgraph_append_row(
        state, symbol_id, context, /*existing_version=*/NULL,
        /*artifact_root=*/true, /*demanding_call=*/NULL, &row_id));
    if (!state->plan_valid) return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t loom_target_callgraph_plan_reachable_rows(
    loom_target_callgraph_state_t* state) {
  for (loom_target_callgraph_row_id_t row_id = 0;
       state->plan_valid && row_id < state->row_count; ++row_id) {
    const loom_symbol_id_t caller_source_symbol_id =
        state->rows[row_id].source_symbol_id;
    loom_target_callgraph_context_t* caller_context =
        state->rows[row_id].context;
    loom_symbol_reference_occurrence_id_t edge_id =
        state->references.symbols[caller_source_symbol_id]
            .first_outgoing_occurrence_id;
    while (state->plan_valid &&
           edge_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* edge =
          &state->references.occurrences[edge_id];
      edge_id = edge->next_outgoing_occurrence_id;
      if (!loom_symbol_reference_occurrence_is_dependency(edge) ||
          edge->kind != LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL ||
          edge->user_op == NULL) {
        continue;
      }
      loom_call_like_t call =
          loom_call_like_cast(state->module, (loom_op_t*)edge->user_op);
      if (!loom_call_like_isa(call) ||
          !loom_target_callgraph_kind_propagates_target(
              loom_call_like_kind(call))) {
        continue;
      }
      ++state->statistics->call_edges_planned;

      const loom_symbol_id_t callee_symbol_id = edge->target_symbol_id;
      IREE_RETURN_IF_ERROR(
          loom_target_callgraph_prepare_symbol(state, callee_symbol_id));
      loom_target_callgraph_context_t* callee_context = NULL;
      IREE_RETURN_IF_ERROR(loom_target_callgraph_derive_context(
          state, caller_context, callee_symbol_id,
          &state->symbols[callee_symbol_id], edge->user_op, &callee_context));
      if (!state->plan_valid) continue;
      if (loom_target_callgraph_find_row(state, callee_symbol_id,
                                         callee_context) !=
          LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID) {
        continue;
      }

      loom_target_callgraph_row_id_t callee_row_id =
          LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_target_callgraph_append_row(
          state, callee_symbol_id, callee_context, /*existing_version=*/NULL,
          /*artifact_root=*/false, edge->user_op, &callee_row_id));
    }
  }
  return iree_ok_status();
}

static bool loom_target_callgraph_name_is_planned(
    const loom_target_callgraph_state_t* state, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < state->row_count; ++i) {
    const loom_target_callgraph_row_t* row = &state->rows[i];
    if (row->clone_required &&
        iree_string_view_equal(row->concrete_name, name)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_target_callgraph_plan_clone_name(
    loom_target_callgraph_state_t* state,
    loom_target_callgraph_symbol_t* symbol_info, iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  const loom_symbol_ref_t source_ref =
      loom_func_like_callee(symbol_info->function);
  const loom_symbol_t* source_symbol =
      &state->module->symbols.entries[source_ref.symbol_id];
  const iree_string_view_t source_name =
      state->module->strings.entries[source_symbol->name_id];

  for (iree_host_size_t ordinal = symbol_info->next_clone_ordinal;
       ordinal < IREE_HOST_SIZE_MAX; ++ordinal) {
    char suffix[32] = {0};
    const int suffix_length =
        snprintf(suffix, sizeof(suffix), "_spec%" PRIhsz, ordinal);
    if (suffix_length < 0 ||
        (iree_host_size_t)suffix_length >= sizeof(suffix)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "target specialization suffix overflow");
    }
    iree_host_size_t name_length = 0;
    if (!iree_host_size_checked_add(
            source_name.size, (iree_host_size_t)suffix_length, &name_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "target specialization name overflow");
    }
    char* name_storage = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(state->pass->arena, name_length,
                                             (void**)&name_storage));
    memcpy(name_storage, source_name.data, source_name.size);
    memcpy(name_storage + source_name.size, suffix,
           (iree_host_size_t)suffix_length);
    const iree_string_view_t name =
        iree_make_string_view(name_storage, name_length);
    const loom_string_id_t existing_name_id =
        loom_module_lookup_string(state->module, name);
    if ((existing_name_id != LOOM_STRING_ID_INVALID &&
         loom_module_find_symbol(state->module, existing_name_id) !=
             LOOM_SYMBOL_ID_INVALID) ||
        loom_target_callgraph_name_is_planned(state, name)) {
      continue;
    }
    symbol_info->next_clone_ordinal = ordinal + 1;
    *out_name = name;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "could not plan a unique target specialization "
                          "symbol");
}

static iree_status_t loom_target_callgraph_prepare_materializations(
    loom_target_callgraph_state_t* state) {
  iree_host_size_t clone_count = 0;
  iree_host_size_t new_version_count = 0;
  for (loom_target_callgraph_row_id_t row_id = 0; row_id < state->row_count;
       ++row_id) {
    loom_target_callgraph_row_t* row = &state->rows[row_id];
    loom_target_callgraph_symbol_t* info =
        &state->symbols[row->source_symbol_id];
    if (row_id == info->original_row_id) {
      const loom_string_id_t name_id =
          state->module->symbols.entries[row->source_symbol_id].name_id;
      row->concrete_name = state->module->strings.entries[name_id];
    } else {
      IREE_ASSERT(info->module_internal);
      row->clone_required = true;
      IREE_RETURN_IF_ERROR(loom_target_callgraph_plan_clone_name(
          state, info, &row->concrete_name));
      ++clone_count;
    }
    if (row->existing_version == NULL) ++new_version_count;
  }

  if (!iree_host_size_checked_add(state->source_symbol_count, clone_count,
                                  &state->concrete_symbol_capacity) ||
      state->concrete_symbol_capacity > LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target-specialized symbol count overflow");
  }
  if (state->concrete_symbol_capacity > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->pass->arena, state->concrete_symbol_capacity,
        sizeof(*state->concrete_rows_by_symbol),
        (void**)&state->concrete_rows_by_symbol));
    for (iree_host_size_t i = 0; i < state->concrete_symbol_capacity; ++i) {
      state->concrete_rows_by_symbol[i] = LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID;
    }
  }

  iree_host_size_t final_version_count = 0;
  if (!iree_host_size_checked_add(state->version_owner->list.count,
                                  new_version_count, &final_version_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target function-version count overflow");
  }
  IREE_RETURN_IF_ERROR(loom_function_version_owner_reserve(
      state->version_owner, final_version_count));

  loom_target_function_version_t* new_versions = NULL;
  if (new_version_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->version_owner->arena, new_version_count, sizeof(*new_versions),
        (void**)&new_versions));
  }
  iree_host_size_t new_version_index = 0;
  for (loom_target_callgraph_row_id_t row_id = 0; row_id < state->row_count;
       ++row_id) {
    loom_target_callgraph_row_t* row = &state->rows[row_id];
    if (row->existing_version != NULL) continue;
    loom_target_callgraph_symbol_t* info =
        &state->symbols[row->source_symbol_id];
    loom_func_symbol_facts_t concrete_facts = *info->function_facts;
    concrete_facts.name = row->concrete_name;
    const iree_string_view_t target_name =
        !iree_string_view_is_empty(info->authored_target_name)
            ? info->authored_target_name
            : loom_target_facts_identity_name(
                  row->context->resolved_target.facts);
    bool contract_valid = false;
    const loom_target_facts_t* function_target_facts = NULL;
    if (info->module_internal && !row->artifact_root) {
      IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_internal_facts(
          state->module, &concrete_facts, target_name,
          row->context->resolved_target.facts, state->pass->diagnostic_emitter,
          state->version_owner->arena, &contract_valid,
          &function_target_facts));
    } else {
      IREE_RETURN_IF_ERROR(loom_target_function_contract_refine_facts(
          state->module, &concrete_facts, target_name,
          row->context->resolved_target.facts, state->pass->diagnostic_emitter,
          state->version_owner->arena, &contract_valid,
          &function_target_facts));
    }
    if (!contract_valid) {
      state->plan_valid = false;
      continue;
    }

    loom_target_function_version_t* version =
        &new_versions[new_version_index++];
    *version = (loom_target_function_version_t){
        .base =
            {
                .type = &loom_target_function_version_type,
            },
        .authored_target_name = info->authored_target_name,
        .target_requirement_facts = info->authored_target_requirement,
        .resolved_target = row->context->resolved_target,
        .target_context_ordinal = row->context->target_context_ordinal,
        .authored_target_is_exact = info->authored_target_requirement != NULL &&
                                    loom_target_facts_are_equivalent(
                                        info->authored_target_requirement,
                                        row->context->resolved_target.facts),
        .function_target_facts = function_target_facts,
    };
    row->pending_version = version;
  }
  return iree_ok_status();
}

static void loom_target_callgraph_bind_concrete_row(
    loom_target_callgraph_state_t* state, loom_target_callgraph_row_id_t row_id,
    loom_func_like_t function, loom_symbol_ref_t concrete_ref) {
  loom_target_callgraph_row_t* row = &state->rows[row_id];
  row->concrete_function = function;
  row->concrete_ref = concrete_ref;
  IREE_ASSERT_LT(concrete_ref.symbol_id, state->concrete_symbol_capacity);
  IREE_ASSERT_EQ(state->concrete_rows_by_symbol[concrete_ref.symbol_id],
                 LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID);
  state->concrete_rows_by_symbol[concrete_ref.symbol_id] = row_id;
  if (row->pending_version != NULL) {
    row->pending_version->base.function = function;
  }
}

static iree_status_t loom_target_callgraph_materialize_clones(
    loom_target_callgraph_state_t* state) {
  for (iree_host_size_t i = 0; i < state->source_symbol_count; ++i) {
    const loom_symbol_id_t symbol_id = (loom_symbol_id_t)i;
    loom_target_callgraph_symbol_t* info = &state->symbols[symbol_id];
    if (info->original_row_id == LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID) {
      continue;
    }
    loom_target_callgraph_bind_concrete_row(
        state, info->original_row_id, info->function,
        loom_func_like_callee(info->function));
  }

  for (loom_target_callgraph_row_id_t row_id = 0; row_id < state->row_count;
       ++row_id) {
    loom_target_callgraph_row_t* row = &state->rows[row_id];
    if (!row->clone_required) continue;
    loom_target_callgraph_symbol_t* info =
        &state->symbols[row->source_symbol_id];
    loom_string_id_t concrete_name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        state->module, row->concrete_name, &concrete_name_id));
    loom_symbol_id_t clone_symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(state->module, concrete_name_id,
                                                &clone_symbol_id));
    const loom_symbol_ref_t clone_ref = {
        .module_id = 0,
        .symbol_id = clone_symbol_id,
    };

    loom_builder_t builder;
    loom_builder_initialize(state->module, &state->module->arena,
                            info->function.op->parent_block, &builder);
    loom_builder_set_after(&builder, info->insertion_anchor);
    loom_func_like_t cloned = {0};
    IREE_RETURN_IF_ERROR(loom_callable_clone_definition(
        &builder, info->function, clone_ref, &cloned, state->pass->arena));
    info->insertion_anchor = cloned.op;
    loom_target_callgraph_bind_concrete_row(state, row_id, cloned, clone_ref);
    ++state->statistics->functions_cloned;
    loom_pass_mark_changed(state->pass);
  }
  return iree_ok_status();
}

typedef struct loom_target_callgraph_retarget_walk_t {
  // Shared specialization state.
  loom_target_callgraph_state_t* state;

  // Concrete caller row whose body is being rewritten.
  loom_target_callgraph_row_id_t caller_row_id;
} loom_target_callgraph_retarget_walk_t;

static iree_status_t loom_target_callgraph_retarget_call(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_target_callgraph_retarget_walk_t* walk =
      (loom_target_callgraph_retarget_walk_t*)user_data;
  loom_target_callgraph_state_t* state = walk->state;

  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  if (loom_op_defining_symbol_id(state->module, op, vtable) !=
      LOOM_SYMBOL_ID_INVALID) {
    *out_result = LOOM_WALK_SKIP;
    return iree_ok_status();
  }
  const loom_call_like_t call = {
      .op = vtable && vtable->call_like ? op : NULL,
      .vtable = vtable ? vtable->call_like : NULL,
  };
  if (!loom_call_like_isa(call) ||
      !loom_target_callgraph_kind_propagates_target(
          loom_call_like_kind(call))) {
    return iree_ok_status();
  }

  const loom_symbol_ref_t current_callee = loom_call_like_callee(call);
  IREE_ASSERT(loom_symbol_ref_is_valid(current_callee));
  IREE_ASSERT_EQ(current_callee.module_id, 0);
  IREE_ASSERT_LT(current_callee.symbol_id, state->concrete_symbol_capacity);
  const loom_target_callgraph_row_id_t concrete_callee_row_id =
      state->concrete_rows_by_symbol[current_callee.symbol_id];
  IREE_ASSERT_NE(concrete_callee_row_id, LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID);
  const loom_target_callgraph_row_t* concrete_callee_row =
      &state->rows[concrete_callee_row_id];
  const loom_target_callgraph_symbol_t* callee_info =
      &state->symbols[concrete_callee_row->source_symbol_id];

  const loom_target_callgraph_row_t* caller_row =
      &state->rows[walk->caller_row_id];
  loom_target_callgraph_context_t* callee_context =
      loom_target_callgraph_find_derived_context(
          caller_row->context, callee_info->authored_target_requirement);
  IREE_ASSERT(callee_context != NULL);
  const loom_target_callgraph_row_id_t callee_row_id =
      loom_target_callgraph_find_row(
          state, concrete_callee_row->source_symbol_id, callee_context);
  IREE_ASSERT_NE(callee_row_id, LOOM_TARGET_CALLGRAPH_ROW_ID_INVALID);
  const loom_symbol_ref_t target_callee =
      state->rows[callee_row_id].concrete_ref;

  if (current_callee.module_id != target_callee.module_id ||
      current_callee.symbol_id != target_callee.symbol_id) {
    loom_call_like_set_callee(state->module, call, target_callee);
    ++state->statistics->calls_retargeted;
    loom_pass_mark_changed(state->pass);
  }
  return iree_ok_status();
}

static iree_status_t loom_target_callgraph_retarget_calls(
    loom_target_callgraph_state_t* state) {
  iree_arena_allocator_t walk_arena;
  iree_arena_initialize(state->pass->arena->block_pool, &walk_arena);
  iree_status_t status = iree_ok_status();
  for (loom_target_callgraph_row_id_t row_id = 0;
       iree_status_is_ok(status) && row_id < state->row_count; ++row_id) {
    loom_target_callgraph_retarget_walk_t walk = {
        .state = state,
        .caller_row_id = row_id,
    };
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    iree_arena_reset(&walk_arena);
    status = loom_walk_function(
        state->module, state->rows[row_id].concrete_function,
        LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_target_callgraph_retarget_call, &walk},
        &walk_arena, &walk_result);
  }
  iree_arena_deinitialize(&walk_arena);
  return status;
}

static iree_status_t loom_target_callgraph_publish_versions(
    loom_target_callgraph_state_t* state) {
  for (loom_target_callgraph_row_id_t row_id = 0; row_id < state->row_count;
       ++row_id) {
    loom_target_function_version_t* version =
        state->rows[row_id].pending_version;
    if (version == NULL) continue;
    IREE_RETURN_IF_ERROR(loom_function_version_owner_append(
        state->version_owner, &version->base));
    ++state->statistics->versions_created;
    loom_pass_mark_changed(state->pass);
  }
  return iree_ok_status();
}

iree_status_t loom_target_callgraph_specialization_run(loom_pass_t* pass,
                                                       loom_module_t* module) {
  const loom_target_pass_capability_t* capability =
      loom_target_pass_capability_from_pass(pass);
  loom_function_version_owner_t* version_owner =
      loom_target_pass_capability_function_version_owner(capability);
  if (version_owner == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "specialize-target-callgraph requires a mutable function-version "
        "owner");
  }

  loom_target_callgraph_state_t state = {
      .pass = pass,
      .module = module,
      .version_owner = version_owner,
      .target_environment =
          loom_target_pass_capability_target_environment(capability),
      .statistics = loom_target_callgraph_specialization_statistics(pass),
  };
  IREE_RETURN_IF_ERROR(loom_target_callgraph_initialize_state(&state));
  if (state.source_symbol_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_symbol_reference_table_build(module, pass->arena,
                                                         &state.references));
  IREE_RETURN_IF_ERROR(loom_target_callgraph_seed_versions(&state));
  IREE_RETURN_IF_ERROR(loom_target_callgraph_seed_authored_roots(&state));
  if (state.row_count == 0) return iree_ok_status();
  if (!state.plan_valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_target_callgraph_plan_reachable_rows(&state));
  if (!state.plan_valid) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_target_callgraph_prepare_materializations(&state));
  if (!state.plan_valid) return iree_ok_status();

  IREE_RETURN_IF_ERROR(loom_target_callgraph_materialize_clones(&state));
  IREE_RETURN_IF_ERROR(loom_target_callgraph_retarget_calls(&state));
  return loom_target_callgraph_publish_versions(&state);
}
