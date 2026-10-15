// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/target_legalize.h"

#include <string.h>

#include "loom/analysis/contract.h"
#include "loom/codegen/low/lower/source_query.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/transforms/pipeline/target_legalize_options.h"
#include "loom/codegen/low/transforms/pipeline/target_legalize_preflight.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/intern_table.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/pass/pipeline.h"
#include "loom/pass/registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/greedy.h"
#include "loom/target/legalization.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"
#include "loom/target/math_policy.h"
#include "loom/target/pass_environment.h"
#include "loom/target/reporting/report.h"
#include "loom/transforms/vector/component_packet_legalization.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/walk.h"

typedef struct loom_low_target_legalize_final_rejection_t {
  // Op rejected by a final legalizer.
  const loom_op_t* op;
  // Legalizer rejection result copied for final verification.
  loom_target_legalizer_result_t result;
  // Next final rejection record in function-local insertion order.
  struct loom_low_target_legalize_final_rejection_t* next;
} loom_low_target_legalize_final_rejection_t;

// One original source operation's decision, retained across greedy revisits.
// Operation addresses remain stable through erasure until module destruction.
typedef struct loom_low_target_legalize_report_decision_t {
  // Original operation, retained after erasure with its DEAD flag set.
  const loom_op_t* op;
  // Latest decision, or NONE until a legalizer reports an intervention.
  loom_target_compile_report_legalization_action_t action;
  // Provider strategy for the latest decision.
  loom_target_compile_report_legalizer_strategy_t strategy;
  // Whether this source rewrite lineage expanded vector semantics into lanes.
  bool scalarized;
  // Operations created across this source rewrite lineage.
  uint64_t created_op_count;
  // Operations erased across this source rewrite lineage.
  uint64_t erased_op_count;
  // Whether a generated descendant expanded vector semantics into lanes.
  bool descendant_scalarized;
  // Operations created while rewriting generated descendants.
  uint64_t descendant_created_op_count;
  // Operations erased while rewriting generated descendants.
  uint64_t descendant_erased_op_count;
  // Optional detailed row in the function arena, absent for summary collection.
  loom_target_compile_report_legalization_row_t* row;
  // Next decision in first-observation order, independent of pointer ordering.
  struct loom_low_target_legalize_report_decision_t* next;
} loom_low_target_legalize_report_decision_t;

typedef struct loom_low_target_legalize_report_lineage_entry_t {
  // Generated operation retaining source-report ownership after creation.
  const loom_op_t* op;
  // Authored source decision that owns this generated operation.
  loom_low_target_legalize_report_decision_t* decision;
} loom_low_target_legalize_report_lineage_entry_t;

typedef struct loom_low_target_legalize_report_lineage_key_t {
  // Dense ownership entries compared by the intern table.
  const loom_low_target_legalize_report_lineage_entry_t* entries;
  // Generated operation identity being queried.
  const loom_op_t* op;
} loom_low_target_legalize_report_lineage_key_t;

typedef struct loom_low_target_legalize_packet_plan_value_range_t {
  // First result value created by one packet-materializing legalizer.
  loom_value_id_t begin;
  // One past the final result value created by that legalizer.
  loom_value_id_t end;
} loom_low_target_legalize_packet_plan_value_range_t;

typedef struct loom_low_target_legalize_function_state_t {
  // Pass invocation being executed.
  loom_pass_t* pass;
  // Immutable pass options for this invocation.
  const loom_low_target_legalize_pass_state_t* pass_state;
  // Typed statistics storage for the current pass invocation.
  loom_low_target_legalize_statistics_t* statistics;
  // Module being legalized.
  loom_module_t* module;
  // Selected target-bound source function.
  const loom_low_source_selection_t* selection;
  // Source-to-low query options shared by eager and final legality checks.
  loom_low_lower_options_t lower_options;
  // Low descriptor set selected by the function target facts.
  const loom_low_descriptor_set_t* descriptor_set;
  // Active source-to-low contract query scope, rebuilt after IR mutation.
  loom_low_lower_source_query_scope_t* query_scope;
  // Arena receiving query_scope storage for this function run.
  iree_arena_allocator_t* query_scope_arena;
  // Target legalizer registry shared across functions in this pass run.
  const loom_target_legalizer_registry_t* legalizer_registry;
  // Target-low legality providers available to the final verifier.
  loom_target_low_legality_provider_list_t legality_provider_list;
  // Current legalization context passed to legalizer callbacks.
  loom_target_legalization_context_t legalization_context;
  // Authored contract results retained by the current component packet plan.
  loom_vector_component_packet_query_cache_t packet_query_cache;
  // Packet-materialized value ranges awaiting the next component plan.
  struct {
    // Monotonic half-open ranges created during the current worklist.
    loom_low_target_legalize_packet_plan_value_range_t* entries;
    // Number of initialized ranges.
    iree_host_size_t count;
    // Allocated entry capacity.
    iree_host_size_t capacity;
  } pending_packet_plan_values;
  // Optional compile report receiving cold legalization decision rows.
  loom_target_compile_report_t* compile_report;
  // Original source decisions, indexed once before the function is mutated.
  struct {
    // Entries sorted by stable operation address for binary lookup.
    loom_low_target_legalize_report_decision_t* entries;
    // Number of original legalizer-rooted operations.
    iree_host_size_t count;
    // Allocated entry capacity during source capture.
    iree_host_size_t capacity;
    // First observed decision, or NULL when no intervention was needed.
    loom_low_target_legalize_report_decision_t* first;
    // Last observed decision for constant-time append.
    loom_low_target_legalize_report_decision_t* last;
  } report_decisions;
  // Generated-operation ownership retained without encoding pass state in IR.
  struct {
    // Operation identity index into entries.
    loom_intern_table_t index;
    // Dense generated-operation ownership entries.
    loom_low_target_legalize_report_lineage_entry_t* entries;
    // Allocated entry capacity.
    iree_host_size_t capacity;
    // True after the cold report index has been initialized.
    bool initialized;
  } report_lineage;
  // First final legalizer rejection retained for final legality diagnostics.
  loom_low_target_legalize_final_rejection_t* final_rejections;
  // Assumption diagnostics checked before legalizer dispatch.
  loom_low_target_legalize_preflight_t preflight;
  // True when contract queries should surface retained legalizer rejections.
  bool use_final_rejections;
} loom_low_target_legalize_function_state_t;

static loom_target_compile_report_legalization_mode_t
loom_low_target_legalize_report_mode(loom_target_legalization_mode_t mode) {
  switch (mode) {
    case LOOM_TARGET_LEGALIZATION_MODE_EAGER:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_EAGER;
    case LOOM_TARGET_LEGALIZATION_MODE_FINAL:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_FINAL;
    default:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_MODE_NONE;
  }
}

static loom_target_compile_report_legalization_policy_t
loom_low_target_legalize_report_policy(
    loom_target_legalization_policy_t policy) {
  switch (policy) {
    case LOOM_TARGET_LEGALIZATION_POLICY_PREFER_NATIVE:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_PREFER_NATIVE;
    case LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REFERENCE_ONLY;
    case LOOM_TARGET_LEGALIZATION_POLICY_REQUIRE_NATIVE:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_REQUIRE_NATIVE;
    default:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_POLICY_NONE;
  }
}

static loom_target_compile_report_contract_outcome_t
loom_low_target_legalize_report_contract_outcome(
    loom_target_contract_query_outcome_t outcome) {
  switch (outcome) {
    case LOOM_TARGET_CONTRACT_QUERY_UNHANDLED:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNHANDLED;
    case LOOM_TARGET_CONTRACT_QUERY_LEGAL:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_LEGAL;
    case LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED:
    case LOOM_TARGET_CONTRACT_QUERY_TERMINAL_REJECTION:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_UNSUPPORTED;
    case LOOM_TARGET_CONTRACT_QUERY_INVALID_IR:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_INVALID_IR;
    default:
      return LOOM_TARGET_COMPILE_REPORT_CONTRACT_OUTCOME_NONE;
  }
}

static loom_target_compile_report_legalizer_strategy_t
loom_low_target_legalize_report_legalizer_strategy(
    loom_target_legalizer_strategy_t strategy) {
  switch (strategy) {
    case LOOM_TARGET_LEGALIZER_STRATEGY_TARGET:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET;
    case LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE;
    case LOOM_TARGET_LEGALIZER_STRATEGY_UNKNOWN:
    default:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE;
  }
}

static loom_target_compile_report_legalization_outcome_t
loom_low_target_legalize_report_legalization_outcome(
    loom_target_compile_report_legalization_action_t action,
    loom_target_compile_report_legalizer_strategy_t legalizer_strategy) {
  switch (action) {
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_ALREADY_LEGAL;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN:
      switch (legalizer_strategy) {
        case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_TARGET:
          return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_TARGET_REWRITE;
        case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_REFERENCE:
          return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REFERENCE_FALLBACK;
        case LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE:
        default:
          return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_NONE;
      }
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_DEFERRED;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_INVALID_IR;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_REJECT_UNSUPPORTED;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_UNHANDLED;
    case LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE:
    default:
      return LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_OUTCOME_NONE;
  }
}

static iree_string_view_t loom_low_target_legalize_function_name(
    const loom_low_target_legalize_function_state_t* state) {
  const loom_symbol_ref_t callee =
      loom_func_like_callee(state->selection->func);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= state->module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol =
      &state->module->symbols.entries[callee.symbol_id];
  if (symbol->name_id >= state->module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&state->module->strings, symbol->name_id);
}

static iree_string_view_t loom_low_target_legalize_nonempty(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

static iree_string_view_t loom_low_target_legalize_report_descriptor_key(
    const loom_low_target_legalize_function_state_t* state,
    const loom_target_contract_query_result_t* query_result) {
  const loom_low_descriptor_t* descriptor = query_result->selected_descriptor;
  if (descriptor == NULL ||
      descriptor->key_string_ref == LOOM_STRING_REF_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(state->descriptor_set,
                                        descriptor->key_string_ref);
}

static bool loom_low_target_legalize_report_wants_rows(
    const loom_low_target_legalize_function_state_t* state) {
  return loom_target_compile_report_wants_details(
      state->compile_report,
      LOOM_TARGET_COMPILE_REPORT_DETAIL_TARGET_LEGALIZATION_ROWS);
}

static bool loom_low_target_legalize_report_source_less(
    const loom_low_target_legalize_report_decision_t* lhs,
    const loom_low_target_legalize_report_decision_t* rhs) {
  return (uintptr_t)lhs->op < (uintptr_t)rhs->op;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_target_legalize_sort_report_sources,
                          loom_low_target_legalize_report_decision_t,
                          loom_low_target_legalize_report_source_less)

static loom_low_target_legalize_report_decision_t*
loom_low_target_legalize_report_decision(
    const loom_low_target_legalize_function_state_t* state,
    const loom_op_t* op) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = state->report_decisions.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    loom_low_target_legalize_report_decision_t* decision =
        &state->report_decisions.entries[middle];
    if (decision->op == op) {
      return decision;
    }
    if ((uintptr_t)decision->op < (uintptr_t)op) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return NULL;
}

static uint32_t loom_low_target_legalize_report_lineage_hash(
    const loom_op_t* op) {
  uint64_t value = (uint64_t)(uintptr_t)op >> 4;
  value ^= value >> 16;
  return (uint32_t)(value * UINT64_C(0x9e3779b97f4a7c15));
}

static bool loom_low_target_legalize_report_lineage_equal(const void* context,
                                                          uint32_t index) {
  const loom_low_target_legalize_report_lineage_key_t* key =
      (const loom_low_target_legalize_report_lineage_key_t*)context;
  return key->entries[index].op == key->op;
}

static loom_low_target_legalize_report_decision_t*
loom_low_target_legalize_report_lineage_lookup(
    const loom_low_target_legalize_function_state_t* state,
    const loom_op_t* op) {
  if (!state->report_lineage.initialized) {
    return NULL;
  }
  const loom_low_target_legalize_report_lineage_key_t key = {
      .entries = state->report_lineage.entries,
      .op = op,
  };
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &state->report_lineage.index,
      loom_low_target_legalize_report_lineage_hash(op),
      loom_low_target_legalize_report_lineage_equal, &key);
  return probe.index == UINT32_MAX
             ? NULL
             : state->report_lineage.entries[probe.index].decision;
}

static loom_low_target_legalize_report_decision_t*
loom_low_target_legalize_report_owner(
    const loom_low_target_legalize_function_state_t* state,
    const loom_op_t* op) {
  loom_low_target_legalize_report_decision_t* decision =
      loom_low_target_legalize_report_decision(state, op);
  return decision != NULL
             ? decision
             : loom_low_target_legalize_report_lineage_lookup(state, op);
}

static iree_status_t loom_low_target_legalize_report_lineage_insert(
    loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    loom_low_target_legalize_report_decision_t* decision) {
  if (!state->report_lineage.initialized) {
    return iree_ok_status();
  }
  const uint32_t hash = loom_low_target_legalize_report_lineage_hash(op);
  const loom_low_target_legalize_report_lineage_key_t key = {
      .entries = state->report_lineage.entries,
      .op = op,
  };
  loom_intern_probe_t probe = loom_intern_table_probe(
      &state->report_lineage.index, hash,
      loom_low_target_legalize_report_lineage_equal, &key);
  IREE_ASSERT_EQ(probe.index, UINT32_MAX);
  if (state->report_lineage.index.count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target legalization report lineage overflow");
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      state->query_scope_arena, &state->report_lineage.index, hash,
      /*insertion_count=*/1, &probe.slot));
  if (state->report_lineage.index.count == state->report_lineage.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->query_scope_arena, state->report_lineage.index.count,
        state->report_lineage.index.count + 1,
        sizeof(*state->report_lineage.entries), &state->report_lineage.capacity,
        (void**)&state->report_lineage.entries));
  }
  const uint32_t index = (uint32_t)state->report_lineage.index.count;
  state->report_lineage.entries[index] =
      (loom_low_target_legalize_report_lineage_entry_t){
          .op = op,
          .decision = decision,
      };
  loom_intern_table_insert(&state->report_lineage.index, probe.slot, hash,
                           index);
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_report_operation_created(
    void* user_data, const loom_op_t* source_op, loom_op_t* created_op) {
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  loom_low_target_legalize_report_decision_t* decision =
      loom_low_target_legalize_report_owner(state, source_op);
  return decision != NULL ? loom_low_target_legalize_report_lineage_insert(
                                state, created_op, decision)
                          : iree_ok_status();
}

static iree_status_t loom_low_target_legalize_record_report_source_op(
    loom_low_target_legalize_function_state_t* state, loom_op_t* op) {
  if (state->report_decisions.count == state->report_decisions.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->query_scope_arena, state->report_decisions.count,
        iree_max((iree_host_size_t)16, state->report_decisions.count + 1),
        sizeof(*state->report_decisions.entries),
        &state->report_decisions.capacity,
        (void**)&state->report_decisions.entries));
  }
  state->report_decisions.entries[state->report_decisions.count++] =
      (loom_low_target_legalize_report_decision_t){.op = op};
  return iree_ok_status();
}

static bool loom_low_target_legalize_entry_matches(
    const loom_low_target_legalize_function_state_t* state,
    const loom_target_legalizer_entry_t* entry, const loom_op_t* op) {
  if (entry->first_operand_element_types) {
    const loom_type_t operand_type =
        loom_module_value_type(state->module, loom_op_operands(op)[0]);
    if (!loom_scalar_type_set_contains(entry->first_operand_element_types,
                                       loom_type_element_type(operand_type))) {
      return false;
    }
  }
  return entry->match == NULL ||
         entry->match(entry, &state->legalization_context, op);
}

// Filters a nonempty registry span to its first applicable row, or an empty
// span. Later rows remain in provider order and still require their own check.
static loom_target_legalizer_op_entry_t loom_low_target_legalize_filter_entries(
    const loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    loom_target_legalizer_op_entry_t op_entry) {
  do {
    const loom_target_legalizer_entry_t* entry =
        &state->legalizer_registry->entries[op_entry.entry_start];
    if (loom_low_target_legalize_entry_matches(state, entry, op)) {
      break;
    }
    ++op_entry.entry_start;
  } while (--op_entry.entry_count);
  return op_entry;
}

static iree_status_t loom_low_target_legalize_capture_report_source_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  loom_target_legalizer_op_entry_t op_entry =
      loom_target_legalizer_registry_lookup_kind(state->legalizer_registry,
                                                 op->kind);
  const bool is_packet_candidate =
      state->legalization_context.vector_packet_policy != NULL &&
      state->legalization_context.policy !=
          LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY &&
      iree_all_bits_set(loom_op_effective_traits(state->module, op),
                        LOOM_TRAIT_DECOMPOSABLE);
  if (loom_target_legalizer_op_entry_is_empty(op_entry)) {
    return is_packet_candidate
               ? loom_low_target_legalize_record_report_source_op(state, op)
               : iree_ok_status();
  }
  op_entry = loom_low_target_legalize_filter_entries(state, op, op_entry);
  if (loom_target_legalizer_op_entry_is_empty(op_entry) &&
      !is_packet_candidate) {
    return iree_ok_status();
  }
  return loom_low_target_legalize_record_report_source_op(state, op);
}

static iree_status_t loom_low_target_legalize_capture_report_source_ops(
    loom_low_target_legalize_function_state_t* state) {
  if (state->compile_report == NULL) {
    return iree_ok_status();
  }
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      state->module, state->selection->func, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){
          .fn = loom_low_target_legalize_capture_report_source_op,
          .user_data = state,
      },
      &walk_result));
  loom_low_target_legalize_sort_report_sources(state->report_decisions.entries,
                                               state->report_decisions.count);
  if (state->report_decisions.count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_intern_table_initialize(state->query_scope_arena, /*capacity=*/0,
                                     &state->report_lineage.index));
    state->report_lineage.initialized = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_prepare_rewriter(
    void* user_data, loom_greedy_rewrite_driver_t* driver,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  (void)function;
  (void)region;
  (void)parent_op;
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  if (state->report_lineage.initialized) {
    loom_rewriter_set_operation_creation_callback(
        &driver->rewriter,
        (loom_rewriter_operation_creation_callback_t){
            .user_data = state,
            .fn = loom_low_target_legalize_report_operation_created,
        });
  }
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_retain_report_decision(
    loom_low_target_legalize_function_state_t* state,
    loom_low_target_legalize_report_decision_t* decision,
    iree_string_view_t source_op_name, uint32_t source_op_kind,
    loom_target_compile_report_legalization_action_t action,
    const loom_target_contract_query_result_t* query_result,
    const loom_target_legalizer_entry_t* legalizer_entry, bool scalarized,
    uint64_t created_op_count, uint64_t erased_op_count) {
  if (loom_low_target_legalize_report_wants_rows(state) && !decision->row) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(state->query_scope_arena,
                                             sizeof(*decision->row),
                                             (void**)&decision->row));
    memset(decision->row, 0, sizeof(*decision->row));
  }
  if (decision->action == LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE) {
    if (state->report_decisions.last) {
      state->report_decisions.last->next = decision;
    } else {
      state->report_decisions.first = decision;
    }
    state->report_decisions.last = decision;
  }
  const loom_target_compile_report_legalizer_strategy_t legalizer_strategy =
      legalizer_entry ? loom_low_target_legalize_report_legalizer_strategy(
                            legalizer_entry->provider_strategy)
                      : LOOM_TARGET_COMPILE_REPORT_LEGALIZER_STRATEGY_NONE;
  decision->action = action;
  decision->strategy = legalizer_strategy;
  decision->scalarized = scalarized || decision->descendant_scalarized;
  decision->created_op_count =
      created_op_count + decision->descendant_created_op_count;
  decision->erased_op_count =
      erased_op_count + decision->descendant_erased_op_count;
  if (!decision->row) {
    return iree_ok_status();
  }

  const loom_target_bundle_t* bundle =
      loom_low_source_selection_target_bundle(state->selection);
  const loom_target_config_t* config = bundle ? bundle->config : NULL;
  const iree_string_view_t legalizer_name = legalizer_entry
                                                ? legalizer_entry->provider_name
                                                : iree_string_view_empty();
  const loom_target_compile_report_legalization_row_t row = {
      .function_name = loom_low_target_legalize_function_name(state),
      .source_op_name = source_op_name,
      .source_op_kind = source_op_kind,
      .target_bundle_name = bundle ? bundle->name : iree_string_view_empty(),
      .target_config_name = config ? config->name : iree_string_view_empty(),
      .legalizer_name = legalizer_name,
      .legalizer_strategy = legalizer_strategy,
      .mode = loom_low_target_legalize_report_mode(
          state->legalization_context.mode),
      .policy = loom_low_target_legalize_report_policy(
          state->legalization_context.policy),
      .action = action,
      .scalarized = decision->scalarized,
      .legalization_outcome =
          loom_low_target_legalize_report_legalization_outcome(
              action, legalizer_strategy),
      .contract_outcome = loom_low_target_legalize_report_contract_outcome(
          query_result->outcome),
      .descriptor_key =
          loom_low_target_legalize_report_descriptor_key(state, query_result),
      .source_rejection_bits = query_result->source_rejection_bits,
      .source_rejection_detail = query_result->source_rejection_detail,
      .target_rejection_bits = query_result->target_rejection_bits,
      .missing_feature_bits = query_result->missing_feature_bits,
      .missing_fact_bits = query_result->missing_fact_bits,
      .created_op_count = decision->created_op_count,
      .erased_op_count = decision->erased_op_count,
  };
  *decision->row = row;
  return iree_ok_status();
}

static void loom_low_target_legalize_accumulate_descendant_rewrite(
    loom_low_target_legalize_report_decision_t* decision, bool scalarized,
    uint64_t created_op_count, uint64_t erased_op_count) {
  decision->descendant_scalarized |= scalarized;
  decision->descendant_created_op_count += created_op_count;
  decision->descendant_erased_op_count += erased_op_count;
  decision->scalarized |= scalarized;
  decision->created_op_count += created_op_count;
  decision->erased_op_count += erased_op_count;
  if (decision->row != NULL) {
    decision->row->scalarized = decision->scalarized;
    decision->row->created_op_count = decision->created_op_count;
    decision->row->erased_op_count = decision->erased_op_count;
  }
}

// Native acceptance supersedes a tentative decision. A successful in-place
// rewrite still describes work performed on the original source operation.
static void loom_low_target_legalize_report_accept_native(
    loom_low_target_legalize_report_decision_t* decision) {
  if (decision &&
      decision->action != LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_NONE &&
      decision->action !=
          LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN) {
    decision->action = LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL;
    decision->scalarized = false;
  }
}

static iree_status_t loom_low_target_legalize_publish_report(
    const loom_low_target_legalize_function_state_t* state) {
  iree_status_t status = iree_ok_status();
  for (const loom_low_target_legalize_report_decision_t* decision =
           state->report_decisions.first;
       iree_status_is_ok(status) && decision; decision = decision->next) {
    // DCE and enclosing rewrites can erase a pending source without revisiting
    // it. Only its own successful legalizer rewrite remains reportable.
    if (decision->action ==
            LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_LEGAL ||
        (iree_any_bit_set(decision->op->flags, LOOM_OP_FLAG_DEAD) &&
         decision->action !=
             LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN)) {
      continue;
    }
    if (decision->row) {
      status = loom_target_compile_report_record_legalization_row(
          state->compile_report, decision->row);
    } else {
      loom_target_compile_report_record_legalization_summary(
          state->compile_report, decision->action, decision->strategy,
          decision->scalarized, decision->created_op_count);
    }
  }
  return status;
}

static loom_target_contract_query_result_t
loom_low_target_legalize_result_for_legalizer_report(
    const loom_target_contract_query_result_t* query_result,
    const loom_target_legalizer_result_t* legalizer_result) {
  loom_target_contract_query_result_t report_result = *query_result;
  report_result.source_rejection_bits |=
      legalizer_result->source_rejection_bits;
  if (legalizer_result->source_rejection_detail !=
      LOOM_CONTRACT_REJECTION_DETAIL_NONE) {
    report_result.source_rejection_detail =
        legalizer_result->source_rejection_detail;
  }
  report_result.target_rejection_bits |=
      legalizer_result->target_rejection_bits;
  report_result.missing_feature_bits |= legalizer_result->missing_feature_bits;
  report_result.missing_fact_bits |= legalizer_result->missing_fact_bits;
  return report_result;
}

static bool loom_low_target_legalize_op_has_reference_legalizer(
    const loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    loom_target_legalizer_op_entry_t op_entry) {
  for (uint16_t i = 0; i < op_entry.entry_count; ++i) {
    const loom_target_legalizer_entry_t* entry =
        &state->legalizer_registry->entries[op_entry.entry_start + i];
    if (i != 0 && !loom_low_target_legalize_entry_matches(state, entry, op)) {
      continue;
    }
    if (entry->provider_strategy == LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE) {
      return true;
    }
  }
  return false;
}

static bool loom_low_target_legalize_op_has_legal_rewrite_entry(
    const loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    loom_target_legalizer_op_entry_t op_entry) {
  for (uint16_t i = 0; i < op_entry.entry_count; ++i) {
    const loom_target_legalizer_entry_t* entry =
        &state->legalizer_registry->entries[op_entry.entry_start + i];
    if (i != 0 && !loom_low_target_legalize_entry_matches(state, entry, op)) {
      continue;
    }
    if (state->legalization_context.policy ==
            LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY &&
        entry->provider_strategy == LOOM_TARGET_LEGALIZER_STRATEGY_TARGET) {
      continue;
    }
    if (iree_any_bit_set(entry->flags,
                         LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL)) {
      return true;
    }
  }
  return false;
}

static bool loom_low_target_legalize_should_accept_legal_contract(
    const loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    loom_target_legalizer_op_entry_t op_entry) {
  if (loom_low_target_legalize_op_has_legal_rewrite_entry(state, op,
                                                          op_entry)) {
    return false;
  }
  return state->legalization_context.policy !=
             LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY ||
         !loom_low_target_legalize_op_has_reference_legalizer(state, op,
                                                              op_entry);
}

static bool loom_low_target_legalize_should_skip_entry(
    const loom_low_target_legalize_function_state_t* state,
    const loom_target_legalizer_entry_t* entry,
    const loom_target_contract_query_result_t* query_result) {
  if (state->legalization_context.policy ==
      LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY) {
    return entry->provider_strategy == LOOM_TARGET_LEGALIZER_STRATEGY_TARGET;
  }
  // An opted-in legal rewrite may decline an op for another target. That does
  // not make ordinary fallback entries eligible to rewrite an accepted op.
  if (query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
    return !iree_any_bit_set(entry->flags,
                             LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REWRITE_LEGAL);
  }
  if (iree_any_bit_set(
          entry->flags,
          LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION) &&
      query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return true;
  }
  if (iree_any_bit_set(
          entry->flags,
          LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_UNHANDLED) &&
      query_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return true;
  }
  return false;
}

static bool loom_low_target_legalize_should_reject_reference_entry(
    const loom_low_target_legalize_function_state_t* state,
    const loom_target_legalizer_entry_t* entry) {
  return state->legalization_context.policy ==
             LOOM_TARGET_LEGALIZATION_POLICY_REQUIRE_NATIVE &&
         entry->provider_strategy == LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE;
}

static loom_target_legalizer_result_t
loom_low_target_legalize_reference_policy_rejection(
    const loom_low_target_legalize_function_state_t* state) {
  return (loom_target_legalizer_result_t){
      .action = state->legalization_context.mode ==
                        LOOM_TARGET_LEGALIZATION_MODE_FINAL
                    ? LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL
                    : LOOM_TARGET_LEGALIZER_ACTION_DEFER,
      .source_rejection_bits = LOOM_CONTRACT_REJECTION_POLICY,
  };
}

static bool loom_low_target_legalize_should_record_final_rejection(
    const loom_low_target_legalize_function_state_t* state,
    const loom_target_legalizer_entry_t* entry) {
  return state->legalization_context.mode ==
             LOOM_TARGET_LEGALIZATION_MODE_FINAL &&
         state->pass_state->verify_source_legality &&
         entry->provider_strategy == LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE;
}

static iree_status_t loom_low_target_legalize_record_final_rejection(
    loom_low_target_legalize_function_state_t* state, const loom_op_t* op,
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalizer_result_t result) {
  if (!loom_low_target_legalize_should_record_final_rejection(state, entry)) {
    return iree_ok_status();
  }
  loom_low_target_legalize_final_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      state->query_scope_arena, sizeof(*rejection), (void**)&rejection));
  *rejection = (loom_low_target_legalize_final_rejection_t){
      .op = op,
      .result = result,
      .next = state->final_rejections,
  };
  state->final_rejections = rejection;
  return iree_ok_status();
}

static const loom_low_target_legalize_final_rejection_t*
loom_low_target_legalize_find_final_rejection(
    const loom_low_target_legalize_function_state_t* state,
    const loom_op_t* op) {
  if (!state->use_final_rejections) {
    return NULL;
  }
  const loom_low_target_legalize_final_rejection_t* rejection =
      state->final_rejections;
  while (rejection != NULL) {
    if (rejection->op == op) {
      return rejection;
    }
    rejection = rejection->next;
  }
  return NULL;
}

static iree_string_view_t loom_low_target_legalize_reference_source_contract(
    void) {
  return IREE_SV("target.reference");
}

static iree_string_view_t loom_low_target_legalize_reference_constraint_key(
    const loom_target_legalizer_result_t* result) {
  switch ((loom_contract_rejection_detail_t)result->source_rejection_detail) {
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_LHS_FRAGMENT_OWNERSHIP:
      return IREE_SV("reference.lhs_fragment_ownership");
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RHS_FRAGMENT_OWNERSHIP:
      return IREE_SV("reference.rhs_fragment_ownership");
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_INIT_FRAGMENT_OWNERSHIP:
      return IREE_SV("reference.init_fragment_ownership");
    case LOOM_CONTRACT_REJECTION_DETAIL_MATRIX_RESULT_FRAGMENT_PAYLOAD:
      return IREE_SV("reference.result_fragment_payload");
    case LOOM_CONTRACT_REJECTION_DETAIL_NONE:
    default:
      break;
  }
  const uint32_t bits = result->source_rejection_bits;
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_INVALID_REQUEST)) {
    return IREE_SV("reference.request_shape");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_ROLE)) {
    return IREE_SV("reference.fragment_roles");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SHAPE)) {
    return IREE_SV("reference.fragment_shape");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_SCHEMA)) {
    return IREE_SV("reference.fragment_schema");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_AUXILIARY_OPERAND)) {
    return IREE_SV("reference.auxiliary_operands");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_NUMERIC)) {
    return IREE_SV("reference.numeric_type");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_FRAGMENT)) {
    return IREE_SV("reference.fragment_ownership");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_CAPABILITY)) {
    return IREE_SV("reference.capability");
  }
  if (iree_any_bit_set(bits, LOOM_CONTRACT_REJECTION_POLICY)) {
    return IREE_SV("reference.policy");
  }
  if (result->missing_feature_bits != 0) {
    return IREE_SV("reference.target_features");
  }
  if (result->missing_fact_bits != 0) {
    return IREE_SV("reference.value_facts");
  }
  if (result->target_rejection_bits != 0) {
    return IREE_SV("reference.target_constraints");
  }
  return IREE_SV("reference.unsupported");
}

static iree_status_t loom_low_target_legalize_make_final_rejection(
    loom_low_target_legalize_function_state_t* state,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    const loom_low_target_legalize_final_rejection_t* final_rejection,
    const loom_target_contract_rejection_t** out_rejection) {
  *out_rejection = NULL;
  loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      environment->arena, sizeof(*rejection), (void**)&rejection));
  loom_diagnostic_param_t* params = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      environment->arena, 7, sizeof(*params), (void**)&params));
  const loom_target_bundle_t* bundle =
      loom_target_contract_query_environment_bundle(environment);
  const loom_target_config_t* config = bundle ? bundle->config : NULL;
  const loom_target_export_plan_t* export_plan =
      bundle ? bundle->export_plan : NULL;
  params[0] = loom_param_string(loom_low_target_legalize_nonempty(
      bundle ? bundle->name : iree_string_view_empty(), IREE_SV("<empty>")));
  params[1] = loom_param_string(loom_low_target_legalize_nonempty(
      export_plan ? export_plan->name : iree_string_view_empty(),
      IREE_SV("<empty>")));
  params[2] = loom_param_string(loom_low_target_legalize_nonempty(
      config ? config->name : iree_string_view_empty(), IREE_SV("<empty>")));
  params[3] = loom_param_string(loom_low_target_legalize_nonempty(
      loom_low_target_legalize_function_name(state), IREE_SV("<module>")));
  params[4] = loom_param_string(loom_op_name(environment->module, source_op));
  params[5] =
      loom_param_string(loom_low_target_legalize_reference_source_contract());
  params[6] =
      loom_param_string(loom_low_target_legalize_reference_constraint_key(
          &final_rejection->result));
  *rejection = (loom_target_contract_rejection_t){
      .error_ref = LOOM_ERR_TARGET_039_REF,
      .params = params,
      .param_count = 7,
  };
  *out_rejection = rejection;
  return iree_ok_status();
}

static iree_status_t
loom_low_target_legalize_apply_final_rejection_to_contract_query(
    loom_low_target_legalize_function_state_t* state,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_target_contract_query_result_t* result) {
  // Final rejections describe the authored operation. A projected query asks
  // whether a different candidate representation is legal and cannot reuse an
  // authored-shape rejection when its target contract has no opinion.
  if (environment->vector_lane_projection.source_lane_count != 0 ||
      result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  const loom_low_target_legalize_final_rejection_t* final_rejection =
      loom_low_target_legalize_find_final_rejection(state, source_op);
  if (final_rejection == NULL) {
    return iree_ok_status();
  }
  const loom_target_contract_rejection_t* rejection = NULL;
  IREE_RETURN_IF_ERROR(loom_low_target_legalize_make_final_rejection(
      state, environment, source_op, final_rejection, &rejection));
  *result = loom_target_contract_query_result_empty();
  result->outcome = final_rejection->result.action ==
                            LOOM_TARGET_LEGALIZER_ACTION_REJECT_INVALID_IR
                        ? LOOM_TARGET_CONTRACT_QUERY_INVALID_IR
                        : LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  result->source_rejection_bits = final_rejection->result.source_rejection_bits;
  result->source_rejection_detail =
      final_rejection->result.source_rejection_detail;
  result->target_rejection_bits = final_rejection->result.target_rejection_bits;
  result->missing_feature_bits = final_rejection->result.missing_feature_bits;
  result->missing_fact_bits = final_rejection->result.missing_fact_bits;
  result->rejection = rejection;
  return iree_ok_status();
}

static void loom_low_target_legalize_deinitialize_query_scope(
    loom_low_target_legalize_function_state_t* state) {
  state->packet_query_cache = loom_vector_component_packet_query_cache_empty();
  if (state->query_scope == NULL) {
    return;
  }
  state->legalization_context.value_domain = NULL;
  state->legalization_context.view_regions = NULL;
  loom_low_lower_source_query_scope_deinitialize(state->query_scope);
  state->query_scope = NULL;
}

static iree_status_t loom_low_target_legalize_refresh_query_scope(
    loom_low_target_legalize_function_state_t* state,
    const loom_value_fact_table_t* fact_table) {
  loom_low_target_legalize_deinitialize_query_scope(state);
  state->lower_options.fact_table = (loom_value_fact_table_t*)fact_table;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_scope_allocate(
      state->module, state->selection->func, &state->lower_options,
      state->query_scope_arena, &state->query_scope));
  state->legalization_context.value_domain =
      loom_low_lower_source_query_scope_value_domain(state->query_scope);
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_scope_view_regions(
      state->query_scope, &state->legalization_context.view_regions));
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_ensure_query_scope(
    loom_low_target_legalize_function_state_t* state,
    const loom_value_fact_table_t* fact_table) {
  if (state->query_scope != NULL &&
      state->lower_options.fact_table == fact_table) {
    return iree_ok_status();
  }
  return loom_low_target_legalize_refresh_query_scope(state, fact_table);
}

static iree_status_t loom_low_target_legalize_query_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_low_target_legalize_ensure_query_scope(
      state, environment->fact_table));
  const loom_target_contract_query_callback_t callback =
      loom_low_lower_source_query_scope_callback(state->query_scope);
  IREE_RETURN_IF_ERROR(
      callback.fn(callback.user_data, environment, source_op, out_result));
  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_apply_final_rejection_to_contract_query(
          state, environment, source_op, out_result));
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_report_packet_rewrite(
    void* user_data, const loom_op_t* source_op,
    const loom_target_contract_query_result_t* authored_query_result,
    uint64_t created_op_count, uint64_t erased_op_count) {
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  loom_low_target_legalize_report_decision_t* report_decision =
      loom_low_target_legalize_report_owner(state, source_op);
  if (report_decision == NULL) {
    return iree_ok_status();
  }
  // Component packetization can revisit generated operations. Their work
  // remains part of the authored rewrite that created them.
  if (report_decision->op != source_op) {
    loom_low_target_legalize_accumulate_descendant_rewrite(
        report_decision, /*scalarized=*/false, created_op_count,
        erased_op_count);
    return iree_ok_status();
  }
  static const loom_target_legalizer_entry_t packet_legalizer_entry = {
      .provider_name = IREE_SVL("vector-packet"),
      .provider_strategy = LOOM_TARGET_LEGALIZER_STRATEGY_TARGET,
  };
  const bool report_wants_rows =
      loom_low_target_legalize_report_wants_rows(state);
  return loom_low_target_legalize_retain_report_decision(
      state, report_decision,
      report_wants_rows ? loom_op_name(state->module, source_op)
                        : iree_string_view_empty(),
      source_op->kind, LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
      authored_query_result, &packet_legalizer_entry,
      /*scalarized=*/false, created_op_count, erased_op_count);
}

// Returns whether reference legalization of |op| would consume a value whose
// producer owns a terminal target rejection. The producer must remain intact
// for its structured diagnostic; expanding a direct consumer would only hide
// that rejection behind lane-wise fallback IR.
static iree_status_t
loom_low_target_legalize_reference_entry_has_terminal_input(
    loom_low_target_legalize_function_state_t* state,
    const loom_target_legalizer_entry_t* entry, const loom_op_t* op,
    bool* out_has_terminal_input) {
  *out_has_terminal_input = false;
  const loom_low_lower_op_contract_query_t* target_query =
      &state->selection->policy->query_op_contract;
  if (entry->provider_strategy != LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE ||
      target_query->fn == NULL) {
    return iree_ok_status();
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_t* value = loom_module_value(state->module, operands[i]);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    const loom_op_t* producer = loom_value_def_op(value);
    if (!loom_low_lower_op_contract_query_accepts(target_query,
                                                  producer->kind)) {
      continue;
    }
    loom_target_contract_query_result_t producer_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
        &state->legalization_context, producer, &producer_result));
    if (loom_target_contract_query_result_is_terminal_rejection(
            &producer_result)) {
      *out_has_terminal_input = true;
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_retain_packet_plan_values(
    loom_low_target_legalize_function_state_t* state, loom_value_id_t begin,
    loom_value_id_t end) {
  if (begin == LOOM_VALUE_ID_INVALID || end == LOOM_VALUE_ID_INVALID ||
      begin == end) {
    return iree_ok_status();
  }
  IREE_ASSERT_LT(begin, end);
  if (state->pending_packet_plan_values.count != 0) {
    IREE_ASSERT_LE(state->pending_packet_plan_values
                       .entries[state->pending_packet_plan_values.count - 1u]
                       .end,
                   begin);
  }
  if (state->pending_packet_plan_values.count ==
      state->pending_packet_plan_values.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->query_scope_arena, state->pending_packet_plan_values.count,
        state->pending_packet_plan_values.count + 1u,
        sizeof(*state->pending_packet_plan_values.entries),
        &state->pending_packet_plan_values.capacity,
        (void**)&state->pending_packet_plan_values.entries));
  }
  state->pending_packet_plan_values
      .entries[state->pending_packet_plan_values.count++] =
      (loom_low_target_legalize_packet_plan_value_range_t){
          .begin = begin,
          .end = end,
      };
  return iree_ok_status();
}

static bool loom_low_target_legalize_value_needs_packet_plan(
    const loom_low_target_legalize_function_state_t* state,
    loom_value_id_t value) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = state->pending_packet_plan_values.count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2u;
    const loom_low_target_legalize_packet_plan_value_range_t range =
        state->pending_packet_plan_values.entries[middle];
    if (value < range.begin) {
      end = middle;
    } else if (value >= range.end) {
      begin = middle + 1u;
    } else {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_target_legalize_rewrite_op(
    void* user_data, loom_greedy_rewrite_driver_t* driver, loom_op_t* op,
    loom_greedy_rewrite_result_t* result, bool* out_changed) {
  *out_changed = false;
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  loom_low_target_legalize_report_decision_t* report_decision =
      loom_low_target_legalize_report_owner(state, op);
  const bool is_report_source =
      report_decision != NULL && report_decision->op == op;
  const bool observe_operation_creation = report_decision != NULL;
  driver->rewriter.flags = 0;
  const uint64_t dead_erased_op_count_before = driver->rewriter.erased_op_count;
  bool erased = false;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase_if_dead(&driver->rewriter, op, &erased));
  if (erased) {
    if (report_decision != NULL && !is_report_source) {
      loom_low_target_legalize_accumulate_descendant_rewrite(
          report_decision, /*scalarized=*/false, /*created_op_count=*/0,
          driver->rewriter.erased_op_count - dead_erased_op_count_before);
    }
    loom_greedy_rewrite_result_record_change(
        result, &driver->rewriter,
        LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
    *out_changed = true;
    return iree_ok_status();
  }
  if (state->preflight.error_count != 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_target_legalize_preflight_check_assume(
      &state->preflight, op, driver->fact_table));
  if (state->preflight.error_count != 0) {
    return iree_ok_status();
  }

  state->legalization_context.fact_table = driver->fact_table;

  loom_target_legalizer_op_entry_t op_entry =
      loom_target_legalizer_registry_lookup_kind(state->legalizer_registry,
                                                 op->kind);
  if (loom_target_legalizer_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }
  op_entry = loom_low_target_legalize_filter_entries(state, op, op_entry);
  if (loom_target_legalizer_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }
  const bool report_wants_rows =
      is_report_source && loom_low_target_legalize_report_wants_rows(state);
  const iree_string_view_t source_op_name =
      report_wants_rows ? loom_op_name(state->module, op)
                        : iree_string_view_empty();
  const uint32_t source_op_kind = op->kind;

  state->legalization_context.rewriter = &driver->rewriter;
  state->legalization_context.arena = driver->scratch_arena;
  loom_target_contract_query_result_t query_result_storage =
      loom_target_contract_query_result_empty();
  const loom_target_contract_query_result_t* query_result =
      loom_vector_component_packet_query_cache_lookup(
          &state->packet_query_cache, op);
  if (query_result == NULL) {
    IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
        &state->legalization_context, op, &query_result_storage));
    query_result = &query_result_storage;
  }
  // Packet materialization earlier in this worklist can create a
  // shape-preserving vector operation after the iteration's component plan has
  // run. Give the next plan first claim when the target admits a multi-lane
  // packet width. Ordinary target and reference rewrites never enter this
  // handoff and consume their generated operations in the current worklist.
  if (!state->legalization_context
           .source_function_has_unsupported_vector_carrier &&
      state->legalization_context.vector_packet_policy != NULL &&
      state->legalization_context.vector_packet_policy
              ->native_lane_count_count != 0 &&
      state->legalization_context.policy !=
          LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY &&
      op->result_count == 1 &&
      loom_low_target_legalize_value_needs_packet_plan(
          state, loom_op_const_results(op)[0])) {
    bool has_packet_candidate = false;
    IREE_RETURN_IF_ERROR(loom_vector_component_packet_op_has_vector_candidate(
        &state->legalization_context, op, query_result, &has_packet_candidate));
    if (has_packet_candidate) {
      return iree_ok_status();
    }
  }
  if (query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
    if (is_report_source) {
      IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
          state, report_decision, source_op_name, source_op_kind,
          LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR,
          query_result, /*legalizer_entry=*/NULL, /*scalarized=*/false,
          /*created_op_count=*/0,
          /*erased_op_count=*/0));
    }
    ++state->statistics->ops_deferred;
    return iree_ok_status();
  }
  if (loom_target_contract_query_result_is_terminal_rejection(query_result)) {
    if (is_report_source) {
      const loom_target_compile_report_legalization_action_t report_action =
          state->legalization_context.mode ==
                  LOOM_TARGET_LEGALIZATION_MODE_FINAL
              ? LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL
              : LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED;
      IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
          state, report_decision, source_op_name, source_op_kind, report_action,
          query_result, /*legalizer_entry=*/NULL,
          /*scalarized=*/false, /*created_op_count=*/0,
          /*erased_op_count=*/0));
    }
    ++state->statistics->ops_deferred;
    return iree_ok_status();
  }
  if (query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL &&
      loom_low_target_legalize_should_accept_legal_contract(state, op,
                                                            op_entry)) {
    loom_low_target_legalize_report_accept_native(
        is_report_source ? report_decision : NULL);
    ++state->statistics->ops_legal;
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < op_entry.entry_count; ++i) {
    const loom_target_legalizer_entry_t* entry =
        &state->legalizer_registry->entries[op_entry.entry_start + i];
    if (i != 0 && !loom_low_target_legalize_entry_matches(state, entry, op)) {
      continue;
    }
    if (loom_low_target_legalize_should_skip_entry(state, entry,
                                                   query_result)) {
      continue;
    }
    driver->rewriter.flags = 0;
    state->legalization_context.component_packet_materialization.recorded =
        false;
    const uint64_t created_op_count_before = driver->rewriter.created_op_count;
    const uint64_t erased_op_count_before = driver->rewriter.erased_op_count;
    const uint64_t scalarized_op_count_before =
        driver->rewriter.scalarized_op_count;
    loom_target_legalizer_result_t legalizer_result = {
        .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
    };
    bool has_terminal_input = false;
    IREE_RETURN_IF_ERROR(
        loom_low_target_legalize_reference_entry_has_terminal_input(
            state, entry, op, &has_terminal_input));
    if (has_terminal_input) {
      legalizer_result.action = LOOM_TARGET_LEGALIZER_ACTION_DEFER;
    } else if (loom_low_target_legalize_should_reject_reference_entry(state,
                                                                      entry)) {
      legalizer_result =
          loom_low_target_legalize_reference_policy_rejection(state);
    } else {
      state->legalization_context.contract_query_result = query_result;
      if (observe_operation_creation) {
        loom_rewriter_set_operation_creation_source(&driver->rewriter, op);
      }
      iree_status_t legalizer_status = entry->legalize(
          entry, &state->legalization_context, op, &legalizer_result);
      if (observe_operation_creation) {
        loom_rewriter_set_operation_creation_source(&driver->rewriter, NULL);
      }
      state->legalization_context.contract_query_result = NULL;
      IREE_RETURN_IF_ERROR(legalizer_status);
    }
    const bool legalizer_mutated =
        iree_any_bit_set(driver->rewriter.flags, LOOM_REWRITER_FLAG_CHANGED);
    if (legalizer_result.action == LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN) {
      if (!legalizer_mutated) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "target legalizer reported REWRITTEN without mutating IR");
      }
    } else if (legalizer_mutated) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target legalizer action %d mutated IR without reporting REWRITTEN",
          (int)legalizer_result.action);
    }
    switch (legalizer_result.action) {
      case LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT:
        continue;
      case LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN: {
        if (state->legalization_context.component_packet_materialization
                .recorded) {
          IREE_RETURN_IF_ERROR(
              loom_low_target_legalize_retain_packet_plan_values(
                  state,
                  state->legalization_context.component_packet_materialization
                      .value_checkpoint,
                  loom_rewriter_value_checkpoint(&driver->rewriter)));
        }
        loom_greedy_rewrite_result_record_change(
            result, &driver->rewriter,
            LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
        const loom_target_contract_query_result_t rewritten_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                query_result, &legalizer_result);
        const bool scalarized =
            driver->rewriter.scalarized_op_count != scalarized_op_count_before;
        const uint64_t created_op_count =
            driver->rewriter.created_op_count - created_op_count_before;
        const uint64_t erased_op_count =
            driver->rewriter.erased_op_count - erased_op_count_before;
        if (is_report_source) {
          IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
              state, report_decision, source_op_name, source_op_kind,
              LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REWRITTEN,
              &rewritten_report_result, entry, scalarized, created_op_count,
              erased_op_count));
        } else if (report_decision != NULL) {
          loom_low_target_legalize_accumulate_descendant_rewrite(
              report_decision, scalarized, created_op_count, erased_op_count);
        }
        ++state->statistics->ops_rewritten;
        *out_changed = true;
        return iree_ok_status();
      }
      case LOOM_TARGET_LEGALIZER_ACTION_DEFER: {
        const loom_target_contract_query_result_t deferred_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                query_result, &legalizer_result);
        if (is_report_source) {
          IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
              state, report_decision, source_op_name, source_op_kind,
              LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_DEFERRED,
              &deferred_report_result, entry, /*scalarized=*/false,
              /*created_op_count=*/0, /*erased_op_count=*/0));
        }
        ++state->statistics->ops_deferred;
        return iree_ok_status();
      }
      case LOOM_TARGET_LEGALIZER_ACTION_REJECT_INVALID_IR: {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_record_final_rejection(
            state, op, entry, legalizer_result));
        const loom_target_contract_query_result_t invalid_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                query_result, &legalizer_result);
        if (is_report_source) {
          IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
              state, report_decision, source_op_name, source_op_kind,
              LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_INVALID_IR,
              &invalid_report_result, entry, /*scalarized=*/false,
              /*created_op_count=*/0, /*erased_op_count=*/0));
        }
        ++state->statistics->ops_deferred;
        return iree_ok_status();
      }
      case LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL: {
        IREE_RETURN_IF_ERROR(loom_low_target_legalize_record_final_rejection(
            state, op, entry, legalizer_result));
        const loom_target_contract_query_result_t unsupported_report_result =
            loom_low_target_legalize_result_for_legalizer_report(
                query_result, &legalizer_result);
        if (is_report_source) {
          IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
              state, report_decision, source_op_name, source_op_kind,
              LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_REJECT_UNSUPPORTED_FINAL,
              &unsupported_report_result, entry, /*scalarized=*/false,
              /*created_op_count=*/0, /*erased_op_count=*/0));
        }
        ++state->statistics->ops_deferred;
        return iree_ok_status();
      }
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "target legalizer returned unknown action %d",
                                (int)legalizer_result.action);
    }
  }

  if (query_result->outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
    loom_low_target_legalize_report_accept_native(
        is_report_source ? report_decision : NULL);
    ++state->statistics->ops_legal;
    return iree_ok_status();
  }

  ++state->statistics->ops_deferred;
  if (is_report_source) {
    IREE_RETURN_IF_ERROR(loom_low_target_legalize_retain_report_decision(
        state, report_decision, source_op_name, source_op_kind,
        LOOM_TARGET_COMPILE_REPORT_LEGALIZATION_ACTION_UNHANDLED, query_result,
        /*legalizer_entry=*/NULL, /*scalarized=*/false,
        /*created_op_count=*/0,
        /*erased_op_count=*/0));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_target_legalize_packet_components(
    void* user_data, loom_greedy_rewrite_driver_t* driver,
    loom_region_t* region, loom_greedy_rewrite_result_t* result,
    bool* out_changed) {
  (void)region;
  *out_changed = false;
  loom_low_target_legalize_function_state_t* state =
      (loom_low_target_legalize_function_state_t*)user_data;
  state->pending_packet_plan_values.count = 0;
  if (state->preflight.error_count != 0 ||
      state->legalization_context.vector_packet_policy == NULL ||
      state->legalization_context.policy ==
          LOOM_TARGET_LEGALIZATION_POLICY_REFERENCE_ONLY ||
      !loom_value_fact_table_has_result_type_kind(driver->fact_table,
                                                  LOOM_TYPE_VECTOR)) {
    return iree_ok_status();
  }

  state->legalization_context.fact_table = driver->fact_table;
  state->legalization_context.rewriter = &driver->rewriter;
  state->legalization_context.arena = driver->scratch_arena;
  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_ensure_query_scope(state, driver->fact_table));
  uint32_t rewritten_op_count = 0;
  iree_status_t packet_status = loom_vector_component_packet_legalize(
      &state->legalization_context,
      state->compile_report != NULL
          ? (loom_vector_component_packet_rewrite_callback_t){
                .fn = loom_low_target_legalize_report_packet_rewrite,
                .user_data = state,
            }
          : (loom_vector_component_packet_rewrite_callback_t){0},
      &state->packet_query_cache,
      &rewritten_op_count);
  if (state->report_lineage.initialized) {
    loom_rewriter_set_operation_creation_source(&driver->rewriter, NULL);
  }
  IREE_RETURN_IF_ERROR(packet_status);
  if (rewritten_op_count == 0) {
    return iree_ok_status();
  }
  loom_greedy_rewrite_result_record_change(
      result, &driver->rewriter,
      LOOM_GREEDY_REWRITE_CHANGE_FLAG_COUNT_MODIFIED_OP);
  state->statistics->ops_rewritten += rewritten_op_count;
  *out_changed = true;
  return iree_ok_status();
}

static void loom_low_target_legalize_changed(
    void* user_data, loom_greedy_rewrite_driver_t* driver) {
  (void)driver;
  loom_low_target_legalize_deinitialize_query_scope(
      (loom_low_target_legalize_function_state_t*)user_data);
}

static iree_status_t loom_low_target_legalize_verify_final(
    loom_module_t* module, loom_low_target_legalize_function_state_t* state,
    const loom_low_target_legalize_pass_state_t* pass_state,
    const loom_value_fact_table_t* fact_table, uint32_t* out_error_count) {
  *out_error_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_refresh_query_scope(state, fact_table));
  const loom_target_contract_query_callback_t contract_query = {
      .fn = loom_low_target_legalize_query_contract,
      .user_data = state,
  };
  loom_target_contract_query_environment_t environment = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_scope_environment_initialize(
      state->query_scope, &environment));
  loom_target_low_legality_result_t result = {0};
  const loom_target_low_legality_options_t legality_options = {
      .environment = &environment,
      .error_catalog = state->selection->policy->error_catalog,
      .provider_list = state->legality_provider_list,
      .contract_query = contract_query,
      .type_supported = state->selection->policy->source_type_supported,
      .source_vector_carrier_supported =
          state->selection->policy->source_vector_carrier_supported,
      .source_function_vector_carrier_supported =
          state->selection->policy->source_function_vector_carrier_supported,
      .structural_legality_flags =
          LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_SCF |
          LOOM_TARGET_LOW_STRUCTURAL_LEGALITY_ALLOW_SOURCE_CFG,
      .emitter = state->pass->diagnostic_emitter,
      .max_errors = pass_state->max_errors,
  };
  state->use_final_rejections = true;
  iree_status_t status = loom_target_low_verify_function_legality(
      module, state->selection->func, &legality_options, &result);
  state->use_final_rejections = false;
  *out_error_count = result.error_count;
  return status;
}

static bool loom_low_target_legalize_function_has_source_vector_carriers(
    const loom_module_t* module, loom_func_like_t function,
    const loom_value_fact_table_t* fact_table,
    const loom_low_lower_policy_t* policy) {
  loom_target_vector_carrier_supported_callback_t callback =
      policy->source_function_vector_carrier_supported;
  if (callback.fn == NULL) {
    callback = policy->source_vector_carrier_supported;
  }
  if (callback.fn == NULL) {
    return true;
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, argument_ids[i]);
    if (loom_type_is_vector(type) &&
        !callback.fn(callback.user_data, module, fact_table, argument_ids[i],
                     type)) {
      return false;
    }
  }
  const loom_value_id_t* result_ids = loom_op_const_results(function.op);
  for (uint16_t i = 0; i < function.op->result_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, result_ids[i]);
    if (loom_type_is_vector(type) &&
        !callback.fn(callback.user_data, module, fact_table, result_ids[i],
                     type)) {
      return false;
    }
  }
  return true;
}

// Keeps per-function state and cold report machinery out of the module loop.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_low_target_legalize_function(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_target_legalize_pass_state_t* pass_state,
    const loom_low_source_selection_t* selection,
    const loom_target_legalizer_registry_t* legalizer_registry,
    loom_target_low_legality_provider_list_t legality_provider_list,
    iree_arena_allocator_t* arena, bool* out_emitted_error_diagnostics) {
  *out_emitted_error_diagnostics = false;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_descriptor_registry_t* descriptor_registry =
      loom_low_pass_capability_descriptor_registry(low_capability);

  loom_low_target_legalize_function_state_t state = {
      .pass = pass,
      .pass_state = pass_state,
      .statistics = loom_low_target_legalize_statistics(pass),
      .module = module,
      .selection = selection,
      .query_scope_arena = arena,
      .legalizer_registry = legalizer_registry,
      .legality_provider_list = legality_provider_list,
      .compile_report = loom_low_pass_capability_compile_report(low_capability),
  };
  IREE_RETURN_IF_ERROR(
      loom_target_low_descriptor_set_select_for_source_lowering(
          descriptor_registry,
          loom_low_source_selection_target_bundle(selection),
          &state.descriptor_set));
  loom_low_target_legalize_preflight_initialize(
      module, selection->func,
      loom_low_source_selection_target_bundle(selection),
      pass->diagnostic_emitter, pass_state->max_errors, &state.preflight);

  const loom_pass_value_fact_scope_t fact_scope =
      loom_pass_value_fact_scope_function_for_target(selection->func,
                                                     selection->target_facts);
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(
      loom_pass_value_facts_acquire(pass, module, fact_scope, &fact_table));
  state.lower_options = (loom_low_lower_options_t){
      .target_ref = selection->target_ref,
      .target_facts = selection->target_facts,
      .descriptor_registry = descriptor_registry,
      .legality_provider_list = legality_provider_list,
      .policy = selection->policy,
      .fact_table = fact_table,
      .emitter = pass->diagnostic_emitter,
      .max_errors = pass_state->max_errors,
  };

  state.legalization_context = (loom_target_legalization_context_t){
      .pass = pass,
      .module = module,
      .function = selection->func,
      .target_facts = selection->target_facts,
      .descriptor_set = state.descriptor_set,
      .vector_packet_policy = selection->policy->vector_packet_policy,
      .bitstream_policy = selection->policy->bitstream_policy,
      .source_vector_carrier_supported =
          selection->policy->source_vector_carrier_supported,
      .source_function_has_unsupported_vector_carrier =
          !loom_low_target_legalize_function_has_source_vector_carriers(
              module, selection->func, fact_table, selection->policy),
      .mode = pass_state->mode,
      .boundary_representation = pass_state->boundary_representation,
      .policy = pass_state->policy,
      .fact_table = fact_table,
      .contract_query =
          {
              .fn = loom_low_target_legalize_query_contract,
              .user_data = &state,
          },
  };
  IREE_RETURN_IF_ERROR(
      loom_low_target_legalize_capture_report_source_ops(&state));

  iree_arena_allocator_t rewrite_arena;
  iree_arena_initialize(module->arena.block_pool, &rewrite_arena);
  loom_greedy_rewrite_driver_t rewrite_driver;
  loom_greedy_rewrite_driver_initialize(module, &rewrite_arena, fact_table,
                                        &rewrite_driver);
  const loom_greedy_rewrite_options_t rewrite_options = {
      .max_iterations = pass_state->max_iterations,
      .math_policy = loom_target_math_policy_registry_lookup_for_bundle(
          loom_target_math_pass_capability_policy_registry(
              loom_target_math_pass_capability_from_pass(pass)),
          loom_low_source_selection_target_bundle(selection)),
      .math_target_bundle = loom_low_source_selection_target_bundle(selection),
  };
  const loom_greedy_rewrite_callbacks_t rewrite_callbacks = {
      .user_data = &state,
      .prepare_region = loom_low_target_legalize_prepare_rewriter,
      .before_worklist = loom_low_target_legalize_packet_components,
      .rewrite_op = loom_low_target_legalize_rewrite_op,
      .changed = loom_low_target_legalize_changed,
  };
  loom_greedy_rewrite_result_t rewrite_result = {0};
  iree_status_t status = loom_greedy_rewrite_run_region(
      &rewrite_driver, selection->func, loom_func_like_body(selection->func),
      selection->func.op, &rewrite_options, &rewrite_callbacks,
      &rewrite_result);
  // Queries and rewrites borrow the pass-owned facts. Release their state
  // before invalidation so final legality can acquire a fresh scope after IR
  // changes.
  loom_low_target_legalize_deinitialize_query_scope(&state);
  loom_greedy_rewrite_driver_deinitialize(&rewrite_driver);
  iree_arena_deinitialize(&rewrite_arena);
  if (!iree_status_is_ok(status) || rewrite_result.changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  if (iree_status_is_ok(status) && rewrite_result.changed) {
    loom_pass_mark_changed(pass);
  }

  // Publish source decisions once, including completed decisions from a run
  // that failed. Report allocation failures join the original failure.
  status =
      iree_status_join(status, loom_low_target_legalize_publish_report(&state));

  // Source legality describes the input to source-to-low. Low functions also
  // participate in rewriting (for example, to discharge assumptions), but their
  // register, storage, and instruction contracts belong to Low verification.
  const bool is_source_function =
      selection->func.op->kind == LOOM_OP_FUNC_DEF ||
      selection->func.op->kind == LOOM_OP_KERNEL_DEF;
  uint32_t final_error_count = 0;
  if (iree_status_is_ok(status) && state.preflight.error_count == 0 &&
      pass_state->verify_source_legality && is_source_function) {
    status = loom_pass_value_facts_acquire(
        pass, module,
        loom_pass_value_fact_scope_conditioned_function_for_target(
            selection->func, selection->target_facts),
        &fact_table);
    if (iree_status_is_ok(status)) {
      status = loom_low_target_legalize_verify_final(
          module, &state, pass_state, fact_table, &final_error_count);
    }
  }
  loom_low_target_legalize_deinitialize_query_scope(&state);
  IREE_RETURN_IF_ERROR(status);

  loom_low_target_legalize_statistics(pass)->errors +=
      (int64_t)(state.preflight.error_count + final_error_count);
  *out_emitted_error_diagnostics =
      state.preflight.error_count != 0 || final_error_count != 0;
  return iree_ok_status();
}

iree_status_t loom_low_target_legalize_run(loom_pass_t* pass,
                                           loom_module_t* module) {
  const loom_low_target_legalize_pass_state_t* state =
      (const loom_low_target_legalize_pass_state_t*)pass->state;
  const loom_low_pass_capability_t* low_capability =
      loom_low_pass_capability_from_pass(pass);
  const loom_low_lower_policy_registry_t* policy_registry =
      loom_low_pass_capability_lower_policy_registry(low_capability);
  const loom_target_pass_capability_t* target_capability =
      loom_target_pass_capability_from_pass(pass);
  const loom_target_low_legality_provider_list_t* legality_provider_list =
      loom_low_pass_capability_legality_provider_list(low_capability);
  const loom_target_legalizer_registry_t* legalizer_registry =
      loom_low_pass_capability_legalizer_registry(low_capability);

  iree_arena_allocator_t run_arena;
  iree_arena_initialize(module->arena.block_pool, &run_arena);

  loom_low_source_selection_list_t selection_list = {0};
  const loom_low_source_selection_options_t selection_options = {
      .policy_registry = policy_registry,
      .diagnostic_emitter = pass->diagnostic_emitter,
      .function_versions =
          loom_target_pass_capability_function_versions(target_capability),
  };
  iree_status_t status = loom_low_select_target_bound_funcs(
      module, &selection_options, &run_arena, &selection_list);

  bool emitted_error_diagnostics = false;
  uint32_t function_count = 0;
  const loom_target_low_legality_provider_list_t legality_list =
      legality_provider_list ? *legality_provider_list
                             : loom_target_low_legality_provider_list_empty();
  for (iree_host_size_t i = 0;
       i < selection_list.count && iree_status_is_ok(status) &&
       !emitted_error_diagnostics;
       ++i) {
    const loom_target_vector_packet_policy_t* vector_packet_policy =
        selection_list.values[i].policy->vector_packet_policy;
    const loom_target_function_version_t* target_version =
        loom_target_function_version_const_cast(
            selection_list.values[i].version_handle);
    if (state->boundary_representation ==
            LOOM_TARGET_BOUNDARY_REPRESENTATION_PROJECTED &&
        (target_version == NULL ||
         !iree_any_bit_set(
             target_version->representation_flags,
             LOOM_TARGET_FUNCTION_REPRESENTATION_FLAG_VECTOR_PACKET_BOUNDARIES) ||
         vector_packet_policy == NULL ||
         !iree_any_bit_set(
             vector_packet_policy->flags,
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_BOUNDARY_PROJECTION))) {
      continue;
    }
    status = loom_low_target_legalize_function(
        pass, module, state, &selection_list.values[i], legalizer_registry,
        legality_list, &run_arena, &emitted_error_diagnostics);
    if (iree_status_is_ok(status)) {
      ++function_count;
    }
  }

  iree_arena_deinitialize(&run_arena);
  IREE_RETURN_IF_ERROR(status);

  loom_low_target_legalize_statistics(pass)->functions +=
      (int64_t)function_count;
  return iree_ok_status();
}
