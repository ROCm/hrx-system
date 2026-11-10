// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/pipeline/source_invocation_facts.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/conditioned_value_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/fact_propagation.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/provider.h"
#include "loom/util/reference_transfer.h"

typedef struct loom_low_source_invocation_function_t
    loom_low_source_invocation_function_t;

typedef struct loom_low_source_invocation_call_t {
  // Immutable source call, including its actual arguments and result IDs.
  loom_call_like_t source;
  // Function receiving the call's result facts.
  loom_low_source_invocation_function_t* caller;
  // Function receiving the call's argument facts.
  loom_low_source_invocation_function_t* callee;
  // Next call in the caller's outgoing list.
  struct loom_low_source_invocation_call_t* next_outgoing;
  // Next call in the callee's incoming list.
  struct loom_low_source_invocation_call_t* next_incoming;
  // Next retained edge in the source-operation lookup bucket.
  struct loom_low_source_invocation_call_t* next_bucket;
  // Conservative actual facts rebased to the callee's formal coordinates.
  loom_value_facts_t* arguments;
  // Latest local observations, published only after the local solve converges.
  loom_value_facts_t* observed_arguments;
  // Next observed call awaiting the caller's publication boundary.
  struct loom_low_source_invocation_call_t* next_observed;
  // Whether this call is linked into its caller's observed list.
  bool observed;
} loom_low_source_invocation_call_t;

typedef struct loom_low_source_invocation_return_t {
  // One verified direct-body exit and its complete result tuple.
  const loom_op_t* source;
  // Next exit of the same function.
  struct loom_low_source_invocation_return_t* next;
} loom_low_source_invocation_return_t;

typedef struct loom_low_source_invocation_local_t {
  // Caller-owned table retained until entry facts or conditioned facts change.
  loom_pass_value_fact_owner_t owner;
  // Shared incremental equation engine; never used to mutate source IR.
  loom_rewriter_t rewriter;
} loom_low_source_invocation_local_t;

struct loom_low_source_invocation_function_t {
  // Immutable source callable.
  loom_func_like_t source;
  // Exact source target selection, or NULL for an unselected callable.
  const loom_low_source_selection_t* selection;
  // Retained calls made by this function.
  loom_low_source_invocation_call_t* outgoing;
  // Retained calls entering this function.
  loom_low_source_invocation_call_t* incoming;
  // Direct-body returns joined into the result summary.
  loom_low_source_invocation_return_t* returns;
  // Result summary in function-result coordinates, owned by boundary facts.
  loom_value_facts_t* results;
  // Retained caller equations, allocated only for calls with result facts.
  loom_low_source_invocation_local_t* local;
  // Calls evaluated since the last completed local solve.
  loom_low_source_invocation_call_t* observed_calls;
  // Next pending function in the intrusive worklist.
  loom_low_source_invocation_function_t* next_pending;
  // Number of call results appended to this function's argument seeds.
  iree_host_size_t call_result_count;
  // Whether all possible callers are present and nonrecursive.
  bool refine_arguments;
  // Whether this function is already pending.
  bool pending;
  // Whether the function has been analyzed at least once.
  bool analyzed;
  // Whether a changed callee has replaced this function's call-result seeds.
  bool changed_results;
};

typedef struct loom_low_source_invocation_worklist_t {
  // Verified module shared by the graph and local fact computations.
  loom_module_t* module;
  // Scratch storage for call edges and transient function state.
  iree_arena_allocator_t* arena;
  // Durable seed facts consumed by both legalization and source planning.
  loom_low_source_invocation_facts_t* facts;
  // Function states indexed by module symbol.
  loom_low_source_invocation_function_t* functions;
  // Source-operation index populated once by the retained call graph.
  loom_low_source_invocation_call_t** call_buckets;
  // Power-of-two number of source-operation buckets.
  iree_host_size_t call_bucket_count;
  // Number of retained semantic call sites.
  iree_host_size_t call_count;
  // First function awaiting changed-input analysis.
  loom_low_source_invocation_function_t* first;
  // Last pending function, or NULL for an empty worklist.
  loom_low_source_invocation_function_t* last;
} loom_low_source_invocation_worklist_t;

static iree_host_size_t loom_low_source_invocation_call_bucket(
    const loom_low_source_invocation_worklist_t* work, const loom_op_t* op) {
  uintptr_t bits = (uintptr_t)op >> 4;
  bits ^= bits >> 16;
  return (bits * UINT32_C(0x9e3779b9)) & (work->call_bucket_count - 1);
}

static void loom_low_source_invocation_enqueue(
    loom_low_source_invocation_worklist_t* work,
    loom_low_source_invocation_function_t* function) {
  if (function->pending) {
    return;
  }
  function->pending = true;
  function->next_pending = NULL;
  if (work->last) {
    work->last->next_pending = function;
  } else {
    work->first = function;
  }
  work->last = function;
}

static loom_value_facts_t loom_low_source_invocation_unknown(
    const loom_module_t* module, loom_value_id_t value,
    const loom_low_source_selection_t* selection) {
  const loom_type_t type = loom_module_value_type(module, value);
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (loom_type_is_scalar(type) || loom_type_is_vector(type)) {
    int64_t lower = INT64_MIN;
    int64_t upper = INT64_MAX;
    if (loom_value_facts_scalar_type_domain(loom_type_element_type(type),
                                            &lower, &upper)) {
      facts = loom_value_facts_make(lower, upper, 1);
    }
  }
  if (selection && selection->target_facts &&
      selection->target_facts->storage.snapshot.subgroup_size > 1 &&
      (loom_type_is_scalar(type) || loom_type_is_vector(type) ||
       loom_type_is_buffer(type) || loom_type_is_view(type))) {
    loom_value_facts_mark_lane_distribution_for_type(type, &facts);
  }
  return facts;
}

static iree_status_t loom_low_source_invocation_collect_call(
    void* user_data, loom_symbol_id_t caller_symbol, loom_call_like_t call) {
  loom_low_source_invocation_worklist_t* work = user_data;
  if (loom_call_like_kind(call) != LOOM_CALL_LIKE_KIND_SEMANTIC) {
    return iree_ok_status();
  }
  loom_low_source_invocation_function_t* caller =
      &work->functions[caller_symbol];
  const loom_symbol_ref_t callee_symbol = loom_call_like_callee(call);
  loom_low_source_invocation_function_t* callee =
      &work->functions[callee_symbol.symbol_id];
  if (!loom_func_like_isa(callee->source) || !caller->selection) {
    return iree_ok_status();
  }
  const loom_target_function_version_t* version =
      loom_target_function_version_const_cast(
          caller->selection->version_handle);
  if (!version ||
      loom_call_like_inline_policy(call) == LOOM_INLINE_POLICY_INLINE ||
      loom_func_like_inline_policy(callee->source) ==
          LOOM_INLINE_POLICY_INLINE ||
      loom_target_provider_select_call_policy(
          &version->resolved_target, work->module, LOOM_CALL_LIKE_KIND_SEMANTIC,
          call, callee->source) != LOOM_TARGET_CALL_POLICY_DIRECT) {
    return iree_ok_status();
  }
  ((uint8_t*)work->facts->direct_call_targets)[callee_symbol.symbol_id] = 1;
  loom_low_source_invocation_call_t* edge = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(work->arena, sizeof(*edge), (void**)&edge));
  *edge =
      (loom_low_source_invocation_call_t){.source = call,
                                          .caller = caller,
                                          .callee = callee,
                                          .next_outgoing = caller->outgoing,
                                          .next_incoming = callee->incoming};
  const loom_value_slice_t operands = loom_call_like_operands(call);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(work->arena, operands.count,
                                                 sizeof(*edge->arguments),
                                                 (void**)&edge->arguments));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      work->arena, operands.count, sizeof(*edge->observed_arguments),
      (void**)&edge->observed_arguments));
  for (uint16_t i = 0; i < operands.count; ++i) {
    edge->arguments[i] = loom_low_source_invocation_unknown(
        work->module, operands.values[i], caller->selection);
  }
  caller->outgoing = edge;
  callee->incoming = edge;
  caller->call_result_count += loom_call_like_results(call).count;
  ++work->call_count;
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_narrow(
    const loom_module_t* module, loom_value_fact_table_t* target,
    loom_type_t type, const loom_value_fact_table_t* source,
    loom_value_facts_t next, loom_value_facts_t* current, bool* changed) {
  loom_value_facts_t narrowed = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_fact_table_narrow_for_type(
      target, module, type, target, *current, source, next, &narrowed));
  if (!loom_value_fact_table_facts_equal_for_type(module, type, target,
                                                  *current, target, narrowed)) {
    *current = narrowed;
    *changed = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_rebase(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    const loom_module_t* module, loom_value_id_t receiving_value,
    loom_value_fact_reference_origin_t origin, loom_value_facts_t source_facts,
    loom_value_facts_t* result) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
      target, source, module, loom_module_value_type(module, receiving_value),
      source_facts, result));
  return loom_value_facts_rebase_reference(&target->context, receiving_value,
                                           origin, result);
}

static iree_status_t loom_low_source_invocation_arguments(
    loom_low_source_invocation_worklist_t* work,
    loom_low_source_invocation_function_t* function, bool* changed) {
  if (!function->refine_arguments || !function->incoming) {
    return iree_ok_status();
  }
  uint16_t count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function->source, &count);
  loom_value_fact_table_t* facts = &work->facts->boundaries;
  for (uint16_t i = 0; i < count; ++i) {
    const loom_type_t type = loom_module_value_type(work->module, arguments[i]);
    loom_value_facts_t joined = function->incoming->arguments[i];
    for (loom_low_source_invocation_call_t* call =
             function->incoming->next_incoming;
         call; call = call->next_incoming) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          facts, work->module, type, facts, joined, facts, call->arguments[i],
          &joined));
    }
    loom_value_facts_t previous =
        loom_value_fact_table_lookup(facts, arguments[i]);
    IREE_RETURN_IF_ERROR(loom_low_source_invocation_narrow(
        work->module, facts, type, facts, joined, &previous, changed));
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(facts, arguments[i], previous));
  }
  return iree_ok_status();
}

// Local inference consumes the retained edge, records its actual arguments,
// and imports the current result summary. No callee traversal occurs here.
static iree_status_t loom_low_source_invocation_resolve_call(
    void* user_data, loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operands,
    loom_value_facts_t* results) {
  loom_low_source_invocation_worklist_t* work = user_data;
  loom_low_source_invocation_call_t* call =
      work->call_buckets[loom_low_source_invocation_call_bucket(work, op)];
  while (call && call->source.op != op) {
    call = call->next_bucket;
  }
  if (!call) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      results[i] = loom_value_facts_unknown();
    }
    return iree_ok_status();
  }
  loom_value_fact_table_t* facts = &work->facts->boundaries;
  if (context->publication && call->callee->refine_arguments) {
    uint16_t count = 0;
    const loom_value_id_t* formals =
        loom_func_like_arg_ids(call->callee->source, &count);
    for (uint16_t i = 0; i < count; ++i) {
      const loom_value_fact_reference_origin_t origin = {
          .function_symbol_id =
              loom_func_like_callee(call->callee->source).symbol_id,
          .region_index =
              loom_func_like_body_region_index(call->callee->source),
          .kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
          .entry_value_id = formals[i]};
      IREE_RETURN_IF_ERROR(loom_low_source_invocation_rebase(
          facts, context->table, module, formals[i], origin, operands[i],
          &call->observed_arguments[i]));
    }
    if (!call->observed) {
      call->observed = true;
      call->next_observed = call->caller->observed_calls;
      call->caller->observed_calls = call;
    }
  }
  const loom_value_id_t* values = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        context->table, facts, module,
        loom_module_value_type(module, values[i]),
        loom_value_fact_table_lookup(facts, values[i]), &results[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_publish_arguments(
    loom_low_source_invocation_worklist_t* work,
    loom_low_source_invocation_function_t* function) {
  loom_value_fact_table_t* facts = &work->facts->boundaries;
  while (function->observed_calls) {
    loom_low_source_invocation_call_t* call = function->observed_calls;
    function->observed_calls = call->next_observed;
    call->observed = false;
    uint16_t count = 0;
    const loom_value_id_t* formals =
        loom_func_like_arg_ids(call->callee->source, &count);
    bool changed = false;
    for (uint16_t i = 0; i < count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_source_invocation_narrow(
          work->module, facts, loom_module_value_type(work->module, formals[i]),
          facts, call->observed_arguments[i], &call->arguments[i], &changed));
    }
    if (changed) {
      loom_low_source_invocation_enqueue(work, call->callee);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_publish_results(
    loom_low_source_invocation_worklist_t* work,
    loom_low_source_invocation_function_t* function,
    const loom_value_fact_table_t* local) {
  if (!function->returns || !function->incoming) {
    return iree_ok_status();
  }
  loom_value_fact_table_t* facts = &work->facts->boundaries;
  const uint16_t count = function->source.op->result_count;
  const loom_value_id_t* results = loom_op_const_results(function->source.op);
  bool changed = false;
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_facts_t joined = loom_value_facts_unknown();
    for (loom_low_source_invocation_return_t* exit = function->returns; exit;
         exit = exit->next) {
      loom_value_facts_t incoming = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_low_source_invocation_rebase(
          facts, local, work->module, results[i],
          (loom_value_fact_reference_origin_t){0},
          loom_value_fact_table_lookup(local,
                                       loom_op_const_operands(exit->source)[i]),
          &incoming));
      if (exit == function->returns) {
        joined = incoming;
      } else {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
            facts, work->module,
            loom_module_value_type(work->module, results[i]), facts, joined,
            facts, incoming, &joined));
      }
    }
    IREE_RETURN_IF_ERROR(loom_low_source_invocation_narrow(
        work->module, facts, loom_module_value_type(work->module, results[i]),
        facts, joined, &function->results[i], &changed));
  }
  if (!changed) {
    return iree_ok_status();
  }
  for (loom_low_source_invocation_call_t* call = function->incoming; call;
       call = call->next_incoming) {
    const loom_value_slice_t actual_results =
        loom_call_like_results(call->source);
    bool caller_changed = false;
    for (uint16_t i = 0; i < count; ++i) {
      loom_value_facts_t incoming = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_low_source_invocation_rebase(
          facts, facts, work->module, actual_results.values[i],
          (loom_value_fact_reference_origin_t){0}, function->results[i],
          &incoming));
      loom_value_facts_t previous =
          loom_value_fact_table_lookup(facts, actual_results.values[i]);
      IREE_RETURN_IF_ERROR(loom_low_source_invocation_narrow(
          work->module, facts,
          loom_module_value_type(work->module, actual_results.values[i]), facts,
          incoming, &previous, &caller_changed));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
          facts, actual_results.values[i], previous));
    }
    if (caller_changed) {
      call->caller->changed_results = true;
      if (call->caller->local && call->caller->local->rewriter.fact_table) {
        IREE_RETURN_IF_ERROR(loom_rewriter_add_fact_user_to_worklist(
            &call->caller->local->rewriter, call->source.op));
      }
      loom_low_source_invocation_enqueue(work, call->caller);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_initialize_function(
    loom_low_source_invocation_worklist_t* work,
    const loom_call_graph_node_t* node, iree_arena_allocator_t* durable_arena) {
  loom_low_source_invocation_function_t* function =
      &work->functions[node->symbol_id];
  const loom_module_t* module = work->module;
  loom_value_fact_table_t* facts = &work->facts->boundaries;
  if (!function->selection && !function->incoming && !function->outgoing) {
    return iree_ok_status();
  }
  function->refine_arguments =
      loom_func_like_is_module_internal(function->source) &&
      !loom_call_graph_is_recursive(&work->facts->call_graph, node->symbol_id);
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function->source, &argument_count);
  const loom_region_descriptor_t* body_descriptor =
      loom_op_vtable_region_descriptor(
          loom_op_vtable(module, function->source.op),
          loom_func_like_body_region_index(function->source));
  const bool declared_uniform =
      body_descriptor && iree_any_bit_set(body_descriptor->flags,
                                          LOOM_REGION_CLUSTER_UNIFORM_ARGS);
  const uint16_t seed_argument_count =
      (function->selection && !declared_uniform) ||
              (function->refine_arguments && function->incoming)
          ? argument_count
          : 0;
  const iree_host_size_t seed_count =
      seed_argument_count + function->call_result_count;
  loom_value_id_t* seeds = NULL;
  if (seed_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        durable_arena, seed_count, sizeof(*seeds), (void**)&seeds));
    work->facts->functions[node->symbol_id] = (loom_value_fact_table_view_t){
        .table = facts, .value_ids = seeds, .value_count = seed_count};
  }
  const loom_target_facts_t* target =
      function->selection ? function->selection->target_facts : NULL;
  const bool launch_only =
      target && node->caller_count == 0 &&
      target->storage.export_plan.abi_kind == LOOM_TARGET_ABI_HAL_KERNEL;
  for (uint16_t i = 0; i < seed_argument_count; ++i) {
    seeds[i] = arguments[i];
    loom_value_facts_t initial = loom_low_source_invocation_unknown(
        module, arguments[i], function->selection);
    const loom_type_t type = loom_module_value_type(module, arguments[i]);
    if (launch_only &&
        (loom_type_is_scalar(type) || loom_type_is_vector(type) ||
         loom_type_is_buffer(type) || loom_type_is_view(type))) {
      loom_value_facts_mark_cluster_uniform(&initial);
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_define(facts, arguments[i], initial));
  }
  iree_host_size_t next_seed = seed_argument_count;
  for (loom_low_source_invocation_call_t* call = function->outgoing; call;
       call = call->next_outgoing) {
    const iree_host_size_t bucket =
        loom_low_source_invocation_call_bucket(work, call->source.op);
    call->next_bucket = work->call_buckets[bucket];
    work->call_buckets[bucket] = call;
    const loom_value_slice_t results = loom_call_like_results(call->source);
    for (uint16_t i = 0; i < results.count; ++i) {
      seeds[next_seed++] = results.values[i];
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
          facts, results.values[i],
          loom_low_source_invocation_unknown(module, results.values[i],
                                             function->selection)));
    }
  }
  // Selected entries still publish their launch seeds, but a function without
  // retained call edges has no invocation summary to solve or propagate.
  if (!function->incoming && !function->outgoing) {
    return iree_ok_status();
  }
  const uint16_t result_count = function->source.op->result_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(work->arena, result_count,
                                                 sizeof(*function->results),
                                                 (void**)&function->results));
  for (uint16_t i = 0; i < result_count; ++i) {
    function->results[i] = loom_low_source_invocation_unknown(
        module, loom_op_const_results(function->source.op)[i],
        function->selection);
  }
  if (result_count) {
    loom_block_t* block = NULL;
    loom_region_for_each_block(loom_func_like_body(function->source), block) {
      const loom_op_t* terminator = loom_block_const_last_op(block);
      if (!loom_func_like_op_is_body_exit(module, function->source,
                                          terminator)) {
        continue;
      }
      loom_low_source_invocation_return_t* exit = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(work->arena, sizeof(*exit), (void**)&exit));
      *exit = (loom_low_source_invocation_return_t){.source = terminator,
                                                    .next = function->returns};
      function->returns = exit;
    }
  }
  loom_low_source_invocation_enqueue(work, function);
  return iree_ok_status();
}

static iree_status_t loom_low_source_invocation_analyze(
    loom_pass_t* pass, loom_low_source_invocation_worklist_t* work,
    loom_low_source_invocation_function_t* function, bool changed_arguments,
    loom_value_fact_table_t** out_local) {
  if (function->call_result_count && !function->local) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        work->arena, sizeof(*function->local), (void**)&function->local));
    loom_pass_value_fact_owner_initialize(work->module->arena.block_pool,
                                          &function->local->owner);
    loom_rewriter_initialize(&function->local->rewriter, work->module,
                             work->arena);
  }
  loom_low_source_invocation_local_t* local = function->local;
  if (local && local->rewriter.fact_table && !changed_arguments) {
    IREE_RETURN_IF_ERROR(loom_rewriter_propagate_value_facts(&local->rewriter));
    *out_local = local->rewriter.fact_table;
    return iree_ok_status();
  }
  if (local) {
    loom_rewriter_attach_value_facts(&local->rewriter, NULL);
  }
  loom_pass_value_fact_scope_t scope =
      loom_pass_value_fact_scope_function_for_target(
          function->source,
          function->selection ? function->selection->target_facts : NULL);
  scope.seed_facts = loom_low_source_invocation_facts_for_function(
      work->facts, function->source);
  // A retained caller owns storage for its definitions instead of reserving
  // the whole module's value domain. Its first seed is an entry or call result;
  // the ordinary fact table grows this interval as definitions are computed.
  const loom_value_fact_range_t reserved_values =
      local ? (loom_value_fact_range_t){.first_value_id =
                                            scope.seed_facts.value_ids[0],
                                        .capacity = 1}
            : (loom_value_fact_range_t){0};
  // Entry seeds stay immutable for the lifetime of this local scope. Call
  // results instead come from the resolver and notify their indexed users.
  scope.seed_facts.value_count -= function->call_result_count;
  if (scope.seed_facts.value_count == 0) {
    // Empty seeds retain the ordinary fact-cache scope, including entries whose
    // uniformity is already declared by their region descriptor.
    scope.seed_facts = (loom_value_fact_table_view_t){0};
  }
  loom_value_fact_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_prepare_range(
      local ? &local->owner : pass->value_facts, work->module, scope,
      reserved_values, &table));
  table->context.resolve_call_results.user_data = work;
  table->context.resolve_call_results.fn =
      loom_low_source_invocation_resolve_call;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(table, work->module, function->source));
  IREE_RETURN_IF_ERROR(loom_conditioned_value_facts_compute(table, work->module,
                                                            function->source));
  // Guard-dependent facts have a whole-scope invalidation contract. Ordinary
  // equations can retain their structure and update only changed-result users.
  if (local && !table->has_conditioned_results) {
    loom_rewriter_attach_value_facts(&local->rewriter, table);
  }
  *out_local = table;
  return iree_ok_status();
}

iree_status_t loom_low_source_invocation_facts_compute(
    loom_pass_t* pass, loom_module_t* module,
    const loom_low_source_selection_list_t* selections,
    iree_arena_allocator_t* arena,
    loom_low_source_invocation_facts_t* out_facts) {
  memset(out_facts, 0, sizeof(*out_facts));
  if (!loom_module_has_semantic_calls(module)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_initialize(&out_facts->boundaries, arena, 0));
  loom_type_registry_configure_fact_context(&out_facts->boundaries.context);
  out_facts->function_count = module->symbols.count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*out_facts->functions),
      (void**)&out_facts->functions));
  memset(out_facts->functions, 0,
         module->symbols.count * sizeof(*out_facts->functions));
  uint8_t* direct_call_targets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*direct_call_targets),
                                                 (void**)&direct_call_targets));
  memset(direct_call_targets, 0,
         module->symbols.count * sizeof(*direct_call_targets));
  out_facts->direct_call_targets = direct_call_targets;
  out_facts->direct_call_target_count = module->symbols.count;

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(module->arena.block_pool, &analysis_arena);
  const loom_fact_context_t previous_context = pass->value_facts->table.context;
  loom_low_source_invocation_worklist_t work = {
      .module = module, .arena = &analysis_arena, .facts = out_facts};
  iree_status_t status = iree_arena_allocate_array(
      &analysis_arena, module->symbols.count, sizeof(*work.functions),
      (void**)&work.functions);
  if (iree_status_is_ok(status)) {
    memset(work.functions, 0, module->symbols.count * sizeof(*work.functions));
    loom_symbol_t* symbol = NULL;
    loom_module_for_each_symbol(module, symbol) {
      if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
          !symbol->defining_op) {
        continue;
      }
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_body(function)) {
        work.functions[symbol - module->symbols.entries].source = function;
      }
    }
    for (iree_host_size_t i = 0; i < selections->count; ++i) {
      const loom_low_source_selection_t* selection = &selections->values[i];
      if (selection->kind == LOOM_LOW_SOURCE_SELECTION_FUNCTION) {
        work.functions[loom_func_like_callee(selection->func).symbol_id]
            .selection = selection;
      }
    }
    status = loom_call_graph_build_with_calls(
        module, arena,
        (loom_call_graph_call_visitor_t){
            .user_data = &work, .fn = loom_low_source_invocation_collect_call},
        &out_facts->call_graph);
  }
  if (iree_status_is_ok(status)) {
    work.call_bucket_count = iree_math_round_up_to_pow2_u64(
        iree_max((iree_host_size_t)1, work.call_count));
    status = iree_arena_allocate_array(&analysis_arena, work.call_bucket_count,
                                       sizeof(*work.call_buckets),
                                       (void**)&work.call_buckets);
    if (iree_status_is_ok(status)) {
      memset(work.call_buckets, 0,
             work.call_bucket_count * sizeof(*work.call_buckets));
    }
  }
  for (uint16_t i = 0;
       i < out_facts->call_graph.node_count && iree_status_is_ok(status); ++i) {
    status = loom_low_source_invocation_initialize_function(
        &work, &out_facts->call_graph.nodes[i], arena);
  }
  while (work.first && iree_status_is_ok(status)) {
    loom_low_source_invocation_function_t* function = work.first;
    work.first = function->next_pending;
    if (!work.first) {
      work.last = NULL;
    }
    function->pending = false;
    bool changed = function->changed_results || !function->analyzed;
    function->changed_results = false;
    bool changed_arguments = false;
    status = loom_low_source_invocation_arguments(&work, function,
                                                  &changed_arguments);
    if (!iree_status_is_ok(status) || (!changed && !changed_arguments)) {
      continue;
    }
    loom_value_fact_table_t* local = NULL;
    status = loom_low_source_invocation_analyze(pass, &work, function,
                                                changed_arguments, &local);
    if (iree_status_is_ok(status)) {
      status = loom_low_source_invocation_publish_arguments(&work, function);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_low_source_invocation_publish_results(&work, function, local);
    }
    function->analyzed = true;
  }
  pass->value_facts->table.context.resolve_call_results =
      previous_context.resolve_call_results;
  loom_pass_value_fact_owner_invalidate(pass->value_facts);
  for (uint16_t i = 0; i < out_facts->call_graph.node_count; ++i) {
    loom_low_source_invocation_local_t* local =
        work.functions[out_facts->call_graph.nodes[i].symbol_id].local;
    if (local) {
      loom_rewriter_deinitialize(&local->rewriter);
      loom_pass_value_fact_owner_deinitialize(&local->owner);
    }
  }
  iree_arena_deinitialize(&analysis_arena);
  return status;
}

loom_value_fact_table_view_t loom_low_source_invocation_facts_for_function(
    const loom_low_source_invocation_facts_t* facts,
    loom_func_like_t function) {
  const loom_symbol_ref_t symbol = loom_func_like_callee(function);
  return symbol.module_id == 0 && symbol.symbol_id < facts->function_count
             ? facts->functions[symbol.symbol_id]
             : (loom_value_fact_table_view_t){0};
}

bool loom_low_source_invocation_facts_has_direct_callers(
    const loom_low_source_invocation_facts_t* facts,
    loom_func_like_t function) {
  if (facts->direct_call_targets == NULL || !loom_func_like_isa(function)) {
    return false;
  }
  const loom_symbol_ref_t callee = loom_func_like_callee(function);
  return loom_symbol_ref_is_valid(callee) && callee.module_id == 0 &&
         callee.symbol_id < facts->direct_call_target_count &&
         facts->direct_call_targets[callee.symbol_id] != 0;
}
