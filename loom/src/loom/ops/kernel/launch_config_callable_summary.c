// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/launch_config_callable_summary.h"

#include <stdlib.h>
#include <string.h>

#include "loom/analysis/value_relation.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config_dependence.h"
#include "loom/ops/op_defs.h"
#include "loom/util/reference_transfer.h"
#include "loom/util/walk.h"

enum loom_kernel_launch_config_callable_summary_state_e {
  LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_EMPTY = 0,
  LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_VISITING = 1,
  LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_COMPLETE = 2,
};

typedef struct loom_kernel_launch_config_reference_edge_t {
  // Reference-like value receiving storage provenance.
  loom_value_id_t destination;
  // Reference-like value supplying storage provenance, or INVALID when the
  // source lies outside the summarized function's formal arguments.
  loom_value_id_t source;
} loom_kernel_launch_config_reference_edge_t;

typedef struct loom_kernel_launch_config_read_source_t {
  // Storage value observed by a deterministic load.
  loom_value_id_t value;
  // Direct load or summarized call establishing the read.
  const loom_op_t* op;
} loom_kernel_launch_config_read_source_t;

typedef struct loom_kernel_launch_config_callable_build_t {
  // Retained cache receiving the completed summary.
  loom_kernel_launch_config_callable_cache_t* cache;
  // Summary currently being populated.
  loom_kernel_launch_config_callable_summary_t* summary;
  // Function-like helper definition being summarized.
  loom_func_like_t function;
  // Temporary storage released after the summary is complete.
  iree_arena_allocator_t* arena;
  // Complete local facts for the helper body.
  loom_value_fact_table_t facts;
  // Storage-dependence closure within the helper body.
  loom_kernel_launch_config_dependence_t dependence;
  // Reference transport edges retained during the one body walk.
  loom_kernel_launch_config_reference_edge_t* reference_edges;
  // Number of initialized entries in reference_edges.
  iree_host_size_t reference_edge_count;
  // Allocated entry capacity in reference_edges.
  iree_host_size_t reference_edge_capacity;
  // Direct and nested storage reads requiring formal-source resolution.
  loom_kernel_launch_config_read_source_t* read_sources;
  // Number of initialized entries in read_sources.
  iree_host_size_t read_source_count;
  // Allocated entry capacity in read_sources.
  iree_host_size_t read_source_capacity;
  // Callable body exits retained in source order.
  const loom_op_t** exits;
  // Number of initialized entries in exits.
  iree_host_size_t exit_count;
  // Allocated entry capacity in exits.
  iree_host_size_t exit_capacity;
  // Reusable module-value-indexed traversal membership bits.
  uint64_t* visited_bits;
  // Number of entries in visited_bits.
  iree_host_size_t visited_word_count;
  // Reusable reference-source traversal queue and touched-value list.
  loom_value_id_t* source_queue;
  // Number of initialized entries in source_queue.
  iree_host_size_t source_queue_count;
  // Allocated entry capacity in source_queue.
  iree_host_size_t source_queue_capacity;
} loom_kernel_launch_config_callable_build_t;

static void loom_kernel_launch_config_callable_fail(
    loom_kernel_launch_config_callable_summary_t* summary,
    loom_kernel_launch_config_contract_failure_t failure, const loom_op_t* op,
    loom_trait_flags_t traits) {
  if (summary->failure != LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
    return;
  }
  summary->failure = failure;
  summary->failure_op = op;
  summary->failure_traits = traits;
}

bool loom_kernel_launch_config_callable_reads_argument(
    const loom_kernel_launch_config_callable_summary_t* summary,
    uint16_t argument_index) {
  return argument_index < summary->argument_count &&
         (summary->read_argument_bits[argument_index / 64] &
          (UINT64_C(1) << (argument_index % 64))) != 0;
}

bool loom_kernel_launch_config_callable_result_is_dependent(
    const loom_kernel_launch_config_callable_summary_t* summary,
    uint16_t result_index) {
  return result_index < summary->result_count &&
         (summary->dependent_result_bits[result_index / 64] &
          (UINT64_C(1) << (result_index % 64))) != 0;
}

bool loom_kernel_launch_config_callable_result_references_argument(
    const loom_kernel_launch_config_callable_summary_t* summary,
    uint16_t result_index, uint16_t argument_index) {
  if (result_index >= summary->result_count ||
      argument_index >= summary->argument_count) {
    return false;
  }
  const uint64_t* row =
      summary->result_reference_argument_bits +
      (iree_host_size_t)result_index * summary->argument_word_count;
  return (row[argument_index / 64] & (UINT64_C(1) << (argument_index % 64))) !=
         0;
}

void loom_kernel_launch_config_callable_cache_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_kernel_launch_config_callable_cache_t* out_cache) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_cache);
  *out_cache = (loom_kernel_launch_config_callable_cache_t){
      .module = module,
      .arena = arena,
  };
}

static bool loom_kernel_launch_config_callable_resolve(
    const loom_kernel_launch_config_callable_cache_t* cache,
    const loom_op_t* call_op, loom_symbol_id_t* out_symbol_id,
    loom_func_like_t* out_function) {
  const loom_call_like_t call =
      loom_call_like_const_cast(cache->module, call_op);
  if (!loom_call_like_is_direct_semantic(call)) {
    return false;
  }
  const loom_symbol_ref_t callee = loom_call_like_callee(call);
  if (callee.module_id != 0 ||
      callee.symbol_id >= cache->module->symbols.count) {
    return false;
  }
  const loom_func_like_t function = loom_func_like_const_cast(
      cache->module,
      cache->module->symbols.entries[callee.symbol_id].defining_op);
  if (!loom_func_like_isa(function) || !loom_func_like_body(function)) {
    return false;
  }
  const loom_inline_policy_t call_policy = loom_call_like_inline_policy(call);
  const loom_inline_policy_t function_policy =
      loom_func_like_inline_policy(function);
  if (call_policy == LOOM_INLINE_POLICY_NOINLINE ||
      function_policy == LOOM_INLINE_POLICY_NOINLINE ||
      (call_policy != LOOM_INLINE_POLICY_INLINE &&
       function_policy != LOOM_INLINE_POLICY_INLINE)) {
    return false;
  }
  *out_symbol_id = callee.symbol_id;
  *out_function = function;
  return true;
}

static iree_status_t loom_kernel_launch_config_callable_append_reference_edge(
    loom_kernel_launch_config_callable_build_t* build,
    loom_value_id_t destination, loom_value_id_t source) {
  if (destination == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (build->reference_edge_count == build->reference_edge_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        build->arena, build->reference_edge_count,
        build->reference_edge_count + 1, sizeof(*build->reference_edges),
        &build->reference_edge_capacity, (void**)&build->reference_edges));
  }
  build->reference_edges[build->reference_edge_count++] =
      (loom_kernel_launch_config_reference_edge_t){
          .destination = destination,
          .source = source,
      };
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_append_read(
    loom_kernel_launch_config_callable_build_t* build, const loom_op_t* op,
    loom_value_id_t value) {
  if (build->read_source_count == build->read_source_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        build->arena, build->read_source_count, build->read_source_count + 1,
        sizeof(*build->read_sources), &build->read_source_capacity,
        (void**)&build->read_sources));
  }
  build->read_sources[build->read_source_count++] =
      (loom_kernel_launch_config_read_source_t){
          .value = value,
          .op = op,
      };
  build->summary->effect_contract = LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_append_exit(
    loom_kernel_launch_config_callable_build_t* build, const loom_op_t* op) {
  if (build->exit_count == build->exit_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        build->arena, build->exit_count, build->exit_count + 1,
        sizeof(*build->exits), &build->exit_capacity, (void**)&build->exits));
  }
  build->exits[build->exit_count++] = op;
  return iree_ok_status();
}

static iree_status_t
loom_kernel_launch_config_callable_append_operation_reference_edges(
    loom_kernel_launch_config_callable_build_t* build, const loom_op_t* op) {
  const loom_value_relation_mask_t relation_mask =
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_TIED_RESULT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_BYPASS) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT) |
      LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REFERENCE_SOURCE);
  loom_value_relation_iterator_t relations;
  loom_value_relation_iterator_initialize(build->cache->module, op,
                                          relation_mask, &relations);
  loom_value_relation_t relation;
  while (loom_value_relation_iterator_next(&relations, &relation)) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_config_callable_append_reference_edge(
            build, relation.destination_value_id, relation.source_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_append_call_summary(
    loom_kernel_launch_config_callable_build_t* build, const loom_op_t* op,
    const loom_kernel_launch_config_callable_summary_t* callee) {
  const loom_call_like_t call =
      loom_call_like_const_cast(build->cache->module, op);
  const loom_value_slice_t operands = loom_call_like_operands(call);
  const loom_value_slice_t results = loom_call_like_results(call);
  const uint16_t argument_count =
      iree_min((iree_host_size_t)callee->argument_count, operands.count);
  const uint16_t result_count =
      iree_min((iree_host_size_t)callee->result_count, results.count);
  for (uint16_t argument_index = 0; argument_index < argument_count;
       ++argument_index) {
    if (loom_kernel_launch_config_callable_reads_argument(callee,
                                                          argument_index)) {
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_append_read(
          build, op, operands.values[argument_index]));
    }
  }
  for (uint16_t result_index = 0; result_index < result_count; ++result_index) {
    if (loom_kernel_launch_config_callable_result_is_dependent(callee,
                                                               result_index)) {
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_mark(
          &build->dependence, results.values[result_index]));
    }
    for (uint16_t argument_index = 0; argument_index < argument_count;
         ++argument_index) {
      if (loom_kernel_launch_config_callable_result_references_argument(
              callee, result_index, argument_index)) {
        IREE_RETURN_IF_ERROR(
            loom_kernel_launch_config_callable_append_reference_edge(
                build, results.values[result_index],
                operands.values[argument_index]));
      }
    }
    if ((callee->nonargument_result_bits[result_index / 64] &
         (UINT64_C(1) << (result_index % 64))) != 0) {
      IREE_RETURN_IF_ERROR(
          loom_kernel_launch_config_callable_append_reference_edge(
              build, results.values[result_index], LOOM_VALUE_ID_INVALID));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_classify_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_kernel_launch_config_callable_build_t* build = user_data;
  if (loom_func_like_op_is_body_exit(build->cache->module, build->function,
                                     op)) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_config_callable_append_exit(build, op));
  }
  IREE_RETURN_IF_ERROR(
      loom_kernel_launch_config_callable_append_operation_reference_edges(build,
                                                                          op));

  const loom_trait_flags_t traits =
      loom_op_effective_traits(build->cache->module, op);
  const loom_trait_flags_t forbidden_traits =
      LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_NON_DETERMINISTIC |
      LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_UNIQUE_IDENTITY |
      LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_MEMORY_FENCE |
      LOOM_TRAIT_OBSERVABLE_EFFECT;
  if (iree_any_bit_set(traits, forbidden_traits)) {
    loom_kernel_launch_config_callable_fail(
        build->summary, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_EFFECT, op,
        traits);
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }

  const loom_kernel_launch_config_callable_summary_t* callee = NULL;
  if (loom_func_call_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_cache_lookup(
        build->cache, op, &callee));
  }
  if (callee) {
    if (callee->failure != LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
      loom_kernel_launch_config_callable_fail(
          build->summary, callee->failure,
          callee->failure_op ? callee->failure_op : op, callee->failure_traits);
      *out_result = LOOM_WALK_ABORT;
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_append_call_summary(
        build, op, callee));
    return iree_ok_status();
  }

  if (!iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY)) {
    return iree_ok_status();
  }
  const loom_memory_access_t access =
      loom_memory_access_cast(build->cache->module, op);
  if (!loom_memory_access_isa(access) ||
      loom_memory_access_operation_kind(access) !=
          LOOM_MEMORY_ACCESS_OPERATION_LOAD) {
    loom_kernel_launch_config_callable_fail(
        build->summary,
        LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_OPERATION, op, 0);
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_append_read(
      build, op, loom_memory_access_view(access)));
  return loom_kernel_launch_config_dependence_mark_results(&build->dependence,
                                                           op);
}

static int loom_kernel_launch_config_reference_edge_compare(const void* lhs,
                                                            const void* rhs) {
  const loom_kernel_launch_config_reference_edge_t* lhs_edge = lhs;
  const loom_kernel_launch_config_reference_edge_t* rhs_edge = rhs;
  if (lhs_edge->destination != rhs_edge->destination) {
    return lhs_edge->destination < rhs_edge->destination ? -1 : 1;
  }
  if (lhs_edge->source != rhs_edge->source) {
    return lhs_edge->source < rhs_edge->source ? -1 : 1;
  }
  return 0;
}

static iree_host_size_t loom_kernel_launch_config_callable_edge_lower_bound(
    const loom_kernel_launch_config_callable_build_t* build,
    loom_value_id_t destination) {
  iree_host_size_t lower = 0;
  iree_host_size_t upper = build->reference_edge_count;
  while (lower < upper) {
    const iree_host_size_t middle = lower + (upper - lower) / 2;
    if (build->reference_edges[middle].destination < destination) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return lower;
}

static iree_status_t loom_kernel_launch_config_callable_enqueue_source(
    loom_kernel_launch_config_callable_build_t* build,
    loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= build->cache->module->values.count ||
      (build->visited_bits[value_id / 64] & (UINT64_C(1) << (value_id % 64))) !=
          0) {
    return iree_ok_status();
  }
  build->visited_bits[value_id / 64] |= UINT64_C(1) << (value_id % 64);
  if (build->source_queue_count == build->source_queue_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        build->arena, build->source_queue_count, build->source_queue_count + 1,
        sizeof(*build->source_queue), &build->source_queue_capacity,
        (void**)&build->source_queue));
  }
  build->source_queue[build->source_queue_count++] = value_id;
  return iree_ok_status();
}

static bool loom_kernel_launch_config_callable_formal_ordinal(
    const loom_value_id_t* arguments, uint16_t argument_count,
    loom_value_id_t value_id, uint16_t* out_ordinal) {
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (arguments[i] == value_id) {
      *out_ordinal = i;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_kernel_launch_config_callable_trace_sources(
    loom_kernel_launch_config_callable_build_t* build, loom_value_id_t source,
    uint64_t* inout_argument_bits, bool* out_found_formal,
    bool* out_found_nonformal) {
  *out_found_formal = false;
  *out_found_nonformal = false;
  IREE_RETURN_IF_ERROR(
      loom_kernel_launch_config_callable_enqueue_source(build, source));
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(build->function, &argument_count);
  for (iree_host_size_t cursor = 0; cursor < build->source_queue_count;
       ++cursor) {
    const loom_value_id_t value_id = build->source_queue[cursor];
    uint16_t argument_index = 0;
    if (loom_kernel_launch_config_callable_formal_ordinal(
            arguments, argument_count, value_id, &argument_index)) {
      inout_argument_bits[argument_index / 64] |= UINT64_C(1)
                                                  << (argument_index % 64);
      *out_found_formal = true;
      continue;
    }
    const iree_host_size_t first_edge =
        loom_kernel_launch_config_callable_edge_lower_bound(build, value_id);
    if (first_edge == build->reference_edge_count ||
        build->reference_edges[first_edge].destination != value_id) {
      *out_found_nonformal = true;
      continue;
    }
    for (iree_host_size_t edge_index = first_edge;
         edge_index < build->reference_edge_count &&
         build->reference_edges[edge_index].destination == value_id;
         ++edge_index) {
      if (build->reference_edges[edge_index].source == LOOM_VALUE_ID_INVALID) {
        *out_found_nonformal = true;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_enqueue_source(
          build, build->reference_edges[edge_index].source));
    }
  }
  for (iree_host_size_t i = 0; i < build->source_queue_count; ++i) {
    const loom_value_id_t value_id = build->source_queue[i];
    build->visited_bits[value_id / 64] &= ~(UINT64_C(1) << (value_id % 64));
  }
  build->source_queue_count = 0;
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_allocate_summary_bits(
    loom_kernel_launch_config_callable_build_t* build) {
  loom_kernel_launch_config_callable_summary_t* summary = build->summary;
  summary->argument_word_count = (summary->argument_count + 63) / 64;
  summary->result_word_count = (summary->result_count + 63) / 64;
  if (summary->argument_word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->cache->arena, summary->argument_word_count,
        sizeof(*summary->read_argument_bits),
        (void**)&summary->read_argument_bits));
    memset(summary->read_argument_bits, 0,
           summary->argument_word_count * sizeof(*summary->read_argument_bits));
  }
  if (summary->result_word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->cache->arena, summary->result_word_count,
        sizeof(*summary->dependent_result_bits),
        (void**)&summary->dependent_result_bits));
    memset(
        summary->dependent_result_bits, 0,
        summary->result_word_count * sizeof(*summary->dependent_result_bits));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->cache->arena, summary->result_word_count,
        sizeof(*summary->nonargument_result_bits),
        (void**)&summary->nonargument_result_bits));
    memset(
        summary->nonargument_result_bits, 0,
        summary->result_word_count * sizeof(*summary->nonargument_result_bits));
  }
  const iree_host_size_t reference_word_count =
      (iree_host_size_t)summary->result_count * summary->argument_word_count;
  if (reference_word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->cache->arena, reference_word_count,
        sizeof(*summary->result_reference_argument_bits),
        (void**)&summary->result_reference_argument_bits));
    memset(summary->result_reference_argument_bits, 0,
           reference_word_count *
               sizeof(*summary->result_reference_argument_bits));
  }
  if (summary->result_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->cache->arena, summary->result_count,
        sizeof(*summary->result_facts), (void**)&summary->result_facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_summarize_results(
    loom_kernel_launch_config_callable_build_t* build) {
  loom_kernel_launch_config_callable_summary_t* summary = build->summary;
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(build->function, &argument_count);
  summary->argument_count = argument_count;
  if (build->exit_count == 0) {
    loom_kernel_launch_config_callable_fail(
        summary, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_OPERATION,
        build->function.op, 0);
    return iree_ok_status();
  }
  summary->result_count = build->exits[0]->operand_count;
  for (iree_host_size_t exit_index = 1; exit_index < build->exit_count;
       ++exit_index) {
    if (build->exits[exit_index]->operand_count != summary->result_count) {
      loom_kernel_launch_config_callable_fail(
          summary, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_OPERATION,
          build->exits[exit_index], 0);
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_kernel_launch_config_callable_allocate_summary_bits(build));

  for (iree_host_size_t read_index = 0; read_index < build->read_source_count;
       ++read_index) {
    bool found_formal = false;
    bool found_nonformal = false;
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_trace_sources(
        build, build->read_sources[read_index].value,
        summary->read_argument_bits, &found_formal, &found_nonformal));
    if (!found_formal || found_nonformal) {
      loom_kernel_launch_config_callable_fail(
          summary, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_ORIGIN,
          build->read_sources[read_index].op, 0);
      return iree_ok_status();
    }
  }

  for (uint16_t result_index = 0; result_index < summary->result_count;
       ++result_index) {
    const loom_value_id_t first_value =
        loom_op_const_operands(build->exits[0])[result_index];
    const loom_type_t result_type =
        loom_module_value_type(build->cache->module, first_value);
    loom_value_facts_t joined_facts =
        loom_value_fact_table_lookup(&build->facts, first_value);
    bool dependent = loom_kernel_launch_config_dependence_contains(
        &build->dependence, first_value);
    for (iree_host_size_t exit_index = 1; exit_index < build->exit_count;
         ++exit_index) {
      const loom_value_id_t value =
          loom_op_const_operands(build->exits[exit_index])[result_index];
      dependent |= loom_kernel_launch_config_dependence_contains(
          &build->dependence, value);
      loom_value_facts_t next_facts = loom_value_facts_unknown();
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          &build->facts, build->cache->module, result_type, &build->facts,
          joined_facts, &build->facts,
          loom_value_fact_table_lookup(&build->facts, value), &next_facts));
      joined_facts = next_facts;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        build->cache->summary_facts, &build->facts, build->cache->module,
        result_type, joined_facts, &summary->result_facts[result_index]));
    if (dependent) {
      summary->dependent_result_bits[result_index / 64] |=
          UINT64_C(1) << (result_index % 64);
    }
    if (!loom_type_is_buffer(result_type) && !loom_type_is_view(result_type)) {
      continue;
    }
    uint64_t* source_bits =
        summary->result_reference_argument_bits +
        (iree_host_size_t)result_index * summary->argument_word_count;
    for (iree_host_size_t exit_index = 0; exit_index < build->exit_count;
         ++exit_index) {
      bool found_formal = false;
      bool found_nonformal = false;
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_trace_sources(
          build, loom_op_const_operands(build->exits[exit_index])[result_index],
          source_bits, &found_formal, &found_nonformal));
      if (found_nonformal) {
        summary->nonargument_result_bits[result_index / 64] |=
            UINT64_C(1) << (result_index % 64);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_callable_build_with_arena(
    loom_kernel_launch_config_callable_cache_t* cache,
    loom_func_like_t function,
    loom_kernel_launch_config_callable_summary_t* summary,
    iree_arena_allocator_t* arena) {
  loom_kernel_launch_config_callable_build_t build = {
      .cache = cache,
      .summary = summary,
      .function = function,
      .arena = arena,
      .visited_word_count = (cache->module->values.count + 63) / 64,
  };
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
      &build.facts, arena, /*initial_capacity=*/0));
  build.facts.context.resolve_call_results.user_data = cache;
  build.facts.context.resolve_call_results.fn =
      loom_kernel_launch_config_callable_resolve_results;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(&build.facts, cache->module, function));
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_initialize(
      cache->module, arena, &build.dependence));
  if (build.visited_word_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, build.visited_word_count, sizeof(*build.visited_bits),
        (void**)&build.visited_bits));
    memset(build.visited_bits, 0,
           build.visited_word_count * sizeof(*build.visited_bits));
  }

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      cache->module, loom_func_like_body(function), LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){
          .fn = loom_kernel_launch_config_callable_classify_op,
          .user_data = &build,
      },
      &walk_result));
  if (summary->failure != LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_kernel_launch_config_dependence_propagate(&build.dependence));
  if (build.reference_edge_count > 1) {
    qsort(build.reference_edges, build.reference_edge_count,
          sizeof(*build.reference_edges),
          loom_kernel_launch_config_reference_edge_compare);
  }
  return loom_kernel_launch_config_callable_summarize_results(&build);
}

static iree_status_t loom_kernel_launch_config_callable_build(
    loom_kernel_launch_config_callable_cache_t* cache,
    loom_func_like_t function,
    loom_kernel_launch_config_callable_summary_t* summary) {
  summary->state = LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_VISITING;
  summary->effect_contract = LOOM_CALLABLE_EFFECT_CONTRACT_PURE;
  iree_arena_allocator_t arena;
  iree_arena_initialize(cache->arena->block_pool, &arena);
  iree_status_t status = loom_kernel_launch_config_callable_build_with_arena(
      cache, function, summary, &arena);
  iree_arena_deinitialize(&arena);
  if (!iree_status_is_ok(status)) {
    summary->state = LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_EMPTY;
    return status;
  }
  summary->state = LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_COMPLETE;
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_callable_cache_lookup(
    loom_kernel_launch_config_callable_cache_t* cache, const loom_op_t* call_op,
    const loom_kernel_launch_config_callable_summary_t** out_summary) {
  *out_summary = NULL;
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  loom_func_like_t function = {0};
  if (!loom_kernel_launch_config_callable_resolve(cache, call_op, &symbol_id,
                                                  &function)) {
    return iree_ok_status();
  }
  if (!cache->summaries) {
    loom_kernel_launch_config_callable_summary_t** summaries = NULL;
    loom_value_fact_table_t* summary_facts = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(cache->arena, cache->module->symbols.count,
                                  sizeof(*summaries), (void**)&summaries));
    memset(summaries, 0, cache->module->symbols.count * sizeof(*summaries));
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        cache->arena, sizeof(*summary_facts), (void**)&summary_facts));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
        summary_facts, cache->arena, /*initial_capacity=*/0));
    cache->summaries = summaries;
    cache->summary_facts = summary_facts;
  }
  loom_kernel_launch_config_callable_summary_t* summary =
      cache->summaries[symbol_id];
  if (!summary) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(cache->arena, sizeof(*summary), (void**)&summary));
    *summary = (loom_kernel_launch_config_callable_summary_t){0};
    cache->summaries[symbol_id] = summary;
  }
  if (summary->state ==
      LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_VISITING) {
    loom_kernel_launch_config_callable_fail(
        summary, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_OPERATION,
        call_op, 0);
  } else if (summary->state ==
             LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_EMPTY) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_config_callable_build(cache, function, summary));
  }
  *out_summary = summary;
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_config_callable_resolve_results(
    void* user_data, loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_kernel_launch_config_callable_cache_t* cache = user_data;
  const loom_kernel_launch_config_callable_summary_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      loom_kernel_launch_config_callable_cache_lookup(cache, op, &summary));
  if (!summary ||
      summary->state !=
          LOOM_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_STATE_COMPLETE ||
      summary->failure != LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
    return iree_ok_status();
  }
  const loom_call_like_t call = loom_call_like_const_cast(module, op);
  const loom_value_slice_t operands = loom_call_like_operands(call);
  const loom_value_slice_t results = loom_call_like_results(call);
  const uint16_t result_count =
      iree_min((iree_host_size_t)summary->result_count, results.count);
  loom_reference_call_t reference_call;
  loom_reference_call_initialize(module, context->table, operands,
                                 context->reference_origin, &reference_call);
  for (uint16_t result_index = 0; result_index < result_count; ++result_index) {
    const loom_type_t result_type =
        loom_module_value_type(module, results.values[result_index]);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
        context->table, cache->summary_facts, module, result_type,
        summary->result_facts[result_index], &result_facts[result_index]));
    if (!loom_type_is_buffer(result_type) && !loom_type_is_view(result_type)) {
      continue;
    }
    bool has_source = false;
    loom_value_fact_reference_origin_t origin = {0};
    const uint16_t argument_count =
        iree_min((iree_host_size_t)summary->argument_count, operands.count);
    for (uint16_t argument_index = 0; argument_index < argument_count;
         ++argument_index) {
      if (!loom_kernel_launch_config_callable_result_references_argument(
              summary, result_index, argument_index)) {
        continue;
      }
      const loom_value_fact_reference_origin_t argument_origin =
          loom_value_facts_reference_origin(context,
                                            operand_facts[argument_index]);
      origin = has_source ? loom_value_fact_reference_origin_meet(
                                origin, argument_origin)
                          : argument_origin;
      has_source = true;
    }
    const bool has_nonargument_source =
        (summary->nonargument_result_bits[result_index / 64] &
         (UINT64_C(1) << (result_index % 64))) != 0;
    if (has_source && has_nonargument_source) {
      origin = (loom_value_fact_reference_origin_t){0};
    } else if (!has_source) {
      origin = loom_reference_call_result_origin(
          &reference_call, loom_value_facts_reference_origin(
                               context, result_facts[result_index]));
    }
    IREE_RETURN_IF_ERROR(loom_value_facts_rebind_reference_origin(
        context, origin, &result_facts[result_index]));
  }
  return iree_ok_status();
}
