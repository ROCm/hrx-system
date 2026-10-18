// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/kernel/launch_config_contract.h"

#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/launch_config_callable_summary.h"
#include "loom/ops/kernel/launch_config_dependence.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_table.h"
#include "loom/util/reference_facts.h"
#include "loom/util/walk.h"

typedef struct loom_kernel_launch_config_contract_read_t {
  // Direct load or summarized helper call establishing the read.
  const loom_op_t* op;
  // Storage value whose origin must be a projected workload argument.
  loom_value_id_t storage;
} loom_kernel_launch_config_contract_read_t;

typedef struct loom_kernel_launch_config_contract_analysis_t {
  // Module containing the kernel definition.
  const loom_module_t* module;
  // Kernel definition whose configuration is being analyzed.
  const loom_op_t* kernel_op;
  // Configuration region projected from the kernel workload signature.
  loom_region_t* config_region;
  // Function-like view of kernel_op.
  loom_func_like_t function;
  // Root-region ordinal of config_region in function.
  uint8_t config_region_index;
  // Caller-owned transient storage.
  iree_arena_allocator_t* arena;
  // Facts computed once for config_region and its nested regions.
  loom_value_fact_table_t facts;
  // Required-inline callable summaries shared by classification and facts.
  loom_kernel_launch_config_callable_cache_t callables;
  // Reads retained in source order during effect classification.
  loom_kernel_launch_config_contract_read_t* reads;
  // Number of entries in reads.
  iree_host_size_t read_count;
  // Allocated entry capacity in reads.
  iree_host_size_t read_capacity;
  // Storage-derived value closure shared with callable summaries.
  loom_kernel_launch_config_dependence_t dependence;
  // Analysis result populated at the first semantic failure.
  loom_kernel_launch_config_contract_t* contract;
} loom_kernel_launch_config_contract_analysis_t;

static void loom_kernel_launch_config_contract_fail(
    loom_kernel_launch_config_contract_analysis_t* analysis,
    loom_kernel_launch_config_contract_failure_t failure, const loom_op_t* op) {
  if (analysis->contract->failure !=
      LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
    return;
  }
  analysis->contract->failure = failure;
  analysis->contract->failure_op = op;
}

static iree_status_t loom_kernel_launch_config_contract_record_read(
    loom_kernel_launch_config_contract_analysis_t* analysis,
    const loom_op_t* op, loom_value_id_t storage) {
  if (analysis->read_count == analysis->read_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        analysis->arena, analysis->read_count, analysis->read_count + 1,
        sizeof(*analysis->reads), &analysis->read_capacity,
        (void**)&analysis->reads));
  }
  analysis->reads[analysis->read_count++] =
      (loom_kernel_launch_config_contract_read_t){
          .op = op,
          .storage = storage,
      };
  return iree_ok_status();
}

static iree_status_t loom_kernel_launch_config_contract_classify_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_kernel_launch_config_contract_analysis_t* analysis = user_data;
  const loom_trait_flags_t traits =
      loom_op_effective_traits(analysis->module, op);
  const loom_trait_flags_t forbidden_traits =
      LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_NON_DETERMINISTIC |
      LOOM_TRAIT_UNKNOWN_EFFECTS | LOOM_TRAIT_UNIQUE_IDENTITY |
      LOOM_TRAIT_CONVERGENT | LOOM_TRAIT_MEMORY_FENCE |
      LOOM_TRAIT_OBSERVABLE_EFFECT;
  if (iree_any_bit_set(traits, forbidden_traits)) {
    loom_kernel_launch_config_contract_fail(
        analysis, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_EFFECT, op);
    analysis->contract->failure_traits = traits;
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }

  const loom_kernel_launch_config_callable_summary_t* summary = NULL;
  if (loom_func_call_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_kernel_launch_config_callable_cache_lookup(
        &analysis->callables, op, &summary));
  }
  if (summary) {
    if (summary->failure != LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
      loom_kernel_launch_config_contract_fail(
          analysis, summary->failure,
          summary->failure_op ? summary->failure_op : op);
      analysis->contract->failure_traits = summary->failure_traits;
      *out_result = LOOM_WALK_ABORT;
      return iree_ok_status();
    }
    const loom_call_like_t call =
        loom_call_like_const_cast(analysis->module, op);
    const loom_value_slice_t operands = loom_call_like_operands(call);
    const loom_value_slice_t results = loom_call_like_results(call);
    const uint16_t argument_count = (uint16_t)iree_min(
        (iree_host_size_t)summary->argument_count, operands.count);
    const uint16_t result_count = (uint16_t)iree_min(
        (iree_host_size_t)summary->result_count, results.count);
    for (uint16_t i = 0; i < argument_count; ++i) {
      if (loom_kernel_launch_config_callable_reads_argument(summary, i)) {
        IREE_RETURN_IF_ERROR(loom_kernel_launch_config_contract_record_read(
            analysis, op, operands.values[i]));
      }
    }
    for (uint16_t i = 0; i < result_count; ++i) {
      if (loom_kernel_launch_config_callable_result_is_dependent(summary, i)) {
        IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_mark(
            &analysis->dependence, results.values[i]));
      }
    }
    if (summary->effect_contract == LOOM_CALLABLE_EFFECT_CONTRACT_READONLY) {
      analysis->contract->effect_contract =
          LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
    }
    return iree_ok_status();
  }
  if (!iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY)) {
    return iree_ok_status();
  }

  const loom_memory_access_t access =
      loom_memory_access_cast(analysis->module, op);
  if (!loom_memory_access_isa(access) ||
      loom_memory_access_operation_kind(access) !=
          LOOM_MEMORY_ACCESS_OPERATION_LOAD) {
    loom_kernel_launch_config_contract_fail(
        analysis, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_OPERATION,
        op);
    *out_result = LOOM_WALK_ABORT;
    return iree_ok_status();
  }
  analysis->contract->effect_contract = LOOM_CALLABLE_EFFECT_CONTRACT_READONLY;
  return loom_kernel_launch_config_contract_record_read(
      analysis, op, loom_memory_access_view(access));
}

static bool loom_kernel_launch_config_contract_read_origin_is_workload(
    const loom_kernel_launch_config_contract_analysis_t* analysis,
    loom_value_id_t storage) {
  if (storage == LOOM_VALUE_ID_INVALID ||
      storage >= analysis->module->values.count) {
    return false;
  }
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(&analysis->facts, storage);
  loom_value_fact_reference_origin_t origin = {0};
  loom_value_fact_buffer_reference_t buffer_reference = {0};
  loom_value_fact_view_reference_t view_reference = {0};
  if (loom_value_facts_query_buffer_reference(&analysis->facts.context, facts,
                                              &buffer_reference)) {
    origin = buffer_reference.origin;
  } else if (loom_value_facts_query_view_reference(&analysis->facts.context,
                                                   facts, &view_reference)) {
    origin = view_reference.origin;
  } else {
    return false;
  }
  return origin.kind == LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY &&
         origin.function_symbol_id ==
             loom_func_like_callee(analysis->function).symbol_id &&
         origin.region_index == analysis->config_region_index;
}

static void loom_kernel_launch_config_contract_check_sizes(
    loom_kernel_launch_config_contract_analysis_t* analysis) {
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(analysis->kernel_op);
  for (uint8_t dimension = 0; dimension < 3; ++dimension) {
    const loom_kernel_dimension_t kernel_dimension =
        (loom_kernel_dimension_t)dimension;
    const loom_value_id_t workgroup_size =
        loom_kernel_launch_config_workgroup_size_operand(launch_config,
                                                         kernel_dimension);
    if (loom_kernel_launch_config_dependence_contains(&analysis->dependence,
                                                      workgroup_size)) {
      loom_kernel_launch_config_contract_fail(
          analysis, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_WORKGROUP_SIZE,
          launch_config);
      analysis->contract->failure_dimension = dimension;
      return;
    }
    const loom_value_id_t cluster_size =
        loom_kernel_launch_config_workgroup_cluster_size_operand(
            launch_config, kernel_dimension);
    if (loom_kernel_launch_config_dependence_contains(&analysis->dependence,
                                                      cluster_size)) {
      loom_kernel_launch_config_contract_fail(
          analysis, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_CLUSTER_SIZE,
          launch_config);
      analysis->contract->failure_dimension = dimension;
      return;
    }
  }
}

static void loom_kernel_launch_config_contract_check_count_ranges(
    loom_kernel_launch_config_contract_analysis_t* analysis) {
  const loom_op_t* launch_config =
      loom_kernel_def_launch_config_op(analysis->kernel_op);
  for (uint8_t dimension = 0; dimension < 3; ++dimension) {
    const loom_value_id_t count =
        loom_kernel_launch_config_workgroup_count_operand(
            launch_config, (loom_kernel_dimension_t)dimension);
    if (!loom_kernel_launch_config_dependence_contains(&analysis->dependence,
                                                       count)) {
      continue;
    }
    const loom_value_facts_t facts =
        loom_value_fact_table_lookup(&analysis->facts, count);
    if (!loom_value_facts_is_float(facts) && facts.range_lo >= 0 &&
        facts.range_hi <= UINT32_MAX) {
      continue;
    }
    loom_kernel_launch_config_contract_fail(
        analysis, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_WORKGROUP_COUNT,
        launch_config);
    analysis->contract->failure_dimension = dimension;
    analysis->contract->failure_range_lower = facts.range_lo;
    analysis->contract->failure_range_upper = facts.range_hi;
    return;
  }
}

iree_status_t loom_kernel_launch_config_contract_analyze(
    const loom_module_t* module, const loom_op_t* kernel_op,
    iree_arena_allocator_t* scratch_arena,
    loom_kernel_launch_config_contract_t* out_contract) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(kernel_op);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_contract);
  IREE_ASSERT(loom_kernel_def_isa(kernel_op));
  *out_contract = (loom_kernel_launch_config_contract_t){
      .effect_contract = LOOM_CALLABLE_EFFECT_CONTRACT_PURE,
  };

  loom_kernel_launch_config_contract_analysis_t analysis = {
      .module = module,
      .kernel_op = kernel_op,
      .config_region = loom_kernel_def_config(kernel_op),
      .function = loom_func_like_const_cast(module, kernel_op),
      .config_region_index = LOOM_REGION_INDEX_NONE,
      .arena = scratch_arena,
      .contract = out_contract,
  };
  for (uint8_t i = 0; i < loom_func_like_region_count(analysis.function); ++i) {
    if (loom_func_like_region(analysis.function, i) == analysis.config_region) {
      analysis.config_region_index = i;
      break;
    }
  }
  IREE_ASSERT_NE(analysis.config_region_index, LOOM_REGION_INDEX_NONE);

  loom_kernel_launch_config_callable_cache_initialize(module, scratch_arena,
                                                      &analysis.callables);
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_initialize(
      module, scratch_arena, &analysis.dependence));

  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(
      loom_walk_region(module, analysis.config_region, LOOM_WALK_PRE_ORDER,
                       (loom_walk_callback_t){
                           .fn = loom_kernel_launch_config_contract_classify_op,
                           .user_data = &analysis,
                       },
                       &walk_result));
  if (out_contract->failure !=
      LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
      &analysis.facts, scratch_arena, /*initial_capacity=*/0));
  analysis.facts.context.resolve_call_results.user_data = &analysis.callables;
  analysis.facts.context.resolve_call_results.fn =
      loom_kernel_launch_config_callable_resolve_results;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region(
      &analysis.facts, module, analysis.function, analysis.config_region,
      (loom_op_t*)kernel_op));
  for (iree_host_size_t i = 0; i < analysis.read_count; ++i) {
    const loom_kernel_launch_config_contract_read_t read = analysis.reads[i];
    if (!loom_kernel_launch_config_contract_read_origin_is_workload(
            &analysis, read.storage)) {
      loom_kernel_launch_config_contract_fail(
          &analysis, LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_READ_ORIGIN,
          read.op);
      return iree_ok_status();
    }
    const loom_memory_access_t access =
        loom_memory_access_cast(module, read.op);
    if (loom_memory_access_isa(access)) {
      IREE_RETURN_IF_ERROR(loom_kernel_launch_config_dependence_mark_results(
          &analysis.dependence, read.op));
    }
  }

  if (!loom_kernel_launch_config_dependence_is_empty(&analysis.dependence)) {
    IREE_RETURN_IF_ERROR(
        loom_kernel_launch_config_dependence_propagate(&analysis.dependence));
    loom_kernel_launch_config_contract_check_sizes(&analysis);
    if (out_contract->failure ==
        LOOM_KERNEL_LAUNCH_CONFIG_CONTRACT_FAILURE_NONE) {
      loom_kernel_launch_config_contract_check_count_ranges(&analysis);
    }
  }
  return iree_ok_status();
}
