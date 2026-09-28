// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/plan.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static const loom_symbol_t* loom_testbench_scenario_symbol_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref) {
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  return &module->symbols.entries[ref.symbol_id];
}

static iree_string_view_t loom_testbench_scenario_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol == NULL || symbol->name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, symbol->name_id);
}

static void loom_testbench_count_comparison_plan(
    const loom_op_t* compare_op,
    loom_testbench_scenario_plan_counts_t* counts) {
  const loom_region_t* comparison = loom_check_compare_comparison(compare_op);
  if (comparison == NULL || comparison->block_count == 0) {
    return;
  }
  const loom_block_t* block = loom_region_const_entry_block(comparison);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    ++counts->issue_capacity;
    if (loom_testbench_is_expectation_op(op)) {
      ++counts->expectation_count;
    }
  }
}

static void loom_testbench_count_trial_plan(
    const loom_op_t* trial_op, loom_testbench_scenario_plan_counts_t* counts) {
  ++counts->trial_count;
  ++counts->issue_capacity;
  const loom_region_t* body = loom_check_trial_body(trial_op);
  if (body == NULL || body->block_count == 0) {
    return;
  }
  const loom_block_t* block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    ++counts->issue_capacity;
    if (loom_testbench_is_value_source_op(op)) {
      ++counts->value_source_count;
    } else if (loom_check_compare_isa(op)) {
      loom_testbench_count_comparison_plan(op, counts);
    }
  }
}

void loom_testbench_count_scenario_plans(
    const loom_module_t* module,
    loom_testbench_scenario_plan_counts_t* out_counts) {
  memset(out_counts, 0, sizeof(*out_counts));
  if (module->body == NULL || module->body->block_count == 0) {
    return;
  }
  const loom_block_t* module_block =
      loom_region_const_entry_block(module->body);
  const loom_op_t* scenario_op = NULL;
  loom_block_for_each_op(module_block, scenario_op) {
    if (!loom_check_scenario_isa(scenario_op)) {
      continue;
    }
    ++out_counts->scenario_count;
    ++out_counts->issue_capacity;
    const loom_region_t* body = loom_check_scenario_body(scenario_op);
    if (body == NULL || body->block_count == 0) {
      continue;
    }
    const loom_block_t* block = loom_region_const_entry_block(body);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      ++out_counts->issue_capacity;
      if (loom_check_trial_isa(op)) {
        loom_testbench_count_trial_plan(op, out_counts);
      } else if (loom_testbench_is_value_source_op(op)) {
        ++out_counts->value_source_count;
      }
    }
  }
}

static iree_status_t loom_testbench_scenario_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, element_size, out_ptr));
  memset(*out_ptr, 0, count * element_size);
  return iree_ok_status();
}

static void loom_testbench_append_scenario_issue(
    loom_testbench_issue_t* issues, iree_host_size_t issue_capacity,
    iree_host_size_t* inout_issue_count, loom_testbench_issue_kind_t kind,
    iree_host_size_t scenario_index, const loom_op_t* op,
    loom_symbol_ref_t scenario_ref) {
  IREE_ASSERT(*inout_issue_count < issue_capacity);
  issues[(*inout_issue_count)++] = (loom_testbench_issue_t){
      .kind = kind,
      .case_index = LOOM_TESTBENCH_CASE_INDEX_INVALID,
      .scenario_index = scenario_index,
      .benchmark_index = LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID,
      .op = op,
      .record_ref = scenario_ref,
  };
}

static bool loom_testbench_scenario_configuration_source_isa(
    const loom_op_t* op) {
  // Configuration scope owns immutable scalar values and entropy identities.
  // Mutable shaped storage remains trial-local so target and oracle
  // realizations can never share it.
  return loom_check_literal_isa(op) || loom_check_entropy_fork_isa(op) ||
         loom_check_entropy_read_isa(op);
}

static loom_testbench_invocation_kind_t
loom_testbench_scenario_subject_invocation_kind(const loom_symbol_t* symbol) {
  if (symbol == NULL) {
    return LOOM_TESTBENCH_INVOCATION_NONE;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CALLABLE)) {
    return LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL)) {
    return LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM)) {
    return LOOM_TESTBENCH_INVOCATION_COMMAND_PROGRAM;
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_PIPELINE)) {
    return LOOM_TESTBENCH_INVOCATION_PIPELINE;
  }
  return LOOM_TESTBENCH_INVOCATION_NONE;
}

static bool loom_testbench_plan_scenario_invocation(
    const loom_module_t* module, const loom_op_t* action_op,
    loom_symbol_ref_t callee, const loom_value_id_t* result_value_ids,
    iree_host_size_t result_count,
    loom_testbench_invocation_plan_t* out_invocation) {
  const bool is_compare = loom_check_compare_isa(action_op);
  const loom_value_slice_t call_parameters =
      is_compare ? loom_check_compare_call_parameters(action_op)
                 : loom_check_invoke_call_parameters(action_op);
  const loom_value_slice_t arguments =
      is_compare ? loom_check_compare_arguments(action_op)
                 : loom_check_invoke_arguments(action_op);
  const loom_symbol_t* symbol =
      loom_testbench_scenario_symbol_from_ref(module, callee);
  const loom_testbench_invocation_kind_t kind =
      loom_testbench_scenario_subject_invocation_kind(symbol);
  if (kind == LOOM_TESTBENCH_INVOCATION_NONE) {
    return false;
  }

  *out_invocation = (loom_testbench_invocation_plan_t){
      .kind = kind,
      .module = module,
      .op = action_op,
      .callee_ref = callee,
      .provider_id = LOOM_STRING_ID_INVALID,
      .provider = iree_string_view_empty(),
      .attrs = loom_named_attr_slice_empty(),
      .execution_epoch = LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID,
      .launch_schedule_depth = 0,
      .workload_value_ids = call_parameters.values,
      .workload_count = call_parameters.count,
      .input_value_ids = arguments.values,
      .input_count = arguments.count,
      .result_value_ids = result_value_ids,
      .result_count = result_count,
  };
  return true;
}

static void loom_testbench_plan_compare_expectations(
    const loom_module_t* module, iree_host_size_t scenario_index,
    loom_symbol_ref_t scenario_ref, const loom_op_t* compare_op,
    loom_testbench_expectation_plan_t* expectations,
    iree_host_size_t* inout_expectation_count, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    loom_testbench_scenario_action_plan_t* action) {
  action->expectations =
      expectations ? expectations + *inout_expectation_count : NULL;
  const loom_region_t* comparison = loom_check_compare_comparison(compare_op);
  if (comparison == NULL || comparison->block_count == 0) {
    return;
  }
  const loom_block_t* block = loom_region_const_entry_block(comparison);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_check_return_isa(op)) {
      continue;
    }
    if (!loom_testbench_is_expectation_op(op)) {
      loom_testbench_append_scenario_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_INVALID_EXPECTATION, scenario_index, op,
          scenario_ref);
      continue;
    }
    loom_testbench_expectation_plan_t* expectation =
        &expectations[(*inout_expectation_count)++];
    if (!loom_testbench_plan_expectation(module, op, expectation)) {
      loom_testbench_append_scenario_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_INVALID_EXPECTATION, scenario_index, op,
          scenario_ref);
    } else if (expectation->kind == LOOM_TESTBENCH_EXPECTATION_EVENT &&
               iree_string_view_equal(expectation->event.provider,
                                      IREE_SV("device"))) {
      action->expects_device_events = true;
    }
  }
  action->expectation_count =
      action->expectations
          ? (expectations + *inout_expectation_count) - action->expectations
          : 0;
}

static bool loom_testbench_plan_scenario_action(
    const loom_module_t* module, iree_host_size_t scenario_index,
    loom_symbol_ref_t scenario_ref, const loom_op_t* action_op,
    loom_testbench_expectation_plan_t* expectations,
    iree_host_size_t* inout_expectation_count, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    loom_testbench_scenario_action_plan_t* out_action) {
  memset(out_action, 0, sizeof(*out_action));
  out_action->op = action_op;
  if (loom_check_compare_isa(action_op)) {
    out_action->kind = LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE;
    const loom_region_t* comparison = loom_check_compare_comparison(action_op);
    if (comparison == NULL || comparison->block_count == 0) {
      return false;
    }
    const loom_block_t* block = loom_region_const_entry_block(comparison);
    const loom_attribute_t actual_count_attr =
        loom_op_const_attrs(action_op)[1];
    const iree_host_size_t actual_count =
        loom_attr_is_absent(actual_count_attr)
            ? 0
            : (iree_host_size_t)loom_check_compare_actual_count(action_op);
    if (actual_count > block->arg_count ||
        block->arg_count - actual_count != actual_count) {
      return false;
    }
    const loom_symbol_ref_t target_callee =
        loom_check_compare_callee(action_op);
    if (!loom_testbench_plan_scenario_invocation(
            module, action_op, target_callee, block->arg_ids, actual_count,
            &out_action->target)) {
      return false;
    }
    loom_symbol_ref_t oracle_callee =
        loom_check_compare_oracle_callee(action_op);
    if (!loom_symbol_ref_is_valid(oracle_callee)) {
      oracle_callee = target_callee;
    }
    if (!loom_testbench_plan_scenario_invocation(
            module, action_op, oracle_callee,
            actual_count == 0 ? NULL : block->arg_ids + actual_count,
            actual_count, &out_action->oracle)) {
      return false;
    }
    loom_testbench_plan_compare_expectations(
        module, scenario_index, scenario_ref, action_op, expectations,
        inout_expectation_count, issues, issue_capacity, inout_issue_count,
        out_action);
    return true;
  }
  if (loom_check_invoke_isa(action_op)) {
    out_action->kind = LOOM_TESTBENCH_SCENARIO_ACTION_INVOKE;
    const loom_value_slice_t results = loom_check_invoke_results(action_op);
    return loom_testbench_plan_scenario_invocation(
        module, action_op, loom_check_invoke_callee(action_op), results.values,
        results.count, &out_action->target);
  }
  return false;
}

static void loom_testbench_plan_trial(
    const loom_module_t* module, iree_host_size_t scenario_index,
    loom_symbol_ref_t scenario_ref, const loom_op_t* trial_op,
    loom_testbench_value_source_plan_t* value_sources,
    iree_host_size_t* inout_value_source_count,
    loom_testbench_expectation_plan_t* expectations,
    iree_host_size_t* inout_expectation_count, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    loom_testbench_trial_plan_t* out_trial) {
  memset(out_trial, 0, sizeof(*out_trial));
  out_trial->op = trial_op;
  out_trial->ordinal_value_id = LOOM_VALUE_ID_INVALID;
  out_trial->entropy_value_id = LOOM_VALUE_ID_INVALID;
  out_trial->issues = issues ? issues + *inout_issue_count : NULL;
  out_trial->value_sources =
      value_sources ? value_sources + *inout_value_source_count : NULL;
  const int64_t trial_count = loom_check_trial_trial_count(trial_op);
  if (trial_count > 0 && (uint64_t)trial_count <= IREE_HOST_SIZE_MAX) {
    out_trial->trial_count = (iree_host_size_t)trial_count;
  }

  const loom_region_t* body = loom_check_trial_body(trial_op);
  if (body != NULL && body->block_count != 0) {
    const loom_block_t* block = loom_region_const_entry_block(body);
    if (block->arg_count == 2) {
      out_trial->ordinal_value_id = block->arg_ids[0];
      out_trial->entropy_value_id = block->arg_ids[1];
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_testbench_is_value_source_op(op)) {
        loom_testbench_value_source_plan_t* source =
            &value_sources[(*inout_value_source_count)++];
        if (!loom_testbench_plan_value_source(module, op, source)) {
          loom_testbench_append_scenario_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_VALUE_SOURCE, scenario_index, op,
              scenario_ref);
        }
        continue;
      }
      if (loom_check_compare_isa(op) || loom_check_invoke_isa(op)) {
        if (!loom_testbench_plan_scenario_action(
                module, scenario_index, scenario_ref, op, expectations,
                inout_expectation_count, issues, issue_capacity,
                inout_issue_count, &out_trial->action)) {
          loom_testbench_append_scenario_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_SCENARIO_ACTION, scenario_index, op,
              scenario_ref);
        }
        continue;
      }
      loom_testbench_append_scenario_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_UNSUPPORTED_TRIAL_BODY_OP, scenario_index, op,
          scenario_ref);
    }
  }

  out_trial->value_source_count =
      out_trial->value_sources ? (value_sources + *inout_value_source_count) -
                                     out_trial->value_sources
                               : 0;
  out_trial->issue_count =
      out_trial->issues ? (issues + *inout_issue_count) - out_trial->issues : 0;
}

static void loom_testbench_plan_scenario(
    const loom_module_t* module, iree_host_size_t scenario_index,
    const loom_op_t* scenario_op, loom_testbench_trial_plan_t* trials,
    iree_host_size_t* inout_trial_count,
    loom_testbench_value_source_plan_t* value_sources,
    iree_host_size_t* inout_value_source_count,
    loom_testbench_expectation_plan_t* expectations,
    iree_host_size_t* inout_expectation_count, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    loom_testbench_scenario_plan_t* out_scenario) {
  memset(out_scenario, 0, sizeof(*out_scenario));
  out_scenario->op = scenario_op;
  out_scenario->ref = loom_check_scenario_scenario_symbol(scenario_op);
  out_scenario->symbol =
      loom_testbench_scenario_symbol_from_ref(module, out_scenario->ref);
  out_scenario->name =
      loom_testbench_scenario_symbol_name(module, out_scenario->symbol);
  out_scenario->is_public = loom_check_scenario_visibility(scenario_op) ==
                            LOOM_CHECK_VISIBILITY_PUBLIC;
  out_scenario->configuration_count = 1;
  out_scenario->configuration_ordinal_value_id = LOOM_VALUE_ID_INVALID;
  out_scenario->configuration_entropy_value_id = LOOM_VALUE_ID_INVALID;
  out_scenario->configuration_sources =
      value_sources ? value_sources + *inout_value_source_count : NULL;
  out_scenario->trials = trials ? trials + *inout_trial_count : NULL;
  out_scenario->issues = issues ? issues + *inout_issue_count : NULL;

  const loom_attribute_t configuration_count_attr =
      loom_op_const_attrs(scenario_op)[2];
  out_scenario->is_configured = !loom_attr_is_absent(configuration_count_attr);
  const loom_region_t* body = loom_check_scenario_body(scenario_op);
  if (out_scenario->is_configured) {
    const int64_t configuration_count =
        loom_check_scenario_configuration_count(scenario_op);
    if (configuration_count > 0 &&
        (uint64_t)configuration_count <= IREE_HOST_SIZE_MAX) {
      out_scenario->configuration_count = (iree_host_size_t)configuration_count;
    } else {
      out_scenario->configuration_count = 0;
    }
    if (body != NULL && body->block_count != 0) {
      const loom_block_t* block = loom_region_const_entry_block(body);
      if (block->arg_count == 2) {
        out_scenario->configuration_ordinal_value_id = block->arg_ids[0];
        out_scenario->configuration_entropy_value_id = block->arg_ids[1];
      }
    }
  }

  if (body != NULL && body->block_count != 0) {
    const loom_block_t* block = loom_region_const_entry_block(body);
    const loom_op_t* op = NULL;

    // Configuration-scoped recipes execute before any runtime trial domain.
    // Planning them first keeps their shared arena slice independent of nested
    // trial value sources even when the source interleaves declarations.
    loom_block_for_each_op(block, op) {
      if (loom_testbench_scenario_configuration_source_isa(op)) {
        loom_testbench_value_source_plan_t* source =
            &value_sources[(*inout_value_source_count)++];
        if (!loom_testbench_plan_value_source(module, op, source)) {
          loom_testbench_append_scenario_issue(
              issues, issue_capacity, inout_issue_count,
              LOOM_TESTBENCH_ISSUE_INVALID_VALUE_SOURCE, scenario_index, op,
              out_scenario->ref);
        }
      }
    }
    out_scenario->configuration_source_count =
        out_scenario->configuration_sources
            ? (value_sources + *inout_value_source_count) -
                  out_scenario->configuration_sources
            : 0;

    loom_block_for_each_op(block, op) {
      if (loom_check_return_isa(op) || loom_check_requires_isa(op) ||
          loom_check_skip_if_isa(op) ||
          loom_testbench_scenario_configuration_source_isa(op)) {
        continue;
      }
      if (loom_check_trial_isa(op)) {
        loom_testbench_trial_plan_t* trial = &trials[(*inout_trial_count)++];
        loom_testbench_plan_trial(module, scenario_index, out_scenario->ref, op,
                                  value_sources, inout_value_source_count,
                                  expectations, inout_expectation_count, issues,
                                  issue_capacity, inout_issue_count, trial);
        continue;
      }
      loom_testbench_append_scenario_issue(
          issues, issue_capacity, inout_issue_count,
          LOOM_TESTBENCH_ISSUE_UNSUPPORTED_SCENARIO_BODY_OP, scenario_index, op,
          out_scenario->ref);
    }
  }

  out_scenario->trial_count =
      out_scenario->trials
          ? (trials + *inout_trial_count) - out_scenario->trials
          : 0;
  out_scenario->issue_count =
      out_scenario->issues
          ? (issues + *inout_issue_count) - out_scenario->issues
          : 0;
}

iree_status_t loom_testbench_plan_scenarios(
    const loom_module_t* module,
    const loom_testbench_scenario_plan_counts_t* counts,
    iree_arena_allocator_t* arena, loom_testbench_issue_t* issues,
    iree_host_size_t issue_capacity, iree_host_size_t* inout_issue_count,
    const loom_testbench_scenario_plan_t** out_scenarios,
    iree_host_size_t* out_scenario_count) {
  *out_scenarios = NULL;
  *out_scenario_count = 0;

  loom_testbench_scenario_plan_t* scenarios = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_allocate_array(
      arena, counts->scenario_count, sizeof(*scenarios), (void**)&scenarios));
  loom_testbench_trial_plan_t* trials = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_allocate_array(
      arena, counts->trial_count, sizeof(*trials), (void**)&trials));
  loom_testbench_value_source_plan_t* value_sources = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_allocate_array(
      arena, counts->value_source_count, sizeof(*value_sources),
      (void**)&value_sources));
  loom_testbench_expectation_plan_t* expectations = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_allocate_array(
      arena, counts->expectation_count, sizeof(*expectations),
      (void**)&expectations));

  iree_host_size_t scenario_count = 0;
  iree_host_size_t trial_count = 0;
  iree_host_size_t value_source_count = 0;
  iree_host_size_t expectation_count = 0;
  if (module->body != NULL && module->body->block_count != 0) {
    const loom_block_t* module_block =
        loom_region_const_entry_block(module->body);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(module_block, op) {
      if (!loom_check_scenario_isa(op)) {
        continue;
      }
      loom_testbench_plan_scenario(
          module, scenario_count, op, trials, &trial_count, value_sources,
          &value_source_count, expectations, &expectation_count, issues,
          issue_capacity, inout_issue_count, &scenarios[scenario_count]);
      ++scenario_count;
    }
  }

  IREE_ASSERT(scenario_count == counts->scenario_count);
  IREE_ASSERT(trial_count == counts->trial_count);
  IREE_ASSERT(value_source_count <= counts->value_source_count);
  IREE_ASSERT(expectation_count <= counts->expectation_count);
  *out_scenarios = scenarios;
  *out_scenario_count = scenario_count;
  return iree_ok_status();
}
