// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/testing/source_workload_pipeline.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/lower/source_selection.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/pipeline/pipeline.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/pass/builder.h"
#include "loom/pass/interpreter.h"
#include "loom/pass/program.h"
#include "loom/pass/value_facts.h"
#include "loom/verify/verify.h"

static iree_status_t loom_low_source_workload_verify_general_module(
    const loom_module_t* module, iree_string_view_t phase) {
  loom_verify_options_t options = {0};
  loom_verify_result_t result = {0};
  IREE_RETURN_IF_ERROR(loom_verify_module(module, &options, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated %.*s failed verification with %" PRIu32
                            " error%s",
                            (int)phase.size, phase.data, result.error_count,
                            result.error_count == 1 ? "" : "s");
  }
  return iree_ok_status();
}

static void loom_low_source_workload_count_module_source_ops(
    const loom_module_t* module,
    loom_low_source_workload_counts_t* out_counts) {
  memset(out_counts, 0, sizeof(*out_counts));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* op = module->symbols.entries[i].defining_op;
    if (!loom_func_def_isa(op)) {
      continue;
    }
    loom_low_source_workload_counts_t func_counts;
    loom_low_source_workload_count_func_ops(module, op, &func_counts);
    loom_low_source_workload_counts_accumulate(out_counts, &func_counts);
  }
}

static iree_status_t loom_low_source_workload_verify_low_module(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options) {
  const loom_low_verify_options_t verify_options = {
      .descriptor_registry = options->descriptor_registry,
      .max_errors = 20,
  };
  loom_low_verify_result_t result = {0};
  loom_low_verify_scratch_t scratch =
      loom_low_verify_scratch_for_module(module);
  IREE_RETURN_IF_ERROR(
      loom_low_verify_module(module, &verify_options, &scratch, &result));
  if (result.error_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generated low function failed verification");
  }
  return iree_ok_status();
}

static uint32_t loom_low_source_workload_count_low_descriptor_ops(
    const loom_op_t* low_func_op) {
  uint32_t count = 0;
  const loom_region_t* body = loom_low_function_const_body(low_func_op);
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_op_isa(op) || loom_low_const_isa(op)) {
        ++count;
      }
    }
  }
  return count;
}

static iree_status_t loom_low_source_workload_build_preparation_pipeline(
    loom_builder_t* builder, void* user_data) {
  return loom_low_pipeline_build_packetization_preparation(builder);
}

static iree_status_t loom_low_source_workload_prepare_low_functions(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options,
    loom_op_t* const* low_func_ops, iree_host_size_t low_func_count,
    iree_arena_block_pool_t* block_pool) {
  if (low_func_count == 0) {
    return iree_ok_status();
  }

  loom_module_t* pipeline_module = NULL;
  iree_status_t status = loom_module_allocate(
      module->context, IREE_SV("__low_source_workload_prepare_packetization"),
      block_pool, NULL, module->allocator, &pipeline_module);
  const loom_op_t* pipeline_op = NULL;
  if (iree_status_is_ok(status)) {
    loom_op_t* mutable_pipeline_op = NULL;
    status = loom_pass_ir_build_pipeline(
        pipeline_module, IREE_SV("__low_source_workload_prepare_packetization"),
        LOOM_PASS_ANCHOR_FUNC,
        loom_low_source_workload_build_preparation_pipeline, NULL,
        &mutable_pipeline_op);
    pipeline_op = mutable_pipeline_op;
  }

  loom_low_pass_environment_storage_t environment_storage = {0};
  loom_pass_environment_t environment =
      loom_low_pass_environment_storage_initialize(
          options->descriptor_registry, /*lower_policy_registry=*/NULL,
          /*legality_provider_list=*/NULL, /*legalizer_provider_list=*/NULL,
          /*math_policy_registry=*/NULL, /*compile_report=*/NULL,
          loom_target_selection_empty(), loom_symbol_ref_null(),
          &environment_storage);
  loom_pass_program_t program = {0};
  if (iree_status_is_ok(status)) {
    const loom_pass_program_compile_options_t compile_options = {
        .registry = options->pass_registry,
        .environment = environment,
    };
    status = loom_pass_program_compile_pipeline(
        pipeline_module, pipeline_op, &compile_options, block_pool, &program);
  }
  if (iree_status_is_ok(status)) {
    const loom_pass_interpreter_options_t interpreter_options = {
        .block_pool = block_pool,
        .environment = environment,
    };
    for (iree_host_size_t i = 0;
         i < low_func_count && iree_status_is_ok(status); ++i) {
      loom_pass_run_result_t run_result = {0};
      status = loom_pass_interpreter_run_function(
          &program, module, loom_func_like_cast(module, low_func_ops[i]),
          &interpreter_options, &run_result);
      if (iree_status_is_ok(status) && run_result.error_count != 0) {
        break;
      }
    }
  }

  loom_pass_program_deinitialize(&program);
  if (pipeline_module != NULL) {
    loom_module_free(pipeline_module);
  }
  return status;
}

iree_status_t loom_low_source_workload_run_pipeline(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_low_source_workload_pipeline_counters_t* out_counters) {
  memset(out_counters, 0, sizeof(*out_counters));

  loom_low_source_workload_count_module_source_ops(
      module, &out_counters->source_counts);
  iree_status_t status =
      loom_low_source_workload_verify_general_module(module, IREE_SV("source"));

  iree_arena_allocator_t lowering_arena;
  bool lowering_arena_initialized = false;
  loom_pass_value_fact_owner_t value_facts = {0};
  bool value_facts_initialized = false;
  loom_low_lower_result_t lower_result = {0};
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &lowering_arena);
    lowering_arena_initialized = true;
    loom_pass_value_fact_owner_initialize(block_pool, &value_facts);
    value_facts_initialized = true;
    const loom_low_source_selection_options_t selection_options = {
        .policy_registry = options->policy_registry,
    };
    loom_low_source_selection_list_t selection_list = {0};
    status = loom_low_select_source_funcs(module, &selection_options,
                                          &lowering_arena, &selection_list);
    loom_op_t** lowered_funcs = NULL;
    if (iree_status_is_ok(status) && selection_list.count == 0) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "generated workload has no compatible source functions");
    }
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate_array(&lowering_arena, selection_list.count,
                                         sizeof(*lowered_funcs),
                                         (void**)&lowered_funcs);
    }
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      const loom_low_source_selection_t* selection = &selection_list.values[i];
      loom_value_fact_table_t* fact_table = NULL;
      status = loom_pass_value_fact_owner_acquire(
          &value_facts, module,
          loom_pass_value_fact_scope_function_for_target(
              selection->func, selection->target_bundle),
          &fact_table);
      if (!iree_status_is_ok(status)) {
        break;
      }
      const loom_low_lower_options_t lower_options = {
          .target_ref = selection->target_ref,
          .bundle = selection->target_bundle,
          .target_data = selection->target_data,
          .descriptor_registry = options->descriptor_registry,
          .policy = selection->policy,
          .fact_table = fact_table,
          .max_errors = 20,
      };
      loom_low_lower_result_t func_lower_result = {0};
      status = loom_low_lower_function(module, selection->func, &lower_options,
                                       &func_lower_result);
      loom_pass_value_fact_owner_invalidate(&value_facts);
      lower_result.error_count += func_lower_result.error_count;
      lower_result.remark_count += func_lower_result.remark_count;
      if (iree_status_is_ok(status) && func_lower_result.error_count != 0) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source lowering produced errors");
      } else if (iree_status_is_ok(status) &&
                 func_lower_result.low_func_op == NULL) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "source lowering did not emit a target-low function");
      }
      if (iree_status_is_ok(status)) {
        lowered_funcs[i] = func_lower_result.low_func_op;
      }
      loom_low_lower_result_deinitialize(&func_lower_result);
    }
    if (iree_status_is_ok(status)) {
      out_counters->lower_error_count = lower_result.error_count;
      out_counters->lower_remark_count = lower_result.remark_count;
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_source_workload_verify_general_module(module,
                                                              IREE_SV("low"));
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_source_workload_prepare_low_functions(
          module, options, lowered_funcs, selection_list.count, block_pool);
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_source_workload_verify_general_module(
          module, IREE_SV("prepared low"));
    }
    if (iree_status_is_ok(status)) {
      status = loom_low_source_workload_verify_low_module(module, options);
    }
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      out_counters->low_descriptor_op_count +=
          loom_low_source_workload_count_low_descriptor_ops(lowered_funcs[i]);
    }

    iree_arena_allocator_t frame_arena;
    bool frame_arena_initialized = false;
    if (iree_status_is_ok(status)) {
      iree_arena_initialize(block_pool, &frame_arena);
      frame_arena_initialized = true;
    }
    const loom_low_emission_frame_options_t frame_options = {
        .descriptor_registry = options->descriptor_registry,
        .schedule_strategy = options->schedule_strategy,
    };
    for (iree_host_size_t i = 0;
         i < selection_list.count && iree_status_is_ok(status); ++i) {
      loom_low_emission_frame_t frame = {0};
      status = loom_low_emission_frame_build(
          module, lowered_funcs[i], &frame_options, &frame_arena, &frame);
      if (iree_status_is_ok(status)) {
        out_counters->schedule_node_count +=
            frame.schedule.scheduled_node_count;
        out_counters->schedule_dependency_count +=
            frame.schedule.dependency_count;
        out_counters->schedule_resource_use_count +=
            frame.schedule.resource_use_count;
        out_counters->schedule_hazard_gap_count +=
            frame.schedule.hazard_gap_count;
        out_counters->allocation_assignment_count +=
            frame.allocation.assignment_count;
        out_counters->allocation_spill_count += frame.allocation.spill_count;
        out_counters->allocation_coalesced_copy_count +=
            frame.allocation.coalesced_copy_count;
        out_counters->allocation_materialized_copy_count +=
            frame.allocation.materialized_copy_count;
      }
    }
    if (frame_arena_initialized) {
      iree_arena_deinitialize(&frame_arena);
    }
  }

  if (value_facts_initialized) {
    loom_pass_value_fact_owner_deinitialize(&value_facts);
  }
  if (lowering_arena_initialized) {
    iree_arena_deinitialize(&lowering_arena);
  }
  return status;
}
