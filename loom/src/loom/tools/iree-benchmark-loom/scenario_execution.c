// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/scenario_execution.h"

#include <string.h>

#include "iree/hal/api.h"
#include "loom/error/diagnostic.h"
#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/scenario/entropy.h"
#include "loom/tooling/testbench/scenario/executor.h"
#include "loom/tooling/testbench/scenario/values.h"
#include "loom/tools/iree-benchmark-loom/case_execution.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"

struct iree_benchmark_loom_scenario_execution_t {
  // Borrowed run dependencies shared by every scenario work item.
  const iree_benchmark_loom_work_plan_execution_options_t* options;
  // HAL profile used to prepare ordinary target products.
  loom_run_hal_testbench_scenario_profile_t profile;
  // Benchmark-only scenario execution policy bound to |profile|.
  loom_testbench_scenario_execution_options_t scenario_options;
  // Host-visible generator materialization staged by the HAL product.
  loom_testbench_value_materializer_options_t materializer_options;
  // True after the HAL runtime and profile have been initialized.
  bool profile_initialized;
  // True after the selected device has been emitted to the event sink.
  bool device_emitted;
  // Materialized values for |prepared_scenario| and
  // |prepared_configuration_ordinal|.
  loom_testbench_scenario_configuration_values_t configuration;
  // True when |configuration| owns initialized storage.
  bool configuration_initialized;
  // Products prepared from |configuration| before any trial values exist.
  loom_testbench_prepared_scenario_configuration_t prepared;
  // True when |prepared| owns prepared target products.
  bool prepared_initialized;
  // Scenario owning the currently prepared configuration.
  const loom_testbench_scenario_plan_t* prepared_scenario;
  // Configuration ordinal represented by |configuration|.
  iree_host_size_t prepared_configuration_ordinal;
  // Reusable independent value graphs for one selected trial domain.
  loom_testbench_scenario_trial_executor_t trial_executor;
  // True when |trial_executor| owns reusable trial storage.
  bool trial_executor_initialized;
  // Trial domain represented by |trial_executor|.
  iree_host_size_t prepared_trial_index;
  // Physical replica capacity represented by |trial_executor|.
  iree_host_size_t prepared_batch_capacity;
};

static void iree_benchmark_loom_scenario_reset_trial_executor(
    iree_benchmark_loom_scenario_execution_t* execution) {
  if (execution->trial_executor_initialized) {
    loom_testbench_scenario_trial_executor_deinitialize(
        &execution->trial_executor);
    execution->trial_executor_initialized = false;
  }
}

static void iree_benchmark_loom_scenario_reset_configuration(
    iree_benchmark_loom_scenario_execution_t* execution) {
  iree_benchmark_loom_scenario_reset_trial_executor(execution);
  if (execution->prepared_initialized) {
    loom_testbench_prepared_scenario_configuration_deinitialize(
        &execution->prepared);
    execution->prepared_initialized = false;
  }
  if (execution->configuration_initialized) {
    loom_testbench_scenario_configuration_values_deinitialize(
        &execution->configuration);
    execution->configuration_initialized = false;
  }
  execution->prepared_scenario = NULL;
  execution->prepared_configuration_ordinal = 0;
}

static iree_status_t iree_benchmark_loom_scenario_ensure_profile(
    iree_benchmark_loom_scenario_execution_t* execution) {
  if (execution->profile_initialized) {
    return iree_ok_status();
  }
  const iree_benchmark_loom_work_plan_execution_options_t* options =
      execution->options;
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(
      &options->hal_context->execution));

  execution->materializer_options =
      options->case_execution_options->materializer;
  execution->materializer_options.target_pool =
      options->hal_context->execution.runtime.host_pool;
  execution->materializer_options.staging_pool =
      options->hal_context->execution.runtime.staging_pool;
  execution->materializer_options.transfer_queue =
      options->hal_context->execution.runtime.transfer_queue;

  const iree_string_view_t target = options->benchmark_options->target;
  const iree_benchmark_loom_hal_compilation_options_t* compilation =
      options->compilation;
  const loom_run_hal_testbench_actual_provider_options_t provider_options = {
      .context = &options->hal_context->execution,
      .compilation = compilation->compilation,
      .module = compilation->compilation->module,
      .native_module = compilation->native_module,
      .pass_program = compilation->pass_program,
      .requested_target_profile = compilation->requested_target_profile,
      .sanitizer = compilation->sanitizer,
      .result_callback = compilation->result_callback,
  };
  loom_run_hal_testbench_scenario_profile_initialize(
      iree_string_view_is_empty(target)
          ? options->hal_context->execution.driver_name
          : target,
      &provider_options, compilation->source_table,
      (loom_diagnostic_sink_t){.fn = loom_diagnostic_stderr_sink},
      &execution->profile);
  loom_testbench_scenario_execution_options_initialize(
      &execution->scenario_options);
  execution->scenario_options.target =
      loom_run_hal_testbench_scenario_execution_profile(&execution->profile);
  execution->scenario_options.function_call =
      options->case_execution_options->invocation.function_call;
  execution->scenario_options.host_allocator = options->host_allocator;
  execution->profile_initialized = true;
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_scenario_ensure_configuration(
    iree_benchmark_loom_scenario_execution_t* execution,
    const loom_testbench_scenario_plan_t* scenario,
    iree_host_size_t configuration_ordinal) {
  if (execution->prepared_initialized &&
      execution->prepared_scenario == scenario &&
      execution->prepared_configuration_ordinal == configuration_ordinal) {
    return iree_ok_status();
  }
  iree_benchmark_loom_scenario_reset_configuration(execution);
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_configuration_values_initialize(
      execution->options->module_plan->module, scenario,
      execution->options->host_allocator, &execution->configuration));
  execution->configuration_initialized = true;

  iree_status_t status =
      loom_testbench_scenario_configuration_values_materialize(
          &execution->materializer_options, loom_testbench_entropy_root(0),
          configuration_ordinal, &execution->configuration);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_prepare_scenario_configuration(
        &execution->scenario_options, &execution->configuration,
        LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK, &execution->prepared);
  }
  if (iree_status_is_ok(status)) {
    execution->prepared_initialized = true;
    execution->prepared_scenario = scenario;
    execution->prepared_configuration_ordinal = configuration_ordinal;
  } else {
    iree_benchmark_loom_scenario_reset_configuration(execution);
  }
  return status;
}

static iree_status_t iree_benchmark_loom_scenario_ensure_trial_executor(
    iree_benchmark_loom_scenario_execution_t* execution,
    iree_host_size_t trial_index, iree_host_size_t batch_capacity) {
  if (execution->trial_executor_initialized &&
      execution->prepared_trial_index == trial_index &&
      execution->prepared_batch_capacity == batch_capacity) {
    return iree_ok_status();
  }
  iree_benchmark_loom_scenario_reset_trial_executor(execution);
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_trial_executor_initialize(
      &execution->prepared, trial_index, &execution->materializer_options,
      batch_capacity, &execution->trial_executor));
  execution->trial_executor_initialized = true;
  execution->prepared_trial_index = trial_index;
  execution->prepared_batch_capacity = batch_capacity;
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_scenario_execution_create(
    const iree_benchmark_loom_work_plan_execution_options_t* options,
    iree_benchmark_loom_scenario_execution_t** out_execution) {
  *out_execution = NULL;
  iree_benchmark_loom_scenario_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      options->host_allocator, sizeof(*execution), (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->options = options;
  *out_execution = execution;
  return iree_ok_status();
}

void iree_benchmark_loom_scenario_execution_destroy(
    iree_benchmark_loom_scenario_execution_t* execution) {
  if (execution == NULL) {
    return;
  }
  const iree_allocator_t host_allocator = execution->options->host_allocator;
  iree_benchmark_loom_scenario_reset_configuration(execution);
  iree_allocator_free(host_allocator, execution);
}

iree_status_t iree_benchmark_loom_run_scenario_work_item(
    iree_benchmark_loom_scenario_execution_t* execution,
    const iree_benchmark_loom_work_item_t* work_item,
    iree_host_size_t* inout_failed_benchmark_count) {
  const iree_benchmark_loom_work_plan_execution_options_t* options =
      execution->options;
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &options->work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  const loom_testbench_scenario_sample_coordinate_t coordinate =
      work_item->scenario_coordinate;

  IREE_RETURN_IF_ERROR(iree_benchmark_loom_scenario_ensure_profile(execution));
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_scenario_ensure_configuration(
      execution, selection->scenario_plan, coordinate.configuration_ordinal));
  if (!execution->device_emitted) {
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context));
    execution->device_emitted = true;
  }
  const iree_host_size_t batch_capacity =
      selection->policy.hal_options.timing.batch_size;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_scenario_ensure_trial_executor(
      execution, coordinate.trial_index, batch_capacity));

  loom_run_benchmark_result_t timing_result = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_benchmark_scenario_trial(
      &execution->trial_executor, coordinate.trial_ordinal,
      &selection->policy.hal_options.timing, &timing_result));
  iree_benchmark_loom_benchmark_result_t benchmark_result = {
      .executed = true,
      .passed = true,
      .samples_per_iteration = 1,
      .has_hal_benchmark = true,
      .hal_benchmark = {.timing = timing_result},
  };
  return iree_benchmark_loom_emit_work_item_result_aliases(
      options->run, options->module_plan, options->work_plan, work_item,
      &benchmark_result, /*correctness_sample_count=*/0,
      /*correctness_failed_sample_count=*/0, options->event_sink,
      inout_failed_benchmark_count);
}
