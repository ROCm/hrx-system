// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/executor.h"

#include <inttypes.h>
#include <string.h>

#include "loom/tooling/testbench/source_report.h"
#include "loom/util/json.h"

static void loom_testbench_prepared_product_deinitialize(
    loom_testbench_prepared_product_t* product) {
  if (product->destroy != NULL) {
    product->destroy(product->user_data);
  }
  *product = (loom_testbench_prepared_product_t){0};
}

void loom_testbench_scenario_execution_options_initialize(
    loom_testbench_scenario_execution_options_t* out_options) {
  *out_options = (loom_testbench_scenario_execution_options_t){
      .host_allocator = iree_allocator_system(),
  };
}

static iree_status_t loom_testbench_prepare_scenario_product(
    const loom_testbench_execution_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    loom_testbench_scenario_execution_mode_t mode,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  *out_product = (loom_testbench_prepared_product_t){0};
  if (profile->prepare == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "scenario execution profile is not configured");
  }
  iree_status_t status =
      profile->prepare(profile->user_data, invocation, configuration,
                       host_allocator, out_product);
  if (!iree_status_is_ok(status)) {
    loom_testbench_prepared_product_deinitialize(out_product);
    return status;
  }
  if (mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS &&
      out_product->execute == NULL) {
    loom_testbench_prepared_product_deinitialize(out_product);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scenario execution profile '%.*s' prepared no execution callback",
        (int)profile->name.size, profile->name.data);
  }
  if (mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK &&
      out_product->benchmark == NULL) {
    loom_testbench_prepared_product_deinitialize(out_product);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scenario execution profile '%.*s' prepared no benchmark callback",
        (int)profile->name.size, profile->name.data);
  }
  out_product->profile = profile->name;
  out_product->invocation = invocation;
  return iree_ok_status();
}

// Product preparation is rooted at the callee under one concrete scenario
// configuration. Workload and ordinary values remain runtime call data; their
// IDs, storage extents, source locations, and expectations are trial-local.
// Arity is retained because prepared providers size their call transport from
// the representative invocation.
static bool loom_testbench_scenario_invocations_share_product(
    const loom_testbench_invocation_plan_t* lhs,
    const loom_testbench_invocation_plan_t* rhs) {
  return lhs->module == rhs->module && lhs->kind == rhs->kind &&
         lhs->callee_ref.module_id == rhs->callee_ref.module_id &&
         lhs->callee_ref.symbol_id == rhs->callee_ref.symbol_id &&
         lhs->workload_count == rhs->workload_count &&
         lhs->input_count == rhs->input_count &&
         lhs->result_count == rhs->result_count;
}

static iree_status_t loom_testbench_prepare_or_reuse_scenario_product(
    const loom_testbench_execution_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    loom_testbench_scenario_execution_mode_t mode,
    iree_allocator_t host_allocator,
    const loom_testbench_prepared_scenario_trial_t* previous_trials,
    iree_host_size_t previous_trial_count, bool use_oracle,
    loom_testbench_prepared_product_t* out_product) {
  for (iree_host_size_t i = 0; i < previous_trial_count; ++i) {
    const loom_testbench_prepared_product_t* product =
        use_oracle ? &previous_trials[i].oracle : &previous_trials[i].target;
    if (product->invocation != NULL &&
        loom_testbench_scenario_invocations_share_product(product->invocation,
                                                          invocation)) {
      // Reused products borrow the earliest matching product's state. Trials
      // are released in reverse order, so the sole owning handle outlives every
      // borrower and performs teardown exactly once.
      *out_product = *product;
      out_product->destroy = NULL;
      return iree_ok_status();
    }
  }

  return loom_testbench_prepare_scenario_product(
      profile, invocation, configuration, mode, host_allocator, out_product);
}

iree_status_t loom_testbench_prepare_scenario_configuration(
    const loom_testbench_scenario_execution_options_t* options,
    const loom_testbench_scenario_configuration_values_t* configuration,
    loom_testbench_scenario_execution_mode_t mode,
    loom_testbench_prepared_scenario_configuration_t* out_prepared) {
  *out_prepared = (loom_testbench_prepared_scenario_configuration_t){0};
  if (mode != LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS &&
      mode != LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid scenario execution mode %u",
                            (unsigned)mode);
  }
  if (!iree_any_bit_set(configuration->flags,
                        LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "scenario configuration is not materialized");
  }
  const loom_testbench_scenario_plan_t* scenario = configuration->scenario_plan;
  if (scenario->issue_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "scenario '%.*s' has %zu planning issues",
                            (int)scenario->name.size, scenario->name.data,
                            scenario->issue_count);
  }

  const iree_allocator_t host_allocator =
      iree_allocator_is_null(options->host_allocator) ? iree_allocator_system()
                                                      : options->host_allocator;
  *out_prepared = (loom_testbench_prepared_scenario_configuration_t){
      .configuration = configuration,
      .mode = mode,
      .function_call = options->function_call,
      .device_event_capture = options->device_event_capture,
      .host_allocator = host_allocator,
  };
  iree_status_t status = iree_ok_status();
  if (scenario->trial_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, scenario->trial_count,
                                         sizeof(*out_prepared->trials),
                                         (void**)&out_prepared->trials);
    if (iree_status_is_ok(status)) {
      memset(out_prepared->trials, 0,
             scenario->trial_count * sizeof(*out_prepared->trials));
      out_prepared->trial_count = scenario->trial_count;
    }
  }

  for (iree_host_size_t trial_index = 0;
       iree_status_is_ok(status) && trial_index < scenario->trial_count;
       ++trial_index) {
    const loom_testbench_trial_plan_t* trial = &scenario->trials[trial_index];
    loom_testbench_prepared_scenario_trial_t* prepared_trial =
        &out_prepared->trials[trial_index];
    prepared_trial->trial_plan = trial;
    if (trial->issue_count != 0 ||
        trial->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_NONE) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "scenario '%.*s' trial %zu has no executable action",
          (int)scenario->name.size, scenario->name.data, trial_index);
      break;
    }
    status = loom_testbench_prepare_or_reuse_scenario_product(
        &options->target, &trial->action.target, &configuration->values, mode,
        host_allocator, out_prepared->trials, trial_index,
        /*use_oracle=*/false, &prepared_trial->target);
    if (iree_status_is_ok(status) &&
        mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS &&
        trial->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
      status = loom_testbench_prepare_or_reuse_scenario_product(
          &options->oracle, &trial->action.oracle, &configuration->values,
          LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, host_allocator,
          out_prepared->trials, trial_index, /*use_oracle=*/true,
          &prepared_trial->oracle);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_prepared_scenario_configuration_deinitialize(out_prepared);
  }
  return status;
}

void loom_testbench_prepared_scenario_configuration_deinitialize(
    loom_testbench_prepared_scenario_configuration_t* prepared) {
  if (prepared == NULL) {
    return;
  }
  for (iree_host_size_t trial_index = prepared->trial_count; trial_index > 0;
       --trial_index) {
    loom_testbench_prepared_scenario_trial_t* trial =
        &prepared->trials[trial_index - 1];
    loom_testbench_prepared_product_deinitialize(&trial->oracle);
    loom_testbench_prepared_product_deinitialize(&trial->target);
  }
  iree_allocator_free(prepared->host_allocator, prepared->trials);
  *prepared = (loom_testbench_prepared_scenario_configuration_t){0};
}

static iree_status_t loom_testbench_scenario_allocate_array(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(allocator, count, element_size, out_ptr));
  memset(*out_ptr, 0, count * element_size);
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_allocate_value_matrix(
    iree_allocator_t allocator, iree_host_size_t row_count,
    iree_host_size_t column_count, loom_testbench_value_t** out_values) {
  iree_host_size_t value_count = 0;
  if (!iree_host_size_checked_mul(row_count, column_count, &value_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "scenario batch value count overflowed");
  }
  return loom_testbench_scenario_allocate_array(
      allocator, value_count, sizeof(**out_values), (void**)out_values);
}

static iree_status_t loom_testbench_scenario_prepare_generators(
    loom_testbench_invocation_provider_t function_call,
    const loom_testbench_trial_plan_t* trial, iree_allocator_t host_allocator,
    loom_testbench_scenario_trial_executor_t* executor) {
  iree_host_size_t generator_count = 0;
  iree_host_size_t max_input_count = 0;
  iree_host_size_t max_result_count = 0;
  for (iree_host_size_t step_index = 0; step_index < trial->recipe_step_count;
       ++step_index) {
    const loom_testbench_trial_recipe_step_t* step =
        &trial->recipe_steps[step_index];
    if (step->kind != LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR) {
      continue;
    }
    ++generator_count;
    max_input_count = iree_max(max_input_count, step->generator.input_count);
    max_result_count = iree_max(max_result_count, step->generator.result_count);
  }
  if (generator_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_testbench_scenario_allocate_array(
      host_allocator, generator_count, sizeof(*executor->prepared_generators),
      (void**)&executor->prepared_generators));
  const loom_testbench_invocation_options_t invocation_options = {
      .function_call = function_call,
  };
  iree_host_size_t generator_index = 0;
  for (iree_host_size_t step_index = 0; step_index < trial->recipe_step_count;
       ++step_index) {
    const loom_testbench_trial_recipe_step_t* step =
        &trial->recipe_steps[step_index];
    if (step->kind != LOOM_TESTBENCH_TRIAL_RECIPE_STEP_GENERATOR) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_testbench_prepare_invocation(
        &invocation_options, &step->generator,
        &executor->prepared_generators[generator_index++]));
  }
  IREE_ASSERT(generator_index == generator_count);

  executor->generator_schedule = (loom_testbench_invocation_schedule_t){
      .invocations = executor->prepared_generators,
      .invocation_count = generator_count,
      .max_input_count = max_input_count,
      .max_result_count = max_result_count,
  };
  return loom_testbench_invocation_executor_initialize(
      &executor->generator_schedule, host_allocator,
      &executor->generator_executor);
}

static void loom_testbench_scenario_reset_values(
    loom_testbench_value_t* values, iree_host_size_t row_count,
    iree_host_size_t column_count) {
  if (values == NULL) {
    return;
  }
  const iree_host_size_t value_count = row_count * column_count;
  for (iree_host_size_t value_index = 0; value_index < value_count;
       ++value_index) {
    loom_testbench_value_deinitialize(&values[value_index]);
  }
}

static loom_testbench_value_t* loom_testbench_scenario_value_matrix_row(
    loom_testbench_value_t* values, iree_host_size_t row_index,
    iree_host_size_t column_count) {
  return column_count == 0 ? NULL : values + row_index * column_count;
}

static void loom_testbench_scenario_trial_executor_reset_calls(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t call_count) {
  const loom_testbench_invocation_plan_t* target =
      &executor->prepared_trial->trial_plan->action.target;
  loom_testbench_scenario_reset_values(executor->target_call_parameters,
                                       call_count, target->workload_count);
  loom_testbench_scenario_reset_values(executor->target_arguments, call_count,
                                       target->input_count);
  loom_testbench_scenario_reset_values(executor->target_results, call_count,
                                       target->result_count);
  const loom_testbench_invocation_plan_t* oracle =
      &executor->prepared_trial->trial_plan->action.oracle;
  if (executor->mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS &&
      executor->prepared_trial->trial_plan->action.kind ==
          LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
    loom_testbench_scenario_reset_values(executor->oracle_call_parameters,
                                         call_count, oracle->workload_count);
    loom_testbench_scenario_reset_values(executor->oracle_arguments, call_count,
                                         oracle->input_count);
    loom_testbench_scenario_reset_values(executor->oracle_results, call_count,
                                         oracle->result_count);
  }
}

static uint8_t* loom_testbench_scenario_expected_device_events(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t call_index) {
  return executor->expected_device_event_capacity == 0
             ? NULL
             : executor->expected_device_events +
                   call_index * executor->expected_device_event_capacity;
}

static void loom_testbench_scenario_trial_executor_reset_events(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t call_count) {
  if (executor->device_event_capture == NULL) {
    return;
  }
  for (iree_host_size_t call_index = 0; call_index < call_count; ++call_index) {
    loom_testbench_device_event_snapshot_deinitialize(
        &executor->device_event_snapshots[call_index]);
    if (executor->expected_device_event_capacity != 0) {
      memset(
          loom_testbench_scenario_expected_device_events(executor, call_index),
          0, executor->expected_device_event_capacity);
    }
  }
}

iree_status_t loom_testbench_scenario_trial_executor_initialize(
    const loom_testbench_prepared_scenario_configuration_t* prepared,
    iree_host_size_t trial_index,
    const loom_testbench_value_materializer_options_t* materializer_options,
    iree_host_size_t batch_capacity,
    loom_testbench_scenario_trial_executor_t* out_executor) {
  *out_executor = (loom_testbench_scenario_trial_executor_t){0};
  if (trial_index >= prepared->trial_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "scenario trial index %zu exceeds count %zu",
                            trial_index, prepared->trial_count);
  }
  if (batch_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scenario batch capacity must be positive");
  }

  const loom_testbench_prepared_scenario_trial_t* prepared_trial =
      &prepared->trials[trial_index];
  const loom_testbench_invocation_plan_t* target =
      &prepared_trial->trial_plan->action.target;
  const loom_testbench_invocation_plan_t* oracle =
      prepared->mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS &&
              prepared_trial->trial_plan->action.kind ==
                  LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE
          ? &prepared_trial->trial_plan->action.oracle
          : NULL;
  *out_executor = (loom_testbench_scenario_trial_executor_t){
      .prepared_trial = prepared_trial,
      .configuration = prepared->configuration,
      .mode = prepared->mode,
      .materializer_options = *materializer_options,
      .host_allocator = prepared->host_allocator,
      .device_event_capture =
          prepared->mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS
              ? prepared->device_event_capture
              : NULL,
      .batch_capacity = batch_capacity,
  };
  iree_status_t status = loom_testbench_scenario_prepare_generators(
      prepared->function_call, prepared_trial->trial_plan,
      out_executor->host_allocator, out_executor);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_allocate_array(
        out_executor->host_allocator, batch_capacity,
        sizeof(*out_executor->trial_values),
        (void**)&out_executor->trial_values);
  }
  if (iree_status_is_ok(status) &&
      prepared->mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS) {
    status = loom_testbench_scenario_allocate_array(
        out_executor->host_allocator, batch_capacity,
        sizeof(*out_executor->results), (void**)&out_executor->results);
  }
  if (iree_status_is_ok(status) && oracle != NULL) {
    status = loom_testbench_scenario_allocate_array(
        out_executor->host_allocator, batch_capacity,
        sizeof(*out_executor->expectation_reports),
        (void**)&out_executor->expectation_reports);
  }
  if (iree_status_is_ok(status) &&
      prepared->mode == LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS &&
      out_executor->device_event_capture != NULL) {
    status = loom_testbench_scenario_allocate_array(
        out_executor->host_allocator, batch_capacity,
        sizeof(*out_executor->device_event_snapshots),
        (void**)&out_executor->device_event_snapshots);
  }
  if (iree_status_is_ok(status) && out_executor->device_event_capture != NULL &&
      prepared_trial->trial_plan->action.expects_device_events) {
    out_executor->expected_device_event_capacity =
        out_executor->device_event_capture->record_capacity;
    iree_host_size_t event_flag_count = 0;
    if (!iree_host_size_checked_mul(
            batch_capacity, out_executor->expected_device_event_capacity,
            &event_flag_count)) {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "scenario batch device-event flag count overflowed");
    } else {
      status = loom_testbench_scenario_allocate_array(
          out_executor->host_allocator, event_flag_count,
          sizeof(*out_executor->expected_device_events),
          (void**)&out_executor->expected_device_events);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_allocate_array(
        out_executor->host_allocator, batch_capacity,
        sizeof(*out_executor->target_calls),
        (void**)&out_executor->target_calls);
  }
  if (iree_status_is_ok(status) && oracle != NULL) {
    status = loom_testbench_scenario_allocate_array(
        out_executor->host_allocator, batch_capacity,
        sizeof(*out_executor->oracle_calls),
        (void**)&out_executor->oracle_calls);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_allocate_value_matrix(
        out_executor->host_allocator, batch_capacity, target->workload_count,
        &out_executor->target_call_parameters);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_allocate_value_matrix(
        out_executor->host_allocator, batch_capacity, target->input_count,
        &out_executor->target_arguments);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_allocate_value_matrix(
        out_executor->host_allocator, batch_capacity, target->result_count,
        &out_executor->target_results);
  }
  if (iree_status_is_ok(status) && oracle != NULL) {
    status = loom_testbench_scenario_allocate_value_matrix(
        out_executor->host_allocator, batch_capacity, oracle->workload_count,
        &out_executor->oracle_call_parameters);
  }
  if (iree_status_is_ok(status) && oracle != NULL) {
    status = loom_testbench_scenario_allocate_value_matrix(
        out_executor->host_allocator, batch_capacity, oracle->input_count,
        &out_executor->oracle_arguments);
  }
  if (iree_status_is_ok(status) && oracle != NULL) {
    status = loom_testbench_scenario_allocate_value_matrix(
        out_executor->host_allocator, batch_capacity, oracle->result_count,
        &out_executor->oracle_results);
  }

  const loom_module_t* module = prepared->configuration->values.module;
  const loom_testbench_trial_plan_t* trial = prepared_trial->trial_plan;
  const loom_testbench_scenario_trial_realization_t realization =
      oracle != NULL
          ? LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_AND_ORACLE
          : LOOM_TESTBENCH_SCENARIO_TRIAL_REALIZATION_TARGET_ONLY;
  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < batch_capacity; ++call_index) {
    status = loom_testbench_scenario_trial_values_initialize(
        module, prepared->configuration->scenario_plan, trial_index,
        realization, out_executor->host_allocator,
        &out_executor->trial_values[call_index]);
    if (iree_status_is_ok(status)) {
      out_executor->initialized_slot_count = call_index + 1;
    }
    if (iree_status_is_ok(status) && oracle != NULL) {
      status = loom_testbench_expectation_report_initialize(
          trial->action.expectation_count, out_executor->host_allocator,
          &out_executor->expectation_reports[call_index]);
    }
    if (iree_status_is_ok(status)) {
      out_executor->target_calls[call_index] = (loom_testbench_product_call_t){
          .identity = &out_executor->trial_values[call_index].identity,
          .call_parameters = loom_testbench_scenario_value_matrix_row(
              out_executor->target_call_parameters, call_index,
              target->workload_count),
          .arguments = loom_testbench_scenario_value_matrix_row(
              out_executor->target_arguments, call_index, target->input_count),
          .results = loom_testbench_scenario_value_matrix_row(
              out_executor->target_results, call_index, target->result_count),
      };
    }
    if (iree_status_is_ok(status) && oracle != NULL) {
      out_executor->oracle_calls[call_index] = (loom_testbench_product_call_t){
          .identity = &out_executor->trial_values[call_index].identity,
          .call_parameters = loom_testbench_scenario_value_matrix_row(
              out_executor->oracle_call_parameters, call_index,
              oracle->workload_count),
          .arguments = loom_testbench_scenario_value_matrix_row(
              out_executor->oracle_arguments, call_index, oracle->input_count),
          .results = loom_testbench_scenario_value_matrix_row(
              out_executor->oracle_results, call_index, oracle->result_count),
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_scenario_trial_executor_deinitialize(out_executor);
  }
  return status;
}

void loom_testbench_scenario_trial_executor_deinitialize(
    loom_testbench_scenario_trial_executor_t* executor) {
  if (executor == NULL) {
    return;
  }
  if (executor->prepared_trial != NULL) {
    loom_testbench_scenario_trial_executor_reset_calls(
        executor, executor->batch_capacity);
  }
  for (iree_host_size_t call_index = executor->initialized_slot_count;
       call_index > 0; --call_index) {
    if (executor->expectation_reports != NULL) {
      loom_testbench_expectation_report_deinitialize(
          &executor->expectation_reports[call_index - 1]);
    }
    loom_testbench_scenario_trial_values_deinitialize(
        &executor->trial_values[call_index - 1]);
  }
  for (iree_host_size_t call_index = executor->batch_capacity; call_index > 0;
       --call_index) {
    if (executor->device_event_snapshots != NULL) {
      loom_testbench_device_event_snapshot_deinitialize(
          &executor->device_event_snapshots[call_index - 1]);
    }
  }
  iree_allocator_free(executor->host_allocator, executor->oracle_results);
  iree_allocator_free(executor->host_allocator, executor->oracle_arguments);
  iree_allocator_free(executor->host_allocator,
                      executor->oracle_call_parameters);
  iree_allocator_free(executor->host_allocator, executor->target_results);
  iree_allocator_free(executor->host_allocator, executor->target_arguments);
  iree_allocator_free(executor->host_allocator,
                      executor->target_call_parameters);
  iree_allocator_free(executor->host_allocator, executor->oracle_calls);
  iree_allocator_free(executor->host_allocator, executor->target_calls);
  iree_allocator_free(executor->host_allocator,
                      executor->expected_device_events);
  iree_allocator_free(executor->host_allocator,
                      executor->device_event_snapshots);
  iree_allocator_free(executor->host_allocator, executor->expectation_reports);
  iree_allocator_free(executor->host_allocator, executor->results);
  iree_allocator_free(executor->host_allocator, executor->trial_values);
  if (executor->generator_executor.schedule != NULL) {
    loom_testbench_invocation_executor_deinitialize(
        &executor->generator_executor);
  }
  iree_allocator_free(executor->host_allocator, executor->prepared_generators);
  *executor = (loom_testbench_scenario_trial_executor_t){0};
}

static iree_status_t loom_testbench_scenario_load_values(
    const loom_testbench_value_table_t* table, const loom_value_id_t* value_ids,
    iree_host_size_t value_count, loom_testbench_value_t* values) {
  for (iree_host_size_t value_index = 0; value_index < value_count;
       ++value_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_retain(
        table, value_ids[value_index], &values[value_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_load_call(
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* table,
    loom_testbench_product_call_t* call) {
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_load_values(
      table, invocation->workload_value_ids, invocation->workload_count,
      call->call_parameters));
  return loom_testbench_scenario_load_values(table, invocation->input_value_ids,
                                             invocation->input_count,
                                             call->arguments);
}

static iree_status_t loom_testbench_scenario_store_call_results(
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_product_call_t* call, loom_testbench_value_table_t* table) {
  for (iree_host_size_t result_index = 0;
       result_index < invocation->result_count; ++result_index) {
    if (call->results[result_index].kind == LOOM_TESTBENCH_VALUE_KIND_NONE) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "scenario product result %zu was not assigned",
                              result_index);
    }
  }
  for (iree_host_size_t result_index = 0;
       result_index < invocation->result_count; ++result_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_assign_move(
        table, invocation->result_value_ids[result_index],
        &call->results[result_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_load_batch_calls(
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_scenario_trial_values_t* trial_values, bool use_oracle,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  for (iree_host_size_t call_index = 0; call_index < call_count; ++call_index) {
    loom_testbench_value_table_t* table =
        use_oracle ? &trial_values[call_index].oracle
                   : &trial_values[call_index].target;
    IREE_RETURN_IF_ERROR(loom_testbench_scenario_load_call(invocation, table,
                                                           &calls[call_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_store_batch_results(
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_scenario_trial_values_t* trial_values, bool use_oracle,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  for (iree_host_size_t call_index = 0; call_index < call_count; ++call_index) {
    loom_testbench_value_table_t* table =
        use_oracle ? &trial_values[call_index].oracle
                   : &trial_values[call_index].target;
    IREE_RETURN_IF_ERROR(loom_testbench_scenario_store_call_results(
        invocation, &calls[call_index], table));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_execute_product(
    const loom_testbench_prepared_product_t* product,
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_scenario_trial_values_t* trial_values, bool use_oracle,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_load_batch_calls(
      invocation, trial_values, use_oracle, call_count, calls));
  IREE_RETURN_IF_ERROR(
      product->execute(product->user_data, invocation, call_count, calls));
  return loom_testbench_scenario_store_batch_results(
      invocation, trial_values, use_oracle, call_count, calls);
}

static iree_status_t loom_testbench_scenario_materialize_trials(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t first_trial_ordinal, iree_host_size_t trial_count) {
  for (iree_host_size_t call_index = 0; call_index < trial_count;
       ++call_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_scenario_trial_values_materialize(
        &executor->materializer_options,
        executor->generator_schedule.invocation_count != 0
            ? &executor->generator_executor
            : NULL,
        executor->configuration, first_trial_ordinal + call_index,
        &executor->trial_values[call_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_device_event_total(
    const loom_testbench_device_event_list_t* events,
    iree_host_size_t* out_total) {
  if (!iree_host_size_checked_add(events->count, events->dropped_count,
                                  out_total)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "scenario device-event count overflowed");
  }
  return iree_ok_status();
}

static bool loom_testbench_scenario_device_event_key_equal(
    const loom_testbench_device_event_record_t* lhs,
    const loom_testbench_device_event_record_t* rhs) {
  const iree_hal_device_event_t* lhs_event = &lhs->event;
  const iree_hal_device_event_t* rhs_event = &rhs->event;
  if (lhs_event->type != rhs_event->type ||
      lhs_event->severity != rhs_event->severity ||
      lhs_event->flags != rhs_event->flags ||
      !iree_string_view_equal(lhs_event->source.driver_id,
                              rhs_event->source.driver_id) ||
      !iree_string_view_equal(lhs_event->source.device_id,
                              rhs_event->source.device_id) ||
      lhs_event->source.physical_device_ordinal !=
          rhs_event->source.physical_device_ordinal ||
      lhs_event->source.queue_ordinal != rhs_event->source.queue_ordinal ||
      lhs_event->source.executable_id != rhs_event->source.executable_id ||
      lhs_event->source.export_ordinal != rhs_event->source.export_ordinal ||
      lhs->has_site != rhs->has_site) {
    return false;
  }
  if (!lhs->has_site) {
    return true;
  }
  return lhs->site.flags == rhs->site.flags &&
         lhs->site.site_id == rhs->site.site_id &&
         iree_string_view_equal(lhs->site.source_file, rhs->site.source_file) &&
         lhs->site.start_line == rhs->site.start_line &&
         lhs->site.start_column == rhs->site.start_column &&
         lhs->site.end_line == rhs->site.end_line &&
         lhs->site.end_column == rhs->site.end_column &&
         iree_string_view_equal(lhs->site.function_name,
                                rhs->site.function_name) &&
         iree_string_view_equal(lhs->site.operation_name,
                                rhs->site.operation_name);
}

static iree_host_size_t loom_testbench_scenario_device_event_key_count(
    const loom_testbench_device_event_list_t* events,
    const loom_testbench_device_event_record_t* key) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < events->count; ++i) {
    count += loom_testbench_scenario_device_event_key_equal(&events->records[i],
                                                            key);
  }
  return count;
}

static iree_host_size_t loom_testbench_scenario_isolated_device_event_key_count(
    const loom_testbench_device_event_snapshot_t* snapshots,
    iree_host_size_t snapshot_count,
    const loom_testbench_device_event_record_t* key) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < snapshot_count; ++i) {
    count += loom_testbench_scenario_device_event_key_count(
        &snapshots[i].events, key);
  }
  return count;
}

static bool loom_testbench_scenario_device_events_reproduced(
    const loom_testbench_device_event_list_t* batch_events,
    const loom_testbench_device_event_snapshot_t* isolated_snapshots,
    iree_host_size_t isolated_snapshot_count) {
  for (iree_host_size_t i = 0; i < batch_events->count; ++i) {
    const loom_testbench_device_event_record_t* key = &batch_events->records[i];
    bool already_counted = false;
    for (iree_host_size_t j = 0; j < i; ++j) {
      already_counted |= loom_testbench_scenario_device_event_key_equal(
          &batch_events->records[j], key);
    }
    if (already_counted) {
      continue;
    }
    const iree_host_size_t batch_count =
        loom_testbench_scenario_device_event_key_count(batch_events, key);
    const iree_host_size_t isolated_count =
        loom_testbench_scenario_isolated_device_event_key_count(
            isolated_snapshots, isolated_snapshot_count, key);
    if (batch_events->dropped_count == 0 ? isolated_count != batch_count
                                         : isolated_count < batch_count) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_testbench_scenario_isolated_device_event_counts(
    const loom_testbench_device_event_snapshot_t* snapshots,
    iree_host_size_t snapshot_count, iree_host_size_t* out_captured_count,
    iree_host_size_t* out_dropped_count) {
  *out_captured_count = 0;
  *out_dropped_count = 0;
  for (iree_host_size_t i = 0; i < snapshot_count; ++i) {
    if (!iree_host_size_checked_add(*out_captured_count,
                                    snapshots[i].events.count,
                                    out_captured_count) ||
        !iree_host_size_checked_add(*out_dropped_count,
                                    snapshots[i].events.dropped_count,
                                    out_dropped_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "isolated scenario device-event count overflowed");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_snapshot_device_events(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t call_index) {
  return loom_testbench_device_event_capture_snapshot(
      executor->device_event_capture, executor->host_allocator,
      &executor->device_event_snapshots[call_index]);
}

static iree_status_t loom_testbench_scenario_execute_target_isolated(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t trial_count) {
  const loom_testbench_prepared_product_t* target =
      &executor->prepared_trial->target;
  const loom_testbench_invocation_plan_t* invocation =
      &executor->prepared_trial->trial_plan->action.target;
  for (iree_host_size_t call_index = 0; call_index < trial_count;
       ++call_index) {
    loom_testbench_device_event_capture_reset(executor->device_event_capture);
    IREE_RETURN_IF_ERROR(loom_testbench_scenario_execute_product(
        target, invocation, &executor->trial_values[call_index],
        /*use_oracle=*/false, /*call_count=*/1,
        &executor->target_calls[call_index]));
    IREE_RETURN_IF_ERROR(
        loom_testbench_scenario_snapshot_device_events(executor, call_index));
  }
  loom_testbench_device_event_capture_reset(executor->device_event_capture);
  return iree_ok_status();
}

static iree_status_t loom_testbench_scenario_execute_target(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t first_trial_ordinal, iree_host_size_t trial_count) {
  const loom_testbench_scenario_action_plan_t* action =
      &executor->prepared_trial->trial_plan->action;
  const loom_testbench_prepared_product_t* target =
      &executor->prepared_trial->target;
  const loom_testbench_invocation_plan_t* invocation = &action->target;
  if (executor->device_event_capture == NULL) {
    return loom_testbench_scenario_execute_product(
        target, invocation, executor->trial_values, /*use_oracle=*/false,
        trial_count, executor->target_calls);
  }
  if (action->expects_device_events) {
    return loom_testbench_scenario_execute_target_isolated(executor,
                                                           trial_count);
  }

  loom_testbench_device_event_capture_reset(executor->device_event_capture);
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_execute_product(
      target, invocation, executor->trial_values, /*use_oracle=*/false,
      trial_count, executor->target_calls));
  loom_testbench_device_event_list_t batch_events = {0};
  loom_testbench_device_event_capture_events(executor->device_event_capture,
                                             &batch_events);
  const bool requires_attribution =
      batch_events.dropped_count != 0 ||
      loom_testbench_device_event_unhandled_error_count(&batch_events, NULL) !=
          0;
  if (!requires_attribution) {
    loom_testbench_device_event_capture_reset(executor->device_event_capture);
    return iree_ok_status();
  }

  loom_testbench_device_event_snapshot_t batch_snapshot = {0};
  iree_status_t status = loom_testbench_device_event_capture_snapshot(
      executor->device_event_capture, executor->host_allocator,
      &batch_snapshot);
  iree_host_size_t batch_event_total = 0;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_device_event_total(&batch_snapshot.events,
                                                        &batch_event_total);
  }
  if (iree_status_is_ok(status)) {
    loom_testbench_scenario_trial_executor_reset_calls(executor, trial_count);
    status = loom_testbench_scenario_materialize_trials(
        executor, first_trial_ordinal, trial_count);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_testbench_scenario_execute_target_isolated(executor, trial_count);
  }

  iree_host_size_t isolated_captured_count = 0;
  iree_host_size_t isolated_dropped_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_isolated_device_event_counts(
        executor->device_event_snapshots, trial_count, &isolated_captured_count,
        &isolated_dropped_count);
  }
  iree_host_size_t isolated_event_total = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_add(isolated_captured_count,
                                  isolated_dropped_count,
                                  &isolated_event_total)) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "isolated scenario device-event count overflowed");
  }
  if (iree_status_is_ok(status) &&
      (isolated_event_total != batch_event_total ||
       !loom_testbench_scenario_device_events_reproduced(
           &batch_snapshot.events, executor->device_event_snapshots,
           trial_count))) {
    status = iree_make_status(
        IREE_STATUS_ABORTED,
        "target batch device events were not reproduced by deterministic "
        "single-trial replay of configuration %zu trial domain %zu range "
        "[%zu, %zu) (batch %zu captured/%zu dropped, replay %zu captured/%zu "
        "dropped); refusing to guess event ownership",
        executor->configuration->configuration_ordinal,
        executor->trial_values[0].identity.trial_index, first_trial_ordinal,
        first_trial_ordinal + trial_count, batch_snapshot.events.count,
        batch_snapshot.events.dropped_count, isolated_captured_count,
        isolated_dropped_count);
  }
  loom_testbench_device_event_snapshot_deinitialize(&batch_snapshot);
  return status;
}

static iree_status_t loom_testbench_scenario_validate_oracle_device_events(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t first_trial_ordinal, iree_host_size_t trial_count) {
  if (executor->device_event_capture == NULL) {
    return iree_ok_status();
  }
  loom_testbench_device_event_list_t events = {0};
  loom_testbench_device_event_capture_events(executor->device_event_capture,
                                             &events);
  const iree_host_size_t error_count =
      loom_testbench_device_event_unhandled_error_count(&events, NULL);
  if (events.dropped_count != 0 || error_count != 0) {
    return iree_make_status(
        IREE_STATUS_ABORTED,
        "oracle profile emitted %zu error event(s) and dropped %zu event(s) "
        "while executing configuration %zu trial domain %zu range [%zu, %zu)",
        error_count, events.dropped_count,
        executor->configuration->configuration_ordinal,
        executor->trial_values[0].identity.trial_index, first_trial_ordinal,
        first_trial_ordinal + trial_count);
  }
  loom_testbench_device_event_capture_reset(executor->device_event_capture);
  return iree_ok_status();
}

iree_status_t loom_testbench_run_scenario_trial_batch(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t first_trial_ordinal, iree_host_size_t trial_count,
    loom_testbench_scenario_trial_result_list_t* out_results) {
  *out_results = (loom_testbench_scenario_trial_result_list_t){0};
  if (executor->mode != LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario trial executor was not prepared for correctness");
  }
  const loom_testbench_trial_plan_t* trial =
      executor->prepared_trial->trial_plan;
  if (trial_count > executor->batch_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "scenario trial batch count %zu exceeds capacity %zu", trial_count,
        executor->batch_capacity);
  }
  if (first_trial_ordinal > trial->trial_count ||
      trial_count > trial->trial_count - first_trial_ordinal) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "scenario trial range [%zu, %zu) exceeds domain count %zu",
        first_trial_ordinal, first_trial_ordinal + trial_count,
        trial->trial_count);
  }
  loom_testbench_scenario_trial_executor_reset_calls(executor, trial_count);
  loom_testbench_scenario_trial_executor_reset_events(executor, trial_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < trial_count; ++call_index) {
    if (trial->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
      loom_testbench_expectation_report_reset(
          &executor->expectation_reports[call_index]);
    }
  }
  if (iree_status_is_ok(status) && trial_count != 0) {
    status = loom_testbench_scenario_materialize_trials(
        executor, first_trial_ordinal, trial_count);
  }
  if (iree_status_is_ok(status) && trial_count != 0) {
    status = loom_testbench_scenario_execute_target(
        executor, first_trial_ordinal, trial_count);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "executing target profile '%.*s'",
          (int)executor->prepared_trial->target.profile.size,
          executor->prepared_trial->target.profile.data);
    }
  }
  if (iree_status_is_ok(status) && trial_count != 0 &&
      trial->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
    status = loom_testbench_scenario_execute_product(
        &executor->prepared_trial->oracle, &trial->action.oracle,
        executor->trial_values, /*use_oracle=*/true, trial_count,
        executor->oracle_calls);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "executing oracle profile '%.*s'",
          (int)executor->prepared_trial->oracle.profile.size,
          executor->prepared_trial->oracle.profile.data);
    }
  }
  if (iree_status_is_ok(status) && trial_count != 0 &&
      trial->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
    status = loom_testbench_scenario_validate_oracle_device_events(
        executor, first_trial_ordinal, trial_count);
  }

  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < trial_count; ++call_index) {
    loom_testbench_scenario_trial_result_t* result =
        &executor->results[call_index];
    *result = (loom_testbench_scenario_trial_result_t){
        .identity = executor->trial_values[call_index].identity,
        .scenario_plan = executor->configuration->scenario_plan,
        .trial_plan = trial,
        .passed = true,
    };
    loom_testbench_sample_observations_t observations =
        loom_testbench_sample_observations_empty();
    if (executor->device_event_capture != NULL) {
      loom_testbench_device_event_snapshot_t* snapshot =
          &executor->device_event_snapshots[call_index];
      uint8_t* expected_device_events =
          loom_testbench_scenario_expected_device_events(executor, call_index);
      observations.device_events = &snapshot->events;
      observations.expected_device_events = expected_device_events;
      observations.expected_device_event_capacity =
          executor->expected_device_event_capacity;
      result->device_events = &snapshot->events;
      result->expected_device_events = expected_device_events;
    }
    if (trial->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
      loom_testbench_expectation_report_t* report =
          &executor->expectation_reports[call_index];
      status = loom_testbench_evaluate_scenario_action_expectations(
          &trial->action, &executor->trial_values[call_index].target,
          &executor->trial_values[call_index].oracle, &observations, report);
      result->expectation_report = report;
    }
    if (iree_status_is_ok(status) && result->device_events != NULL) {
      result->unhandled_device_event_count =
          loom_testbench_device_event_unhandled_error_count(
              result->device_events, result->expected_device_events);
    }
    result->passed = iree_status_is_ok(status) &&
                     (result->expectation_report == NULL ||
                      result->expectation_report->failure_count == 0) &&
                     (result->device_events == NULL ||
                      (result->device_events->dropped_count == 0 &&
                       result->unhandled_device_event_count == 0));
  }

  if (!iree_status_is_ok(status)) {
    loom_testbench_scenario_trial_executor_reset_calls(executor, trial_count);
    return iree_status_annotate_f(
        status, "executing check.scenario '@%.*s' trial domain",
        (int)executor->configuration->scenario_plan->name.size,
        executor->configuration->scenario_plan->name.data);
  }
  loom_testbench_scenario_trial_executor_reset_calls(executor, trial_count);
  out_results->values = executor->results;
  out_results->count = trial_count;
  return iree_ok_status();
}

iree_status_t loom_testbench_benchmark_scenario_trial(
    loom_testbench_scenario_trial_executor_t* executor,
    iree_host_size_t trial_ordinal, const loom_run_benchmark_options_t* options,
    loom_run_benchmark_result_t* out_result) {
  loom_run_benchmark_result_initialize(out_result);
  if (executor->mode != LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario trial executor was not prepared for benchmarking");
  }
  if (options->batch_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scenario benchmark batch size must be positive");
  }
  if (options->batch_size > executor->batch_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "scenario benchmark batch size %zu exceeds capacity %zu",
        options->batch_size, executor->batch_capacity);
  }
  const loom_testbench_trial_plan_t* trial =
      executor->prepared_trial->trial_plan;
  if (trial_ordinal >= trial->trial_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "trial ordinal %zu exceeds trial domain count %zu",
                            trial_ordinal, trial->trial_count);
  }

  const iree_host_size_t call_count = options->batch_size;
  loom_testbench_scenario_trial_executor_reset_calls(executor, call_count);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < call_count; ++call_index) {
    status = loom_testbench_scenario_trial_values_materialize(
        &executor->materializer_options,
        executor->generator_schedule.invocation_count != 0
            ? &executor->generator_executor
            : NULL,
        executor->configuration, trial_ordinal,
        &executor->trial_values[call_index]);
  }
  const loom_testbench_prepared_product_t* product =
      &executor->prepared_trial->target;
  const loom_testbench_invocation_plan_t* invocation = &trial->action.target;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_load_batch_calls(
        invocation, executor->trial_values, /*use_oracle=*/false, call_count,
        executor->target_calls);
  }
  if (iree_status_is_ok(status)) {
    status = product->benchmark(product->user_data, invocation, call_count,
                                executor->target_calls, options,
                                executor->host_allocator, out_result);
  }
  if (iree_status_is_ok(status) &&
      out_result->batch_size != options->batch_size) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario product benchmark reported batch size %zu instead of %zu",
        out_result->batch_size, options->batch_size);
  }
  loom_testbench_scenario_trial_executor_reset_calls(executor, call_count);
  if (!iree_status_is_ok(status)) {
    loom_run_benchmark_result_initialize(out_result);
    return iree_status_annotate_f(
        status, "benchmarking target profile '%.*s' for check.scenario '@%.*s'",
        (int)product->profile.size, product->profile.data,
        (int)executor->configuration->scenario_plan->name.size,
        executor->configuration->scenario_plan->name.data);
  }
  return iree_ok_status();
}

iree_status_t loom_testbench_scenario_trial_result_write_json(
    const loom_testbench_scenario_trial_result_t* result,
    loom_output_stream_t* stream) {
  const char* action =
      result->trial_plan->action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE
          ? "compare"
          : "invoke";
  char entropy[35];
  const int entropy_length = iree_snprintf(
      entropy, sizeof(entropy), "0x%016" PRIx64 "%016" PRIx64,
      result->identity.entropy_root.high, result->identity.entropy_root.low);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("scenario"), result->scenario_plan->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("action"), iree_make_cstring_view(action)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("entropy"),
      iree_make_string_view(entropy, (iree_host_size_t)entropy_length)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("configuration_ordinal"),
      result->identity.configuration_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("trial_index"), result->identity.trial_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("trial_ordinal"), result->identity.trial_ordinal));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("passed"), result->passed));
  if (result->expectation_report != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("expectations")));
    IREE_RETURN_IF_ERROR(loom_testbench_expectation_report_write_json(
        result->expectation_report, stream));
  }
  if (result->device_events != NULL &&
      (result->unhandled_device_event_count != 0 ||
       result->device_events->dropped_count != 0)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("device_events")));
    IREE_RETURN_IF_ERROR(loom_testbench_device_event_failure_write_json(
        result->device_events, result->expected_device_events,
        result->unhandled_device_event_count, stream));
    IREE_RETURN_IF_ERROR(loom_testbench_write_source_location_json(
        result->trial_plan->action.target.module,
        result->trial_plan->action.op->location, &object));
  }
  return loom_json_object_end(&object);
}
