// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/scenario_execution.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "iree/hal/api.h"
#include "loom/error/diagnostic.h"
#include "loom/tooling/execution/hal/scenario_profile.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/scenario/entropy.h"
#include "loom/tooling/testbench/scenario/executor.h"
#include "loom/tooling/testbench/scenario/values.h"
#include "loom/tools/loom-benchmark/case_execution.h"
#include "loom/tools/loom-benchmark/hal_actual.h"
#include "loom/tools/loom-benchmark/input_ring.h"
#include "loom/tools/loom-benchmark/profile_report.h"

struct loom_benchmark_scenario_execution_t {
  // Borrowed run dependencies shared by every scenario work item.
  const loom_benchmark_work_plan_execution_options_t* options;
  // Shared entropy domain for controlled comparisons, or empty to use each
  // scenario's own name.
  iree_string_view_t entropy_domain;
  // HAL profile used to prepare ordinary target products.
  loom_run_hal_testbench_scenario_profile_t profile;
  // Target, independent oracle, and correctness observation dependencies.
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
  // True when |prepared| owns prepared target and oracle products.
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
  // Benchmark selection owning the prepared concrete trial.
  const loom_benchmark_selected_benchmark_t* prepared_selection;
  // Benchmark-local sample ordinal owning |prepared_coordinate|.
  iree_host_size_t prepared_benchmark_sample_ordinal;
  // Concrete trial represented by the current layout and value ring.
  loom_testbench_scenario_sample_coordinate_t prepared_coordinate;
  // HAL binding layout for one realization of |prepared_coordinate|.
  loom_run_hal_testbench_scenario_input_layout_t prepared_layout;
  // Physical input realizations materialized for each measurement window.
  iree_host_size_t prepared_ring_count;
};

static void loom_benchmark_scenario_reset_trial_executor(
    loom_benchmark_scenario_execution_t* execution) {
  if (execution->trial_executor_initialized) {
    loom_testbench_scenario_trial_executor_deinitialize(
        &execution->trial_executor);
    execution->trial_executor_initialized = false;
  }
  execution->prepared_selection = NULL;
  execution->prepared_benchmark_sample_ordinal = 0;
  execution->prepared_coordinate =
      (loom_testbench_scenario_sample_coordinate_t){0};
  execution->prepared_layout =
      (loom_run_hal_testbench_scenario_input_layout_t){0};
  execution->prepared_ring_count = 0;
}

static void loom_benchmark_scenario_reset_configuration(
    loom_benchmark_scenario_execution_t* execution) {
  loom_benchmark_scenario_reset_trial_executor(execution);
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

static iree_status_t loom_benchmark_scenario_ensure_profile(
    loom_benchmark_scenario_execution_t* execution) {
  if (execution->profile_initialized) {
    return iree_ok_status();
  }
  const loom_benchmark_work_plan_execution_options_t* options =
      execution->options;
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_context_ensure_runtime(
      &options->hal_context->execution));

  execution->materializer_options =
      options->case_execution_options->materializer;
  execution->materializer_options.staging_pool =
      options->hal_context->execution.runtime.staging_pool;
  execution->materializer_options.transfer_queue =
      options->hal_context->execution.runtime.transfer_queue;

  const iree_string_view_t target = options->benchmark_options->target;
  const loom_benchmark_hal_compilation_options_t* compilation =
      options->compilation;
  loom_run_hal_testbench_actual_provider_options_t provider_options;
  loom_benchmark_hal_provider_options_initialize(
      options->hal_context, compilation, /*invocation=*/NULL,
      compilation->result_callback, &provider_options);
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
  execution->scenario_options.oracle = options->scenario_oracle_profile;
  execution->scenario_options.prepare_oracles =
      options->benchmark_options->scenario_correctness;
  execution->scenario_options.device_event_capture =
      options->case_execution_options->device_event_capture;
  execution->scenario_options.function_call =
      options->case_execution_options->invocation.function_call;
  execution->scenario_options.host_allocator = options->host_allocator;
  execution->profile_initialized = true;
  return iree_ok_status();
}

static iree_status_t loom_benchmark_scenario_ensure_configuration(
    loom_benchmark_scenario_execution_t* execution,
    const loom_testbench_scenario_plan_t* scenario,
    iree_host_size_t configuration_ordinal) {
  if (execution->prepared_initialized &&
      execution->prepared_scenario == scenario &&
      execution->prepared_configuration_ordinal == configuration_ordinal) {
    return iree_ok_status();
  }
  loom_benchmark_scenario_reset_configuration(execution);
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_configuration_values_initialize(
      execution->options->module_plan->module, scenario,
      execution->options->host_allocator, &execution->configuration));
  execution->configuration_initialized = true;

  iree_status_t status =
      loom_testbench_scenario_configuration_values_materialize_in_entropy_domain(
          &execution->materializer_options, loom_testbench_entropy_root(0),
          iree_string_view_is_empty(execution->entropy_domain)
              ? scenario->name
              : execution->entropy_domain,
          configuration_ordinal, &execution->configuration);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_prepare_scenario_configuration(
        &execution->scenario_options, &execution->configuration,
        &execution->prepared);
  }
  if (iree_status_is_ok(status)) {
    execution->prepared_initialized = true;
    execution->prepared_scenario = scenario;
    execution->prepared_configuration_ordinal = configuration_ordinal;
  } else {
    loom_benchmark_scenario_reset_configuration(execution);
  }
  return status;
}

static iree_status_t loom_benchmark_scenario_ensure_trial_executor(
    loom_benchmark_scenario_execution_t* execution,
    iree_host_size_t trial_index, iree_host_size_t batch_capacity) {
  if (execution->trial_executor_initialized &&
      execution->prepared_trial_index == trial_index &&
      execution->prepared_batch_capacity == batch_capacity) {
    return iree_ok_status();
  }
  loom_benchmark_scenario_reset_trial_executor(execution);
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_trial_executor_initialize(
      &execution->prepared, trial_index,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK,
      &execution->materializer_options, batch_capacity,
      &execution->trial_executor));
  execution->trial_executor_initialized = true;
  execution->prepared_trial_index = trial_index;
  execution->prepared_batch_capacity = batch_capacity;
  return iree_ok_status();
}

iree_status_t loom_benchmark_scenario_execution_allocate(
    const loom_benchmark_work_plan_execution_options_t* options,
    iree_string_view_t entropy_domain,
    loom_benchmark_scenario_execution_t** out_execution) {
  *out_execution = NULL;
  loom_benchmark_scenario_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      options->host_allocator, sizeof(*execution), (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->options = options;
  execution->entropy_domain = entropy_domain;
  *out_execution = execution;
  return iree_ok_status();
}

void loom_benchmark_scenario_execution_free(
    loom_benchmark_scenario_execution_t* execution) {
  if (execution == NULL) {
    return;
  }
  const iree_allocator_t host_allocator = execution->options->host_allocator;
  loom_benchmark_scenario_reset_configuration(execution);
  iree_allocator_free(host_allocator, execution);
}

typedef struct loom_benchmark_scenario_layout_query_t {
  // HAL input layout reported for one physical call.
  loom_run_hal_testbench_scenario_input_layout_t layout;
} loom_benchmark_scenario_layout_query_t;

static iree_status_t loom_benchmark_scenario_query_layout(
    void* user_data, const loom_testbench_prepared_product_t* product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  loom_benchmark_scenario_layout_query_t* query =
      (loom_benchmark_scenario_layout_query_t*)user_data;
  if (call_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "scenario layout query requires one call");
  }
  return loom_run_hal_testbench_query_scenario_product_input_layout(
      product, invocation, &calls[0], &query->layout);
}

static iree_status_t loom_benchmark_scenario_check_trial(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_selected_benchmark_t* selection,
    iree_host_size_t work_item_index, iree_host_size_t benchmark_sample_ordinal,
    loom_testbench_scenario_sample_coordinate_t coordinate, bool* out_passed) {
  // Correctness owns only one target/oracle realization. Release its graphs
  // and observations before allocating the independent target-only timing ring.
  loom_testbench_scenario_trial_executor_t checker = {0};
  iree_status_t status = loom_testbench_scenario_trial_executor_initialize(
      &execution->prepared, coordinate.trial_index,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS,
      &execution->materializer_options, /*batch_capacity=*/1, &checker);
  loom_testbench_scenario_trial_result_list_t results = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_run_scenario_trial_batch(
        &checker, coordinate.trial_ordinal, /*trial_count=*/1, &results);
  }
  if (iree_status_is_ok(status)) {
    *out_passed = results.values[0].passed;
    status = loom_benchmark_event_sink_emit(
        execution->options->event_sink,
        &(loom_benchmark_event_t){
            .kind = LOOM_BENCHMARK_EVENT_SCENARIO_SAMPLE,
            .scenario_sample =
                {
                    .run = execution->options->run,
                    .selection = selection,
                    .work_item_index = work_item_index,
                    .benchmark_sample_ordinal = benchmark_sample_ordinal,
                    .result = &results.values[0],
                },
        });
  }
  loom_testbench_scenario_trial_executor_deinitialize(&checker);
  return status;
}

iree_status_t loom_benchmark_scenario_execution_prepare(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_selected_benchmark_t* selection,
    iree_host_size_t work_item_index, iree_host_size_t benchmark_sample_ordinal,
    loom_testbench_scenario_sample_coordinate_t coordinate,
    bool* out_runnable) {
  *out_runnable = false;
  loom_benchmark_scenario_reset_trial_executor(execution);
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_ensure_profile(execution));
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_ensure_configuration(
      execution, selection->scenario_plan, coordinate.configuration_ordinal));

  const loom_benchmark_options_t* benchmark_options =
      execution->options->benchmark_options;
  if (benchmark_options->scenario_correctness) {
    bool passed = false;
    IREE_RETURN_IF_ERROR(loom_benchmark_scenario_check_trial(
        execution, selection, work_item_index, benchmark_sample_ordinal,
        coordinate, &passed));
    if (!passed) {
      return iree_ok_status();
    }
  }

  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_ensure_trial_executor(
      execution, coordinate.trial_index, /*batch_capacity=*/1));

  loom_benchmark_scenario_layout_query_t layout_query = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_trial_with_materialized_calls(
      &execution->trial_executor, coordinate.trial_ordinal, /*call_count=*/1,
      (loom_testbench_scenario_trial_calls_callback_t){
          .fn = loom_benchmark_scenario_query_layout,
          .user_data = &layout_query,
      }));
  const iree_host_size_t ring_count = loom_benchmark_input_ring_select_count(
      benchmark_options->input_ring_count,
      (uint64_t)benchmark_options->input_ring_min_bytes,
      layout_query.layout.binding_range_set_bytes,
      selection->policy.hal_options.timing.batch_size);
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_ensure_trial_executor(
      execution, coordinate.trial_index, ring_count));

  execution->prepared_selection = selection;
  execution->prepared_benchmark_sample_ordinal = benchmark_sample_ordinal;
  execution->prepared_coordinate = coordinate;
  execution->prepared_layout = layout_query.layout;
  execution->prepared_ring_count = ring_count;
  *out_runnable = true;
  return iree_ok_status();
}

static iree_status_t loom_benchmark_scenario_compare_type_slice(
    const loom_testbench_invocation_plan_t* baseline,
    const loom_value_id_t* baseline_values,
    const loom_testbench_invocation_plan_t* candidate,
    const loom_value_id_t* candidate_values, iree_host_size_t value_count,
    const char* value_kind) {
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    if (!loom_type_equal(
            loom_module_value_type(baseline->module, baseline_values[i]),
            loom_module_value_type(candidate->module, candidate_values[i]))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s %" PRIhsz " type differs", value_kind, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_benchmark_scenario_compare_invocation_layout(
    const loom_testbench_invocation_plan_t* baseline,
    const loom_testbench_invocation_plan_t* candidate) {
  if (baseline->kind != candidate->kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation kinds differ");
  }
  if (baseline->workload_count != candidate->workload_count ||
      baseline->input_count != candidate->input_count ||
      baseline->result_count != candidate->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invocation arities differ (workload=%" PRIhsz
                            "/%" PRIhsz ", inputs=%" PRIhsz "/%" PRIhsz
                            ", results=%" PRIhsz "/%" PRIhsz ")",
                            candidate->workload_count, baseline->workload_count,
                            candidate->input_count, baseline->input_count,
                            candidate->result_count, baseline->result_count);
  }
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_compare_type_slice(
      baseline, baseline->workload_value_ids, candidate,
      candidate->workload_value_ids, baseline->workload_count,
      "workload parameter"));
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_compare_type_slice(
      baseline, baseline->input_value_ids, candidate,
      candidate->input_value_ids, baseline->input_count, "argument"));
  return loom_benchmark_scenario_compare_type_slice(
      baseline, baseline->result_value_ids, candidate,
      candidate->result_value_ids, baseline->result_count, "result");
}

static bool loom_benchmark_scenario_buffer_view_metadata_equal(
    iree_hal_buffer_view_t* baseline, iree_hal_buffer_view_t* candidate) {
  if ((baseline == NULL) != (candidate == NULL)) {
    return false;
  }
  if (baseline == NULL) {
    return true;
  }
  if (iree_hal_buffer_view_element_type(baseline) !=
          iree_hal_buffer_view_element_type(candidate) ||
      iree_hal_buffer_view_encoding_type(baseline) !=
          iree_hal_buffer_view_encoding_type(candidate) ||
      iree_hal_buffer_view_shape_rank(baseline) !=
          iree_hal_buffer_view_shape_rank(candidate)) {
    return false;
  }
  const iree_host_size_t rank = iree_hal_buffer_view_shape_rank(baseline);
  const iree_hal_dim_t* baseline_shape =
      iree_hal_buffer_view_shape_dims(baseline);
  const iree_hal_dim_t* candidate_shape =
      iree_hal_buffer_view_shape_dims(candidate);
  for (iree_host_size_t i = 0; i < rank; ++i) {
    if (baseline_shape[i] != candidate_shape[i]) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_benchmark_scenario_compare_buffer_contents(
    const iree_tooling_buffer_binding_t* baseline,
    const iree_tooling_buffer_binding_t* candidate, iree_host_size_t ordinal) {
  iree_hal_buffer_mapping_t baseline_mapping = {0};
  iree_hal_buffer_mapping_t candidate_mapping = {0};
  bool baseline_mapped = false;
  bool candidate_mapped = false;
  iree_status_t status = iree_hal_buffer_map_range(
      baseline->buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE,
      baseline->byte_offset, baseline->byte_length, &baseline_mapping);
  if (iree_status_is_ok(status)) {
    baseline_mapped = true;
    status = iree_hal_buffer_map_range(
        candidate->buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE,
        candidate->byte_offset, candidate->byte_length, &candidate_mapping);
  }
  if (iree_status_is_ok(status)) {
    candidate_mapped = true;
    if (baseline_mapping.contents.data_length != 0 &&
        !iree_all_bits_set(iree_hal_buffer_memory_type(baseline_mapping.buffer),
                           IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
      status = iree_hal_buffer_mapping_invalidate_range(&baseline_mapping, 0,
                                                        IREE_HAL_WHOLE_BUFFER);
    }
  }
  if (iree_status_is_ok(status) &&
      candidate_mapping.contents.data_length != 0 &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(candidate_mapping.buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_invalidate_range(&candidate_mapping, 0,
                                                      IREE_HAL_WHOLE_BUFFER);
  }
  if (iree_status_is_ok(status) && baseline_mapping.contents.data_length != 0 &&
      memcmp(baseline_mapping.contents.data, candidate_mapping.contents.data,
             baseline_mapping.contents.data_length) != 0) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "value %" PRIhsz " buffer contents differ", ordinal);
  }
  if (candidate_mapped) {
    status = iree_status_join(status,
                              iree_hal_buffer_unmap_range(&candidate_mapping));
  }
  if (baseline_mapped) {
    status = iree_status_join(status,
                              iree_hal_buffer_unmap_range(&baseline_mapping));
  }
  return status;
}

typedef struct loom_benchmark_scenario_buffer_location_t {
  bool traceable;
  loom_value_id_t allocation_value_id;
  iree_hal_buffer_t* allocated_buffer;
  iree_device_size_t byte_offset;
} loom_benchmark_scenario_buffer_location_t;

static loom_benchmark_scenario_buffer_location_t
loom_benchmark_scenario_buffer_location(const loom_testbench_value_t* value) {
  if (value->buffer_reference.is_traceable) {
    return (loom_benchmark_scenario_buffer_location_t){
        .traceable = true,
        .allocation_value_id = value->buffer_reference.allocation_value_id,
        .byte_offset = value->buffer_reference.byte_offset,
    };
  }
  return (loom_benchmark_scenario_buffer_location_t){
      .allocated_buffer =
          iree_hal_buffer_allocated_buffer(value->buffer.buffer),
      .byte_offset = iree_hal_buffer_byte_offset(value->buffer.buffer) +
                     value->buffer.byte_offset,
  };
}

static bool loom_benchmark_scenario_buffer_locations_share_root(
    loom_benchmark_scenario_buffer_location_t lhs,
    loom_benchmark_scenario_buffer_location_t rhs) {
  if (lhs.traceable != rhs.traceable) {
    return false;
  }
  return lhs.traceable ? lhs.allocation_value_id == rhs.allocation_value_id
                       : lhs.allocated_buffer == rhs.allocated_buffer;
}

static bool loom_benchmark_scenario_buffer_relative_offsets_equal(
    loom_benchmark_scenario_buffer_location_t baseline_lhs,
    loom_benchmark_scenario_buffer_location_t baseline_rhs,
    loom_benchmark_scenario_buffer_location_t candidate_lhs,
    loom_benchmark_scenario_buffer_location_t candidate_rhs) {
  if (baseline_lhs.byte_offset >= baseline_rhs.byte_offset) {
    return candidate_lhs.byte_offset >= candidate_rhs.byte_offset &&
           baseline_lhs.byte_offset - baseline_rhs.byte_offset ==
               candidate_lhs.byte_offset - candidate_rhs.byte_offset;
  }
  return candidate_lhs.byte_offset < candidate_rhs.byte_offset &&
         baseline_rhs.byte_offset - baseline_lhs.byte_offset ==
             candidate_rhs.byte_offset - candidate_lhs.byte_offset;
}

static iree_status_t loom_benchmark_scenario_compare_value(
    const loom_testbench_value_t* baseline,
    const loom_testbench_value_t* candidate, iree_host_size_t ordinal) {
  if (baseline->kind != candidate->kind) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "value %" PRIhsz " kinds differ", ordinal);
  }
  switch ((loom_testbench_value_kind_t)baseline->kind) {
    case LOOM_TESTBENCH_VALUE_KIND_NONE:
      return iree_ok_status();
    case LOOM_TESTBENCH_VALUE_KIND_SCALAR: {
      if (baseline->scalar.kind != candidate->scalar.kind) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "value %" PRIhsz " scalar kinds differ",
                                ordinal);
      }
      uint32_t baseline_words[2] = {0};
      uint32_t candidate_words[2] = {0};
      iree_host_size_t baseline_word_count = 0;
      iree_host_size_t candidate_word_count = 0;
      IREE_RETURN_IF_ERROR(iree_tooling_value_write_abi_words(
          &baseline->scalar, IREE_ARRAYSIZE(baseline_words), baseline_words,
          &baseline_word_count));
      IREE_RETURN_IF_ERROR(iree_tooling_value_write_abi_words(
          &candidate->scalar, IREE_ARRAYSIZE(candidate_words), candidate_words,
          &candidate_word_count));
      if (baseline_word_count != candidate_word_count ||
          memcmp(baseline_words, candidate_words,
                 baseline_word_count * sizeof(baseline_words[0])) != 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "value %" PRIhsz " scalar payloads differ",
                                ordinal);
      }
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_VALUE_KIND_BUFFER: {
      if (baseline->buffer.kind != candidate->buffer.kind ||
          baseline->buffer.byte_length != candidate->buffer.byte_length ||
          !loom_benchmark_scenario_buffer_view_metadata_equal(
              baseline->buffer.buffer_view, candidate->buffer.buffer_view)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "value %" PRIhsz " buffer layouts differ",
                                ordinal);
      }
      return loom_benchmark_scenario_compare_buffer_contents(
          &baseline->buffer, &candidate->buffer, ordinal);
    }
    case LOOM_TESTBENCH_VALUE_KIND_ENTROPY:
      if (baseline->entropy.low != candidate->entropy.low ||
          baseline->entropy.high != candidate->entropy.high) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "value %" PRIhsz " entropy payloads differ",
                                ordinal);
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "value %" PRIhsz " has unsupported kind %u",
                              ordinal, (unsigned)baseline->kind);
  }
}

static const loom_testbench_value_t* loom_benchmark_scenario_call_value(
    const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_product_call_t* call, iree_host_size_t ordinal) {
  return ordinal < invocation->workload_count
             ? &call->call_parameters[ordinal]
             : &call->arguments[ordinal - invocation->workload_count];
}

static iree_status_t loom_benchmark_scenario_compare_call_values(
    const loom_testbench_invocation_plan_t* baseline_invocation,
    const loom_testbench_product_call_t* baseline_call,
    const loom_testbench_invocation_plan_t* candidate_invocation,
    const loom_testbench_product_call_t* candidate_call) {
  const iree_host_size_t value_count =
      baseline_invocation->workload_count + baseline_invocation->input_count;
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_benchmark_scenario_compare_value(
        loom_benchmark_scenario_call_value(baseline_invocation, baseline_call,
                                           i),
        loom_benchmark_scenario_call_value(candidate_invocation, candidate_call,
                                           i),
        i));
  }
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    const loom_testbench_value_t* baseline_i =
        loom_benchmark_scenario_call_value(baseline_invocation, baseline_call,
                                           i);
    if (!loom_testbench_value_is_buffer(baseline_i)) {
      continue;
    }
    const loom_testbench_value_t* candidate_i =
        loom_benchmark_scenario_call_value(candidate_invocation, candidate_call,
                                           i);
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_testbench_value_t* baseline_j =
          loom_benchmark_scenario_call_value(baseline_invocation, baseline_call,
                                             j);
      if (!loom_testbench_value_is_buffer(baseline_j)) {
        continue;
      }
      const loom_testbench_value_t* candidate_j =
          loom_benchmark_scenario_call_value(candidate_invocation,
                                             candidate_call, j);
      const bool baseline_aliases =
          loom_benchmark_scenario_buffer_locations_share_root(
              loom_benchmark_scenario_buffer_location(baseline_i),
              loom_benchmark_scenario_buffer_location(baseline_j));
      const bool candidate_aliases =
          loom_benchmark_scenario_buffer_locations_share_root(
              loom_benchmark_scenario_buffer_location(candidate_i),
              loom_benchmark_scenario_buffer_location(candidate_j));
      if (baseline_aliases != candidate_aliases) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "values %" PRIhsz " and %" PRIhsz
                                " have different buffer alias relationships",
                                j, i);
      }
      if (baseline_aliases &&
          !loom_benchmark_scenario_buffer_relative_offsets_equal(
              loom_benchmark_scenario_buffer_location(baseline_i),
              loom_benchmark_scenario_buffer_location(baseline_j),
              loom_benchmark_scenario_buffer_location(candidate_i),
              loom_benchmark_scenario_buffer_location(candidate_j))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "values %" PRIhsz " and %" PRIhsz
                                " have different relative buffer offsets",
                                j, i);
      }
    }
  }
  return iree_ok_status();
}

typedef struct loom_benchmark_scenario_candidate_compare_t {
  loom_benchmark_scenario_execution_t* candidate;
  const loom_testbench_invocation_plan_t* baseline_invocation;
  iree_host_size_t baseline_call_count;
  const loom_testbench_product_call_t* baseline_calls;
} loom_benchmark_scenario_candidate_compare_t;

static iree_status_t loom_benchmark_scenario_compare_candidate_call(
    void* user_data, const loom_testbench_prepared_product_t* product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  (void)product;
  loom_benchmark_scenario_candidate_compare_t* compare =
      (loom_benchmark_scenario_candidate_compare_t*)user_data;
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_compare_invocation_layout(
      compare->baseline_invocation, invocation));
  if (call_count != compare->baseline_call_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "physical input ring counts differ");
  }
  for (iree_host_size_t i = 0; i < call_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_benchmark_scenario_compare_call_values(
        compare->baseline_invocation, &compare->baseline_calls[i], invocation,
        &calls[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_benchmark_scenario_compare_baseline_call(
    void* user_data, const loom_testbench_prepared_product_t* product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  (void)product;
  loom_benchmark_scenario_candidate_compare_t* compare =
      (loom_benchmark_scenario_candidate_compare_t*)user_data;
  compare->baseline_invocation = invocation;
  compare->baseline_call_count = call_count;
  compare->baseline_calls = calls;
  return loom_testbench_scenario_trial_with_materialized_calls(
      &compare->candidate->trial_executor,
      compare->candidate->prepared_coordinate.trial_ordinal,
      compare->candidate->prepared_ring_count,
      (loom_testbench_scenario_trial_calls_callback_t){
          .fn = loom_benchmark_scenario_compare_candidate_call,
          .user_data = compare,
      });
}

iree_status_t loom_benchmark_scenario_execution_validate_equivalent(
    loom_benchmark_scenario_execution_t* baseline,
    loom_benchmark_scenario_execution_t* candidate) {
  const iree_string_view_t baseline_name =
      baseline->prepared_selection->benchmark_plan->name;
  const iree_string_view_t candidate_name =
      candidate->prepared_selection->benchmark_plan->name;
  if (baseline->prepared_layout.binding_count !=
          candidate->prepared_layout.binding_count ||
      baseline->prepared_layout.binding_set_bytes !=
          candidate->prepared_layout.binding_set_bytes ||
      baseline->prepared_layout.binding_range_count !=
          candidate->prepared_layout.binding_range_count ||
      baseline->prepared_layout.binding_range_set_bytes !=
          candidate->prepared_layout.binding_range_set_bytes ||
      baseline->prepared_layout.materialized_storage_count !=
          candidate->prepared_layout.materialized_storage_count ||
      baseline->prepared_layout.materialized_storage_set_bytes !=
          candidate->prepared_layout.materialized_storage_set_bytes ||
      baseline->prepared_ring_count != candidate->prepared_ring_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "comparison candidate `%.*s` HAL binding layout differs from baseline "
        "`%.*s`",
        (int)candidate_name.size, candidate_name.data, (int)baseline_name.size,
        baseline_name.data);
  }

  loom_benchmark_scenario_candidate_compare_t compare = {
      .candidate = candidate,
  };
  iree_status_t status = loom_testbench_scenario_trial_with_materialized_calls(
      &baseline->trial_executor, baseline->prepared_coordinate.trial_ordinal,
      baseline->prepared_ring_count,
      (loom_testbench_scenario_trial_calls_callback_t){
          .fn = loom_benchmark_scenario_compare_baseline_call,
          .user_data = &compare,
      });
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        status,
        "comparison candidate `%.*s` materialized trial does not match "
        "baseline `%.*s`",
        (int)candidate_name.size, candidate_name.data, (int)baseline_name.size,
        baseline_name.data);
  }
  return status;
}

typedef struct loom_benchmark_scenario_measurement_t {
  // Complete HAL timing and profiling policy.
  const loom_run_hal_benchmark_options_t* options;
  // Host allocator used for replay metadata and timing samples.
  iree_allocator_t host_allocator;
  // HAL benchmark and physical binding-ring evidence.
  loom_run_hal_testbench_scenario_benchmark_result_t result;
} loom_benchmark_scenario_measurement_t;

static iree_status_t loom_benchmark_scenario_measure_calls(
    void* user_data, const loom_testbench_prepared_product_t* product,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  loom_benchmark_scenario_measurement_t* measurement =
      (loom_benchmark_scenario_measurement_t*)user_data;
  return loom_run_hal_testbench_benchmark_scenario_product(
      product, invocation, call_count, calls, measurement->options,
      measurement->host_allocator, &measurement->result);
}

iree_status_t loom_benchmark_scenario_execution_measure(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_benchmark_policy_t* policy,
    loom_benchmark_benchmark_result_t* out_result) {
  *out_result = (loom_benchmark_benchmark_result_t){0};
  const loom_benchmark_work_plan_execution_options_t* options =
      execution->options;
  const loom_benchmark_selected_benchmark_t* selection =
      execution->prepared_selection;
  const loom_testbench_scenario_sample_coordinate_t coordinate =
      execution->prepared_coordinate;
  const loom_benchmark_options_t* benchmark_options =
      options->benchmark_options;
  const iree_host_size_t ring_count = execution->prepared_ring_count;
  const loom_run_hal_testbench_scenario_input_layout_t layout =
      execution->prepared_layout;
  loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
  iree_string_builder_t profile_artifact_path;
  iree_status_t status = loom_benchmark_prepare_profile_artifact(
      options->run, &selection->identity, options->hal_context,
      (loom_benchmark_profile_replay_identity_t){
          .has_scenario_coordinate = true,
          .scenario_coordinate = coordinate,
      },
      options->host_allocator, &hal_options, &profile_artifact_path);

  loom_benchmark_scenario_measurement_t measurement = {
      .options = &hal_options,
      .host_allocator = options->host_allocator,
  };
  if (iree_status_is_ok(status)) {
    status = loom_testbench_scenario_trial_with_materialized_calls(
        &execution->trial_executor, coordinate.trial_ordinal, ring_count,
        (loom_testbench_scenario_trial_calls_callback_t){
            .fn = loom_benchmark_scenario_measure_calls,
            .user_data = &measurement,
        });
  }
  if (iree_status_is_ok(status) &&
      (measurement.result.binding_count_per_call != layout.binding_count ||
       measurement.result.binding_set_bytes != layout.binding_set_bytes ||
       measurement.result.binding_range_count_per_call !=
           layout.binding_range_count ||
       measurement.result.binding_range_set_bytes !=
           layout.binding_range_set_bytes ||
       measurement.result.materialized_storage_count_per_call !=
           layout.materialized_storage_count ||
       measurement.result.materialized_storage_set_bytes !=
           layout.materialized_storage_set_bytes)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario input layout changed between ring sizing and measurement");
  }
  uint64_t expected_binding_ring_bytes = 0;
  if (iree_status_is_ok(status) && layout.binding_set_bytes != 0 &&
      ring_count > UINT64_MAX / layout.binding_set_bytes) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "scenario binding ring byte count overflowed");
  } else if (iree_status_is_ok(status)) {
    expected_binding_ring_bytes =
        layout.binding_set_bytes * (uint64_t)ring_count;
  }
  if (iree_status_is_ok(status) &&
      measurement.result.binding_ring_bytes != expected_binding_ring_bytes) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario physical calls materialized different binding byte lengths");
  }
  uint64_t expected_binding_range_ring_bytes = 0;
  if (iree_status_is_ok(status) && layout.binding_range_set_bytes != 0 &&
      ring_count > UINT64_MAX / layout.binding_range_set_bytes) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "scenario binding range byte count overflowed");
  } else if (iree_status_is_ok(status)) {
    expected_binding_range_ring_bytes =
        layout.binding_range_set_bytes * (uint64_t)ring_count;
  }
  if (iree_status_is_ok(status) &&
      measurement.result.binding_range_ring_bytes !=
          expected_binding_range_ring_bytes) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario physical calls materialized different binding ranges");
  }
  uint64_t expected_storage_ring_bytes = 0;
  if (iree_status_is_ok(status) && layout.materialized_storage_set_bytes != 0 &&
      ring_count > UINT64_MAX / layout.materialized_storage_set_bytes) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "scenario storage ring byte count overflowed");
  } else if (iree_status_is_ok(status)) {
    expected_storage_ring_bytes =
        layout.materialized_storage_set_bytes * (uint64_t)ring_count;
  }
  if (iree_status_is_ok(status) &&
      measurement.result.materialized_storage_ring_bytes !=
          expected_storage_ring_bytes) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "scenario physical calls materialized different storage byte lengths");
  }

  const loom_run_hal_testbench_actual_provider_t* provider =
      loom_run_hal_testbench_scenario_product_provider(
          &execution->prepared.trials[coordinate.trial_index].target);
  iree_host_size_t dispatches_per_batch = 0;
  if (iree_status_is_ok(status)) {
    if (!iree_host_size_checked_mul(hal_options.timing.batch_size,
                                    measurement.result.dispatch_count_per_call,
                                    &dispatches_per_batch)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "scenario physical dispatch count overflowed host size limits");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_result = (loom_benchmark_benchmark_result_t){
        .compile_report = provider ? provider->artifacts.compile_report : NULL,
        .executed = true,
        .passed = true,
        .has_benchmark_sample_ordinal = true,
        .benchmark_sample_ordinal =
            execution->prepared_benchmark_sample_ordinal,
        .has_scenario_coordinate = true,
        .scenario_coordinate = coordinate,
        .samples_per_iteration = 1,
        .has_hal_benchmark = true,
        .hal_benchmark = measurement.result.benchmark,
        .data_cache =
            {
                .populated = true,
                .correctness_materialization =
                    benchmark_options->scenario_correctness
                        ? LOOM_BENCHMARK_BUFFER_MATERIALIZATION_HOST_VISIBLE
                        : LOOM_BENCHMARK_BUFFER_MATERIALIZATION_UNKNOWN,
                .measurement_materialization =
                    LOOM_BENCHMARK_BUFFER_MATERIALIZATION_DEVICE_LOCAL,
                .binding_count = measurement.result.binding_count_per_call,
                .binding_ring_count = ring_count,
                .command_buffer_ring_count =
                    measurement.result.benchmark.command_buffer_ring_count,
                .dispatches_per_batch = dispatches_per_batch,
                .requested_min_ring_bytes =
                    (uint64_t)benchmark_options->input_ring_min_bytes,
                .binding_set_bytes = measurement.result.binding_set_bytes,
                .binding_ring_bytes = measurement.result.binding_ring_bytes,
                .binding_range_count =
                    measurement.result.binding_range_count_per_call,
                .binding_range_set_bytes =
                    measurement.result.binding_range_set_bytes,
                .binding_range_ring_bytes =
                    measurement.result.binding_range_ring_bytes,
                .materialized_storage_count =
                    measurement.result.materialized_storage_count_per_call,
                .materialized_storage_set_bytes =
                    measurement.result.materialized_storage_set_bytes,
                .materialized_storage_ring_bytes =
                    measurement.result.materialized_storage_ring_bytes,
            },
    };
  }
  iree_string_builder_deinitialize(&profile_artifact_path);
  return status;
}

iree_status_t loom_benchmark_run_scenario_work_item(
    loom_benchmark_scenario_execution_t* execution,
    const loom_benchmark_work_item_t* work_item,
    iree_host_size_t* inout_correctness_sample_count,
    iree_host_size_t* inout_correctness_failed_sample_count,
    iree_host_size_t* inout_failed_benchmark_count) {
  const loom_benchmark_work_plan_execution_options_t* options =
      execution->options;
  const loom_benchmark_selected_benchmark_t* selection =
      &options->work_plan
           ->selected_benchmarks[work_item->representative_selection_index];
  bool runnable = false;
  IREE_RETURN_IF_ERROR(loom_benchmark_scenario_execution_prepare(
      execution, selection, work_item->work_item_index,
      work_item->begin_benchmark_sample, work_item->scenario_coordinate,
      &runnable));
  const iree_host_size_t correctness_sample_count =
      options->benchmark_options->scenario_correctness ? 1 : 0;
  *inout_correctness_sample_count += correctness_sample_count;
  *inout_correctness_failed_sample_count += !runnable;
  if (!execution->device_emitted) {
    IREE_RETURN_IF_ERROR(loom_benchmark_event_sink_emit_device(
        options->event_sink, options->run, options->hal_context));
    execution->device_emitted = true;
  }

  loom_benchmark_benchmark_result_t benchmark_result = {
      .has_benchmark_sample_ordinal = true,
      .benchmark_sample_ordinal = work_item->begin_benchmark_sample,
      .has_scenario_coordinate = true,
      .scenario_coordinate = work_item->scenario_coordinate,
      .samples_per_iteration = 1,
      .failed_sample_count = !runnable,
  };
  if (runnable) {
    IREE_RETURN_IF_ERROR(loom_benchmark_scenario_execution_measure(
        execution, &selection->policy, &benchmark_result));
  }
  return loom_benchmark_emit_work_item_result_aliases(
      options->run, options->module_plan, options->work_plan, work_item,
      &benchmark_result, correctness_sample_count,
      /*correctness_failed_sample_count=*/!runnable, options->event_sink,
      inout_failed_benchmark_count);
}
