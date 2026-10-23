// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/dispatch_benchmark.h"

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tools/loom-benchmark/case_execution.h"
#include "loom/tools/loom-benchmark/input_ring.h"
#include "loom/tools/loom-benchmark/profile_report.h"

typedef struct loom_benchmark_hal_case_sample_preparation_t {
  // Materializer options used to create each independent ring slot.
  loom_testbench_value_materializer_options_t materializer_options;
  // Provider-resolved source-order invocation schedule.
  loom_testbench_prepared_case_t prepared_case;
  // Reusable table containing the current materialized sample.
  loom_testbench_value_table_t value_table;
  // Reusable scratch used to execute pre-dispatch invocations.
  loom_testbench_invocation_executor_t invocation_executor;
  // Number of complete provider spans before the first kernel launch.
  iree_host_size_t preparation_span_count;
} loom_benchmark_hal_case_sample_preparation_t;

static iree_status_t loom_benchmark_hal_case_sample_preparation_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    loom_benchmark_hal_case_sample_preparation_t* out_preparation) {
  *out_preparation = (loom_benchmark_hal_case_sample_preparation_t){
      .materializer_options = execution_options->materializer,
  };

  const loom_testbench_case_plan_t* case_plan =
      &module_plan->cases[benchmark_plan->case_index];
  const iree_host_size_t first_kernel_index =
      (iree_host_size_t)(case_plan->kernel_launches[0] -
                         case_plan->invocations);
  iree_status_t status = loom_testbench_prepare_case_execution(
      execution_options, module_plan, benchmark_plan->case_index, arena,
      &out_preparation->prepared_case);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_value_table_initialize_case(
        module_plan->module, case_plan, allocator,
        &out_preparation->value_table);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_invocation_executor_initialize(
        &out_preparation->prepared_case.invocation_schedule, allocator,
        &out_preparation->invocation_executor);
  }
  const loom_testbench_invocation_schedule_t* schedule =
      &out_preparation->prepared_case.invocation_schedule;
  for (iree_host_size_t span_index = 0;
       iree_status_is_ok(status) && span_index < schedule->span_count;
       ++span_index) {
    const loom_testbench_prepared_invocation_span_t* span =
        &schedule->spans[span_index];
    const iree_host_size_t first_span_invocation =
        (iree_host_size_t)(span->invocations[0].plan - case_plan->invocations);
    if (first_span_invocation >= first_kernel_index) {
      break;
    }
    IREE_ASSERT(first_span_invocation + span->invocation_count <=
                first_kernel_index);
    ++out_preparation->preparation_span_count;
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_invocation_executor_deinitialize(
        &out_preparation->invocation_executor);
    loom_testbench_value_table_deinitialize(&out_preparation->value_table);
    *out_preparation = (loom_benchmark_hal_case_sample_preparation_t){0};
  }
  return status;
}

static void loom_benchmark_hal_case_sample_preparation_deinitialize(
    loom_benchmark_hal_case_sample_preparation_t* preparation) {
  loom_testbench_invocation_executor_deinitialize(
      &preparation->invocation_executor);
  loom_testbench_value_table_deinitialize(&preparation->value_table);
  *preparation = (loom_benchmark_hal_case_sample_preparation_t){0};
}

static iree_status_t loom_benchmark_hal_case_sample_prepare(
    loom_benchmark_hal_case_sample_preparation_t* preparation,
    loom_testbench_case_sample_t sample, iree_host_size_t execution_index) {
  const loom_testbench_case_plan_t* case_plan =
      preparation->prepared_case.case_plan;
  loom_testbench_value_table_reset(&preparation->value_table);
  loom_testbench_invocation_executor_reset_issues(
      &preparation->invocation_executor);
  iree_status_t status = loom_testbench_materialize_case_sample(
      &preparation->materializer_options, case_plan, sample,
      &preparation->value_table);
  for (iree_host_size_t span_index = 0;
       iree_status_is_ok(status) &&
       preparation->invocation_executor.issue_count == 0 &&
       span_index < preparation->preparation_span_count;
       ++span_index) {
    status = loom_testbench_run_prepared_invocation_span(
        &preparation->invocation_executor, execution_index,
        &preparation->prepared_case.invocation_schedule.spans[span_index],
        &preparation->value_table);
  }
  if (iree_status_is_ok(status) &&
      preparation->invocation_executor.issue_count != 0) {
    const loom_testbench_sample_issue_t* issue =
        &preparation->invocation_executor.issues[0];
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pre-dispatch invocation rejected sample preparation during %.*s "
        "(%.*s): %.*s",
        (int)issue->stage.size, issue->stage.data, (int)issue->kind.size,
        issue->kind.data, (int)issue->message.size, issue->message.data);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        status,
        "preparing dispatch_complete check.case '@%.*s' sample %" PRIhsz,
        (int)case_plan->name.size, case_plan->name.data, sample.ordinal);
  }
  return status;
}

typedef struct loom_benchmark_hal_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans materialized from check ops.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed binding lists into |plans| for HAL benchmark setup.
  loom_run_hal_binding_list_t* binding_lists;
  // Number of entries in |plans| and |binding_lists|.
  iree_host_size_t plan_count;
  // Data/cache summary derived while materializing the ring.
  loom_benchmark_data_cache_summary_t summary;
} loom_benchmark_hal_input_ring_t;

static iree_status_t loom_benchmark_prepare_hal_invocation_plan_for_sample(
    loom_benchmark_hal_case_sample_preparation_t* preparation,
    loom_benchmark_hal_actual_provider_t* provider,
    loom_testbench_case_sample_t sample, iree_host_size_t execution_index,
    iree_allocator_t allocator, loom_run_hal_invocation_plan_t* out_plan,
    loom_testbench_buffer_footprint_t* out_input_footprint) {
  loom_run_hal_invocation_options_t invocation_options = {0};
  loom_run_hal_binding_list_t bindings = {0};
  iree_status_t status = loom_benchmark_hal_case_sample_prepare(
      preparation, sample, execution_index);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_value_table_resolve_buffer_footprint(
        &preparation->value_table, provider->execution.invocation->input_count,
        provider->execution.invocation->input_value_ids, out_input_footprint);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_materialize_invocation_from_table(
        &preparation->value_table, &provider->execution, allocator,
        &invocation_options, &bindings);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_prepare_from_lists(
        &invocation_options, &bindings, /*expected_bindings=*/NULL,
        /*max_output_element_count=*/0, allocator, out_plan);
  }
  loom_run_hal_binding_list_deinitialize(&bindings);
  return status;
}

static iree_status_t loom_benchmark_hal_input_ring_initialize(
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_options_t* options,
    loom_benchmark_hal_case_sample_preparation_t* preparation,
    loom_benchmark_hal_actual_provider_t* provider,
    loom_testbench_case_sample_t sample, iree_allocator_t allocator,
    loom_benchmark_hal_input_ring_t* out_ring) {
  *out_ring = (loom_benchmark_hal_input_ring_t){
      .host_allocator = allocator,
  };

  loom_run_hal_invocation_plan_t first_plan = {0};
  loom_testbench_buffer_footprint_t first_input_footprint = {0};
  iree_status_t status = loom_benchmark_prepare_hal_invocation_plan_for_sample(
      preparation, provider, sample, /*execution_index=*/0, allocator,
      &first_plan, &first_input_footprint);
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_binding_list_total_byte_length(
        &first_plan.bindings, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    ring_count = loom_benchmark_input_ring_select_count(
        options->input_ring_count, (uint64_t)options->input_ring_min_bytes,
        first_input_footprint.binding_range_byte_length,
        policy->hal_options.timing.batch_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, ring_count,
                                         sizeof(*out_ring->plans),
                                         (void**)&out_ring->plans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, ring_count,
                                         sizeof(*out_ring->binding_lists),
                                         (void**)&out_ring->binding_lists);
  }
  if (iree_status_is_ok(status)) {
    memset(out_ring->plans, 0, ring_count * sizeof(*out_ring->plans));
    out_ring->plan_count = ring_count;
    out_ring->plans[0] = first_plan;
    first_plan = (loom_run_hal_invocation_plan_t){0};
    out_ring->binding_lists[0] = out_ring->plans[0].bindings;
    out_ring->summary = (loom_benchmark_data_cache_summary_t){
        .populated = true,
        .correctness_materialization =
            LOOM_BENCHMARK_BUFFER_MATERIALIZATION_HOST_VISIBLE,
        .measurement_materialization =
            LOOM_BENCHMARK_BUFFER_MATERIALIZATION_DEVICE_LOCAL,
        .binding_count = out_ring->plans[0].bindings.count,
        .binding_ring_count = ring_count,
        .dispatches_per_batch = policy->hal_options.timing.batch_size,
        .requested_min_ring_bytes = (uint64_t)options->input_ring_min_bytes,
        .binding_set_bytes = first_binding_set_bytes,
        .binding_ring_bytes = first_binding_set_bytes,
        .binding_range_count = first_input_footprint.binding_range_count,
        .binding_range_set_bytes =
            first_input_footprint.binding_range_byte_length,
        .binding_range_ring_bytes =
            first_input_footprint.binding_range_byte_length,
        .materialized_storage_count =
            first_input_footprint.storage_allocation_count,
        .materialized_storage_set_bytes =
            first_input_footprint.storage_byte_length,
        .materialized_storage_ring_bytes =
            first_input_footprint.storage_byte_length,
    };
  }
  for (iree_host_size_t i = 1; iree_status_is_ok(status) && i < ring_count;
       ++i) {
    loom_testbench_buffer_footprint_t input_footprint = {0};
    status = loom_benchmark_prepare_hal_invocation_plan_for_sample(
        preparation, provider, sample, /*execution_index=*/i, allocator,
        &out_ring->plans[i], &input_footprint);
    if (iree_status_is_ok(status) &&
        !loom_run_hal_invocation_options_equal(&out_ring->plans[0].options,
                                               &out_ring->plans[i].options)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL benchmark input ring materialization changed dispatch constants "
          "or geometry for selected sweep sample %" PRIhsz,
          sample.ordinal);
    }
    if (iree_status_is_ok(status) &&
        !loom_testbench_buffer_footprint_equal(&input_footprint,
                                               &first_input_footprint)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL benchmark input ring materialization changed input buffer "
          "footprint for selected sweep sample %" PRIhsz,
          sample.ordinal);
    }
    uint64_t binding_set_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_binding_list_total_byte_length(
          &out_ring->plans[i].bindings, &binding_set_bytes);
    }
    if (iree_status_is_ok(status)) {
      if (UINT64_MAX - out_ring->summary.binding_ring_bytes <
              binding_set_bytes ||
          UINT64_MAX - out_ring->summary.binding_range_ring_bytes <
              input_footprint.binding_range_byte_length ||
          UINT64_MAX - out_ring->summary.materialized_storage_ring_bytes <
              input_footprint.storage_byte_length) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "HAL benchmark input ring byte accounting "
                                  "overflowed uint64");
      } else {
        out_ring->binding_lists[i] = out_ring->plans[i].bindings;
        out_ring->summary.binding_ring_bytes += binding_set_bytes;
        out_ring->summary.binding_range_ring_bytes +=
            input_footprint.binding_range_byte_length;
        out_ring->summary.materialized_storage_ring_bytes +=
            input_footprint.storage_byte_length;
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < out_ring->plan_count; ++i) {
      loom_run_hal_invocation_plan_deinitialize(&out_ring->plans[i]);
    }
    iree_allocator_free(out_ring->host_allocator, out_ring->binding_lists);
    iree_allocator_free(out_ring->host_allocator, out_ring->plans);
    *out_ring = (loom_benchmark_hal_input_ring_t){0};
  }
  loom_run_hal_invocation_plan_deinitialize(&first_plan);
  return status;
}

static void loom_benchmark_hal_input_ring_deinitialize(
    loom_benchmark_hal_input_ring_t* ring) {
  for (iree_host_size_t i = 0; i < ring->plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  iree_allocator_free(ring->host_allocator, ring->binding_lists);
  iree_allocator_free(ring->host_allocator, ring->plans);
  *ring = (loom_benchmark_hal_input_ring_t){0};
}

typedef struct loom_benchmark_hal_sequence_input_ring_t {
  // Host allocator used for ring-owned arrays.
  iree_allocator_t host_allocator;
  // Ring-owned invocation plans in ring-slot, sequence-step order.
  loom_run_hal_invocation_plan_t* plans;
  // Borrowed plan pointers into |plans| for HAL benchmark setup.
  const loom_run_hal_invocation_plan_t** plan_ptrs;
  // Borrowed prepared candidates in sequence-step order.
  const loom_run_hal_prepared_candidate_t** candidates;
  // Execution epochs in sequence-step order.
  iree_host_size_t* execution_epochs;
  // Number of logical ring slots in |plans|.
  iree_host_size_t plan_ring_count;
  // Number of actual dispatch invocations per logical ring slot.
  iree_host_size_t sequence_count;
  // Data/cache summary derived while materializing the ring.
  loom_benchmark_data_cache_summary_t summary;
} loom_benchmark_hal_sequence_input_ring_t;

static iree_host_size_t loom_benchmark_hal_sequence_input_ring_plan_count(
    const loom_benchmark_hal_sequence_input_ring_t* ring) {
  return ring->plan_ring_count * ring->sequence_count;
}

static iree_status_t loom_benchmark_hal_sequence_plans_total_byte_length(
    const loom_run_hal_invocation_plan_t* plans, iree_host_size_t plan_count,
    uint64_t* out_byte_length) {
  *out_byte_length = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    uint64_t plan_byte_length = 0;
    IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_total_byte_length(
        &plans[i].bindings, &plan_byte_length));
    if (UINT64_MAX - *out_byte_length < plan_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL benchmark sequence binding byte count "
                              "overflowed uint64");
    }
    *out_byte_length += plan_byte_length;
  }
  return iree_ok_status();
}

static iree_host_size_t loom_benchmark_hal_sequence_binding_count(
    const loom_run_hal_invocation_plan_t* plans, iree_host_size_t plan_count) {
  iree_host_size_t binding_count = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    binding_count += plans[i].bindings.count;
  }
  return binding_count;
}

static iree_status_t loom_benchmark_hal_sequence_resolve_input_footprint(
    const loom_testbench_value_table_t* table,
    const loom_benchmark_hal_actual_sequence_t* sequence,
    iree_allocator_t allocator,
    loom_testbench_buffer_footprint_t* out_footprint) {
  *out_footprint = (loom_testbench_buffer_footprint_t){0};
  iree_host_size_t input_count = 0;
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    if (!iree_host_size_checked_add(
            input_count,
            sequence->providers[i].execution.invocation->input_count,
            &input_count)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL benchmark sequence input count overflowed host size limits");
    }
  }

  loom_value_id_t* input_value_ids = NULL;
  if (input_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, input_count,
                                                     sizeof(*input_value_ids),
                                                     (void**)&input_value_ids));
  }
  iree_host_size_t input_offset = 0;
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        sequence->providers[i].execution.invocation;
    if (invocation->input_count != 0) {
      memcpy(&input_value_ids[input_offset], invocation->input_value_ids,
             invocation->input_count * sizeof(*input_value_ids));
      input_offset += invocation->input_count;
    }
  }
  IREE_ASSERT(input_offset == input_count);
  iree_status_t status = loom_testbench_value_table_resolve_buffer_footprint(
      table, input_count, input_value_ids, out_footprint);
  iree_allocator_free(allocator, input_value_ids);
  return status;
}

static iree_status_t loom_benchmark_prepare_hal_sequence_plans_for_sample(
    loom_benchmark_hal_case_sample_preparation_t* preparation,
    loom_benchmark_hal_actual_sequence_t* sequence,
    loom_testbench_case_sample_t sample, iree_host_size_t execution_index,
    iree_allocator_t allocator, loom_run_hal_invocation_plan_t* out_plans,
    loom_testbench_buffer_footprint_t* out_input_footprint) {
  *out_input_footprint = (loom_testbench_buffer_footprint_t){0};
  iree_status_t status = loom_benchmark_hal_case_sample_prepare(
      preparation, sample, execution_index);
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < sequence->provider_count; ++i) {
    loom_run_hal_testbench_actual_provider_t* provider =
        &sequence->providers[i].execution;
    loom_run_hal_invocation_options_t invocation_options = {0};
    loom_run_hal_binding_list_t bindings = {0};
    status = loom_run_hal_testbench_materialize_invocation_from_table(
        &preparation->value_table, provider, allocator, &invocation_options,
        &bindings);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_invocation_plan_prepare_from_lists(
          &invocation_options, &bindings, /*expected_bindings=*/NULL,
          /*max_output_element_count=*/0, allocator, &out_plans[i]);
    }
    loom_run_hal_binding_list_deinitialize(&bindings);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_hal_sequence_resolve_input_footprint(
        &preparation->value_table, sequence, allocator, out_input_footprint);
  }
  return status;
}

static iree_status_t loom_benchmark_hal_sequence_input_ring_initialize(
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_options_t* options,
    loom_benchmark_hal_case_sample_preparation_t* preparation,
    loom_benchmark_hal_actual_sequence_t* sequence,
    loom_testbench_case_sample_t sample, iree_allocator_t allocator,
    loom_benchmark_hal_sequence_input_ring_t* out_ring) {
  *out_ring = (loom_benchmark_hal_sequence_input_ring_t){
      .host_allocator = allocator,
      .sequence_count = sequence->provider_count,
  };

  loom_run_hal_invocation_plan_t* first_plans = NULL;
  loom_testbench_buffer_footprint_t first_input_footprint = {0};
  iree_host_size_t first_plan_count = 0;
  iree_status_t status =
      iree_allocator_malloc_array(allocator, sequence->provider_count,
                                  sizeof(*first_plans), (void**)&first_plans);
  if (iree_status_is_ok(status)) {
    memset(first_plans, 0, sequence->provider_count * sizeof(*first_plans));
    first_plan_count = sequence->provider_count;
    status = loom_benchmark_prepare_hal_sequence_plans_for_sample(
        preparation, sequence, sample, /*execution_index=*/0, allocator,
        first_plans, &first_input_footprint);
  }
  uint64_t first_binding_set_bytes = 0;
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_hal_sequence_plans_total_byte_length(
        first_plans, sequence->provider_count, &first_binding_set_bytes);
  }

  iree_host_size_t ring_count = 1;
  if (iree_status_is_ok(status)) {
    ring_count = loom_benchmark_input_ring_select_count(
        options->input_ring_count, (uint64_t)options->input_ring_min_bytes,
        first_input_footprint.binding_range_byte_length,
        policy->hal_options.timing.batch_size);
  }
  iree_host_size_t plan_count = 0;
  if (iree_status_is_ok(status) &&
      !iree_host_size_checked_mul(ring_count, sequence->provider_count,
                                  &plan_count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL benchmark sequence input ring plan count "
                              "overflowed host size limits");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan_count,
                                         sizeof(*out_ring->plans),
                                         (void**)&out_ring->plans);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, plan_count,
                                         sizeof(*out_ring->plan_ptrs),
                                         (void**)&out_ring->plan_ptrs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, sequence->provider_count,
                                         sizeof(*out_ring->candidates),
                                         (void**)&out_ring->candidates);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, sequence->provider_count,
                                         sizeof(*out_ring->execution_epochs),
                                         (void**)&out_ring->execution_epochs);
  }
  if (iree_status_is_ok(status)) {
    memset(out_ring->plans, 0, plan_count * sizeof(*out_ring->plans));
    out_ring->plan_ring_count = ring_count;
    for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
      out_ring->plans[i] = first_plans[i];
      first_plans[i] = (loom_run_hal_invocation_plan_t){0};
      out_ring->plan_ptrs[i] = &out_ring->plans[i];
      out_ring->candidates[i] =
          &sequence->providers[i].execution.prepared_candidate;
      out_ring->execution_epochs[i] =
          sequence->providers[i].execution.invocation->execution_epoch;
    }
    iree_host_size_t dispatches_per_batch = 0;
    if (!iree_host_size_checked_mul(policy->hal_options.timing.batch_size,
                                    sequence->provider_count,
                                    &dispatches_per_batch)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "HAL benchmark sequence dispatch count "
                                "overflowed host size limits");
    } else {
      out_ring->summary = (loom_benchmark_data_cache_summary_t){
          .populated = true,
          .correctness_materialization =
              LOOM_BENCHMARK_BUFFER_MATERIALIZATION_HOST_VISIBLE,
          .measurement_materialization =
              LOOM_BENCHMARK_BUFFER_MATERIALIZATION_DEVICE_LOCAL,
          .binding_count = loom_benchmark_hal_sequence_binding_count(
              out_ring->plans, sequence->provider_count),
          .binding_ring_count = ring_count,
          .dispatches_per_batch = dispatches_per_batch,
          .requested_min_ring_bytes = (uint64_t)options->input_ring_min_bytes,
          .binding_set_bytes = first_binding_set_bytes,
          .binding_ring_bytes = first_binding_set_bytes,
          .binding_range_count = first_input_footprint.binding_range_count,
          .binding_range_set_bytes =
              first_input_footprint.binding_range_byte_length,
          .binding_range_ring_bytes =
              first_input_footprint.binding_range_byte_length,
          .materialized_storage_count =
              first_input_footprint.storage_allocation_count,
          .materialized_storage_set_bytes =
              first_input_footprint.storage_byte_length,
          .materialized_storage_ring_bytes =
              first_input_footprint.storage_byte_length,
      };
    }
  }
  for (iree_host_size_t ring_index = 1;
       iree_status_is_ok(status) && ring_index < ring_count; ++ring_index) {
    loom_run_hal_invocation_plan_t* ring_plans =
        &out_ring->plans[ring_index * sequence->provider_count];
    loom_testbench_buffer_footprint_t input_footprint = {0};
    status = loom_benchmark_prepare_hal_sequence_plans_for_sample(
        preparation, sequence, sample, /*execution_index=*/ring_index,
        allocator, ring_plans, &input_footprint);
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence->provider_count;
         ++step_index) {
      if (!loom_run_hal_invocation_options_equal(
              &out_ring->plans[step_index].options,
              &ring_plans[step_index].options)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HAL benchmark sequence input ring materialization changed "
            "dispatch constants or geometry for selected sweep sample %" PRIhsz
            " step %" PRIhsz,
            sample.ordinal, step_index);
      }
    }
    if (iree_status_is_ok(status) &&
        !loom_testbench_buffer_footprint_equal(&input_footprint,
                                               &first_input_footprint)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL benchmark sequence input ring materialization changed input "
          "buffer footprint for selected sweep sample %" PRIhsz,
          sample.ordinal);
    }
    uint64_t binding_set_bytes = 0;
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_hal_sequence_plans_total_byte_length(
          ring_plans, sequence->provider_count, &binding_set_bytes);
    }
    if (iree_status_is_ok(status)) {
      if (UINT64_MAX - out_ring->summary.binding_ring_bytes <
              binding_set_bytes ||
          UINT64_MAX - out_ring->summary.binding_range_ring_bytes <
              input_footprint.binding_range_byte_length ||
          UINT64_MAX - out_ring->summary.materialized_storage_ring_bytes <
              input_footprint.storage_byte_length) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "HAL benchmark sequence input ring byte "
                                  "accounting overflowed uint64");
      } else {
        out_ring->summary.binding_ring_bytes += binding_set_bytes;
        out_ring->summary.binding_range_ring_bytes +=
            input_footprint.binding_range_byte_length;
        out_ring->summary.materialized_storage_ring_bytes +=
            input_footprint.storage_byte_length;
      }
    }
    for (iree_host_size_t step_index = 0;
         iree_status_is_ok(status) && step_index < sequence->provider_count;
         ++step_index) {
      const iree_host_size_t plan_index =
          ring_index * sequence->provider_count + step_index;
      out_ring->plan_ptrs[plan_index] = &out_ring->plans[plan_index];
    }
  }
  if (!iree_status_is_ok(status)) {
    const iree_host_size_t plan_count =
        loom_benchmark_hal_sequence_input_ring_plan_count(out_ring);
    for (iree_host_size_t i = 0; i < plan_count; ++i) {
      loom_run_hal_invocation_plan_deinitialize(&out_ring->plans[i]);
    }
    iree_allocator_free(out_ring->host_allocator, out_ring->execution_epochs);
    iree_allocator_free(out_ring->host_allocator, out_ring->candidates);
    iree_allocator_free(out_ring->host_allocator, out_ring->plan_ptrs);
    iree_allocator_free(out_ring->host_allocator, out_ring->plans);
    *out_ring = (loom_benchmark_hal_sequence_input_ring_t){0};
  }
  for (iree_host_size_t i = 0; i < first_plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&first_plans[i]);
  }
  iree_allocator_free(allocator, first_plans);
  return status;
}

static void loom_benchmark_hal_sequence_input_ring_deinitialize(
    loom_benchmark_hal_sequence_input_ring_t* ring) {
  const iree_host_size_t plan_count =
      loom_benchmark_hal_sequence_input_ring_plan_count(ring);
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    loom_run_hal_invocation_plan_deinitialize(&ring->plans[i]);
  }
  iree_allocator_free(ring->host_allocator, ring->execution_epochs);
  iree_allocator_free(ring->host_allocator, ring->candidates);
  iree_allocator_free(ring->host_allocator, ring->plan_ptrs);
  iree_allocator_free(ring->host_allocator, ring->plans);
  *ring = (loom_benchmark_hal_sequence_input_ring_t){0};
}

iree_status_t loom_benchmark_run_hal_benchmark_sample(
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_options_t* options,
    loom_benchmark_hal_actual_provider_t* provider,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    loom_benchmark_benchmark_result_t* out_result) {
  const loom_testbench_case_sample_t sample =
      loom_testbench_benchmark_case_sample(benchmark_plan,
                                           benchmark_sample_ordinal);
  memset(out_result, 0, sizeof(*out_result));
  out_result->has_benchmark_sample_ordinal = true;
  out_result->benchmark_sample_ordinal = benchmark_sample_ordinal;
  out_result->samples_per_iteration = 1;

  out_result->compile_report = provider->execution.artifacts.compile_report;
  out_result->compile_report_artifact_path =
      provider->compile_report_artifact_path;
  out_result->artifact_manifest_path = provider->artifact_manifest_path;
  out_result->target_artifact_path = provider->target_artifact_path;
  out_result->target_listing_path = provider->target_listing_path;
  out_result->hal_executable_path = provider->hal_executable_path;
  if (provider->execution.compile_rejected) {
    loom_benchmark_benchmark_result_set_compile_rejection(provider, out_result);
    out_result->has_benchmark_sample_ordinal = true;
    out_result->benchmark_sample_ordinal = benchmark_sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  loom_benchmark_hal_input_ring_t input_ring = {0};
  loom_benchmark_hal_case_sample_preparation_t preparation = {0};
  iree_status_t status = loom_benchmark_hal_case_sample_preparation_initialize(
      module_plan, benchmark_plan, execution_options, execution_arena,
      allocator, &preparation);
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_hal_input_ring_initialize(
        policy, options, &preparation, provider, sample, allocator,
        &input_ring);
  }
  loom_benchmark_hal_case_sample_preparation_deinitialize(&preparation);
  iree_arena_reset(execution_arena);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    status = loom_benchmark_prepare_profile_artifact(
        run, candidate, provider->context,
        (loom_benchmark_profile_replay_identity_t){
            .benchmark_sample_ordinal = benchmark_sample_ordinal,
        },
        allocator, &hal_options, &profile_artifact_path);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_benchmark_dispatch_binding_ring(
          &provider->context->execution.runtime,
          &provider->execution.prepared_candidate, &input_ring.plans[0],
          input_ring.plan_count, input_ring.binding_lists, &hal_options,
          allocator, &out_result->hal_benchmark);
    }
    if (iree_status_is_ok(status)) {
      out_result->data_cache.command_buffer_ring_count =
          out_result->hal_benchmark.command_buffer_ring_count;
    }
    iree_string_builder_deinitialize(&profile_artifact_path);
  }
  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = true;
  }
  loom_benchmark_hal_input_ring_deinitialize(&input_ring);
  return status;
}

iree_status_t loom_benchmark_run_hal_sequence_benchmark_sample(
    const loom_benchmark_run_identity_t* run,
    const loom_benchmark_candidate_identity_t* candidate,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_benchmark_benchmark_policy_t* policy,
    const loom_benchmark_options_t* options,
    loom_benchmark_hal_context_t* hal_context,
    loom_benchmark_hal_actual_sequence_t* sequence,
    const loom_testbench_case_execution_options_t* execution_options,
    iree_arena_allocator_t* execution_arena,
    iree_host_size_t benchmark_sample_ordinal, iree_allocator_t allocator,
    loom_benchmark_benchmark_result_t* out_result) {
  const loom_testbench_case_sample_t sample =
      loom_testbench_benchmark_case_sample(benchmark_plan,
                                           benchmark_sample_ordinal);
  memset(out_result, 0, sizeof(*out_result));
  out_result->has_benchmark_sample_ordinal = true;
  out_result->benchmark_sample_ordinal = benchmark_sample_ordinal;
  out_result->samples_per_iteration = 1;

  const loom_benchmark_hal_actual_provider_t* rejected_provider =
      loom_benchmark_hal_actual_sequence_first_rejection(sequence);
  if (rejected_provider != NULL) {
    loom_benchmark_benchmark_result_set_compile_rejection(rejected_provider,
                                                          out_result);
    out_result->has_benchmark_sample_ordinal = true;
    out_result->benchmark_sample_ordinal = benchmark_sample_ordinal;
    out_result->samples_per_iteration = 1;
    return iree_ok_status();
  }

  loom_benchmark_hal_sequence_input_ring_t input_ring = {0};
  loom_benchmark_hal_case_sample_preparation_t preparation = {0};
  iree_status_t status = loom_benchmark_hal_case_sample_preparation_initialize(
      module_plan, benchmark_plan, execution_options, execution_arena,
      allocator, &preparation);
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_hal_sequence_input_ring_initialize(
        policy, options, &preparation, sequence, sample, allocator,
        &input_ring);
  }
  loom_benchmark_hal_case_sample_preparation_deinitialize(&preparation);
  iree_arena_reset(execution_arena);
  if (iree_status_is_ok(status)) {
    out_result->has_hal_benchmark = true;
    out_result->data_cache = input_ring.summary;
    loom_run_hal_benchmark_options_t hal_options = policy->hal_options;
    iree_string_builder_t profile_artifact_path;
    status = loom_benchmark_prepare_profile_artifact(
        run, candidate, hal_context,
        (loom_benchmark_profile_replay_identity_t){
            .benchmark_sample_ordinal = benchmark_sample_ordinal,
        },
        allocator, &hal_options, &profile_artifact_path);
    if (iree_status_is_ok(status)) {
      status = loom_run_hal_benchmark_dispatch_sequence_plan_ring(
          &hal_context->execution.runtime, sequence->provider_count,
          input_ring.candidates, input_ring.execution_epochs,
          input_ring.plan_ring_count, input_ring.plan_ptrs, &hal_options,
          allocator, &out_result->hal_benchmark);
    }
    if (iree_status_is_ok(status)) {
      out_result->data_cache.command_buffer_ring_count =
          out_result->hal_benchmark.command_buffer_ring_count;
    }
    iree_string_builder_deinitialize(&profile_artifact_path);
  }
  if (iree_status_is_ok(status)) {
    out_result->executed = true;
    out_result->passed = true;
  }
  loom_benchmark_hal_sequence_input_ring_deinitialize(&input_ring);
  return status;
}
