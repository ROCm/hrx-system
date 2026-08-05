// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/invocation.h"

#include <string.h>

void loom_testbench_invocation_options_initialize(
    loom_testbench_invocation_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->oracle_providers = loom_testbench_oracle_provider_list_empty();
}

static bool loom_testbench_find_oracle_provider(
    const loom_testbench_invocation_options_t* options, iree_string_view_t name,
    loom_testbench_invocation_provider_t* out_provider) {
  for (iree_host_size_t provider_index = 0;
       provider_index < options->oracle_providers.count; ++provider_index) {
    const loom_testbench_oracle_provider_t* provider =
        &options->oracle_providers.values[provider_index];
    if (iree_string_view_equal(provider->name, name)) {
      *out_provider = provider->provider;
      return true;
    }
  }
  memset(out_provider, 0, sizeof(*out_provider));
  return false;
}

static bool loom_testbench_invocation_providers_equal(
    const loom_testbench_invocation_provider_t* lhs,
    const loom_testbench_invocation_provider_t* rhs) {
  return lhs->invoke == rhs->invoke &&
         lhs->invoke_sequence == rhs->invoke_sequence &&
         lhs->query_issue == rhs->query_issue &&
         lhs->user_data == rhs->user_data;
}

iree_status_t loom_testbench_prepare_case_invocations(
    const loom_testbench_invocation_options_t* options,
    const loom_testbench_case_plan_t* case_plan, iree_arena_allocator_t* arena,
    loom_testbench_invocation_schedule_t* out_schedule) {
  memset(out_schedule, 0, sizeof(*out_schedule));

  loom_testbench_prepared_invocation_t* prepared_invocations = NULL;
  if (case_plan->invocation_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, case_plan->invocation_count, sizeof(*prepared_invocations),
        (void**)&prepared_invocations));
    memset(prepared_invocations, 0,
           case_plan->invocation_count * sizeof(*prepared_invocations));
  }

  loom_testbench_prepared_invocation_span_t* spans = NULL;
  if (case_plan->invocation_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, case_plan->invocation_count, sizeof(*spans), (void**)&spans));
    memset(spans, 0, case_plan->invocation_count * sizeof(*spans));
  }

  iree_host_size_t max_workload_count = 0;
  iree_host_size_t max_input_count = 0;
  iree_host_size_t max_result_count = 0;
  for (iree_host_size_t invocation_index = 0;
       invocation_index < case_plan->invocation_count; ++invocation_index) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[invocation_index];
    loom_testbench_invocation_provider_t provider = {0};
    switch (invocation->kind) {
      case LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL:
        provider = options->function_call;
        if (!provider.invoke) {
          return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "no function call provider is configured");
        }
        break;
      case LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH:
        provider = options->kernel_launch;
        if (!provider.invoke) {
          return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "no kernel launch provider is configured");
        }
        break;
      case LOOM_TESTBENCH_INVOCATION_ORACLE:
        if (!loom_testbench_find_oracle_provider(options, invocation->provider,
                                                 &provider)) {
          return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "oracle provider `%.*s` is not configured",
                                  (int)invocation->provider.size,
                                  invocation->provider.data);
        }
        if (!provider.invoke) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "oracle provider `%.*s` has no callback",
                                  (int)invocation->provider.size,
                                  invocation->provider.data);
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "invalid invocation plan kind %u",
                                (unsigned)invocation->kind);
    }

    prepared_invocations[invocation_index].plan = invocation;
    prepared_invocations[invocation_index].provider = provider;
    max_workload_count =
        iree_max(max_workload_count, invocation->workload_count);
    max_input_count = iree_max(max_input_count, invocation->input_count);
    max_result_count = iree_max(max_result_count, invocation->result_count);
  }

  iree_host_size_t span_count = 0;
  for (iree_host_size_t begin_index = 0;
       begin_index < case_plan->invocation_count;) {
    const loom_testbench_invocation_provider_t* provider =
        &prepared_invocations[begin_index].provider;
    iree_host_size_t end_index = begin_index + 1;
    if (provider->invoke_sequence != NULL) {
      while (end_index < case_plan->invocation_count &&
             loom_testbench_invocation_providers_equal(
                 provider, &prepared_invocations[end_index].provider)) {
        ++end_index;
      }
    }
    spans[span_count++] = (loom_testbench_prepared_invocation_span_t){
        .invocations = &prepared_invocations[begin_index],
        .invocation_count = end_index - begin_index,
    };
    begin_index = end_index;
  }

  out_schedule->invocations = prepared_invocations;
  out_schedule->invocation_count = case_plan->invocation_count;
  out_schedule->spans = spans;
  out_schedule->span_count = span_count;
  out_schedule->max_workload_count = max_workload_count;
  out_schedule->max_input_count = max_input_count;
  out_schedule->max_result_count = max_result_count;
  return iree_ok_status();
}

static iree_status_t loom_testbench_allocate_value_array(
    iree_allocator_t allocator, iree_host_size_t count,
    loom_testbench_value_t** out_values) {
  *out_values = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, count, sizeof(**out_values), (void**)out_values));
  memset(*out_values, 0, count * sizeof(**out_values));
  return iree_ok_status();
}

static void loom_testbench_reset_values(loom_testbench_value_t* values,
                                        iree_host_size_t count) {
  for (iree_host_size_t value_index = 0; value_index < count; ++value_index) {
    loom_testbench_value_deinitialize(&values[value_index]);
  }
}

iree_status_t loom_testbench_invocation_executor_initialize(
    const loom_testbench_invocation_schedule_t* schedule,
    iree_allocator_t host_allocator,
    loom_testbench_invocation_executor_t* out_executor) {
  memset(out_executor, 0, sizeof(*out_executor));
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  out_executor->schedule = schedule;
  out_executor->host_allocator = host_allocator;
  out_executor->workload_capacity = schedule->max_workload_count;
  out_executor->input_capacity = schedule->max_input_count;
  out_executor->result_capacity = schedule->max_result_count;

  iree_status_t status = loom_testbench_allocate_value_array(
      host_allocator, out_executor->workload_capacity,
      &out_executor->workloads);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_allocate_value_array(
        host_allocator, out_executor->input_capacity, &out_executor->inputs);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_allocate_value_array(
        host_allocator, out_executor->result_capacity, &out_executor->results);
  }
  if (iree_status_is_ok(status) && schedule->invocation_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, schedule->invocation_count,
        sizeof(*out_executor->issues), (void**)&out_executor->issues);
    if (iree_status_is_ok(status)) {
      out_executor->issue_capacity = schedule->invocation_count;
      memset(out_executor->issues, 0,
             out_executor->issue_capacity * sizeof(*out_executor->issues));
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_testbench_invocation_executor_deinitialize(out_executor);
    return status;
  }
  return iree_ok_status();
}

void loom_testbench_invocation_executor_deinitialize(
    loom_testbench_invocation_executor_t* executor) {
  if (!executor) {
    return;
  }
  if (executor->workloads) {
    loom_testbench_reset_values(executor->workloads,
                                executor->workload_capacity);
    iree_allocator_free(executor->host_allocator, executor->workloads);
  }
  if (executor->inputs) {
    loom_testbench_reset_values(executor->inputs, executor->input_capacity);
    iree_allocator_free(executor->host_allocator, executor->inputs);
  }
  if (executor->results) {
    loom_testbench_reset_values(executor->results, executor->result_capacity);
    iree_allocator_free(executor->host_allocator, executor->results);
  }
  iree_allocator_free(executor->host_allocator, executor->issues);
  memset(executor, 0, sizeof(*executor));
}

static iree_status_t loom_testbench_invocation_executor_query_issue(
    loom_testbench_invocation_executor_t* executor,
    const loom_testbench_prepared_invocation_t* prepared) {
  if (prepared->provider.query_issue == NULL) {
    return iree_ok_status();
  }
  loom_testbench_sample_issue_t issue = {0};
  IREE_RETURN_IF_ERROR(prepared->provider.query_issue(
      prepared->provider.user_data, prepared->plan, &issue));
  if (issue.category == LOOM_TESTBENCH_SAMPLE_ISSUE_NONE) {
    return iree_ok_status();
  }
  if (executor->issue_count >= executor->issue_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "testbench sample issue capacity exceeded invocation count");
  }
  executor->issues[executor->issue_count++] = issue;
  return iree_ok_status();
}

static iree_status_t loom_testbench_load_invocation_values(
    const loom_testbench_value_table_t* table, const loom_value_id_t* value_ids,
    iree_host_size_t value_count, loom_testbench_value_t* values) {
  for (iree_host_size_t value_index = 0; value_index < value_count;
       ++value_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_lookup_retain(
        table, value_ids[value_index], &values[value_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_store_invocation_results(
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_value_table_t* table, loom_testbench_value_t* results) {
  for (iree_host_size_t result_index = 0;
       result_index < invocation->result_count; ++result_index) {
    if (results[result_index].kind == LOOM_TESTBENCH_VALUE_KIND_NONE) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "invocation result %zu was not assigned",
                              result_index);
    }
  }
  for (iree_host_size_t result_index = 0;
       result_index < invocation->result_count; ++result_index) {
    IREE_RETURN_IF_ERROR(loom_testbench_value_table_assign_move(
        table, invocation->result_value_ids[result_index],
        &results[result_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_run_single_invocation(
    loom_testbench_invocation_executor_t* executor,
    const loom_testbench_prepared_invocation_t* prepared,
    loom_testbench_value_table_t* table) {
  const loom_testbench_invocation_plan_t* invocation = prepared->plan;
  IREE_ASSERT(invocation->workload_count <= executor->workload_capacity);
  IREE_ASSERT(invocation->input_count <= executor->input_capacity);
  IREE_ASSERT(invocation->result_count <= executor->result_capacity);

  iree_status_t status = loom_testbench_load_invocation_values(
      table, invocation->workload_value_ids, invocation->workload_count,
      executor->workloads);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_load_invocation_values(
        table, invocation->input_value_ids, invocation->input_count,
        executor->inputs);
  }
  if (iree_status_is_ok(status)) {
    status = prepared->provider.invoke(
        prepared->provider.user_data, invocation, invocation->workload_count,
        executor->workloads, invocation->input_count, executor->inputs,
        invocation->result_count, executor->results);
  }
  if (iree_status_is_ok(status)) {
    status = loom_testbench_invocation_executor_query_issue(executor, prepared);
  }
  loom_testbench_reset_values(executor->workloads, invocation->workload_count);
  loom_testbench_reset_values(executor->inputs, invocation->input_count);
  if (iree_status_is_ok(status) && executor->issue_count == 0) {
    status = loom_testbench_store_invocation_results(invocation, table,
                                                     executor->results);
  }
  loom_testbench_reset_values(executor->results, invocation->result_count);
  return status;
}

static iree_status_t loom_testbench_query_invocation_span_issues(
    loom_testbench_invocation_executor_t* executor,
    const loom_testbench_prepared_invocation_span_t* span) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t invocation_index = 0;
       iree_status_is_ok(status) && executor->issue_count == 0 &&
       invocation_index < span->invocation_count;
       ++invocation_index) {
    status = loom_testbench_invocation_executor_query_issue(
        executor, &span->invocations[invocation_index]);
  }
  return status;
}

iree_status_t loom_testbench_run_case_invocations(
    loom_testbench_invocation_executor_t* executor,
    iree_host_size_t sample_ordinal, loom_testbench_value_table_t* table) {
  executor->issue_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t span_index = 0;
       iree_status_is_ok(status) && executor->issue_count == 0 &&
       span_index < executor->schedule->span_count;
       ++span_index) {
    const loom_testbench_prepared_invocation_span_t* span =
        &executor->schedule->spans[span_index];
    const loom_testbench_prepared_invocation_t* prepared = span->invocations;
    if (span->invocation_count > 1) {
      IREE_ASSERT(prepared->provider.invoke_sequence != NULL);
      status = prepared->provider.invoke_sequence(
          prepared->provider.user_data, sample_ordinal, span->invocation_count,
          prepared, table);
      if (iree_status_is_ok(status)) {
        status = loom_testbench_query_invocation_span_issues(executor, span);
      }
    } else {
      status = loom_testbench_run_single_invocation(executor, prepared, table);
    }
  }
  return status;
}
