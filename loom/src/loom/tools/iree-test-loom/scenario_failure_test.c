// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>

#include "iree/hal/api.h"
#include "loom/ir/module.h"
#include "loom/tools/iree-test-loom/main.h"

typedef struct iree_test_loom_scenario_failure_profile_t {
  // Stable diagnostic name for the profile.
  iree_string_view_t name;
  // True when trial ordinal one must inject the selected subject failure.
  bool inject_failures;
} iree_test_loom_scenario_failure_profile_t;

static iree_string_view_t iree_test_loom_scenario_failure_subject_name(
    const loom_testbench_invocation_plan_t* invocation) {
  const loom_module_t* module = invocation->module;
  const loom_symbol_ref_t ref = invocation->callee_ref;
  if (ref.module_id != 0 || ref.symbol_id >= module->symbols.count) {
    return iree_string_view_empty();
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  return symbol->name_id < module->strings.count
             ? loom_string_table_get(&module->strings, symbol->name_id)
             : iree_string_view_empty();
}

static iree_status_t iree_test_loom_scenario_failure_require_call_shape(
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t expected_input_count,
    iree_host_size_t expected_result_count) {
  if (invocation->workload_count != 0 ||
      invocation->input_count != expected_input_count ||
      invocation->result_count != expected_result_count) {
    const iree_string_view_t subject_name =
        iree_test_loom_scenario_failure_subject_name(invocation);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "controlled subject '%.*s' expected %zu input(s) and %zu result(s)",
        (int)subject_name.size, subject_name.data, expected_input_count,
        expected_result_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_scenario_failure_execute_result(
    const iree_test_loom_scenario_failure_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  IREE_RETURN_IF_ERROR(iree_test_loom_scenario_failure_require_call_shape(
      invocation, /*expected_input_count=*/1, /*expected_result_count=*/1));
  for (iree_host_size_t i = 0; i < call_count; ++i) {
    const loom_testbench_value_t* input = &calls[i].arguments[0];
    if (input->kind != LOOM_TESTBENCH_VALUE_KIND_SCALAR ||
        input->scalar.kind != IREE_TOOLING_VALUE_KIND_I32) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "controlled result subject requires i32 input");
    }
    loom_testbench_value_t* result = &calls[i].results[0];
    result->kind = LOOM_TESTBENCH_VALUE_KIND_SCALAR;
    result->scalar.kind = IREE_TOOLING_VALUE_KIND_I32;
    result->scalar.storage.i32 =
        input->scalar.storage.i32 +
        (profile->inject_failures && calls[i].identity->trial_ordinal == 1);
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_scenario_failure_execute_post_state(
    const iree_test_loom_scenario_failure_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  IREE_RETURN_IF_ERROR(iree_test_loom_scenario_failure_require_call_shape(
      invocation, /*expected_input_count=*/1, /*expected_result_count=*/0));
  for (iree_host_size_t i = 0; i < call_count; ++i) {
    const loom_testbench_value_t* storage = &calls[i].arguments[0];
    if (storage->kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "controlled post-state subject requires a buffer input");
    }
    int32_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(storage->buffer.buffer,
                                                  storage->buffer.byte_offset,
                                                  &value, sizeof(value)));
    value +=
        1 + (profile->inject_failures && calls[i].identity->trial_ordinal == 1);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(storage->buffer.buffer,
                                                   storage->buffer.byte_offset,
                                                   &value, sizeof(value)));
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_scenario_failure_execute_guard(
    const iree_test_loom_scenario_failure_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  IREE_RETURN_IF_ERROR(iree_test_loom_scenario_failure_require_call_shape(
      invocation, /*expected_input_count=*/1, /*expected_result_count=*/0));
  for (iree_host_size_t i = 0; i < call_count; ++i) {
    const loom_testbench_value_t* payload = &calls[i].arguments[0];
    if (payload->kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "controlled guard subject requires an interior buffer view");
    }
    const iree_device_size_t payload_byte_offset =
        iree_hal_buffer_byte_offset(payload->buffer.buffer) +
        payload->buffer.byte_offset;
    if (payload_byte_offset < sizeof(int32_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "controlled guard subject requires an interior buffer view");
    }
    if (profile->inject_failures && calls[i].identity->trial_ordinal == 1) {
      const int32_t corrupted_guard = -1;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(
          iree_hal_buffer_allocated_buffer(payload->buffer.buffer),
          payload_byte_offset - sizeof(corrupted_guard), &corrupted_guard,
          sizeof(corrupted_guard)));
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_scenario_failure_execute_reference(
    const iree_test_loom_scenario_failure_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  IREE_RETURN_IF_ERROR(iree_test_loom_scenario_failure_require_call_shape(
      invocation, /*expected_input_count=*/2, /*expected_result_count=*/1));
  for (iree_host_size_t i = 0; i < call_count; ++i) {
    const bool select_second =
        profile->inject_failures && calls[i].identity->trial_ordinal == 1;
    loom_testbench_value_retain(&calls[i].arguments[select_second ? 1 : 0],
                                &calls[i].results[0]);
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_scenario_failure_execute(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  const iree_test_loom_scenario_failure_profile_t* profile =
      (const iree_test_loom_scenario_failure_profile_t*)user_data;
  const iree_string_view_t subject =
      iree_test_loom_scenario_failure_subject_name(invocation);
  if (iree_string_view_equal(subject, IREE_SV("scenario_wrong_result")) ||
      iree_string_view_equal(subject, IREE_SV("scenario_missing_outcome"))) {
    return iree_test_loom_scenario_failure_execute_result(profile, invocation,
                                                          call_count, calls);
  }
  if (iree_string_view_equal(subject, IREE_SV("scenario_wrong_post_state"))) {
    return iree_test_loom_scenario_failure_execute_post_state(
        profile, invocation, call_count, calls);
  }
  if (iree_string_view_equal(subject, IREE_SV("scenario_wrong_guard"))) {
    return iree_test_loom_scenario_failure_execute_guard(profile, invocation,
                                                         call_count, calls);
  }
  if (iree_string_view_equal(subject, IREE_SV("scenario_wrong_reference"))) {
    return iree_test_loom_scenario_failure_execute_reference(
        profile, invocation, call_count, calls);
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "unrecognized controlled scenario subject '%.*s'",
                          (int)subject.size, subject.data);
}

static const loom_testbench_product_vtable_t
    iree_test_loom_scenario_failure_product_vtable = {
        .execute = iree_test_loom_scenario_failure_execute,
};

static iree_status_t iree_test_loom_scenario_failure_prepare(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  (void)invocation;
  (void)configuration;
  (void)host_allocator;
  *out_product = (loom_testbench_prepared_product_t){
      .vtable = &iree_test_loom_scenario_failure_product_vtable,
      .user_data = user_data,
  };
  return iree_ok_status();
}

static loom_testbench_execution_profile_t
iree_test_loom_scenario_failure_bind_profile(
    void* user_data, const loom_testbench_compilation_t* compilation,
    const loom_source_table_resolver_t* sources,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_testbench_compile_result_callback_t result_callback) {
  (void)compilation;
  (void)sources;
  (void)diagnostic_sink;
  (void)result_callback;
  iree_test_loom_scenario_failure_profile_t* profile =
      (iree_test_loom_scenario_failure_profile_t*)user_data;
  return (loom_testbench_execution_profile_t){
      .name = profile->name,
      .prepare = iree_test_loom_scenario_failure_prepare,
      .user_data = profile,
  };
}

static iree_status_t iree_test_loom_scenario_query_requirement(
    void* user_data, const loom_module_t* module, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_result) {
  (void)user_data;
  iree_string_view_t state = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_read_string_attr(
      module, attrs, IREE_SV("state"), &state));
  if (iree_string_view_equal(state, IREE_SV("satisfied"))) {
    out_result->state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED;
  } else if (iree_string_view_equal(state, IREE_SV("unsatisfied"))) {
    out_result->state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSATISFIED;
  } else if (iree_string_view_equal(state, IREE_SV("unavailable"))) {
    out_result->state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE;
  } else {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "controlled requirement provider failed");
  }
  return iree_ok_status();
}

static iree_status_t iree_test_loom_scenario_populate_requirements(
    void* user_data, loom_run_hal_testbench_context_t* hal_context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count) {
  (void)user_data;
  (void)hal_context;
  if (*inout_provider_count >= provider_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "no capacity for controlled requirement provider");
  }
  providers[(*inout_provider_count)++] =
      (loom_testbench_requirement_provider_t){
          .name = IREE_SV("controlled.requirement"),
          .query = iree_test_loom_scenario_query_requirement,
      };
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_test_loom_scenario_failure_profile_t target = {
      .name = IREE_SV("controlled-target"),
      .inject_failures = true,
  };
  iree_test_loom_scenario_failure_profile_t oracle = {
      .name = IREE_SV("controlled-oracle"),
      .inject_failures = false,
  };
  const iree_test_loom_configuration_t configuration = {
      .tool_name = "iree-test-loom-scenario-failure-test",
      .populate_requirement_providers =
          {
              .fn = iree_test_loom_scenario_populate_requirements,
          },
      .scenario_target_profile =
          {
              .fn = iree_test_loom_scenario_failure_bind_profile,
              .user_data = &target,
          },
      .scenario_oracle_profile =
          {
              .fn = iree_test_loom_scenario_failure_bind_profile,
              .user_data = &oracle,
          },
  };
  return iree_test_loom_main(argc, argv, &configuration);
}
