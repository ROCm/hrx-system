// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/execution_backend.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/hal/candidate.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/io/file.h"

static const loom_run_hal_execution_backend_t*
loom_run_hal_execution_backend_from_base(
    const loom_run_execution_backend_t* backend) {
  IREE_ASSERT_ARGUMENT(backend);
  return iree_containerof(backend, loom_run_hal_execution_backend_t, base);
}

static const loom_run_hal_artifact_provider_t*
loom_run_hal_execution_backend_artifact_provider(
    const loom_run_execution_backend_t* backend) {
  const loom_run_hal_execution_backend_t* hal_execution_backend =
      loom_run_hal_execution_backend_from_base(backend);
  return hal_execution_backend->artifact_provider;
}

static iree_status_t loom_run_hal_write_artifact(
    iree_string_view_t path, iree_const_byte_span_t contents,
    iree_string_view_t artifact_name, iree_allocator_t allocator) {
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  if (contents.data == NULL || contents.data_length == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL %.*s artifact is empty",
                            (int)artifact_name.size, artifact_name.data);
  }
  return loom_tooling_write_output_file(
      path,
      iree_make_string_view((const char*)contents.data, contents.data_length),
      allocator);
}

static iree_status_t loom_run_hal_write_candidate_artifacts(
    const loom_run_one_shot_request_t* request,
    const loom_run_hal_candidate_t* candidate) {
  IREE_RETURN_IF_ERROR(loom_run_hal_write_artifact(
      request->options->hal_target_artifact_output_path,
      candidate->artifact.target_artifact_data, IREE_SV("target-native"),
      request->host_allocator));
  return loom_run_hal_write_artifact(
      request->options->hal_executable_output_path,
      candidate->artifact.executable_data, IREE_SV("executable"),
      request->host_allocator);
}

iree_status_t loom_run_hal_execution_backend_probe(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_probe_request_t* request) {
  const loom_run_hal_artifact_provider_t* artifact_provider =
      loom_run_hal_execution_backend_artifact_provider(backend);
  if (artifact_provider->select_device_target == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL artifact provider '%.*s' is missing required device target "
        "selection hook",
        (int)artifact_provider->name.size, artifact_provider->name.data);
  }

  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_device_target_t target = {0};

  iree_status_t status = loom_run_hal_runtime_initialize(
      artifact_provider->hal_driver_name, request->host_allocator, &runtime);
  if (iree_status_is_ok(status)) {
    status = artifact_provider->select_device_target(
        artifact_provider, &runtime, request->host_allocator, &target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &request->result->output,
        "backend: %.*s\nhal provider: %.*s\nhal driver: %.*s\n",
        (int)backend->name.size, backend->name.data,
        (int)artifact_provider->name.size, artifact_provider->name.data,
        (int)artifact_provider->hal_driver_name.size,
        artifact_provider->hal_driver_name.data);
  }
  if (iree_status_is_ok(status) &&
      !iree_string_view_is_empty(target.target_key)) {
    status = iree_string_builder_append_format(
        &request->result->output, "hal target key: %.*s\n",
        (int)target.target_key.size, target.target_key.data);
  }

  if (artifact_provider->deinitialize_device_target != NULL) {
    artifact_provider->deinitialize_device_target(artifact_provider, &target,
                                                  request->host_allocator);
  }
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}

iree_status_t loom_run_hal_execution_backend_run_one_shot(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request) {
  const loom_run_hal_artifact_provider_t* artifact_provider =
      loom_run_hal_execution_backend_artifact_provider(backend);

  loom_run_hal_runtime_t runtime = {0};
  loom_run_hal_candidate_t candidate = {0};
  loom_run_hal_invocation_result_t invocation_result = {0};
  loom_run_hal_invocation_result_initialize(request->host_allocator,
                                            &invocation_result);

  iree_status_t status = loom_run_hal_runtime_initialize(
      artifact_provider->hal_driver_name, request->host_allocator, &runtime);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_candidate_compile(
        artifact_provider, &runtime, request->run_module,
        request->compile_options, request->host_allocator, &candidate);
  }
  if (iree_status_is_ok(status) && !candidate.compiled) {
    request->result->exit_code = 1;
  }
  if (iree_status_is_ok(status) && candidate.compiled) {
    status = loom_run_hal_write_candidate_artifacts(request, &candidate);
  }
  if (iree_status_is_ok(status) && candidate.compiled &&
      !request->options->hal_emit_only) {
    loom_run_hal_invocation_request_t invocation_request = {0};
    loom_run_hal_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.artifact = &candidate.artifact;
    invocation_request.options.function_name =
        request->options->hal_function_name;
    invocation_request.options.workgroup_count[0] =
        request->options->hal_workgroup_count[0];
    invocation_request.options.workgroup_count[1] =
        request->options->hal_workgroup_count[1];
    invocation_request.options.workgroup_count[2] =
        request->options->hal_workgroup_count[2];
    if (request->options->hal_constant_count >
        IREE_ARRAYSIZE(invocation_request.options.constants)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch constant count %" PRIhsz " exceeds maximum %" PRIhsz,
          request->options->hal_constant_count,
          IREE_ARRAYSIZE(invocation_request.options.constants));
    }
    if (iree_status_is_ok(status)) {
      invocation_request.options.constant_count =
          request->options->hal_constant_count;
      memcpy(invocation_request.options.constants,
             request->options->hal_constants,
             request->options->hal_constant_count *
                 sizeof(request->options->hal_constants[0]));
      invocation_request.bindings = (loom_run_hal_binding_specs_t){
          .values = request->options->hal_bindings.values,
          .conventions = request->options->hal_bindings.conventions,
          .count = request->options->hal_bindings.count,
      };
      invocation_request.expected_bindings = (loom_run_hal_binding_specs_t){
          .values = request->options->hal_expected_bindings.values,
          .conventions = request->options->hal_expected_bindings.conventions,
          .count = request->options->hal_expected_bindings.count,
      };
      invocation_request.max_output_element_count =
          request->options->hal_max_output_element_count;
      status = loom_run_hal_invocation_run(
          &invocation_request, request->host_allocator, &invocation_result);
    }
  }
  if (iree_status_is_ok(status) && candidate.compiled &&
      !request->options->hal_emit_only) {
    request->result->exit_code = invocation_result.exit_code;
    status = iree_string_builder_append_string(
        &request->result->output,
        iree_string_builder_view(&invocation_result.output));
  }
  if (iree_status_is_ok(status) && request->compile_report_capture != NULL) {
    status = loom_run_compile_report_capture_append_output(
        request->compile_report_capture, &request->result->output);
  }

  loom_run_hal_invocation_result_deinitialize(&invocation_result);
  loom_run_hal_candidate_deinitialize(&candidate);
  loom_run_hal_runtime_deinitialize(&runtime);
  return status;
}
