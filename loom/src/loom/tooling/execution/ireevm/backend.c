// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/ireevm/backend.h"

#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/ireevm/candidate.h"
#include "loom/tooling/execution/ireevm/invocation.h"
#include "loom/tooling/execution/one_shot.h"

static iree_status_t loom_ireevm_execution_backend_run_one_shot(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request) {
  loom_ireevm_run_candidate_t candidate = {0};
  loom_run_vm_runtime_t runtime = {0};
  loom_run_vm_invocation_result_t invocation_result = {0};
  loom_run_vm_invocation_result_initialize(request->host_allocator,
                                           &invocation_result);

  iree_status_t status = loom_ireevm_run_candidate_emit(
      request->run_module, request->compile_options, request->host_allocator,
      &candidate);
  if (iree_status_is_ok(status) && !candidate.emitted) {
    request->result->exit_code = 1;
  }
  if (iree_status_is_ok(status) && candidate.emitted) {
    status = loom_run_vm_runtime_initialize(request->host_allocator, &runtime);
  }
  if (iree_status_is_ok(status) && candidate.emitted) {
    loom_run_vm_invocation_request_t invocation_request = {0};
    loom_run_vm_invocation_request_initialize(&invocation_request);
    invocation_request.runtime = &runtime;
    invocation_request.archive = &candidate.archive;
    invocation_request.options.function_name =
        request->options->vm_function_name;
    invocation_request.options.inputs = (loom_run_vm_value_specs_t){
        .values = request->options->vm_inputs.values,
        .count = request->options->vm_inputs.count,
    };
    invocation_request.options.outputs = (loom_run_vm_value_specs_t){
        .values = request->options->vm_outputs.values,
        .count = request->options->vm_outputs.count,
    };
    invocation_request.options.expected_outputs = (loom_run_vm_value_specs_t){
        .values = request->options->vm_expected_outputs.values,
        .count = request->options->vm_expected_outputs.count,
    };
    invocation_request.options.max_output_element_count =
        request->options->vm_max_output_element_count;
    status = loom_run_vm_invocation_run(
        &invocation_request, request->host_allocator, &invocation_result);
  }
  if (iree_status_is_ok(status) && candidate.emitted) {
    request->result->exit_code = invocation_result.exit_code;
    status = iree_string_builder_append_string(
        &request->result->output,
        iree_string_builder_view(&invocation_result.output));
  }
  if (iree_status_is_ok(status) && request->compile_report_capture != NULL) {
    status = loom_run_compile_report_capture_append_output(
        request->compile_report_capture, &request->result->output);
  }

  loom_run_vm_invocation_result_deinitialize(&invocation_result);
  loom_run_vm_runtime_deinitialize(&runtime);
  loom_ireevm_run_candidate_deinitialize(&candidate);
  return status;
}

const loom_run_execution_backend_t loom_ireevm_execution_backend = {
    .name = IREE_SVL("vm"),
    .flags = LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS,
    .run_one_shot = loom_ireevm_execution_backend_run_one_shot,
};
