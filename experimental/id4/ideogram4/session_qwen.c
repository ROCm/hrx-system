// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/ideogram4/session_state.h"
#include "experimental/id4/ideogram4/session_support.h"
#include "experimental/id4/stages/qwen3_vl.h"

static iree_status_t id4_ideogram4_validate_qwen_issue_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Ideogram 4 session must be loaded before issue");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options), IREE_SV("Qwen issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Qwen issue extension structures are not "
                            "supported");
  }
  if (!options->request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue request is required");
  }
  if (!options->tokenizer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue tokenizer is required");
  }
  if (!options->parameter_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue parameter provider is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue kernel library is required");
  }
  switch (options->qwen_weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Qwen issue weight execution strategy %" PRIu32 " is invalid",
          (uint32_t)options->qwen_weight_execution_strategy);
  }
  const iree_host_size_t device_count = iree_hal_device_group_device_count(
      id4_pipeline_stage_services(session->qwen_stage)->device_group);
  if (options->device_index >= device_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue device index %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            options->device_index, device_count);
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_diagnostic_tap_names(
      options->diagnostic_tap_names));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Qwen issue wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("Qwen issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen issue final signal is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Qwen issue")));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_qwen_execution_allocate(
    id4_ideogram4_session_t* session, iree_allocator_t host_allocator,
    id4_ideogram4_qwen_execution_t** out_execution) {
  *out_execution = NULL;
  id4_ideogram4_qwen_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->host_allocator = host_allocator;
  execution->session = session;
  id4_ideogram4_session_retain(session);
  *out_execution = execution;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_qwen_create_plan(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options,
    const id4_ideogram4_qwen_inputs_t* inputs,
    id4_ideogram4_qwen_execution_t* execution) {
  id4_qwen3_vl_stage_plan_options_t qwen_options;
  memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = inputs->token_count;
  qwen_options.request.token_ids = inputs->token_ids;
  qwen_options.weight_execution_strategy =
      options->qwen_weight_execution_strategy;

  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &qwen_options;
  plan_options.flags =
      options->diagnostic_tap_names.count == 0
          ? 0
          : ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = options->device_index;
  plan_options.queue_affinity = options->queue_affinity;
  plan_options.diagnostic_tap_names = options->diagnostic_tap_names;
  plan_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_plan(session->qwen_stage, &plan_options,
                                 &execution->plan);
}

static iree_status_t id4_ideogram4_qwen_create_semaphores(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_status_t status = iree_hal_semaphore_create(
      device, queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &execution->prepare_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, queue_affinity, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->upload_semaphore);
  }
  return status;
}

static iree_status_t id4_ideogram4_qwen_prepare_bundle(
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->prepare_semaphore,
      signal_payload_value);

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = options->parameter_provider;
  prepare_options.kernel_library = options->kernel_library;
  prepare_options.wait_semaphore_list = options->wait_semaphore_list;
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.command_buffer_mode = options->command_buffer_mode;
  prepare_options.kernel_diagnostic_artifact_flags =
      options->kernel_diagnostic_artifact_flags;
  prepare_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_prepare(execution->session->qwen_stage,
                                    execution->plan, &prepare_options,
                                    &execution->bundle);
}

static iree_status_t id4_ideogram4_qwen_allocate_bindings(
    iree_hal_device_t* device,
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_status_t status = id4_pipeline_allocate_boundary_bindings(
      device, options->queue_affinity, execution->plan,
      execution->host_allocator, &execution->boundary_bindings);
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_allocate_diagnostic_tap_bindings(
        device, options->queue_affinity, execution->plan,
        execution->host_allocator, &execution->diagnostic_tap_bindings);
  }
  return status;
}

static iree_status_t id4_ideogram4_qwen_upload_inputs(
    iree_hal_device_t* device,
    const id4_ideogram4_qwen_issue_options_t* options,
    const id4_ideogram4_qwen_inputs_t* inputs,
    id4_ideogram4_qwen_execution_t* execution) {
  const iree_host_size_t attention_mask_length =
      inputs->token_capacity * (iree_host_size_t)inputs->token_capacity *
      sizeof(inputs->attention_mask[0]);
  const iree_host_size_t token_weights_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_weights[0]);

  id4_ideogram4_boundary_upload_context_t upload_context = {
      .device = device,
      .queue_affinity = options->queue_affinity,
      .plan = execution->plan,
      .boundary_bindings = &execution->boundary_bindings,
      .semaphore = execution->upload_semaphore,
      .payload_value = &execution->upload_payload_value,
  };

  IREE_RETURN_IF_ERROR(id4_ideogram4_upload_boundary_tensor(
      &upload_context, IREE_SV("attention_mask"), inputs->attention_mask,
      attention_mask_length, options->wait_semaphore_list));
  return id4_ideogram4_upload_boundary_tensor(
      &upload_context, IREE_SV("token_weights"), inputs->token_weights,
      token_weights_length, iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_qwen_find_outputs(
    id4_ideogram4_qwen_execution_t* execution) {
  return id4_pipeline_find_boundary_binding(
      execution->plan, &execution->boundary_bindings, IREE_SV("condition"),
      &execution->condition_binding);
}

static iree_status_t id4_ideogram4_qwen_issue_bundle(
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_two_semaphore_list(
      wait_semaphores, wait_payload_values, execution->prepare_semaphore, 1,
      execution->upload_semaphore, execution->upload_payload_value);

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = execution->boundary_bindings.count;
  issue_options.boundary_bindings = execution->boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count =
      execution->diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings =
      execution->diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = wait_list;
  issue_options.signal_semaphore_list = options->signal_semaphore_list;
  issue_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_issue(execution->session->qwen_stage,
                                  execution->bundle, &issue_options);
}

iree_status_t id4_ideogram4_session_issue_qwen(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_qwen_issue_options(session, options));

  id4_ideogram4_qwen_inputs_t inputs;
  memset(&inputs, 0, sizeof(inputs));
  id4_ideogram4_qwen_execution_t* execution = NULL;

  iree_status_t status = id4_ideogram4_qwen_execution_allocate(
      session, session->host_allocator, &execution);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_qwen_lowering_options_t lowering_options;
    memset(&lowering_options, 0, sizeof(lowering_options));
    lowering_options.structure_size = sizeof(lowering_options);
    lowering_options.tokenizer = options->tokenizer;
    lowering_options.request = options->request;
    lowering_options.tokenizer_flags = options->tokenizer_flags;
    lowering_options.max_token_count = session->qwen_model.max_token_count;
    lowering_options.vocab_size = session->qwen_model.vocab_size;
    uint32_t token_count = 0;
    status = id4_ideogram4_request_count_qwen_tokens(
        &lowering_options, session->host_allocator, &token_count);
    if (iree_status_is_ok(status)) {
      status = id4_qwen3_vl_program_calculate_bf16_token_capacity(
          token_count, &lowering_options.token_capacity);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_request_lower_qwen_inputs(
          &lowering_options, session->host_allocator, &inputs);
    }
  }
  if (iree_status_is_ok(status)) {
    execution->token_count = inputs.token_count;
    status =
        id4_ideogram4_qwen_create_plan(session, options, &inputs, execution);
  }
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    device = iree_hal_device_group_device_at(
        id4_pipeline_stage_services(session->qwen_stage)->device_group,
        options->device_index);
    status = id4_ideogram4_qwen_create_semaphores(
        device, options->queue_affinity, execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_prepare_bundle(options, execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_allocate_bindings(device, options, execution);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_ideogram4_qwen_upload_inputs(device, options, &inputs, execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_find_outputs(execution);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_qwen_issue_bundle(options, execution);
  }
  id4_ideogram4_qwen_inputs_deinitialize(&inputs, session->host_allocator);
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_qwen_execution_release(execution);
  }
  return status;
}

void id4_ideogram4_qwen_execution_release(
    id4_ideogram4_qwen_execution_t* execution) {
  if (!execution) return;
  id4_pipeline_buffer_binding_set_deinitialize(
      &execution->diagnostic_tap_bindings);
  id4_pipeline_buffer_binding_set_deinitialize(&execution->boundary_bindings);
  id4_pipeline_bundle_release(execution->bundle);
  id4_pipeline_plan_release(execution->plan);
  iree_hal_semaphore_release(execution->upload_semaphore);
  iree_hal_semaphore_release(execution->prepare_semaphore);
  id4_ideogram4_session_release(execution->session);
  iree_allocator_t host_allocator = execution->host_allocator;
  iree_allocator_free(host_allocator, execution);
}

const id4_pipeline_plan_t* id4_ideogram4_qwen_execution_plan(
    const id4_ideogram4_qwen_execution_t* execution) {
  IREE_ASSERT_ARGUMENT(execution);
  return execution->plan;
}

iree_status_t id4_ideogram4_qwen_execution_result(
    const id4_ideogram4_qwen_execution_t* execution,
    id4_ideogram4_qwen_result_t* out_result) {
  if (!execution || !out_result) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen execution and result output are required");
  }
  out_result->token_count = execution->token_count;
  out_result->condition_binding = execution->condition_binding;
  return iree_ok_status();
}
