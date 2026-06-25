// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "experimental/id4/pipeline/binding.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"

struct id4_ideogram4_session_t {
  // Reference count for shared session ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for session-owned storage.
  iree_allocator_t host_allocator;
  // Qwen3-VL model dimensions used by the text conditioning stage.
  id4_qwen3_vl_model_config_t qwen_model;
  // Ideogram 4 DiT model dimensions used by conditioned/unconditioned stages.
  id4_ideogram4_dit_model_config_t dit_model;
  // Ideogram 4 latent-to-image decode model contract.
  id4_ideogram4_decode_model_config_t decode_model;
  // Coarse Qwen3-VL forward stage owned by the session.
  id4_pipeline_stage_t* qwen_stage;
  // Conditioned Ideogram 4 DiT forward stage owned by the session.
  id4_pipeline_stage_t* dit_conditioned_stage;
  // Unconditioned Ideogram 4 DiT forward stage owned by the session.
  id4_pipeline_stage_t* dit_unconditioned_stage;
  // Device-side sampler denoise-step stage owned by the session.
  id4_pipeline_stage_t* sampler_stage;
  // Ideogram 4 VAE-backed latent decode stage owned by the session.
  id4_pipeline_stage_t* decode_stage;
  // True after immutable session state has loaded.
  bool is_loaded;
};

struct id4_ideogram4_qwen_execution_t {
  // Allocator used for execution-owned storage.
  iree_allocator_t host_allocator;
  // Session retained for the lifetime of this asynchronous execution.
  id4_ideogram4_session_t* session;
  // Plan retained for diagnostics and boundary metadata.
  id4_pipeline_plan_t* plan;
  // Prepared Qwen bundle retained until execution release.
  id4_pipeline_bundle_t* bundle;
  // Owned boundary tensor buffers in plan order.
  id4_pipeline_buffer_binding_set_t boundary_bindings;
  // Owned diagnostic tap buffers in plan order.
  id4_pipeline_buffer_binding_set_t diagnostic_tap_bindings;
  // Semaphore signaled when parameter loading and preparation complete.
  iree_hal_semaphore_t* prepare_semaphore;
  // Semaphore chaining host-to-device request tensor uploads.
  iree_hal_semaphore_t* upload_semaphore;
  // Final payload value signaled on |upload_semaphore|.
  uint64_t upload_payload_value;
  // Number of token positions in the execution.
  uint32_t token_count;
  // Exported condition tensor binding owned by |boundary_bindings|.
  iree_hal_buffer_binding_t condition_binding;
};

static iree_status_t id4_ideogram4_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_ideogram4_validate_semaphore_list(
    iree_hal_semaphore_list_t semaphore_list, iree_string_view_t list_name) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (!semaphore_list.semaphores) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore array is required",
                            (int)list_name.size, list_name.data);
  }
  if (!semaphore_list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s payload value array is required",
                            (int)list_name.size, list_name.data);
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    if (!semaphore_list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)list_name.size, list_name.data, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_diagnostic_tap_names(
    iree_string_view_list_t names) {
  if (names.count == 0) {
    if (names.values) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "diagnostic tap name array requires at least one tap name");
    }
    return iree_ok_status();
  }
  if (!names.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "diagnostic tap name array is required");
  }
  for (iree_host_size_t i = 0; i < names.count; ++i) {
    if (iree_string_view_is_empty(names.values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "diagnostic tap name %" PRIhsz " is empty", i);
    }
  }
  return iree_ok_status();
}

static iree_hal_semaphore_list_t id4_ideogram4_single_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  *semaphore_storage = semaphore;
  *payload_storage = payload_value;
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = 1,
      // Stack-backed semaphore handle.
      .semaphores = semaphore_storage,
      // Stack-backed payload value.
      .payload_values = payload_storage,
  };
  return list;
}

static iree_hal_semaphore_list_t id4_ideogram4_two_semaphore_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_hal_semaphore_t* first_semaphore, uint64_t first_payload_value,
    iree_hal_semaphore_t* second_semaphore, uint64_t second_payload_value) {
  semaphore_storage[0] = first_semaphore;
  semaphore_storage[1] = second_semaphore;
  payload_storage[0] = first_payload_value;
  payload_storage[1] = second_payload_value;
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = 2,
      // Stack-backed semaphore handles.
      .semaphores = semaphore_storage,
      // Stack-backed payload values.
      .payload_values = payload_storage,
  };
  return list;
}

static iree_status_t id4_ideogram4_validate_session_create_options(
    const id4_ideogram4_session_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 session create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 session create extension structures are not supported");
  }
  if (!options->services.device_group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session device group is required");
  }
  if (!options->services.executable_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session executable cache is required");
  }
  if (iree_allocator_is_null(options->services.host_allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session host allocator is required");
  }
  if (!options->kernel_cache) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session kernel cache is required");
  }
  if (options->vae_activation_format == ID4_VAE_ACTIVATION_FORMAT_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 session VAE activation format is required");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_create_qwen_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_qwen3_vl_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = options->parameter_scopes.qwen;
  stage_options.model = session->qwen_model;
  return id4_qwen3_vl_stage_create(&stage_options, session->host_allocator,
                                   &session->qwen_stage);
}

static iree_status_t id4_ideogram4_create_dit_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session, iree_string_view_t parameter_scope,
    id4_pipeline_stage_t** out_stage) {
  id4_ideogram4_dit_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = parameter_scope;
  stage_options.model = session->dit_model;
  return id4_ideogram4_dit_stage_create(&stage_options, session->host_allocator,
                                        out_stage);
}

static iree_status_t id4_ideogram4_create_sampler_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_sampler_denoise_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  return id4_sampler_denoise_stage_create(
      &stage_options, session->host_allocator, &session->sampler_stage);
}

static iree_status_t id4_ideogram4_create_decode_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_ideogram4_decode_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = options->parameter_scopes.vae;
  stage_options.model = session->decode_model;
  stage_options.vae_activation_format = options->vae_activation_format;
  return id4_ideogram4_decode_stage_create(
      &stage_options, session->host_allocator, &session->decode_stage);
}

static iree_status_t id4_ideogram4_create_stages(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  session->qwen_model = *id4_qwen3_vl_program_ideogram4_model_config();
  session->dit_model = *id4_ideogram4_dit_program_ideogram4_model_config();
  session->decode_model =
      *id4_ideogram4_decode_program_ideogram4_model_config();

  iree_status_t status = id4_ideogram4_create_qwen_stage(options, session);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_dit_stage(
        options, session, options->parameter_scopes.dit_conditioned,
        &session->dit_conditioned_stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_dit_stage(
        options, session, options->parameter_scopes.dit_unconditioned,
        &session->dit_unconditioned_stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_sampler_stage(options, session);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_decode_stage(options, session);
  }
  return status;
}

iree_status_t id4_ideogram4_session_create(
    const id4_ideogram4_session_create_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_session_create_options(options));

  id4_ideogram4_session_t* session = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*session), (void**)&session);
  if (iree_status_is_ok(status)) {
    memset(session, 0, sizeof(*session));
    iree_atomic_ref_count_init(&session->ref_count);
    session->host_allocator = host_allocator;
    status = id4_ideogram4_create_stages(options, session);
  }
  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    id4_ideogram4_session_release(session);
  }
  return status;
}

static void id4_ideogram4_session_retain(id4_ideogram4_session_t* session) {
  if (!session) return;
  iree_atomic_ref_count_inc(&session->ref_count);
}

void id4_ideogram4_session_release(id4_ideogram4_session_t* session) {
  if (session && iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_allocator_t host_allocator = session->host_allocator;
    id4_pipeline_stage_release(session->decode_stage);
    id4_pipeline_stage_release(session->sampler_stage);
    id4_pipeline_stage_release(session->dit_unconditioned_stage);
    id4_pipeline_stage_release(session->dit_conditioned_stage);
    id4_pipeline_stage_release(session->qwen_stage);
    iree_allocator_free(host_allocator, session);
  }
}

static iree_status_t id4_ideogram4_validate_session_load_options(
    const id4_ideogram4_session_load_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session load options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 session load")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 session load extension structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 session load")));
  return iree_ok_status();
}

iree_status_t id4_ideogram4_session_load(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_session_load_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_session_load_options(options));
  if (session->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Ideogram 4 session is already loaded");
  }
  id4_pipeline_stage_load_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.diagnostics_sink = options->diagnostics_sink;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->qwen_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->dit_conditioned_stage, &stage_options));
  IREE_RETURN_IF_ERROR(id4_pipeline_stage_load(session->dit_unconditioned_stage,
                                               &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->sampler_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->decode_stage, &stage_options));
  session->is_loaded = true;
  return iree_ok_status();
}

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

static iree_status_t id4_ideogram4_qwen_upload_binding(
    iree_hal_device_t* device,
    const id4_ideogram4_qwen_issue_options_t* options,
    id4_ideogram4_qwen_execution_t* execution, iree_string_view_t binding_name,
    const void* source_data, iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  iree_hal_buffer_binding_t binding;
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->plan, &execution->boundary_bindings, binding_name, &binding));
  return id4_pipeline_queue_update_binding(
      device, options->queue_affinity, &binding, source_data, source_length,
      initial_wait_semaphore_list, execution->upload_semaphore,
      &execution->upload_payload_value);
}

static iree_status_t id4_ideogram4_qwen_upload_inputs(
    iree_hal_device_t* device,
    const id4_ideogram4_qwen_issue_options_t* options,
    const id4_ideogram4_qwen_inputs_t* inputs,
    id4_ideogram4_qwen_execution_t* execution) {
  const iree_host_size_t token_ids_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_ids[0]);
  const iree_host_size_t attention_mask_length =
      inputs->token_count * (iree_host_size_t)inputs->token_count *
      sizeof(inputs->attention_mask[0]);
  const iree_host_size_t token_weights_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_weights[0]);

  IREE_RETURN_IF_ERROR(id4_ideogram4_qwen_upload_binding(
      device, options, execution, IREE_SV("token_ids"), inputs->token_ids,
      token_ids_length, options->wait_semaphore_list));
  IREE_RETURN_IF_ERROR(id4_ideogram4_qwen_upload_binding(
      device, options, execution, IREE_SV("attention_mask"),
      inputs->attention_mask, attention_mask_length,
      iree_hal_semaphore_list_empty()));
  return id4_ideogram4_qwen_upload_binding(
      device, options, execution, IREE_SV("token_weights"),
      inputs->token_weights, token_weights_length,
      iree_hal_semaphore_list_empty());
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
    status = id4_ideogram4_request_lower_qwen_inputs(
        &lowering_options, session->host_allocator, &inputs);
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
