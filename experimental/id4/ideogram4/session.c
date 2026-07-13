// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/ideogram4/session_generation.h"
#include "experimental/id4/ideogram4/session_prepare.h"
#include "experimental/id4/ideogram4/session_state.h"
#include "experimental/id4/ideogram4/session_support.h"
#include "experimental/id4/pipeline/binding.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"

static iree_status_t id4_ideogram4_validate_generation_issue_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation issue");
  }
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  if (bundle->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation bundle was prepared by a different session");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_policy.mode,
      bundle->residency_policy.request_stage_mask,
      bundle->residency_policy.phase_stage_masks));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation issue")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation issue extension structures are not supported");
  }
  switch (options->issue_policy) {
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT:
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation issue policy %" PRIu32
                              " is invalid",
                              (uint32_t)options->issue_policy);
  }
  const id4_pipeline_stage_issue_flags_t allowed_stage_issue_flags =
      ID4_PIPELINE_STAGE_ISSUE_FLAG_WAIT_AFTER_EACH_EXECUTION_SEGMENT;
  if (iree_any_bit_set(options->stage_issue_flags,
                       ~allowed_stage_issue_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue stage flags 0x%x are unsupported",
        options->stage_issue_flags);
  }
  if (!options->tokenizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue tokenizer is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  if (id4_ideogram4_sampler_preset_step_count(
          options->request->generation.sampler_preset) !=
      bundle->summary.denoise_step_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue step count does not match prepared "
        "bundle");
  }
  id4_pipeline_program_shape_t request_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (request_latent_shape.rank !=
      bundle->summary.diffusion_latent_shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue latent shape does not match prepared "
        "bundle");
  }
  for (uint32_t i = 0; i < request_latent_shape.rank; ++i) {
    if (request_latent_shape.dims[i] !=
        bundle->summary.diffusion_latent_shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation issue latent shape does not match prepared "
          "bundle");
    }
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation issue "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue final signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation issue"));
}

static iree_status_t id4_ideogram4_validate_generation_begin_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation begin");
  }
  if (!bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation bundle is required");
  }
  if (bundle->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation bundle was prepared by a different session");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_bundle_residency(
      bundle->residency_policy.mode,
      bundle->residency_policy.request_stage_mask,
      bundle->residency_policy.phase_stage_masks));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation begin options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation begin")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation begin extension structures are not supported");
  }
  if (!options->tokenizer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin tokenizer is required");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_request(session, options->request));
  if (id4_ideogram4_sampler_preset_step_count(
          options->request->generation.sampler_preset) !=
      bundle->summary.denoise_step_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin step count does not match prepared "
        "bundle");
  }
  id4_pipeline_program_shape_t request_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (request_latent_shape.rank !=
      bundle->summary.diffusion_latent_shape.rank) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation begin latent shape does not match prepared "
        "bundle");
  }
  for (uint32_t i = 0; i < request_latent_shape.rank; ++i) {
    if (request_latent_shape.dims[i] !=
        bundle->summary.diffusion_latent_shape.dims[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation begin latent shape does not match prepared "
          "bundle");
    }
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation begin "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation begin signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation begin signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation begin"));
}

static iree_status_t id4_ideogram4_validate_generation_phase_issue_options(
    const id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options) {
  if (!execution) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation execution is required");
  }
  if (!phase_bundle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation phase bundle is required");
  }
  if (phase_bundle->generation_bundle &&
      phase_bundle->generation_bundle != execution->bundle) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase bundle does not belong to execution "
        "bundle");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_single_generation_phase_mask(
      phase_bundle->phase_mask));
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation phase issue")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation phase issue extension "
                            "structures are not supported");
  }
  const id4_pipeline_stage_issue_flags_t allowed_stage_issue_flags =
      ID4_PIPELINE_STAGE_ISSUE_FLAG_WAIT_AFTER_EACH_EXECUTION_SEGMENT;
  if (iree_any_bit_set(options->stage_issue_flags,
                       ~allowed_stage_issue_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue stage flags 0x%x are unsupported",
        options->stage_issue_flags);
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list,
      IREE_SV("Ideogram 4 generation phase issue wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation phase issue signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation phase issue signal is required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation phase issue"));
}

static iree_status_t id4_ideogram4_generation_execution_allocate(
    id4_ideogram4_generation_bundle_t* bundle, iree_allocator_t host_allocator,
    id4_ideogram4_generation_execution_t** out_execution) {
  *out_execution = NULL;
  id4_ideogram4_generation_execution_t* execution = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*execution),
                                             (void**)&execution));
  memset(execution, 0, sizeof(*execution));
  execution->host_allocator = host_allocator;
  execution->bundle = bundle;
  id4_ideogram4_generation_bundle_retain(bundle);
  *out_execution = execution;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_execution_create_semaphores(
    id4_ideogram4_generation_execution_t* execution) {
  id4_ideogram4_generation_bundle_t* bundle = execution->bundle;
  iree_status_t status = iree_hal_semaphore_create(
      bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
      &execution->upload_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->qwen_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->noise_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &execution->dit_conditioned_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        bundle->device, bundle->queue_affinity, 0, IREE_HAL_SEMAPHORE_FLAG_NONE,
        &execution->dit_unconditioned_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->sampler_done_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &execution->decode_done_semaphore);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_upload_boundary_tensor(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_string_view_t binding_name, const void* source_data,
    iree_host_size_t source_length,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  id4_ideogram4_generation_bundle_t* bundle = execution->bundle;
  id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[stage_ordinal];
  id4_ideogram4_boundary_upload_context_t upload_context = {
      .device = bundle->device,
      .queue_affinity = bundle->queue_affinity,
      .plan = slot->plan,
      .boundary_bindings = &slot->boundary_bindings,
      .semaphore = execution->upload_semaphore,
      .payload_value = &execution->upload_payload_value,
  };
  return id4_ideogram4_upload_boundary_tensor(&upload_context, binding_name,
                                              source_data, source_length,
                                              initial_wait_semaphore_list);
}

static iree_status_t id4_ideogram4_generation_upload_noise_seed(
    id4_ideogram4_generation_execution_t* execution, uint64_t seed,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  int32_t seed_words[2] = {
      // Low 32 bits of the request seed.
      (int32_t)(seed & 0xFFFFFFFFull),
      // High 32 bits of the request seed.
      (int32_t)(seed >> 32),
  };
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, IREE_SV("seed"),
      seed_words, sizeof(seed_words), initial_wait_semaphore_list);
}

static iree_status_t id4_ideogram4_generation_upload_qwen_inputs(
    id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_qwen_inputs_t* inputs,
    iree_hal_semaphore_list_t initial_wait_semaphore_list) {
  const iree_host_size_t attention_mask_length =
      inputs->token_capacity * (iree_host_size_t)inputs->token_capacity *
      sizeof(inputs->attention_mask[0]);
  const iree_host_size_t token_weights_length =
      inputs->token_count * (iree_host_size_t)sizeof(inputs->token_weights[0]);
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("attention_mask"),
      inputs->attention_mask, attention_mask_length,
      initial_wait_semaphore_list));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SV("token_weights"),
      inputs->token_weights, token_weights_length,
      iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_generation_upload_dit_branch_inputs(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    const id4_ideogram4_dit_branch_inputs_t* inputs) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, stage_ordinal, IREE_SV("image_indicator"),
      inputs->image_indicator, inputs->image_indicator_byte_length,
      iree_hal_semaphore_list_empty()));
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, stage_ordinal, IREE_SV("position_embedding"),
      inputs->position_embedding, inputs->position_embedding_byte_length,
      iree_hal_semaphore_list_empty());
}

static iree_status_t id4_ideogram4_generation_upload_denoise_step(
    id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_denoise_step_t* step) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
      IREE_SV("timestep"), &step->flow_time, sizeof(step->flow_time),
      iree_hal_semaphore_list_empty()));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      IREE_SV("timestep"), &step->flow_time, sizeof(step->flow_time),
      iree_hal_semaphore_list_empty()));
  const float step_values[] = {
      step->flow_time,
      step->next_flow_time,
      step->guidance_scale,
  };
  return id4_ideogram4_generation_upload_boundary_tensor(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, IREE_SV("step"),
      step_values, sizeof(step_values), iree_hal_semaphore_list_empty());
}

static iree_hal_semaphore_list_t id4_ideogram4_generation_make_wait_list(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_host_size_t count) {
  iree_hal_semaphore_list_t list = {
      // Number of semaphore edges in this stack-backed list.
      .count = count,
      // Stack-backed semaphore handles.
      .semaphores = semaphore_storage,
      // Stack-backed payload values.
      .payload_values = payload_storage,
  };
  return list;
}

static void id4_ideogram4_generation_push_wait(
    iree_hal_semaphore_t** semaphore_storage, uint64_t* payload_storage,
    iree_host_size_t* inout_count, iree_hal_semaphore_t* semaphore,
    uint64_t payload_value) {
  if (!semaphore) return;
  semaphore_storage[*inout_count] = semaphore;
  payload_storage[*inout_count] = payload_value;
  ++*inout_count;
}

static iree_status_t id4_ideogram4_generation_chain_upload_after_sampler(
    id4_ideogram4_generation_execution_t* execution,
    uint64_t sampler_payload_value) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->sampler_done_semaphore, sampler_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = execution->upload_payload_value + 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->upload_semaphore,
      signal_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  execution->upload_payload_value = signal_payload_value;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_chain_phase_issue_wait(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t phase_wait_list) {
  if (phase_wait_list.count == 0) return iree_ok_status();
  iree_host_size_t wait_count_capacity = 0;
  if (!iree_host_size_checked_add(phase_wait_list.count, 1,
                                  &wait_count_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 generation phase issue wait list count overflow");
  }
  iree_hal_semaphore_t** wait_semaphores = (iree_hal_semaphore_t**)iree_alloca(
      wait_count_capacity * sizeof(wait_semaphores[0]));
  uint64_t* wait_payload_values = (uint64_t*)iree_alloca(
      wait_count_capacity * sizeof(wait_payload_values[0]));
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  for (iree_host_size_t i = 0; i < phase_wait_list.count; ++i) {
    id4_ideogram4_generation_push_wait(
        wait_semaphores, wait_payload_values, &wait_count,
        phase_wait_list.semaphores[i], phase_wait_list.payload_values[i]);
  }
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = execution->upload_payload_value + 1;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore, &signal_payload_value, execution->upload_semaphore,
      signal_payload_value);
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_list, IREE_HAL_EXECUTE_FLAG_NONE));
  execution->upload_payload_value = signal_payload_value;
  if (phase_mask == ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING) {
    execution->qwen_upload_payload_value = signal_payload_value;
    execution->seed_upload_payload_value = signal_payload_value;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_issue_stage(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_bundle_t* stage_bundle,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_t* signal_semaphore, uint64_t signal_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_slot_t* slot =
      &execution->bundle->stages[stage_ordinal];
  iree_hal_semaphore_t* signal_semaphore_storage = NULL;
  uint64_t signal_payload_storage = 0;
  iree_hal_semaphore_list_t signal_list = id4_ideogram4_single_semaphore_list(
      &signal_semaphore_storage, &signal_payload_storage, signal_semaphore,
      signal_payload_value);

  id4_pipeline_stage_issue_options_t issue_options;
  memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.execution_segment_submission_window = 1;
  issue_options.flags = execution->stage_issue_flags;
  issue_options.boundary_binding_count = slot->boundary_bindings.count;
  issue_options.boundary_bindings = slot->boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count =
      slot->diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings =
      slot->diagnostic_tap_bindings.bindings;
  id4_pipeline_parameter_slab_set_t* parameter_slabs =
      id4_pipeline_bundle_parameter_slabs(stage_bundle);
  if (parameter_slabs &&
      id4_pipeline_parameter_slab_set_has_deferred_load_context(
          parameter_slabs)) {
    issue_options.parameter_load_prefetch_segment_distance =
        execution->parameter_load_prefetch_segment_distance;
  }
  issue_options.wait_semaphore_list = wait_semaphore_list;
  issue_options.signal_semaphore_list = signal_list;
  issue_options.diagnostics_sink = diagnostics_sink;
  iree_status_t status =
      id4_pipeline_stage_issue(slot->stage, stage_bundle, &issue_options);
  if (!iree_status_is_ok(status)) {
    status =
        iree_status_join(status, id4_pipeline_bundle_check_readiness_failures(
                                     stage_bundle, diagnostics_sink));
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_qwen(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* qwen_bundle, uint64_t qwen_upload_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     qwen_upload_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, qwen_bundle, wait_list,
      execution->qwen_done_semaphore, 1, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_noise(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* noise_bundle, uint64_t seed_upload_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     seed_upload_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, noise_bundle, wait_list,
      execution->noise_done_semaphore, 1, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_dit_branch(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_bundle_t* branch_bundle,
    iree_hal_semaphore_t* latent_semaphore, uint64_t latent_payload_value,
    iree_hal_semaphore_t* condition_semaphore, uint64_t condition_payload_value,
    iree_hal_semaphore_t* branch_done_semaphore, uint64_t branch_payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[4];
  uint64_t wait_payload_values[4];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, latent_semaphore,
                                     latent_payload_value);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, condition_semaphore,
                                     condition_payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, stage_ordinal, branch_bundle, wait_list, branch_done_semaphore,
      branch_payload_value, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_sampler(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* sampler_bundle, uint64_t payload_value,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[3];
  uint64_t wait_payload_values[3];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->dit_conditioned_done_semaphore, payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->dit_unconditioned_done_semaphore, payload_value);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER, sampler_bundle,
      wait_list, execution->sampler_done_semaphore, payload_value,
      diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_decode(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_bundle_t* decode_bundle, uint64_t sampler_payload,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_hal_semaphore_t* wait_semaphores[3];
  uint64_t wait_payload_values[3];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count, execution->upload_semaphore,
                                     execution->upload_payload_value);
  id4_ideogram4_generation_push_wait(
      wait_semaphores, wait_payload_values, &wait_count,
      execution->sampler_done_semaphore, sampler_payload);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_stage(
      execution, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE, decode_bundle,
      wait_list, execution->decode_done_semaphore, 1, diagnostics_sink));
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = 1;
  wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, execution->decode_done_semaphore,
      wait_payload_value);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      final_signal_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_find_outputs(
    id4_ideogram4_generation_execution_t* execution) {
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
           .boundary_bindings,
      IREE_SV("velocity"), &execution->conditioned_velocity_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle
          ->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .plan,
      &execution->bundle
           ->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
           .boundary_bindings,
      IREE_SV("velocity"), &execution->unconditioned_velocity_binding));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
           .boundary_bindings,
      IREE_SV("x_next"), &execution->final_latent_binding));
  return id4_pipeline_find_boundary_binding(
      execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE].plan,
      &execution->bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
           .boundary_bindings,
      IREE_SV("media.image.decoded"), &execution->decoded_image_binding);
}

static iree_status_t id4_ideogram4_generation_begin_execution(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_request_t* request, const iree_tokenizer_t* tokenizer,
    iree_tokenizer_encode_flags_t tokenizer_flags,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    id4_ideogram4_generation_execution_t** out_execution) {
  id4_ideogram4_generation_execution_t* execution = NULL;
  iree_status_t status = id4_ideogram4_generation_execution_allocate(
      bundle, session->host_allocator, &execution);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_qwen_lowering_options_t qwen_lowering_options;
    memset(&qwen_lowering_options, 0, sizeof(qwen_lowering_options));
    qwen_lowering_options.structure_size = sizeof(qwen_lowering_options);
    qwen_lowering_options.tokenizer = tokenizer;
    qwen_lowering_options.request = request;
    qwen_lowering_options.tokenizer_flags = tokenizer_flags;
    qwen_lowering_options.max_token_count = session->qwen_model.max_token_count;
    qwen_lowering_options.token_capacity = bundle->summary.qwen_token_capacity;
    qwen_lowering_options.vocab_size = session->qwen_model.vocab_size;
    status = id4_ideogram4_request_lower_qwen_inputs(&qwen_lowering_options,
                                                     execution->host_allocator,
                                                     &execution->qwen_inputs);
  }
  if (iree_status_is_ok(status) &&
      execution->qwen_inputs.token_count != bundle->summary.qwen_token_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation issue Qwen token count %" PRIu32
        " does not match prepared bundle count %" PRIu32,
        execution->qwen_inputs.token_count, bundle->summary.qwen_token_count);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_dit_lowering_options_t dit_lowering_options;
    memset(&dit_lowering_options, 0, sizeof(dit_lowering_options));
    dit_lowering_options.structure_size = sizeof(dit_lowering_options);
    dit_lowering_options.generation = &request->generation;
    dit_lowering_options.text_token_count = execution->qwen_inputs.token_count;
    dit_lowering_options.attention_head_size =
        session->dit_model.hidden_size /
        session->dit_model.attention_head_count;
    status = id4_ideogram4_request_lower_dit_inputs(&dit_lowering_options,
                                                    execution->host_allocator,
                                                    &execution->dit_inputs);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_sampler_preset_lower_schedule(
        request->generation.sampler_preset,
        (uint32_t)bundle->summary.decoded_image_shape.dims[0],
        (uint32_t)bundle->summary.decoded_image_shape.dims[1],
        execution->host_allocator, &execution->denoise_schedule);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_execution_create_semaphores(execution);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(wait_semaphore_list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_qwen_inputs(
        execution, &execution->qwen_inputs, iree_hal_semaphore_list_empty());
  }
  if (iree_status_is_ok(status)) {
    execution->qwen_upload_payload_value = execution->upload_payload_value;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_noise_seed(
        execution, request->generation.seed, iree_hal_semaphore_list_empty());
  }
  if (iree_status_is_ok(status)) {
    execution->seed_upload_payload_value = execution->upload_payload_value;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_dit_branch_inputs(
        execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
        &execution->dit_inputs.conditioned);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_dit_branch_inputs(
        execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
        &execution->dit_inputs.unconditioned);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_upload_denoise_step(
        execution, &execution->denoise_schedule.steps[0]);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_find_outputs(execution);
  }
  if (iree_status_is_ok(status) && signal_semaphore_list.count != 0) {
    iree_hal_semaphore_t* wait_semaphore = NULL;
    uint64_t wait_payload_value = execution->upload_payload_value;
    iree_hal_semaphore_list_t upload_wait_list =
        id4_ideogram4_single_semaphore_list(
            &wait_semaphore, &wait_payload_value, execution->upload_semaphore,
            wait_payload_value);
    status = iree_hal_device_queue_barrier(
        bundle->device, bundle->queue_affinity, upload_wait_list,
        signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

iree_status_t id4_ideogram4_session_begin_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_begin_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_begin_options(
      session, bundle, options));
  return id4_ideogram4_generation_begin_execution(
      session, bundle, options->request, options->tokenizer,
      options->tokenizer_flags, options->wait_semaphore_list,
      options->signal_semaphore_list, out_execution);
}

static iree_status_t id4_ideogram4_generation_issue_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_qwen(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN),
      execution->qwen_upload_payload_value, diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN]
      .was_issued = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_noise(
        execution,
        id4_ideogram4_generation_phase_stage_bundle(
            phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE),
        execution->seed_upload_payload_value, diagnostics_sink);
    phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_NOISE]
        .was_issued = iree_status_is_ok(status);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_wait_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution) {
  iree_hal_semaphore_t* qwen_done_semaphore = execution->qwen_done_semaphore;
  uint64_t qwen_done_payload_value = 1;
  iree_hal_semaphore_list_t qwen_done_list = {
      // One Qwen completion edge required before allocating DiT weights.
      .count = 1,
      // Stack-backed Qwen completion semaphore.
      .semaphores = &qwen_done_semaphore,
      // Stack-backed Qwen completion payload value.
      .payload_values = &qwen_done_payload_value,
  };
  return iree_hal_semaphore_list_wait(qwen_done_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_wait_one(
    iree_hal_semaphore_t* semaphore, uint64_t payload_value) {
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = payload_value;
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, semaphore, payload_value);
  return iree_hal_semaphore_list_wait(wait_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_check_phase_bundle_failures(
    const id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    id4_pipeline_bundle_t* stage_bundle =
        phase_bundle->stage_bundle_refs[i].bundle;
    if (!stage_bundle) continue;
    status =
        iree_status_join(status, id4_pipeline_bundle_check_readiness_failures(
                                     stage_bundle, diagnostics_sink));
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_wait_stage_bundle(
    id4_pipeline_bundle_t* stage_bundle, iree_hal_semaphore_t* semaphore,
    uint64_t payload_value, id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status =
      id4_ideogram4_generation_wait_one(semaphore, payload_value);
  if (!iree_status_is_ok(status) && stage_bundle) {
    status =
        iree_status_join(status, id4_pipeline_bundle_check_readiness_failures(
                                     stage_bundle, diagnostics_sink));
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_qwen_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t qwen_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
      ID4_IDEOGRAM4_GENERATION_PHASE_NONE, iree_hal_semaphore_list_empty(),
      kernel_diagnostic_artifact_flags, diagnostics_sink, &qwen_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_qwen(
        execution, qwen_ref.bundle, execution->qwen_upload_payload_value,
        diagnostics_sink);
    qwen_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_stage_bundle(
        qwen_ref.bundle, execution->qwen_done_semaphore, 1, diagnostics_sink);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&qwen_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_noise_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t noise_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
      ID4_IDEOGRAM4_GENERATION_PHASE_NONE, iree_hal_semaphore_list_empty(),
      kernel_diagnostic_artifact_flags, diagnostics_sink, &noise_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_noise(
        execution, noise_ref.bundle, execution->seed_upload_payload_value,
        diagnostics_sink);
    noise_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_stage_bundle(
        noise_ref.bundle, execution->noise_done_semaphore, 1, diagnostics_sink);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&noise_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_dit_branch_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_hal_semaphore_t* latent_semaphore, uint64_t latent_payload_value,
    iree_hal_semaphore_t* condition_semaphore, uint64_t condition_payload_value,
    iree_hal_semaphore_t* branch_done_semaphore, uint64_t branch_payload_value,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t branch_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, stage_ordinal, ID4_IDEOGRAM4_GENERATION_PHASE_NONE,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink, &branch_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_dit_branch(
        execution, stage_ordinal, branch_ref.bundle, latent_semaphore,
        latent_payload_value, condition_semaphore, condition_payload_value,
        branch_done_semaphore, branch_payload_value, diagnostics_sink);
    branch_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_stage_bundle(
        branch_ref.bundle, branch_done_semaphore, branch_payload_value,
        diagnostics_sink);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&branch_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_sampler_stage_serial(
    id4_ideogram4_generation_execution_t* execution, uint64_t payload_value,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t sampler_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
      ID4_IDEOGRAM4_GENERATION_PHASE_NONE, iree_hal_semaphore_list_empty(),
      kernel_diagnostic_artifact_flags, diagnostics_sink, &sampler_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_sampler(
        execution, sampler_ref.bundle, payload_value, diagnostics_sink);
    sampler_ref.was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_stage_bundle(
        sampler_ref.bundle, execution->sampler_done_semaphore, payload_value,
        diagnostics_sink);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&sampler_ref));
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_stage_bundle_ref_t decode_ref = {0};
  iree_status_t status = id4_ideogram4_generation_acquire_stage_bundle_ref(
      execution->bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
      ID4_IDEOGRAM4_GENERATION_PHASE_NONE, iree_hal_semaphore_list_empty(),
      kernel_diagnostic_artifact_flags, diagnostics_sink, &decode_ref);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_decode(
        execution, decode_ref.bundle, execution->denoise_schedule.step_count,
        final_signal_list, diagnostics_sink);
    decode_ref.was_issued = iree_status_is_ok(status);
  }
  status = iree_status_join(
      status, id4_ideogram4_generation_release_stage_bundle_ref(&decode_ref));
  return status;
}

static bool id4_ideogram4_generation_uses_phase_stage_bundles(
    const id4_ideogram4_generation_bundle_t* bundle) {
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_PHASE_COUNT; ++i) {
    if (bundle->residency_policy.phase_stage_masks[i] !=
        ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
      return true;
    }
  }
  return false;
}

static iree_status_t id4_ideogram4_generation_issue_conditioning_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_qwen_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink));
  return id4_ideogram4_generation_issue_noise_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_denoise_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch_stage_serial(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          kernel_diagnostic_artifact_flags, diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch_stage_serial(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          kernel_diagnostic_artifact_flags, diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_sampler_stage_serial(
          execution, payload_value, kernel_diagnostic_artifact_flags,
          diagnostics_sink);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_conditioning_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_denoise_stage_serial(
      execution, kernel_diagnostic_artifact_flags, diagnostics_sink));
  return id4_ideogram4_generation_issue_decode_stage_serial(
      execution, final_signal_list, kernel_diagnostic_artifact_flags,
      diagnostics_sink);
}

static iree_status_t
id4_ideogram4_generation_issue_conditioning_phase_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_bundle_t* qwen_bundle =
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN);
  iree_status_t status = id4_ideogram4_generation_issue_qwen(
      execution, qwen_bundle, execution->qwen_upload_payload_value,
      diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN]
      .was_issued = iree_status_is_ok(status);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_stage_bundle(
        qwen_bundle, execution->qwen_done_semaphore, 1, diagnostics_sink);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_bundle_t* noise_bundle =
        id4_ideogram4_generation_phase_stage_bundle(
            phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE);
    status = id4_ideogram4_generation_issue_noise(
        execution, noise_bundle, execution->seed_upload_payload_value,
        diagnostics_sink);
    phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_NOISE]
        .was_issued = iree_status_is_ok(status);
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_wait_stage_bundle(
          noise_bundle, execution->noise_done_semaphore, 1, diagnostics_sink);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_denoise_phase_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      id4_pipeline_bundle_t* conditioned_bundle =
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED);
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          conditioned_bundle, latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .was_issued |= iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_wait_stage_bundle(
            conditioned_bundle, execution->dit_conditioned_done_semaphore,
            payload_value, diagnostics_sink);
      }
    }
    if (iree_status_is_ok(status)) {
      id4_pipeline_bundle_t* unconditioned_bundle =
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED);
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          unconditioned_bundle, latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .was_issued |= iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_wait_stage_bundle(
            unconditioned_bundle, execution->dit_unconditioned_done_semaphore,
            payload_value, diagnostics_sink);
      }
    }
    if (iree_status_is_ok(status)) {
      id4_pipeline_bundle_t* sampler_bundle =
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER);
      status = id4_ideogram4_generation_issue_sampler(
          execution, sampler_bundle, payload_value, diagnostics_sink);
      phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
          .was_issued |= iree_status_is_ok(status);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_wait_stage_bundle(
            sampler_bundle, execution->sampler_done_semaphore, payload_value,
            diagnostics_sink);
      }
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_phase_stage_serial(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_decode(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE),
      execution->denoise_schedule.step_count, final_signal_list,
      diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
      .was_issued = iree_status_is_ok(status);
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_stage_serial_phase_bundle(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_mask_t phase_mask,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_ideogram4_generation_phase_bundle_t phase_bundle = {0};
  id4_ideogram4_generation_phase_bundle_initialize(phase_mask, &phase_bundle);
  iree_status_t status = id4_ideogram4_generation_prepare_phase_bundle(
      execution->bundle, phase_mask, iree_hal_semaphore_list_empty(),
      kernel_diagnostic_artifact_flags, diagnostics_sink, &phase_bundle);
  if (iree_status_is_ok(status)) {
    switch (phase_mask) {
      case ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING:
        status = id4_ideogram4_generation_issue_conditioning_phase_stage_serial(
            execution, &phase_bundle, diagnostics_sink);
        break;
      case ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE:
        status = id4_ideogram4_generation_issue_denoise_phase_stage_serial(
            execution, &phase_bundle, diagnostics_sink);
        break;
      case ID4_IDEOGRAM4_GENERATION_PHASE_DECODE:
        status = id4_ideogram4_generation_issue_decode_phase_stage_serial(
            execution, &phase_bundle, final_signal_list, diagnostics_sink);
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 generation phase mask 0x%x does not identify one "
            "generation phase",
            phase_mask);
        break;
    }
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, id4_ideogram4_generation_check_phase_bundle_failures(
                    &phase_bundle, diagnostics_sink));
  }
  return iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&phase_bundle));
}

static iree_status_t id4_ideogram4_generation_issue_stage_serial_phase_bundles(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_stage_serial_phase_bundle(
      execution, ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_issue_stage_serial_phase_bundle(
      execution, ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
      iree_hal_semaphore_list_empty(), kernel_diagnostic_artifact_flags,
      diagnostics_sink));
  return id4_ideogram4_generation_issue_stage_serial_phase_bundle(
      execution, ID4_IDEOGRAM4_GENERATION_PHASE_DECODE, final_signal_list,
      kernel_diagnostic_artifact_flags, diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_stage_serial_dispatch(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_kernel_diagnostic_artifact_flags_t
        kernel_diagnostic_artifact_flags,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (id4_ideogram4_generation_uses_phase_stage_bundles(execution->bundle)) {
    return id4_ideogram4_generation_issue_stage_serial_phase_bundles(
        execution, final_signal_list, kernel_diagnostic_artifact_flags,
        diagnostics_sink);
  }
  return id4_ideogram4_generation_issue_stage_serial(
      execution, final_signal_list, kernel_diagnostic_artifact_flags,
      diagnostics_sink);
}

static iree_status_t id4_ideogram4_generation_issue_denoise_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < execution->denoise_schedule.step_count && iree_status_is_ok(status);
       ++i) {
    const uint64_t payload_value = (uint64_t)i + 1;
    iree_hal_semaphore_t* latent_semaphore =
        i == 0 ? execution->noise_done_semaphore
               : execution->sampler_done_semaphore;
    const uint64_t latent_payload_value = i == 0 ? 1 : (uint64_t)i;
    if (i > 0) {
      status = id4_ideogram4_generation_chain_upload_after_sampler(execution,
                                                                   (uint64_t)i);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_upload_denoise_step(
            execution, &execution->denoise_schedule.steps[i]);
      }
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED),
          latent_semaphore, latent_payload_value,
          execution->qwen_done_semaphore, 1,
          execution->dit_conditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED]
          .was_issued |= iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_dit_branch(
          execution, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED),
          latent_semaphore, latent_payload_value, NULL, 0,
          execution->dit_unconditioned_done_semaphore, payload_value,
          diagnostics_sink);
      phase_bundle
          ->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED]
          .was_issued |= iree_status_is_ok(status);
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_sampler(
          execution,
          id4_ideogram4_generation_phase_stage_bundle(
              phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER),
          payload_value, diagnostics_sink);
      phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER]
          .was_issued |= iree_status_is_ok(status);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_issue_decode_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    iree_hal_semaphore_list_t final_signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_ideogram4_generation_issue_decode(
      execution,
      id4_ideogram4_generation_phase_stage_bundle(
          phase_bundle, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE),
      execution->denoise_schedule.step_count, final_signal_list,
      diagnostics_sink);
  phase_bundle->stage_bundle_refs[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE]
      .was_issued = iree_status_is_ok(status);
  return status;
}

static iree_status_t id4_ideogram4_generation_signal_conditioning_phase(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_semaphore_t* wait_semaphores[2];
  uint64_t wait_payload_values[2];
  iree_host_size_t wait_count = 0;
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count,
                                     execution->qwen_done_semaphore, 1);
  id4_ideogram4_generation_push_wait(wait_semaphores, wait_payload_values,
                                     &wait_count,
                                     execution->noise_done_semaphore, 1);
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_generation_make_wait_list(
      wait_semaphores, wait_payload_values, wait_count);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t id4_ideogram4_generation_signal_denoise_phase(
    id4_ideogram4_generation_execution_t* execution,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_hal_semaphore_t* wait_semaphore = NULL;
  uint64_t wait_payload_value = execution->denoise_schedule.step_count;
  iree_hal_semaphore_list_t wait_list = id4_ideogram4_single_semaphore_list(
      &wait_semaphore, &wait_payload_value, execution->sampler_done_semaphore,
      wait_payload_value);
  return iree_hal_device_queue_barrier(
      execution->bundle->device, execution->bundle->queue_affinity, wait_list,
      signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

iree_status_t id4_ideogram4_generation_execution_issue_phase(
    id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_phase_bundle_t* phase_bundle,
    const id4_ideogram4_generation_phase_issue_options_t* options) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_phase_issue_options(
      execution, phase_bundle, options));
  execution->parameter_load_prefetch_segment_distance =
      options->parameter_load_prefetch_segment_distance;
  execution->stage_issue_flags = options->stage_issue_flags;
  iree_status_t status = id4_ideogram4_generation_chain_phase_issue_wait(
      execution, phase_bundle->phase_mask, options->wait_semaphore_list);
  if (!iree_status_is_ok(status)) return status;
  switch (phase_bundle->phase_mask) {
    case ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING:
      status = id4_ideogram4_generation_issue_conditioning_phase(
          execution, phase_bundle, options->diagnostics_sink);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_signal_conditioning_phase(
            execution, options->signal_semaphore_list);
      }
      return status;
    case ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE:
      status = id4_ideogram4_generation_issue_denoise_phase(
          execution, phase_bundle, options->diagnostics_sink);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_generation_signal_denoise_phase(
            execution, options->signal_semaphore_list);
      }
      return status;
    case ID4_IDEOGRAM4_GENERATION_PHASE_DECODE:
      return id4_ideogram4_generation_issue_decode_phase(
          execution, phase_bundle, options->signal_semaphore_list,
          options->diagnostics_sink);
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase mask 0x%x does not identify one "
          "generation phase",
          phase_bundle->phase_mask);
  }
}

iree_status_t id4_ideogram4_session_issue_generation(
    id4_ideogram4_session_t* session, id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_issue_options_t* options,
    id4_ideogram4_generation_execution_t** out_execution) {
  IREE_ASSERT_ARGUMENT(out_execution);
  *out_execution = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_issue_options(
      session, bundle, options));

  id4_ideogram4_generation_execution_t* execution = NULL;
  if (options->issue_policy ==
      ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL) {
    iree_status_t status = id4_ideogram4_generation_begin_execution(
        session, bundle, options->request, options->tokenizer,
        options->tokenizer_flags, options->wait_semaphore_list,
        iree_hal_semaphore_list_empty(), &execution);
    if (iree_status_is_ok(status)) {
      execution->parameter_load_prefetch_segment_distance =
          options->parameter_load_prefetch_segment_distance;
      execution->stage_issue_flags = options->stage_issue_flags;
    }
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_generation_issue_stage_serial_dispatch(
          execution, options->signal_semaphore_list,
          options->kernel_diagnostic_artifact_flags, options->diagnostics_sink);
    }
    if (iree_status_is_ok(status)) {
      *out_execution = execution;
    } else {
      status = iree_status_join(
          status, id4_ideogram4_generation_bundle_check_resident_failures(
                      bundle, options->diagnostics_sink));
      id4_ideogram4_generation_execution_release(execution);
    }
    return status;
  }

  id4_ideogram4_generation_phase_bundle_t conditioning_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING, &conditioning_phase);
  id4_ideogram4_generation_phase_bundle_t denoise_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE, &denoise_phase);
  id4_ideogram4_generation_phase_bundle_t decode_phase = {0};
  id4_ideogram4_generation_phase_bundle_initialize(
      ID4_IDEOGRAM4_GENERATION_PHASE_DECODE, &decode_phase);

  iree_status_t status = id4_ideogram4_generation_begin_execution(
      session, bundle, options->request, options->tokenizer,
      options->tokenizer_flags, options->wait_semaphore_list,
      iree_hal_semaphore_list_empty(), &execution);
  if (iree_status_is_ok(status)) {
    execution->parameter_load_prefetch_segment_distance =
        options->parameter_load_prefetch_segment_distance;
    execution->stage_issue_flags = options->stage_issue_flags;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_CONDITIONING,
        iree_hal_semaphore_list_empty(),
        options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
        &conditioning_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_conditioning_phase(
        execution, &conditioning_phase, options->diagnostics_sink);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_wait_conditioning_phase(execution);
  }
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, id4_ideogram4_generation_check_phase_bundle_failures(
                    &conditioning_phase, options->diagnostics_sink));
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&conditioning_phase));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DENOISE,
        iree_hal_semaphore_list_empty(),
        options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
        &denoise_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_denoise_phase(
        execution, &denoise_phase, options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&denoise_phase));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_prepare_phase_bundle(
        bundle, ID4_IDEOGRAM4_GENERATION_PHASE_DECODE,
        iree_hal_semaphore_list_empty(),
        options->kernel_diagnostic_artifact_flags, options->diagnostics_sink,
        &decode_phase);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_issue_decode_phase(
        execution, &decode_phase, options->signal_semaphore_list,
        options->diagnostics_sink);
  }
  status = iree_status_join(
      status,
      id4_ideogram4_generation_phase_bundle_deinitialize(&decode_phase));
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(
        status, id4_ideogram4_generation_bundle_check_resident_failures(
                    bundle, options->diagnostics_sink));
  }
  if (iree_status_is_ok(status)) {
    *out_execution = execution;
  } else {
    id4_ideogram4_generation_execution_release(execution);
  }
  return status;
}

void id4_ideogram4_generation_execution_release(
    id4_ideogram4_generation_execution_t* execution) {
  if (!execution) return;
  id4_ideogram4_denoise_schedule_deinitialize(&execution->denoise_schedule,
                                              execution->host_allocator);
  id4_ideogram4_dit_inputs_deinitialize(&execution->dit_inputs,
                                        execution->host_allocator);
  id4_ideogram4_qwen_inputs_deinitialize(&execution->qwen_inputs,
                                         execution->host_allocator);
  iree_hal_semaphore_release(execution->decode_done_semaphore);
  iree_hal_semaphore_release(execution->sampler_done_semaphore);
  iree_hal_semaphore_release(execution->dit_unconditioned_done_semaphore);
  iree_hal_semaphore_release(execution->dit_conditioned_done_semaphore);
  iree_hal_semaphore_release(execution->noise_done_semaphore);
  iree_hal_semaphore_release(execution->qwen_done_semaphore);
  iree_hal_semaphore_release(execution->upload_semaphore);
  id4_ideogram4_generation_bundle_release(execution->bundle);
  iree_allocator_t host_allocator = execution->host_allocator;
  iree_allocator_free(host_allocator, execution);
}

iree_status_t id4_ideogram4_generation_execution_check_failures(
    const id4_ideogram4_generation_execution_t* execution,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!execution) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation execution is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      diagnostics_sink, IREE_SV("Ideogram 4 generation failure check")));
  return id4_ideogram4_generation_bundle_check_resident_failures(
      execution->bundle, diagnostics_sink);
}

iree_status_t id4_ideogram4_generation_execution_result(
    const id4_ideogram4_generation_execution_t* execution,
    id4_ideogram4_generation_result_t* out_result) {
  if (!execution || !out_result) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation execution and result output are required");
  }
  out_result->conditioned_velocity_binding =
      execution->conditioned_velocity_binding;
  out_result->unconditioned_velocity_binding =
      execution->unconditioned_velocity_binding;
  out_result->final_latent_binding = execution->final_latent_binding;
  out_result->decoded_image_binding = execution->decoded_image_binding;
  return iree_ok_status();
}

iree_status_t id4_ideogram4_generation_execution_qwen_inputs(
    const id4_ideogram4_generation_execution_t* execution,
    const id4_ideogram4_qwen_inputs_t** out_inputs) {
  if (!execution || !out_inputs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation execution and Qwen input output are required");
  }
  *out_inputs = &execution->qwen_inputs;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_find_diagnostic_tap_plan(
    const id4_pipeline_plan_t* plan, iree_string_view_t tap_name,
    const id4_pipeline_diagnostic_tap_plan_t** out_tap) {
  *out_tap = NULL;
  const iree_host_size_t tap_count =
      id4_pipeline_plan_diagnostic_tap_count(plan);
  for (iree_host_size_t i = 0; i < tap_count; ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (tap && iree_string_view_equal(tap->name, tap_name)) {
      *out_tap = tap;
      return iree_ok_status();
    }
  }
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation stage %.*s has no diagnostic tap `%.*s`",
      (int)stage_name.size, stage_name.data, (int)tap_name.size, tap_name.data);
}

iree_status_t id4_ideogram4_generation_execution_find_diagnostic_tap(
    const id4_ideogram4_generation_execution_t* execution,
    iree_string_view_t stage_key, iree_string_view_t tap_name,
    const id4_pipeline_tensor_layout_t** out_layout,
    iree_hal_buffer_binding_t* out_binding) {
  if (!execution || iree_string_view_is_empty(stage_key) ||
      iree_string_view_is_empty(tap_name) || !out_layout || !out_binding) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation diagnostic tap lookup requires execution, "
        "stage key, tap name, layout output, and binding output");
  }
  *out_layout = NULL;
  memset(out_binding, 0, sizeof(*out_binding));
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor_for_key(stage_key);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "Ideogram 4 generation has no stage `%.*s`",
                            (int)stage_key.size, stage_key.data);
  }
  const id4_ideogram4_generation_stage_slot_t* slot =
      &execution->bundle->stages[descriptor->ordinal];
  const id4_pipeline_diagnostic_tap_plan_t* tap = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_diagnostic_tap_plan(
      slot->plan, tap_name, &tap));
  IREE_RETURN_IF_ERROR(id4_pipeline_find_diagnostic_tap_binding(
      slot->plan, &slot->diagnostic_tap_bindings, tap_name, out_binding));
  *out_layout = &tap->layout;
  return iree_ok_status();
}
