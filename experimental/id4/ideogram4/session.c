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

struct id4_ideogram4_generation_plan_t {
  // Allocator used for plan-owned storage.
  iree_allocator_t host_allocator;
  // Session retained for the lifetime of the generation plan.
  id4_ideogram4_session_t* session;
  // Stable generation-level shape and scheduling summary.
  id4_ideogram4_generation_plan_summary_t summary;
  // Planned Qwen3-VL prompt conditioning stage.
  id4_pipeline_plan_t* qwen_plan;
  // Planned conditioned Ideogram 4 DiT stage.
  id4_pipeline_plan_t* dit_conditioned_plan;
  // Planned unconditioned Ideogram 4 DiT stage.
  id4_pipeline_plan_t* dit_unconditioned_plan;
  // Planned device-side sampler denoise-step stage.
  id4_pipeline_plan_t* sampler_plan;
  // Planned VAE-backed latent decode stage.
  id4_pipeline_plan_t* decode_plan;
};

typedef enum id4_ideogram4_generation_stage_ordinal_e {
  ID4_IDEOGRAM4_GENERATION_STAGE_QWEN = 0,
  ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED = 1,
  ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED = 2,
  ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER = 3,
  ID4_IDEOGRAM4_GENERATION_STAGE_DECODE = 4,
  ID4_IDEOGRAM4_GENERATION_STAGE_COUNT = 5,
} id4_ideogram4_generation_stage_ordinal_t;

typedef struct id4_ideogram4_generation_stage_slot_t {
  // Session-owned stage implementation selected for this slot.
  id4_pipeline_stage_t* stage;
  // Retained plan for this coarse stage.
  id4_pipeline_plan_t* plan;
  // Prepared reusable bundle for this coarse stage.
  id4_pipeline_bundle_t* bundle;
  // Owned boundary tensor buffers in plan order.
  id4_pipeline_buffer_binding_set_t boundary_bindings;
  // Owned diagnostic tap buffers in plan order.
  id4_pipeline_buffer_binding_set_t diagnostic_tap_bindings;
  // Semaphore signaled when this stage's async parameter loads complete.
  iree_hal_semaphore_t* prepare_semaphore;
} id4_ideogram4_generation_stage_slot_t;

typedef struct id4_ideogram4_generation_boundary_alias_t {
  // Stage slot producing the replacement binding.
  id4_ideogram4_generation_stage_ordinal_t source_stage;
  // Boundary tensor exported by the producer stage.
  iree_string_view_t source_name;
  // Stage slot consuming the replacement binding.
  id4_ideogram4_generation_stage_ordinal_t target_stage;
  // Boundary tensor imported by the consumer stage.
  iree_string_view_t target_name;
} id4_ideogram4_generation_boundary_alias_t;

struct id4_ideogram4_generation_bundle_t {
  // Allocator used for bundle-owned storage.
  iree_allocator_t host_allocator;
  // Session retained for the lifetime of this prepared generation.
  id4_ideogram4_session_t* session;
  // Stable generation-level shape and scheduling summary.
  id4_ideogram4_generation_plan_summary_t summary;
  // Device shared by every prepared stage in the current single-device plan.
  iree_hal_device_t* device;
  // Queue affinity shared by every prepared stage in the current plan.
  iree_hal_queue_affinity_t queue_affinity;
  // Coarse stage slots owned by this generation bundle.
  id4_ideogram4_generation_stage_slot_t
      stages[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
};

static const id4_ideogram4_generation_boundary_alias_t
    id4_ideogram4_generation_boundary_aliases[] = {
        {
            // Qwen text conditioning consumed by the conditioned DiT branch.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
            .source_name = IREE_SVL("condition"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
            .target_name = IREE_SVL("condition"),
        },
        {
            // Conditioned DiT velocity consumed by CFG sampling.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
            .source_name = IREE_SVL("velocity"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .target_name = IREE_SVL("cond_out"),
        },
        {
            // Unconditioned DiT velocity consumed by CFG sampling.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
            .source_name = IREE_SVL("velocity"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .target_name = IREE_SVL("uncond_out"),
        },
        {
            // Final Euler latent consumed by VAE-backed decode.
            .source_stage = ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
            .source_name = IREE_SVL("x_next"),
            .target_stage = ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
            .target_name = IREE_SVL("media.latent.diffusion"),
        },
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

static iree_status_t id4_ideogram4_validate_generation_config(
    const id4_ideogram4_session_t* session,
    id4_ideogram4_generation_config_t config) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      config.structure_size, sizeof(config),
      IREE_SV("Ideogram 4 generation config")));
  if (config.next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation config extension structures are not supported");
  }
  if (config.denoise_step_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation step count is zero");
  }
  if (config.diffusion_latent_shape.rank != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation latent shape rank must be "
                            "4");
  }
  if (config.diffusion_latent_shape.dims[2] !=
      session->dit_model.input_channel_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation latent channel count %" PRIu64
        " does not match DiT channel count %" PRIu32,
        config.diffusion_latent_shape.dims[2],
        session->dit_model.input_channel_count);
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_decode_program_validate_diffusion_latent_shape(
          session->decode_model, config.diffusion_latent_shape));
  switch (config.dit_activation_format) {
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL:
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT activation format %" PRIu32 " is invalid",
          (uint32_t)config.dit_activation_format);
  }
  switch (config.vae_tiling.mode) {
    case ID4_VAE_TILING_MODE_DISABLED:
    case ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE:
    case ID4_VAE_TILING_MODE_RELATIVE_TILE_SIZE:
    case ID4_VAE_TILING_MODE_MEMORY_BUDGET:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation VAE tiling mode %" PRIu32
                              " is invalid",
                              (uint32_t)config.vae_tiling.mode);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_generation_plan_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Ideogram 4 session must be loaded before "
                            "generation planning");
  }
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation plan")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation plan extension structures are not supported");
  }
  if (!options->request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation request is required");
  }
  if (!options->tokenizer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation tokenizer is required");
  }
  const iree_host_size_t device_count = iree_hal_device_group_device_count(
      id4_pipeline_stage_services(session->qwen_stage)->device_group);
  if (options->device_index >= device_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation device index %" PRIhsz
                            " exceeds device count %" PRIhsz,
                            options->device_index, device_count);
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_config(session, options->generation));
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation plan")));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_allocate(
    id4_ideogram4_session_t* session, iree_allocator_t host_allocator,
    id4_ideogram4_generation_plan_t** out_plan) {
  *out_plan = NULL;
  id4_ideogram4_generation_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*plan), (void**)&plan));
  memset(plan, 0, sizeof(*plan));
  plan->host_allocator = host_allocator;
  plan->session = session;
  id4_ideogram4_session_retain(session);
  *out_plan = plan;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_plan_stage(
    id4_pipeline_stage_t* stage, const void* stage_options,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = stage_options;
  plan_options.device_index = options->device_index;
  plan_options.queue_affinity = options->queue_affinity;
  plan_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_qwen(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    uint32_t token_count, id4_pipeline_plan_t** out_plan) {
  id4_qwen3_vl_stage_plan_options_t qwen_options;
  memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = token_count;
  return id4_ideogram4_plan_stage(session->qwen_stage, &qwen_options, options,
                                  out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_dit(
    id4_pipeline_stage_t* stage,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode,
    uint32_t text_token_count, id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_dit_stage_plan_options_t dit_options;
  memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request.latent_shape = options->generation.diffusion_latent_shape;
  dit_options.request.conditioning_mode = conditioning_mode;
  dit_options.request.text_token_count = text_token_count;
  dit_options.activation_format = options->generation.dit_activation_format;
  return id4_ideogram4_plan_stage(stage, &dit_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_sampler(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_denoise_stage_plan_options_t sampler_options;
  memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape =
      options->generation.diffusion_latent_shape;
  return id4_ideogram4_plan_stage(session->sampler_stage, &sampler_options,
                                  options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_decode(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_decode_stage_plan_options_t decode_options;
  memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape =
      options->generation.diffusion_latent_shape;
  decode_options.request.vae_tiling = options->generation.vae_tiling;
  return id4_ideogram4_plan_stage(session->decode_stage, &decode_options,
                                  options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_stages(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_plan_t* plan) {
  uint32_t token_count = 0;
  id4_ideogram4_qwen_lowering_options_t qwen_lowering_options;
  memset(&qwen_lowering_options, 0, sizeof(qwen_lowering_options));
  qwen_lowering_options.structure_size = sizeof(qwen_lowering_options);
  qwen_lowering_options.tokenizer = options->tokenizer;
  qwen_lowering_options.request = options->request;
  qwen_lowering_options.tokenizer_flags = options->tokenizer_flags;
  qwen_lowering_options.max_token_count = session->qwen_model.max_token_count;
  IREE_RETURN_IF_ERROR(id4_ideogram4_request_count_qwen_tokens(
      &qwen_lowering_options, session->host_allocator, &token_count));

  plan->summary.qwen_token_count = token_count;
  plan->summary.denoise_step_count = options->generation.denoise_step_count;
  plan->summary.diffusion_latent_shape =
      options->generation.diffusion_latent_shape;
  plan->summary.dit_activation_format =
      options->generation.dit_activation_format;
  plan->summary.vae_tiling = options->generation.vae_tiling;

  iree_status_t status = id4_ideogram4_plan_generation_qwen(
      session, options, token_count, &plan->qwen_plan);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_dit(
        session->dit_conditioned_stage, options,
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED, token_count,
        &plan->dit_conditioned_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_dit(
        session->dit_unconditioned_stage, options,
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED, 0,
        &plan->dit_unconditioned_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_sampler(session, options,
                                                   &plan->sampler_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_decode(session, options,
                                                  &plan->decode_plan);
  }
  return status;
}

iree_status_t id4_ideogram4_session_plan_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_plan_options(session, options));

  id4_ideogram4_generation_plan_t* plan = NULL;
  iree_status_t status = id4_ideogram4_generation_plan_allocate(
      session, session->host_allocator, &plan);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_stages(session, options, plan);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    id4_ideogram4_generation_plan_release(plan);
  }
  return status;
}

void id4_ideogram4_generation_plan_release(
    id4_ideogram4_generation_plan_t* plan) {
  if (!plan) return;
  id4_pipeline_plan_release(plan->decode_plan);
  id4_pipeline_plan_release(plan->sampler_plan);
  id4_pipeline_plan_release(plan->dit_unconditioned_plan);
  id4_pipeline_plan_release(plan->dit_conditioned_plan);
  id4_pipeline_plan_release(plan->qwen_plan);
  id4_ideogram4_session_release(plan->session);
  iree_allocator_t host_allocator = plan->host_allocator;
  iree_allocator_free(host_allocator, plan);
}

iree_status_t id4_ideogram4_generation_plan_summary(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_plan_summary_t* out_summary) {
  if (!plan || !out_summary) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan and summary output are required");
  }
  *out_summary = plan->summary;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_append_shape_json(
    iree_string_builder_t* builder, id4_pipeline_program_shape_t shape) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"rank\":%u,\"dims\":[", shape.rank));
  for (uint32_t i = 0; i < shape.rank; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu64, shape.dims[i]));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t id4_ideogram4_generation_plan_append_tiling_json(
    iree_string_builder_t* builder, id4_vae_tiling_config_t tiling) {
  return iree_string_builder_append_format(
      builder,
      "{\"mode\":%u,\"tile_size_x\":%" PRIu32 ",\"tile_size_y\":%" PRIu32
      ",\"relative_size_x\":%g"
      ",\"relative_size_y\":%g,\"overlap\":%g,\"memory_budget\":%" PRIu64 "}",
      (uint32_t)tiling.mode, tiling.tile_size_x, tiling.tile_size_y,
      (double)tiling.relative_size_x, (double)tiling.relative_size_y,
      (double)tiling.overlap, (uint64_t)tiling.memory_budget);
}

static iree_status_t id4_ideogram4_generation_plan_append_stage_json(
    iree_string_builder_t* builder, const char* key,
    const id4_pipeline_plan_t* stage_plan) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", key));
  return id4_pipeline_plan_format_json(stage_plan, builder);
}

iree_status_t id4_ideogram4_generation_plan_format_json(
    const id4_ideogram4_generation_plan_t* plan,
    iree_string_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "{\"kind\":\"ideogram4_generation\",\"summary\":{\"qwen_token_count\":"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "%" PRIu32 ",\"denoise_step_count\":%" PRIu32
      ",\"diffusion_latent_shape\":",
      plan->summary.qwen_token_count, plan->summary.denoise_step_count));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_shape_json(
      builder, plan->summary.diffusion_latent_shape));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"dit_activation_format\":%u,\"vae_tiling\":",
      (uint32_t)plan->summary.dit_activation_format));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_tiling_json(
      builder, plan->summary.vae_tiling));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "},\"stages\":{"));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, "qwen", plan->qwen_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, "dit_conditioned", plan->dit_conditioned_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, "dit_unconditioned", plan->dit_unconditioned_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, "sampler", plan->sampler_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, "decode", plan->decode_plan));
  return iree_string_builder_append_cstring(builder, "}}");
}

static iree_status_t id4_ideogram4_validate_generation_prepare_options(
    const id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options) {
  if (!session) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 session is required");
  }
  if (!session->is_loaded) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Ideogram 4 session must be loaded before generation preparation");
  }
  if (!plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan is required");
  }
  if (plan->session != session) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan was created by a different session");
  }
  if (!plan->qwen_plan || !plan->dit_conditioned_plan ||
      !plan->dit_unconditioned_plan || !plan->sampler_plan ||
      !plan->decode_plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan must contain every coarse stage plan");
  }
  if (!options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare options are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation prepare")));
  if (options->next) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Ideogram 4 generation prepare extension "
                            "structures are not supported");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("Ideogram 4 generation prepare "
                                            "wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_semaphore_list(
      options->signal_semaphore_list,
      IREE_SV("Ideogram 4 generation prepare signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare final signal is required");
  }
  if (!options->parameter_providers.qwen) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation Qwen parameter provider is required");
  }
  if (!options->parameter_providers.dit_conditioned) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation conditioned DiT parameter provider is required");
  }
  if (!options->parameter_providers.dit_unconditioned) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation unconditioned DiT "
                            "parameter provider is required");
  }
  if (!options->parameter_providers.vae) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation VAE parameter provider is required");
  }
  if (!options->kernel_library) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation prepare kernel library is required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 generation prepare")));
  return iree_ok_status();
}

static void id4_ideogram4_generation_stage_slot_deinitialize(
    id4_ideogram4_generation_stage_slot_t* slot) {
  id4_pipeline_buffer_binding_set_deinitialize(&slot->diagnostic_tap_bindings);
  id4_pipeline_buffer_binding_set_deinitialize(&slot->boundary_bindings);
  id4_pipeline_bundle_release(slot->bundle);
  id4_pipeline_plan_release(slot->plan);
  iree_hal_semaphore_release(slot->prepare_semaphore);
  memset(slot, 0, sizeof(*slot));
}

static void id4_ideogram4_generation_bundle_assign_slot(
    id4_ideogram4_generation_stage_slot_t* slot, id4_pipeline_stage_t* stage,
    id4_pipeline_plan_t* plan) {
  slot->stage = stage;
  slot->plan = plan;
  id4_pipeline_plan_retain(plan);
}

static iree_status_t id4_ideogram4_generation_bundle_allocate(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  *out_bundle = NULL;
  id4_ideogram4_generation_bundle_t* bundle = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*bundle), (void**)&bundle));
  memset(bundle, 0, sizeof(*bundle));
  bundle->host_allocator = host_allocator;
  bundle->session = session;
  bundle->summary = plan->summary;
  id4_ideogram4_session_retain(session);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_QWEN], session->qwen_stage,
      plan->qwen_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED],
      session->dit_conditioned_stage, plan->dit_conditioned_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED],
      session->dit_unconditioned_stage, plan->dit_unconditioned_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER],
      session->sampler_stage, plan->sampler_plan);
  id4_ideogram4_generation_bundle_assign_slot(
      &bundle->stages[ID4_IDEOGRAM4_GENERATION_STAGE_DECODE],
      session->decode_stage, plan->decode_plan);
  *out_bundle = bundle;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_stage_plan_placement(
    const id4_pipeline_plan_t* plan, iree_host_size_t* out_device_index,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  const iree_host_size_t placement_count =
      id4_pipeline_plan_placement_count(plan);
  if (placement_count != 1) {
    iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation stage %.*s has %" PRIhsz
        " placements; generation preparation currently requires one "
        "placement per coarse stage",
        (int)stage_name.size, stage_name.data, placement_count);
  }
  const id4_pipeline_device_placement_t* placement =
      id4_pipeline_plan_placement_at(plan, 0);
  if (!placement) {
    iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation stage %.*s placement is missing",
        (int)stage_name.size, stage_name.data);
  }
  *out_device_index = placement->device_index;
  *out_queue_affinity = placement->queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_select_placement(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_host_size_t device_index = 0;
  iree_hal_queue_affinity_t queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_plan_placement(
      bundle->stages[0].plan, &device_index, &queue_affinity));
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(bundle->stages[0].plan);
  for (iree_host_size_t i = 1; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    iree_host_size_t stage_device_index = 0;
    iree_hal_queue_affinity_t stage_queue_affinity =
        IREE_HAL_QUEUE_AFFINITY_ANY;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_plan_placement(
        bundle->stages[i].plan, &stage_device_index, &stage_queue_affinity));
    if (id4_pipeline_plan_device_group(bundle->stages[i].plan) !=
        device_group) {
      iree_string_view_t stage_name =
          id4_pipeline_plan_stage_name(bundle->stages[i].plan);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation stage %.*s was planned for a different "
          "device group",
          (int)stage_name.size, stage_name.data);
    }
    if (stage_device_index != device_index ||
        stage_queue_affinity != queue_affinity) {
      iree_string_view_t stage_name =
          id4_pipeline_plan_stage_name(bundle->stages[i].plan);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Ideogram 4 generation stage %.*s placement differs from the first "
          "stage; multi-placement generation preparation is not implemented",
          (int)stage_name.size, stage_name.data);
    }
  }
  bundle->device = iree_hal_device_group_device_at(device_group, device_index);
  if (!bundle->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation selected device %" PRIhsz
                            " is missing",
                            device_index);
  }
  bundle->queue_affinity = queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_allocate_bindings(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT && iree_status_is_ok(status);
       ++i) {
    id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[i];
    status = id4_pipeline_allocate_boundary_bindings(
        bundle->device, bundle->queue_affinity, slot->plan,
        bundle->host_allocator, &slot->boundary_bindings);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_allocate_diagnostic_tap_bindings(
          bundle->device, bundle->queue_affinity, slot->plan,
          bundle->host_allocator, &slot->diagnostic_tap_bindings);
    }
  }
  return status;
}

static bool id4_ideogram4_generation_shapes_equal(
    id4_pipeline_tensor_shape_t lhs, id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_status_t id4_ideogram4_generation_find_boundary_tensor(
    const id4_pipeline_plan_t* plan, iree_string_view_t name,
    const id4_pipeline_boundary_tensor_plan_t** out_boundary) {
  *out_boundary = NULL;
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_boundary = boundary;
      return iree_ok_status();
    }
  }
  iree_string_view_t stage_name = id4_pipeline_plan_stage_name(plan);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation stage %.*s has no boundary tensor `%.*s`",
      (int)stage_name.size, stage_name.data, (int)name.size, name.data);
}

static iree_status_t id4_ideogram4_generation_validate_boundary_alias(
    const id4_ideogram4_generation_stage_slot_t* source_slot,
    iree_string_view_t source_name,
    const id4_ideogram4_generation_stage_slot_t* target_slot,
    iree_string_view_t target_name) {
  const id4_pipeline_boundary_tensor_plan_t* source_boundary = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_boundary_tensor(
      source_slot->plan, source_name, &source_boundary));
  const id4_pipeline_boundary_tensor_plan_t* target_boundary = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_find_boundary_tensor(
      target_slot->plan, target_name, &target_boundary));

  iree_string_view_t source_stage_name =
      id4_pipeline_plan_stage_name(source_slot->plan);
  iree_string_view_t target_stage_name =
      id4_pipeline_plan_stage_name(target_slot->plan);
  if (!iree_all_bits_set(source_boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation alias source %.*s/%.*s is not exported",
        (int)source_stage_name.size, source_stage_name.data,
        (int)source_name.size, source_name.data);
  }
  if (!iree_all_bits_set(target_boundary->flags,
                         ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_IMPORTED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation alias target %.*s/%.*s is not imported",
        (int)target_stage_name.size, target_stage_name.data,
        (int)target_name.size, target_name.data);
  }
  const id4_pipeline_tensor_layout_t* source_layout = &source_boundary->layout;
  const id4_pipeline_tensor_layout_t* target_layout = &target_boundary->layout;
  if (source_layout->dtype != target_layout->dtype ||
      !id4_ideogram4_generation_shapes_equal(source_layout->shape,
                                             target_layout->shape) ||
      source_layout->byte_length != target_layout->byte_length ||
      source_layout->alignment != target_layout->alignment) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation boundary alias %.*s/%.*s to %.*s/%.*s has "
        "incompatible tensor layouts",
        (int)source_stage_name.size, source_stage_name.data,
        (int)source_name.size, source_name.data, (int)target_stage_name.size,
        target_stage_name.data, (int)target_name.size, target_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_apply_boundary_alias(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_boundary_alias_t* alias) {
  id4_ideogram4_generation_stage_slot_t* source_slot =
      &bundle->stages[alias->source_stage];
  id4_ideogram4_generation_stage_slot_t* target_slot =
      &bundle->stages[alias->target_stage];
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_validate_boundary_alias(
      source_slot, alias->source_name, target_slot, alias->target_name));

  iree_hal_buffer_binding_t replacement;
  IREE_RETURN_IF_ERROR(id4_pipeline_find_boundary_binding(
      source_slot->plan, &source_slot->boundary_bindings, alias->source_name,
      &replacement));
  return id4_pipeline_replace_boundary_binding(target_slot->plan,
                                               &target_slot->boundary_bindings,
                                               alias->target_name, replacement);
}

static iree_status_t id4_ideogram4_generation_bundle_apply_boundary_aliases(
    id4_ideogram4_generation_bundle_t* bundle) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_boundary_aliases); ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_bundle_apply_boundary_alias(
        bundle, &id4_ideogram4_generation_boundary_aliases[i]));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_bundle_create_prepare_semaphores(
    id4_ideogram4_generation_bundle_t* bundle) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT && iree_status_is_ok(status);
       ++i) {
    id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[i];
    if (id4_pipeline_plan_parameter_slab_count(slot->plan) == 0) continue;
    status = iree_hal_semaphore_create(bundle->device, bundle->queue_affinity,
                                       0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &slot->prepare_semaphore);
  }
  return status;
}

static iree_io_parameter_provider_t*
id4_ideogram4_generation_prepare_stage_parameter_provider(
    const id4_ideogram4_generation_parameter_providers_t* providers,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  iree_io_parameter_provider_t*
      stage_providers[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT] = {
          providers->qwen,
          providers->dit_conditioned,
          providers->dit_unconditioned,
          NULL,
          providers->vae,
      };
  if (stage_ordinal >= ID4_IDEOGRAM4_GENERATION_STAGE_COUNT) return NULL;
  return stage_providers[stage_ordinal];
}

static iree_status_t id4_ideogram4_generation_prepare_stage(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  id4_ideogram4_generation_stage_slot_t* slot = &bundle->stages[stage_ordinal];
  iree_hal_semaphore_t* signal_semaphore = NULL;
  uint64_t signal_payload_value = 1;
  iree_hal_semaphore_list_t signal_list = iree_hal_semaphore_list_empty();
  if (slot->prepare_semaphore) {
    signal_list = id4_ideogram4_single_semaphore_list(
        &signal_semaphore, &signal_payload_value, slot->prepare_semaphore,
        signal_payload_value);
  }

  id4_pipeline_stage_prepare_options_t prepare_options;
  memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider =
      id4_ideogram4_generation_prepare_stage_parameter_provider(
          &options->parameter_providers, stage_ordinal);
  prepare_options.kernel_library = options->kernel_library;
  prepare_options.wait_semaphore_list = slot->prepare_semaphore
                                            ? options->wait_semaphore_list
                                            : iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal_list;
  prepare_options.command_buffer_mode = options->command_buffer_mode;
  prepare_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_prepare(slot->stage, slot->plan, &prepare_options,
                                    &slot->bundle);
}

static iree_status_t id4_ideogram4_generation_bundle_prepare_stages(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT && iree_status_is_ok(status);
       ++i) {
    status = id4_ideogram4_generation_prepare_stage(
        bundle, options, (id4_ideogram4_generation_stage_ordinal_t)i);
  }
  return status;
}

static iree_status_t id4_ideogram4_generation_bundle_signal_readiness(
    id4_ideogram4_generation_bundle_t* bundle,
    const id4_ideogram4_generation_prepare_options_t* options) {
  iree_hal_semaphore_t* wait_semaphores[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
  uint64_t wait_payload_values[ID4_IDEOGRAM4_GENERATION_STAGE_COUNT];
  iree_host_size_t wait_count = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    iree_hal_semaphore_t* semaphore = bundle->stages[i].prepare_semaphore;
    if (!semaphore) continue;
    wait_semaphores[wait_count] = semaphore;
    wait_payload_values[wait_count] = 1;
    ++wait_count;
  }
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  if (wait_count != 0) {
    wait_list.count = wait_count;
    wait_list.semaphores = wait_semaphores;
    wait_list.payload_values = wait_payload_values;
  }
  return iree_hal_device_queue_barrier(
      bundle->device, bundle->queue_affinity, wait_list,
      options->signal_semaphore_list, IREE_HAL_EXECUTE_FLAG_NONE);
}

iree_status_t id4_ideogram4_session_prepare_generation(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_prepare_options_t* options,
    id4_ideogram4_generation_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = NULL;
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_prepare_options(
      session, plan, options));

  id4_ideogram4_generation_bundle_t* bundle = NULL;
  iree_status_t status = id4_ideogram4_generation_bundle_allocate(
      session, plan, session->host_allocator, &bundle);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_select_placement(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_allocate_bindings(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_apply_boundary_aliases(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_create_prepare_semaphores(bundle);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_prepare_stages(bundle, options);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_bundle_signal_readiness(bundle, options);
  }
  if (iree_status_is_ok(status)) {
    *out_bundle = bundle;
  } else {
    id4_ideogram4_generation_bundle_release(bundle);
  }
  return status;
}

void id4_ideogram4_generation_bundle_release(
    id4_ideogram4_generation_bundle_t* bundle) {
  if (!bundle) return;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT; ++i) {
    id4_ideogram4_generation_stage_slot_deinitialize(&bundle->stages[i]);
  }
  id4_ideogram4_session_release(bundle->session);
  iree_allocator_t host_allocator = bundle->host_allocator;
  iree_allocator_free(host_allocator, bundle);
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
