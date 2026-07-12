// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/ideogram4/session_parameters.h"
#include "experimental/id4/ideogram4/session_state.h"
#include "experimental/id4/ideogram4/session_support.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"

static iree_status_t id4_ideogram4_validate_qwen_parameter_format(
    const id4_ideogram4_session_create_options_t* options) {
  switch (options->qwen_parameter_format) {
    case ID4_QWEN3_VL_PARAMETER_FORMAT_BF16:
    case ID4_QWEN3_VL_PARAMETER_FORMAT_FP8_E4M3_BLOCK_SCALED:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 Qwen parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)options->qwen_parameter_format);
  }
}

static iree_status_t id4_ideogram4_validate_dit_parameter_format(
    const id4_ideogram4_session_create_options_t* options) {
  switch (options->dit_parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      if (!iree_string_view_is_empty(
              options->parameter_scopes.dit_conditioned_fp8) ||
          !iree_string_view_is_empty(
              options->parameter_scopes.dit_unconditioned_fp8)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 BF16 DiT parameter format must not provide FP8 "
            "parameter scopes");
      }
      return iree_ok_status();
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      if (iree_string_view_is_empty(
              options->parameter_scopes.dit_conditioned_fp8)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 FP8 e4m3 DiT parameter format requires a "
            "conditioned FP8 parameter scope");
      }
      if (iree_string_view_is_empty(
              options->parameter_scopes.dit_unconditioned_fp8)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 FP8 e4m3 DiT parameter format requires an "
            "unconditioned FP8 parameter scope");
      }
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 DiT parameter format %" PRIu32
                              " is invalid",
                              (uint32_t)options->dit_parameter_format);
  }
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
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_qwen_parameter_format(options));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_dit_parameter_format(options));
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
  stage_options.parameter_format = options->qwen_parameter_format;
  stage_options.model = session->qwen_model;
  return id4_qwen3_vl_stage_create(&stage_options, session->host_allocator,
                                   &session->qwen_stage);
}

static iree_status_t id4_ideogram4_create_dit_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session, iree_string_view_t parameter_scope,
    iree_string_view_t fp8_parameter_scope, id4_pipeline_stage_t** out_stage) {
  id4_ideogram4_dit_parameter_source_rule_list_t source_rules;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      options->dit_parameter_format, session->dit_model, fp8_parameter_scope,
      options->services.host_allocator, &source_rules));

  id4_ideogram4_dit_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  stage_options.parameter_scope = parameter_scope;
  stage_options.parameter_source_rule_count = source_rules.count;
  stage_options.parameter_source_rules = source_rules.values;
  stage_options.model = session->dit_model;
  iree_status_t status = id4_ideogram4_dit_stage_create(
      &stage_options, session->host_allocator, out_stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &source_rules, options->services.host_allocator);
  return status;
}

static iree_status_t id4_ideogram4_create_sampler_noise_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_sampler_noise_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  return id4_sampler_noise_stage_create(&stage_options, session->host_allocator,
                                        &session->sampler_noise_stage);
}

static iree_status_t id4_ideogram4_create_sampler_denoise_stage(
    const id4_ideogram4_session_create_options_t* options,
    id4_ideogram4_session_t* session) {
  id4_sampler_denoise_stage_create_options_t stage_options;
  memset(&stage_options, 0, sizeof(stage_options));
  stage_options.structure_size = sizeof(stage_options);
  stage_options.services = options->services;
  stage_options.kernel_cache = options->kernel_cache;
  return id4_sampler_denoise_stage_create(
      &stage_options, session->host_allocator, &session->sampler_denoise_stage);
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
  session->qwen_parameter_format = options->qwen_parameter_format;
  session->dit_model = *id4_ideogram4_dit_program_ideogram4_model_config();
  session->decode_model =
      *id4_ideogram4_decode_program_ideogram4_model_config();

  iree_status_t status = id4_ideogram4_create_qwen_stage(options, session);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_dit_stage(
        options, session, options->parameter_scopes.dit_conditioned,
        options->parameter_scopes.dit_conditioned_fp8,
        &session->dit_conditioned_stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_dit_stage(
        options, session, options->parameter_scopes.dit_unconditioned,
        options->parameter_scopes.dit_unconditioned_fp8,
        &session->dit_unconditioned_stage);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_sampler_noise_stage(options, session);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_create_sampler_denoise_stage(options, session);
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

void id4_ideogram4_session_retain(id4_ideogram4_session_t* session) {
  if (!session) {
    return;
  }
  iree_atomic_ref_count_inc(&session->ref_count);
}

void id4_ideogram4_session_release(id4_ideogram4_session_t* session) {
  if (session && iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_allocator_t host_allocator = session->host_allocator;
    for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_STAGE_COUNT;
         ++i) {
      id4_ideogram4_resident_parameter_cache_entry_deinitialize(
          &session->resident_stage_parameters[i], host_allocator);
    }
    id4_pipeline_stage_release(session->decode_stage);
    id4_pipeline_stage_release(session->sampler_denoise_stage);
    id4_pipeline_stage_release(session->sampler_noise_stage);
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
      id4_pipeline_stage_load(session->sampler_noise_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->sampler_denoise_stage, &stage_options));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_stage_load(session->decode_stage, &stage_options));
  session->is_loaded = true;
  return iree_ok_status();
}
