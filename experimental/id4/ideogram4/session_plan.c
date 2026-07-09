// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdint.h>
#include <string.h>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/ideogram4/session_generation.h"
#include "experimental/id4/ideogram4/session_state.h"
#include "experimental/id4/ideogram4/session_support.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/qwen3_vl.h"
#include "experimental/id4/stages/sampler.h"

static iree_status_t
id4_ideogram4_validate_generation_stage_diagnostic_tap_lists(
    iree_host_size_t list_count,
    const id4_ideogram4_generation_stage_diagnostic_tap_list_t* lists) {
  if (list_count == 0) {
    if (lists) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "generation diagnostic tap list array requires at least one entry");
    }
    return iree_ok_status();
  }
  if (!lists) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generation diagnostic tap list array is required");
  }
  for (iree_host_size_t i = 0; i < list_count; ++i) {
    if (iree_string_view_is_empty(lists[i].stage_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "generation diagnostic tap list %" PRIhsz
                              " has an empty stage key",
                              i);
    }
    if (!id4_ideogram4_generation_stage_descriptor_for_key(
            lists[i].stage_key)) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "generation diagnostic tap list %" PRIhsz
                              " references unknown stage `%.*s`",
                              i, (int)lists[i].stage_key.size,
                              lists[i].stage_key.data);
    }
    if (lists[i].tap_names.count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "generation diagnostic tap list for stage `%.*s` is empty",
          (int)lists[i].stage_key.size, lists[i].stage_key.data);
    }
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_validate_diagnostic_tap_names(lists[i].tap_names));
    for (iree_host_size_t j = i + 1; j < list_count; ++j) {
      if (iree_string_view_equal(lists[i].stage_key, lists[j].stage_key)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "generation diagnostic tap list contains duplicate stage `%.*s`",
            (int)lists[i].stage_key.size, lists[i].stage_key.data);
      }
    }
  }
  return iree_ok_status();
}

static id4_pipeline_program_shape_t
id4_ideogram4_generation_decoded_image_shape(
    id4_ideogram4_decode_model_config_t model,
    id4_pipeline_program_shape_t diffusion_latent_shape) {
  return id4_pipeline_program_make_shape_rank4(
      diffusion_latent_shape.dims[0] * model.vae.scale_x,
      diffusion_latent_shape.dims[1] * model.vae.scale_y,
      model.vae.decoded_channel_count, diffusion_latent_shape.dims[3]);
}

static iree_status_t id4_ideogram4_validate_generation_policy(
    id4_ideogram4_generation_plan_policy_t policy) {
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      policy.structure_size, sizeof(policy),
      IREE_SV("Ideogram 4 generation plan policy")));
  if (policy.next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation plan policy extension structures are not "
        "supported");
  }
  switch (policy.dit_activation_format) {
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL:
    case ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT activation format %" PRIu32 " is invalid",
          (uint32_t)policy.dit_activation_format);
  }
  switch (policy.dit_weight_execution_format) {
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS_FEED_FORWARD_BF16_RESIDENT:
    case ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_STREAMING_COMPACT_RHS:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT weight execution format %" PRIu32
          " is invalid",
          (uint32_t)policy.dit_weight_execution_format);
  }
  switch (policy.qwen_weight_execution_strategy) {
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_ROW_MAJOR:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_HYBRID_COMPACT_RHS:
    case ID4_QWEN3_VL_WEIGHT_EXECUTION_STRATEGY_STREAMING_COMPACT_RHS:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation Qwen weight execution strategy %" PRIu32
          " is invalid",
          (uint32_t)policy.qwen_weight_execution_strategy);
  }
  switch (policy.qwen_attention_implementation) {
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_AUTO:
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_MATERIALIZED:
    case ID4_QWEN3_VL_ATTENTION_IMPLEMENTATION_WMMA:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation Qwen attention implementation %" PRIu32
          " is invalid",
          (uint32_t)policy.qwen_attention_implementation);
  }
  switch (policy.dit_attention_implementation) {
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_STREAMING:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_MATERIALIZED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_BLOCKED_WMMA:
    case ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT attention implementation %" PRIu32
          " is invalid",
          (uint32_t)policy.dit_attention_implementation);
  }
  switch (policy.dit_feed_forward_implementation) {
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_FUSED_PRODUCT:
    case ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY:
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation DiT feed-forward implementation %" PRIu32
          " is invalid",
          (uint32_t)policy.dit_feed_forward_implementation);
  }
  switch (policy.vae_tiling.mode) {
    case ID4_VAE_TILING_MODE_DISABLED:
    case ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE:
    case ID4_VAE_TILING_MODE_RELATIVE_TILE_SIZE:
    case ID4_VAE_TILING_MODE_MEMORY_BUDGET:
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation VAE tiling mode %" PRIu32
                              " is invalid",
                              (uint32_t)policy.vae_tiling.mode);
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
      id4_ideogram4_validate_generation_request(session, options->request));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_policy(options->policy));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_resident_stage_mask(
      options->region_per_dispatch_stage_mask));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_stage_diagnostic_tap_lists(
          options->stage_diagnostic_tap_list_count,
          options->stage_diagnostic_tap_lists));
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

static iree_string_view_list_t
id4_ideogram4_generation_stage_diagnostic_tap_names(
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  if (!descriptor) return iree_string_view_list_empty();
  iree_string_view_t stage_key = iree_make_cstring_view(descriptor->key);
  for (iree_host_size_t i = 0; i < options->stage_diagnostic_tap_list_count;
       ++i) {
    const id4_ideogram4_generation_stage_diagnostic_tap_list_t* list =
        &options->stage_diagnostic_tap_lists[i];
    if (iree_string_view_equal(list->stage_key, stage_key)) {
      return list->tap_names;
    }
  }
  return iree_string_view_list_empty();
}

static iree_status_t id4_ideogram4_plan_stage(
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    id4_pipeline_stage_t* stage, const void* stage_options,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  iree_string_view_list_t diagnostic_tap_names =
      id4_ideogram4_generation_stage_diagnostic_tap_names(options,
                                                          stage_ordinal);
  id4_pipeline_stage_plan_options_t plan_options;
  memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = stage_options;
  plan_options.flags =
      diagnostic_tap_names.count == 0
          ? 0
          : ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  if (descriptor && iree_any_bit_set(options->region_per_dispatch_stage_mask,
                                     descriptor->resident_stage_bit)) {
    plan_options.flags |= ID4_PIPELINE_STAGE_PLAN_FLAG_REGION_PER_DISPATCH;
  }
  plan_options.device_index = options->device_index;
  plan_options.queue_affinity = options->queue_affinity;
  plan_options.diagnostic_tap_names = diagnostic_tap_names;
  plan_options.diagnostics_sink = options->diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_qwen(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    const id4_ideogram4_qwen_inputs_t* inputs, id4_pipeline_plan_t** out_plan) {
  id4_qwen3_vl_stage_plan_options_t qwen_options;
  memset(&qwen_options, 0, sizeof(qwen_options));
  qwen_options.structure_size = sizeof(qwen_options);
  qwen_options.request.token_count = inputs->token_count;
  qwen_options.request.token_ids = inputs->token_ids;
  qwen_options.weight_execution_strategy =
      options->policy.qwen_weight_execution_strategy;
  qwen_options.attention_implementation =
      options->policy.qwen_attention_implementation;
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_QWEN,
                                  session->qwen_stage, &qwen_options, options,
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
  dit_options.request.latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  dit_options.request.conditioning_mode = conditioning_mode;
  dit_options.request.text_token_count = text_token_count;
  dit_options.activation_format = options->policy.dit_activation_format;
  dit_options.weight_execution_format =
      options->policy.dit_weight_execution_format;
  dit_options.attention_implementation =
      options->policy.dit_attention_implementation;
  dit_options.feed_forward_implementation =
      options->policy.dit_feed_forward_implementation;
  return id4_ideogram4_plan_stage(
      conditioning_mode == ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED
          ? ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED
          : ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      stage, &dit_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_sampler_noise(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_noise_stage_plan_options_t sampler_options;
  memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_NOISE,
                                  session->sampler_noise_stage,
                                  &sampler_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_sampler_denoise(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_denoise_stage_plan_options_t sampler_options;
  memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
                                  session->sampler_denoise_stage,
                                  &sampler_options, options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_decode(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_decode_stage_plan_options_t decode_options;
  memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  decode_options.request.vae_tiling = options->policy.vae_tiling;
  return id4_ideogram4_plan_stage(ID4_IDEOGRAM4_GENERATION_STAGE_DECODE,
                                  session->decode_stage, &decode_options,
                                  options, out_plan);
}

static iree_status_t id4_ideogram4_plan_generation_stages(
    id4_ideogram4_session_t* session,
    const id4_ideogram4_generation_plan_options_t* options,
    id4_ideogram4_generation_plan_t* plan) {
  id4_ideogram4_qwen_lowering_options_t qwen_lowering_options;
  memset(&qwen_lowering_options, 0, sizeof(qwen_lowering_options));
  qwen_lowering_options.structure_size = sizeof(qwen_lowering_options);
  qwen_lowering_options.tokenizer = options->tokenizer;
  qwen_lowering_options.request = options->request;
  qwen_lowering_options.tokenizer_flags = options->tokenizer_flags;
  qwen_lowering_options.max_token_count = session->qwen_model.max_token_count;
  qwen_lowering_options.vocab_size = session->qwen_model.vocab_size;
  uint32_t qwen_token_count = 0;
  iree_status_t status = id4_ideogram4_request_count_qwen_tokens(
      &qwen_lowering_options, session->host_allocator, &qwen_token_count);
  if (iree_status_is_ok(status)) {
    status = id4_qwen3_vl_program_calculate_token_capacity(
        session->qwen_parameter_format, qwen_token_count,
        &qwen_lowering_options.token_capacity);
  }
  id4_ideogram4_qwen_inputs_t qwen_inputs;
  memset(&qwen_inputs, 0, sizeof(qwen_inputs));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_lower_qwen_inputs(
        &qwen_lowering_options, session->host_allocator, &qwen_inputs);
  }

  plan->summary.diffusion_latent_shape =
      id4_ideogram4_generation_request_diffusion_latent_shape(options->request);
  if (iree_status_is_ok(status)) {
    plan->summary.qwen_token_count = qwen_inputs.token_count;
    plan->summary.qwen_token_capacity = qwen_inputs.token_capacity;
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_dit_program_image_token_count(
        session->dit_model, plan->summary.diffusion_latent_shape,
        &plan->summary.image_token_count);
  }
  if (iree_status_is_ok(status) &&
      plan->summary.image_token_count > UINT32_MAX - qwen_inputs.token_count) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Ideogram4 generation conditioned token count "
                              "overflow");
  }
  if (iree_status_is_ok(status)) {
    plan->summary.conditioned_dit_token_count =
        qwen_inputs.token_count + plan->summary.image_token_count;
    status = id4_ideogram4_dit_program_calculate_bf16_token_capacity(
        plan->summary.conditioned_dit_token_count,
        &plan->summary.conditioned_dit_token_capacity);
  }
  if (iree_status_is_ok(status)) {
    plan->summary.unconditioned_dit_token_count =
        plan->summary.image_token_count;
    status = id4_ideogram4_dit_program_calculate_bf16_token_capacity(
        plan->summary.unconditioned_dit_token_count,
        &plan->summary.unconditioned_dit_token_capacity);
  }
  if (iree_status_is_ok(status)) {
    plan->summary.denoise_step_count =
        options->request->generation.denoise_step_count;
    plan->summary.decoded_image_shape =
        id4_ideogram4_generation_decoded_image_shape(
            session->decode_model, plan->summary.diffusion_latent_shape);
    plan->summary.dit_activation_format = options->policy.dit_activation_format;
    plan->summary.dit_weight_execution_format =
        options->policy.dit_weight_execution_format;
    plan->summary.qwen_weight_execution_strategy =
        options->policy.qwen_weight_execution_strategy;
    plan->summary.qwen_attention_implementation =
        options->policy.qwen_attention_implementation;
    plan->summary.dit_attention_implementation =
        options->policy.dit_attention_implementation;
    plan->summary.dit_feed_forward_implementation =
        options->policy.dit_feed_forward_implementation;
    plan->summary.vae_tiling = options->policy.vae_tiling;
  }

  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_qwen(session, options, &qwen_inputs,
                                                &plan->qwen_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_dit(
        session->dit_conditioned_stage, options,
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED,
        qwen_inputs.token_count, &plan->dit_conditioned_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_dit(
        session->dit_unconditioned_stage, options,
        ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED, 0,
        &plan->dit_unconditioned_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_sampler_noise(
        session, options, &plan->sampler_noise_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_sampler_denoise(
        session, options, &plan->sampler_denoise_plan);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_plan_generation_decode(session, options,
                                                  &plan->decode_plan);
  }
  id4_ideogram4_qwen_inputs_deinitialize(&qwen_inputs, session->host_allocator);
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
  id4_pipeline_plan_release(plan->sampler_denoise_plan);
  id4_pipeline_plan_release(plan->sampler_noise_plan);
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

iree_host_size_t id4_ideogram4_generation_plan_stage_count(
    const id4_ideogram4_generation_plan_t* plan) {
  return plan ? IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors) : 0;
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

iree_status_t id4_ideogram4_generation_plan_stage_at(
    const id4_ideogram4_generation_plan_t* plan, iree_host_size_t index,
    iree_string_view_t* out_stage_key,
    const id4_pipeline_plan_t** out_stage_plan) {
  if (!plan || !out_stage_key || !out_stage_plan) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan, stage key output, and stage plan output "
        "are required");
  }
  *out_stage_key = iree_string_view_empty();
  *out_stage_plan = NULL;
  const iree_host_size_t stage_count =
      id4_ideogram4_generation_plan_stage_count(plan);
  if (index >= stage_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation plan stage index %" PRIhsz
                            " exceeds stage count %" PRIhsz,
                            index, stage_count);
  }

  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      &id4_ideogram4_generation_stage_descriptors[index];
  const id4_pipeline_plan_t* stage_plan =
      id4_ideogram4_generation_stage_plan(plan, descriptor->ordinal);
  if (!stage_plan) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 generation plan stage %s is missing",
                            descriptor->key);
  }
  *out_stage_key = iree_make_cstring_view(descriptor->key);
  *out_stage_plan = stage_plan;
  return iree_ok_status();
}

typedef struct id4_ideogram4_generation_stage_resource_statistics_t {
  // Parameter slab bytes retained by one prepared stage bundle.
  iree_device_size_t parameter_byte_length;
  // Peak parameter slab bytes used by one deferred compact issue window.
  iree_device_size_t parameter_issue_peak_byte_length;
  // Parameter source and target bytes moved while materializing one bundle.
  iree_device_size_t parameter_materialization_byte_length;
  // Constant slab bytes retained by one prepared stage bundle.
  iree_device_size_t constant_byte_length;
  // Local transient high-water bytes used while issuing one stage.
  iree_device_size_t local_high_water_mark;
} id4_ideogram4_generation_stage_resource_statistics_t;

static iree_status_t id4_ideogram4_generation_add_device_size(
    iree_device_size_t* inout_value, iree_device_size_t addend,
    iree_string_view_t field_name) {
  if (UINT64_MAX - *inout_value < addend) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 generation resource statistic %.*s overflows",
        (int)field_name.size, field_name.data);
  }
  *inout_value += addend;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_parameter_region_window_size(
    iree_host_size_t prefetch_region_distance,
    iree_host_size_t* out_region_window_size) {
  if (!iree_host_size_checked_add(prefetch_region_distance, 1,
                                  out_region_window_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Ideogram 4 generation parameter prefetch distance overflows");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_stage_resource_statistics(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_host_size_t parameter_region_window_size,
    id4_ideogram4_generation_stage_resource_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  const id4_pipeline_plan_t* stage_plan =
      id4_ideogram4_generation_stage_plan(plan, stage_ordinal);
  if (!stage_plan) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        id4_ideogram4_generation_stage_descriptor(stage_ordinal);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation resource statistics reference missing "
        "stage %s",
        descriptor ? descriptor->key : "<unknown>");
  }
  id4_pipeline_plan_statistics_t stage_statistics =
      id4_pipeline_plan_statistics(stage_plan);
  out_statistics->parameter_byte_length =
      stage_statistics.parameter_slab_byte_length;
  id4_pipeline_parameter_window_statistics_t window_statistics;
  IREE_RETURN_IF_ERROR(id4_pipeline_plan_parameter_window_statistics(
      stage_plan, parameter_region_window_size, &window_statistics));
  out_statistics->parameter_issue_peak_byte_length =
      window_statistics.peak_window_target_byte_length;
  for (iree_host_size_t i = 0;
       i < ID4_PIPELINE_PARAMETER_LOAD_STEP_KIND_CAPACITY; ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &out_statistics->parameter_materialization_byte_length,
        stage_statistics.parameter_load_kind_statistics[i].source_byte_length,
        IREE_SV("parameter.materialization.source")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &out_statistics->parameter_materialization_byte_length,
        stage_statistics.parameter_load_kind_statistics[i].target_byte_length,
        IREE_SV("parameter.materialization.target")));
  }
  out_statistics->constant_byte_length =
      stage_statistics.constant_slab_byte_length;
  out_statistics->local_high_water_mark =
      stage_statistics.memory_slab_high_water_mark;
  return iree_ok_status();
}

static bool id4_ideogram4_generation_resident_mask_contains_stage(
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  return descriptor &&
         iree_any_bit_set(resident_stage_mask, descriptor->resident_stage_bit);
}

static void id4_ideogram4_generation_resource_policy_initialize(
    id4_ideogram4_generation_residency_mode_t residency_mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    const id4_ideogram4_generation_resident_stage_mask_t* phase_stage_masks,
    id4_ideogram4_generation_residency_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  out_policy->mode = residency_mode;
  switch (residency_mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
      out_policy->request_stage_mask =
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_ALL;
      return;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
      out_policy->request_stage_mask = resident_stage_mask;
      return;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES:
      for (iree_host_size_t i = 0;
           i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors);
           ++i) {
        out_policy->phase_stage_masks[i] =
            id4_ideogram4_generation_phase_stage_mask(
                &id4_ideogram4_generation_phase_descriptors[i]);
      }
      return;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_AWARE_STAGE_BUNDLES:
      out_policy->request_stage_mask = resident_stage_mask;
      memcpy(out_policy->phase_stage_masks, phase_stage_masks,
             sizeof(out_policy->phase_stage_masks));
      return;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
    default:
      return;
  }
}

static iree_status_t id4_ideogram4_generation_resource_add_stage(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    bool use_compact_parameter_windows,
    iree_host_size_t parameter_region_window_size,
    id4_ideogram4_generation_stage_resource_statistics_t* io_statistics) {
  id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
      plan, stage_ordinal, parameter_region_window_size, &stage_statistics));
  const iree_device_size_t parameter_byte_length =
      use_compact_parameter_windows
          ? stage_statistics.parameter_issue_peak_byte_length
          : stage_statistics.parameter_byte_length;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &io_statistics->parameter_byte_length, parameter_byte_length,
      IREE_SV("parameter")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &io_statistics->constant_byte_length,
      stage_statistics.constant_byte_length, IREE_SV("constant")));
  return id4_ideogram4_generation_add_device_size(
      &io_statistics->local_high_water_mark,
      stage_statistics.local_high_water_mark, IREE_SV("local"));
}

static iree_status_t id4_ideogram4_generation_resource_add_stage_parameters(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_host_size_t parameter_region_window_size,
    id4_ideogram4_generation_stage_resource_statistics_t* io_statistics) {
  id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
      plan, stage_ordinal, parameter_region_window_size, &stage_statistics));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &io_statistics->parameter_byte_length,
      stage_statistics.parameter_byte_length, IREE_SV("phase.parameter")));
  return id4_ideogram4_generation_add_device_size(
      &io_statistics->constant_byte_length,
      stage_statistics.constant_byte_length, IREE_SV("phase.constant"));
}

static iree_status_t id4_ideogram4_generation_resource_add_stage_local(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_host_size_t parameter_region_window_size,
    id4_ideogram4_generation_stage_resource_statistics_t* io_statistics) {
  id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
      plan, stage_ordinal, parameter_region_window_size, &stage_statistics));
  return id4_ideogram4_generation_add_device_size(
      &io_statistics->local_high_water_mark,
      stage_statistics.local_high_water_mark, IREE_SV("local"));
}

static iree_status_t id4_ideogram4_generation_resource_stage_group(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t request_stage_mask,
    id4_ideogram4_generation_resident_stage_mask_t phase_stage_mask,
    iree_host_size_t parameter_region_window_size, iree_host_size_t stage_count,
    const id4_ideogram4_generation_stage_ordinal_t* stage_ordinals,
    id4_ideogram4_generation_stage_resource_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  for (iree_host_size_t i = 0; i < stage_count; ++i) {
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        stage_ordinals[i];
    if (id4_ideogram4_generation_resident_mask_contains_stage(
            request_stage_mask, stage_ordinal)) {
      id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
          plan, stage_ordinal, parameter_region_window_size,
          &stage_statistics));
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
          &out_statistics->local_high_water_mark,
          stage_statistics.local_high_water_mark, IREE_SV("local")));
      continue;
    }
    const bool use_compact_parameter_windows =
        !id4_ideogram4_generation_resident_mask_contains_stage(phase_stage_mask,
                                                               stage_ordinal);
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_add_stage(
        plan, stage_ordinal, use_compact_parameter_windows,
        parameter_region_window_size, out_statistics));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_resource_resident_stages(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    iree_host_size_t parameter_region_window_size,
    id4_ideogram4_generation_stage_resource_statistics_t* out_statistics) {
  memset(out_statistics, 0, sizeof(*out_statistics));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (!iree_any_bit_set(resident_stage_mask,
                          descriptor->resident_stage_bit)) {
      continue;
    }
    id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
        plan, descriptor->ordinal, parameter_region_window_size,
        &stage_statistics));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &out_statistics->parameter_byte_length,
        stage_statistics.parameter_byte_length, IREE_SV("resident.parameter")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &out_statistics->constant_byte_length,
        stage_statistics.constant_byte_length, IREE_SV("resident.constant")));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_find_boundary_layout(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    iree_string_view_t name, const id4_pipeline_tensor_layout_t** out_layout) {
  *out_layout = NULL;
  const id4_pipeline_plan_t* stage_plan =
      id4_ideogram4_generation_stage_plan(plan, stage_ordinal);
  if (!stage_plan) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        id4_ideogram4_generation_stage_descriptor(stage_ordinal);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation resource statistics reference missing "
        "stage %s",
        descriptor ? descriptor->key : "<unknown>");
  }
  const iree_host_size_t boundary_count =
      id4_pipeline_plan_boundary_tensor_count(stage_plan);
  for (iree_host_size_t i = 0; i < boundary_count; ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(stage_plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_layout = &boundary->layout;
      return iree_ok_status();
    }
  }
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(stage_ordinal);
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "Ideogram 4 generation resource statistics stage %s has no boundary "
      "tensor `%.*s`",
      descriptor ? descriptor->key : "<unknown>", (int)name.size, name.data);
}

static iree_status_t
id4_ideogram4_generation_resource_retained_boundary_buffers(
    const id4_ideogram4_generation_plan_t* plan,
    iree_device_size_t* out_boundary_byte_length,
    iree_device_size_t* out_diagnostic_tap_byte_length) {
  *out_boundary_byte_length = 0;
  *out_diagnostic_tap_byte_length = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_pipeline_plan_t* stage_plan = id4_ideogram4_generation_stage_plan(
        plan, id4_ideogram4_generation_stage_descriptors[i].ordinal);
    if (!stage_plan) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation resource statistics reference missing "
          "stage %s",
          id4_ideogram4_generation_stage_descriptors[i].key);
    }
    id4_pipeline_plan_statistics_t stage_statistics =
        id4_pipeline_plan_statistics(stage_plan);
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        out_boundary_byte_length, stage_statistics.boundary_tensor_byte_length,
        IREE_SV("boundary")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        out_diagnostic_tap_byte_length,
        stage_statistics.diagnostic_tap_byte_length, IREE_SV("tap")));
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_boundary_aliases); ++i) {
    const id4_ideogram4_generation_boundary_alias_t* alias =
        &id4_ideogram4_generation_boundary_aliases[i];
    const id4_pipeline_tensor_layout_t* source_layout = NULL;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_find_boundary_layout(
        plan, alias->source_stage, alias->source_name, &source_layout));
    const id4_pipeline_tensor_layout_t* target_layout = NULL;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_find_boundary_layout(
        plan, alias->target_stage, alias->target_name, &target_layout));
    if (source_layout->byte_length != target_layout->byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation boundary alias %.*s to %.*s has byte "
          "length mismatch",
          (int)alias->source_name.size, alias->source_name.data,
          (int)alias->target_name.size, alias->target_name.data);
    }
    if (*out_boundary_byte_length < target_layout->byte_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram 4 generation boundary alias %.*s to %.*s underflows "
          "retained boundary accounting",
          (int)alias->source_name.size, alias->source_name.data,
          (int)alias->target_name.size, alias->target_name.data);
    }
    *out_boundary_byte_length -= target_layout->byte_length;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_resource_total_peak(
    const id4_ideogram4_generation_resource_statistics_t* statistics,
    iree_device_size_t parameter_byte_length,
    iree_device_size_t constant_byte_length,
    iree_device_size_t local_byte_length,
    iree_device_size_t* out_total_byte_length) {
  iree_device_size_t total_byte_length =
      statistics->boundary_buffer_byte_length;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, statistics->diagnostic_tap_buffer_byte_length,
      IREE_SV("peak.tap")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, parameter_byte_length, IREE_SV("peak.parameter")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, constant_byte_length, IREE_SV("peak.constant")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &total_byte_length, local_byte_length, IREE_SV("peak.local")));
  *out_total_byte_length = total_byte_length;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_resource_stage_serial_phase(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_residency_policy_t* residency_policy,
    const id4_ideogram4_generation_stage_resource_statistics_t*
        request_statistics,
    const id4_ideogram4_generation_phase_descriptor_t* phase,
    id4_ideogram4_generation_resident_stage_mask_t phase_stage_mask,
    iree_host_size_t parameter_region_window_size,
    id4_ideogram4_generation_resource_statistics_t* io_statistics) {
  id4_ideogram4_generation_stage_resource_statistics_t phase_statistics;
  memset(&phase_statistics, 0, sizeof(phase_statistics));
  for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        phase->stage_ordinals[i];
    if (id4_ideogram4_generation_resident_mask_contains_stage(
            residency_policy->request_stage_mask, stage_ordinal) ||
        !id4_ideogram4_generation_resident_mask_contains_stage(phase_stage_mask,
                                                               stage_ordinal)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_add_stage_parameters(
        plan, stage_ordinal, parameter_region_window_size, &phase_statistics));
  }

  for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
    const id4_ideogram4_generation_stage_ordinal_t stage_ordinal =
        phase->stage_ordinals[i];
    id4_ideogram4_generation_stage_resource_statistics_t stage_statistics;
    memset(&stage_statistics, 0, sizeof(stage_statistics));
    if (id4_ideogram4_generation_resident_mask_contains_stage(
            residency_policy->request_stage_mask, stage_ordinal) ||
        id4_ideogram4_generation_resident_mask_contains_stage(phase_stage_mask,
                                                              stage_ordinal)) {
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_add_stage_local(
          plan, stage_ordinal, parameter_region_window_size,
          &stage_statistics));
    } else {
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_add_stage(
          plan, stage_ordinal, /*use_compact_parameter_windows=*/true,
          parameter_region_window_size, &stage_statistics));
    }

    iree_device_size_t parameter_byte_length =
        request_statistics->parameter_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &parameter_byte_length, phase_statistics.parameter_byte_length,
        IREE_SV("stage.phase.parameter")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &parameter_byte_length, stage_statistics.parameter_byte_length,
        IREE_SV("stage.parameter")));

    iree_device_size_t constant_byte_length =
        request_statistics->constant_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &constant_byte_length, phase_statistics.constant_byte_length,
        IREE_SV("stage.phase.constant")));
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &constant_byte_length, stage_statistics.constant_byte_length,
        IREE_SV("stage.constant")));

    if (parameter_byte_length >
        io_statistics->stage_serial_parameter_peak_byte_length) {
      io_statistics->stage_serial_parameter_peak_byte_length =
          parameter_byte_length;
    }
    if (constant_byte_length >
        io_statistics->stage_serial_constant_peak_byte_length) {
      io_statistics->stage_serial_constant_peak_byte_length =
          constant_byte_length;
    }
    if (stage_statistics.local_high_water_mark >
        io_statistics->stage_serial_local_peak_byte_length) {
      io_statistics->stage_serial_local_peak_byte_length =
          stage_statistics.local_high_water_mark;
    }
    iree_device_size_t total_byte_length = 0;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_total_peak(
        io_statistics, parameter_byte_length, constant_byte_length,
        stage_statistics.local_high_water_mark, &total_byte_length));
    if (total_byte_length >
        io_statistics->stage_serial_total_peak_byte_length) {
      io_statistics->stage_serial_total_peak_byte_length = total_byte_length;
    }
  }
  return iree_ok_status();
}

iree_status_t id4_ideogram4_generation_plan_resource_statistics(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_resource_statistics_options_t* options,
    id4_ideogram4_generation_resource_statistics_t* out_statistics) {
  if (!plan || !options || !out_statistics) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan, resource statistics options, and output "
        "are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation resource statistics")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation resource statistics extension structures are "
        "not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_prepare_residency_options(
          options->residency_mode, options->resident_stage_mask,
          options->phase_stage_masks));

  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_generation_resource_retained_boundary_buffers(
          plan, &out_statistics->boundary_buffer_byte_length,
          &out_statistics->diagnostic_tap_buffer_byte_length));

  id4_ideogram4_generation_residency_policy_t residency_policy;
  id4_ideogram4_generation_resource_policy_initialize(
      options->residency_mode, options->resident_stage_mask,
      options->phase_stage_masks, &residency_policy);
  iree_host_size_t parameter_region_window_size = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_parameter_region_window_size(
      options->parameter_load_prefetch_region_distance,
      &parameter_region_window_size));
  id4_ideogram4_generation_stage_resource_statistics_t resident_statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_resident_stages(
      plan, residency_policy.request_stage_mask, parameter_region_window_size,
      &resident_statistics));
  out_statistics->resident_stage_parameter_byte_length =
      resident_statistics.parameter_byte_length;
  out_statistics->resident_stage_constant_byte_length =
      resident_statistics.constant_byte_length;
  out_statistics->resident_stage_bundle_byte_length =
      resident_statistics.parameter_byte_length;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &out_statistics->resident_stage_bundle_byte_length,
      resident_statistics.constant_byte_length, IREE_SV("resident.bundle")));

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    const id4_ideogram4_generation_phase_descriptor_t* phase =
        &id4_ideogram4_generation_phase_descriptors[i];
    id4_ideogram4_generation_stage_resource_statistics_t phase_statistics;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_stage_group(
        plan, residency_policy.request_stage_mask,
        residency_policy.phase_stage_masks[i], parameter_region_window_size,
        phase->stage_count, phase->stage_ordinals, &phase_statistics));
    iree_device_size_t parameter_byte_length =
        resident_statistics.parameter_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &parameter_byte_length, phase_statistics.parameter_byte_length,
        IREE_SV("phase.parameter")));
    iree_device_size_t constant_byte_length =
        resident_statistics.constant_byte_length;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
        &constant_byte_length, phase_statistics.constant_byte_length,
        IREE_SV("phase.constant")));
    if (parameter_byte_length >
        out_statistics->phase_concurrent_parameter_peak_byte_length) {
      out_statistics->phase_concurrent_parameter_peak_byte_length =
          parameter_byte_length;
    }
    if (constant_byte_length >
        out_statistics->phase_concurrent_constant_peak_byte_length) {
      out_statistics->phase_concurrent_constant_peak_byte_length =
          constant_byte_length;
    }
    if (phase_statistics.local_high_water_mark >
        out_statistics->phase_concurrent_local_peak_byte_length) {
      out_statistics->phase_concurrent_local_peak_byte_length =
          phase_statistics.local_high_water_mark;
    }
    iree_device_size_t total_byte_length = 0;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_total_peak(
        out_statistics, parameter_byte_length, constant_byte_length,
        phase_statistics.local_high_water_mark, &total_byte_length));
    if (total_byte_length >
        out_statistics->phase_concurrent_total_peak_byte_length) {
      out_statistics->phase_concurrent_total_peak_byte_length =
          total_byte_length;
    }
  }

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_resource_stage_serial_phase(
        plan, &residency_policy, &resident_statistics,
        &id4_ideogram4_generation_phase_descriptors[i],
        residency_policy.phase_stage_masks[i], parameter_region_window_size,
        out_statistics));
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_validate_generation_issue_policy(
    id4_ideogram4_generation_issue_policy_t issue_policy) {
  switch (issue_policy) {
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_PHASE_CONCURRENT:
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 generation issue policy %" PRIu32
                              " is invalid",
                              (uint32_t)issue_policy);
  }
}

static uint32_t id4_ideogram4_generation_stage_issue_count(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  switch (stage_ordinal) {
    case ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED:
    case ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED:
    case ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER:
      return plan->summary.denoise_step_count;
    default:
      return 1;
  }
}

static iree_device_size_t id4_ideogram4_generation_ceil_mib(
    iree_device_size_t byte_length) {
  const iree_device_size_t mib = 1024ull * 1024ull;
  return (byte_length + mib - 1) / mib;
}

static iree_status_t id4_ideogram4_generation_add_residency_score(
    uint64_t* inout_score, uint64_t addend) {
  if (UINT64_MAX - *inout_score < addend) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation residency score overflow");
  }
  *inout_score += addend;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_stage_residency_unit_score(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    uint64_t* out_score) {
  id4_ideogram4_generation_stage_resource_statistics_t statistics;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_resource_statistics(
      plan, stage_ordinal, /*parameter_region_window_size=*/1, &statistics));
  iree_device_size_t avoided_byte_length =
      statistics.parameter_materialization_byte_length;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_add_device_size(
      &avoided_byte_length, statistics.constant_byte_length,
      IREE_SV("resident.score")));
  const uint64_t avoided_mib =
      (uint64_t)id4_ideogram4_generation_ceil_mib(avoided_byte_length);
  *out_score = avoided_mib;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_stage_residency_score(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal,
    uint64_t materialization_count, uint64_t* out_score) {
  uint64_t unit_score = 0;
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_residency_unit_score(
      plan, stage_ordinal, &unit_score));
  if (materialization_count != 0 &&
      unit_score > UINT64_MAX / materialization_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Ideogram 4 generation residency score overflow");
  }
  *out_score = unit_score * materialization_count;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_residency_mask_score(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    uint64_t* out_score) {
  uint64_t score = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (!iree_any_bit_set(resident_stage_mask,
                          descriptor->resident_stage_bit)) {
      continue;
    }
    uint64_t stage_score = 0;
    const uint64_t issue_count =
        id4_ideogram4_generation_stage_issue_count(plan, descriptor->ordinal);
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_residency_score(
        plan, descriptor->ordinal, issue_count, &stage_score));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_generation_add_residency_score(&score, stage_score));
  }
  *out_score = score;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_phase_residency_score(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_resident_stage_mask_t* phase_stage_masks,
    uint64_t* out_score) {
  uint64_t score = 0;
  for (iree_host_size_t phase_index = 0;
       phase_index < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors);
       ++phase_index) {
    const id4_ideogram4_generation_phase_descriptor_t* phase =
        &id4_ideogram4_generation_phase_descriptors[phase_index];
    for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
      const id4_ideogram4_generation_stage_descriptor_t* descriptor =
          id4_ideogram4_generation_stage_descriptor(phase->stage_ordinals[i]);
      if (!iree_any_bit_set(phase_stage_masks[phase_index],
                            descriptor->resident_stage_bit)) {
        continue;
      }
      uint64_t stage_score = 0;
      const uint64_t issue_count =
          id4_ideogram4_generation_stage_issue_count(plan, descriptor->ordinal);
      const uint64_t saved_materialization_count =
          issue_count == 0 ? 0 : issue_count - 1;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_stage_residency_score(
          plan, descriptor->ordinal, saved_materialization_count,
          &stage_score));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_generation_add_residency_score(&score, stage_score));
    }
  }
  *out_score = score;
  return iree_ok_status();
}

static iree_device_size_t id4_ideogram4_generation_selected_peak(
    id4_ideogram4_generation_issue_policy_t issue_policy,
    const id4_ideogram4_generation_resource_statistics_t* statistics) {
  switch (issue_policy) {
    case ID4_IDEOGRAM4_GENERATION_ISSUE_POLICY_STAGE_SERIAL:
      return statistics->stage_serial_total_peak_byte_length;
    default:
      return statistics->phase_concurrent_total_peak_byte_length;
  }
}

static id4_ideogram4_generation_residency_mode_t
id4_ideogram4_generation_residency_mode_for_mask(
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask) {
  return resident_stage_mask == ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE
             ? ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES
             : ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES;
}

static iree_status_t id4_ideogram4_generation_residency_statistics_for_mask(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_ideogram4_generation_resource_statistics_t* out_statistics) {
  id4_ideogram4_generation_resource_statistics_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.residency_mode =
      id4_ideogram4_generation_residency_mode_for_mask(resident_stage_mask);
  options.resident_stage_mask = resident_stage_mask;
  options.parameter_load_prefetch_region_distance =
      parameter_load_prefetch_region_distance;
  return id4_ideogram4_generation_plan_resource_statistics(plan, &options,
                                                           out_statistics);
}

static iree_status_t id4_ideogram4_generation_phase_aware_residency_statistics(
    const id4_ideogram4_generation_plan_t* plan,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    const id4_ideogram4_generation_resident_stage_mask_t* phase_stage_masks,
    iree_host_size_t parameter_load_prefetch_region_distance,
    id4_ideogram4_generation_resource_statistics_t* out_statistics) {
  id4_ideogram4_generation_resource_statistics_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.residency_mode =
      ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_AWARE_STAGE_BUNDLES;
  options.resident_stage_mask = resident_stage_mask;
  memcpy(options.phase_stage_masks, phase_stage_masks,
         sizeof(options.phase_stage_masks));
  options.parameter_load_prefetch_region_distance =
      parameter_load_prefetch_region_distance;
  return id4_ideogram4_generation_plan_resource_statistics(plan, &options,
                                                           out_statistics);
}

static uint32_t id4_ideogram4_generation_residency_mode_preference(
    id4_ideogram4_generation_residency_mode_t mode) {
  switch (mode) {
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_AWARE_STAGE_BUNDLES:
      return 0;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_SELECTED_STAGE_BUNDLES:
      return 1;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_STAGE_BUNDLES:
      return 2;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ISSUE_PHASES:
      return 3;
    case ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_ALL_STAGE_BUNDLES:
    default:
      return 4;
  }
}

static uint32_t id4_ideogram4_generation_phase_stage_masks_sort_key(
    const id4_ideogram4_generation_resident_stage_mask_t* phase_stage_masks) {
  uint32_t key = 0;
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_PHASE_COUNT; ++i) {
    key |= ((uint32_t)phase_stage_masks[i]) << (i * 8);
  }
  return key;
}

static bool id4_ideogram4_generation_residency_candidate_is_better(
    bool has_selection, id4_ideogram4_generation_residency_mode_t mode,
    id4_ideogram4_generation_resident_stage_mask_t resident_stage_mask,
    const id4_ideogram4_generation_resident_stage_mask_t* phase_stage_masks,
    uint64_t score, iree_device_size_t peak_byte_length,
    id4_ideogram4_generation_residency_mode_t best_mode,
    id4_ideogram4_generation_resident_stage_mask_t best_stage_mask,
    const id4_ideogram4_generation_resident_stage_mask_t*
        best_phase_stage_masks,
    uint64_t best_score, iree_device_size_t best_peak_byte_length) {
  if (!has_selection) {
    return true;
  }
  if (score != best_score) {
    return score > best_score;
  }
  if (peak_byte_length != best_peak_byte_length) {
    return peak_byte_length < best_peak_byte_length;
  }
  const uint32_t preference =
      id4_ideogram4_generation_residency_mode_preference(mode);
  const uint32_t best_preference =
      id4_ideogram4_generation_residency_mode_preference(best_mode);
  if (preference != best_preference) {
    return preference < best_preference;
  }
  if (resident_stage_mask != best_stage_mask) {
    return resident_stage_mask < best_stage_mask;
  }
  return id4_ideogram4_generation_phase_stage_masks_sort_key(
             phase_stage_masks) <
         id4_ideogram4_generation_phase_stage_masks_sort_key(
             best_phase_stage_masks);
}

static bool id4_ideogram4_generation_phase_stage_masks_are_empty(
    const id4_ideogram4_generation_resident_stage_mask_t* phase_stage_masks) {
  for (iree_host_size_t i = 0; i < ID4_IDEOGRAM4_GENERATION_PHASE_COUNT; ++i) {
    if (phase_stage_masks[i] != ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
      return false;
    }
  }
  return true;
}

static id4_ideogram4_generation_resident_stage_mask_t
id4_ideogram4_generation_stage_mask_from_subset(
    id4_ideogram4_generation_resident_stage_mask_t candidate_stage_mask,
    uint32_t subset) {
  id4_ideogram4_generation_resident_stage_mask_t stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors); ++i) {
    const id4_ideogram4_generation_stage_descriptor_t* descriptor =
        &id4_ideogram4_generation_stage_descriptors[i];
    if (!iree_all_bits_set(subset, 1u << i)) {
      continue;
    }
    if (!iree_any_bit_set(candidate_stage_mask,
                          descriptor->resident_stage_bit)) {
      continue;
    }
    stage_mask |= descriptor->resident_stage_bit;
  }
  return stage_mask;
}

static void id4_ideogram4_generation_make_phase_stage_masks(
    id4_ideogram4_generation_resident_stage_mask_t phase_local_stage_mask,
    id4_ideogram4_generation_resident_stage_mask_t* out_phase_stage_masks) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    out_phase_stage_masks[i] =
        id4_ideogram4_generation_phase_stage_mask(
            &id4_ideogram4_generation_phase_descriptors[i]) &
        phase_local_stage_mask;
  }
}

iree_status_t id4_ideogram4_generation_plan_select_residency(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_residency_select_options_t* options,
    id4_ideogram4_generation_residency_selection_t* out_selection) {
  if (!plan || !options || !out_selection) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation plan, residency select options, and output "
        "are required");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("Ideogram 4 generation residency selection")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Ideogram 4 generation residency selection extension structures are "
        "not supported");
  }
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_validate_generation_issue_policy(options->issue_policy));
  IREE_RETURN_IF_ERROR(id4_ideogram4_validate_generation_resident_stage_mask(
      options->candidate_stage_mask));
  if (options->candidate_stage_mask ==
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation residency selection requires candidate "
        "stages");
  }
  if (options->memory_budget_byte_length == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation residency selection requires a non-zero "
        "memory budget");
  }

  bool found_selection = false;
  uint64_t best_score = 0;
  iree_device_size_t best_peak_byte_length = 0;
  id4_ideogram4_generation_residency_mode_t best_residency_mode =
      ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_INVALID;
  id4_ideogram4_generation_resident_stage_mask_t best_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  id4_ideogram4_generation_resident_stage_mask_t
      best_phase_stage_masks[ID4_IDEOGRAM4_GENERATION_PHASE_COUNT];
  memset(best_phase_stage_masks, 0, sizeof(best_phase_stage_masks));
  id4_ideogram4_generation_resident_stage_mask_t
      empty_phase_stage_masks[ID4_IDEOGRAM4_GENERATION_PHASE_COUNT];
  memset(empty_phase_stage_masks, 0, sizeof(empty_phase_stage_masks));
  id4_ideogram4_generation_resource_statistics_t best_statistics;
  memset(&best_statistics, 0, sizeof(best_statistics));

  const uint32_t subset_count =
      1u << IREE_ARRAYSIZE(id4_ideogram4_generation_stage_descriptors);
  for (uint32_t subset = 0; subset < subset_count; ++subset) {
    const id4_ideogram4_generation_resident_stage_mask_t stage_mask =
        id4_ideogram4_generation_stage_mask_from_subset(
            options->candidate_stage_mask, subset);

    id4_ideogram4_generation_resource_statistics_t statistics;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_residency_statistics_for_mask(
        plan, stage_mask, options->parameter_load_prefetch_region_distance,
        &statistics));
    const iree_device_size_t peak_byte_length =
        id4_ideogram4_generation_selected_peak(options->issue_policy,
                                               &statistics);
    if (peak_byte_length > options->memory_budget_byte_length) {
      continue;
    }

    uint64_t score = 0;
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_residency_mask_score(
        plan, stage_mask, &score));
    const id4_ideogram4_generation_residency_mode_t residency_mode =
        id4_ideogram4_generation_residency_mode_for_mask(stage_mask);
    if (id4_ideogram4_generation_residency_candidate_is_better(
            found_selection, residency_mode, stage_mask,
            empty_phase_stage_masks, score, peak_byte_length,
            best_residency_mode, best_stage_mask, best_phase_stage_masks,
            best_score, best_peak_byte_length)) {
      found_selection = true;
      best_score = score;
      best_peak_byte_length = peak_byte_length;
      best_residency_mode = residency_mode;
      best_stage_mask = stage_mask;
      memset(best_phase_stage_masks, 0, sizeof(best_phase_stage_masks));
      best_statistics = statistics;
    }
  }

  for (uint32_t request_subset = 0; request_subset < subset_count;
       ++request_subset) {
    const id4_ideogram4_generation_resident_stage_mask_t request_stage_mask =
        id4_ideogram4_generation_stage_mask_from_subset(
            options->candidate_stage_mask, request_subset);
    for (uint32_t phase_subset = 0; phase_subset < subset_count;
         ++phase_subset) {
      const id4_ideogram4_generation_resident_stage_mask_t
          phase_local_stage_mask =
              id4_ideogram4_generation_stage_mask_from_subset(
                  options->candidate_stage_mask, phase_subset);
      if (phase_local_stage_mask ==
          ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE) {
        continue;
      }
      if (iree_any_bit_set(request_stage_mask, phase_local_stage_mask)) {
        continue;
      }
      id4_ideogram4_generation_resident_stage_mask_t
          phase_stage_masks[ID4_IDEOGRAM4_GENERATION_PHASE_COUNT];
      id4_ideogram4_generation_make_phase_stage_masks(phase_local_stage_mask,
                                                      phase_stage_masks);
      if (id4_ideogram4_generation_phase_stage_masks_are_empty(
              phase_stage_masks)) {
        continue;
      }

      id4_ideogram4_generation_resource_statistics_t statistics;
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_generation_phase_aware_residency_statistics(
              plan, request_stage_mask, phase_stage_masks,
              options->parameter_load_prefetch_region_distance, &statistics));
      const iree_device_size_t peak_byte_length =
          id4_ideogram4_generation_selected_peak(options->issue_policy,
                                                 &statistics);
      if (peak_byte_length > options->memory_budget_byte_length) {
        continue;
      }
      uint64_t score = 0;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_residency_mask_score(
          plan, request_stage_mask, &score));
      uint64_t phase_score = 0;
      IREE_RETURN_IF_ERROR(id4_ideogram4_generation_phase_residency_score(
          plan, phase_stage_masks, &phase_score));
      IREE_RETURN_IF_ERROR(
          id4_ideogram4_generation_add_residency_score(&score, phase_score));
      if (!id4_ideogram4_generation_residency_candidate_is_better(
              found_selection,
              ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_AWARE_STAGE_BUNDLES,
              request_stage_mask, phase_stage_masks, score, peak_byte_length,
              best_residency_mode, best_stage_mask, best_phase_stage_masks,
              best_score, best_peak_byte_length)) {
        continue;
      }
      found_selection = true;
      best_score = score;
      best_peak_byte_length = peak_byte_length;
      best_residency_mode =
          ID4_IDEOGRAM4_GENERATION_RESIDENCY_MODE_PHASE_AWARE_STAGE_BUNDLES;
      best_stage_mask = request_stage_mask;
      memcpy(best_phase_stage_masks, phase_stage_masks,
             sizeof(best_phase_stage_masks));
      best_statistics = statistics;
    }
  }

  if (!found_selection) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "Ideogram 4 generation cannot fit within memory budget %" PRIu64
        " bytes",
        (uint64_t)options->memory_budget_byte_length);
  }

  memset(out_selection, 0, sizeof(*out_selection));
  out_selection->residency_mode = best_residency_mode;
  out_selection->resident_stage_mask = best_stage_mask;
  memcpy(out_selection->phase_stage_masks, best_phase_stage_masks,
         sizeof(out_selection->phase_stage_masks));
  out_selection->selected_peak_byte_length = best_peak_byte_length;
  out_selection->resource_statistics = best_statistics;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_residency_statistics_accumulate(
    const id4_ideogram4_generation_plan_t* plan,
    const id4_ideogram4_generation_phase_descriptor_t* phase,
    id4_ideogram4_generation_residency_statistics_t* io_statistics) {
  for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
    const id4_pipeline_plan_t* stage_plan =
        id4_ideogram4_generation_stage_plan(plan, phase->stage_ordinals[i]);
    if (!stage_plan) {
      const id4_ideogram4_generation_stage_descriptor_t* descriptor =
          id4_ideogram4_generation_stage_descriptor(phase->stage_ordinals[i]);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation phase %.*s references missing "
          "stage %s",
          (int)phase->name.size, phase->name.data,
          descriptor ? descriptor->key : "<unknown>");
    }
    id4_pipeline_plan_statistics_t stage_statistics =
        id4_pipeline_plan_statistics(stage_plan);
    io_statistics->parameter_byte_length +=
        stage_statistics.parameter_slab_byte_length;
    if (stage_statistics.largest_parameter_slab_byte_length >
        io_statistics->largest_stage_parameter_byte_length) {
      io_statistics->largest_stage_parameter_byte_length =
          stage_statistics.largest_parameter_slab_byte_length;
    }
    io_statistics->constant_byte_length +=
        stage_statistics.constant_slab_byte_length;
    io_statistics->local_slab_byte_length +=
        stage_statistics.memory_slab_byte_length;
    io_statistics->local_high_water_mark +=
        stage_statistics.memory_slab_high_water_mark;
    io_statistics->stage_boundary_byte_length +=
        stage_statistics.boundary_tensor_byte_length;
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_generation_plan_append_stage_key_json(
    iree_string_builder_t* builder,
    id4_ideogram4_generation_stage_ordinal_t ordinal) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(ordinal);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Ideogram 4 generation stage ordinal %u",
                            (uint32_t)ordinal);
  }
  return iree_string_builder_append_format(builder, "\"%s\"", descriptor->key);
}

static iree_status_t id4_ideogram4_generation_plan_append_phase_json(
    const id4_ideogram4_generation_plan_t* plan, iree_string_builder_t* builder,
    const id4_ideogram4_generation_phase_descriptor_t* phase) {
  id4_ideogram4_generation_residency_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_residency_statistics_accumulate(
      plan, phase, &statistics));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "{\"name\":\"%.*s\"", (int)phase->name.size, phase->name.data));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"stage_keys\":["));
  for (iree_host_size_t i = 0; i < phase->stage_count; ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_key_json(
        builder, phase->stage_ordinals[i]));
  }
  const bool repeated_per_denoise_step = iree_any_bit_set(
      phase->flags, ID4_IDEOGRAM4_GENERATION_PHASE_REPEATED_PER_DENOISE_STEP);
  return iree_string_builder_append_format(
      builder,
      "],\"repeated_per_denoise_step\":%s"
      ",\"parameter_byte_length\":%" PRIu64
      ",\"largest_stage_parameter_byte_length\":%" PRIu64
      ",\"constant_byte_length\":%" PRIu64
      ",\"local_slab_byte_length\":%" PRIu64
      ",\"local_high_water_mark\":%" PRIu64
      ",\"stage_boundary_byte_length\":%" PRIu64 "}",
      repeated_per_denoise_step ? "true" : "false",
      (uint64_t)statistics.parameter_byte_length,
      (uint64_t)statistics.largest_stage_parameter_byte_length,
      (uint64_t)statistics.constant_byte_length,
      (uint64_t)statistics.local_slab_byte_length,
      (uint64_t)statistics.local_high_water_mark,
      (uint64_t)statistics.stage_boundary_byte_length);
}

static iree_status_t id4_ideogram4_generation_plan_append_residency_json(
    const id4_ideogram4_generation_plan_t* plan,
    iree_string_builder_t* builder) {
  id4_ideogram4_generation_residency_statistics_t total_statistics;
  memset(&total_statistics, 0, sizeof(total_statistics));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_generation_residency_statistics_accumulate(
            plan, &id4_ideogram4_generation_phase_descriptors[i],
            &total_statistics));
  }

  iree_device_size_t parameter_high_water_mark = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    id4_ideogram4_generation_residency_statistics_t phase_statistics;
    memset(&phase_statistics, 0, sizeof(phase_statistics));
    IREE_RETURN_IF_ERROR(
        id4_ideogram4_generation_residency_statistics_accumulate(
            plan, &id4_ideogram4_generation_phase_descriptors[i],
            &phase_statistics));
    if (phase_statistics.parameter_byte_length > parameter_high_water_mark) {
      parameter_high_water_mark = phase_statistics.parameter_byte_length;
    }
  }

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"total_stage_parameter_byte_length\":%" PRIu64
      ",\"phase_parameter_high_water_mark\":%" PRIu64
      ",\"largest_stage_parameter_byte_length\":%" PRIu64
      ",\"total_stage_boundary_byte_length\":%" PRIu64 ",\"phases\":[",
      (uint64_t)total_statistics.parameter_byte_length,
      (uint64_t)parameter_high_water_mark,
      (uint64_t)total_statistics.largest_stage_parameter_byte_length,
      (uint64_t)total_statistics.stage_boundary_byte_length));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_ideogram4_generation_phase_descriptors); ++i) {
    if (i != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_phase_json(
        plan, builder, &id4_ideogram4_generation_phase_descriptors[i]));
  }
  return iree_string_builder_append_cstring(builder, "]}");
}

static iree_status_t id4_ideogram4_generation_plan_append_stage_json(
    iree_string_builder_t* builder,
    id4_ideogram4_generation_stage_ordinal_t ordinal,
    const id4_pipeline_plan_t* stage_plan) {
  const id4_ideogram4_generation_stage_descriptor_t* descriptor =
      id4_ideogram4_generation_stage_descriptor(ordinal);
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown Ideogram 4 generation stage ordinal %u",
                            (uint32_t)ordinal);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, "\"%s\":", descriptor->key));
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
      "%" PRIu32 ",\"qwen_token_capacity\":%" PRIu32
      ",\"image_token_count\":%" PRIu32
      ",\"conditioned_dit_token_count\":%" PRIu32
      ",\"conditioned_dit_token_capacity\":%" PRIu32
      ",\"unconditioned_dit_token_count\":%" PRIu32
      ",\"unconditioned_dit_token_capacity\":%" PRIu32
      ",\"denoise_step_count\":%" PRIu32 ",\"diffusion_latent_shape\":",
      plan->summary.qwen_token_count, plan->summary.qwen_token_capacity,
      plan->summary.image_token_count,
      plan->summary.conditioned_dit_token_count,
      plan->summary.conditioned_dit_token_capacity,
      plan->summary.unconditioned_dit_token_count,
      plan->summary.unconditioned_dit_token_capacity,
      plan->summary.denoise_step_count));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_shape_json(
      builder, plan->summary.diffusion_latent_shape));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, ",\"decoded_image_shape\":"));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_shape_json(
      builder, plan->summary.decoded_image_shape));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"dit_activation_format\":%u,\"dit_weight_execution_format\":%u,"
      "\"qwen_weight_execution_strategy\":%u,"
      "\"qwen_attention_implementation\":%u,"
      "\"dit_attention_implementation\":%u,"
      "\"dit_feed_forward_implementation\":%u,\"vae_tiling\":",
      (uint32_t)plan->summary.dit_activation_format,
      (uint32_t)plan->summary.dit_weight_execution_format,
      (uint32_t)plan->summary.qwen_weight_execution_strategy,
      (uint32_t)plan->summary.qwen_attention_implementation,
      (uint32_t)plan->summary.dit_attention_implementation,
      (uint32_t)plan->summary.dit_feed_forward_implementation));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_tiling_json(
      builder, plan->summary.vae_tiling));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "},\"residency\":"));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_generation_plan_append_residency_json(plan, builder));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"stages\":{"));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, plan->qwen_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
      plan->dit_conditioned_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
      plan->dit_unconditioned_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_NOISE, plan->sampler_noise_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_SAMPLER,
      plan->sampler_denoise_plan));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_plan_append_stage_json(
      builder, ID4_IDEOGRAM4_GENERATION_STAGE_DECODE, plan->decode_plan));
  return iree_string_builder_append_cstring(builder, "}}");
}
