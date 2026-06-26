// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/tooling/image.h"
#include "experimental/id4/tooling/readback.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

IREE_FLAG(string, tokenizer, "", "Path to the HuggingFace tokenizer JSON.");
IREE_FLAG(string, prompt_json, "",
          "Full JSON prompt/configuration payload for one generation.");
IREE_FLAG(string, prompt_json_file, "",
          "Path to a JSON prompt/configuration payload for one generation.");
IREE_FLAG(string, output, "", "Output image path.");
IREE_FLAG(string, dit_parameter_format, "bf16",
          "DiT parameter format: bf16 or mixed_bf16_fp8_e4m3.");
IREE_FLAG(string, dit_activation_format, "bf16_linear_input",
          "DiT activation format: bf16_linear_input or f32_canonical.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT native-FP8 parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT native-FP8 parameter scope.");
IREE_FLAG(bool, dry_run, false,
          "Plan a full generation request and exit without loading parameters "
          "or issuing device work.");
IREE_FLAG(string, dump_plan, "",
          "Path to write the structured pipeline plan JSON.");
IREE_FLAG(string, dump_diagnostics, "",
          "Directory for loomc, HAL, tensor, and stage diagnostics.");
IREE_FLAG(string, profile_output, "",
          "Path to write queue and dispatch-level profiling data.");

static iree_status_t id4_cli_load_tokenizer(iree_string_view_t tokenizer_path,
                                            iree_allocator_t host_allocator,
                                            iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;
  if (iree_string_view_is_empty(tokenizer_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokenizer is required");
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      tokenizer_path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);
  if (iree_status_is_ok(status)) {
    iree_string_view_t json =
        iree_make_string_view((const char*)file_contents->const_buffer.data,
                              file_contents->const_buffer.data_length);
    status = iree_tokenizer_from_huggingface_json(json, host_allocator,
                                                  out_tokenizer);
  }
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t id4_cli_parse_request(
    iree_allocator_t host_allocator, id4_ideogram4_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  iree_string_view_t prompt_json = iree_make_cstring_view(FLAG_prompt_json);
  iree_string_view_t prompt_json_file =
      iree_make_cstring_view(FLAG_prompt_json_file);
  const bool has_inline_prompt = !iree_string_view_is_empty(prompt_json);
  const bool has_prompt_file = !iree_string_view_is_empty(prompt_json_file);
  if (has_inline_prompt == has_prompt_file) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exactly one of --prompt_json or --prompt_json_file is required");
  }
  if (has_inline_prompt) {
    return id4_ideogram4_request_parse_json(prompt_json, host_allocator,
                                            out_request);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status =
      iree_io_file_contents_map(prompt_json_file, IREE_IO_FILE_ACCESS_READ,
                                host_allocator, &file_contents);
  if (iree_status_is_ok(status)) {
    iree_string_view_t file_json =
        iree_make_string_view((const char*)file_contents->const_buffer.data,
                              file_contents->const_buffer.data_length);
    status = id4_ideogram4_request_parse_json(file_json, host_allocator,
                                              out_request);
  }
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t id4_cli_reject_unimplemented_diagnostics_flags(void) {
  if (strlen(FLAG_dump_diagnostics) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "--dump_diagnostics is not wired yet");
  }
  if (strlen(FLAG_profile_output) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "--profile_output is not wired yet");
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_parse_dit_parameter_format(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_status_t id4_cli_parse_dit_activation_format(
    id4_ideogram4_dit_activation_format_t* out_format) {
  IREE_ASSERT_ARGUMENT(out_format);
  iree_string_view_t value = iree_make_cstring_view(FLAG_dit_activation_format);
  if (iree_string_view_equal(value, IREE_SV("bf16_linear_input"))) {
    *out_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(value, IREE_SV("f32_canonical"))) {
    *out_format = ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_F32_CANONICAL;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "--dit_activation_format must be bf16_linear_input or f32_canonical");
}

static iree_status_t id4_cli_validate_execution_flags(void) {
  IREE_RETURN_IF_ERROR(id4_cli_reject_unimplemented_diagnostics_flags());
  if (strlen(FLAG_output) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--output is required when executing generation");
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_validate_dry_run_flags(void) {
  IREE_RETURN_IF_ERROR(id4_cli_reject_unimplemented_diagnostics_flags());
  if (strlen(FLAG_output) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--output requires generation execution; omit --dry_run");
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_write_generation_plan(
    iree_string_view_t output_path, const id4_ideogram4_generation_plan_t* plan,
    iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(output_path)) return iree_ok_status();

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      id4_ideogram4_generation_plan_format_json(plan, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        output_path, iree_make_const_byte_span(json.data, json.size),
        host_allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_hal_semaphore_list_t id4_cli_single_semaphore_list(
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

static iree_status_t id4_cli_create_loaded_session(
    id4_tooling_runtime_context_t* runtime_context,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    iree_allocator_t host_allocator, id4_ideogram4_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;
  id4_ideogram4_session_create_options_t session_options;
  memset(&session_options, 0, sizeof(session_options));
  session_options.structure_size = sizeof(session_options);
  session_options.services =
      id4_tooling_runtime_context_stage_services(runtime_context);
  session_options.kernel_cache = runtime_context->kernel_cache;
  session_options.parameter_scopes.qwen = IREE_SV("qwen");
  session_options.parameter_scopes.dit_conditioned = IREE_SV("dit_cond");
  session_options.parameter_scopes.dit_unconditioned = IREE_SV("dit_uncond");
  session_options.parameter_scopes.vae = IREE_SV("vae");
  IREE_RETURN_IF_ERROR(id4_cli_parse_dit_parameter_format(
      &session_options.dit_parameter_format));
  if (session_options.dit_parameter_format ==
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3) {
    session_options.parameter_scopes.dit_conditioned_fp8 =
        iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
    session_options.parameter_scopes.dit_unconditioned_fp8 =
        iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
  }
  session_options.vae_activation_format =
      ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;
  IREE_RETURN_IF_ERROR(id4_ideogram4_session_create(
      &session_options, host_allocator, out_session));

  id4_ideogram4_session_load_options_t load_options;
  memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = diagnostics_sink;
  iree_status_t status =
      id4_ideogram4_session_load(*out_session, &load_options);
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_session_release(*out_session);
    *out_session = NULL;
  }
  return status;
}

static iree_status_t id4_cli_make_generation_plan_policy(
    id4_ideogram4_generation_plan_policy_t* out_policy) {
  IREE_ASSERT_ARGUMENT(out_policy);
  id4_ideogram4_generation_plan_policy_t policy;
  memset(&policy, 0, sizeof(policy));
  policy.structure_size = sizeof(policy);
  IREE_RETURN_IF_ERROR(
      id4_cli_parse_dit_activation_format(&policy.dit_activation_format));
  policy.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  *out_policy = policy;
  return iree_ok_status();
}

static void id4_cli_release_parameter_providers(
    id4_ideogram4_generation_parameter_providers_t* providers) {
  iree_io_parameter_provider_release(providers->vae);
  iree_io_parameter_provider_release(providers->dit_unconditioned);
  iree_io_parameter_provider_release(providers->dit_conditioned);
  iree_io_parameter_provider_release(providers->qwen);
  memset(providers, 0, sizeof(*providers));
}

static iree_status_t id4_cli_create_parameter_providers(
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_parameter_providers_t* out_providers) {
  memset(out_providers, 0, sizeof(*out_providers));
  id4_ideogram4_dit_parameter_format_t dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_RETURN_IF_ERROR(
      id4_cli_parse_dit_parameter_format(&dit_parameter_format));

  iree_status_t status = iree_ok_status();
  if (dit_parameter_format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    id4_tooling_parameter_provider_request_t requests[] = {
        {
            // Qwen3-VL text encoder parameter scope.
            .scope = IREE_SV("qwen"),
            // Qwen provider output.
            .out_provider = &out_providers->qwen,
        },
        {
            // Conditioned DiT parameter scope.
            .scope = IREE_SV("dit_cond"),
            // Conditioned DiT provider output.
            .out_provider = &out_providers->dit_conditioned,
        },
        {
            // Unconditioned DiT parameter scope.
            .scope = IREE_SV("dit_uncond"),
            // Unconditioned DiT provider output.
            .out_provider = &out_providers->dit_unconditioned,
        },
        {
            // VAE parameter scope.
            .scope = IREE_SV("vae"),
            // VAE provider output.
            .out_provider = &out_providers->vae,
        },
    };
    status = id4_tooling_create_parameter_providers_from_flags(
        IREE_ARRAYSIZE(requests), requests, host_allocator);
  } else {
    iree_io_parameter_provider_t* dit_conditioned_bf16 = NULL;
    iree_io_parameter_provider_t* dit_conditioned_fp8 = NULL;
    iree_io_parameter_provider_t* dit_unconditioned_bf16 = NULL;
    iree_io_parameter_provider_t* dit_unconditioned_fp8 = NULL;
    const iree_string_view_t dit_conditioned_fp8_scope =
        iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
    const iree_string_view_t dit_unconditioned_fp8_scope =
        iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
    id4_tooling_parameter_provider_request_t requests[] = {
        {
            // Qwen3-VL text encoder parameter scope.
            .scope = IREE_SV("qwen"),
            // Qwen provider output.
            .out_provider = &out_providers->qwen,
        },
        {
            // Conditioned DiT BF16 parameter scope.
            .scope = IREE_SV("dit_cond"),
            // Conditioned DiT BF16 provider output.
            .out_provider = &dit_conditioned_bf16,
        },
        {
            // Conditioned DiT native-FP8 parameter scope.
            .scope = dit_conditioned_fp8_scope,
            // Conditioned DiT native-FP8 provider output.
            .out_provider = &dit_conditioned_fp8,
        },
        {
            // Unconditioned DiT BF16 parameter scope.
            .scope = IREE_SV("dit_uncond"),
            // Unconditioned DiT BF16 provider output.
            .out_provider = &dit_unconditioned_bf16,
        },
        {
            // Unconditioned DiT native-FP8 parameter scope.
            .scope = dit_unconditioned_fp8_scope,
            // Unconditioned DiT native-FP8 provider output.
            .out_provider = &dit_unconditioned_fp8,
        },
        {
            // VAE parameter scope.
            .scope = IREE_SV("vae"),
            // VAE provider output.
            .out_provider = &out_providers->vae,
        },
    };
    status = id4_tooling_create_parameter_providers_from_flags(
        IREE_ARRAYSIZE(requests), requests, host_allocator);
    if (iree_status_is_ok(status)) {
      const id4_tooling_parameter_provider_set_entry_t entries[] = {
          {
              // Conditioned DiT BF16 parameter scope.
              .scope = IREE_SV("dit_cond"),
              // Conditioned DiT BF16 provider.
              .provider = dit_conditioned_bf16,
          },
          {
              // Conditioned DiT native-FP8 parameter scope.
              .scope = dit_conditioned_fp8_scope,
              // Conditioned DiT native-FP8 provider.
              .provider = dit_conditioned_fp8,
          },
      };
      status = id4_tooling_create_parameter_provider_set(
          IREE_ARRAYSIZE(entries), entries, host_allocator,
          &out_providers->dit_conditioned);
    }
    if (iree_status_is_ok(status)) {
      const id4_tooling_parameter_provider_set_entry_t entries[] = {
          {
              // Unconditioned DiT BF16 parameter scope.
              .scope = IREE_SV("dit_uncond"),
              // Unconditioned DiT BF16 provider.
              .provider = dit_unconditioned_bf16,
          },
          {
              // Unconditioned DiT native-FP8 parameter scope.
              .scope = dit_unconditioned_fp8_scope,
              // Unconditioned DiT native-FP8 provider.
              .provider = dit_unconditioned_fp8,
          },
      };
      status = id4_tooling_create_parameter_provider_set(
          IREE_ARRAYSIZE(entries), entries, host_allocator,
          &out_providers->dit_unconditioned);
    }
    iree_io_parameter_provider_release(dit_unconditioned_fp8);
    iree_io_parameter_provider_release(dit_unconditioned_bf16);
    iree_io_parameter_provider_release(dit_conditioned_fp8);
    iree_io_parameter_provider_release(dit_conditioned_bf16);
  }
  if (!iree_status_is_ok(status)) {
    id4_cli_release_parameter_providers(out_providers);
  }
  return status;
}

static id4_pipeline_tensor_shape_t id4_cli_convert_program_shape(
    id4_pipeline_program_shape_t shape) {
  id4_pipeline_tensor_shape_t tensor_shape;
  memset(&tensor_shape, 0, sizeof(tensor_shape));
  tensor_shape.rank = shape.rank;
  memcpy(tensor_shape.dims, shape.dims, sizeof(tensor_shape.dims));
  return tensor_shape;
}

static iree_status_t id4_cli_run_generation_dry_run(
    iree_allocator_t host_allocator) {
  id4_ideogram4_request_t request;
  memset(&request, 0, sizeof(request));
  iree_tokenizer_t* tokenizer = NULL;
  id4_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  bool runtime_context_initialized = false;
  id4_ideogram4_session_t* session = NULL;
  id4_ideogram4_generation_plan_t* generation_plan = NULL;

  iree_status_t status = id4_cli_validate_dry_run_flags();
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_request(host_allocator, &request);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_load_tokenizer(iree_make_cstring_view(FLAG_tokenizer),
                                    host_allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    id4_tooling_runtime_context_options_t runtime_options;
    memset(&runtime_options, 0, sizeof(runtime_options));
    runtime_options.structure_size = sizeof(runtime_options);
    runtime_options.executable_cache_identifier = IREE_SV("id4");
    status = id4_tooling_runtime_context_initialize_from_flags(
        &runtime_options, host_allocator, &runtime_context);
    runtime_context_initialized = iree_status_is_ok(status);
  }
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_loaded_session(&runtime_context, &diagnostics_sink,
                                           host_allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = &request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    status = id4_cli_make_generation_plan_policy(&plan_options.policy);
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_session_plan_generation(session, &plan_options,
                                                     &generation_plan);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_write_generation_plan(iree_make_cstring_view(FLAG_dump_plan),
                                      generation_plan, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_plan_summary_t summary;
    memset(&summary, 0, sizeof(summary));
    status = id4_ideogram4_generation_plan_summary(generation_plan, &summary);
    if (iree_status_is_ok(status)) {
      fprintf(stdout,
              "Ideogram 4 generation plan: tokens=%" PRIu32 " latent=%" PRIu64
              "x%" PRIu64 "x%" PRIu64 " image=%" PRIu64 "x%" PRIu64
              " steps=%" PRIu32 "\n",
              summary.qwen_token_count, summary.diffusion_latent_shape.dims[0],
              summary.diffusion_latent_shape.dims[1],
              summary.diffusion_latent_shape.dims[2],
              summary.decoded_image_shape.dims[0],
              summary.decoded_image_shape.dims[1], summary.denoise_step_count);
    }
  }

  id4_ideogram4_generation_plan_release(generation_plan);
  id4_ideogram4_session_release(session);
  if (runtime_context_initialized) {
    id4_tooling_runtime_context_deinitialize(&runtime_context);
  }
  iree_tokenizer_free(tokenizer);
  id4_ideogram4_request_deinitialize(&request, host_allocator);
  return status;
}

static iree_status_t id4_cli_write_decoded_image(
    iree_string_view_t output_path,
    id4_ideogram4_generation_plan_summary_t summary,
    iree_const_byte_span_t decoded_image, iree_allocator_t host_allocator) {
  id4_tooling_write_f32_rgb_ppm_options_t image_options;
  memset(&image_options, 0, sizeof(image_options));
  image_options.structure_size = sizeof(image_options);
  image_options.path = output_path;
  image_options.shape =
      id4_cli_convert_program_shape(summary.decoded_image_shape);
  image_options.pixels = decoded_image;
  image_options.normalization =
      ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE;
  image_options.host_allocator = host_allocator;
  return id4_tooling_write_f32_rgb_ppm(&image_options);
}

static iree_status_t id4_cli_run_generation(iree_allocator_t host_allocator) {
  id4_ideogram4_request_t request;
  memset(&request, 0, sizeof(request));
  iree_tokenizer_t* tokenizer = NULL;
  id4_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  bool runtime_context_initialized = false;
  id4_pipeline_kernel_library_t* kernel_library = NULL;
  id4_ideogram4_generation_parameter_providers_t parameter_providers;
  memset(&parameter_providers, 0, sizeof(parameter_providers));
  id4_ideogram4_session_t* session = NULL;
  id4_ideogram4_generation_plan_t* generation_plan = NULL;
  id4_ideogram4_generation_bundle_t* generation_bundle = NULL;
  iree_hal_semaphore_t* prepare_semaphore = NULL;
  iree_hal_semaphore_t* completion_semaphore = NULL;
  id4_ideogram4_generation_execution_t* execution = NULL;
  bool generation_was_prepared = false;
  bool generation_was_issued = false;
  id4_tooling_host_bytes_t decoded_image_bytes;
  memset(&decoded_image_bytes, 0, sizeof(decoded_image_bytes));
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  iree_status_t status = id4_cli_validate_execution_flags();
  if (iree_status_is_ok(status)) {
    status = id4_cli_parse_request(host_allocator, &request);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_load_tokenizer(iree_make_cstring_view(FLAG_tokenizer),
                                    host_allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    id4_tooling_runtime_context_options_t runtime_options;
    memset(&runtime_options, 0, sizeof(runtime_options));
    runtime_options.structure_size = sizeof(runtime_options);
    runtime_options.executable_cache_identifier = IREE_SV("id4");
    status = id4_tooling_runtime_context_initialize_from_flags(
        &runtime_options, host_allocator, &runtime_context);
    runtime_context_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_tooling_create_embedded_kernel_library(host_allocator,
                                                        &kernel_library);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_loaded_session(&runtime_context, &diagnostics_sink,
                                           host_allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_parameter_providers(host_allocator,
                                                &parameter_providers);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &prepare_semaphore);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &completion_semaphore);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = &request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    status = id4_cli_make_generation_plan_policy(&plan_options.policy);
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_session_plan_generation(session, &plan_options,
                                                     &generation_plan);
    }
  }
  id4_ideogram4_generation_plan_summary_t summary;
  memset(&summary, 0, sizeof(summary));
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_generation_plan_summary(generation_plan, &summary);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_cli_write_generation_plan(iree_make_cstring_view(FLAG_dump_plan),
                                      generation_plan, host_allocator);
  }
  iree_hal_semaphore_t* prepare_semaphore_storage = NULL;
  uint64_t prepare_payload_storage = 1;
  iree_hal_semaphore_list_t prepare_signal_list =
      iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    prepare_signal_list = id4_cli_single_semaphore_list(
        &prepare_semaphore_storage, &prepare_payload_storage, prepare_semaphore,
        prepare_payload_storage);
    id4_ideogram4_generation_prepare_options_t prepare_options;
    memset(&prepare_options, 0, sizeof(prepare_options));
    prepare_options.structure_size = sizeof(prepare_options);
    prepare_options.parameter_providers = parameter_providers;
    prepare_options.kernel_library = kernel_library;
    prepare_options.command_buffer_mode = runtime_context.command_buffer_mode;
    prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
    prepare_options.signal_semaphore_list = prepare_signal_list;
    prepare_options.diagnostics_sink = &diagnostics_sink;
    status = id4_ideogram4_session_prepare_generation(
        session, generation_plan, &prepare_options, &generation_bundle);
    generation_was_prepared = iree_status_is_ok(status);
  }
  iree_hal_semaphore_t* completion_semaphore_storage = NULL;
  uint64_t completion_payload_storage = 1;
  iree_hal_semaphore_list_t completion_list = iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    completion_list = id4_cli_single_semaphore_list(
        &completion_semaphore_storage, &completion_payload_storage,
        completion_semaphore, completion_payload_storage);
    id4_ideogram4_generation_issue_options_t issue_options;
    memset(&issue_options, 0, sizeof(issue_options));
    issue_options.structure_size = sizeof(issue_options);
    issue_options.request = &request;
    issue_options.tokenizer = tokenizer;
    issue_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    issue_options.wait_semaphore_list = prepare_signal_list;
    issue_options.signal_semaphore_list = completion_list;
    issue_options.diagnostics_sink = &diagnostics_sink;
    status = id4_ideogram4_session_issue_generation(session, generation_bundle,
                                                    &issue_options, &execution);
    generation_was_issued = iree_status_is_ok(status);
  }
  if (generation_was_issued) {
    status = iree_status_join(
        status,
        iree_hal_semaphore_list_wait(completion_list, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
  } else if (generation_was_prepared) {
    status = iree_status_join(
        status, iree_hal_semaphore_list_wait(prepare_signal_list,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_result_t result;
    memset(&result, 0, sizeof(result));
    status = id4_ideogram4_generation_execution_result(execution, &result);
    id4_tooling_readback_buffer_binding_options_t readback_options;
    memset(&readback_options, 0, sizeof(readback_options));
    readback_options.structure_size = sizeof(readback_options);
    readback_options.device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    readback_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    readback_options.binding = result.decoded_image_binding;
    readback_options.wait_semaphore_list = completion_list;
    readback_options.host_allocator = host_allocator;
    if (iree_status_is_ok(status)) {
      status = id4_tooling_readback_buffer_binding(&readback_options,
                                                   &decoded_image_bytes);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_write_decoded_image(
        iree_make_cstring_view(FLAG_output), summary,
        iree_make_const_byte_span(decoded_image_bytes.data,
                                  decoded_image_bytes.length),
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout,
            "Ideogram 4 generation complete: tokens=%" PRIu32 " image=%" PRIu64
            "x%" PRIu64 " output=%s\n",
            summary.qwen_token_count, summary.decoded_image_shape.dims[0],
            summary.decoded_image_shape.dims[1], FLAG_output);
  }

  id4_tooling_host_bytes_deinitialize(&decoded_image_bytes, host_allocator);
  id4_ideogram4_generation_execution_release(execution);
  id4_ideogram4_generation_bundle_release(generation_bundle);
  id4_ideogram4_generation_plan_release(generation_plan);
  iree_hal_semaphore_release(prepare_semaphore);
  iree_hal_semaphore_release(completion_semaphore);
  id4_ideogram4_session_release(session);
  id4_cli_release_parameter_providers(&parameter_providers);
  id4_pipeline_kernel_library_release(kernel_library);
  if (runtime_context_initialized) {
    id4_tooling_runtime_context_deinitialize(&runtime_context);
  }
  iree_tokenizer_free(tokenizer);
  id4_ideogram4_request_deinitialize(&request, host_allocator);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "id4",
      "Experimental Ideogram 4 HAL pipeline runner.\n"
      "\n"
      "Loads a prompt JSON request, tokenizes it, and either plans or runs a "
      "full generation.\n"
      "Pass exactly one of --prompt_json=... or --prompt_json_file=....\n"
      "Model parameters are loaded with standard --parameters= flags.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  int exit_code = EXIT_SUCCESS;
  if (argc > 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "no positional arguments expected");
  } else if (FLAG_dry_run) {
    status = id4_cli_run_generation_dry_run(host_allocator);
  } else {
    status = id4_cli_run_generation(host_allocator);
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
