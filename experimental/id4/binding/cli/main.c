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
#include "experimental/id4/pipeline/plan.h"
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
IREE_FLAG(bool, dry_run, false,
          "Plan a full generation request and exit without loading parameters "
          "or issuing device work.");
IREE_FLAG(int32_t, latent_width, 0,
          "Diffusion latent width for full generation planning.");
IREE_FLAG(int32_t, latent_height, 0,
          "Diffusion latent height for full generation planning.");
IREE_FLAG(int32_t, denoise_steps, 0,
          "Denoise step count for full generation planning.");
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

static iree_status_t id4_cli_reject_unimplemented_flags(void) {
  if (strlen(FLAG_output) != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "--output requires downstream sampler, DiT, and VAE CLI wiring");
  }
  if (strlen(FLAG_dump_diagnostics) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "--dump_diagnostics is not wired yet");
  }
  if (strlen(FLAG_profile_output) != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "--profile_output is not wired yet");
  }
  const bool has_generation_flags = FLAG_latent_width != 0 ||
                                    FLAG_latent_height != 0 ||
                                    FLAG_denoise_steps != 0;
  if (!FLAG_dry_run && has_generation_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "generation shape flags require --dry_run until "
                            "full generation issue is wired");
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_write_plan(iree_string_view_t output_path,
                                        const id4_pipeline_plan_t* plan,
                                        iree_allocator_t host_allocator) {
  if (iree_string_view_is_empty(output_path)) return iree_ok_status();

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        output_path, iree_make_const_byte_span(json.data, json.size),
        host_allocator);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
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

static iree_status_t id4_cli_make_generation_config(
    id4_ideogram4_generation_config_t* out_config) {
  IREE_ASSERT_ARGUMENT(out_config);
  memset(out_config, 0, sizeof(*out_config));
  if (FLAG_latent_width <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--latent_width must be positive");
  }
  if (FLAG_latent_height <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--latent_height must be positive");
  }
  if (FLAG_denoise_steps <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--denoise_steps must be positive");
  }
  out_config->structure_size = sizeof(*out_config);
  out_config->diffusion_latent_shape = id4_pipeline_program_make_shape_rank4(
      (uint64_t)FLAG_latent_width, (uint64_t)FLAG_latent_height, 128, 1);
  out_config->denoise_step_count = (uint32_t)FLAG_denoise_steps;
  out_config->dit_activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  out_config->vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  return iree_ok_status();
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

  iree_status_t status = id4_cli_reject_unimplemented_flags();
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
  id4_ideogram4_generation_config_t generation_config;
  memset(&generation_config, 0, sizeof(generation_config));
  if (iree_status_is_ok(status)) {
    status = id4_cli_make_generation_config(&generation_config);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_generation_plan_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = &request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    plan_options.generation = generation_config;
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;
    status = id4_ideogram4_session_plan_generation(session, &plan_options,
                                                   &generation_plan);
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
              "x%" PRIu64 "x%" PRIu64 " steps=%" PRIu32 "\n",
              summary.qwen_token_count, summary.diffusion_latent_shape.dims[0],
              summary.diffusion_latent_shape.dims[1],
              summary.diffusion_latent_shape.dims[2],
              summary.denoise_step_count);
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

static iree_status_t id4_cli_run_qwen(iree_allocator_t host_allocator) {
  id4_ideogram4_request_t request;
  memset(&request, 0, sizeof(request));
  iree_tokenizer_t* tokenizer = NULL;
  id4_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  bool runtime_context_initialized = false;
  id4_pipeline_kernel_library_t* kernel_library = NULL;
  iree_io_parameter_provider_t* parameter_provider = NULL;
  id4_ideogram4_session_t* session = NULL;
  iree_hal_semaphore_t* completion_semaphore = NULL;
  id4_ideogram4_qwen_execution_t* execution = NULL;
  bool qwen_was_issued = false;
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  iree_status_t status = id4_cli_reject_unimplemented_flags();
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
    status = id4_tooling_create_parameter_provider_from_flags(
        iree_string_view_empty(), host_allocator, &parameter_provider);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_create_loaded_session(&runtime_context, &diagnostics_sink,
                                           host_allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_device_t* device =
        id4_tooling_runtime_context_primary_device(&runtime_context);
    status = iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &completion_semaphore);
  }
  iree_hal_semaphore_t* completion_semaphore_storage = NULL;
  uint64_t completion_payload_storage = 1;
  iree_hal_semaphore_list_t completion_list = iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    completion_list = id4_cli_single_semaphore_list(
        &completion_semaphore_storage, &completion_payload_storage,
        completion_semaphore, completion_payload_storage);
    id4_ideogram4_qwen_issue_options_t issue_options;
    memset(&issue_options, 0, sizeof(issue_options));
    issue_options.structure_size = sizeof(issue_options);
    issue_options.request = &request;
    issue_options.tokenizer = tokenizer;
    issue_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    issue_options.parameter_provider = parameter_provider;
    issue_options.kernel_library = kernel_library;
    issue_options.device_index = 0;
    issue_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    issue_options.command_buffer_mode = runtime_context.command_buffer_mode;
    issue_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
    issue_options.signal_semaphore_list = completion_list;
    issue_options.diagnostics_sink = &diagnostics_sink;
    status =
        id4_ideogram4_session_issue_qwen(session, &issue_options, &execution);
    qwen_was_issued = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_write_plan(iree_make_cstring_view(FLAG_dump_plan),
                                id4_ideogram4_qwen_execution_plan(execution),
                                host_allocator);
  }
  if (qwen_was_issued) {
    status = iree_status_join(
        status,
        iree_hal_semaphore_list_wait(completion_list, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_qwen_result_t result;
    memset(&result, 0, sizeof(result));
    status = id4_ideogram4_qwen_execution_result(execution, &result);
    if (iree_status_is_ok(status)) {
      fprintf(stdout,
              "Qwen conditioning complete: tokens=%" PRIu32
              " condition_bytes=%" PRIu64 "\n",
              result.token_count, (uint64_t)result.condition_binding.length);
    }
  }

  id4_ideogram4_qwen_execution_release(execution);
  iree_hal_semaphore_release(completion_semaphore);
  id4_ideogram4_session_release(session);
  iree_io_parameter_provider_release(parameter_provider);
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
      "Loads a prompt JSON request, tokenizes it, and either plans a full "
      "generation or runs the Qwen3-VL conditioning stage.\n"
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
    status = id4_cli_run_qwen(host_allocator);
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
