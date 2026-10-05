// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/krea2/model.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#include "experimental/loom_serve/device.h"
#include "experimental/loom_serve/models/krea2/request.h"
#include "experimental/loom_serve/weights.h"
#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tooling/device_util.h"

struct loom_serve_krea2_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t allocator;
  // Fixed pixel height for the prepared command.
  uint32_t height;
  // Fixed pixel width for the prepared command.
  uint32_t width;
  // Fixed retained text extent for the prepared command.
  uint32_t text_tokens;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Shared device/timeline ownership, outliving all accepted work.
  loom_serve_device_t* owner;
  // Optional profiling session, ended after accepted work drains.
  iree_hal_profiling_from_flags_t* profiling;
  // Cold live-source compiler and its task pool.
  loom_serve_jit_t* jit;
  // Compiled image root and reflection.
  loom_serve_jit_stage_t* stage;
  // Reusable command retaining immutable parameter domains.
  iree_hal_command_buffer_t* command;
  // Encoder, Turbo, optional adapter and VAE fixed buffers.
  iree_hal_buffer_t* weights[4];
  // Inputs, final RGB and reflected workspace; partial slots are NULL.
  iree_hal_buffer_binding_t bindings[LOOM_SERVE_KREA2_INPUT_COUNT + 2];
  // Number of input slots; the adapted command has the extra strength slot.
  iree_host_size_t input_count;
  // Completed NCHW F32 RGB feedback, overwritten by the next generation.
  iree_byte_span_t output;
};

iree_status_t loom_serve_krea2_model_destroy(loom_serve_krea2_model_t* model) {
  if (!model) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  if (model->owner) {
    status =
        loom_serve_execution_drain(loom_serve_device_execution(model->owner));
  }
  status = iree_status_join(
      status, iree_hal_end_profiling_from_flags(model->profiling));
  iree_hal_command_buffer_release(model->command);
  loom_serve_jit_stage_destroy(model->stage);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->weights); ++i) {
    iree_hal_buffer_release(model->weights[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(model->bindings); ++i) {
    iree_hal_buffer_release(model->bindings[i].buffer);
  }
  loom_serve_jit_destroy(model->jit);
  iree_tokenizer_free(model->tokenizer);
  iree_allocator_free(model->allocator, model->output.data);
  loom_serve_device_destroy(model->owner);
  iree_allocator_free(model->allocator, model);
  return status;
}

static iree_status_t krea2_load_tokenizer(loom_serve_krea2_model_t* model,
                                          iree_string_view_t directory) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      directory, IREE_SV("tokenizer/tokenizer.json"), model->allocator, &path));
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(path), model->allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        model->allocator, &model->tokenizer);
  }
  iree_io_file_contents_free(contents);
  iree_allocator_free(model->allocator, path);
  return status;
}

static iree_status_t krea2_model_initialize(
    loom_serve_krea2_model_t* model,
    const loom_serve_krea2_model_options_t* options,
    const iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT]) {
  const iree_allocator_t allocator = model->allocator;
  const bool adapted = options->adapter_path.size != 0;
  bool retain_profile_metadata = false;
  IREE_RETURN_IF_ERROR(
      iree_hal_profiling_from_flags_requires_retained_command_buffer_metadata(
          &retain_profile_metadata));
  const iree_hal_command_buffer_mode_t command_mode =
      retain_profile_metadata
          ? IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA
          : IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
  IREE_RETURN_IF_ERROR(
      krea2_load_tokenizer(model, options->checkpoint_directory));
  model->output.data_length =
      (iree_host_size_t)options->height * options->width * 12;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, model->output.data_length, (void**)&model->output.data));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create(IREE_SV("amdgpu"), allocator, &model->owner));
  iree_hal_device_t* device = loom_serve_device_handle(model->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(model->owner);
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(device, dispatch,
                                             options->source_directory, NULL,
                                             allocator, &model->jit));
  const uint32_t images = (options->height / 16) * (options->width / 16);
  const char* keys[] = {"krea2.block_tokens",  "krea2.image_tokens",
                        "krea2.text_tokens",   "krea2.time_count",
                        "krea2.latent_height", "krea2.latent_width"};
  const uint32_t values[] = {images + options->text_tokens, images,
                             options->text_tokens,          8,
                             options->height / 8,           options->width / 8};
  char strings[IREE_ARRAYSIZE(keys)][16];
  loomc_config_binding_t config_bindings[IREE_ARRAYSIZE(keys)];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(keys); ++i) {
    snprintf(strings[i], sizeof(strings[i]), "%u", values[i]);
    config_bindings[i] = (loomc_config_binding_t){
        loomc_make_cstring_view(keys[i]), loomc_make_cstring_view(strings[i])};
  }
  const loomc_config_options_t config = {
      config_bindings,
      IREE_ARRAYSIZE(keys),
      {0},
      LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  IREE_RETURN_IF_ERROR(loom_serve_jit_compile(
      model->jit,
      adapted ? IREE_SV("sample_image_adapted") : IREE_SV("sample_image"),
      &config, &model->stage));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stage);
  model->input_count = LOOM_SERVE_KREA2_INPUT_COUNT - !adapted;
  const iree_host_size_t binding_count = model->input_count + 2;
  const uint32_t roots = adapted ? 4 : 3;
  if (program->requirements.rebindable_binding_count != binding_count ||
      program->requirements.transient.binding_index != model->input_count + 1 ||
      program->requirements.fixed_buffer_count != roots ||
      program->parameter_roots.count != roots ||
      program->requirements.launch_counts.binding_index != UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source root does not implement the Krea image ABI");
  }
  uint64_t parameter_bytes = 0;
  for (uint32_t i = 0; i < roots; ++i) {
    parameter_bytes +=
        loom_cmd_program_parameter_root_at(program, i).required_byte_length;
  }
  printf("{\"event\":\"image_residency\",\"parameter_bytes\":%" PRIu64
         ",\"workspace_bytes\":%" PRIu64 ",\"kernels\":%u}\n",
         parameter_bytes, program->requirements.transient.required_byte_length,
         program->requirements.executable_count);
  fflush(stdout);
  IREE_RETURN_IF_ERROR(iree_hal_begin_device_group_profiling_from_flags(
      loom_serve_device_group(model->owner), allocator, &model->profiling));
  char* policy = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("weights.loom"), allocator, &policy));
  const char* names[] = {"text_encoder/model.safetensors", "turbo.safetensors",
                         "vae/diffusion_pytorch_model.safetensors"};
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < roots && iree_status_is_ok(status); ++i) {
    char* path = NULL;
    const bool adapter_root = adapted && i == 2;
    if (!adapter_root) {
      const uint32_t name_index = i < 2 ? i : 2;
      status = iree_file_path_join(options->checkpoint_directory,
                                   iree_make_cstring_view(names[name_index]),
                                   allocator, &path);
    }
    if (iree_status_is_ok(status)) {
      const loom_cmd_program_parameter_root_t reflected =
          loom_cmd_program_parameter_root_at(program, i);
      const loom_serve_weight_root_t root = {
          program, reflected, &model->weights[reflected.fixed_buffer_index]};
      status = loom_serve_weights_load(
          device, loom_serve_device_transfer_queue(model->owner), dispatch,
          model->jit, command_mode, 0, 1, &root,
          adapter_root ? options->adapter_path : iree_make_cstring_view(path),
          iree_make_cstring_view(policy), allocator);
    }
    iree_allocator_free(allocator, path);
  }
  iree_allocator_free(allocator, policy);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
      model->stage, iree_hal_queue_family(dispatch), command_mode,
      model->weights, &model->command));
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    const bool workspace = i == model->input_count + 1;
    const iree_device_size_t length =
        i < model->input_count
            ? sizes[i + (!adapted && i >= LOOM_SERVE_KREA2_INPUT_STRENGTH)]
        : workspace ? program->requirements.transient.required_byte_length
                    : model->output.data_length;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment =
        workspace ? program->requirements.transient.minimum_alignment : 256;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, length,
        &model->bindings[i].buffer);
    model->bindings[i].length = length;
  }
  return status;
}

iree_status_t loom_serve_krea2_model_create(
    const loom_serve_krea2_model_options_t* options,
    loom_serve_krea2_model_t** out_model, iree_allocator_t host_allocator) {
  *out_model = NULL;
  iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT];
  IREE_RETURN_IF_ERROR(loom_serve_krea2_request_measure(
      options->height, options->width, options->text_tokens, sizes));
  loom_serve_krea2_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*model), (void**)&model));
  model->allocator = host_allocator;
  model->height = options->height;
  model->width = options->width;
  model->text_tokens = options->text_tokens;
  iree_status_t status = krea2_model_initialize(model, options, sizes);
  if (iree_status_is_ok(status)) {
    *out_model = model;
  } else {
    status = iree_status_join(status, loom_serve_krea2_model_destroy(model));
  }
  return status;
}

iree_status_t loom_serve_krea2_model_generate(loom_serve_krea2_model_t* model,
                                              iree_string_view_t prompt,
                                              uint64_t seed, float strength,
                                              iree_const_byte_span_t* out_rgb) {
  *out_rgb = iree_const_byte_span_empty();
  const bool adapted = model->input_count == LOOM_SERVE_KREA2_INPUT_COUNT;
  if (!isfinite(strength)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "adapter strength must be finite");
  }
  if (!adapted && strength != 1.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "strength requires an adapter");
  }
  const loom_serve_krea2_request_options_t options = {
      model->height, model->width, model->text_tokens, seed, strength};
  const iree_time_t prepare_begin = iree_time_now();
  fprintf(stderr, "{\"event\":\"image_preparing\"}\n");
  loom_serve_krea2_prompt_t* prepared = NULL;
  loom_serve_krea2_request_t* request = NULL;
  iree_status_t status =
      loom_serve_krea2_prompt_create(model->tokenizer, model->text_tokens,
                                     prompt, &prepared, model->allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_krea2_request_create(prepared, options, &request,
                                             model->allocator);
  }
  loom_serve_krea2_prompt_destroy(prepared);
  IREE_RETURN_IF_ERROR(status);
  const iree_time_t prepare_end = iree_time_now();
  fprintf(stderr, "{\"event\":\"image_prepared\",\"prepare_ns\":%" PRId64 "}\n",
          prepare_end - prepare_begin);
  iree_hal_transfer_operation_t uploads[LOOM_SERVE_KREA2_INPUT_COUNT] = {0};
  for (iree_host_size_t i = 0; i < model->input_count; ++i) {
    const loom_serve_krea2_input_t kind =
        (loom_serve_krea2_input_t)(i + (!adapted &&
                                        i >= LOOM_SERVE_KREA2_INPUT_STRENGTH));
    const iree_const_byte_span_t input =
        loom_serve_krea2_request_input(request, kind);
    uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i].upload.source = input.data;
    uploads[i].upload.target_buffer = model->bindings[i].buffer;
    uploads[i].upload.length = input.data_length;
  }
  loom_serve_execution_t* execution = loom_serve_device_execution(model->owner);
  uint64_t completion = 0;
  status = loom_serve_execution_transfer(execution, model->input_count, uploads,
                                         &completion);
  if (iree_status_is_ok(status)) {
    status = loom_serve_execution_execute(
        execution, model->command,
        (iree_hal_buffer_binding_table_t){model->input_count + 2,
                                          model->bindings},
        &completion);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = {.source_buffer =
                         model->bindings[model->input_count].buffer,
                     .target = model->output.data,
                     .length = model->output.data_length},
    };
    status =
        loom_serve_execution_feedback(execution, 1, &download, &completion);
  }
  const iree_time_t submit_end = iree_time_now();
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "{\"event\":\"image_submitted\",\"submit_ns\":%" PRId64 "}\n",
            submit_end - prepare_end);
    status = loom_serve_execution_feedback_wait(execution, completion);
  }
  const iree_time_t completion_end = iree_time_now();
  if (iree_status_is_ok(status)) {
    *out_rgb = iree_make_const_byte_span(model->output.data,
                                         model->output.data_length);
  } else {
    status = iree_status_join(status, loom_serve_execution_drain(execution));
  }
  loom_serve_krea2_request_destroy(request);
  if (iree_status_is_ok(status)) {
    // Submission overlaps device work. The remaining wait includes queued
    // transfers and final readback, not an isolated GPU execution interval.
    fprintf(stderr,
            "{\"event\":\"image_execution\",\"prepare_ns\":%" PRId64
            ",\"submit_ns\":%" PRId64 ",\"completion_wait_ns\":%" PRId64 "}\n",
            prepare_end - prepare_begin, submit_end - prepare_end,
            completion_end - submit_end);
  }
  return status;
}
