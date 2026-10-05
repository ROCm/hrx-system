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

typedef struct krea2_model_stage_t {
  // Retained text extent specialized into this command and its kernels.
  uint32_t text_tokens;
  // Compiled image root, executable ownership and reflection.
  loom_serve_jit_stage_t* compiled;
  // Reusable command retaining the model's immutable parameter domains.
  iree_hal_command_buffer_t* command;
} krea2_model_stage_t;

struct loom_serve_krea2_model_t {
  // Allocator owning this model and its host feedback storage.
  iree_allocator_t allocator;
  // Fixed pixel height for the prepared command.
  uint32_t height;
  // Fixed pixel width for the prepared command.
  uint32_t width;
  // Immutable tokenizer reused by each independent request encoder.
  iree_tokenizer_t* tokenizer;
  // Shared device/timeline ownership, outliving all accepted work.
  loom_serve_device_t* owner;
  // Optional profiling session, ended after accepted work drains.
  iree_hal_profiling_from_flags_t* profiling;
  // Cold live-source compiler and its task pool.
  loom_serve_jit_t* jit;
  // Cold retained shapes sharing one parameter and issue-time buffer bank.
  struct {
    // Number of owned slots, including partially initialized slots on failure.
    uint32_t count;
    // Slot zero is the configured maximum; optional slot one is text128.
    krea2_model_stage_t values[2];
  } stages;
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
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    iree_hal_command_buffer_release(model->stages.values[i].command);
    loom_serve_jit_stage_destroy(model->stages.values[i].compiled);
  }
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

static iree_status_t krea2_compile_stage(
    loom_serve_krea2_model_t* model,
    const loom_serve_krea2_model_options_t* options,
    krea2_model_stage_t* stage) {
  const uint32_t images = (options->height / 16) * (options->width / 16);
  const char* keys[] = {"krea2.block_tokens",  "krea2.image_tokens",
                        "krea2.text_tokens",   "krea2.time_count",
                        "krea2.latent_height", "krea2.latent_width"};
  const uint32_t values[] = {images + stage->text_tokens, images,
                             stage->text_tokens,          8,
                             options->height / 8,         options->width / 8};
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
  return loom_serve_jit_compile(model->jit,
                                options->adapter_path.size
                                    ? IREE_SV("sample_image_adapted")
                                    : IREE_SV("sample_image"),
                                &config, &stage->compiled);
}

// The source catalog is external. Every retained shape must describe the same
// immutable placement before a single checkpoint byte is loaded or shared.
static iree_status_t krea2_check_layouts(loom_serve_krea2_model_t* model) {
  const loom_cmd_program_t* canonical =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const bool adapted = model->input_count == LOOM_SERVE_KREA2_INPUT_COUNT;
  const iree_host_size_t binding_count = model->input_count + 2;
  const uint32_t roots = adapted ? 4 : 3;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const krea2_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    if (program->requirements.rebindable_binding_count != binding_count ||
        program->requirements.transient.binding_index !=
            model->input_count + 1 ||
        program->requirements.fixed_buffer_count != roots ||
        program->parameter_roots.count != roots ||
        program->requirements.launch_counts.binding_index != UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "text%u source root does not implement the Krea image ABI",
          stage->text_tokens);
    }
    if (!i) {
      continue;
    }
    if (canonical->parameters.count != program->parameters.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "text%u parameter count differs from maximum",
                              stage->text_tokens);
    }
    for (uint32_t root = 0; root < roots; ++root) {
      const loom_cmd_program_parameter_root_t lhs =
          loom_cmd_program_parameter_root_at(canonical, root);
      const loom_cmd_program_parameter_root_t rhs =
          loom_cmd_program_parameter_root_at(program, root);
      if (lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
          lhs.required_byte_length != rhs.required_byte_length ||
          lhs.minimum_alignment != rhs.minimum_alignment) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "text%u parameter root %u differs from maximum",
                                stage->text_tokens, root);
      }
    }
    for (uint32_t parameter = 0; parameter < canonical->parameters.count;
         ++parameter) {
      const loom_cmd_program_parameter_t lhs =
          loom_cmd_program_parameter_at(canonical, parameter);
      const loom_cmd_program_parameter_t rhs =
          loom_cmd_program_parameter_at(program, parameter);
      if (!iree_string_view_equal(lhs.key, rhs.key) ||
          lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
          lhs.byte_offset != rhs.byte_offset ||
          lhs.byte_length != rhs.byte_length ||
          lhs.minimum_alignment != rhs.minimum_alignment) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "text%u parameter placement %u differs from maximum",
            stage->text_tokens, parameter);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t krea2_model_initialize(
    loom_serve_krea2_model_t* model,
    const loom_serve_krea2_model_options_t* options,
    iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT]) {
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
  uint64_t workspace_length = 0;
  uint64_t workspace_alignment = 0;
  uint64_t executable_count = 0;
  uint64_t entry_count = 0;
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    krea2_model_stage_t* stage = &model->stages.values[i];
    IREE_RETURN_IF_ERROR(krea2_compile_stage(model, options, stage));
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(stage->compiled);
    workspace_length = iree_max(
        workspace_length, program->requirements.transient.required_byte_length);
    workspace_alignment = iree_max(
        workspace_alignment, program->requirements.transient.minimum_alignment);
    executable_count += program->requirements.executable_count;
    entry_count += program->requirements.entry_count;
    if (i) {
      iree_host_size_t stage_sizes[LOOM_SERVE_KREA2_INPUT_COUNT];
      IREE_RETURN_IF_ERROR(loom_serve_krea2_request_measure(
          options->height, options->width, stage->text_tokens, stage_sizes));
      for (iree_host_size_t input = 0; input < IREE_ARRAYSIZE(stage_sizes);
           ++input) {
        sizes[input] = iree_max(sizes[input], stage_sizes[input]);
      }
    }
  }
  IREE_RETURN_IF_ERROR(krea2_check_layouts(model));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(model->stages.values[0].compiled);
  const uint32_t roots = adapted ? 4 : 3;
  const iree_host_size_t binding_count = model->input_count + 2;
  uint64_t parameter_bytes = 0;
  for (uint32_t i = 0; i < roots; ++i) {
    parameter_bytes +=
        loom_cmd_program_parameter_root_at(program, i).required_byte_length;
  }
  uint64_t input_bytes = 0;
  for (iree_host_size_t i = 0; i < model->input_count; ++i) {
    input_bytes +=
        sizes[i + (!adapted && i >= LOOM_SERVE_KREA2_INPUT_STRENGTH)];
  }
  printf("{\"event\":\"image_residency\",\"parameter_bytes\":%" PRIu64
         ",\"input_bytes\":%" PRIu64
         ",\"output_bytes\":%zu"
         ",\"workspace_bytes\":%" PRIu64 ",\"workspace_alignment\":%" PRIu64
         ",\"kernels\":%" PRIu64 ",\"entries\":%" PRIu64 ",\"stages\":[",
         parameter_bytes, input_bytes, model->output.data_length,
         workspace_length, workspace_alignment, executable_count, entry_count);
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    const krea2_model_stage_t* stage = &model->stages.values[i];
    const loom_cmd_program_t* reflected =
        loom_serve_jit_stage_program(stage->compiled);
    printf("%s{\"text_tokens\":%u,\"workspace_bytes\":%" PRIu64
           ",\"workspace_alignment\":%" PRIu64
           ",\"kernels\":%u,\"entries\":%u}",
           i ? "," : "", stage->text_tokens,
           reflected->requirements.transient.required_byte_length,
           reflected->requirements.transient.minimum_alignment,
           reflected->requirements.executable_count,
           reflected->requirements.entry_count);
  }
  printf("]}\n");
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
  for (uint32_t i = 0; i < model->stages.count; ++i) {
    krea2_model_stage_t* stage = &model->stages.values[i];
    IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
        stage->compiled, iree_hal_queue_family(dispatch), command_mode,
        model->weights, &stage->command));
  }
  for (iree_host_size_t i = 0; i < binding_count && iree_status_is_ok(status);
       ++i) {
    const bool workspace = i == model->input_count + 1;
    const iree_device_size_t length =
        i < model->input_count
            ? sizes[i + (!adapted && i >= LOOM_SERVE_KREA2_INPUT_STRENGTH)]
        : workspace ? workspace_length
                    : model->output.data_length;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment = workspace ? workspace_alignment : 256;
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
  model->stages.count = 1;
  model->stages.values[0].text_tokens = options->text_tokens;
  if (options->text_tokens > 128 && options->text_tokens % 64 == 0) {
    model->stages.count = 2;
    model->stages.values[1].text_tokens = 128;
  }
  model->input_count =
      LOOM_SERVE_KREA2_INPUT_COUNT - !options->adapter_path.size;
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
  const iree_time_t prepare_begin = iree_time_now();
  fprintf(stderr, "{\"event\":\"image_preparing\"}\n");
  loom_serve_krea2_prompt_t* prepared = NULL;
  loom_serve_krea2_request_t* request = NULL;
  const krea2_model_stage_t* stage = &model->stages.values[0];
  uint32_t token_count = 0;
  iree_status_t status =
      loom_serve_krea2_prompt_create(model->tokenizer, stage->text_tokens,
                                     prompt, &prepared, model->allocator);
  if (iree_status_is_ok(status)) {
    token_count = loom_serve_krea2_prompt_token_count(prepared);
    // This bound preserves the prompt/suffix live-key partitions in text128.
    if (model->stages.count == 2 && token_count <= 98) {
      stage = &model->stages.values[1];
    }
    const loom_serve_krea2_request_options_t options = {
        model->height, model->width, stage->text_tokens, seed, strength};
    status = loom_serve_krea2_request_create(prepared, options, &request,
                                             model->allocator);
  }
  loom_serve_krea2_prompt_destroy(prepared);
  IREE_RETURN_IF_ERROR(status);
  const iree_time_t prepare_end = iree_time_now();
  fprintf(stderr,
          "{\"event\":\"image_prepared\",\"prepare_ns\":%" PRId64
          ",\"text_tokens\":%u,\"prefix_prompt_tokens\":%u}\n",
          prepare_end - prepare_begin, stage->text_tokens, token_count);
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
        execution, stage->command,
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
