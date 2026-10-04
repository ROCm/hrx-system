// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native prompt-to-image CLI. All model arithmetic and stage control are source
// commands; this cold caller owns request I/O, residency and final RGB
// feedback.

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/device.h"
#include "experimental/loom_serve/models/krea2/request.h"
#include "experimental/loom_serve/weights.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

IREE_FLAG(string, model, "experimental/loom_serve/models/krea2",
          "Live source catalog directory.");
IREE_FLAG(string, checkpoint, "",
          "Official Krea 2 Turbo checkpoint directory.");
IREE_FLAG(string, adapter, "", "Optional softwatercolor safetensors file.");
IREE_FLAG(string, prompt, "", "Image prompt; the model supplies its template.");
IREE_FLAG(string, seed, "0", "Unsigned decimal 64-bit native noise seed.");
IREE_FLAG(int32_t, height, 384, "Output pixel height, divisible by 16.");
IREE_FLAG(int32_t, width, 384, "Output pixel width, divisible by 16.");
IREE_FLAG(int32_t, text_tokens, 512, "Retained text extent, divisible by 16.");
IREE_FLAG(float, strength, 1.0f, "Adapter strength; zero is base identity.");
IREE_FLAG(string, output, "", "Output PPM image path.");

typedef struct krea2_generator_t {
  // Shared device/timeline ownership, outliving all accepted work.
  loom_serve_device_t* owner;
  // Cold live-source compiler and its task pool.
  loom_serve_jit_t* jit;
  // Compiled image root and reflection.
  loom_serve_jit_stage_t* stage;
  // Reusable command, retaining immutable parameter domains.
  iree_hal_command_buffer_t* command;
  // Encoder, Turbo, optional adapter and VAE fixed buffers.
  iree_hal_buffer_t* weights[4];
  // Inputs, final RGB and one reflected workspace; partial slots are NULL.
  iree_hal_buffer_binding_t bindings[LOOM_SERVE_KREA2_INPUT_COUNT + 2];
  // Number of input slots, excluding output and workspace.
  iree_host_size_t input_count;
  // Number of slots including output and workspace.
  iree_host_size_t binding_count;
  // Upload descriptors borrowing request bytes until execution drains.
  iree_hal_transfer_operation_t uploads[LOOM_SERVE_KREA2_INPUT_COUNT];
  // Immutable request slab retained through all accepted uploads.
  loom_serve_krea2_request_t* request;
  // Completed NCHW F32 RGB feedback, alive through execution drain.
  uint8_t* output;
  // Length in bytes of final RGB feedback.
  iree_host_size_t output_bytes;
} krea2_generator_t;

static iree_status_t krea2_prepare_request(
    krea2_generator_t* generator, loom_serve_krea2_request_options_t options,
    iree_allocator_t allocator) {
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      iree_make_cstring_view(FLAG_checkpoint),
      IREE_SV("tokenizer/tokenizer.json"), allocator, &path));
  iree_io_file_contents_t* contents = NULL;
  iree_tokenizer_t* tokenizer = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(path), allocator, &contents);
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_krea2_request_create(
        tokenizer, options, iree_make_cstring_view(FLAG_prompt), allocator,
        &generator->request);
  }
  iree_tokenizer_free(tokenizer);
  iree_io_file_contents_free(contents);
  iree_allocator_free(allocator, path);
  return status;
}

static iree_status_t krea2_initialize(krea2_generator_t* generator,
                                      iree_allocator_t allocator) {
  if (!FLAG_output[0] || !FLAG_checkpoint[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--checkpoint and --output are required");
  }
  const bool adapted = FLAG_adapter[0] != 0;
  if (!adapted && FLAG_strength != 1.0f) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--strength requires --adapter");
  }
  loom_serve_krea2_request_options_t options = {
      (uint32_t)FLAG_height, (uint32_t)FLAG_width, (uint32_t)FLAG_text_tokens,
      0, FLAG_strength};
  IREE_RETURN_IF_ERROR(
      iree_json_parse_uint64(iree_make_cstring_view(FLAG_seed), &options.seed));
  IREE_RETURN_IF_ERROR(krea2_prepare_request(generator, options, allocator));
  generator->output_bytes =
      (iree_host_size_t)options.height * options.width * 12;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, generator->output_bytes,
                                             (void**)&generator->output));
  IREE_RETURN_IF_ERROR(loom_serve_device_create(IREE_SV("amdgpu"), allocator,
                                                &generator->owner));
  iree_hal_device_t* device = loom_serve_device_handle(generator->owner);
  iree_hal_queue_t* dispatch =
      loom_serve_device_dispatch_queue(generator->owner);
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(device, dispatch,
                                             iree_make_cstring_view(FLAG_model),
                                             NULL, allocator, &generator->jit));
  const uint32_t images = (options.height / 16) * (options.width / 16);
  const char* keys[] = {"krea2.block_tokens",  "krea2.image_tokens",
                        "krea2.text_tokens",   "krea2.time_count",
                        "krea2.latent_height", "krea2.latent_width"};
  const uint32_t values[] = {images + options.text_tokens, images,
                             options.text_tokens,          8,
                             options.height / 8,           options.width / 8};
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
      generator->jit,
      adapted ? IREE_SV("sample_image_adapted") : IREE_SV("sample_image"),
      &config, &generator->stage));
  const loom_cmd_program_t* program =
      loom_serve_jit_stage_program(generator->stage);
  generator->input_count = LOOM_SERVE_KREA2_INPUT_COUNT - !adapted;
  generator->binding_count = generator->input_count + 2;
  const uint32_t roots = adapted ? 4 : 3;
  if (program->requirements.rebindable_binding_count !=
          generator->binding_count ||
      program->requirements.transient.binding_index !=
          generator->input_count + 1 ||
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
  char* policy = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(iree_make_cstring_view(FLAG_model),
                                           IREE_SV("weights.loom"), allocator,
                                           &policy));
  const char* names[] = {"text_encoder/model.safetensors", "turbo.safetensors",
                         "vae/diffusion_pytorch_model.safetensors"};
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < roots && iree_status_is_ok(status); ++i) {
    char* path = NULL;
    const bool adapter_root = adapted && i == 2;
    if (!adapter_root) {
      const uint32_t name_index = i < 2 ? i : 2;
      status = iree_file_path_join(iree_make_cstring_view(FLAG_checkpoint),
                                   iree_make_cstring_view(names[name_index]),
                                   allocator, &path);
    }
    if (iree_status_is_ok(status)) {
      const loom_cmd_program_parameter_root_t reflected =
          loom_cmd_program_parameter_root_at(program, i);
      const loom_serve_weight_root_t root = {
          program, reflected,
          &generator->weights[reflected.fixed_buffer_index]};
      status = loom_serve_weights_load(
          device, loom_serve_device_transfer_queue(generator->owner), dispatch,
          generator->jit, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, 0, 1, &root,
          iree_make_cstring_view(adapter_root ? FLAG_adapter : path),
          iree_make_cstring_view(policy), allocator);
    }
    iree_allocator_free(allocator, path);
  }
  iree_allocator_free(allocator, policy);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_jit_stage_record(
      generator->stage, iree_hal_queue_family(dispatch),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, generator->weights,
      &generator->command));
  for (iree_host_size_t i = 0;
       i < generator->binding_count && iree_status_is_ok(status); ++i) {
    const bool workspace = i == generator->input_count + 1;
    iree_const_byte_span_t input = iree_const_byte_span_empty();
    if (i < generator->input_count) {
      const loom_serve_krea2_input_t kind =
          (loom_serve_krea2_input_t)(i +
                                     (!adapted &&
                                      i >= LOOM_SERVE_KREA2_INPUT_STRENGTH));
      input = loom_serve_krea2_request_input(generator->request, kind);
    }
    const iree_device_size_t length =
        i < generator->input_count ? input.data_length
        : workspace ? program->requirements.transient.required_byte_length
                    : generator->output_bytes;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
    params.min_alignment =
        workspace ? program->requirements.transient.minimum_alignment : 256;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, length,
        &generator->bindings[i].buffer);
    generator->bindings[i].length = length;
    if (i < generator->input_count) {
      generator->uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      generator->uploads[i].upload.source = input.data;
      generator->uploads[i].upload.target_buffer =
          generator->bindings[i].buffer;
      generator->uploads[i].upload.length = length;
    }
  }
  return status;
}

static iree_status_t krea2_generate(krea2_generator_t* generator) {
  loom_serve_execution_t* execution =
      loom_serve_device_execution(generator->owner);
  uint64_t completion = 0;
  IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(
      execution, generator->input_count, generator->uploads, &completion));
  IREE_RETURN_IF_ERROR(loom_serve_execution_execute(
      execution, generator->command,
      (iree_hal_buffer_binding_table_t){generator->binding_count,
                                        generator->bindings},
      &completion));
  iree_hal_transfer_operation_t download = {0};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download.source_buffer =
      generator->bindings[generator->input_count].buffer;
  download.download.target = generator->output;
  download.download.length = generator->output_bytes;
  IREE_RETURN_IF_ERROR(
      loom_serve_execution_feedback(execution, 1, &download, &completion));
  return loom_serve_execution_feedback_wait(execution, completion);
}

static iree_status_t krea2_write_image(krea2_generator_t* generator,
                                       iree_allocator_t allocator) {
  char header[64];
  const iree_host_size_t header_length = (iree_host_size_t)snprintf(
      header, sizeof(header), "P6\n%d %d\n255\n", FLAG_width, FLAG_height);
  const iree_host_size_t pixels = generator->output_bytes / 12;
  uint8_t* image = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, header_length + pixels * 3, (void**)&image));
  memcpy(image, header, header_length);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < pixels * 3 && iree_status_is_ok(status);
       ++i) {
    const float value = iree_unaligned_load_le_f32(generator->output + i * 4);
    if (!isfinite(value) || value < -1.0f || value > 1.0f) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "nonfinite or out-of-range final RGB at %zu", i);
    } else {
      const iree_host_size_t channel = i / pixels;
      const iree_host_size_t pixel = i % pixels;
      image[header_length + pixel * 3 + channel] =
          (uint8_t)nearbyintf((value / 2.0f + 0.5f) * 255.0f);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_output),
        iree_make_const_byte_span(image, header_length + pixels * 3),
        allocator);
  }
  iree_allocator_free(allocator, image);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_allocator_t allocator = iree_allocator_system();
  krea2_generator_t generator = {0};
  iree_status_t status = krea2_initialize(&generator, allocator);
  if (iree_status_is_ok(status)) {
    status = krea2_generate(&generator);
  }
  if (iree_status_is_ok(status)) {
    status = krea2_write_image(&generator, allocator);
  }
  if (generator.owner) {
    status = iree_status_join(
        status, loom_serve_execution_drain(
                    loom_serve_device_execution(generator.owner)));
  }
  iree_hal_command_buffer_release(generator.command);
  loom_serve_jit_stage_destroy(generator.stage);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(generator.weights); ++i) {
    iree_hal_buffer_release(generator.weights[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(generator.bindings); ++i) {
    iree_hal_buffer_release(generator.bindings[i].buffer);
  }
  loom_serve_jit_destroy(generator.jit);
  loom_serve_krea2_request_destroy(generator.request);
  iree_allocator_free(allocator, generator.output);
  loom_serve_device_destroy(generator.owner);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  printf("Wrote %s\n", FLAG_output);
  return EXIT_SUCCESS;
}
