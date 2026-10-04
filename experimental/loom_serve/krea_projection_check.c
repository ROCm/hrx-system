// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-checkpoint component qualification, not an image-generation server.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/device.h"
#include "experimental/loom_serve/weights.h"
#include "iree/base/internal/math.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, model, "experimental/loom_serve/models/krea2",
          "Krea source catalog directory.");
IREE_FLAG(string, weights, "", "Original Turbo safetensors checkpoint.");
IREE_FLAG(string, adapter, "", "Official softwatercolor safetensors adapter.");
IREE_FLAG(string, input, "", "Packed latent input as little-endian BF16.");
IREE_FLAG(string, expected_base, "", "Reference base projection as BF16.");
IREE_FLAG(string, expected_adapter, "",
          "Reference strength-one projection as BF16.");
IREE_FLAG(int32_t, rows, 576, "Image token count, a positive multiple of 16.");

typedef struct projection_check_t {
  // Shared device and queues, destroyed after all model and host I/O storage.
  loom_serve_device_t* owner;
  // Compiler used only during cold setup.
  loom_serve_jit_t* jit;
  // Independently compiled base and adapter command roots.
  loom_serve_jit_stage_t* stages[2];
  // Separate immutable checkpoint roots, one per compiled stage.
  iree_hal_buffer_t* weights[2];
  // Reusable recorded base and adapter commands.
  iree_hal_command_buffer_t* commands[2];
  // Input, output, strength, and command-planned workspace allocations.
  iree_hal_buffer_t* buffers[4];
  // Actual input plus independent base and adapter reference file bytes.
  iree_io_file_contents_t* files[3];
  // Completed outputs retained for numerical comparison and zero identity.
  uint16_t* output;
  // First base result retained across adapter use.
  uint16_t* baseline;
  // Upload payload kept alive until accepted transfer completion.
  float strength;
  // Bytes in one packed input matrix.
  iree_host_size_t input_bytes;
  // Bytes in one output matrix.
  iree_host_size_t output_bytes;
} projection_check_t;

static iree_status_t projection_allocate(projection_check_t* check,
                                         iree_device_size_t length,
                                         iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  params.min_alignment = 256;
  return iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(loom_serve_device_handle(check->owner)), params,
      length, out_buffer);
}

static iree_status_t projection_initialize(projection_check_t* check,
                                           iree_allocator_t allocator) {
  if (FLAG_rows < 16 || FLAG_rows > 65536 || FLAG_rows % 16 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rows must be a multiple of 16 in [16, 65536]");
  }
  check->input_bytes = (iree_host_size_t)FLAG_rows * 64 * sizeof(uint16_t);
  check->output_bytes = (iree_host_size_t)FLAG_rows * 6144 * sizeof(uint16_t);
  const char* paths[] = {FLAG_input, FLAG_expected_base, FLAG_expected_adapter};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(paths) && iree_status_is_ok(status); ++i) {
    status = iree_io_file_contents_read(iree_make_cstring_view(paths[i]),
                                        allocator, &check->files[i]);
    const iree_host_size_t expected =
        i ? check->output_bytes : check->input_bytes;
    if (iree_status_is_ok(status) &&
        check->files[i]->const_buffer.data_length != expected) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "file '%s' must contain exactly %zu bytes",
                                paths[i], expected);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, check->output_bytes,
                                             (void**)&check->output));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, check->output_bytes,
                                             (void**)&check->baseline));
  IREE_RETURN_IF_ERROR(
      loom_serve_device_create(IREE_SV("amdgpu"), allocator, &check->owner));
  iree_hal_device_t* device = loom_serve_device_handle(check->owner);
  iree_hal_queue_t* dispatch = loom_serve_device_dispatch_queue(check->owner);
  IREE_RETURN_IF_ERROR(loom_serve_jit_create(device, dispatch,
                                             iree_make_cstring_view(FLAG_model),
                                             NULL, allocator, &check->jit));
  char rows[32];
  snprintf(rows, sizeof(rows), "%d", FLAG_rows);
  const loomc_config_binding_t binding = {
      loomc_make_cstring_view("krea2.image_tokens"),
      loomc_make_cstring_view(rows)};
  const loomc_config_options_t config = {
      &binding, 1, {0}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
  const iree_string_view_t roots[] = {IREE_SVL("image_projection"),
                                      IREE_SVL("image_projection_adapter")};
  const char* checkpoints[] = {FLAG_weights, FLAG_adapter};
  char* policy_path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(iree_make_cstring_view(FLAG_model),
                                           IREE_SV("weights.loom"), allocator,
                                           &policy_path));
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(roots) && iree_status_is_ok(status); ++i) {
    status = loom_serve_jit_compile(check->jit, roots[i], &config,
                                    &check->stages[i]);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    const loom_cmd_program_t* program =
        loom_serve_jit_stage_program(check->stages[i]);
    if (program->requirements.fixed_buffer_count != 1 ||
        program->requirements.rebindable_binding_count != (i ? 4 : 2) ||
        (i && program->requirements.transient.binding_index != 3)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "projection source has incompatible bindings");
      continue;
    }
    const loom_serve_weight_stage_t stage = {program, &check->weights[i]};
    status = loom_serve_weights_load(
        device, loom_serve_device_transfer_queue(check->owner), dispatch,
        check->jit, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, 0, 1, &stage,
        iree_make_cstring_view(checkpoints[i]),
        iree_make_cstring_view(policy_path), allocator);
    if (iree_status_is_ok(status)) {
      status = loom_serve_jit_stage_record(
          check->stages[i], iree_hal_queue_family(dispatch),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, &check->weights[i],
          &check->commands[i]);
    }
  }
  iree_allocator_free(allocator, policy_path);
  IREE_RETURN_IF_ERROR(status);
  const iree_device_size_t lengths[] = {
      check->input_bytes, check->output_bytes, sizeof(check->strength),
      loom_serve_jit_stage_program(check->stages[1])
          ->requirements.transient.required_byte_length};
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(lengths) && iree_status_is_ok(status); ++i) {
    status = projection_allocate(check, lengths[i], &check->buffers[i]);
  }
  return status;
}

static iree_status_t projection_compare(projection_check_t* check, int phase) {
  const char* names[] = {"base", "zero_adapter", "adapter",
                         "base_after_adapter"};
  const uint16_t* expected =
      (const uint16_t*)check->files[phase == 2 ? 2 : 1]->const_buffer.data;
  const iree_host_size_t count = check->output_bytes / sizeof(uint16_t);
  iree_host_size_t different = 0;
  iree_host_size_t outside_tolerance = 0;
  double squared_error = 0;
  double squared_expected = 0;
  float maximum = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const float actual_value = iree_math_bf16_to_f32(check->output[i]);
    const float expected_value = iree_math_bf16_to_f32(expected[i]);
    const float error = fabsf(actual_value - expected_value);
    different += check->output[i] != expected[i];
    // One BF16 relative rounding interval plus a small absolute floor near
    // cancellation. The complete tensor is checked, including finite values.
    outside_tolerance +=
        !isfinite(actual_value) || !isfinite(expected_value) ||
        error > 0.0001220703125f + 0.0078125f * fabsf(expected_value);
    maximum = fmaxf(maximum, error);
    squared_error += (double)error * error;
    squared_expected += (double)expected_value * expected_value;
  }
  printf(
      "{\"phase\":\"%s\",\"elements\":%zu,\"different\":%zu,"
      "\"outside_tolerance\":%zu,\"maximum_absolute_error\":%.9g,"
      "\"relative_l2\":%.9g}\n",
      names[phase], count, different, outside_tolerance, maximum,
      sqrt(squared_error / fmax(squared_expected, 1e-30)));
  if (outside_tolerance) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "%zu projection elements exceed tolerance",
                            outside_tolerance);
  }
  if (!phase) {
    memcpy(check->baseline, check->output, check->output_bytes);
  } else if ((phase == 1 || phase == 3) &&
             memcmp(check->baseline, check->output, check->output_bytes)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "adapter use changed the base projection");
  }
  return iree_ok_status();
}

static iree_status_t projection_run(projection_check_t* check) {
  loom_serve_execution_t* execution = loom_serve_device_execution(check->owner);
  iree_status_t status = iree_ok_status();
  for (int phase = 0; phase < 4 && iree_status_is_ok(status); ++phase) {
    check->strength = phase == 2 ? 1.0f : 0.0f;
    iree_hal_transfer_operation_t uploads[2] = {0};
    uploads[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[0].upload.source = check->files[0]->const_buffer.data;
    uploads[0].upload.target_buffer = check->buffers[0];
    uploads[0].upload.length = check->input_bytes;
    uploads[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[1].upload.source = &check->strength;
    uploads[1].upload.target_buffer = check->buffers[2];
    uploads[1].upload.length = sizeof(check->strength);
    uint64_t completion = 0;
    status = loom_serve_execution_transfer(execution, 2, uploads, &completion);
    const iree_hal_buffer_binding_t base_bindings[] = {
        {check->buffers[0], 0, check->input_bytes},
        {check->buffers[1], 0, check->output_bytes}};
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_execute(
          execution, check->commands[0],
          (iree_hal_buffer_binding_table_t){2, base_bindings}, &completion);
    }
    const iree_hal_buffer_binding_t adapter_bindings[] = {
        base_bindings[0],
        {check->buffers[2], 0, sizeof(check->strength)},
        base_bindings[1],
        {check->buffers[3], 0, iree_hal_buffer_byte_length(check->buffers[3])}};
    if (iree_status_is_ok(status) && (phase == 1 || phase == 2)) {
      status = loom_serve_execution_execute(
          execution, check->commands[1],
          (iree_hal_buffer_binding_table_t){4, adapter_bindings}, &completion);
    }
    iree_hal_transfer_operation_t download = {0};
    download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    download.download.source_buffer = check->buffers[1];
    download.download.target = check->output;
    download.download.length = check->output_bytes;
    if (iree_status_is_ok(status)) {
      status =
          loom_serve_execution_feedback(execution, 1, &download, &completion);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback_wait(execution, completion);
    }
    if (iree_status_is_ok(status)) {
      status = projection_compare(check, phase);
    }
  }
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_allocator_t allocator = iree_allocator_system();
  projection_check_t check = {0};
  iree_status_t status = projection_initialize(&check, allocator);
  if (iree_status_is_ok(status)) {
    status = projection_run(&check);
  }
  if (check.owner) {
    status = iree_status_join(
        status,
        loom_serve_execution_drain(loom_serve_device_execution(check.owner)));
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(check.commands); ++i) {
    iree_hal_command_buffer_release(check.commands[i]);
    loom_serve_jit_stage_destroy(check.stages[i]);
    iree_hal_buffer_release(check.weights[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(check.buffers); ++i) {
    iree_hal_buffer_release(check.buffers[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(check.files); ++i) {
    iree_io_file_contents_free(check.files[i]);
  }
  iree_allocator_free(allocator, check.output);
  iree_allocator_free(allocator, check.baseline);
  loom_serve_jit_destroy(check.jit);
  loom_serve_device_destroy(check.owner);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  fprintf(stderr,
          "PASS: base, zero/nonzero adapter, and retained base identity.\n");
  return EXIT_SUCCESS;
}
