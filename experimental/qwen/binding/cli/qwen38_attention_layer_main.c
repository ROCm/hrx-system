// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/qwen/programs/qwen38_attention_layer_decode_source.h"
#include "experimental/qwen/tooling/command_program.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

#define QWEN38_HIDDEN_ELEMENT_COUNT 5120
#define QWEN38_HIDDEN_BYTE_LENGTH 20480
#define QWEN38_ATTENTION_ELEMENT_COUNT 6144
#define QWEN38_ATTENTION_BYTE_LENGTH 24576
#define QWEN38_ATTENTION_CACHE_CAPACITY 128
#define QWEN38_ATTENTION_CACHE_ELEMENT_COUNT 262144
#define QWEN38_ATTENTION_CACHE_BYTE_LENGTH 524288

IREE_FLAG(string, output, "", "Optional raw F32 residual output path.");
IREE_FLAG(string, cache_output, "", "Optional raw F16 K/V cache output path.");
IREE_FLAG(string, attention_output, "",
          "Optional raw F32 gated-attention output path.");
IREE_FLAG(int32_t, transition_count, 1,
          "Number of consecutive layer transitions to execute.");

static const char* const qwen38_attention_layer_usage =
    "Runs the exact first Qwen3.8 full-attention layer through a reusable "
    "Loom command program.\n"
    "\n"
    "Required flags:\n"
    "  --device=<AMDGPU device URI>\n"
    "  --parameters=<Qwen3.8-27B UD-Q5_K_XL GGUF path>\n"
    "\n"
    "Optional output:\n"
    "  --output=<raw F32 residual path>\n"
    "  --cache_output=<raw F16 token-major K/V cache path>\n"
    "  --attention_output=<raw F32 gated-attention output path>\n"
    "  --transition_count=<positive replay count up to 128>\n";

static iree_status_t qwen38_allocate_buffer(
    iree_hal_device_t* device, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t usage, iree_device_size_t minimum_alignment,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = minimum_alignment,
  };
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t qwen38_prepare_layer_program(
    qwen_tooling_runtime_context_t* runtime_context,
    iree_string_view_t root_name, iree_allocator_t host_allocator,
    qwen_tooling_command_program_set_t** out_program_set) {
  if (qwen38_attention_layer_decode_source_size() != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen3.8 attention layer source must contain exactly one file");
  }
  const iree_file_toc_t* files = qwen38_attention_layer_decode_source_create();
  const iree_string_view_t root_names[] = {root_name};
  const qwen_tooling_command_program_set_options_t options = {
      .source_identifier = iree_make_cstring_view(files[0].name),
      .source_contents =
          iree_make_const_byte_span(files[0].data, files[0].size),
      .root_names = root_names,
      .root_count = IREE_ARRAYSIZE(root_names),
  };
  return qwen_tooling_command_program_set_create(
      runtime_context, &options, host_allocator, out_program_set);
}

static iree_status_t qwen38_fill_and_wait(iree_hal_device_t* device,
                                          iree_hal_buffer_t* buffer) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  const uint32_t zero = 0;
  iree_status_t status = iree_hal_device_queue_fill(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signals, buffer, 0, iree_hal_buffer_byte_length(buffer), &zero,
      sizeof(zero), IREE_HAL_FILL_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

static iree_status_t qwen38_submit_and_wait(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t signal_value = 1;
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signals, command_buffer, binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
}

static iree_status_t qwen38_attention_layer_run(void) {
  const iree_allocator_t host_allocator = iree_allocator_system();
  qwen_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  qwen_tooling_command_program_set_t* program_set = NULL;
  qwen_tooling_command_program_t* program = NULL;
  iree_string_view_t root_name = iree_string_view_empty();
  const loomc_cmd_program_info_t* program_info = NULL;
  iree_hal_buffer_t* residual_buffer = NULL;
  iree_hal_buffer_t* control_buffer = NULL;
  iree_hal_buffer_t* cache_buffer = NULL;
  iree_hal_buffer_t* attention_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  float* residual_values = NULL;
  float* attention_values = NULL;
  uint16_t* cache_values = NULL;
  iree_hal_profiling_from_flags_t* profiling = NULL;

  iree_status_t status = iree_ok_status();
  if (FLAG_transition_count < 1 ||
      FLAG_transition_count > QWEN38_ATTENTION_CACHE_CAPACITY) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--transition_count must be between 1 and %d",
                              QWEN38_ATTENTION_CACHE_CAPACITY);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: indexed GGUF; compiling and gathering layer 3...\n");
    root_name = FLAG_attention_output[0] == '\0'
                    ? IREE_SV("qwen38_attention_layer3_decode")
                    : IREE_SV("qwen38_attention_layer3_decode_witness");
    status = qwen38_prepare_layer_program(&runtime_context, root_name,
                                          host_allocator, &program_set);
  }
  if (iree_status_is_ok(status)) {
    program = qwen_tooling_command_program_set_lookup(program_set, root_name);
  }
  if (iree_status_is_ok(status)) {
    program_info = qwen_tooling_command_program_info(program);
    const iree_host_size_t expected_binding_count =
        FLAG_attention_output[0] == '\0' ? 4 : 5;
    const uint32_t expected_transient_binding =
        FLAG_attention_output[0] == '\0' ? 3 : 4;
    if (program_info->rebindable_binding_count != expected_binding_count ||
        program_info->transient.binding_index != expected_transient_binding ||
        program_info->transient.required_byte_length == 0 ||
        program_info->config.binding_index !=
            LOOMC_CMD_PROGRAM_BINDING_INVALID) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the exact layer command program published an incompatible ABI");
    }
  }

  iree_hal_device_t* device =
      qwen_tooling_runtime_context_device(&runtime_context);
  if (iree_status_is_ok(status)) {
    if (FLAG_attention_output[0] != '\0') {
      status = qwen38_allocate_buffer(
          device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
          /*minimum_alignment=*/256, QWEN38_ATTENTION_BYTE_LENGTH,
          &attention_buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_HIDDEN_BYTE_LENGTH, &residual_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/64, sizeof(int32_t), &control_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_ATTENTION_CACHE_BYTE_LENGTH,
        &cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        program_info->transient.minimum_alignment,
        program_info->transient.required_byte_length, &transient_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_HIDDEN_ELEMENT_COUNT, sizeof(*residual_values),
        (void**)&residual_values);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < QWEN38_HIDDEN_ELEMENT_COUNT; ++i) {
      const int32_t centered = (int32_t)(i % 257) - 128;
      residual_values[i] = (float)centered / 128.0f;
    }
    status = iree_hal_device_transfer_h2d(
        device, residual_values, residual_buffer, 0, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, cache_buffer);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: materialized %" PRIhsz " commands; executing layer 3...\n",
            program_info->command_count);
    iree_hal_buffer_binding_t bindings[5] = {
        [0] =
            {
                .buffer = residual_buffer,
                .offset = 0,
                .length = IREE_HAL_WHOLE_BUFFER,
            },
        [1] =
            {
                .buffer = control_buffer,
                .offset = 0,
                .length = IREE_HAL_WHOLE_BUFFER,
            },
        [2] =
            {
                .buffer = cache_buffer,
                .offset = 0,
                .length = IREE_HAL_WHOLE_BUFFER,
            },
    };
    iree_host_size_t binding_count = 4;
    if (attention_buffer) {
      bindings[3] = (iree_hal_buffer_binding_t){
          .buffer = attention_buffer,
          .offset = 0,
          .length = IREE_HAL_WHOLE_BUFFER,
      };
      bindings[4] = (iree_hal_buffer_binding_t){
          .buffer = transient_buffer,
          .offset = 0,
          .length = IREE_HAL_WHOLE_BUFFER,
      };
      binding_count = 5;
    } else {
      bindings[3] = (iree_hal_buffer_binding_t){
          .buffer = transient_buffer,
          .offset = 0,
          .length = IREE_HAL_WHOLE_BUFFER,
      };
    }
    status =
        iree_hal_begin_profiling_from_flags(device, host_allocator, &profiling);
    for (int32_t position = 0;
         position < FLAG_transition_count && iree_status_is_ok(status);
         ++position) {
      status = iree_hal_device_transfer_h2d(
          device, &position, control_buffer, 0, sizeof(position),
          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
      if (iree_status_is_ok(status)) {
        status = qwen38_submit_and_wait(
            device, qwen_tooling_command_program_command_buffer(program),
            (iree_hal_buffer_binding_table_t){
                .count = binding_count,
                .bindings = bindings,
            });
      }
    }
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
    profiling = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, residual_buffer, 0, residual_values, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && FLAG_cache_output[0] != '\0') {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_ATTENTION_CACHE_ELEMENT_COUNT,
        sizeof(*cache_values), (void**)&cache_values);
  }
  if (iree_status_is_ok(status) && cache_values) {
    status = iree_hal_device_transfer_d2h(device, cache_buffer, 0, cache_values,
                                          QWEN38_ATTENTION_CACHE_BYTE_LENGTH,
                                          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                          iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && attention_buffer) {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_ATTENTION_ELEMENT_COUNT,
        sizeof(*attention_values), (void**)&attention_values);
  }
  if (iree_status_is_ok(status) && attention_values) {
    status = iree_hal_device_transfer_d2h(
        device, attention_buffer, 0, attention_values,
        QWEN38_ATTENTION_BYTE_LENGTH, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }

  // Preserve explicitly requested witness data before validating the final
  // residual. A failed transition is precisely when the intermediate outputs
  // are needed to identify the first corrupt command boundary.
  if (iree_status_is_ok(status) && FLAG_output[0] != '\0') {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_output),
        iree_make_const_byte_span(residual_values, QWEN38_HIDDEN_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status) && cache_values) {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_cache_output),
        iree_make_const_byte_span(cache_values,
                                  QWEN38_ATTENTION_CACHE_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status) && attention_values) {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_attention_output),
        iree_make_const_byte_span(attention_values,
                                  QWEN38_ATTENTION_BYTE_LENGTH),
        host_allocator);
  }

  double sum = 0.0;
  double sum_squares = 0.0;
  float minimum = FLT_MAX;
  float maximum = -FLT_MAX;
  iree_host_size_t finite_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < QWEN38_HIDDEN_ELEMENT_COUNT; ++i) {
      const float value = residual_values[i];
      if (isfinite(value)) ++finite_count;
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
      sum += value;
      sum_squares += (double)value * (double)value;
    }
    if (finite_count != QWEN38_HIDDEN_ELEMENT_COUNT) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "the Qwen3.8 layer produced %" PRIhsz
                                "/%d finite values",
                                finite_count, QWEN38_HIDDEN_ELEMENT_COUNT);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout,
            "Qwen3.8 attention layer 3 executed %d time(s): %" PRIhsz
            " commands, %" PRIhsz " parameters, %" PRIu64
            " parameter bytes, %" PRIu64
            " transient bytes, residual[min=%g max=%g sum=%.9g "
            "sum_squares=%.9g]\n",
            FLAG_transition_count, program_info->command_count,
            program_info->parameter_count,
            (uint64_t)qwen_tooling_command_program_set_parameter_byte_length(
                program_set),
            (uint64_t)program_info->transient.required_byte_length, minimum,
            maximum, sum, sum_squares);
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_allocator_free(host_allocator, cache_values);
  iree_allocator_free(host_allocator, attention_values);
  iree_allocator_free(host_allocator, residual_values);
  iree_hal_buffer_release(transient_buffer);
  iree_hal_buffer_release(attention_buffer);
  iree_hal_buffer_release(cache_buffer);
  iree_hal_buffer_release(control_buffer);
  iree_hal_buffer_release(residual_buffer);
  qwen_tooling_command_program_set_release(program_set);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen38-attention-layer-cli",
                       qwen38_attention_layer_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) status = qwen38_attention_layer_run();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
