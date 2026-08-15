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

#include "experimental/qwen/programs/qwen38_text_model_source.h"
#include "experimental/qwen/tooling/command_program.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tooling/device_util.h"

#define QWEN38_HIDDEN_ELEMENT_COUNT 5120
#define QWEN38_HIDDEN_BYTE_LENGTH 20480
#define QWEN38_MTP_CACHE_BYTE_LENGTH 2097152
#define QWEN38_TOKEN_BUFFER_ELEMENT_COUNT 129
#define QWEN38_VOCABULARY_SIZE 248320

IREE_FLAG(string, hidden_input, "",
          "Optional raw F32 normalized target hidden-state input path.");
IREE_FLAG(string, hidden_output, "",
          "Optional raw F32 normalized MTP hidden-state output path.");
IREE_FLAG(string, residual_output, "",
          "Optional raw F32 MTP residual output path.");
IREE_FLAG(int32_t, initial_token_id, 0,
          "Target token consumed by the MTP transition.");
IREE_FLAG(int32_t, initial_position, 0,
          "Absolute position of the target token.");

static const char* const qwen38_mtp_usage =
    "Advances the exact Qwen3.8 block-64 NextN/MTP transition through one "
    "reusable Loom command program. Only the embedding, output, and block-64 "
    "parameter roots are gathered.\n"
    "\n"
    "Required flags:\n"
    "  --device=<AMDGPU device URI>\n"
    "  --parameters=<Qwen3.8-27B UD-Q5_K_XL GGUF path>\n"
    "\n"
    "Optional input and output:\n"
    "  --hidden_input=<raw 5120-element F32 target hidden state>\n"
    "  --hidden_output=<raw 5120-element F32 MTP hidden state>\n"
    "  --residual_output=<raw 5120-element F32 MTP residual>\n"
    "  --initial_token_id=<token ID in [0, 248320)>\n"
    "  --initial_position=<absolute token position in [0, 512)>\n";

static iree_status_t qwen38_allocate_buffer(
    iree_hal_device_t* device, iree_device_size_t minimum_alignment,
    iree_device_size_t byte_length, iree_hal_buffer_t** out_buffer) {
  const iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = minimum_alignment,
  };
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t qwen38_prepare_mtp(
    qwen_tooling_runtime_context_t* runtime_context,
    iree_allocator_t host_allocator,
    qwen_tooling_command_program_set_t** out_program_set) {
  if (qwen38_text_model_source_size() != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen3.8 text model source must contain exactly one file");
  }
  const iree_file_toc_t* files = qwen38_text_model_source_create();
  const iree_string_view_t root_names[] = {
      IREE_SV("qwen38_mtp_decode_greedy"),
  };
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

static iree_status_t qwen38_initialize_hidden(iree_allocator_t host_allocator,
                                              float* hidden_values) {
  if (FLAG_hidden_input[0] == '\0') {
    for (iree_host_size_t i = 0; i < QWEN38_HIDDEN_ELEMENT_COUNT; ++i) {
      const int32_t centered = (int32_t)(i % 257) - 128;
      hidden_values[i] = (float)centered / 128.0f;
    }
    return iree_ok_status();
  }

  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_hidden_input), host_allocator, &contents));
  iree_status_t status = iree_ok_status();
  if (contents->const_buffer.data_length != QWEN38_HIDDEN_BYTE_LENGTH) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--hidden_input contains %" PRIhsz " bytes; expected %d",
        contents->const_buffer.data_length, QWEN38_HIDDEN_BYTE_LENGTH);
  } else {
    memcpy(hidden_values, contents->const_buffer.data,
           QWEN38_HIDDEN_BYTE_LENGTH);
  }
  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t qwen38_validate_finite(iree_string_view_t name,
                                            const float* values,
                                            float* out_minimum,
                                            float* out_maximum,
                                            double* out_sum) {
  float minimum = FLT_MAX;
  float maximum = -FLT_MAX;
  double sum = 0.0;
  for (iree_host_size_t i = 0; i < QWEN38_HIDDEN_ELEMENT_COUNT; ++i) {
    const float value = values[i];
    if (!isfinite(value)) {
      return iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "%.*s contains a non-finite value at index %" PRIhsz, (int)name.size,
          name.data, i);
    }
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    sum += value;
  }
  *out_minimum = minimum;
  *out_maximum = maximum;
  *out_sum = sum;
  return iree_ok_status();
}

static iree_status_t qwen38_mtp_run(void) {
  const iree_allocator_t host_allocator = iree_allocator_system();
  qwen_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  qwen_tooling_command_program_set_t* program_set = NULL;
  qwen_tooling_command_program_t* program = NULL;
  const loomc_cmd_program_info_t* program_info = NULL;
  iree_hal_buffer_t* residual_buffer = NULL;
  iree_hal_buffer_t* control_buffer = NULL;
  iree_hal_buffer_t* attention_cache_buffer = NULL;
  iree_hal_buffer_t* hidden_buffer = NULL;
  iree_hal_buffer_t* token_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  float* residual_values = NULL;
  float* hidden_values = NULL;
  int32_t control_values[2] = {FLAG_initial_position, 0};
  int32_t token_ids[QWEN38_TOKEN_BUFFER_ELEMENT_COUNT];
  memset(token_ids, 0, sizeof(token_ids));
  token_ids[0] = FLAG_initial_token_id;
  iree_hal_profiling_from_flags_t* profiling = NULL;

  iree_status_t status = iree_ok_status();
  if (FLAG_initial_token_id < 0 ||
      FLAG_initial_token_id >= QWEN38_VOCABULARY_SIZE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--initial_token_id must be between 0 and %d",
                              QWEN38_VOCABULARY_SIZE - 1);
  } else if (FLAG_initial_position < 0 || FLAG_initial_position >= 512) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--initial_position must be between 0 and 511");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_HIDDEN_ELEMENT_COUNT, sizeof(*hidden_values),
        (void**)&hidden_values);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, QWEN38_HIDDEN_ELEMENT_COUNT, sizeof(*residual_values),
        (void**)&residual_values);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_initialize_hidden(host_allocator, hidden_values);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: indexing GGUF; compiling and gathering the isolated "
            "MTP transition...\n");
    status = qwen38_prepare_mtp(&runtime_context, host_allocator, &program_set);
  }
  if (iree_status_is_ok(status)) {
    program = qwen_tooling_command_program_set_lookup(
        program_set, IREE_SV("qwen38_mtp_decode_greedy"));
    program_info = qwen_tooling_command_program_info(program);
    if (program_info->rebindable_binding_count != 6 ||
        program_info->transient.binding_index != 5 ||
        program_info->transient.required_byte_length == 0 ||
        program_info->config.binding_index !=
            LOOMC_CMD_PROGRAM_BINDING_INVALID) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the MTP command program published an incompatible ABI");
    }
  }

  iree_hal_device_t* device =
      qwen_tooling_runtime_context_device(&runtime_context);
  if (iree_status_is_ok(status)) {
    status =
        qwen38_allocate_buffer(device, /*minimum_alignment=*/256,
                               QWEN38_HIDDEN_BYTE_LENGTH, &residual_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(device, /*minimum_alignment=*/64,
                                    sizeof(control_values), &control_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(device, /*minimum_alignment=*/256,
                                    QWEN38_MTP_CACHE_BYTE_LENGTH,
                                    &attention_cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(device, /*minimum_alignment=*/256,
                                    QWEN38_HIDDEN_BYTE_LENGTH, &hidden_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(device, /*minimum_alignment=*/64,
                                    sizeof(token_ids), &token_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, program_info->transient.minimum_alignment,
        program_info->transient.required_byte_length, &transient_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(
        device, control_values, control_buffer, 0, sizeof(control_values),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(
        device, hidden_values, hidden_buffer, 0, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(
        device, token_ids, token_buffer, 0, sizeof(token_ids),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, attention_cache_buffer);
  }

  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: materialized %" PRIhsz " MTP commands with %" PRIu64
            " parameter bytes; executing once...\n",
            program_info->command_count,
            (uint64_t)qwen_tooling_command_program_set_parameter_byte_length(
                program_set));
    const iree_hal_buffer_binding_t bindings[6] = {
        [0] = {.buffer = residual_buffer,
               .offset = 0,
               .length = IREE_HAL_WHOLE_BUFFER},
        [1] = {.buffer = control_buffer,
               .offset = 0,
               .length = IREE_HAL_WHOLE_BUFFER},
        [2] = {.buffer = attention_cache_buffer,
               .offset = 0,
               .length = IREE_HAL_WHOLE_BUFFER},
        [3] = {.buffer = hidden_buffer,
               .offset = 0,
               .length = IREE_HAL_WHOLE_BUFFER},
        [4] = {.buffer = token_buffer,
               .offset = 0,
               .length = IREE_HAL_WHOLE_BUFFER},
        [5] = {.buffer = transient_buffer,
               .offset = 0,
               .length = IREE_HAL_WHOLE_BUFFER},
    };
    status =
        iree_hal_begin_profiling_from_flags(device, host_allocator, &profiling);
    if (iree_status_is_ok(status)) {
      status = qwen38_submit_and_wait(
          device, qwen_tooling_command_program_command_buffer(program),
          (iree_hal_buffer_binding_table_t){
              .count = IREE_ARRAYSIZE(bindings),
              .bindings = bindings,
          });
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, hidden_buffer, 0, hidden_values, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, control_buffer, 0, control_values, sizeof(control_values),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, token_buffer, 0, token_ids, sizeof(token_ids),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) &&
      (control_values[0] != FLAG_initial_position + 1 ||
       control_values[1] != 1 || token_ids[0] != token_ids[1] ||
       token_ids[0] < 0 || token_ids[0] >= QWEN38_VOCABULARY_SIZE)) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "MTP commit produced control=[%" PRId32 ", %" PRId32
        "] and token pair=[%" PRId32 ", %" PRId32 "]",
        control_values[0], control_values[1], token_ids[0], token_ids[1]);
  }

  float residual_minimum = 0.0f;
  float residual_maximum = 0.0f;
  double residual_sum = 0.0;
  float hidden_minimum = 0.0f;
  float hidden_maximum = 0.0f;
  double hidden_sum = 0.0;
  if (iree_status_is_ok(status)) {
    status = qwen38_validate_finite(IREE_SV("MTP residual"), residual_values,
                                    &residual_minimum, &residual_maximum,
                                    &residual_sum);
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen38_validate_finite(IREE_SV("MTP hidden state"), hidden_values,
                               &hidden_minimum, &hidden_maximum, &hidden_sum);
  }
  if (iree_status_is_ok(status) && FLAG_residual_output[0] != '\0') {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_residual_output),
        iree_make_const_byte_span(residual_values, QWEN38_HIDDEN_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status) && FLAG_hidden_output[0] != '\0') {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_hidden_output),
        iree_make_const_byte_span(hidden_values, QWEN38_HIDDEN_BYTE_LENGTH),
        host_allocator);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout,
            "Qwen3.8 MTP transition: %" PRIhsz " commands, %" PRIhsz
            " parameters, %" PRIu64 " parameter bytes, %" PRIu64
            " transient bytes, token=%" PRId32
            ", residual[min=%g max=%g sum=%.9g], "
            "hidden[min=%g max=%g sum=%.9g]\n",
            program_info->command_count, program_info->parameter_count,
            (uint64_t)qwen_tooling_command_program_set_parameter_byte_length(
                program_set),
            (uint64_t)program_info->transient.required_byte_length,
            token_ids[1], residual_minimum, residual_maximum, residual_sum,
            hidden_minimum, hidden_maximum, hidden_sum);
  }

  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  iree_hal_buffer_release(transient_buffer);
  iree_hal_buffer_release(token_buffer);
  iree_hal_buffer_release(hidden_buffer);
  iree_hal_buffer_release(attention_cache_buffer);
  iree_hal_buffer_release(control_buffer);
  iree_hal_buffer_release(residual_buffer);
  qwen_tooling_command_program_set_release(program_set);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  iree_allocator_free(host_allocator, residual_values);
  iree_allocator_free(host_allocator, hidden_values);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen38-mtp-cli", qwen38_mtp_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) status = qwen38_mtp_run();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
