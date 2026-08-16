// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/qwen/programs/qwen38_text_model_source.h"
#include "experimental/qwen/tooling/command_program.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tooling/device_util.h"
#include "loomc/target/cmd/hal.h"

#define QWEN38_CONTEXT_CAPACITY 512
#define QWEN38_GENERATION_CAPACITY 128
#define QWEN38_HIDDEN_BYTE_LENGTH 20480
#define QWEN38_HIDDEN_CAPACITY_BYTE_LENGTH 10485760
#define QWEN38_GDN_STATE_BYTE_LENGTH 156893184
#define QWEN38_ATTENTION_CACHE_BYTE_LENGTH 33554432
#define QWEN38_MTP_ATTENTION_CACHE_BYTE_LENGTH 2097152
#define QWEN38_CONTROL_BYTE_LENGTH 8
#define QWEN38_MTP_DRAFT_DEPTH 4
#define QWEN38_MTP_PROGRESS_WORD_COUNT 7
#define QWEN38_MTP_PROGRESS_BYTE_LENGTH \
  (QWEN38_MTP_PROGRESS_WORD_COUNT * sizeof(int32_t))
#define QWEN38_DECODE_TEXT_CAPACITY 65536

IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, prompt,
          "Explain why reproducible benchmarks need controlled clocks in "
          "three short paragraphs.",
          "User prompt to tokenize and run.");
IREE_FLAG(bool, raw_prompt, false,
          "Treat --prompt as an already formatted model prompt.");
IREE_FLAG(int32_t, max_tokens, 16,
          "Maximum number of greedy output tokens, from 1 through 128.");
IREE_FLAG(bool, mtp, true,
          "Enable the model's depth-4 MTP proposal and verification path.");
IREE_FLAG(string, residual_output, "",
          "Optional raw F32 final residual-row output path.");
IREE_FLAG(string, hidden_output, "",
          "Optional raw F32 final normalized-hidden output path.");

static const char* const qwen38_text_usage =
    "Runs Qwen3.8 text prefill and greedy MTP generation through reusable "
    "Loom command programs sharing one model parameter pack.\n"
    "\n"
    "Required flags:\n"
    "  --device=<AMDGPU device URI>\n"
    "  --parameters=<Qwen3.8-27B UD-Q5_K_XL GGUF path>\n"
    "  --tokenizer=<Qwen3.8 tokenizer.json path>\n"
    "\n"
    "Input and generation:\n"
    "  --prompt=<user text>\n"
    "  --raw_prompt=true to bypass the no-thinking chat wrapper\n"
    "  --max_tokens=<greedy output count from 1 through 128>\n"
    "\n"
    "Optional diagnostic output:\n"
    "  --residual_output=<raw 5120-element F32 final residual row>\n"
    "  --hidden_output=<raw 5120-element F32 final normalized hidden row>\n";

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

static iree_status_t qwen38_queue_program(
    iree_hal_device_t* device, iree_hal_semaphore_t* execution_semaphore,
    uint64_t* inout_execution_value, qwen_tooling_command_program_t* program,
    iree_hal_buffer_binding_table_t binding_table) {
  uint64_t wait_value = *inout_execution_value;
  uint64_t signal_value = wait_value + 1;
  const iree_hal_semaphore_list_t waits = {
      .count = 1,
      .semaphores = &execution_semaphore,
      .payload_values = &wait_value,
  };
  const iree_hal_semaphore_list_t signals = {
      .count = 1,
      .semaphores = &execution_semaphore,
      .payload_values = &signal_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, waits, signals,
      qwen_tooling_command_program_command_buffer(program), binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) *inout_execution_value = signal_value;
  return status;
}

static iree_status_t qwen38_prepare_text_model(
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
      IREE_SV("qwen38_text_prefill_mtp_greedy"),
      IREE_SV("qwen38_text_prefill_greedy_with_hidden"),
      IREE_SV("qwen38_text_decode_greedy_with_hidden"),
      IREE_SV("qwen38_mtp_decode_greedy"),
      IREE_SV("qwen38_mtp_propose_depth4"),
      IREE_SV("qwen38_mtp_target_verify_depth4"),
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

static iree_status_t qwen38_build_prompt(iree_string_view_t prompt,
                                         bool raw_prompt,
                                         iree_string_builder_t* out_builder) {
  if (raw_prompt) {
    return iree_string_builder_append_string(out_builder, prompt);
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(out_builder, "<|im_start|>user\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(out_builder, prompt));
  return iree_string_builder_append_cstring(
      out_builder,
      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

static double qwen38_elapsed_ms(iree_time_t start, iree_time_t end) {
  return (double)(end - start) / 1000000.0;
}

static double qwen38_rate(iree_host_size_t item_count, iree_time_t start,
                          iree_time_t end) {
  return end != start
             ? (double)item_count * 1000000000.0 / (double)(end - start)
             : 0.0;
}

static bool qwen38_place_control_range(iree_host_size_t byte_length,
                                       iree_host_size_t* cursor,
                                       iree_host_size_t* out_offset) {
  iree_host_size_t byte_offset = 0;
  if (!iree_host_size_checked_align(*cursor, /*alignment=*/64, &byte_offset) ||
      !iree_host_size_checked_add(byte_offset, byte_length, cursor)) {
    return false;
  }
  *out_offset = byte_offset;
  return true;
}

static iree_status_t qwen38_text_run(void) {
  const iree_allocator_t host_allocator = iree_allocator_system();
  const iree_time_t run_start = iree_time_now();
  qwen_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  iree_io_file_contents_t* tokenizer_contents = NULL;
  iree_tokenizer_t* tokenizer = NULL;
  iree_string_builder_t formatted_prompt;
  iree_string_builder_initialize(host_allocator, &formatted_prompt);
  qwen_tooling_command_program_set_t* program_set = NULL;
  qwen_tooling_command_program_t* prefill_program = NULL;
  qwen_tooling_command_program_t* decode_program = NULL;
  qwen_tooling_command_program_t* propose_program = NULL;
  qwen_tooling_command_program_t* verify_program = NULL;
  const loomc_cmd_program_info_t* prefill_info = NULL;
  const loomc_cmd_program_info_t* decode_info = NULL;
  const loomc_cmd_program_info_t* propose_info = NULL;
  const loomc_cmd_program_info_t* verify_info = NULL;
  iree_hal_buffer_t* residual_buffer = NULL;
  iree_hal_buffer_t* state_buffer = NULL;
  iree_hal_buffer_t* cache_buffer = NULL;
  iree_hal_buffer_t* mtp_cache_buffer = NULL;
  iree_hal_buffer_t* normalized_hidden_buffer = NULL;
  iree_hal_buffer_t* token_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  iree_hal_buffer_t* config_buffer = NULL;
  iree_hal_buffer_t* progress_buffer = NULL;
  iree_hal_semaphore_t* execution_semaphore = NULL;
  iree_hal_profiling_from_flags_t* profiling = NULL;
  iree_hal_buffer_mapping_t progress_mapping = {0};
  bool progress_mapping_active = false;
  uint8_t* config_data = NULL;
  char* decoded_text = NULL;
  int32_t token_ids[QWEN38_CONTEXT_CAPACITY];
  memset(token_ids, 0, sizeof(token_ids));
  int32_t control_values[2] = {0, 0};
  float residual_values[QWEN38_HIDDEN_BYTE_LENGTH / sizeof(float)];
  float hidden_values[QWEN38_HIDDEN_BYTE_LENGTH / sizeof(float)];
  iree_host_size_t prompt_token_count = 0;
  iree_host_size_t control_data_offset = 0;
  iree_host_size_t mtp_control_data_offset = 0;
  iree_host_size_t current_token_data_offset = 0;
  iree_host_size_t draft_tokens_data_offset = 0;
  iree_host_size_t verify_tokens_data_offset = 0;
  iree_host_size_t replay_control_data_offset = 0;
  iree_host_size_t acceptance_data_offset = 0;
  iree_host_size_t config_data_length = 0;
  iree_host_size_t mtp_state_data_length = 0;
  iree_time_t ready_time = run_start;
  iree_time_t prefill_start = run_start;
  iree_time_t first_token_time = run_start;
  iree_time_t decode_start = run_start;
  iree_time_t decode_end = run_start;

  iree_status_t status = iree_ok_status();
  if (FLAG_tokenizer[0] == '\0') {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--tokenizer must name tokenizer.json");
  } else if (FLAG_max_tokens < 1 ||
             FLAG_max_tokens > QWEN38_GENERATION_CAPACITY) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "--max_tokens must be between 1 and %d",
                              QWEN38_GENERATION_CAPACITY);
  }

  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_map(iree_make_cstring_view(FLAG_tokenizer),
                                       IREE_IO_FILE_ACCESS_READ, host_allocator,
                                       &tokenizer_contents);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t tokenizer_json = iree_make_string_view(
        (const char*)tokenizer_contents->const_buffer.data,
        tokenizer_contents->const_buffer.data_length);
    status = iree_tokenizer_from_huggingface_json(tokenizer_json,
                                                  host_allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_build_prompt(iree_make_cstring_view(FLAG_prompt),
                                 FLAG_raw_prompt, &formatted_prompt);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode(
        tokenizer, iree_string_builder_view(&formatted_prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(token_ids, /*token_offsets=*/NULL,
                                         /*type_ids=*/NULL,
                                         IREE_ARRAYSIZE(token_ids)),
        host_allocator, &prompt_token_count);
  }
  if (iree_status_is_ok(status) &&
      (prompt_token_count == 0 ||
       prompt_token_count + (iree_host_size_t)FLAG_max_tokens - 1 >
           QWEN38_CONTEXT_CAPACITY)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prompt (%" PRIhsz ") plus generation (%d) exceeds context %d",
        prompt_token_count, FLAG_max_tokens, QWEN38_CONTEXT_CAPACITY);
  }

  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "qwen38: tokenized %" PRIhsz
            " prompt tokens; preparing prefill, decode, and MTP together...\n",
            prompt_token_count);
    status = qwen38_prepare_text_model(&runtime_context, host_allocator,
                                       &program_set);
  }
  if (iree_status_is_ok(status)) {
    prefill_program = qwen_tooling_command_program_set_lookup(
        program_set, FLAG_mtp
                         ? IREE_SV("qwen38_text_prefill_mtp_greedy")
                         : IREE_SV("qwen38_text_prefill_greedy_with_hidden"));
    decode_program = qwen_tooling_command_program_set_lookup(
        program_set, IREE_SV("qwen38_text_decode_greedy_with_hidden"));
    propose_program = qwen_tooling_command_program_set_lookup(
        program_set, IREE_SV("qwen38_mtp_propose_depth4"));
    verify_program = qwen_tooling_command_program_set_lookup(
        program_set, IREE_SV("qwen38_mtp_target_verify_depth4"));
    prefill_info = qwen_tooling_command_program_info(prefill_program);
    decode_info = qwen_tooling_command_program_info(decode_program);
    propose_info = qwen_tooling_command_program_info(propose_program);
    verify_info = qwen_tooling_command_program_info(verify_program);
    const iree_host_size_t expected_prefill_binding_count = FLAG_mtp ? 9 : 8;
    const iree_host_size_t expected_prefill_transient_binding =
        FLAG_mtp ? 7 : 6;
    const iree_host_size_t expected_prefill_config_binding = FLAG_mtp ? 8 : 7;
    if (!prefill_info || !decode_info || !propose_info || !verify_info ||
        prefill_info->rebindable_binding_count !=
            expected_prefill_binding_count ||
        prefill_info->transient.binding_index !=
            expected_prefill_transient_binding ||
        prefill_info->config.binding_index != expected_prefill_config_binding ||
        prefill_info->config.required_byte_length == 0 ||
        decode_info->rebindable_binding_count != 7 ||
        decode_info->transient.binding_index != 6 ||
        decode_info->config.binding_index !=
            LOOMC_CMD_PROGRAM_BINDING_INVALID ||
        propose_info->rebindable_binding_count != 10 ||
        propose_info->transient.binding_index != 9 ||
        propose_info->config.binding_index !=
            LOOMC_CMD_PROGRAM_BINDING_INVALID ||
        verify_info->rebindable_binding_count != 13 ||
        verify_info->transient.binding_index != 12 ||
        verify_info->config.binding_index !=
            LOOMC_CMD_PROGRAM_BINDING_INVALID) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "the text command programs published an incompatible ABI");
    }
  }
  if (iree_status_is_ok(status)) {
    config_data_length =
        (iree_host_size_t)prefill_info->config.required_byte_length;
    mtp_state_data_length = config_data_length;
    if (!qwen38_place_control_range(sizeof(control_values),
                                    &mtp_state_data_length,
                                    &control_data_offset) ||
        !qwen38_place_control_range(sizeof(control_values),
                                    &mtp_state_data_length,
                                    &mtp_control_data_offset) ||
        !qwen38_place_control_range(sizeof(int32_t), &mtp_state_data_length,
                                    &current_token_data_offset) ||
        !qwen38_place_control_range(QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t),
                                    &mtp_state_data_length,
                                    &draft_tokens_data_offset) ||
        !qwen38_place_control_range(QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t),
                                    &mtp_state_data_length,
                                    &verify_tokens_data_offset) ||
        !qwen38_place_control_range(sizeof(control_values),
                                    &mtp_state_data_length,
                                    &replay_control_data_offset) ||
        !qwen38_place_control_range(QWEN38_CONTROL_BYTE_LENGTH,
                                    &mtp_state_data_length,
                                    &acceptance_data_offset)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command control allocation is too large");
    }
  }

  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    device = qwen_tooling_runtime_context_device(&runtime_context);
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_HIDDEN_CAPACITY_BYTE_LENGTH,
        &residual_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_HIDDEN_BYTE_LENGTH,
        &normalized_hidden_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/256, QWEN38_GDN_STATE_BYTE_LENGTH, &state_buffer);
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
        /*minimum_alignment=*/256, QWEN38_MTP_ATTENTION_CACHE_BYTE_LENGTH,
        &mtp_cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*minimum_alignment=*/64, sizeof(token_ids), &token_buffer);
  }
  if (iree_status_is_ok(status)) {
    iree_device_size_t transient_byte_length = iree_max(
        prefill_info->transient.required_byte_length,
        iree_max(decode_info->transient.required_byte_length,
                 iree_max(propose_info->transient.required_byte_length,
                          verify_info->transient.required_byte_length)));
    iree_device_size_t transient_alignment =
        iree_max(prefill_info->transient.minimum_alignment,
                 iree_max(decode_info->transient.minimum_alignment,
                          iree_max(propose_info->transient.minimum_alignment,
                                   verify_info->transient.minimum_alignment)));
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
        transient_alignment, transient_byte_length, &transient_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
            IREE_HAL_BUFFER_USAGE_TRANSFER,
        LOOMC_CMD_HAL_CONFIG_ALIGNMENT, mtp_state_data_length, &config_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_allocate_buffer(
        device,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
        /*minimum_alignment=*/64, QWEN38_MTP_PROGRESS_BYTE_LENGTH,
        &progress_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_range(
        progress_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        /*byte_offset=*/0, QWEN38_MTP_PROGRESS_BYTE_LENGTH, &progress_mapping);
    if (iree_status_is_ok(status)) {
      progress_mapping_active = true;
      memset(progress_mapping.contents.data, 0,
             QWEN38_MTP_PROGRESS_BYTE_LENGTH);
    }
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t config_alignment =
        (iree_host_size_t)iree_max((uint64_t)LOOMC_CMD_HAL_CONFIG_ALIGNMENT,
                                   prefill_info->config.minimum_alignment);
    status = iree_allocator_malloc_aligned(
        host_allocator, mtp_state_data_length, config_alignment, /*offset=*/0,
        (void**)&config_data);
    if (iree_status_is_ok(status)) {
      memset(config_data, 0, mtp_state_data_length);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, QWEN38_DECODE_TEXT_CAPACITY,
                                   (void**)&decoded_text);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &execution_semaphore);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_h2d(
        device, token_ids, token_buffer, 0, sizeof(token_ids),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, state_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, mtp_cache_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = qwen38_fill_and_wait(device, normalized_hidden_buffer);
  }
  if (iree_status_is_ok(status)) {
    const uint64_t argument_bits[] = {(uint64_t)prompt_token_count};
    status = qwen_tooling_command_program_populate_config(
        prefill_program, argument_bits, IREE_ARRAYSIZE(argument_bits),
        iree_make_byte_span(
            config_data,
            (iree_host_size_t)prefill_info->config.required_byte_length));
  }
  if (iree_status_is_ok(status)) {
    iree_unaligned_store_le_u32(config_data + control_data_offset,
                                (uint32_t)prompt_token_count);
    iree_unaligned_store_le_u32(
        config_data + control_data_offset + sizeof(uint32_t), 0);
    status = iree_hal_device_transfer_h2d(
        device, config_data, config_buffer, 0, mtp_state_data_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }

  const iree_hal_buffer_binding_t prefill_bindings[9] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = sizeof(control_values)},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [7] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [8] = {.buffer = config_buffer,
             .offset = 0,
             .length =
                 prefill_info ? prefill_info->config.required_byte_length : 0},
  };
  const iree_hal_buffer_binding_t ordinary_prefill_bindings[8] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = sizeof(control_values)},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [7] = {.buffer = config_buffer,
             .offset = 0,
             .length =
                 prefill_info ? prefill_info->config.required_byte_length : 0},
  };
  const iree_hal_buffer_binding_t decode_bindings[7] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = sizeof(control_values)},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
  };
  const iree_hal_buffer_binding_t propose_bindings[10] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = config_buffer,
             .offset = mtp_control_data_offset,
             .length = QWEN38_CONTROL_BYTE_LENGTH},
      [3] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = config_buffer,
             .offset = current_token_data_offset,
             .length = sizeof(int32_t)},
      [7] = {.buffer = config_buffer,
             .offset = draft_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [8] = {.buffer = config_buffer,
             .offset = verify_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [9] = {.buffer = transient_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
  };
  const iree_hal_buffer_binding_t verify_bindings[13] = {
      [0] = {.buffer = residual_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [1] = {.buffer = config_buffer,
             .offset = control_data_offset,
             .length = QWEN38_CONTROL_BYTE_LENGTH},
      [2] = {.buffer = state_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [3] = {.buffer = cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [4] = {.buffer = mtp_cache_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [5] = {.buffer = normalized_hidden_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [6] = {.buffer = token_buffer,
             .offset = 0,
             .length = IREE_HAL_WHOLE_BUFFER},
      [7] = {.buffer = config_buffer,
             .offset = draft_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [8] = {.buffer = config_buffer,
             .offset = verify_tokens_data_offset,
             .length = QWEN38_MTP_DRAFT_DEPTH * sizeof(int32_t)},
      [9] = {.buffer = config_buffer,
             .offset = replay_control_data_offset,
             .length = QWEN38_CONTROL_BYTE_LENGTH},
      [10] = {.buffer = config_buffer,
              .offset = acceptance_data_offset,
              .length = QWEN38_CONTROL_BYTE_LENGTH},
      [11] = {.buffer = progress_buffer,
              .offset = 0,
              .length = QWEN38_MTP_PROGRESS_BYTE_LENGTH},
      [12] = {.buffer = transient_buffer,
              .offset = 0,
              .length = IREE_HAL_WHOLE_BUFFER},
  };
  uint64_t execution_value = 1;
  if (iree_status_is_ok(status)) {
    ready_time = iree_time_now();
    fprintf(stderr,
            "qwen38: materialized prefill=%" PRIhsz " decode=%" PRIhsz
            " MTP-propose=%" PRIhsz " MTP-verify=%" PRIhsz
            " commands; executing...\n",
            prefill_info->command_count, decode_info->command_count,
            propose_info->command_count, verify_info->command_count);
    status =
        iree_hal_begin_profiling_from_flags(device, host_allocator, &profiling);
  }
  if (iree_status_is_ok(status)) {
    const iree_hal_semaphore_list_t signals = {
        .count = 1,
        .semaphores = &execution_semaphore,
        .payload_values = &execution_value,
    };
    prefill_start = iree_time_now();
    const iree_hal_buffer_binding_table_t prefill_binding_table =
        FLAG_mtp
            ? (iree_hal_buffer_binding_table_t){
                  .count = IREE_ARRAYSIZE(prefill_bindings),
                  .bindings = prefill_bindings,
              }
            : (iree_hal_buffer_binding_table_t){
                  .count = IREE_ARRAYSIZE(ordinary_prefill_bindings),
                  .bindings = ordinary_prefill_bindings,
              };
    status = iree_hal_device_queue_execute(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signals, qwen_tooling_command_program_command_buffer(prefill_program),
        prefill_binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(execution_semaphore, execution_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
    first_token_time = iree_time_now();
    decode_start = first_token_time;
  }

  int32_t generated_count = 1;
  int32_t mtp_cycle_count = 0;
  int32_t mtp_committed_count = 0;
  int32_t mtp_accepted_draft_count = 0;
  int32_t mtp_depth_counts[QWEN38_MTP_DRAFT_DEPTH + 1] = {0};
  // A cycle always emits at least one token, so this conservative batch can
  // execute without observing intermediate acceptance decisions and cannot
  // cross the requested output limit. The host reads cumulative progress only
  // after the batch's final timeline signal.
  while (iree_status_is_ok(status) && FLAG_mtp &&
         FLAG_max_tokens - generated_count >= QWEN38_MTP_DRAFT_DEPTH) {
    const int32_t prior_generated_count = generated_count;
    const int32_t batch_cycle_count =
        (FLAG_max_tokens - generated_count) / QWEN38_MTP_DRAFT_DEPTH;
    for (int32_t cycle = 0;
         iree_status_is_ok(status) && cycle < batch_cycle_count; ++cycle) {
      status = qwen38_queue_program(
          device, execution_semaphore, &execution_value, propose_program,
          (iree_hal_buffer_binding_table_t){
              .count = IREE_ARRAYSIZE(propose_bindings),
              .bindings = propose_bindings,
          });
      if (iree_status_is_ok(status)) {
        status = qwen38_queue_program(
            device, execution_semaphore, &execution_value, verify_program,
            (iree_hal_buffer_binding_table_t){
                .count = IREE_ARRAYSIZE(verify_bindings),
                .bindings = verify_bindings,
            });
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(execution_semaphore, execution_value,
                                       iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
    }
    int32_t next_generated_count = prior_generated_count;
    int32_t next_accepted_draft_count = mtp_accepted_draft_count;
    if (iree_status_is_ok(status)) {
      const uint8_t* progress_data = progress_mapping.contents.data;
      next_generated_count = (int32_t)iree_unaligned_load_le_u32(progress_data);
      next_accepted_draft_count =
          (int32_t)iree_unaligned_load_le_u32(progress_data + sizeof(uint32_t));
      const int32_t minimum_generated_count =
          prior_generated_count + batch_cycle_count;
      const int32_t maximum_generated_count =
          prior_generated_count + batch_cycle_count * QWEN38_MTP_DRAFT_DEPTH;
      const int32_t maximum_accepted_draft_count =
          (mtp_cycle_count + batch_cycle_count) * QWEN38_MTP_DRAFT_DEPTH;
      if (next_generated_count < minimum_generated_count ||
          next_generated_count > maximum_generated_count ||
          next_generated_count > FLAG_max_tokens ||
          next_accepted_draft_count < mtp_accepted_draft_count ||
          next_accepted_draft_count > maximum_accepted_draft_count) {
        status = iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "MTP batch of %" PRId32 " cycles advanced from %" PRId32
            " to %" PRId32 " outputs with %" PRId32
            " cumulative accepted drafts",
            batch_cycle_count, prior_generated_count, next_generated_count,
            next_accepted_draft_count);
      }
    }
    if (iree_status_is_ok(status)) {
      generated_count = next_generated_count;
      mtp_cycle_count += batch_cycle_count;
      mtp_committed_count = generated_count - 1;
      mtp_accepted_draft_count = next_accepted_draft_count;
      const uint8_t* depth_data =
          progress_mapping.contents.data + 2 * sizeof(uint32_t);
      int32_t observed_cycle_count = 0;
      for (int32_t depth = 0; depth <= QWEN38_MTP_DRAFT_DEPTH; ++depth) {
        mtp_depth_counts[depth] = (int32_t)iree_unaligned_load_le_u32(
            depth_data + depth * sizeof(uint32_t));
        observed_cycle_count += mtp_depth_counts[depth];
      }
      if (observed_cycle_count != mtp_cycle_count) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "MTP depth histogram accounts for %" PRId32
                                  " of %" PRId32 " completed cycles",
                                  observed_cycle_count, mtp_cycle_count);
      }
    }
  }

  while (iree_status_is_ok(status) && generated_count < FLAG_max_tokens) {
    status = qwen38_queue_program(device, execution_semaphore, &execution_value,
                                  decode_program,
                                  (iree_hal_buffer_binding_table_t){
                                      .count = IREE_ARRAYSIZE(decode_bindings),
                                      .bindings = decode_bindings,
                                  });
    if (iree_status_is_ok(status)) ++generated_count;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(execution_semaphore, execution_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) decode_end = iree_time_now();
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  profiling = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, token_buffer, 0, token_ids, sizeof(token_ids),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, config_buffer, control_data_offset, control_values,
        sizeof(control_values), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && FLAG_residual_output[0] != '\0') {
    status = iree_hal_device_transfer_d2h(
        device, residual_buffer, 0, residual_values, QWEN38_HIDDEN_BYTE_LENGTH,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  if (iree_status_is_ok(status) && FLAG_hidden_output[0] != '\0') {
    status = iree_hal_device_transfer_d2h(
        device, normalized_hidden_buffer, 0, hidden_values,
        QWEN38_HIDDEN_BYTE_LENGTH, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
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
  if (iree_status_is_ok(status) &&
      (control_values[0] != (int32_t)prompt_token_count + FLAG_max_tokens - 1 ||
       control_values[1] != FLAG_max_tokens)) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "generation control advanced to [position=%" PRId32
                         ", generated=%" PRId32 "]",
                         control_values[0], control_values[1]);
  }

  iree_host_size_t visible_token_count = (iree_host_size_t)FLAG_max_tokens;
  if (iree_status_is_ok(status)) {
    const int32_t eos_token = iree_tokenizer_vocab_lookup(
        iree_tokenizer_vocab(tokenizer), IREE_SV("<|im_end|>"));
    if (eos_token >= 0) {
      for (iree_host_size_t i = 0; i < visible_token_count; ++i) {
        if (token_ids[i + 1] == eos_token) {
          visible_token_count = i + 1;
          break;
        }
      }
    }
    iree_host_size_t decoded_text_length = 0;
    status = iree_tokenizer_decode(
        tokenizer,
        iree_tokenizer_make_token_id_list(token_ids + 1, visible_token_count),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(decoded_text,
                                      QWEN38_DECODE_TEXT_CAPACITY - 1),
        host_allocator, &decoded_text_length);
    if (iree_status_is_ok(status)) decoded_text[decoded_text_length] = '\0';
  }

  if (iree_status_is_ok(status)) {
    const iree_host_size_t decode_transition_count =
        (iree_host_size_t)FLAG_max_tokens - 1;
    fprintf(stdout,
            "Qwen3.8 command-program generation\n"
            "  prompt tokens:        %" PRIhsz
            "\n"
            "  generated tokens:     %d\n"
            "  parameter bytes:      %" PRIu64
            "\n"
            "  MTP cycles:           %" PRId32 " (%" PRId32 " tokens, %" PRId32
            " accepted drafts)\n"
            "  MTP accepted depths:  0:%" PRId32 " 1:%" PRId32 " 2:%" PRId32
            " 3:%" PRId32 " 4:%" PRId32
            "\n"
            "  setup to ready:       %.3f ms\n"
            "  prefill:              %.3f ms (%.2f tok/s)\n"
            "  cold time to token:   %.3f ms\n"
            "  decode:               %.3f ms (%.2f tok/s)\n"
            "  total through output: %.3f ms\n",
            prompt_token_count, FLAG_max_tokens,
            (uint64_t)qwen_tooling_command_program_set_parameter_byte_length(
                program_set),
            mtp_cycle_count, mtp_committed_count, mtp_accepted_draft_count,
            mtp_depth_counts[0], mtp_depth_counts[1], mtp_depth_counts[2],
            mtp_depth_counts[3], mtp_depth_counts[4],
            qwen38_elapsed_ms(run_start, ready_time),
            qwen38_elapsed_ms(prefill_start, first_token_time),
            qwen38_rate(prompt_token_count, prefill_start, first_token_time),
            qwen38_elapsed_ms(run_start, first_token_time),
            qwen38_elapsed_ms(decode_start, decode_end),
            qwen38_rate(decode_transition_count, decode_start, decode_end),
            qwen38_elapsed_ms(run_start, decode_end));
    fprintf(stdout, "  output token IDs:");
    for (int32_t i = 0; i < FLAG_max_tokens; ++i) {
      fprintf(stdout, " %" PRId32, token_ids[i + 1]);
    }
    fprintf(stdout, "\n\n%s\n", decoded_text);
  }

  if (progress_mapping_active) {
    status = iree_status_join(status,
                              iree_hal_buffer_unmap_range(&progress_mapping));
  }
  iree_hal_semaphore_release(execution_semaphore);
  iree_allocator_free(host_allocator, decoded_text);
  iree_allocator_free_aligned(host_allocator, config_data);
  iree_hal_buffer_release(progress_buffer);
  iree_hal_buffer_release(config_buffer);
  iree_hal_buffer_release(transient_buffer);
  iree_hal_buffer_release(token_buffer);
  iree_hal_buffer_release(normalized_hidden_buffer);
  iree_hal_buffer_release(mtp_cache_buffer);
  iree_hal_buffer_release(cache_buffer);
  iree_hal_buffer_release(state_buffer);
  iree_hal_buffer_release(residual_buffer);
  qwen_tooling_command_program_set_release(program_set);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  iree_string_builder_deinitialize(&formatted_prompt);
  iree_tokenizer_free(tokenizer);
  if (tokenizer_contents) iree_io_file_contents_free(tokenizer_contents);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen38-cli", qwen38_text_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) status = qwen38_text_run();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
