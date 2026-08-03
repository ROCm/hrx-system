// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/qwen/runtime/model.h"
#include "experimental/qwen/runtime/program.h"
#include "experimental/qwen/runtime/request.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/types.h"
#include "iree/tooling/device_util.h"

#define QWEN_PREFILL_TOKEN_COUNT 512
#define QWEN_PREFILL_TOKEN_BYTE_LENGTH \
  (QWEN_PREFILL_TOKEN_COUNT * sizeof(iree_tokenizer_token_id_t))

IREE_FLAG(string, tokens, "",
          "Raw token IDs: exactly 512 little-endian I32 values.");
IREE_FLAG(int32_t, expected_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Expected selected token (pinned oracle: 264); omit to skip "
          "validation.");
IREE_FLAG(bool, decode_one, false,
          "Consume the device-published prefill token at position 512 and "
          "execute one exact-count decode issue.");
IREE_FLAG(int32_t, expected_decode_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Expected token selected by --decode_one; omit to report without "
          "external validation.");

static const char* const qwen_prefill_cli_usage =
    "Runs complete Qwen prefill-512 and optional exact-count one-token "
    "decode.\n"
    "\n"
    "Required flags:\n"
    "  --device=<device URI>\n"
    "  --parameters=<GGUF or parameter archive path>\n"
    "  --tokens=<raw 512-element little-endian I32 token path>\n"
    "\n"
    "Optional validation:\n"
    "  --expected_token=<prefill-selected token ID; pinned oracle is 264>\n"
    "  --decode_one\n"
    "  --expected_decode_token=<decode-selected token ID>\n"
    "\n"
    "Profiling flags surround the prefill issue. Use the filtered decode "
    "benchmark row for an isolated decode profile.\n";

typedef struct qwen_prefill_cli_timepoint_t {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} qwen_prefill_cli_timepoint_t;

static iree_hal_semaphore_list_t qwen_prefill_cli_timepoint_list(
    qwen_prefill_cli_timepoint_t* timepoint) {
  iree_hal_semaphore_list_t list = {
      .count = 1,
      .semaphores = &timepoint->semaphore,
      .payload_values = &timepoint->value,
  };
  return list;
}

// Transient, non-sanctioned containment for an AMDGPU async file-action
// teardown defect. A host preparation failure must not release the tooling
// runtime while the model gather remains active. Keep this wait tool-only and
// delete it when asynchronous device teardown is safe.
static iree_status_t qwen_wait_for_model_ready_bringup_workaround(
    iree_status_t status, qwen_model_t* model,
    const qwen_prefill_cli_timepoint_t* model_ready) {
  if (!model) return status;
  return iree_status_join(
      status, iree_hal_semaphore_wait(
                  model_ready->semaphore, model_ready->value,
                  iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

static iree_status_t qwen_prefill_cli_load_tokens(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents,
    iree_tokenizer_token_id_t out_token_ids[QWEN_PREFILL_TOKEN_COUNT]) {
  *out_contents = NULL;
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokens must specify a token-ID file");
  }

  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 host_allocator, out_contents));
  if ((*out_contents)->const_buffer.data_length !=
      QWEN_PREFILL_TOKEN_BYTE_LENGTH) {
    const iree_host_size_t actual_byte_length =
        (*out_contents)->const_buffer.data_length;
    iree_io_file_contents_free(*out_contents);
    *out_contents = NULL;
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "token-ID file '%.*s' has %" PRIhsz
        " bytes; expected exactly %zu little-endian I32 bytes",
        (int)path.size, path.data, actual_byte_length,
        QWEN_PREFILL_TOKEN_BYTE_LENGTH);
  }

  const uint8_t* source_data = (*out_contents)->const_buffer.data;
  for (iree_host_size_t i = 0; i < QWEN_PREFILL_TOKEN_COUNT; ++i) {
    const uint32_t token_bits =
        iree_unaligned_load_le_u32(source_data + i * sizeof(uint32_t));
    memcpy(&out_token_ids[i], &token_bits, sizeof(token_bits));
  }
  return iree_ok_status();
}

static iree_status_t qwen_prefill_cli_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();

  if (FLAG_expected_token < IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_token must be nonnegative when validation is requested");
  }
  if (FLAG_expected_decode_token < IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_decode_token must be nonnegative when validation is "
        "requested");
  }
  if (!FLAG_decode_one &&
      FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--expected_decode_token requires --decode_one");
  }
  const iree_host_size_t decode_context_class =
      FLAG_decode_one
          ? qwen_program_decode_context_class(QWEN_PREFILL_TOKEN_COUNT)
          : 0;
  const iree_host_size_t request_context_capacity =
      FLAG_decode_one ? decode_context_class : QWEN_PREFILL_TOKEN_COUNT;

  iree_tokenizer_token_id_t token_ids[QWEN_PREFILL_TOKEN_COUNT];
  iree_io_file_contents_t* token_contents = NULL;
  status =
      qwen_prefill_cli_load_tokens(iree_make_cstring_view(FLAG_tokens),
                                   host_allocator, &token_contents, token_ids);

  qwen_tooling_runtime_context_t runtime_context;
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  } else {
    memset(&runtime_context, 0, sizeof(runtime_context));
  }

  iree_hal_semaphore_t* timeline = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        qwen_tooling_runtime_context_device(&runtime_context),
        IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline);
  }

  qwen_prefill_cli_timepoint_t model_ready = {
      .semaphore = timeline,
      .value = 1,
  };
  qwen_model_t* model = NULL;
  if (iree_status_is_ok(status)) {
    qwen_model_options_t model_options;
    qwen_model_options_initialize(&model_options);
    model_options.device_group = runtime_context.device_group;
    if (runtime_context.jit_worker_count != 0) {
      model_options.jit_worker_count = runtime_context.jit_worker_count;
    }
    qwen_parameter_source_t parameter_source = {
        .index = runtime_context.parameter_index,
        .provider = runtime_context.parameter_provider,
        .scope = iree_string_view_empty(),
    };
    status = qwen_model_load(
        &model_options, &parameter_source, iree_hal_semaphore_list_empty(),
        qwen_prefill_cli_timepoint_list(&model_ready), host_allocator, &model);
  }

  // Program preparation is synchronous host work and intentionally overlaps
  // the asynchronous model gather above.
  qwen_program_t* prefill_program = NULL;
  if (iree_status_is_ok(status)) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_PREFILL;
    program_options.token_count = QWEN_PREFILL_TOKEN_COUNT;
    program_options.context_count = QWEN_PREFILL_TOKEN_COUNT;
    program_options.token_capacity = QWEN_PREFILL_TOKEN_COUNT;
    program_options.context_capacity = request_context_capacity;
    program_options.command_buffer_mode = runtime_context.command_buffer_mode;
    status = qwen_program_prepare(model, &program_options, host_allocator,
                                  &prefill_program);
  }

  qwen_program_t* decode_program = NULL;
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_DECODE;
    program_options.token_count = 1;
    program_options.context_count = decode_context_class;
    program_options.token_capacity = QWEN_PREFILL_TOKEN_COUNT;
    program_options.context_capacity = request_context_capacity;
    program_options.command_buffer_mode = runtime_context.command_buffer_mode;
    status = qwen_program_prepare(model, &program_options, host_allocator,
                                  &decode_program);
  }

  qwen_prefill_cli_timepoint_t request_ready = {
      .semaphore = timeline,
      .value = 2,
  };
  qwen_request_t* request = NULL;
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_capacity = QWEN_PREFILL_TOKEN_COUNT;
    request_options.context_capacity = request_context_capacity;
    status = qwen_request_create(
        model, &request_options, qwen_prefill_cli_timepoint_list(&model_ready),
        qwen_prefill_cli_timepoint_list(&request_ready), host_allocator,
        &request);
  }

  qwen_prefill_cli_timepoint_t tokens_ready = {
      .semaphore = timeline,
      .value = 3,
  };
  if (iree_status_is_ok(status)) {
    status = qwen_request_reset_tokens(
        request, /*context_base=*/0,
        iree_tokenizer_make_token_id_list(token_ids, QWEN_PREFILL_TOKEN_COUNT),
        qwen_prefill_cli_timepoint_list(&request_ready),
        qwen_prefill_cli_timepoint_list(&tokens_ready));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(timeline, tokens_ready.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  // Profiling surrounds only the issue and host-visible completion wait.
  iree_hal_profiling_from_flags_t* profiling = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        runtime_context.device_group, host_allocator, &profiling);
  }

  qwen_prefill_cli_timepoint_t issue_complete = {
      .semaphore = timeline,
      .value = 4,
  };
  if (iree_status_is_ok(status)) {
    status =
        qwen_program_issue(prefill_program, request,
                           qwen_prefill_cli_timepoint_list(&tokens_ready),
                           qwen_prefill_cli_timepoint_list(&issue_complete));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(timeline, issue_complete.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (profiling) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  }

  iree_tokenizer_token_id_t selected_token = IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (iree_status_is_ok(status)) {
    status = qwen_request_read_selected_token(request, &selected_token);
  }
  if (iree_status_is_ok(status) &&
      FLAG_expected_token != IREE_TOKENIZER_TOKEN_ID_INVALID &&
      selected_token != FLAG_expected_token) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "selected token %" PRId32
                              " differs from expected token %" PRId32,
                              selected_token, FLAG_expected_token);
  }
  if (iree_status_is_ok(status)) {
    const qwen_model_statistics_t model_statistics =
        qwen_model_statistics(model);
    fprintf(stdout,
            "Qwen prefill 512 selected token %" PRId32 ": %" PRIhsz
            " dispatches, %" PRIu64 " resident bytes, %" PRIu64
            " transient bytes\n",
            selected_token, qwen_program_dispatch_count(prefill_program),
            (uint64_t)model_statistics.allocation_bytes,
            (uint64_t)qwen_program_transient_byte_length(prefill_program));
  }

  qwen_prefill_cli_timepoint_t decode_complete = {
      .semaphore = timeline,
      .value = 5,
  };
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    status =
        qwen_program_issue(decode_program, request,
                           qwen_prefill_cli_timepoint_list(&issue_complete),
                           qwen_prefill_cli_timepoint_list(&decode_complete));
  }
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    status = iree_hal_semaphore_wait(timeline, decode_complete.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_tokenizer_token_id_t decode_token = IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    status = qwen_request_read_selected_token(request, &decode_token);
  }
  if (iree_status_is_ok(status) && FLAG_decode_one &&
      FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID &&
      decode_token != FLAG_expected_decode_token) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "decode selected token %" PRId32
                              " differs from expected decode token %" PRId32,
                              decode_token, FLAG_expected_decode_token);
  }
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    fprintf(stdout,
            "Qwen decode at context 513 selected token %" PRId32 ": %" PRIhsz
            " dispatches, %" PRIu64 " transient bytes\n",
            decode_token, qwen_program_dispatch_count(decode_program),
            (uint64_t)qwen_program_transient_byte_length(decode_program));
  }

  status =
      qwen_wait_for_model_ready_bringup_workaround(status, model, &model_ready);
  qwen_request_release(request);
  qwen_program_release(decode_program);
  qwen_program_release(prefill_program);
  qwen_model_release(model);
  iree_hal_semaphore_release(timeline);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  iree_io_file_contents_free(token_contents);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen-prefill-cli", qwen_prefill_cli_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_prefill_cli_run();
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
