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
#include "experimental/qwen/runtime/model_shape.h"
#include "experimental/qwen/runtime/program.h"
#include "experimental/qwen/runtime/request.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tooling/device_util.h"

// Temporary bound imposed by the cooperative decode-attention bring-up
// adapter. The canonical long-context path supports a larger model context.
#define QWEN_GENERATION_DECODE_CONTEXT_LIMIT 2048
#define QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE 64
#define QWEN_GENERATION_DECODE_CONTEXT_CLASS_COUNT \
  (QWEN_GENERATION_DECODE_CONTEXT_LIMIT /          \
   QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE)

IREE_FLAG(string, tokenizer, "", "HuggingFace tokenizer.json path.");
IREE_FLAG(string, prompt, "", "User message to generate a response for.");
IREE_FLAG(string, system, "", "Optional system message.");
IREE_FLAG(int32_t, max_tokens, 16,
          "Maximum generated token count, including a terminating EOS.");
IREE_FLAG(int32_t, context_length, QWEN_GENERATION_DECODE_CONTEXT_LIMIT,
          "Request K/V capacity; currently at most 2048.");
IREE_FLAG(bool, print_token_ids, false,
          "Print each generated token ID to stderr.");

static const char* const qwen_generation_cli_usage =
    "Generates Qwen text from one system/user chat turn with greedy "
    "selection.\n"
    "\n"
    "Required flags:\n"
    "  --device=<device URI>\n"
    "  --parameters=<Qwen3-30B-A3B GGUF path>\n"
    "  --tokenizer=<matching HuggingFace tokenizer.json path>\n"
    "  --prompt=<user message>\n"
    "\n"
    "The prompt is uploaded once. Every continuation token remains on the "
    "device; host reads are observation-only for text streaming and EOS.\n";

typedef struct qwen_generation_cli_timepoint_t {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} qwen_generation_cli_timepoint_t;

static iree_hal_semaphore_list_t qwen_generation_cli_timepoint_list(
    qwen_generation_cli_timepoint_t* timepoint) {
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
static iree_status_t qwen_generation_cli_wait_for_model_ready_workaround(
    iree_status_t status, qwen_model_t* model,
    const qwen_generation_cli_timepoint_t* model_ready) {
  if (!model) return status;
  return iree_status_join(
      status, iree_hal_semaphore_wait(
                  model_ready->semaphore, model_ready->value,
                  iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

static iree_status_t qwen_generation_cli_load_tokenizer(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_tokenizer_t** out_tokenizer) {
  *out_tokenizer = NULL;
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokenizer is required");
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);
  if (iree_status_is_ok(status)) {
    const iree_string_view_t json =
        iree_make_string_view((const char*)file_contents->const_buffer.data,
                              file_contents->const_buffer.data_length);
    status = iree_tokenizer_from_huggingface_json(json, host_allocator,
                                                  out_tokenizer);
  }
  iree_io_file_contents_free(file_contents);
  return status;
}

static iree_status_t qwen_generation_cli_build_chat_prompt(
    iree_string_view_t system_message, iree_string_view_t user_message,
    iree_string_builder_t* out_builder) {
  if (iree_string_view_is_empty(user_message)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--prompt is required and must not be empty");
  }

  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(system_message)) {
    status =
        iree_string_builder_append_cstring(out_builder, "<|im_start|>system\n");
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(out_builder, system_message);
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(out_builder, "<|im_end|>\n");
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_string_builder_append_cstring(out_builder, "<|im_start|>user\n");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(out_builder, user_message);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(
        out_builder, "<|im_end|>\n<|im_start|>assistant\n");
  }
  return status;
}

static iree_status_t qwen_generation_cli_encode_prompt(
    const iree_tokenizer_t* tokenizer, iree_string_view_t chat_prompt,
    iree_allocator_t host_allocator, iree_tokenizer_token_id_t** out_token_ids,
    iree_host_size_t* out_token_count) {
  *out_token_ids = NULL;
  *out_token_count = 0;

  // A byte-level BPE cannot emit more model tokens than input bytes. Explicit
  // chat control tokens replace multi-byte spans, so the same bound applies.
  const iree_host_size_t token_capacity = chat_prompt.size;
  iree_tokenizer_token_id_t* token_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, token_capacity, sizeof(*token_ids), (void**)&token_ids));
  iree_status_t status = iree_tokenizer_encode(
      tokenizer, chat_prompt, IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(token_ids, /*token_offsets=*/NULL,
                                       /*type_ids=*/NULL, token_capacity),
      host_allocator, out_token_count);
  if (iree_status_is_ok(status) && *out_token_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Qwen chat prompt encoded to no tokens");
  }
  if (iree_status_is_ok(status)) {
    *out_token_ids = token_ids;
  } else {
    iree_allocator_free(host_allocator, token_ids);
  }
  return status;
}

static iree_status_t qwen_generation_cli_write_text(const char* data,
                                                    iree_host_size_t length) {
  if (length == 0) return iree_ok_status();
  if (fwrite(data, 1, length, stdout) != length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to write generated text to stdout");
  }
  if (fflush(stdout) != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to flush generated text to stdout");
  }
  return iree_ok_status();
}

static iree_status_t qwen_generation_cli_emit_token(
    iree_tokenizer_decode_state_t* decode_state,
    iree_tokenizer_token_id_t token_id) {
  char text_buffer[IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE];
  iree_tokenizer_token_id_list_t remaining_tokens =
      iree_tokenizer_make_token_id_list(&token_id, 1);
  while (remaining_tokens.count != 0) {
    iree_host_size_t token_count = 0;
    iree_host_size_t text_length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_feed(
        decode_state, remaining_tokens,
        iree_make_mutable_string_view(text_buffer, sizeof(text_buffer)),
        &token_count, &text_length));
    if (token_count == 0 && text_length == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "Qwen token %" PRId32
                              " exceeds the streaming decode buffer",
                              token_id);
    }
    IREE_RETURN_IF_ERROR(
        qwen_generation_cli_write_text(text_buffer, text_length));
    remaining_tokens.values += token_count;
    remaining_tokens.count -= token_count;
  }
  return iree_ok_status();
}

static iree_status_t qwen_generation_cli_finalize_text(
    iree_tokenizer_decode_state_t* decode_state) {
  char text_buffer[IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE];
  iree_host_size_t text_length = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_decode_state_finalize(
      decode_state,
      iree_make_mutable_string_view(text_buffer, sizeof(text_buffer)),
      &text_length));
  return qwen_generation_cli_write_text(text_buffer, text_length);
}

static iree_host_size_t qwen_generation_cli_decode_context_class(
    iree_host_size_t context_base) {
  return (context_base / QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE + 1) *
         QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE;
}

static iree_status_t qwen_generation_cli_prepare_decode_program(
    qwen_model_t* model, iree_host_size_t context_class,
    iree_host_size_t token_capacity, iree_host_size_t context_capacity,
    iree_hal_command_buffer_mode_t command_buffer_mode,
    iree_allocator_t host_allocator, qwen_program_t** out_program) {
  qwen_program_options_t program_options;
  qwen_program_options_initialize(&program_options);
  program_options.kind = QWEN_PROGRAM_KIND_DECODE;
  program_options.token_count = 1;
  program_options.context_count = context_class;
  program_options.token_capacity = token_capacity;
  program_options.context_capacity = context_capacity;
  program_options.command_buffer_mode = command_buffer_mode;
  return qwen_program_prepare(model, &program_options, host_allocator,
                              out_program);
}

static void qwen_generation_cli_release_decode_programs(
    qwen_program_t** programs) {
  for (iree_host_size_t i = 0; i < QWEN_GENERATION_DECODE_CONTEXT_CLASS_COUNT;
       ++i) {
    qwen_program_release(programs[i]);
  }
}

static iree_status_t qwen_generation_cli_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();
  if (FLAG_max_tokens <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--max_tokens must be greater than zero");
  }
  if (FLAG_context_length < QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE ||
      FLAG_context_length > QWEN_GENERATION_DECODE_CONTEXT_LIMIT ||
      FLAG_context_length % QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--context_length must be a multiple of %d in [%d, %d] while the "
        "decode-attention bring-up adapter is active",
        QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE,
        QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE,
        QWEN_GENERATION_DECODE_CONTEXT_LIMIT);
  }

  iree_status_t status = iree_ok_status();
  iree_tokenizer_t* tokenizer = NULL;
  status = qwen_generation_cli_load_tokenizer(
      iree_make_cstring_view(FLAG_tokenizer), host_allocator, &tokenizer);

  iree_string_builder_t chat_prompt_builder;
  iree_string_builder_initialize(host_allocator, &chat_prompt_builder);
  if (iree_status_is_ok(status)) {
    status = qwen_generation_cli_build_chat_prompt(
        iree_make_cstring_view(FLAG_system),
        iree_make_cstring_view(FLAG_prompt), &chat_prompt_builder);
  }

  iree_tokenizer_token_id_t* prompt_token_ids = NULL;
  iree_host_size_t prompt_token_count = 0;
  if (iree_status_is_ok(status)) {
    status = qwen_generation_cli_encode_prompt(
        tokenizer, iree_string_builder_view(&chat_prompt_builder),
        host_allocator, &prompt_token_ids, &prompt_token_count);
  }

  const iree_host_size_t max_tokens =
      FLAG_max_tokens > 0 ? (iree_host_size_t)FLAG_max_tokens : 0;
  const iree_host_size_t context_capacity =
      FLAG_context_length > 0 ? (iree_host_size_t)FLAG_context_length : 0;
  if (iree_status_is_ok(status) &&
      (prompt_token_count > context_capacity ||
       max_tokens - 1 > context_capacity - prompt_token_count)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "prompt has %" PRIhsz " tokens and generation requests %" PRIhsz
        "; final decode context exceeds --context_length=%" PRIhsz,
        prompt_token_count, max_tokens, context_capacity);
  }

  iree_host_size_t decode_state_storage_size = 0;
  uint8_t* decode_state_storage = NULL;
  iree_tokenizer_decode_state_t* decode_state = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decode_state_calculate_size(
        tokenizer, &decode_state_storage_size);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, decode_state_storage_size,
                                   (void**)&decode_state_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_decode_state_initialize(
        tokenizer, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_byte_span(decode_state_storage, decode_state_storage_size),
        &decode_state);
  }

  qwen_tooling_runtime_context_t runtime_context;
  memset(&runtime_context, 0, sizeof(runtime_context));
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  }

  // Generation profiling includes model residency and program preparation so a
  // single capture describes the complete text-to-text lifecycle.
  iree_hal_profiling_from_flags_t* profiling = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        runtime_context.device_group, host_allocator, &profiling);
  }

  iree_hal_semaphore_t* timeline = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        qwen_tooling_runtime_context_device(&runtime_context),
        IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline);
  }

  qwen_generation_cli_timepoint_t model_ready = {
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
    status = qwen_model_load(&model_options, &parameter_source,
                             iree_hal_semaphore_list_empty(),
                             qwen_generation_cli_timepoint_list(&model_ready),
                             host_allocator, &model);
  }

  qwen_program_t* prefill_program = NULL;
  if (iree_status_is_ok(status)) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_PREFILL;
    program_options.token_count = prompt_token_count;
    program_options.context_count = prompt_token_count;
    program_options.token_capacity = prompt_token_count;
    program_options.context_capacity = context_capacity;
    program_options.command_buffer_mode = runtime_context.command_buffer_mode;
    status = qwen_program_prepare(model, &program_options, host_allocator,
                                  &prefill_program);
  }

  qwen_program_t* decode_programs[QWEN_GENERATION_DECODE_CONTEXT_CLASS_COUNT] =
      {0};
  if (iree_status_is_ok(status) && max_tokens > 1) {
    const iree_host_size_t context_class =
        qwen_generation_cli_decode_context_class(prompt_token_count);
    const iree_host_size_t class_ordinal =
        context_class / QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE - 1;
    status = qwen_generation_cli_prepare_decode_program(
        model, context_class, prompt_token_count, context_capacity,
        runtime_context.command_buffer_mode, host_allocator,
        &decode_programs[class_ordinal]);
  }

  qwen_generation_cli_timepoint_t request_ready = {
      .semaphore = timeline,
      .value = 2,
  };
  qwen_request_t* request = NULL;
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_capacity = prompt_token_count;
    request_options.context_capacity = context_capacity;
    status =
        qwen_request_create(model, &request_options,
                            qwen_generation_cli_timepoint_list(&model_ready),
                            qwen_generation_cli_timepoint_list(&request_ready),
                            host_allocator, &request);
  }

  qwen_generation_cli_timepoint_t tokens_ready = {
      .semaphore = timeline,
      .value = 3,
  };
  if (iree_status_is_ok(status)) {
    status = qwen_request_reset_tokens(
        request, /*context_base=*/0,
        iree_tokenizer_make_token_id_list(prompt_token_ids, prompt_token_count),
        qwen_generation_cli_timepoint_list(&request_ready),
        qwen_generation_cli_timepoint_list(&tokens_ready));
  }

  qwen_generation_cli_timepoint_t issue_complete = {
      .semaphore = timeline,
      .value = 4,
  };
  if (iree_status_is_ok(status)) {
    status =
        qwen_program_issue(prefill_program, request,
                           qwen_generation_cli_timepoint_list(&tokens_ready),
                           qwen_generation_cli_timepoint_list(&issue_complete));
  }

  iree_host_size_t generated_token_count = 0;
  bool reached_end_of_sequence = false;
  for (iree_host_size_t i = 0; i < max_tokens && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_semaphore_wait(
        issue_complete.semaphore, issue_complete.value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
    iree_tokenizer_token_id_t token_id = IREE_TOKENIZER_TOKEN_ID_INVALID;
    if (iree_status_is_ok(status)) {
      status = qwen_request_read_selected_token(request, &token_id);
    }
    if (iree_status_is_ok(status) &&
        (token_id < 0 || token_id >= QWEN_MODEL_VOCABULARY_SIZE)) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "model selected invalid token %" PRId32, token_id);
    }
    if (!iree_status_is_ok(status)) break;

    ++generated_token_count;
    if (FLAG_print_token_ids) {
      fprintf(stderr, "token[%" PRIhsz "]=%" PRId32 "\n",
              generated_token_count - 1, token_id);
    }
    if (token_id == QWEN_MODEL_END_OF_SEQUENCE_TOKEN) {
      reached_end_of_sequence = true;
      break;
    }
    status = qwen_generation_cli_emit_token(decode_state, token_id);
    if (!iree_status_is_ok(status) || i + 1 == max_tokens) break;

    qwen_generation_cli_timepoint_t next_issue_complete = {
        .semaphore = timeline,
        .value = issue_complete.value + 1,
    };
    const iree_host_size_t decode_context_base =
        prompt_token_count + generated_token_count - 1;
    const iree_host_size_t context_class =
        qwen_generation_cli_decode_context_class(decode_context_base);
    const iree_host_size_t class_ordinal =
        context_class / QWEN_GENERATION_DECODE_CONTEXT_CLASS_SIZE - 1;
    if (!decode_programs[class_ordinal]) {
      status = qwen_generation_cli_prepare_decode_program(
          model, context_class, prompt_token_count, context_capacity,
          runtime_context.command_buffer_mode, host_allocator,
          &decode_programs[class_ordinal]);
    }
    if (!iree_status_is_ok(status)) break;
    status = qwen_program_issue(
        decode_programs[class_ordinal], request,
        qwen_generation_cli_timepoint_list(&issue_complete),
        qwen_generation_cli_timepoint_list(&next_issue_complete));
    issue_complete = next_issue_complete;
  }

  if (iree_status_is_ok(status)) {
    status = qwen_generation_cli_finalize_text(decode_state);
  }
  if (iree_status_is_ok(status)) {
    fputc('\n', stdout);
    fprintf(stderr,
            "Qwen generated %" PRIhsz " token%s from %" PRIhsz
            " prompt tokens%s.\n",
            generated_token_count, generated_token_count == 1 ? "" : "s",
            prompt_token_count,
            reached_end_of_sequence ? " through EOS" : " at the token limit");
  }

  status = qwen_generation_cli_wait_for_model_ready_workaround(status, model,
                                                               &model_ready);
  if (profiling) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  }
  qwen_request_release(request);
  qwen_generation_cli_release_decode_programs(decode_programs);
  qwen_program_release(prefill_program);
  qwen_model_release(model);
  iree_hal_semaphore_release(timeline);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  if (decode_state) {
    iree_tokenizer_decode_state_deinitialize(decode_state);
  }
  iree_allocator_free(host_allocator, decode_state_storage);
  iree_allocator_free(host_allocator, prompt_token_ids);
  iree_string_builder_deinitialize(&chat_prompt_builder);
  iree_tokenizer_free(tokenizer);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen-cli", qwen_generation_cli_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_generation_cli_run();
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
