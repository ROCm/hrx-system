// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Observes the source chat/input/completion path without device execution.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loom/serve/service/text/chat.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

IREE_FLAG(string, control, "", "Chat control VM program (.loom or .loombc).");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, request, "", "Chat Completions request JSON file.");
IREE_FLAG(string, options, "{}", "Deployment options JSON object.");
IREE_FLAG(string, response, "",
          "Optional raw generated text file to complete.");
IREE_FLAG(int32_t, capacity, 262144, "Maximum complete prompt token count.");
IREE_FLAG(
    int32_t, response_chunk_size, 0,
    "Incremental decoded output bytes per policy call; zero completes once.");

static iree_string_view_t loom_serve_text_request_check_file_text(
    iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_status_t loom_serve_text_request_check_report(
    const loom_serve_text_chat_t* chat, iree_host_size_t token_count,
    const int32_t* tokens, const loom_serve_text_chat_completion_t* completion,
    iree_string_view_t tool_calls) {
  loom_output_stream_t stream;
  loom_output_stream_for_file(stdout, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("model"), chat->policy->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("prompt"), iree_string_builder_view(&chat->prompt)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("declared_tool_count"), chat->tool_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("tokens")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(&stream, &array));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < token_count && iree_status_is_ok(status);
       ++i) {
    status = loom_json_array_write_int32_element(&array, tokens[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_array_end(&array);
  }
  iree_const_byte_span_t options = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        chat->options, 0, iree_vm_buffer_length(chat->options), &options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(&object, IREE_SV("request_options"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_array_begin(&stream, &array);
  }
  for (iree_host_size_t i = 0;
       i < options.data_length && iree_status_is_ok(status); ++i) {
    status = loom_json_array_write_int32_element(&array, options.data[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_array_end(&array);
  }
  if (iree_status_is_ok(status) && completion) {
    status = loom_json_object_write_host_size_field(
        &object, IREE_SV("output_cursor"), chat->output.cursor);
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &object, IREE_SV("content"),
          iree_string_builder_view(&chat->output.content));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &object, IREE_SV("reasoning_content"),
          iree_string_builder_view(&chat->output.reasoning));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_bool_field(&object, IREE_SV("reusable"),
                                                 completion->reusable);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &object, IREE_SV("checkpoint"), completion->transcript);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_begin_field(&object, IREE_SV("tool_calls"));
    }
    if (iree_status_is_ok(status)) {
      status = loom_output_stream_write(&stream, tool_calls);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_end(&object);
  }
  if (iree_status_is_ok(status)) {
    status = loom_output_stream_write_cstring(&stream, "\n");
  }
  if (iree_status_is_ok(status) && fflush(stdout) != 0) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "failed to flush request observation");
  }
  return status;
}

static iree_status_t loom_serve_text_request_check_run(
    iree_allocator_t host_allocator) {
  loom_serve_compilation_report_t compilation = {0};
  if (!FLAG_control[0] || !FLAG_tokenizer[0] || !FLAG_request[0] ||
      FLAG_capacity <= 0 || FLAG_capacity > 262144 ||
      FLAG_response_chunk_size < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--control, --tokenizer and --request are required; "
        "capacity is [1, 262144]");
  }
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(host_allocator, &environment));
  iree_io_file_contents_t* tokenizer_contents = NULL;
  iree_status_t status =
      iree_io_file_contents_read(iree_make_cstring_view(FLAG_tokenizer),
                                 host_allocator, &tokenizer_contents);
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        loom_serve_text_request_check_file_text(tokenizer_contents),
        host_allocator, &tokenizer);
  }
  iree_io_file_contents_free(tokenizer_contents);
  iree_vm_module_t* libraries[3] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(environment, tokenizer,
                                            host_allocator, &libraries[0]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(environment, host_allocator,
                                           &libraries[1]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_tools_module_create(
        environment, host_allocator, &libraries[2]);
  }
  const iree_string_view_t roots[] = {
      IREE_SVL("model_name"),       IREE_SVL("chat_image_limits"),
      IREE_SVL("chat_begin"),       IREE_SVL("chat_message"),
      IREE_SVL("chat_end"),         IREE_SVL("render_tool"),
      IREE_SVL("parse_tools"),      IREE_SVL("prepare_input"),
      IREE_SVL("output_next"),      IREE_SVL("complete_text"),
      IREE_SVL("request_defaults"), IREE_SVL("request_options")};
  loom_serve_program_t* program = NULL;
  const loom_serve_sources_t sources = {.base_directory = IREE_SV(".")};
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_allocate(
        environment, &sources, iree_make_cstring_view(FLAG_control),
        IREE_ARRAYSIZE(roots), roots, iree_vm_module_span_from_array(libraries),
        &compilation, host_allocator, &program);
  }
  loom_serve_text_chat_policy_t policy = {0};
  iree_vm_buffer_t* options = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(FLAG_options, strlen(FLAG_options)), 1,
        host_allocator, &options);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_policy_initialize(environment, program,
                                                    options, &policy);
  }
  iree_vm_buffer_release(options);
  iree_io_file_contents_t* request = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_read(iree_make_cstring_view(FLAG_request),
                                        host_allocator, &request);
  }
  loom_serve_text_chat_t chat = {0};
  bool chat_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_initialize(
        &policy, policy.name, loom_serve_text_request_check_file_text(request),
        1, 128, host_allocator, &chat);
    chat_initialized = iree_status_is_ok(status);
  }
  int32_t* tokens = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, FLAG_capacity,
                                         sizeof(*tokens), (void**)&tokens);
  }
  iree_host_size_t token_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_prepare_input(
        &policy, iree_string_builder_view(&chat.prompt),
        LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
        LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, FLAG_capacity, host_allocator,
        tokens, &token_count);
  }
  iree_io_file_contents_t* response = NULL;
  if (iree_status_is_ok(status) && FLAG_response[0]) {
    status = iree_io_file_contents_read(iree_make_cstring_view(FLAG_response),
                                        host_allocator, &response);
  }
  iree_string_builder_t tool_calls;
  iree_string_builder_initialize(host_allocator, &tool_calls);
  loom_serve_text_chat_completion_t completion = {0};
  if (iree_status_is_ok(status) && response && FLAG_response_chunk_size) {
    const iree_string_view_t text =
        loom_serve_text_request_check_file_text(response);
    for (iree_host_size_t length = 0;
         length < text.size && iree_status_is_ok(status);) {
      length += iree_min((iree_host_size_t)FLAG_response_chunk_size,
                         text.size - length);
      loom_serve_text_chat_output_t output;
      do {
        status = loom_serve_text_chat_output_next(
            &chat, iree_string_view_substr(text, 0, length),
            LOOM_SERVE_TEXT_CHAT_OUTPUT_STREAMING, &output);
      } while (iree_status_is_ok(status) && output.advanced);
    }
  }
  if (iree_status_is_ok(status) && response) {
    status = loom_serve_text_chat_complete(
        &chat, loom_serve_text_request_check_file_text(response), 1,
        &tool_calls, &completion);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_request_check_report(
        &chat, token_count, tokens, response ? &completion : NULL,
        iree_string_builder_view(&tool_calls));
  }
  loom_serve_text_chat_completion_deinitialize(&completion);
  iree_string_builder_deinitialize(&tool_calls);
  iree_io_file_contents_free(response);
  iree_allocator_free(host_allocator, tokens);
  if (chat_initialized) {
    loom_serve_text_chat_deinitialize(&chat);
  }
  iree_io_file_contents_free(request);
  loom_serve_text_chat_policy_deinitialize(&policy);
  loom_serve_program_free(program);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(libraries); ++i) {
    iree_vm_module_release(libraries[i]);
  }
  iree_tokenizer_free(tokenizer);
  iree_vm_environment_free(environment);
  loom_output_stream_t output;
  loom_output_stream_for_file(stderr, &output);
  status = iree_status_join(
      status, loom_serve_compilation_write(&compilation, &output));
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status =
      loom_serve_text_request_check_run(iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
