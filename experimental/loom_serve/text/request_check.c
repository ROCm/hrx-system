// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Observes the actual source chat/input/completion path without GPU residency.
#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/json.h"
#include "experimental/loom_serve/text/chat.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

IREE_FLAG(string, model, "",
          "Source package directory containing control.loom.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, request, "", "OpenAI streaming request JSON file.");
IREE_FLAG(string, response, "",
          "Optional raw generated text file to complete.");
IREE_FLAG(int32_t, capacity, 262144, "Maximum complete prompt token count.");

static iree_string_view_t file_text(iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_status_t report(const loom_serve_text_chat_t* chat,
                            iree_host_size_t token_count, const int32_t* tokens,
                            const loom_serve_text_chat_completion_t* completion,
                            iree_string_view_t tool_calls) {
  loom_output_stream_t stream;
  loom_output_stream_for_file(stdout, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("model"), chat->policy->name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("prompt"), iree_string_builder_view(&chat->prompt)));
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
  if (iree_status_is_ok(status) && completion) {
    status = loom_json_object_write_host_size_field(
        &object, IREE_SV("text_end"), completion->text_end);
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

static iree_status_t run(iree_allocator_t allocator) {
  if (!FLAG_model[0] || !FLAG_tokenizer[0] || !FLAG_request[0] ||
      FLAG_capacity <= 0 || FLAG_capacity > 262144) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--model, --tokenizer and --request are required; "
                            "capacity is [1, 262144]");
  }
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(allocator, &environment));
  iree_io_file_contents_t* tokenizer_contents = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_tokenizer), allocator, &tokenizer_contents);
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(file_text(tokenizer_contents),
                                                  allocator, &tokenizer);
  }
  iree_io_file_contents_free(tokenizer_contents);
  iree_vm_module_t* libraries[3] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(environment, tokenizer,
                                            &libraries[0], allocator);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_json_module_create(environment, &libraries[1], allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_tools_module_create(environment,
                                                      &libraries[2], allocator);
  }
  char* path = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_file_path_join(iree_make_cstring_view(FLAG_model),
                                 IREE_SV("control.loom"), allocator, &path);
  }
  const iree_string_view_t roots[] = {
      IREE_SVL("model_name"),    IREE_SVL("chat_begin"),
      IREE_SVL("chat_message"),  IREE_SVL("chat_end"),
      IREE_SVL("render_tool"),   IREE_SVL("parse_tools"),
      IREE_SVL("prepare_input"), IREE_SVL("text_end"),
      IREE_SVL("complete_text")};
  loom_serve_program_t* program = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_create(
        environment, iree_make_cstring_view(path), IREE_ARRAYSIZE(roots), roots,
        iree_vm_module_span_from_array(libraries), allocator, &program);
  }
  iree_allocator_free(allocator, path);
  loom_serve_text_chat_policy_t policy = {0};
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_text_chat_policy_initialize(environment, program, &policy);
  }
  iree_io_file_contents_t* request = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_read(iree_make_cstring_view(FLAG_request),
                                        allocator, &request);
  }
  loom_serve_text_chat_t chat = {0};
  bool chat_initialized = false;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_initialize(&policy, file_text(request), 128,
                                             allocator, &chat);
    chat_initialized = iree_status_is_ok(status);
  }
  int32_t* tokens = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(allocator, FLAG_capacity,
                                         sizeof(*tokens), (void**)&tokens);
  }
  iree_host_size_t token_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_prepare_input(
        &policy, iree_string_builder_view(&chat.prompt),
        LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
        LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, FLAG_capacity, tokens,
        &token_count, allocator);
  }
  iree_io_file_contents_t* response = NULL;
  if (iree_status_is_ok(status) && FLAG_response[0]) {
    status = iree_io_file_contents_read(iree_make_cstring_view(FLAG_response),
                                        allocator, &response);
  }
  iree_string_builder_t tool_calls;
  iree_string_builder_initialize(allocator, &tool_calls);
  loom_serve_text_chat_completion_t completion = {0};
  if (iree_status_is_ok(status) && response) {
    status = loom_serve_text_chat_complete(&chat, file_text(response), 0, 1,
                                           &tool_calls, &completion);
  }
  if (iree_status_is_ok(status)) {
    status = report(&chat, token_count, tokens, response ? &completion : NULL,
                    iree_string_builder_view(&tool_calls));
  }
  loom_serve_text_chat_completion_deinitialize(&completion);
  iree_string_builder_deinitialize(&tool_calls);
  iree_io_file_contents_free(response);
  iree_allocator_free(allocator, tokens);
  if (chat_initialized) {
    loom_serve_text_chat_deinitialize(&chat);
  }
  iree_io_file_contents_free(request);
  loom_serve_text_chat_policy_deinitialize(&policy);
  loom_serve_program_destroy(program);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(libraries); ++i) {
    iree_vm_module_release(libraries[i]);
  }
  iree_tokenizer_free(tokenizer);
  iree_vm_environment_free(environment);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status = run(iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
