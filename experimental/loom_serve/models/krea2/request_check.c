// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-tokenizer qualification boundary; production consumes spans in memory.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/models/krea2/request.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

IREE_FLAG(string, tokenizer, "", "Real model tokenizer.json.");
IREE_FLAG(string, prompt, "", "Prompt to encode.");
IREE_FLAG(string, seed, "0", "Unsigned decimal 64-bit noise seed.");
IREE_FLAG(int32_t, height, 384, "Output height in pixels.");
IREE_FLAG(int32_t, width, 384, "Output width in pixels.");
IREE_FLAG(int32_t, text_tokens, 512, "Retained text extent.");
IREE_FLAG(float, strength, 1.0f, "LoRA strength.");
IREE_FLAG(string, output, "", "Existing directory for request input files.");

static iree_status_t krea2_check_materialization(
    const loom_serve_krea2_prompt_t* prompt,
    loom_serve_krea2_request_options_t options,
    loom_serve_krea2_request_t** out_request, iree_allocator_t allocator) {
  *out_request = NULL;
  if (loom_serve_krea2_prompt_token_count(prompt) > 45) {
    loom_serve_krea2_request_options_t small = options;
    small.text_tokens = 16;
    loom_serve_krea2_request_t* rejected = NULL;
    iree_status_t status =
        loom_serve_krea2_request_create(prompt, small, &rejected, allocator);
    const bool rejected_without_ownership =
        iree_status_code(status) == IREE_STATUS_INVALID_ARGUMENT && !rejected;
    loom_serve_krea2_request_destroy(rejected);
    if (!rejected_without_ownership) {
      return iree_status_join(
          iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "non-fitting prompt did not reject without ownership"),
          status);
    }
    // The expected failure is observed at this qualification boundary.
    iree_status_free(status);
  }
  loom_serve_krea2_request_t* request = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_krea2_request_create(prompt, options, &request, allocator));
  loom_serve_krea2_request_t* repeated = NULL;
  iree_status_t status =
      loom_serve_krea2_request_create(prompt, options, &repeated, allocator);
  for (int i = 0; i < LOOM_SERVE_KREA2_INPUT_COUNT && iree_status_is_ok(status);
       ++i) {
    const iree_const_byte_span_t first =
        loom_serve_krea2_request_input(request, (loom_serve_krea2_input_t)i);
    const iree_const_byte_span_t second =
        loom_serve_krea2_request_input(repeated, (loom_serve_krea2_input_t)i);
    if (first.data_length != second.data_length ||
        memcmp(first.data, second.data, first.data_length)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "repeated materialization changed input %d", i);
    }
  }
  loom_serve_krea2_request_destroy(repeated);
  if (iree_status_is_ok(status)) {
    *out_request = request;
  } else {
    loom_serve_krea2_request_destroy(request);
  }
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_allocator_t allocator = iree_allocator_system();
  loom_serve_krea2_request_options_t options = {
      (uint32_t)FLAG_height, (uint32_t)FLAG_width, (uint32_t)FLAG_text_tokens,
      0, FLAG_strength};
  iree_status_t status =
      iree_json_parse_uint64(iree_make_cstring_view(FLAG_seed), &options.seed);
  iree_io_file_contents_t* contents = NULL;
  iree_tokenizer_t* tokenizer = NULL;
  loom_serve_krea2_prompt_t* prompt = NULL;
  loom_serve_krea2_request_t* request = NULL;
  uint32_t token_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_read(iree_make_cstring_view(FLAG_tokenizer),
                                        allocator, &contents);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        allocator, &tokenizer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_krea2_prompt_create(tokenizer, options.text_tokens,
                                            iree_make_cstring_view(FLAG_prompt),
                                            &prompt, allocator);
  }
  if (iree_status_is_ok(status)) {
    token_count = loom_serve_krea2_prompt_token_count(prompt);
    status = krea2_check_materialization(prompt, options, &request, allocator);
  }
  loom_serve_krea2_prompt_destroy(prompt);
  for (int i = 0; i < LOOM_SERVE_KREA2_INPUT_COUNT && iree_status_is_ok(status);
       ++i) {
    char name[32];
    snprintf(name, sizeof(name), "input-%d", i);
    char* path = NULL;
    status =
        iree_file_path_join(iree_make_cstring_view(FLAG_output),
                            iree_make_cstring_view(name), allocator, &path);
    if (iree_status_is_ok(status)) {
      const iree_const_byte_span_t bytes =
          loom_serve_krea2_request_input(request, (loom_serve_krea2_input_t)i);
      status = iree_io_file_contents_write(iree_make_cstring_view(path), bytes,
                                           allocator);
    }
    iree_allocator_free(allocator, path);
  }
  loom_serve_krea2_request_destroy(request);
  iree_tokenizer_free(tokenizer);
  iree_io_file_contents_free(contents);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  printf(
      "{\"text_tokens\":%u,\"prefix_prompt_tokens\":%u,"
      "\"non_fit_checked\":%s,\"exact_materializations\":2}\n",
      options.text_tokens, token_count, token_count > 45 ? "true" : "false");
  return EXIT_SUCCESS;
}
