// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-tokenizer qualification boundary; production consumes spans in memory.

#include <stdio.h>
#include <stdlib.h>

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
  loom_serve_krea2_request_t* request = NULL;
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
    status = loom_serve_krea2_request_create(
        tokenizer, options, iree_make_cstring_view(FLAG_prompt), allocator,
        &request);
  }
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
  return EXIT_SUCCESS;
}
