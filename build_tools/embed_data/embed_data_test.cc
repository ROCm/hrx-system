// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "build_tools/embed_data/testdata/empty_embed.h"
#include "build_tools/embed_data/testdata/flat_embed.h"
#include "build_tools/embed_data/testdata/path_embed.h"

static int expect_string(const char* label, const char* actual,
                         const char* expected) {
  if (strcmp(actual, expected) == 0) return 0;
  fprintf(stderr, "%s: expected '%s', got '%s'\n", label, expected, actual);
  return 1;
}

static int expect_data(const char* label, const iree_file_toc_t* entry,
                       const char* expected) {
  const size_t expected_length = strlen(expected);
  if (entry->size != expected_length ||
      memcmp(entry->data, expected, expected_length) != 0 ||
      entry->data[entry->size] != '\0') {
    fprintf(stderr, "%s: embedded contents do not match\n", label);
    return 1;
  }
  return 0;
}

static int expect_sentinel(const char* label, const iree_file_toc_t* entry) {
  if (entry->name == NULL && entry->data == NULL && entry->size == 0) return 0;
  fprintf(stderr, "%s: missing sentinel entry\n", label);
  return 1;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  int result = 0;

  const iree_file_toc_t* empty = empty_embed_create();
  result |= expect_sentinel("empty[0]", &empty[0]);
  if (empty_embed_size() != 0) {
    fprintf(stderr, "empty_embed_size: expected 0, got %zu\n",
            empty_embed_size());
    result = 1;
  }

  const iree_file_toc_t* flat = flat_embed_create();
  result |= expect_string("flat[0].name", flat[0].name, "crlf.bin");
  result |= expect_data("flat[0].data", &flat[0],
                        "crlf-line-one\r\ncrlf-line-two\r\n");
  result |= expect_string("flat[1].name", flat[1].name, "first.txt");
  result |= expect_data("flat[1].data", &flat[1], "alpha\n");
  result |= expect_string("flat[2].name", flat[2].name, "second.txt");
  result |= expect_data("flat[2].data", &flat[2], "bravo\n");
  result |= expect_sentinel("flat[3]", &flat[3]);
  if (flat_embed_size() != 3) {
    fprintf(stderr, "flat_embed_size: expected 3, got %zu\n",
            flat_embed_size());
    result = 1;
  }

  const iree_file_toc_t* path = path_embed_create();
  result |= expect_string("path[0].name", path[0].name, "crlf.bin");
  result |= expect_data("path[0].data", &path[0],
                        "crlf-line-one\r\ncrlf-line-two\r\n");
  result |= expect_string("path[1].name", path[1].name, "first.txt");
  result |= expect_data("path[1].data", &path[1], "alpha\n");
  result |= expect_string("path[2].name", path[2].name, "nested/second.txt");
  result |= expect_data("path[2].data", &path[2], "bravo\n");
  result |= expect_sentinel("path[3]", &path[3]);
  if (path_embed_size() != 3) {
    fprintf(stderr, "path_embed_size: expected 3, got %zu\n",
            path_embed_size());
    result = 1;
  }

  return result;
}
