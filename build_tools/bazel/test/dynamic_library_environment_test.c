// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int verify_file_contents(const char* path, const char* expected,
                                const char* description) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "unable to open %s: %s\n", description, path);
    return 1;
  }

  char contents[64] = {0};
  size_t length = fread(contents, 1, sizeof(contents) - 1, file);
  if (ferror(file)) {
    fprintf(stderr, "unable to read %s: %s\n", description, path);
    fclose(file);
    return 1;
  }
  fclose(file);

  if (length != strlen(expected) || strcmp(contents, expected) != 0) {
    fprintf(stderr, "unexpected %s contents: %s\n", description, contents);
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  const char* shard_status_path = getenv("TEST_SHARD_STATUS_FILE");
  if (shard_status_path) {
    FILE* shard_status_file = fopen(shard_status_path, "wb");
    if (!shard_status_file) {
      fprintf(stderr, "unable to advertise test sharding support: %s\n",
              shard_status_path);
      return 1;
    }
    fclose(shard_status_file);
  }

  const char* source_environment = getenv("IREE_TEST_SOURCE_ENV");
  if (!source_environment || strcmp(source_environment, "source") != 0) {
    fprintf(stderr, "IREE_TEST_SOURCE_ENV was not preserved\n");
    return 1;
  }

  const char* path = getenv("IREE_TEST_DYNAMIC_LIBRARY_PATH");
  if (!path) {
    fprintf(stderr, "IREE_TEST_DYNAMIC_LIBRARY_PATH is not set\n");
    return 1;
  }
  if (verify_file_contents(path, "runtime root fixture\n",
                           "dynamic-library environment runfile") != 0) {
    return 1;
  }

  if (argc != 2) {
    fprintf(stderr, "expected one runfile argument, got %d\n", argc - 1);
    return 1;
  }
  if (verify_file_contents(argv[1], "runtime dependency fixture\n",
                           "dynamic-library argument runfile") != 0) {
    return 1;
  }
  return 0;
}
