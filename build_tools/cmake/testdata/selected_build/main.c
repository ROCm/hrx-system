// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

int fixture_value(void);

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  FILE* input = fopen(argv[1], "r");
  if (!input) {
    return 2;
  }
  int expected_value = 0;
  int count = fscanf(input, "%d", &expected_value);
  fclose(input);
  if (count != 1) {
    return 2;
  }
  const char* temporary_directory = getenv("TEST_TMPDIR");
  if (!temporary_directory) {
    return 2;
  }
  if (getenv("IREE_CTEST_SKIP_GLOBAL_TMPDIR_CLEANUP")) {
    return 4;
  }
  char temporary_marker[4096];
  int temporary_marker_length =
      snprintf(temporary_marker, sizeof(temporary_marker), "%s/ran",
               temporary_directory);
  if (temporary_marker_length < 0 ||
      (size_t)temporary_marker_length >= sizeof(temporary_marker)) {
    return 2;
  }
  FILE* stale_marker = fopen(temporary_marker, "r");
  if (stale_marker) {
    fclose(stale_marker);
    return 3;
  }
  FILE* fresh_marker = fopen(temporary_marker, "w");
  if (!fresh_marker || fclose(fresh_marker) != 0) {
    return 2;
  }
  FILE* marker = fopen(argv[2], "a");
  if (!marker) {
    return 2;
  }
  fputs("run\n", marker);
  if (fclose(marker) != 0) {
    return 2;
  }
  return fixture_value() == expected_value ? 0 : 1;
}
