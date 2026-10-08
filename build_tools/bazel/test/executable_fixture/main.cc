// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "build_tools/bazel/test/executable_fixture/library.h"

int main(int argc, char** argv) {
  const char* const expected[] = {
      "argument with spaces",
      "",
      "embedded\"quote",
      "trailing\\",
  };
  const int expected_count = sizeof(expected) / sizeof(expected[0]);
  bool arguments_match = argc == 1 || argc == expected_count + 1;
  for (int index = 1; index < argc && arguments_match; ++index) {
    arguments_match = strcmp(argv[index], expected[index - 1]) == 0;
  }
  if (!arguments_match) {
    fprintf(stderr,
            "expected no arguments or %d exact fixture arguments; got %d:\n",
            expected_count, argc - 1);
    for (int index = 1; index < argc; ++index) {
      fprintf(stderr, "  [%s]\n", argv[index]);
    }
    return 2;
  }
  return executable_fixture_value() == 42 ? 0 : 1;
}
