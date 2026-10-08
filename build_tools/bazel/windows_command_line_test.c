// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/bazel/windows_command_line.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const struct {
    // Original argument received by the launcher.
    const wchar_t* argument;
    // Command-line spelling consumed by the child's Windows CRT.
    const wchar_t* quoted;
  } cases[] = {
      {L"", L"\"\""},
      {L"simple", L"\"simple\""},
      {L"argument with spaces", L"\"argument with spaces\""},
      {L"a\tb", L"\"a\tb\""},
      {L"a\"b", L"\"a\\\"b\""},
      {L"a\\\"b", L"\"a\\\\\\\"b\""},
      {L"a\\b", L"\"a\\b\""},
      {L"tail\\", L"\"tail\\\\\""},
      {L"tail\\\\", L"\"tail\\\\\\\\\""},
      {L"caf\u00e9", L"\"caf\u00e9\""},
  };
  int failures = 0;
  for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
    wchar_t* quoted = iree_bazel_quote_windows_argument(cases[index].argument);
    if (!quoted || wcscmp(quoted, cases[index].quoted) != 0) {
      fwprintf(stderr, L"case %zu: expected [%ls], got [%ls]\n", index,
               cases[index].quoted, quoted ? quoted : L"<allocation failed>");
      ++failures;
    }
    free(quoted);
  }
  return failures != 0;
}
