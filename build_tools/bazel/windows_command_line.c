// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/bazel/windows_command_line.h"

#include <stdint.h>
#include <stdlib.h>

wchar_t* iree_bazel_quote_windows_argument(const wchar_t* argument) {
  size_t length = wcslen(argument);
  if (length > (SIZE_MAX / sizeof(wchar_t) - 3) / 2) {
    return NULL;
  }
  wchar_t* quoted = (wchar_t*)malloc((length * 2 + 3) * sizeof(wchar_t));
  if (!quoted) {
    return NULL;
  }
  wchar_t* output = quoted;
  *output++ = L'"';
  for (;;) {
    size_t backslashes = 0;
    while (*argument == L'\\') {
      ++backslashes;
      ++argument;
    }
    // A quote consumes one escape slash; every literal slash immediately
    // before that quote (including the closing delimiter) must be doubled.
    if (*argument == L'"' || *argument == L'\0') {
      backslashes *= 2;
    }
    while (backslashes > 0) {
      *output++ = L'\\';
      --backslashes;
    }
    if (*argument == L'\0') {
      break;
    }
    if (*argument == L'"') {
      *output++ = L'\\';
    }
    *output++ = *argument++;
  }
  *output++ = L'"';
  *output = L'\0';
  return quoted;
}
