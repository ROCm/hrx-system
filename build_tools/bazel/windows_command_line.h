// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_BAZEL_WINDOWS_COMMAND_LINE_H_
#define IREE_BUILD_TOOLS_BAZEL_WINDOWS_COMMAND_LINE_H_

#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

// Encodes one non-NULL, NUL-terminated argument for Windows CRT command-line
// parsing. The caller owns the returned allocation and releases it with free.
// Returns NULL if the required allocation cannot be represented or obtained.
wchar_t* iree_bazel_quote_windows_argument(const wchar_t* argument);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BUILD_TOOLS_BAZEL_WINDOWS_COMMAND_LINE_H_
