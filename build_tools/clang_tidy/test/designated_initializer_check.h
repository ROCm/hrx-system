// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_CLANG_TIDY_TEST_DESIGNATED_INITIALIZER_CHECK_H_
#define IREE_BUILD_TOOLS_CLANG_TIDY_TEST_DESIGNATED_INITIALIZER_CHECK_H_

struct HeaderConfig {
  int value;
};

inline HeaderConfig HeaderValue() { return {/*.value=*/1}; }

#endif  // IREE_BUILD_TOOLS_CLANG_TIDY_TEST_DESIGNATED_INITIALIZER_CHECK_H_
