// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_
#define IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_

// Re-exports the BCR libbacktrace header through this package so Bazel layering
// checks can see the direct dependency that provides it.
#include <backtrace.h>

#endif  // IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_

