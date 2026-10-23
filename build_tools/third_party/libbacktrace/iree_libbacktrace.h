// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_
#define IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_

// The pinned implementation has private symbols. An explicitly supplied CMake
// package retains its own public ABI and does not set this definition. This
// wrapper is used only by the formatter, not by IREE's public headers.
#if defined(IREE_LIBBACKTRACE_PRIVATE)
#include "symbol_prefix.h"
#endif
#include <backtrace.h>

#endif  // IREE_BUILD_TOOLS_THIRD_PARTY_LIBBACKTRACE_IREE_LIBBACKTRACE_H_
