// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared command-line flags for Loom compile requests.

#ifndef LOOM_TOOLING_COMPILE_REQUEST_FLAGS_H_
#define LOOM_TOOLING_COMPILE_REQUEST_FLAGS_H_

#include "iree/base/api.h"
#include "loom/tooling/compile/request.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the shared --product, --format, and --target constraints with
// caller-selected roots. All returned string views borrow flag storage.
loom_compile_request_options_t loom_compile_request_options_from_flags(
    iree_string_view_list_t roots);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_REQUEST_FLAGS_H_
