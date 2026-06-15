// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Synthetic loom-check provider used by checked-in .loom-test files.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_PROVIDER_H_

#include "loom/tools/loom-check/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_check_provider_t loom_check_test_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_PROVIDER_H_
