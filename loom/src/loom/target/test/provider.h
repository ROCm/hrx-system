// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Synthetic target provider for loom-check and target-independent tests.

#ifndef LOOM_TARGET_TEST_PROVIDER_H_
#define LOOM_TARGET_TEST_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const loom_target_provider_t loom_test_target_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_PROVIDER_H_
