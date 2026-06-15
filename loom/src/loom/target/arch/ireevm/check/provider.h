// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check provider for IREE VM target-low tests.

#ifndef LOOM_TARGET_ARCH_IREEVM_CHECK_PROVIDER_H_
#define LOOM_TARGET_ARCH_IREEVM_CHECK_PROVIDER_H_

#include "loom/tools/loom-check/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// IREE VM target-low descriptor and lowering package.
extern const loom_check_provider_t loom_ireevm_check_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_IREEVM_CHECK_PROVIDER_H_
