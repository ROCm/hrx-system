// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom-check provider for the LLVMIR target provider.

#ifndef LOOM_TARGET_ARCH_LLVMIR_CHECK_PROVIDER_H_
#define LOOM_TARGET_ARCH_LLVMIR_CHECK_PROVIDER_H_

#include "loom/tools/loom-check/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Check provider that contributes the LLVMIR target provider.
extern const loom_check_provider_t loom_llvmir_target_check_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_LLVMIR_CHECK_PROVIDER_H_
