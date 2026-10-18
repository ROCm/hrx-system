// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_TEST_IR_PROVIDER_H_
#define LOOM_TARGET_TEST_IR_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Synthetic target IR for tools that read, verify, and print test modules.
// Supplies the same dialect, descriptors, and facts as the compiler provider
// without its test passes or lowering policies. Select one of the two providers
// when composing a target environment; their IR contributions overlap.
extern const loom_target_provider_t loom_test_ir_target_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_IR_PROVIDER_H_
