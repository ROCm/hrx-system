// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Backend-independent target-low registry package for compiler infrastructure
// tests. This package intentionally models a tiny synthetic machine so core
// tools can exercise low scheduling/allocation without linking production
// backend descriptor tables.

#ifndef LOOM_TARGET_TEST_LOW_REGISTRY_H_
#define LOOM_TARGET_TEST_LOW_REGISTRY_H_

#include "loom/target/low_descriptor_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

void loom_test_low_descriptor_registry_initialize(
    loom_target_low_descriptor_registry_t* out_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_TEST_LOW_REGISTRY_H_
