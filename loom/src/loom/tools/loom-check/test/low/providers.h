// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only loom-check queries over Low compiler state.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_LOW_PROVIDERS_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_LOW_PROVIDERS_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reports focused allocation decisions for authored Low values.
extern const loom_check_emit_provider_t loom_check_test_low_allocation_provider;

// Reports focused schedule relationships for authored Low operations.
extern const loom_check_emit_provider_t loom_check_test_low_schedule_provider;

// Reports complete scheduling, allocation, packet, and report observations for
// authored Low compiler tests.
extern const loom_check_emit_provider_t
    loom_check_test_low_observation_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_LOW_PROVIDERS_H_
