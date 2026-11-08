// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Controlled scenario execution and requirement providers for native tests.

#ifndef LOOM_TOOLS_LOOM_TEST_SCENARIO_TEST_PROVIDER_H_
#define LOOM_TOOLS_LOOM_TEST_SCENARIO_TEST_PROVIDER_H_

#include "loom/tools/loom-test/main.h"

#ifdef __cplusplus
extern "C" {
#endif

// Installs deterministic failing target, passing oracle, and requirement
// providers into |configuration| while preserving its other fields.
void loom_test_scenario_test_provider_configure(
    loom_test_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_TEST_SCENARIO_TEST_PROVIDER_H_
