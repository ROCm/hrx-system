// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only loom-check adapters for configuration behavior.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_CONFIG_PROVIDERS_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_CONFIG_PROVIDERS_H_

#include "loom/tools/loom-check/test/native_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Applies one textual binding and prints its event, summary, and resulting IR.
extern const loom_check_test_native_emit_provider_t
    loom_check_test_config_materialize_provider;

// Prints the complete structured configuration schema for a module.
extern const loom_check_test_native_emit_provider_t
    loom_check_test_config_schema_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_CONFIG_PROVIDERS_H_
