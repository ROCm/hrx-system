// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL testbench providers selected by the Loom build configuration.

#ifndef LOOM_TOOLING_EXECUTION_CONFIGURED_TESTBENCH_H_
#define LOOM_TOOLING_EXECUTION_CONFIGURED_TESTBENCH_H_

#include "loom/tooling/execution/hal/testbench_requirement_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the immutable configured HAL requirement provider initializers.
const loom_run_hal_testbench_requirement_initializer_set_t*
loom_tooling_configured_testbench_requirement_initializers(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_CONFIGURED_TESTBENCH_H_
