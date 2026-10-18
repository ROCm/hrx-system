// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_SERVE_ACTIVITY_TEST_H_
#define LOOM_TOOLS_LOOM_SERVE_ACTIVITY_TEST_H_

#include "iree/hal/driver_registry.h"
#include "loom/serve/runtime/host/host.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs the native activity/recognition suite with the executable's exact
// compiler and runtime-driver composition. Options and registry remain borrowed
// until the call returns; one application owns every test's shared resources.
int loom_serve_activity_test_main(
    int argc, char** argv,
    const loom_serve_compiler_options_t* compiler_options,
    iree_hal_driver_registry_t* driver_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_SERVE_ACTIVITY_TEST_H_
