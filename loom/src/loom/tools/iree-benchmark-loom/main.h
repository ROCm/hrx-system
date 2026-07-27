// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-benchmark-loom command-line implementation.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_MAIN_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_MAIN_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/configuration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Runs the configured iree-benchmark-loom command-line tool.
int iree_benchmark_loom_main(
    int argc, char** argv,
    const iree_benchmark_loom_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_MAIN_H_
