// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Human-facing help text for iree-benchmark-loom.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_HELP_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_HELP_H_

#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints the compact AGENTS.md integration guide to |file|.
void iree_benchmark_loom_print_agents_md(FILE* file);

// Installs the detailed command-line usage text for the IREE flag parser.
void iree_benchmark_loom_set_usage(const char* tool_name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_HELP_H_
