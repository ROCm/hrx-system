// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_CLI_HELP_H_
#define LOOM_TOOLING_CLI_HELP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Filters normal `--help` output to Loom-owned CLI flags and the base
// help/flagfile controls. `--help=all` still prints the full linked flag
// registry.
void loom_tooling_cli_set_default_help_filter(void);

// Returns true when |arg| is the conventional agent-facing Markdown help flag.
bool loom_tooling_cli_is_agents_markdown_arg(const char* arg);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TOOLING_CLI_HELP_H_
