// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command program preparation products exposed to loom-check.

#ifndef LOOM_TARGET_ARCH_CMD_CHECK_PROGRAM_PLAN_H_
#define LOOM_TARGET_ARCH_CMD_CHECK_PROGRAM_PLAN_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emit provider for prepared portable command programs.
extern const loom_check_emit_provider_t
    loom_cmd_program_plan_check_emit_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_CHECK_PROGRAM_PLAN_H_
