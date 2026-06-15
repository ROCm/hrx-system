// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_CHECK_REQUIREMENTS_H_
#define LOOM_TOOLS_LOOM_CHECK_REQUIREMENTS_H_

#include "iree/base/api.h"
#include "loom/tools/loom-check/check.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Checks a test case's // REQUIRES declarations before executing its IR body.
//
// Unavailable declared requirements set |result| to SKIP and return OK with
// |out_continue_execution| false. Harness errors, such as unknown requirement
// names or missing declarations for external emit tools, set |result| to FAIL
// and return OK with |out_continue_execution| false. These are final outcomes:
// callers must not apply XFAIL inversion after this helper stops execution.
iree_status_t loom_check_preflight_requirements(
    const loom_check_case_t* test_case,
    const loom_check_environment_t* environment, iree_allocator_t allocator,
    loom_check_result_t* result, bool* out_continue_execution);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TOOLS_LOOM_CHECK_REQUIREMENTS_H_
