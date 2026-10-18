// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PASS_RUN_H_
#define LOOMC_PASS_RUN_H_

#include "loom/pass/environment.h"
#include "loomc/pass.h"
#include "option_chain.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Executes an admitted module using its context and retained compiler facts.
// The caller has validated public descriptors and context compatibility and
// owns terminal failure cleanup. This operation appends to the caller's result
// and invalidates target verification after the interpreter mutates the IR.
LOOMC_API_PRIVATE loomc_status_t loomc_pass_program_run_into_result(
    const loomc_pass_program_t* pass_program, loomc_workspace_t* workspace,
    loomc_module_t* module,
    const loom_pass_environment_capability_t* launch_config_capability,
    const loomc_option_chain_t* options, loomc_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PASS_RUN_H_
