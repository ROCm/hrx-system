// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PASS_PROGRAM_STORAGE_H_
#define LOOMC_PASS_PROGRAM_STORAGE_H_

#include "loom/pass/program.h"
#include "loomc/pass.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the context retained by the public pass program handle.
LOOMC_API_PRIVATE loomc_context_t* loomc_pass_program_context(
    const loomc_pass_program_t* pass_program);

// Returns the immutable Loom pass program owned by the public handle.
LOOMC_API_PRIVATE const loom_pass_program_t*
loomc_pass_program_loom_pass_program(const loomc_pass_program_t* pass_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PASS_PROGRAM_STORAGE_H_
