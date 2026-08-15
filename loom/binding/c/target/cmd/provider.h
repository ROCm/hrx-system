// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_PROVIDER_H_
#define LOOMC_TARGET_CMD_PROVIDER_H_

#include "loom/binding/c/src/program_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Command-program provider package linked beside target-family providers.
LOOMC_API_PRIVATE extern const loomc_program_provider_set_t
    loomc_cmd_program_provider_set;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_PROVIDER_H_
