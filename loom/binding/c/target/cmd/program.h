// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_TARGET_CMD_PROGRAM_H_
#define LOOM_BINDING_C_TARGET_CMD_PROGRAM_H_

#include "loom/target/arch/cmd/package.h"
#include "loomc/target/cmd/program.h"

// Returns the validated package view owned by |package|.
const loom_cmd_program_package_t* loomc_cmd_program_package_parsed(
    const loomc_cmd_program_package_t* package);

// Resolves one package-local export token, or returns NULL when invalid.
const loom_cmd_program_package_export_t*
loomc_cmd_program_package_resolve_export(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export);

#endif  // LOOM_BINDING_C_TARGET_CMD_PROGRAM_H_
