// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_EMIT_PRIVATE_H_
#define LOOMC_EMIT_PRIVATE_H_

#include "loomc/emit.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends ordinary module emission to an existing successful result.
//
// The target emitter is selected from |options->artifact_format| through the
// same target-environment policy as loomc_emit_module. The result owns the
// allocator used for all appended artifacts and diagnostics. On successful
// emission |out_primary_artifact_index| identifies the primary artifact added
// before any emitter sidecars or reports.
LOOMC_API_PRIVATE loomc_status_t loomc_emit_module_append(
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module,
    const loomc_emit_options_t* options, loomc_result_t* result,
    loomc_host_size_t* out_primary_artifact_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_EMIT_PRIVATE_H_
