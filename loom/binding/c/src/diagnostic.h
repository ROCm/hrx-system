// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_DIAGNOSTIC_STORAGE_H_
#define LOOMC_DIAGNOSTIC_STORAGE_H_

#include "loom/error/diagnostic.h"
#include "loom/ir/module.h"
#include "loomc/diagnostic.h"
#include "result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Adds a rendered Loom diagnostic to result.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_loom_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    const loom_diagnostic_t* diagnostic);

// Materializes and adds a Loom diagnostic emission to result.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_loom_diagnostic_emission(
    loomc_result_t* result, const loomc_source_t* source,
    loom_emitter_t emitter, const loom_diagnostic_emission_t* emission);

// Verifies a Loom module and adds verifier diagnostics to result.
LOOMC_API_PRIVATE loomc_status_t loomc_result_verify_loom_module(
    const loom_module_t* module, const loomc_source_t* source,
    loomc_result_t* result);

// Adds a rendered status as a result diagnostic without consuming status.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status);

// Returns true when status should be represented as an operation diagnostic.
LOOMC_API_PRIVATE bool loomc_status_is_result_diagnostic(loomc_status_t status);

// Adds a status diagnostic and marks the result failed.
LOOMC_API_PRIVATE loomc_status_t loomc_result_fail_status_diagnostic(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status);

// Adds a status diagnostic, marks the result failed, and frees status.
LOOMC_API_PRIVATE loomc_status_t loomc_result_fail_status_diagnostic_consume(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_diagnostic_severity_t severity, loomc_string_view_t code,
    loomc_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_DIAGNOSTIC_STORAGE_H_
