// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_RESULT_STORAGE_H_
#define LOOMC_RESULT_STORAGE_H_

#include "loomc/context.h"
#include "loomc/result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an empty result with a terminal code and source retention policy.
LOOMC_API_PRIVATE loomc_status_t loomc_result_create(
    loomc_status_code_t status_code, loomc_source_retention_t source_retention,
    loomc_allocator_t allocator, loomc_result_t** out_result);

// Returns the allocator owned by result.
LOOMC_API_PRIVATE loomc_allocator_t
loomc_result_allocator(const loomc_result_t* result);

// Returns the diagnostic source retention policy owned by result.
LOOMC_API_PRIVATE loomc_source_retention_t
loomc_result_source_retention(const loomc_result_t* result);

// Sets the terminal code while the result is still being built. Diagnostic-only
// failures use FAILED_PRECONDITION; status normalization preserves its code.
LOOMC_API_PRIVATE void loomc_result_set_status_code(
    loomc_result_t* result, loomc_status_code_t status_code);

// Adds a copied diagnostic to result.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_diagnostic(
    loomc_result_t* result, const loomc_diagnostic_t* diagnostic);

// Adds an artifact by copying metadata and retaining immutable contents.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_artifact(
    loomc_result_t* result, const loomc_artifact_t* artifact);

// Adds an artifact whose contents storage is transferred on success.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_artifact_take_contents(
    loomc_result_t* result, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    loomc_byte_span_t contents);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_RESULT_STORAGE_H_
