// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_RESULT_STORAGE_H_
#define LOOMC_RESULT_STORAGE_H_

#include "loomc/result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an empty result with state.
LOOMC_API_PRIVATE loomc_status_t
loomc_result_create(loomc_result_state_t state, loomc_allocator_t allocator,
                    loomc_result_t** out_result);

// Returns the allocator owned by result.
LOOMC_API_PRIVATE loomc_allocator_t
loomc_result_allocator(const loomc_result_t* result);

// Sets the result state while the result is still being built.
LOOMC_API_PRIVATE loomc_status_t
loomc_result_set_state(loomc_result_t* result, loomc_result_state_t state);

// Adds a copied diagnostic to result.
LOOMC_API_PRIVATE loomc_status_t loomc_result_add_diagnostic(
    loomc_result_t* result, const loomc_diagnostic_t* diagnostic);

// Adds a copied artifact to result.
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
