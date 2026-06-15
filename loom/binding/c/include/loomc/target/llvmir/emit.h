// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_LLVMIR_EMIT_H_
#define LOOMC_TARGET_LLVMIR_EMIT_H_

#include "loomc/emit.h"
#include "loomc/target/llvmir/base.h"

/// @file
/// LLVMIR emission option space.
///
/// Link the LLVMIR target binding package to make the generic
/// `loomc_emit_module` operation capable of producing
/// `LOOMC_ARTIFACT_FORMAT_LLVMIR_TEXT` and
/// `LOOMC_ARTIFACT_FORMAT_LLVMIR_BITCODE` artifacts.
///
/// @par Example
/// Emit a prepared LLVMIR target-low module as textual LLVM IR:
///
/// @code{.c}
/// loomc_emit_options_t emit_options = {
///     .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
///     .structure_size = sizeof(loomc_emit_options_t),
///     .artifact_format =
///         loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LLVMIR_TEXT),
///     .identifier = loomc_make_cstring_view("kernel.ll"),
/// };
///
/// loomc_result_t* emit_result = NULL;
/// loomc_status_t status = loomc_emit_module(
///     target_environment, workspace, module, &emit_options,
///     loomc_allocator_system(), &emit_result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(emit_result)) {
///   // Inspect typed diagnostics from emit_result.
/// }
/// loomc_result_release(emit_result);
/// @endcode

#endif  // LOOMC_TARGET_LLVMIR_EMIT_H_
