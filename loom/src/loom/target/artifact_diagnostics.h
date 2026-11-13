// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-backed rejections of executable artifact contracts. Callers retain
// semantic acceptance separately; these statuses only report sink failures.
#ifndef LOOM_TARGET_ARTIFACT_DIAGNOSTICS_H_
#define LOOM_TARGET_ARTIFACT_DIAGNOSTICS_H_

#include "loom/error/emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits an unmet artifact requirement at its owning operation.
iree_status_t loom_target_artifact_emit_requirement(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t format, iree_string_view_t constraint);

// Emits an unmet symbol requirement at its declaration or use.
iree_status_t loom_target_artifact_emit_symbol_requirement(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t format, iree_string_view_t name,
    iree_string_view_t constraint);

// Diagnoses the second export and relates the first declaration with its name.
iree_status_t loom_target_artifact_emit_duplicate_export(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_op_t* previous_op, iree_string_view_t format,
    iree_string_view_t name);

// Emits a representational limit with its actual and maximum extents. A NULL
// operation is appropriate only for a limit owned by the complete module.
iree_status_t loom_target_artifact_emit_limit(iree_diagnostic_emitter_t emitter,
                                              const loom_op_t* op,
                                              iree_string_view_t format,
                                              iree_string_view_t quantity,
                                              uint64_t actual,
                                              uint64_t maximum);

// Emits an out-of-bounds symbol offset at the operation requesting its address.
iree_status_t loom_target_artifact_emit_symbol_offset(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t format, iree_string_view_t name, uint64_t offset,
    uint64_t length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARTIFACT_DIAGNOSTICS_H_
