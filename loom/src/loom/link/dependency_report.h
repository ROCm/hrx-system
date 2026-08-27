// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic diagnostics and JSON serialization for dependency analysis.

#ifndef LOOM_LINK_DEPENDENCY_REPORT_H_
#define LOOM_LINK_DEPENDENCY_REPORT_H_

#include "iree/base/api.h"
#include "loom/link/dependency_analysis.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when every requirement has valid direct ownership and exact
// definition resolution is unambiguous.
bool loom_link_dependency_analysis_succeeded(
    const loom_link_dependency_analysis_t* analysis);

// Returns the stable diagnostic code for a failed |requirement|.
// Returns an empty string for a satisfied requirement.
iree_string_view_t loom_link_dependency_diagnostic_code(
    const loom_link_dependency_requirement_t* requirement);

// Formats one human-readable diagnostic for a failed |requirement|.
// |component_name| is caller-owned provenance and may be empty.
iree_status_t loom_link_dependency_format_diagnostic(
    const loom_link_dependency_analysis_t* analysis,
    const loom_link_dependency_requirement_t* requirement,
    iree_string_view_t component_name, loom_output_stream_t* stream);

// Writes the complete schema-versioned dependency report as compact JSON.
// Serialization is allocation-free apart from storage owned by |stream|.
iree_status_t loom_link_dependency_format_json(
    const loom_link_dependency_analysis_t* analysis,
    iree_string_view_t component_name, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_DEPENDENCY_REPORT_H_
