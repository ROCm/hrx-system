// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_REPORT_INTERNAL_H_
#define LOOMC_COMPILE_REPORT_INTERNAL_H_

#include "loom/target/reporting/format.h"
#include "loomc/compile_report.h"
#include "result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates one compile-report option descriptor.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_options_validate(
    const loomc_compile_report_options_t* options);

// Converts a public report mode to the target report formatter mode.
LOOMC_API_PRIVATE loom_target_compile_report_format_mode_t
loomc_compile_report_format_mode(loomc_compile_report_mode_t mode);

// Returns detail categories collected for a public report mode.
LOOMC_API_PRIVATE loom_target_compile_report_detail_flags_t
loomc_compile_report_requested_detail_flags(loomc_compile_report_mode_t mode);

// Creates an owned report identifier. An explicit identifier is preserved;
// otherwise ".compile-report.json" is appended to |default_identifier|.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_make_identifier(
    const loomc_compile_report_options_t* options,
    loomc_string_view_t default_identifier, loomc_allocator_t allocator,
    loomc_string_view_t* out_identifier);

// Formats and appends one typed compile report to |result|.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_report_add_artifact(
    loomc_result_t* result, loomc_compile_report_mode_t mode,
    loomc_string_view_t identifier, const loom_target_compile_report_t* report);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_REPORT_INTERNAL_H_
