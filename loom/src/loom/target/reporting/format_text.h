// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared line-oriented compile report field formatting.

#ifndef LOOM_TARGET_REPORTING_FORMAT_TEXT_H_
#define LOOM_TARGET_REPORTING_FORMAT_TEXT_H_

#include "loom/target/reporting/format.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns |value| or the stable placeholder for an unavailable text value.
iree_string_view_t loom_target_compile_report_text_non_empty(
    iree_string_view_t value);

// Appends a named string field using the stable unavailable-value placeholder.
iree_status_t loom_target_compile_report_text_append_string_field(
    iree_string_builder_t* builder, iree_string_view_t name,
    iree_string_view_t value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_REPORTING_FORMAT_TEXT_H_
