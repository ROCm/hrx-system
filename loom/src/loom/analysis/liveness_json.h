// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Machine-readable JSON formatting for Loom liveness analyses.
//
// This is intentionally split from liveness.h so schedulers and allocators can
// consume the core analysis without depending on text printing or JSON helpers.

#ifndef LOOM_ANALYSIS_LIVENESS_JSON_H_
#define LOOM_ANALYSIS_LIVENESS_JSON_H_

#include "iree/base/api.h"
#include "loom/analysis/liveness.h"
#include "loom/format/text/printer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends a compact JSON object describing |analysis| to |builder|. The format
// is diagnostic/test output, not bytecode-stable artifact identity.
iree_status_t loom_liveness_format_json(
    const loom_liveness_analysis_t* analysis,
    const loom_text_print_options_t* type_print_options,
    iree_string_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_LIVENESS_JSON_H_
