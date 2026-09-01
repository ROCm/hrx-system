// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low lowering report construction.

#ifndef LOOM_CODEGEN_LOW_LOWER_REPORT_H_
#define LOOM_CODEGEN_LOW_LOWER_REPORT_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns exact source execution evidence for an operation when loop and CFG
// facts can prove it without target execution.
iree_status_t loom_low_lower_source_op_execution_count_plus_one(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint64_t* out_execution_count_plus_one);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REPORT_H_
