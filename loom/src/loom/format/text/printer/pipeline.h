// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_PIPELINE_H_
#define LOOM_FORMAT_TEXT_PRINTER_PIPELINE_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |region| has a lossless pass.pipeline friendly spelling.
bool loom_print_pipeline_region_is_friendly(
    const loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor);

// Prints a region using mandatory pass.pipeline friendly syntax.
iree_status_t loom_print_pipeline_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor);

// Prints a skipped pass.pipeline friendly region body.
iree_status_t loom_print_pipeline_skipped_region(loom_print_context_t* ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_PIPELINE_H_
