// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source CFG and structured-control emission.

#ifndef LOOM_CODEGEN_LOW_LOWER_STRUCTURAL_H_
#define LOOM_CODEGEN_LOW_LOWER_STRUCTURAL_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates and binds Low blocks for the source function CFG.
iree_status_t loom_low_lower_map_blocks(loom_low_lower_context_t* context,
                                        loom_region_t* source_body);

// Selects target-specific branch plans after Low blocks have been created.
iree_status_t loom_low_lower_prepare_branches(loom_low_lower_context_t* context,
                                              loom_region_t* source_body);

// Emits the source region into the mapped Low function body.
iree_status_t loom_low_lower_emit_body(loom_low_lower_context_t* context,
                                       loom_region_t* source_body);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_STRUCTURAL_H_
