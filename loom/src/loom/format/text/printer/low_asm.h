// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_LOW_ASM_H_
#define LOOM_FORMAT_TEXT_PRINTER_LOW_ASM_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |region| must use low asm syntax under the active source or
// printer policy. Nested regions inherit an enclosing low asm region.
bool loom_print_low_asm_is_requested(loom_print_context_t* ctx,
                                     const loom_region_t* region);

// Returns true when a requested low asm region starts with an explicit `asm`
// marker instead of inheriting syntax from its enclosing region.
bool loom_print_low_asm_uses_marker(loom_print_context_t* ctx,
                                    const loom_region_t* region);

// Attempts to print a region using optional low asm syntax. When canonical
// fallback is allowed, sets |out_printed| to false if no lossless asm spelling
// exists.
iree_status_t loom_print_low_asm_optional_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent, bool* out_printed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_LOW_ASM_H_
