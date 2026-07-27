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

// Returns true when low asm region syntax was requested by print options.
bool loom_print_low_asm_is_requested(loom_print_context_t* ctx);

// Prints a region using mandatory low asm syntax.
iree_status_t loom_print_low_asm_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent);

// Attempts to print a region using optional low asm syntax. Sets
// |out_printed| to false when the region has no lossless low asm spelling.
iree_status_t loom_print_low_asm_optional_region(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent, bool* out_printed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_LOW_ASM_H_
