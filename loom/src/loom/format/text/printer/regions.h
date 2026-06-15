// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_REGIONS_H_
#define LOOM_FORMAT_TEXT_PRINTER_REGIONS_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |block| has an explicit printable label in the module.
bool loom_print_block_has_label(const loom_print_context_t* ctx,
                                const loom_block_t* block);

// Returns true when |block| needs a synthetic label to preserve region block
// boundaries or to give a successor reference a target name.
bool loom_print_block_needs_synthetic_label(const loom_print_context_t* ctx,
                                            const loom_region_t* region,
                                            const loom_block_t* block);

// Prints one block label line, including block arguments when requested.
iree_status_t loom_print_block_label_line_with_options(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_block_t* block, bool print_block_args);

// Prints one block label line, including any block arguments.
iree_status_t loom_print_block_label_line(loom_print_context_t* ctx,
                                          const loom_region_t* region,
                                          const loom_block_t* block);

// Prints a successor reference and reports its emitted field range.
iree_status_t loom_print_successor_ref(loom_print_context_t* ctx,
                                       const loom_op_t* op,
                                       uint8_t successor_index);

// Prints one ordinary operation at the current indentation level.
iree_status_t loom_print_op(loom_print_context_t* ctx, const loom_op_t* op);

// Returns true when |op| is the region's default terminator and has no payload
// that must be preserved textually.
bool loom_print_should_elide_implicit_terminator(
    const loom_region_descriptor_t* region_descriptor, const loom_op_t* op);

// Prints the body of |region| using canonical block/op traversal. When
// |entry_args_declared_by_parent| is true, the entry block arguments are part
// of the parent op signature and are not repeated on an entry block label.
iree_status_t loom_print_region_body(
    loom_print_context_t* ctx, const loom_region_t* region,
    const loom_region_descriptor_t* region_descriptor,
    bool entry_args_declared_by_parent);

// Prints a module body using canonical top-level block/op traversal.
iree_status_t loom_print_module_body(loom_print_context_t* ctx,
                                     const loom_region_t* region);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_REGIONS_H_
