// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_FORMAT_TABLES_H_
#define LOOM_FORMAT_TEXT_PRINTER_FORMAT_TABLES_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints an inline attribute dictionary from attributes not covered elsewhere
// by the generated format.
iree_status_t loom_print_inline_attr_dict(
    loom_print_context_t* ctx, const loom_op_t* op,
    const loom_op_vtable_t* vtable,
    const loom_format_element_t* inline_element);

// Prints a named operand dictionary.
iree_status_t loom_print_operand_dict(loom_print_context_t* ctx,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable,
                                      const loom_format_element_t* element);

// Prints an attribute-keyed operand row table.
iree_status_t loom_print_attr_table(loom_print_context_t* ctx,
                                    const loom_op_t* op,
                                    const loom_op_vtable_t* vtable,
                                    const loom_format_element_t* element);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_FORMAT_TABLES_H_
