// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_FORMAT_SIGNATURES_H_
#define LOOM_FORMAT_TEXT_PRINTER_FORMAT_SIGNATURES_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints a result type list, including tied result and named symbol result
// spelling.
iree_status_t loom_print_result_type_list(loom_print_context_t* ctx,
                                          const loom_op_t* op,
                                          const loom_op_vtable_t* vtable,
                                          const loom_format_element_t* element);

// Prints a region binding list.
iree_status_t loom_print_binding_list(loom_print_context_t* ctx,
                                      const loom_op_t* op,
                                      const loom_op_vtable_t* vtable,
                                      const loom_format_element_t* element);

// Prints explicit region block arguments.
iree_status_t loom_print_block_args(loom_print_context_t* ctx,
                                    const loom_op_t* op,
                                    const loom_format_element_t* element);

// Prints func-like signature arguments.
iree_status_t loom_print_func_args(loom_print_context_t* ctx,
                                   const loom_op_t* op,
                                   const loom_op_vtable_t* vtable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_FORMAT_SIGNATURES_H_
