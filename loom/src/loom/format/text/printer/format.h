// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_TEXT_PRINTER_FORMAT_H_
#define LOOM_FORMAT_TEXT_PRINTER_FORMAT_H_

#include "iree/base/api.h"
#include "loom/format/text/printer/context.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints |op| using its generated assembly-format element stream.
iree_status_t loom_print_format_elements(loom_print_context_t* ctx,
                                         const loom_op_t* op,
                                         const loom_op_vtable_t* vtable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_TEXT_PRINTER_FORMAT_H_
