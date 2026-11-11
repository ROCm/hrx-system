// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Operation decoding for symbolic integer conversion analysis.

#ifndef LOOM_ANALYSIS_SYMBOLIC_INTEGER_CAST_H_
#define LOOM_ANALYSIS_SYMBOLIC_INTEGER_CAST_H_

#include "loom/ir/integer_cast.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Describes the exact fixed-width payload interpretation performed by |op|.
// Returns false when the op is not a supported integer/address cast.
bool loom_symbolic_integer_cast_from_op(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_integer_cast_t* out_cast);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOLIC_INTEGER_CAST_H_
