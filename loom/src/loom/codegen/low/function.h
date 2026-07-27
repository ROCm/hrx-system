// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low function-entry utilities.
//
// Low helper functions and low kernel entries share the same structural body
// model and descriptor/scheduler/allocation machinery. These helpers keep that
// production contract centralized so adding a specialized entry op does not
// fork the backend pipeline.

#ifndef LOOM_CODEGEN_LOW_FUNCTION_H_
#define LOOM_CODEGEN_LOW_FUNCTION_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true for low function definitions with executable bodies.
bool loom_low_function_def_isa(const loom_op_t* op);

// Returns the module-local symbol ref naming |function_op|.
loom_symbol_ref_t loom_low_function_callee(const loom_op_t* function_op);

// Returns the module-local target record symbol ref selected by |function_op|.
loom_symbol_ref_t loom_low_function_target(const loom_op_t* function_op);

// Returns the low allocation mode attr value, or 0 when absent.
uint8_t loom_low_function_allocation(const loom_op_t* function_op);

// Returns the low schedule mode attr value, or 0 when absent.
uint8_t loom_low_function_schedule(const loom_op_t* function_op);

// Returns the executable body region of |function_op|, or NULL when absent.
loom_region_t* loom_low_function_body(loom_op_t* function_op);

// Returns the executable body region of |function_op|, or NULL when absent.
const loom_region_t* loom_low_function_const_body(const loom_op_t* function_op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_FUNCTION_H_
