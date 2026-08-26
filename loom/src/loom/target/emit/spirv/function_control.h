// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V structured-control emission for target-low functions.

#ifndef LOOM_TARGET_EMIT_SPIRV_FUNCTION_CONTROL_H_
#define LOOM_TARGET_EMIT_SPIRV_FUNCTION_CONTROL_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_emit_state_t loom_spirv_emit_state_t;

// Emits a verified low.scf.if operation.
iree_status_t loom_spirv_emit_scf_if(loom_spirv_emit_state_t* state,
                                     const loom_op_t* op);

// Emits a verified low.scf.for operation.
iree_status_t loom_spirv_emit_scf_for(loom_spirv_emit_state_t* state,
                                      const loom_op_t* op);

// Emits a verified low.scf.while operation.
iree_status_t loom_spirv_emit_scf_while(loom_spirv_emit_state_t* state,
                                        const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_FUNCTION_CONTROL_H_
