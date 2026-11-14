// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_EMIT_SPIRV_COMPONENT_MATH_H_
#define LOOM_TARGET_EMIT_SPIRV_COMPONENT_MATH_H_

#include "loom/target/emit/spirv/function_emitter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the module's typed helper for |row|, emitting it on first use. The
// helper maps the selected scalar primitive over the implementation-defined
// components of a cooperative matrix without exposing their physical layout.
iree_status_t loom_spirv_emit_component_math_function(
    loom_spirv_function_emission_context_t* context,
    const loom_spirv_packet_row_t* row, uint32_t matrix_type_id,
    uint32_t instruction_set_id, uint32_t* out_function_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_COMPONENT_MATH_H_
