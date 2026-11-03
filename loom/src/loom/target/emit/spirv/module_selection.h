// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_SELECTION_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_SELECTION_H_

#include "loom/target/emit/spirv/module_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selects the exact payload bits into a reserved result ID. Vector payloads
// have a Boolean vector condition with the same lane count. Floating payloads
// use integer carriers so selection is independent of floating-point modes.
iree_status_t loom_spirv_module_emit_selection(
    loom_spirv_type_context_t* context, loom_spirv_value_type_t value_type,
    uint32_t result_type_id, uint32_t result_id, uint32_t condition_id,
    uint32_t true_id, uint32_t false_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_SELECTION_H_
