// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Subgroup-uniform inputs to SPIR-V cooperative matrix constructors.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_COOPERATIVE_INITIALIZER_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_COOPERATIVE_INITIALIZER_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns whether the fragment describes one matrix with native component bits,
// without encoded scale, sparse, or packed payload interpretation.
bool loom_spirv_cooperative_initializer_has_native_payload(
    const loom_value_fact_table_t* facts, const loom_op_t* source_op);

// Proves native component interpretation, one logical matrix block, and
// subgroup-uniform value and participation for a vector.fragment constructor.
iree_status_t loom_spirv_can_materialize_cooperative_initializer_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, bool* out_can_materialize);

// Preserves the scalar producer's selected native carrier.
loom_type_t loom_spirv_cooperative_initializer_component_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    const void* plan);

// Reuses the bound value after selection proved subgroup uniformity.
iree_status_t loom_spirv_materialize_cooperative_initializer_component(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_COOPERATIVE_INITIALIZER_H_
