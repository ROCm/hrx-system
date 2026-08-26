// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source value materializers used by generated lower-rule tables.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATERIALIZERS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATERIALIZERS_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when a source i32 scalar or vector value can be materialized as
// a VGPR operand for vector-style packets.
iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Returns true when a source f32 scalar or vector value can be materialized as
// a VGPR operand for vector-style packets.
iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Returns true when a source i64 scalar can be materialized as a VGPR pair for
// vector-style packets.
iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Returns true when a source address scalar can be materialized as a VGPR
// operand for vector-style address arithmetic.
iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Returns true when a source address scalar can be materialized as a one-unit
// SGPR operand for scalar address arithmetic.
iree_status_t loom_amdgpu_value_can_materialize_as_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Returns true when a source scalar i1 value can be materialized as an
// EXEC-width SGPR mask for divergent predicate arithmetic.
iree_status_t loom_amdgpu_value_can_materialize_as_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Looks up a lowered i32 scalar or vector value and materializes exact source
// constants into VGPRs when a vector-style packet cannot consume the existing
// lowering.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Looks up a lowered f32 scalar or vector value and materializes exact source
// constants into VGPRs when a vector-style packet cannot consume the existing
// lowering.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_f32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Looks up a lowered i64 scalar value and materializes each 32-bit register
// unit into a VGPR pair when a vector-style packet cannot consume the existing
// lowering.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Looks up a lowered address scalar and returns its low 32 bits in one VGPR,
// materializing constants and uniform values when required.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Looks up a lowered address scalar and returns its low 32-bit SGPR unit.
iree_status_t loom_amdgpu_lookup_or_materialize_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Looks up a lowered i1 value and materializes subgroup-uniform SCC predicates
// as EXEC-width SGPR masks for divergent predicate arithmetic.
iree_status_t loom_amdgpu_lookup_or_materialize_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATERIALIZERS_H_
