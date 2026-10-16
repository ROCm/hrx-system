// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source value and scalar register-carrier materializers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATERIALIZERS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATERIALIZERS_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"

#ifdef __cplusplus
extern "C" {
#endif

// Retains exact i32, f32, or address literals when a VGPR use cannot reuse the
// actual producer carrier. NULL selects the ordinary register conversion.
iree_status_t loom_amdgpu_prepare_vgpr_literal(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan);

// Emits retained literal bits or copies the bound register payload to VGPRs.
iree_status_t loom_amdgpu_emit_prepared_vgpr_value(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan,
    loom_value_id_t* out_low_value);

// Emits a retained address literal or projects the bound low word to a VGPR.
iree_status_t loom_amdgpu_emit_prepared_vgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan,
    loom_value_id_t* out_low_value);

// Retains exact truth/falsehood as immutable recipes. NULL selects conversion
// of the source's canonical Low predicate. Exact truth reads EXEC at the use.
iree_status_t loom_amdgpu_prepare_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t source_value,
    const void** out_plan);

// Executes a retained mask choice without consulting source facts.
iree_status_t loom_amdgpu_emit_prepared_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, const void* plan,
    loom_value_id_t* out_low_value);

// Preserves VGPR carriers and projects SGPR tuples into equally wide VGPRs.
loom_type_t loom_amdgpu_materialized_vgpr_register_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value);

// Applies the active target's scalar-source limit to a binary VOP3 RHS.
loom_type_t loom_amdgpu_materialized_vop3_binary_rhs_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value);

// Exposes the address's low word in a VGPR.
loom_type_t loom_amdgpu_materialized_vgpr_address_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value);

// Exposes the address's low word in an SGPR.
loom_type_t loom_amdgpu_materialized_sgpr_address_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value);

// Exposes the address's low word in its producer's register class.
loom_type_t loom_amdgpu_materialized_address_register_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value);

// Projects the selected 32-bit address operand without changing register class.
iree_status_t loom_amdgpu_lookup_address_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Native predicates carry an EXEC-width mask in an SGPR pair.
loom_type_t loom_amdgpu_materialized_native_i1_mask_type(
    const loom_low_lower_context_t* context, loom_value_id_t source_value);

// Returns true when the source value has an SGPR or VGPR register mapping
// whose bits can be consumed by a VGPR operand.
iree_status_t loom_amdgpu_value_can_materialize_as_vgpr_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Reuses the bound VGPR value or copies each bound SGPR word into a VGPR.
// Preserves all payload bits, including unspecified upper bits of narrow
// values.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Returns true when a source value can occupy the second operand of a binary
// VOP3 packet after applying the active target's scalar-source limit.
iree_status_t loom_amdgpu_value_can_materialize_as_vop3_binary_rhs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Reuses the source value when the target permits two VOP3 scalar sources.
// Targets limited to one scalar source copy an SGPR source into VGPRs; an
// existing VGPR source is reused.
iree_status_t loom_amdgpu_lookup_or_materialize_vop3_binary_rhs(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

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

// Returns true when a source scalar i1 value can be materialized in one of the
// native AMDGPU predicate representations: SCC, a durable SGPR word, or an
// EXEC-width SGPR mask.
iree_status_t loom_amdgpu_value_can_materialize_as_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_can_materialize);

// Looks up a lowered i64 scalar value and materializes each 32-bit register
// unit into a VGPR pair when a vector-style packet cannot consume the existing
// lowering.
iree_status_t loom_amdgpu_lookup_or_materialize_vgpr_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Looks up a uniform address scalar or canonical integer term and returns its
// low 32-bit SGPR unit, projecting predicates to zero/one.
iree_status_t loom_amdgpu_lookup_or_materialize_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Materializes the low 32 bits of a proven subgroup-uniform address scalar in
// one SGPR. A canonical VGPR producer is projected with readfirstlane without
// changing its source-value mapping.
iree_status_t loom_amdgpu_materialize_uniform_sgpr_address(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_value_id_t* out_low_value);

// Converts a one- or two-word scalar register payload to its selected carrier.
// Widening zero-extends; narrowing retains the low word. Selection must prove
// subgroup uniformity before requesting an SGPR for a VGPR payload. Consumes
// only Low values and the retained destination type, preserving source
// bindings.
iree_status_t loom_amdgpu_materialize_low_scalar_carrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_type_t required_type,
    loom_value_id_t* out_low_value);

// Projects a lowered subgroup-uniform predicate to SCC. The caller owns the
// uniformity proof; storage may be SCC, a normalized SGPR word, or a native
// lane mask. Mask projection tests only the lanes active at this use.
iree_status_t loom_amdgpu_materialize_uniform_i1_condition(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_condition);

// Projects an i1 predicate to the numeric value zero or one in the requested
// SGPR or VGPR bank. An SGPR result requires a subgroup-uniform predicate,
// including when its physical representation is a native lane mask.
iree_status_t loom_amdgpu_lookup_or_materialize_i1_integer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t register_class_id,
    loom_value_id_t* out_low_value);

// Converts a lowered SCC or durable SGPR Boolean to an EXEC-width SGPR mask.
// An existing SGPR-pair mask is reused. Source constant choices belong to the
// caller's selected operand plan; this conversion consumes only Low carriers.
iree_status_t loom_amdgpu_materialize_low_native_i1_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, loom_value_id_t* out_low_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATERIALIZERS_H_
