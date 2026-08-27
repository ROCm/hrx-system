// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU constant analysis, planning, and lowering.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_CONSTANTS_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_CONSTANTS_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

enum {
  // Maximum unsigned integer payload encodable by the AMDGPU source inline
  // immediate form.
  LOOM_AMDGPU_SOURCE_INLINE_U32_MAX = 64u,
  // Number of lanes in one DPP row.
  LOOM_AMDGPU_DPP_ROW_LANE_COUNT = 16u,
};

#ifdef __cplusplus
extern "C" {
#endif

// Extracts an exact index constant from a module value.
bool loom_amdgpu_module_value_as_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value);

// Extracts an exact scalar i32 constant from a module value.
bool loom_amdgpu_module_value_as_i32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              int64_t* out_value);

// Extracts an exact scalar f32 constant as a raw IEEE bit pattern.
bool loom_amdgpu_module_value_as_f32_constant(const loom_module_t* module,
                                              loom_value_id_t value_id,
                                              uint32_t* out_bit_pattern);

// Extracts an exact non-negative signed 64-bit integer from value facts.
bool loom_amdgpu_value_facts_as_exact_non_negative_i64(loom_value_facts_t facts,
                                                       int64_t* out_value);

// Extracts an exact signed 32-bit integer from value facts.
bool loom_amdgpu_value_facts_as_exact_i32(loom_value_facts_t facts,
                                          int64_t* out_value);

// Extracts an exact signed 32-bit integer from value facts or a module
// constant.
bool loom_amdgpu_value_as_exact_i32(const loom_module_t* module,
                                    const loom_value_fact_table_t* fact_table,
                                    loom_value_id_t value_id,
                                    int64_t* out_value);

// Extracts an exact f32 constant as a raw IEEE bit pattern from value facts.
bool loom_amdgpu_value_facts_as_f32_bit_pattern(loom_value_facts_t facts,
                                                uint32_t* out_bit_pattern);

// Extracts an exact f32 constant as a raw IEEE bit pattern from value facts or
// a module constant.
bool loom_amdgpu_value_as_f32_bit_pattern(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t value_id, uint32_t* out_bit_pattern);

// Extracts an exact integer or f32 value as a raw 32-bit payload.
bool loom_amdgpu_value_facts_as_u32_bits(loom_value_facts_t facts,
                                         uint32_t* out_bits);

// Extracts one exact scalar/vector lane value as a raw 32-bit payload.
bool loom_amdgpu_source_lane_as_u32_bits(
    const loom_value_fact_table_t* fact_table, const loom_module_t* module,
    loom_value_id_t source, uint32_t lane, uint32_t* out_bits);

// Returns true when value is a non-zero power of two.
bool loom_amdgpu_u32_is_power_of_two(uint32_t value);

// Returns true when an attribute can encode as a signed 32-bit immediate.
bool loom_amdgpu_attr_is_i32_immediate(loom_attribute_t value);

// Returns true when an attribute can encode as an unsigned 32-bit address.
bool loom_amdgpu_attr_is_u32_address_immediate(loom_attribute_t value);

// Returns true when an attribute can encode as an f32 immediate payload.
bool loom_amdgpu_attr_is_f32_immediate(loom_attribute_t value);

// Returns the f32 bit pattern produced by narrowing an attribute.
uint32_t loom_amdgpu_attr_f32_bit_pattern(loom_attribute_t value);

// Returns true when an attribute can encode as an f16/bf16 immediate payload.
bool loom_amdgpu_attr_is_16bit_float_immediate(loom_attribute_t value);

// Returns the low-half f16/bf16 bit pattern produced by narrowing an attribute.
uint32_t loom_amdgpu_attr_16bit_float_bit_pattern(loom_scalar_type_t type,
                                                  loom_attribute_t value);

// Extracts a source scalar i32 constant.
bool loom_amdgpu_value_as_i32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       int64_t* out_value);

// Extracts a source scalar i1 exact constant from value facts.
bool loom_amdgpu_value_as_i1_constant(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id,
                                      bool* out_value);

// Extracts a source scalar f32 constant as a raw IEEE bit pattern.
bool loom_amdgpu_value_as_f32_constant(loom_low_lower_context_t* context,
                                       loom_value_id_t value_id,
                                       uint32_t* out_bit_pattern);

// Extracts a source address scalar constant.
bool loom_amdgpu_value_as_address_constant(loom_low_lower_context_t* context,
                                           loom_value_id_t value_id,
                                           int64_t* out_value);

// Resolves a descriptor and interns its imm32 attribute name when present.
iree_status_t loom_amdgpu_resolve_imm32_descriptor(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_low_lower_resolved_descriptor_t* out_descriptor,
    loom_string_id_t* out_imm32_attr_name_id, bool* out_present);

// Selects an AMDGPU constant materialization plan for index.constant.
iree_status_t loom_amdgpu_select_index_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU constant materialization plan for scalar.constant.
iree_status_t loom_amdgpu_select_scalar_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected);

// Selects an AMDGPU constant materialization plan for vector.constant.
iree_status_t loom_amdgpu_select_vector_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_constant_plan_t* out_plan, bool* out_selected);

// Emits and binds an ordered range of descriptor-backed u32 constants.
iree_status_t loom_amdgpu_bind_register_u32_lane_constants(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result,
    const loom_low_lower_resolved_descriptor_t* descriptor,
    loom_string_id_t imm32_attr_name_id, const uint32_t* lane_bit_patterns,
    uint32_t lane_count);

// Lowers an AMDGPU constant materialization plan.
iree_status_t loom_amdgpu_lower_constant_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_constant_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_CONSTANTS_H_
