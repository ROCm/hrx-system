// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source-to-low type and value mapping helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_TYPES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_TYPES_H_

#include <stdint.h>

#include "loom/analysis/view_regions.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/low_legality.h"
#include "loom/target/registers.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when the source type is a scalar i32.
bool loom_amdgpu_type_is_i32(loom_type_t type);

// Returns true when the source type is a scalar i16.
bool loom_amdgpu_type_is_i16(loom_type_t type);

// Returns true when the source type is a scalar i64.
bool loom_amdgpu_type_is_i64(loom_type_t type);

// Returns true when the source type is a scalar i8.
bool loom_amdgpu_type_is_i8(loom_type_t type);

// Returns true when the source type is a scalar i1.
bool loom_amdgpu_type_is_i1(loom_type_t type);

// Returns true when the source type is an address-sized scalar lowered through
// the current 32-bit AMDGPU scalar path.
bool loom_amdgpu_type_is_address_scalar(loom_type_t type);

// Returns true when the source type is a scalar f32.
bool loom_amdgpu_type_is_f32(loom_type_t type);

// Returns true when the source type is a scalar f64.
bool loom_amdgpu_type_is_f64(loom_type_t type);

// Returns true when the source type is a scalar f16 or bf16.
bool loom_amdgpu_type_is_16bit_float(loom_type_t type);

typedef enum loom_amdgpu_vector_storage_kind_e {
  LOOM_AMDGPU_VECTOR_STORAGE_KIND_NONE = 0,
  LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_32BIT = 1,
  LOOM_AMDGPU_VECTOR_STORAGE_KIND_FULL_64BIT = 2,
  LOOM_AMDGPU_VECTOR_STORAGE_KIND_I1_MASK = 3,
  LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_16BIT_FLOAT = 4,
  LOOM_AMDGPU_VECTOR_STORAGE_KIND_PACKED_INTEGER = 5,
} loom_amdgpu_vector_storage_kind_t;

typedef struct loom_amdgpu_vector_storage_t {
  // Physical storage shape selected for the source vector type.
  loom_amdgpu_vector_storage_kind_t kind;
  // Source IR scalar element type.
  loom_scalar_type_t element_type;
  // Number of source IR logical lanes.
  uint32_t element_count;
  // Number of 32-bit low register units occupied by the storage.
  uint32_t register_count;
  // Number of 32-bit low register units occupied by one logical lane.
  uint32_t element_register_count;
  // Number of payload bits occupied by one logical lane.
  uint32_t element_bit_count;
} loom_amdgpu_vector_storage_t;

// Returns true when the source type maps to one of AMDGPU's supported static
// vector storage classes.
bool loom_amdgpu_type_vector_storage(loom_type_t type,
                                     loom_amdgpu_vector_storage_t* out_storage);

// Returns a static rank-1 vector lane count for the requested element type, or
// zero when the type is not a supported static rank-1 vector.
uint32_t loom_amdgpu_static_vector_lane_count(loom_type_t type,
                                              loom_scalar_type_t element_type,
                                              uint32_t max_lane_count);

// Returns the flattened element count for a static vector of the requested
// element type, or zero when the type is not a supported static vector payload.
uint32_t loom_amdgpu_static_vector_register_count(
    loom_type_t type, loom_scalar_type_t element_type,
    uint32_t max_register_count);

// Returns the rank-1 lane count for a supported AMDGPU 32-bit vector payload,
// or zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_32bit_lane_count(loom_type_t type);

// Returns the flattened register count for a supported static AMDGPU 32-bit
// vector payload, or zero when the source type is not representable as that
// payload.
uint32_t loom_amdgpu_vector_32bit_register_count(loom_type_t type);

// Returns the row-major flattened register ordinal for static vector indices.
bool loom_amdgpu_static_vector_flat_register_from_indices(
    loom_type_t type, const int64_t* indices, uint32_t* out_ordinal);

// Returns the rank-1 i32 lane count for a supported AMDGPU 32-bit vector
// payload, or zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i32_lane_count(loom_type_t type);

// Returns the flattened register count for a supported static i32 vector
// payload, or zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i32_register_count(loom_type_t type);

// Returns true when the source type can be loaded through scalar memory as a
// bitwise 32-bit payload.
bool loom_amdgpu_type_is_32bit_memory_payload(loom_type_t type);

// Returns the rank-1 f32 lane count for a supported AMDGPU 32-bit vector
// payload, or zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_f32_lane_count(loom_type_t type);

// Returns the flattened register count for a supported static f32 vector
// payload, or zero when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_f32_register_count(loom_type_t type);

// Returns the i1 lane count for a supported AMDGPU mask vector payload, or zero
// when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i1_lane_count(loom_type_t type);

// Returns the i8 lane count for a supported AMDGPU packed byte payload, or zero
// when the source type is not representable as that payload.
uint32_t loom_amdgpu_vector_i8_lane_count(loom_type_t type);

// Returns true when the source type is an integer vector payload that can be
// stored in packed 32-bit registers. The payload occupies the low bits of the
// register range; unused high bits in the final register are unspecified.
bool loom_amdgpu_type_packed_integer_storage(loom_type_t type,
                                             uint32_t* out_payload_bit_count,
                                             uint32_t* out_register_count);

// Returns true when the source type is an f16/bf16 vector payload that can be
// stored in packed 32-bit registers. Odd lane counts occupy the low half of the
// final register; the high half is unspecified.
bool loom_amdgpu_type_packed_16bit_float_storage(
    loom_type_t type, uint32_t* out_payload_bit_count,
    uint32_t* out_register_count);

// Returns true when the source type is a byte-addressable view that can map to
// an AMDGPU HAL/global buffer resource or LDS root.
bool loom_amdgpu_type_is_byte_addressable_view(loom_type_t type);

// Returns true when a source value has scalar i32 type.
bool loom_amdgpu_value_is_i32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id);

// Returns true when a source value has an address scalar type.
bool loom_amdgpu_value_is_address_scalar(loom_low_lower_context_t* context,
                                         loom_value_id_t value_id);

// Returns true when a source value has scalar f32 type.
bool loom_amdgpu_value_is_f32(loom_low_lower_context_t* context,
                              loom_value_id_t value_id);

// Returns true when a source value has scalar f16 or bf16 type.
bool loom_amdgpu_value_is_16bit_float(loom_low_lower_context_t* context,
                                      loom_value_id_t value_id);

// Returns true when a source value has a byte-addressable view type.
bool loom_amdgpu_value_is_byte_addressable_view(
    loom_low_lower_context_t* context, loom_value_id_t value_id);

// Builds a one-unit SGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_sgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type);

// Builds a multi-unit SGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_sgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type);

// Builds a one-unit VGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_vgpr_type(loom_low_lower_context_t* context,
                                         loom_type_t* out_type);

// Builds the one-unit SCC register type in the current lowering context.
iree_status_t loom_amdgpu_make_scc_type(loom_low_lower_context_t* context,
                                        loom_type_t* out_type);

// Builds a multi-unit VGPR register type in the current lowering context.
iree_status_t loom_amdgpu_make_vgpr_range_type(
    loom_low_lower_context_t* context, uint32_t unit_count,
    loom_type_t* out_type);

// Builds the register type for a descriptor row's implicit resource operand.
iree_status_t loom_amdgpu_make_descriptor_row_implicit_resource_type(
    loom_low_lower_context_t* context, const loom_low_descriptor_t* descriptor,
    loom_type_t* out_type);

// Returns whether a low register type belongs to the requested AMDGPU register
// class.
iree_status_t loom_amdgpu_low_type_register_class_is(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    bool* out_match);

// Returns whether a low register type belongs to the requested AMDGPU register
// class and occupies the requested number of 32-bit register units.
iree_status_t loom_amdgpu_low_type_is_register_class_count(
    loom_low_lower_context_t* context, loom_type_t type, uint16_t reg_class_id,
    uint32_t register_unit_count, bool* out_match);

// Returns whether a low value belongs to the requested AMDGPU register class
// and occupies the requested number of 32-bit register units.
iree_status_t loom_amdgpu_low_value_is_register_class_count(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    uint16_t reg_class_id, uint32_t register_unit_count, bool* out_match);

// Returns true when the source value should prefer a VGPR mapping even if its
// scalar type could otherwise map to an SGPR. Fact and view-region tables
// enable placement-sensitive proofs for values such as read-only scalar memory
// loads; pass NULL for both tables only on callers that intentionally need the
// conservative module-local answer.
bool loom_amdgpu_source_value_prefers_vgpr(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id);

// Returns true when the source value is an i1 represented by an EXEC-width
// native lane mask using the supplied fact and view-region context.
bool loom_amdgpu_source_value_is_native_i1_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    loom_value_id_t source_value_id);

// Computes whether the source value should prefer a VGPR mapping in the active
// lowering context.
iree_status_t loom_amdgpu_context_value_prefers_vgpr(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_prefers_vgpr);

// Computes whether the source value is represented as an EXEC-width native lane
// mask in the active lowering context.
iree_status_t loom_amdgpu_context_value_is_native_i1_mask(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    bool* out_is_native_mask);

// Returns true when the source value is a subgroup lane-mask integer and every
// active lane observes the same mask payload.
bool loom_amdgpu_source_value_is_uniform_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id);

// Returns true when the source value is a subgroup lane-mask integer whose
// payload may differ per active lane.
bool loom_amdgpu_source_value_is_divergent_subgroup_lane_mask(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t source_value_id);

// Maps a source type to the default AMDGPU low register type.
iree_status_t loom_amdgpu_map_type(void* user_data,
                                   loom_low_lower_context_t* context,
                                   const loom_op_t* source_op,
                                   loom_type_t source_type,
                                   loom_type_t* out_low_type);

// Maps a source value to the AMDGPU low register type selected for its
// placement-sensitive use.
iree_status_t loom_amdgpu_map_value(void* user_data,
                                    loom_low_lower_context_t* context,
                                    const loom_op_t* source_op,
                                    loom_value_id_t source_value_id,
                                    loom_type_t source_type,
                                    loom_type_t* out_low_type);

// Maps a source value to AMDGPU descriptor register metadata for read-only
// target contract queries.
iree_status_t loom_amdgpu_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_TYPES_H_
