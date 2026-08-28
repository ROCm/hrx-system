// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU narrow floating-point payload lowering for FP8/BF8 formats.
//
// This shard owns AMDGPU packet materialization for FP8/BF8 payloads across
// scalar, vector, and matrix-fragment users. Target-independent numeric-format
// semantics live in the generic scalar type and value-fact layers. Descriptor
// availability and architecture feature rows should come from AMDGPU generated
// target tables; this file exposes the selected FP8/BF8 recipe machinery.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_H_

#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_fp8_decode_plan_flag_bits_e {
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32 = 1u << 0,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16 = 1u << 1,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32 = 1u << 2,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR = 1u << 3,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK = 1u << 4,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_ADD3_SRC2_LITERAL = 1u << 5,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM = 1u << 6,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC2_LITERAL = 1u << 7,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MIN_U16 = 1u << 8,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_LO_U16 = 1u << 9,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32_SRC0_LITERAL = 1u << 10,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ADD_U16 = 1u << 11,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_ASHRREV_I16 = 1u << 12,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_NE_I32_SRC1_INLINE = 1u << 13,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64 = 1u << 14,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAX_U16 = 1u << 15,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_CMP_LG_U64_SRC1_INLINE = 1u << 16,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PERM_B32_SRC1_ZERO_SRC2_LIT = 1u << 17,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFI_B32 = 1u << 18,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_LSHL_ADD_U32_SHIFT_IMM_SRC2_LITERAL =
      1u << 19,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHLREV_B16 = 1u << 20,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16 = 1u << 21,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MAD_U16_SRC2_LITERAL = 1u << 22,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_LSHRREV_B16 = 1u << 23,
  LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PK_MUL_F16 = 1u << 24,
} loom_amdgpu_fp8_decode_plan_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_decode_plan_flags_t;

typedef enum loom_amdgpu_fp8_decode_plan_capability_bits_e {
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_NONE = 0u,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_EXACT_REPAIR = 1u << 0,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_ZERO_REPAIR = 1u << 1,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_MASK_REPAIR_SPLIT = 1u << 2,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_INLINE_SGPR64_ZERO_COMPARE = 1u << 3,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_FINITE_NAN_CONDITION = 1u
                                                                         << 4,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_COMBINED_NON_NORMAL_CONDITION = 1u
                                                                         << 5,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_F16_PAYLOAD = 1u << 6,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_NORMAL_BF16_PAYLOAD = 1u << 7,
  LOOM_AMDGPU_FP8_DECODE_PLAN_CAPABILITY_PACKED_EXACT_BF16_VIA_F16 = 1u << 8,
} loom_amdgpu_fp8_decode_plan_capability_bits_t;
typedef uint32_t loom_amdgpu_fp8_decode_plan_capabilities_t;

// Maps target-independent value facts into FP8 decode simplification flags.
loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_fp8_decode_value_flags_from_facts(loom_value_facts_t facts);

// Returns the source format accepted by native conversion descriptors for an
// exact FP8 payload and its proven content facts. NONE requires exact software
// conversion on targets whose native instruction has different semantics.
loom_value_fact_numeric_format_flags_t loom_amdgpu_fp8_descriptor_source_format(
    loom_value_fact_numeric_format_flags_t exact_source_format,
    loom_value_facts_t content_facts);

typedef enum loom_amdgpu_fp8_encoded_operand_schema_kind_e {
  LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_UNSCALED = 0,
  LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_F32,
  LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0,
} loom_amdgpu_fp8_encoded_operand_schema_kind_t;

// Returns true when |schema| matches one of AMDGPU's direct FP8/BF8 vector
// decode routes for |element_type| and the logical source lane count.
bool loom_amdgpu_fp8_encoded_operand_schema_matches(
    loom_value_fact_encoded_operand_schema_t schema,
    loom_scalar_type_t element_type, uint32_t lane_count,
    loom_amdgpu_fp8_encoded_operand_schema_kind_t kind);

typedef enum loom_amdgpu_fp8_packed_bf16_missing_requirement_bits_e {
  LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE = 0u,
  LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PERMUTE_PACKET = 1u << 0,
  LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_FINITE = 1u << 1,
  LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_NOT_SUBNORMAL = 1u << 2,
  LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_ZERO_REPAIR_PACKETS = 1u << 3,
  LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PACKED_SHIFT_PACKET = 1u << 4,
} loom_amdgpu_fp8_packed_bf16_missing_requirement_bits_t;
typedef uint32_t loom_amdgpu_fp8_packed_bf16_missing_requirements_t;

typedef enum loom_amdgpu_fp8_packed_u16_repair_bits_e {
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NONE = 0u,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_ZERO = 1u << 0,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_SUBNORMAL = 1u << 1,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN = 1u << 2,
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF = 1u << 3,
} loom_amdgpu_fp8_packed_u16_repair_bits_t;
typedef uint32_t loom_amdgpu_fp8_packed_u16_repairs_t;

typedef uint8_t loom_amdgpu_fp8_packed_bf16_strategy_t;
enum loom_amdgpu_fp8_packed_bf16_strategy_e {
  // Emit the packed normal payload plus any branchless zero repair.
  LOOM_AMDGPU_FP8_PACKED_BF16_STRATEGY_NORMAL = 0,
  // Emit the exact packed software repair path.
  LOOM_AMDGPU_FP8_PACKED_BF16_STRATEGY_EXACT_REPAIR = 1,
  // Convert through exact packed F16 arithmetic before producing BF16.
  LOOM_AMDGPU_FP8_PACKED_BF16_STRATEGY_EXACT_VIA_F16 = 2,
};

typedef uint8_t loom_amdgpu_fp8_packed_f16_strategy_t;
enum loom_amdgpu_fp8_packed_f16_strategy_e {
  // Emit the packed normal payload plus any branchless zero repair.
  LOOM_AMDGPU_FP8_PACKED_F16_STRATEGY_NORMAL = 0,
  // Emit the exact packed software repair path.
  LOOM_AMDGPU_FP8_PACKED_F16_STRATEGY_EXACT_REPAIR = 1,
};

enum {
  LOOM_AMDGPU_FP8_U16_BYTE_COUNT = 2u,
  LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT = 2u,
  // Number of FP8 lanes stored in one physical VGPR.
  LOOM_AMDGPU_FP8_REGISTER_BYTE_COUNT = 4u,
};

enum {
  LOOM_AMDGPU_FP8_BF16_BYTE_COUNT = 2u,
  LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT = 2u,
  // Raw f32 bit pattern for the identity scale used with scale-f32 packets.
  LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS = 0x3F800000u,
  // Raw E8M0FNU byte payload for a scale of 1.0.
  LOOM_AMDGPU_FP8_E8M0FNU_IDENTITY_SCALE_BYTE = 0x7Fu,
  // Packed E8M0FNU identity scale payload for scale-pk8 packets.
  LOOM_AMDGPU_FP8_E8M0FNU_PACKED_IDENTITY_SCALE_BITS = 0x7F7F7F7Fu,
};

typedef struct loom_amdgpu_fp8_native_descriptor_refs_t {
  // Scalar native conversion descriptor, when the target exposes one.
  loom_amdgpu_descriptor_ref_t lane;
  // Packed pair native conversion descriptor.
  loom_amdgpu_descriptor_ref_t pair;
  // Scalar native conversion descriptors indexed by source-register byte.
  loom_amdgpu_descriptor_ref_t byte_select[LOOM_AMDGPU_FP8_REGISTER_BYTE_COUNT];
} loom_amdgpu_fp8_native_descriptor_refs_t;

typedef enum loom_amdgpu_fp8_native_descriptor_flag_bits_e {
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE = 1u << 0,
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR = 1u << 1,
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_BYTE_SELECT_FAMILY = 1u << 2,
} loom_amdgpu_fp8_native_descriptor_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_native_descriptor_flags_t;

typedef struct loom_amdgpu_fp8_native_descriptors_t {
  // Native packet descriptors available in the active descriptor set.
  loom_amdgpu_fp8_native_descriptor_flags_t flags;
  // Scalar native conversion descriptor, when flags include HAS_LANE.
  loom_low_lower_resolved_descriptor_t lane_descriptor;
  // Packed pair native conversion descriptor, when flags include HAS_PAIR.
  loom_low_lower_resolved_descriptor_t pair_descriptor;
} loom_amdgpu_fp8_native_descriptors_t;

typedef struct loom_amdgpu_fp8_decode_plan_t {
  // Available native packet helpers selected from the active descriptor set.
  loom_amdgpu_fp8_decode_plan_flags_t flags;
  // Derived route capabilities computed from descriptor flags and FP8 format.
  loom_amdgpu_fp8_decode_plan_capabilities_t capabilities;
  // Parsed FP8 source format.
  loom_scalar_type_fp8_format_t format;
  // Packed unsigned BF16 subnormal payload tables for two-bit mantissas.
  uint32_t subnormal_bf16_table_words[2];
  // Packed BF16 subnormal payload byte tables for three-bit mantissas.
  uint32_t subnormal_bf16_byte_table_words
      [LOOM_AMDGPU_FP8_BF16_BYTE_COUNT]
      [LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT];
  // Packed F16 subnormal payload byte tables.
  uint32_t
      subnormal_f16_byte_table_words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                                    [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT];
  // Unsigned VGPR bitfield extract descriptor used to isolate packed bytes.
  loom_low_lower_resolved_descriptor_t bfe_u32_descriptor;
  // Optional signed equality compare descriptor with an inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_eq_i32_src1_inline_descriptor;
  // Optional signed not-equal compare descriptor with an inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_ne_i32_src1_inline_descriptor;
  // Optional unsigned 64-bit not-equal scalar compare descriptor.
  loom_low_lower_resolved_descriptor_t compare_lg_u64_descriptor;
  // Optional unsigned 64-bit not-equal compare with an inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_lg_u64_src1_inline_descriptor;
  // Optional unsigned-greater-equal compare descriptor with inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_uge_u32_src1_inline_descriptor;
  // Optional unsigned-less-than compare descriptor with an inline RHS operand.
  loom_low_lower_resolved_descriptor_t compare_ult_u32_src1_inline_descriptor;
  // Packed low-16-bit pair descriptor used to combine BF16 lanes.
  loom_low_lower_resolved_descriptor_t pack_u16_descriptor;
  // Byte permute descriptor used to select tiny FP8 subnormal BF16 tables.
  loom_low_lower_resolved_descriptor_t perm_b32_descriptor;
  // Byte permute descriptor with an immediate selector operand.
  loom_low_lower_resolved_descriptor_t perm_b32_src2_literal_descriptor;
  // Byte permute descriptor with zero SRC1 and an immediate selector operand.
  loom_low_lower_resolved_descriptor_t
      perm_b32_src1_zero_src2_literal_descriptor;
  // Native F32-pair-to-BF16-pair conversion descriptor.
  loom_low_lower_resolved_descriptor_t native_bf16_pack_descriptor;
  // Integer three-input add descriptor with a source-2 literal.
  loom_low_lower_resolved_descriptor_t add3_src2_literal_descriptor;
  // Integer left-shift-add descriptor with an immediate shift.
  loom_low_lower_resolved_descriptor_t lshl_add_u32_shift_imm_descriptor;
  // Integer left-shift-add descriptor with immediate shift and source-2
  // literal.
  loom_low_lower_resolved_descriptor_t
      lshl_add_u32_shift_imm_src2_literal_descriptor;
  // Packed unsigned 16-bit min descriptor.
  loom_low_lower_resolved_descriptor_t pk_min_u16_descriptor;
  // Packed unsigned 16-bit low-multiply descriptor.
  loom_low_lower_resolved_descriptor_t pk_mul_lo_u16_descriptor;
  // Packed unsigned 16-bit add descriptor.
  loom_low_lower_resolved_descriptor_t pk_add_u16_descriptor;
  // Packed 16-bit left-shift descriptor.
  loom_low_lower_resolved_descriptor_t pk_lshlrev_b16_descriptor;
  // Packed logical 16-bit right-shift descriptor.
  loom_low_lower_resolved_descriptor_t pk_lshrrev_b16_descriptor;
  // Packed F16 multiply descriptor.
  loom_low_lower_resolved_descriptor_t pk_mul_f16_descriptor;
  // Packed unsigned 16-bit multiply-add descriptor.
  loom_low_lower_resolved_descriptor_t pk_mad_u16_descriptor;
  // Packed unsigned 16-bit multiply-add descriptor with source-2 literal.
  loom_low_lower_resolved_descriptor_t pk_mad_u16_src2_literal_descriptor;
  // Packed unsigned 16-bit max descriptor.
  loom_low_lower_resolved_descriptor_t pk_max_u16_descriptor;
  // Packed signed 16-bit arithmetic-right-shift descriptor.
  loom_low_lower_resolved_descriptor_t pk_ashrrev_i16_descriptor;
  // Bitfield insert descriptor with all operands in registers.
  loom_low_lower_resolved_descriptor_t bfi_b32_descriptor;
  // Bitfield insert descriptor with an inline mask operand.
  loom_low_lower_resolved_descriptor_t bfi_b32_src0_literal_descriptor;
} loom_amdgpu_fp8_decode_plan_t;

typedef struct loom_amdgpu_fp8_packed_u16_pair_source_t {
  // Source register containing the selected adjacent FP8 byte pair.
  loom_value_id_t source_register;
  // First FP8 byte offset within source_register.
  uint32_t byte_offset;
  // Number of live result lanes consumed from the source pair.
  uint32_t live_lane_count;
} loom_amdgpu_fp8_packed_u16_pair_source_t;

// Maps one supported FP8 numeric-format bit to its compact table index.
bool loom_amdgpu_fp8_format_index(
    loom_value_fact_numeric_format_flags_t numeric_format,
    uint32_t* out_format_index);

// Returns the exact binary format and physical scalar carrier for one
// supported FP8 numeric-format bit.
bool loom_amdgpu_fp8_format(
    loom_value_fact_numeric_format_flags_t numeric_format,
    loom_scalar_type_t* out_element_type,
    loom_scalar_type_fp8_format_t* out_format);

// Returns the native unscaled FP8/BF8 conversion descriptor refs for the
// descriptor-compatible source format and result element type pair.
bool loom_amdgpu_fp8_native_descriptor_refs(
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs);

// Returns native unscaled FP8/BF8 conversion descriptors for the
// descriptor-compatible source format and result element type pair. The
// descriptor pointer remains valid until the current loom_low_lower_function
// call returns.
iree_status_t loom_amdgpu_get_fp8_native_descriptors(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_scalar_type_t result_element_type,
    const loom_amdgpu_fp8_native_descriptors_t** out_descriptors);

// Returns the native scaled FP8/BF8 conversion descriptor ref for the
// descriptor-compatible source format and result element type pair.
bool loom_amdgpu_fp8_scalef32_descriptor_ref(
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_ref);

// Returns the native scaled FP8/BF8 conversion descriptor for the
// descriptor-compatible source format and result element type pair, when the
// active target exposes one. The descriptor pointer remains valid until the
// current loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_fp8_scalef32_descriptor(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_scalar_type_t result_element_type,
    const loom_low_lower_resolved_descriptor_t** out_descriptor);

// Returns the native scaled FP8/BF8 conversion descriptor ref for
// descriptor-compatible source format/result type pairs using E8M0 scale
// packets.
bool loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_ref);

// Returns the native E8M0 scale-pk8 FP8/BF8 conversion descriptor for the
// descriptor-compatible source format and result element type pair, when
// available. The descriptor pointer remains valid until the current
// loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_fp8_e8m0_pk8_descriptor(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_scalar_type_t result_element_type,
    const loom_low_lower_resolved_descriptor_t** out_descriptor);

// Returns descriptor helpers and exact format tables for |source_format|.
// |descriptor_source_format| is the semantically compatible format used only
// for native descriptor availability. The plan is function-local target
// lowering state and remains valid until the current loom_low_lower_function
// call returns.
iree_status_t loom_amdgpu_get_fp8_decode_plan(
    loom_low_lower_context_t* context,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    const loom_amdgpu_fp8_decode_plan_t** out_plan);

// Initializes descriptor-availability flags and exact format tables without
// resolving low descriptors for emission.
void loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_value_fact_numeric_format_flags_t source_format,
    loom_value_fact_numeric_format_flags_t descriptor_source_format,
    loom_amdgpu_fp8_decode_plan_t* out_plan);

// Returns the target packet and value-fact requirements missing from the
// packed FP8-to-BF16 pair decode path.
loom_amdgpu_fp8_packed_bf16_missing_requirements_t
loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Returns the special-value repairs emitted by the packed FP8-to-BF16 pair
// decode path for |plan| and |value_flags|.
loom_amdgpu_fp8_packed_u16_repairs_t
loom_amdgpu_fp8_pair_to_packed_bf16_repairs(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Selects the packed BF16 emission strategy from target capabilities and value
// facts. The caller must first prove that packed BF16 emission is available.
loom_amdgpu_fp8_packed_bf16_strategy_t
loom_amdgpu_select_fp8_packed_bf16_strategy(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Selects the finite packed F16 emission strategy from target capabilities and
// value facts. The caller must first prove that packed F16 emission is
// available.
loom_amdgpu_fp8_packed_f16_strategy_t
loom_amdgpu_select_fp8_packed_f16_strategy(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Returns the structured report key for the selected packed BF16 strategy.
iree_string_view_t loom_amdgpu_fp8_packed_bf16_strategy_key(
    loom_amdgpu_fp8_packed_bf16_strategy_t strategy,
    loom_amdgpu_fp8_packed_u16_repairs_t repairs);

// Returns the zero/subnormal repairs emitted by the packed finite FP8-to-F16
// pair decode path for |plan| and |value_flags|.
loom_amdgpu_fp8_packed_u16_repairs_t loom_amdgpu_fp8_pair_to_packed_f16_repairs(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Returns the structured report reason key for packed F16 decode repairs.
iree_string_view_t loom_amdgpu_fp8_packed_f16_repair_reason_key(
    loom_amdgpu_fp8_packed_u16_repairs_t repairs);

// Emits one 16-bit BF16 bit payload from an unsigned FP8 byte payload.
iree_status_t loom_amdgpu_emit_fp8_to_bf16_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type, loom_value_id_t* out_lane);

// Emits one 32-bit F32 bit payload from an unsigned FP8 byte payload.
iree_status_t loom_amdgpu_emit_fp8_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type, loom_value_id_t* out_lane);

// Emits one 32-bit F32 bit payload when |value_flags| prove no subnormal
// reconstruction is required. Leaves |out_lane| invalid when the value facts
// are not strong enough.
iree_status_t loom_amdgpu_try_emit_fp8_not_subnormal_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan, loom_value_id_t low_byte,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_lane);

// Returns true when the target packets and value facts can use the packed BF16
// pair decode path without per-lane special-value repair.
bool loom_amdgpu_can_emit_fp8_pair_to_packed_bf16(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Returns true when the packed BF16 pair decode path should be selected ahead
// of native FP8-to-F32 pair decode followed by explicit BF16 packing.
bool loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Emits packed BF16 registers for adjacent FP8 byte-pair sources using the
// selected strategy and repairs. Exact rare subnormal repair is batched across
// all pairs so the hot path branches once per packet instead of once per pair.
iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_packed_bf16_strategy_t strategy,
    loom_amdgpu_fp8_packed_u16_repairs_t repairs, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packets);

// Returns true when the target packets and value facts can use the packed
// finite FP8-to-F16 pair decode path. Finite-only inputs may require exact
// zero/subnormal repair while finite-not-subnormal inputs stay on the normal
// fast path.
bool loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Emits packed F16 registers for adjacent FP8 byte-pair sources using the
// selected strategy and repairs when value facts prove finite storage.
iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count, loom_amdgpu_fp8_packed_f16_strategy_t strategy,
    loom_amdgpu_fp8_packed_u16_repairs_t repairs, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packets);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_H_
