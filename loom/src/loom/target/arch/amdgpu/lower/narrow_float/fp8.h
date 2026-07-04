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
} loom_amdgpu_fp8_decode_plan_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_decode_plan_flags_t;

// Per-value facts that can simplify the generated special-value decode path.
// These describe the actual value being decoded, not the full source FP8 type.
typedef enum loom_amdgpu_fp8_decode_value_flag_bits_e {
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_NAN = 1u << 0,
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_INF = 1u << 1,
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NOT_SUBNORMAL = 1u << 2,
  LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NON_ZERO = 1u << 3,
} loom_amdgpu_fp8_decode_value_flag_bits_t;
typedef uint32_t loom_amdgpu_fp8_decode_value_flags_t;

// Maps target-independent value facts into FP8 decode simplification flags.
loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_fp8_decode_value_flags_from_facts(loom_value_facts_t facts);

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

enum {
  LOOM_AMDGPU_FP8_U16_BYTE_COUNT = 2u,
  LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT = 2u,
};

enum {
  LOOM_AMDGPU_FP8_BF16_BYTE_COUNT = 2u,
  LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT = 2u,
  // Raw f32 bit pattern for the identity scale used with scale-f32 packets.
  LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS = 0x3F800000u,
};

typedef struct loom_amdgpu_fp8_native_descriptor_refs_t {
  // Scalar native conversion descriptor, when the target exposes one.
  loom_amdgpu_descriptor_ref_t lane;
  // Packed pair native conversion descriptor.
  loom_amdgpu_descriptor_ref_t pair;
} loom_amdgpu_fp8_native_descriptor_refs_t;

typedef enum loom_amdgpu_fp8_native_descriptor_flag_bits_e {
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_NONE = 0u,
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE = 1u << 0,
  LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR = 1u << 1,
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
  // Native packed-pair FP8-to-F32 conversion descriptor.
  loom_low_lower_resolved_descriptor_t native_f32_pair_descriptor;
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
  // Packed unsigned 16-bit multiply-add descriptor.
  loom_low_lower_resolved_descriptor_t pk_mad_u16_descriptor;
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

// Returns the native unscaled FP8/BF8 conversion descriptor refs for the source
// and result element type pair.
bool loom_amdgpu_fp8_native_descriptor_refs(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs);

// Returns native unscaled FP8/BF8 conversion descriptors for the source and
// result element type pair. The descriptor pointer remains valid until the
// current loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_fp8_native_descriptors(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    const loom_amdgpu_fp8_native_descriptors_t** out_descriptors);

// Returns the native scaled FP8/BF8 conversion descriptor ref for the source
// and result element type pair.
bool loom_amdgpu_fp8_scalef32_descriptor_ref(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_ref);

// Returns the native scaled FP8/BF8 conversion descriptor for the source and
// result element type pair, when the active target exposes one. The descriptor
// pointer remains valid until the current loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_fp8_scalef32_descriptor(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    const loom_low_lower_resolved_descriptor_t** out_descriptor);

// Returns descriptor helpers and format tables for |element_type|. The plan is
// function-local target lowering state and remains valid until the current
// loom_low_lower_function call returns.
iree_status_t loom_amdgpu_get_fp8_decode_plan(
    loom_low_lower_context_t* context, loom_scalar_type_t element_type,
    const loom_amdgpu_fp8_decode_plan_t** out_plan);

// Initializes descriptor-availability flags and format tables for
// |element_type| without resolving low descriptors for emission.
void loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t element_type, loom_amdgpu_fp8_decode_plan_t* out_plan);

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

// Returns the structured report reason key for packed BF16 decode repairs.
iree_string_view_t loom_amdgpu_fp8_packed_bf16_repair_reason_key(
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

// Emits packed BF16 registers for adjacent FP8 byte-pair sources. Exact rare
// subnormal repair is batched across all pairs so the hot path branches once
// per packet instead of once per pair.
iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packets);

// Returns true when the target packets and value facts can use the packed
// finite FP8-to-F16 pair decode path. Finite-only inputs may require exact
// zero/subnormal repair while finite-not-subnormal inputs stay on the normal
// fast path.
bool loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(
    const loom_amdgpu_fp8_decode_plan_t* plan,
    loom_amdgpu_fp8_decode_value_flags_t value_flags);

// Emits packed F16 registers for adjacent FP8 byte-pair sources when value
// facts prove finite storage.
iree_status_t loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* plan,
    const loom_amdgpu_fp8_packed_u16_pair_source_t* pair_sources,
    iree_host_size_t pair_count,
    loom_amdgpu_fp8_decode_value_flags_t value_flags, loom_type_t vgpr_type,
    loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packets);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_NARROW_FLOAT_FP8_H_
