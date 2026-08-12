// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable AMDGPU FP8 lowering tables.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_FP8_TABLES_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_FP8_TABLES_H_

#include <stddef.h>
#include <stdint.h>

#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_table_counts.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_REASON_COUNT = 16u,
  LOOM_AMDGPU_FP8_PACKED_F16_REPAIR_REASON_COUNT = 4u,
};

typedef enum loom_amdgpu_fp8_scale_group_mode_e {
  LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_NONE = 0,
  LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_ALL_LANES,
  LOOM_AMDGPU_FP8_SCALE_GROUP_MODE_OCTETS_MAX4,
} loom_amdgpu_fp8_scale_group_mode_t;

typedef struct loom_amdgpu_fp8_encoded_operand_schema_requirement_t {
  // Required scale format for the direct decode route.
  loom_value_fact_numeric_format_flags_t scale_format;
  // Required scale topology for the direct decode route.
  loom_value_fact_scale_topology_flags_t scale_topology;
  // Required affine policy for the direct decode route.
  loom_value_fact_affine_policy_flags_t affine_policy;
  // Required scale operand count for the direct decode route.
  uint32_t scale_operand_count;
  // Shape rule for scale_group_element_count.
  loom_amdgpu_fp8_scale_group_mode_t scale_group_mode;
} loom_amdgpu_fp8_encoded_operand_schema_requirement_t;

typedef struct loom_amdgpu_fp8_encoded_operand_format_row_t {
  // Scalar type owning this accepted format row.
  loom_scalar_type_t element_type;
  // Numeric formats accepted for encoded operand facts with this scalar type.
  loom_value_fact_numeric_format_flags_t encoded_operand_formats;
} loom_amdgpu_fp8_encoded_operand_format_row_t;

typedef struct loom_amdgpu_fp8_format_row_t {
  // Exact FP8 numeric format represented by this row.
  loom_value_fact_numeric_format_flags_t source_format;
  // Physical scalar carrier for encoded values of this format.
  loom_scalar_type_t element_type;
  // Exact exponent, mantissa, bias, and special-value semantics.
  loom_scalar_type_fp8_format_t format;
} loom_amdgpu_fp8_format_row_t;

typedef struct loom_amdgpu_fp8_subnormal_table_row_t {
  // Exact FP8 numeric format owning this decode table row.
  loom_value_fact_numeric_format_flags_t source_format;
  // Scalar type owning this FP8/BF8 decode table row.
  loom_scalar_type_t element_type;
  // Encoded FP8 source format used by the decode plan.
  loom_scalar_type_fp8_format_t format;
  // Packed unsigned BF16 payloads for two-bit mantissas.
  uint32_t subnormal_bf16_table_words[2];
  // Packed BF16 payload byte tables for three-bit mantissas.
  uint32_t subnormal_bf16_byte_table_words
      [LOOM_AMDGPU_FP8_BF16_BYTE_COUNT]
      [LOOM_AMDGPU_FP8_BF16_BYTE_TABLE_WORD_COUNT];
  // Packed F16 payload byte tables.
  uint32_t
      subnormal_f16_byte_table_words[LOOM_AMDGPU_FP8_U16_BYTE_COUNT]
                                    [LOOM_AMDGPU_FP8_U16_BYTE_TABLE_WORD_COUNT];
} loom_amdgpu_fp8_subnormal_table_row_t;

typedef struct loom_amdgpu_fp8_native_descriptor_ref_row_t {
  // Exact encoded FP8/BF8 source numeric format.
  loom_value_fact_numeric_format_flags_t source_format;
  // Decoded result scalar type.
  loom_scalar_type_t result_element_type;
  // Native unscaled descriptor refs for the type pair.
  loom_amdgpu_fp8_native_descriptor_refs_t refs;
} loom_amdgpu_fp8_native_descriptor_ref_row_t;

typedef struct loom_amdgpu_fp8_scaled_descriptor_ref_row_t {
  // Exact encoded FP8/BF8 source numeric format.
  loom_value_fact_numeric_format_flags_t source_format;
  // Decoded result scalar type.
  loom_scalar_type_t result_element_type;
  // Descriptor ref for the native scale-f32 pair packet.
  loom_amdgpu_descriptor_ref_t scalef32_pair_descriptor_ref;
  // Descriptor ref for the native E8M0 scale-pk8 packet.
  loom_amdgpu_descriptor_ref_t e8m0_pk8_descriptor_ref;
} loom_amdgpu_fp8_scaled_descriptor_ref_row_t;

typedef struct loom_amdgpu_fp8_decode_plan_descriptor_row_t {
  // Descriptor ref to probe in the active target descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Byte offset of the resolved descriptor field in the decode plan.
  size_t descriptor_offset;
  // Plan flag raised when the descriptor ref is present.
  loom_amdgpu_fp8_decode_plan_flags_t present_flag;
} loom_amdgpu_fp8_decode_plan_descriptor_row_t;

extern const iree_string_view_t kLoomAmdgpuFp8PackedBf16RepairReasons
    [LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_REASON_COUNT];
extern const iree_string_view_t kLoomAmdgpuFp8PackedF16RepairReasons
    [LOOM_AMDGPU_FP8_PACKED_F16_REPAIR_REASON_COUNT];

extern const loom_amdgpu_fp8_encoded_operand_schema_requirement_t
    kLoomAmdgpuFp8EncodedOperandSchemaRequirements
        [LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_E8M0 + 1];
extern const loom_amdgpu_fp8_encoded_operand_format_row_t
    kLoomAmdgpuFp8EncodedOperandFormatRows[LOOM_SCALAR_TYPE_COUNT_];

extern const loom_amdgpu_fp8_format_row_t
    kLoomAmdgpuFp8FormatRows[LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT];
extern const loom_amdgpu_fp8_subnormal_table_row_t
    kLoomAmdgpuFp8SubnormalTableRows[LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT];

extern const loom_amdgpu_fp8_native_descriptor_ref_row_t
    kLoomAmdgpuFp8NativeDescriptorRefRows
        [LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW_COUNT];
extern const uint8_t
    kLoomAmdgpuFp8NativeDescriptorRefRowIndex[LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT]
                                             [LOOM_SCALAR_TYPE_COUNT_];

extern const loom_amdgpu_fp8_scaled_descriptor_ref_row_t
    kLoomAmdgpuFp8ScaledDescriptorRefRows
        [LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW_COUNT];
extern const uint8_t
    kLoomAmdgpuFp8ScaledDescriptorRefRowIndex[LOOM_AMDGPU_FP8_FORMAT_ROW_COUNT]
                                             [LOOM_SCALAR_TYPE_COUNT_];

extern const loom_amdgpu_fp8_decode_plan_descriptor_row_t
    kLoomAmdgpuFp8DecodePlanDescriptorRows
        [LOOM_AMDGPU_FP8_DECODE_PLAN_DESCRIPTOR_ROW_COUNT];

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ENCODING_FP8_TABLES_H_
