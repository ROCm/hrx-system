// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V semantic scalar type facts.
//
// These model the scalar element types Loom selects for SPIR-V type
// declarations and target legality. They are intentionally separate from
// SPIR-V wire enum families such as SpvComponentType, which are only used by
// APIs or instructions that literally take those enum values.

#ifndef LOOM_TARGET_ARCH_SPIRV_SCALAR_TYPES_H_
#define LOOM_TARGET_ARCH_SPIRV_SCALAR_TYPES_H_

#include "iree/base/api.h"
#include "loom/target/arch/spirv/features.h"
#include "loom/util/numeric_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_spirv_scalar_type_kind_e {
  // Unknown or uninitialized scalar type kind.
  LOOM_SPIRV_SCALAR_TYPE_KIND_UNKNOWN = 0,
  // Scalar type declared with OpTypeFloat.
  LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT = 1,
  // Scalar type declared with signed OpTypeInt.
  LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT = 2,
  // Scalar type declared with unsigned OpTypeInt.
  LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT = 3,
} loom_spirv_scalar_type_kind_t;

typedef enum loom_spirv_scalar_type_e {
  // Unknown or uninitialized scalar type.
  LOOM_SPIRV_SCALAR_TYPE_UNKNOWN = 0,
  // IEEE binary16 floating-point scalar type.
  LOOM_SPIRV_SCALAR_TYPE_F16 = 1,
  // IEEE binary32 floating-point scalar type.
  LOOM_SPIRV_SCALAR_TYPE_F32 = 2,
  // IEEE binary64 floating-point scalar type.
  LOOM_SPIRV_SCALAR_TYPE_F64 = 3,
  // KHR bfloat16 scalar type, declared as OpTypeFloat 16 BFloat16KHR.
  LOOM_SPIRV_SCALAR_TYPE_BF16 = 4,
  // Signed 8-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S8 = 5,
  // Signed 16-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S16 = 6,
  // Signed 32-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S32 = 7,
  // Signed 64-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_S64 = 8,
  // Unsigned 8-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U8 = 9,
  // Unsigned 16-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U16 = 10,
  // Unsigned 32-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U32 = 11,
  // Unsigned 64-bit integer scalar type.
  LOOM_SPIRV_SCALAR_TYPE_U64 = 12,
} loom_spirv_scalar_type_t;

typedef struct loom_spirv_scalar_type_descriptor_t {
  // Stable scalar type enum value.
  loom_spirv_scalar_type_t type;
  // Stable diagnostic spelling.
  iree_string_view_t name;
  // OpTypeFloat or OpTypeInt bit width.
  uint8_t bit_width;
  // SPIR-V type declaration family.
  loom_spirv_scalar_type_kind_t kind;
  // Optional FP Encoding operand, or LOOM_SPIRV_FP_ENCODING_MAX for default.
  loom_spirv_fp_encoding_t fp_encoding;
  // Feature bits required to declare or use this scalar type.
  loom_spirv_feature_bits_t required_feature_bits;
} loom_spirv_scalar_type_descriptor_t;

// Returns the scalar type descriptor, or NULL when |type| is unknown.
const loom_spirv_scalar_type_descriptor_t* loom_spirv_scalar_type_descriptor(
    loom_spirv_scalar_type_t type);

// Returns the stable diagnostic spelling for a scalar type.
iree_string_view_t loom_spirv_scalar_type_name(loom_spirv_scalar_type_t type);

// Maps a target-independent numeric format to a SPIR-V semantic scalar type.
// Returns false when the numeric format is unknown, multi-valued, or not
// representable by the currently modeled core/KHR scalar type vocabulary.
bool loom_spirv_scalar_type_from_numeric_format(
    loom_value_fact_numeric_format_flags_t numeric_format,
    loom_spirv_scalar_type_t* out_scalar_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_SCALAR_TYPES_H_
