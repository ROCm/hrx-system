// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/scalar_types.h"

static const loom_spirv_scalar_type_descriptor_t kScalarTypeDescriptors[] = {
    [LOOM_SPIRV_SCALAR_TYPE_UNKNOWN] = {0},
    [LOOM_SPIRV_SCALAR_TYPE_F16] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_F16,
            .name = IREE_SVL("f16"),
            .bit_width = 16,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_FLOAT16,
        },
    [LOOM_SPIRV_SCALAR_TYPE_F32] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_F32,
            .name = IREE_SVL("f32"),
            .bit_width = 32,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = 0,
        },
    [LOOM_SPIRV_SCALAR_TYPE_F64] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_F64,
            .name = IREE_SVL("f64"),
            .bit_width = 64,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_FLOAT64,
        },
    [LOOM_SPIRV_SCALAR_TYPE_BF16] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_BF16,
            .name = IREE_SVL("bf16"),
            .bit_width = 16,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_B_FLOAT16_KHR,
            .required_feature_bits = LOOM_SPIRV_FEATURE_BFLOAT16_TYPE_KHR,
        },
    [LOOM_SPIRV_SCALAR_TYPE_S8] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_S8,
            .name = IREE_SVL("s8"),
            .bit_width = 8,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_INT8,
        },
    [LOOM_SPIRV_SCALAR_TYPE_S16] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_S16,
            .name = IREE_SVL("s16"),
            .bit_width = 16,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_INT16,
        },
    [LOOM_SPIRV_SCALAR_TYPE_S32] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_S32,
            .name = IREE_SVL("s32"),
            .bit_width = 32,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = 0,
        },
    [LOOM_SPIRV_SCALAR_TYPE_S64] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_S64,
            .name = IREE_SVL("s64"),
            .bit_width = 64,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_SIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_INT64,
        },
    [LOOM_SPIRV_SCALAR_TYPE_U8] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_U8,
            .name = IREE_SVL("u8"),
            .bit_width = 8,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_INT8,
        },
    [LOOM_SPIRV_SCALAR_TYPE_U16] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_U16,
            .name = IREE_SVL("u16"),
            .bit_width = 16,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_INT16,
        },
    [LOOM_SPIRV_SCALAR_TYPE_U32] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_U32,
            .name = IREE_SVL("u32"),
            .bit_width = 32,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = 0,
        },
    [LOOM_SPIRV_SCALAR_TYPE_U64] =
        {
            .type = LOOM_SPIRV_SCALAR_TYPE_U64,
            .name = IREE_SVL("u64"),
            .bit_width = 64,
            .kind = LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT,
            .fp_encoding = LOOM_SPIRV_FP_ENCODING_MAX,
            .required_feature_bits = LOOM_SPIRV_FEATURE_INT64,
        },
};

const loom_spirv_scalar_type_descriptor_t* loom_spirv_scalar_type_descriptor(
    loom_spirv_scalar_type_t type) {
  if ((uint32_t)type >= IREE_ARRAYSIZE(kScalarTypeDescriptors) ||
      type == LOOM_SPIRV_SCALAR_TYPE_UNKNOWN) {
    return NULL;
  }
  return &kScalarTypeDescriptors[type];
}

iree_string_view_t loom_spirv_scalar_type_name(loom_spirv_scalar_type_t type) {
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(type);
  return descriptor != NULL ? descriptor->name : IREE_SV("unknown");
}

bool loom_spirv_scalar_type_from_numeric_format(
    loom_value_fact_numeric_format_flags_t numeric_format,
    loom_spirv_scalar_type_t* out_scalar_type) {
  IREE_ASSERT_ARGUMENT(out_scalar_type);
  *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_UNKNOWN;
  switch (numeric_format) {
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F64:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F64;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_F16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_F16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_BF16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_I8:
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_S8;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U32:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U32;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U16:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U16;
      return true;
    case LOOM_VALUE_FACT_NUMERIC_FORMAT_U8:
      *out_scalar_type = LOOM_SPIRV_SCALAR_TYPE_U8;
      return true;
    default:
      return false;
  }
}
