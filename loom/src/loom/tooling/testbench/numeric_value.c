// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/numeric_value.h"

#include <string.h>

#include "iree/base/internal/math.h"

static double loom_testbench_numeric_load_f8e4m3_f64(const uint8_t* data) {
  return (double)iree_math_f8e4m3fn_to_f32(data[0]);
}

static double loom_testbench_numeric_load_f8e5m2_f64(const uint8_t* data) {
  return (double)iree_math_f8e5m2_to_f32(data[0]);
}

static double loom_testbench_numeric_load_f16_f64(const uint8_t* data) {
  uint16_t bits = 0;
  memcpy(&bits, data, sizeof(bits));
  return (double)iree_math_f16_to_f32(bits);
}

static double loom_testbench_numeric_load_bf16_f64(const uint8_t* data) {
  uint16_t bits = 0;
  memcpy(&bits, data, sizeof(bits));
  return (double)iree_math_bf16_to_f32(bits);
}

static double loom_testbench_numeric_load_f32_f64(const uint8_t* data) {
  float value = 0.0f;
  memcpy(&value, data, sizeof(value));
  return (double)value;
}

static double loom_testbench_numeric_load_f64_f64(const uint8_t* data) {
  double value = 0.0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static int64_t loom_testbench_numeric_load_i8_i64(const uint8_t* data) {
  int8_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_i8_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_i8_i64(data);
}

static int64_t loom_testbench_numeric_load_u8_i64(const uint8_t* data) {
  return data[0];
}

static uint64_t loom_testbench_numeric_load_u8_u64(const uint8_t* data) {
  return data[0];
}

static double loom_testbench_numeric_load_u8_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_u8_i64(data);
}

static int64_t loom_testbench_numeric_load_i16_i64(const uint8_t* data) {
  int16_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_i16_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_i16_i64(data);
}

static int64_t loom_testbench_numeric_load_u16_i64(const uint8_t* data) {
  uint16_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static uint64_t loom_testbench_numeric_load_u16_u64(const uint8_t* data) {
  uint16_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_u16_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_u16_i64(data);
}

static int64_t loom_testbench_numeric_load_i32_i64(const uint8_t* data) {
  int32_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_i32_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_i32_i64(data);
}

static int64_t loom_testbench_numeric_load_u32_i64(const uint8_t* data) {
  uint32_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static uint64_t loom_testbench_numeric_load_u32_u64(const uint8_t* data) {
  uint32_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_u32_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_u32_i64(data);
}

static int64_t loom_testbench_numeric_load_i64_i64(const uint8_t* data) {
  int64_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_i64_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_i64_i64(data);
}

static uint64_t loom_testbench_numeric_load_u64_u64(const uint8_t* data) {
  uint64_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

static double loom_testbench_numeric_load_u64_f64(const uint8_t* data) {
  return (double)loom_testbench_numeric_load_u64_u64(data);
}

static const loom_testbench_numeric_type_t kLoomTestbenchNumericTypes[] = {
    [LOOM_TESTBENCH_NUMERIC_F8E4M3] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_F8E4M3,
            .name = "f8e4m3",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN,
            .byte_count = 1,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT,
            .load_f64 = loom_testbench_numeric_load_f8e4m3_f64,
            .load_i64 = NULL,
            .minimum_normal_f64 = 0x1p-6,
        },
    [LOOM_TESTBENCH_NUMERIC_F8E5M2] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_F8E5M2,
            .name = "f8e5m2",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2,
            .byte_count = 1,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT,
            .load_f64 = loom_testbench_numeric_load_f8e5m2_f64,
            .load_i64 = NULL,
            .minimum_normal_f64 = 0x1p-14,
        },
    [LOOM_TESTBENCH_NUMERIC_F16] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_F16,
            .name = "f16",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16,
            .byte_count = 2,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT,
            .load_f64 = loom_testbench_numeric_load_f16_f64,
            .load_i64 = NULL,
            .minimum_normal_f64 = 0x1p-14,
        },
    [LOOM_TESTBENCH_NUMERIC_BF16] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_BF16,
            .name = "bf16",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
            .byte_count = 2,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT,
            .load_f64 = loom_testbench_numeric_load_bf16_f64,
            .load_i64 = NULL,
            .minimum_normal_f64 = 0x1p-126,
        },
    [LOOM_TESTBENCH_NUMERIC_F32] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_F32,
            .name = "f32",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32,
            .byte_count = 4,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT,
            .load_f64 = loom_testbench_numeric_load_f32_f64,
            .load_i64 = NULL,
            .minimum_normal_f64 = 0x1p-126,
        },
    [LOOM_TESTBENCH_NUMERIC_F64] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_F64,
            .name = "f64",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64,
            .byte_count = 8,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT,
            .load_f64 = loom_testbench_numeric_load_f64_f64,
            .load_i64 = NULL,
            .minimum_normal_f64 = 0x1p-1022,
        },
    [LOOM_TESTBENCH_NUMERIC_I8] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_I8,
            .name = "i8",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_SINT_8,
            .byte_count = 1,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_i8_f64,
            .load_i64 = loom_testbench_numeric_load_i8_i64,
            .minimum_i64 = INT8_MIN,
            .maximum_i64 = INT8_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_U8] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_U8,
            .name = "u8",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_UINT_8,
            .byte_count = 1,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_u8_f64,
            .load_i64 = loom_testbench_numeric_load_u8_i64,
            .load_u64 = loom_testbench_numeric_load_u8_u64,
            .minimum_i64 = 0,
            .maximum_i64 = UINT8_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_I16] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_I16,
            .name = "i16",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_SINT_16,
            .byte_count = 2,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_i16_f64,
            .load_i64 = loom_testbench_numeric_load_i16_i64,
            .minimum_i64 = INT16_MIN,
            .maximum_i64 = INT16_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_U16] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_U16,
            .name = "u16",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_UINT_16,
            .byte_count = 2,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_u16_f64,
            .load_i64 = loom_testbench_numeric_load_u16_i64,
            .load_u64 = loom_testbench_numeric_load_u16_u64,
            .minimum_i64 = 0,
            .maximum_i64 = UINT16_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_I32] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_I32,
            .name = "i32",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_SINT_32,
            .byte_count = 4,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_i32_f64,
            .load_i64 = loom_testbench_numeric_load_i32_i64,
            .minimum_i64 = INT32_MIN,
            .maximum_i64 = INT32_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_U32] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_U32,
            .name = "u32",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_UINT_32,
            .byte_count = 4,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_u32_f64,
            .load_i64 = loom_testbench_numeric_load_u32_i64,
            .load_u64 = loom_testbench_numeric_load_u32_u64,
            .minimum_i64 = 0,
            .maximum_i64 = UINT32_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_I64] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_I64,
            .name = "i64",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_SINT_64,
            .byte_count = 8,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_i64_f64,
            .load_i64 = loom_testbench_numeric_load_i64_i64,
            .minimum_i64 = INT64_MIN,
            .maximum_i64 = INT64_MAX,
        },
    [LOOM_TESTBENCH_NUMERIC_U64] =
        {
            .kind = LOOM_TESTBENCH_NUMERIC_U64,
            .name = "u64",
            .hal_element_type = IREE_HAL_ELEMENT_TYPE_UINT_64,
            .byte_count = 8,
            .domain = LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER,
            .load_f64 = loom_testbench_numeric_load_u64_f64,
            .load_i64 = NULL,
            .load_u64 = loom_testbench_numeric_load_u64_u64,
            .minimum_i64 = 0,
            .maximum_i64 = INT64_MAX,
        },
};

static_assert(IREE_ARRAYSIZE(kLoomTestbenchNumericTypes) ==
                  LOOM_TESTBENCH_NUMERIC_COUNT_,
              "numeric descriptor table must cover every numeric kind");

const loom_testbench_numeric_type_t* loom_testbench_numeric_type(
    loom_testbench_numeric_kind_t kind) {
  if (kind <= LOOM_TESTBENCH_NUMERIC_NONE ||
      kind >= LOOM_TESTBENCH_NUMERIC_COUNT_) {
    return NULL;
  }
  return &kLoomTestbenchNumericTypes[kind];
}

const loom_testbench_numeric_type_t* loom_testbench_numeric_type_parse(
    iree_string_view_t value) {
  // U64 is an exact comparison interpretation, not a reference-matmul
  // contract: that reference implementation accumulates through int64_t.
  for (loom_testbench_numeric_kind_t kind = LOOM_TESTBENCH_NUMERIC_F8E4M3;
       kind < LOOM_TESTBENCH_NUMERIC_U64; ++kind) {
    const loom_testbench_numeric_type_t* numeric_type =
        loom_testbench_numeric_type(kind);
    if (iree_string_view_equal(value,
                               iree_make_cstring_view(numeric_type->name))) {
      return numeric_type;
    }
  }
  return NULL;
}

const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_hal_element_type(
    iree_hal_element_type_t element_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E4M3);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E5M2);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F16);
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_BF16);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F32);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F64);
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
    case IREE_HAL_ELEMENT_TYPE_INT_8:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8);
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U8);
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_ELEMENT_TYPE_INT_16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I16);
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U16);
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_INT_32:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I32);
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U32);
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_INT_64:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I64);
    default:
      return NULL;
  }
}

const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_scalar_type(loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E4M3);
    case LOOM_SCALAR_TYPE_F8E5M2:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F8E5M2);
    case LOOM_SCALAR_TYPE_F16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F16);
    case LOOM_SCALAR_TYPE_BF16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_BF16);
    case LOOM_SCALAR_TYPE_F32:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F32);
    case LOOM_SCALAR_TYPE_F64:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_F64);
    case LOOM_SCALAR_TYPE_I8:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I8);
    case LOOM_SCALAR_TYPE_I16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I16);
    case LOOM_SCALAR_TYPE_I32:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I32);
    case LOOM_SCALAR_TYPE_I64:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_I64);
    default:
      return NULL;
  }
}

const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_signed_scalar_type(
    loom_scalar_type_t scalar_type) {
  const loom_testbench_numeric_type_t* numeric_type =
      loom_testbench_numeric_type_from_scalar_type(scalar_type);
  return numeric_type && loom_testbench_numeric_type_is_integer(numeric_type)
             ? numeric_type
             : NULL;
}

const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_unsigned_scalar_type(
    loom_scalar_type_t scalar_type) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U8);
    case LOOM_SCALAR_TYPE_I16:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U16);
    case LOOM_SCALAR_TYPE_I32:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U32);
    case LOOM_SCALAR_TYPE_I64:
      return loom_testbench_numeric_type(LOOM_TESTBENCH_NUMERIC_U64);
    default:
      return NULL;
  }
}

bool loom_testbench_numeric_type_matches_storage(
    const loom_testbench_numeric_type_t* numeric_type,
    iree_hal_element_type_t element_type) {
  if (numeric_type->hal_element_type == element_type) {
    return true;
  }
  if ((numeric_type->kind == LOOM_TESTBENCH_NUMERIC_U8 ||
       numeric_type->kind == LOOM_TESTBENCH_NUMERIC_I8) &&
      (element_type == IREE_HAL_ELEMENT_TYPE_SINT_8 ||
       element_type == IREE_HAL_ELEMENT_TYPE_UINT_8 ||
       element_type == IREE_HAL_ELEMENT_TYPE_INT_8)) {
    return true;
  }
  if ((numeric_type->kind == LOOM_TESTBENCH_NUMERIC_U32 ||
       numeric_type->kind == LOOM_TESTBENCH_NUMERIC_I32) &&
      (element_type == IREE_HAL_ELEMENT_TYPE_SINT_32 ||
       element_type == IREE_HAL_ELEMENT_TYPE_UINT_32 ||
       element_type == IREE_HAL_ELEMENT_TYPE_INT_32)) {
    return true;
  }
  if ((numeric_type->kind == LOOM_TESTBENCH_NUMERIC_U16 ||
       numeric_type->kind == LOOM_TESTBENCH_NUMERIC_I16) &&
      (element_type == IREE_HAL_ELEMENT_TYPE_SINT_16 ||
       element_type == IREE_HAL_ELEMENT_TYPE_UINT_16 ||
       element_type == IREE_HAL_ELEMENT_TYPE_INT_16)) {
    return true;
  }
  if ((numeric_type->kind == LOOM_TESTBENCH_NUMERIC_U64 ||
       numeric_type->kind == LOOM_TESTBENCH_NUMERIC_I64) &&
      (element_type == IREE_HAL_ELEMENT_TYPE_SINT_64 ||
       element_type == IREE_HAL_ELEMENT_TYPE_UINT_64 ||
       element_type == IREE_HAL_ELEMENT_TYPE_INT_64)) {
    return true;
  }
  return false;
}
