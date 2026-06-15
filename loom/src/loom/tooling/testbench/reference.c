// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/reference.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/modules/hal/types.h"

typedef struct loom_testbench_reference_rank2_view_t {
  // Borrowed HAL buffer view.
  iree_hal_buffer_view_t* view;
  // Element type stored by |view|.
  iree_hal_element_type_t element_type;
  // First row-major dimension.
  iree_host_size_t rows;
  // Second row-major dimension.
  iree_host_size_t columns;
} loom_testbench_reference_rank2_view_t;

typedef struct loom_testbench_reference_rank4_view_t {
  // Borrowed HAL buffer view.
  iree_hal_buffer_view_t* view;
  // Element type stored by |view|.
  iree_hal_element_type_t element_type;
  // First tile-grid dimension.
  iree_host_size_t outer_rows;
  // Second tile-grid dimension.
  iree_host_size_t outer_columns;
  // First per-tile matrix dimension.
  iree_host_size_t inner_rows;
  // Second per-tile matrix dimension.
  iree_host_size_t inner_columns;
} loom_testbench_reference_rank4_view_t;

typedef struct loom_testbench_reference_matmul_result_t {
  // Borrowed dense row-major result bytes.
  const uint8_t* data;
  // Number of bytes in |data|.
  iree_host_size_t data_length;
} loom_testbench_reference_matmul_result_t;

typedef enum loom_testbench_reference_numeric_kind_e {
  // Invalid or unset numeric kind.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_NONE = 0,
  // 8-bit E4M3 finite-only float.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3 = 1,
  // 8-bit E5M2 float.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E5M2 = 2,
  // IEEE f16.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_F16 = 3,
  // bfloat16.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16 = 4,
  // IEEE f32.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_F32 = 5,
  // IEEE f64.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_F64 = 6,
  // Signed 8-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_I8 = 7,
  // Unsigned 8-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_U8 = 8,
  // Signed 16-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_I16 = 9,
  // Unsigned 16-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_U16 = 10,
  // Signed 32-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_I32 = 11,
  // Unsigned 32-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_U32 = 12,
  // Signed 64-bit integer.
  LOOM_TESTBENCH_REFERENCE_NUMERIC_I64 = 13,
} loom_testbench_reference_numeric_kind_t;

typedef struct loom_testbench_reference_matmul_contract_t {
  // Numeric interpretation of lhs storage elements.
  loom_testbench_reference_numeric_kind_t lhs;
  // Numeric interpretation of rhs storage elements.
  loom_testbench_reference_numeric_kind_t rhs;
  // Numeric interpretation of init storage elements.
  loom_testbench_reference_numeric_kind_t init;
  // Numeric domain used for reference accumulation.
  loom_testbench_reference_numeric_kind_t accumulator;
  // Numeric element kind written into the result buffer view.
  loom_testbench_reference_numeric_kind_t result;
} loom_testbench_reference_matmul_contract_t;

static iree_status_t loom_testbench_reference_variant_buffer_view(
    const iree_vm_variant_t* variant, iree_hal_buffer_view_t** out_view) {
  *out_view = NULL;
  if (!iree_vm_variant_is_ref(*variant) ||
      !iree_hal_buffer_view_isa(variant->ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference oracle expects buffer view inputs");
  }
  *out_view = iree_hal_buffer_view_deref(variant->ref);
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_dim_to_host_size(
    iree_hal_dim_t dim, iree_host_size_t* out_value) {
  if (dim > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer view dimension %" PRIu64
                            " exceeds host size range",
                            (uint64_t)dim);
  }
  *out_value = (iree_host_size_t)dim;
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_rank2_view_initialize(
    iree_hal_buffer_view_t* view,
    loom_testbench_reference_rank2_view_t* out_view) {
  if (iree_hal_buffer_view_shape_rank(view) != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference.matmul expects rank-2 buffer views");
  }
  *out_view = (loom_testbench_reference_rank2_view_t){
      .view = view,
      .element_type = iree_hal_buffer_view_element_type(view),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 0), &out_view->rows));
  return loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 1), &out_view->columns);
}

static iree_status_t loom_testbench_reference_rank4_view_initialize(
    iree_hal_buffer_view_t* view,
    loom_testbench_reference_rank4_view_t* out_view) {
  if (iree_hal_buffer_view_shape_rank(view) != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.tiled_matmul expects rank-4 buffer views");
  }
  *out_view = (loom_testbench_reference_rank4_view_t){
      .view = view,
      .element_type = iree_hal_buffer_view_element_type(view),
  };
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 0), &out_view->outer_rows));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 1), &out_view->outer_columns));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 2), &out_view->inner_rows));
  return loom_testbench_reference_dim_to_host_size(
      iree_hal_buffer_view_shape_dim(view, 3), &out_view->inner_columns);
}

static iree_string_view_t loom_testbench_reference_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static const char* loom_testbench_reference_numeric_kind_name(
    loom_testbench_reference_numeric_kind_t kind) {
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3:
      return "f8e4m3";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E5M2:
      return "f8e5m2";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F16:
      return "f16";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16:
      return "bf16";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F32:
      return "f32";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F64:
      return "f64";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I8:
      return "i8";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U8:
      return "u8";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I16:
      return "i16";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U16:
      return "u16";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I32:
      return "i32";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U32:
      return "u32";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I64:
      return "i64";
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_NONE:
      return "none";
  }
  return "unknown";
}

static bool loom_testbench_reference_numeric_kind_is_float(
    loom_testbench_reference_numeric_kind_t kind) {
  return kind >= LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3 &&
         kind <= LOOM_TESTBENCH_REFERENCE_NUMERIC_F64;
}

static bool loom_testbench_reference_numeric_kind_is_integer(
    loom_testbench_reference_numeric_kind_t kind) {
  return kind >= LOOM_TESTBENCH_REFERENCE_NUMERIC_I8 &&
         kind <= LOOM_TESTBENCH_REFERENCE_NUMERIC_I64;
}

static bool loom_testbench_reference_numeric_kind_parse(
    iree_string_view_t value,
    loom_testbench_reference_numeric_kind_t* out_kind) {
  static const struct {
    iree_string_view_t name;
    loom_testbench_reference_numeric_kind_t kind;
  } kKinds[] = {
      {IREE_SVL("f8e4m3"), LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3},
      {IREE_SVL("f8e5m2"), LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E5M2},
      {IREE_SVL("f16"), LOOM_TESTBENCH_REFERENCE_NUMERIC_F16},
      {IREE_SVL("bf16"), LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16},
      {IREE_SVL("f32"), LOOM_TESTBENCH_REFERENCE_NUMERIC_F32},
      {IREE_SVL("f64"), LOOM_TESTBENCH_REFERENCE_NUMERIC_F64},
      {IREE_SVL("i8"), LOOM_TESTBENCH_REFERENCE_NUMERIC_I8},
      {IREE_SVL("u8"), LOOM_TESTBENCH_REFERENCE_NUMERIC_U8},
      {IREE_SVL("i16"), LOOM_TESTBENCH_REFERENCE_NUMERIC_I16},
      {IREE_SVL("u16"), LOOM_TESTBENCH_REFERENCE_NUMERIC_U16},
      {IREE_SVL("i32"), LOOM_TESTBENCH_REFERENCE_NUMERIC_I32},
      {IREE_SVL("u32"), LOOM_TESTBENCH_REFERENCE_NUMERIC_U32},
      {IREE_SVL("i64"), LOOM_TESTBENCH_REFERENCE_NUMERIC_I64},
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKinds); ++i) {
    if (iree_string_view_equal(value, kKinds[i].name)) {
      *out_kind = kKinds[i].kind;
      return true;
    }
  }
  return false;
}

static bool loom_testbench_reference_numeric_kind_from_element_type(
    iree_hal_element_type_t element_type,
    loom_testbench_reference_numeric_kind_t* out_kind) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3;
      return true;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E5M2;
      return true;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_F16;
      return true;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16;
      return true;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_F32;
      return true;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_F64;
      return true;
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
    case IREE_HAL_ELEMENT_TYPE_INT_8:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_I8;
      return true;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_U8;
      return true;
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
    case IREE_HAL_ELEMENT_TYPE_INT_16:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_I16;
      return true;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_U16;
      return true;
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
    case IREE_HAL_ELEMENT_TYPE_INT_32:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_I32;
      return true;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_U32;
      return true;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
    case IREE_HAL_ELEMENT_TYPE_INT_64:
      *out_kind = LOOM_TESTBENCH_REFERENCE_NUMERIC_I64;
      return true;
    default:
      return false;
  }
}

static iree_hal_element_type_t
loom_testbench_reference_numeric_kind_hal_element_type(
    loom_testbench_reference_numeric_kind_t kind) {
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E4M3_FN;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E5M2:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_8_E5M2;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F16:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16:
      return IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F32:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F64:
      return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I8:
      return IREE_HAL_ELEMENT_TYPE_SINT_8;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U8:
      return IREE_HAL_ELEMENT_TYPE_UINT_8;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I16:
      return IREE_HAL_ELEMENT_TYPE_SINT_16;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U16:
      return IREE_HAL_ELEMENT_TYPE_UINT_16;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I32:
      return IREE_HAL_ELEMENT_TYPE_SINT_32;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U32:
      return IREE_HAL_ELEMENT_TYPE_UINT_32;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I64:
      return IREE_HAL_ELEMENT_TYPE_SINT_64;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_NONE:
      return IREE_HAL_ELEMENT_TYPE_NONE;
  }
  return IREE_HAL_ELEMENT_TYPE_NONE;
}

static iree_host_size_t loom_testbench_reference_numeric_kind_byte_count(
    loom_testbench_reference_numeric_kind_t kind) {
  return iree_hal_element_dense_byte_count(
      loom_testbench_reference_numeric_kind_hal_element_type(kind));
}

static bool loom_testbench_reference_numeric_kind_matches_storage(
    loom_testbench_reference_numeric_kind_t kind,
    iree_hal_element_type_t element_type) {
  if (loom_testbench_reference_numeric_kind_hal_element_type(kind) ==
      element_type) {
    return true;
  }
  if ((kind == LOOM_TESTBENCH_REFERENCE_NUMERIC_U8 ||
       kind == LOOM_TESTBENCH_REFERENCE_NUMERIC_I8) &&
      (element_type == IREE_HAL_ELEMENT_TYPE_SINT_8 ||
       element_type == IREE_HAL_ELEMENT_TYPE_UINT_8 ||
       element_type == IREE_HAL_ELEMENT_TYPE_INT_8)) {
    return true;
  }
  if ((kind == LOOM_TESTBENCH_REFERENCE_NUMERIC_U32 ||
       kind == LOOM_TESTBENCH_REFERENCE_NUMERIC_I32) &&
      (element_type == IREE_HAL_ELEMENT_TYPE_SINT_32 ||
       element_type == IREE_HAL_ELEMENT_TYPE_UINT_32 ||
       element_type == IREE_HAL_ELEMENT_TYPE_INT_32)) {
    return true;
  }
  return false;
}

static double loom_testbench_reference_load_float_element_at(
    loom_testbench_reference_numeric_kind_t kind, const uint8_t* data,
    iree_host_size_t element_offset) {
  const uint8_t* element_data =
      data +
      element_offset * loom_testbench_reference_numeric_kind_byte_count(kind);
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E4M3: {
      return (double)iree_math_f8e4m3fn_to_f32(element_data[0]);
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F8E5M2: {
      return (double)iree_math_f8e5m2_to_f32(element_data[0]);
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F16: {
      uint16_t bits = 0;
      memcpy(&bits, element_data, sizeof(bits));
      return (double)iree_math_f16_to_f32(bits);
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16: {
      uint16_t bits = 0;
      memcpy(&bits, element_data, sizeof(bits));
      return (double)iree_math_bf16_to_f32(bits);
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F32: {
      float value = 0.0f;
      memcpy(&value, element_data, sizeof(value));
      return (double)value;
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F64: {
      double value = 0.0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    default:
      return 0.0;
  }
}

static int64_t loom_testbench_reference_load_integer_element_at(
    loom_testbench_reference_numeric_kind_t kind, const uint8_t* data,
    iree_host_size_t element_offset) {
  const uint8_t* element_data =
      data +
      element_offset * loom_testbench_reference_numeric_kind_byte_count(kind);
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I8: {
      int8_t value = 0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U8: {
      return element_data[0];
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I16: {
      int16_t value = 0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U16: {
      uint16_t value = 0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I32: {
      int32_t value = 0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U32: {
      uint32_t value = 0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I64: {
      int64_t value = 0;
      memcpy(&value, element_data, sizeof(value));
      return value;
    }
    default:
      return 0;
  }
}

static double loom_testbench_reference_load_numeric_element_as_float_at(
    loom_testbench_reference_numeric_kind_t kind, const uint8_t* data,
    iree_host_size_t element_offset) {
  if (loom_testbench_reference_numeric_kind_is_float(kind)) {
    return loom_testbench_reference_load_float_element_at(kind, data,
                                                          element_offset);
  }
  return (double)loom_testbench_reference_load_integer_element_at(
      kind, data, element_offset);
}

static bool loom_testbench_reference_integer_kind_contains(
    loom_testbench_reference_numeric_kind_t kind, int64_t value) {
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I8:
      return value >= INT8_MIN && value <= INT8_MAX;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U8:
      return value >= 0 && value <= UINT8_MAX;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I16:
      return value >= INT16_MIN && value <= INT16_MAX;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U16:
      return value >= 0 && value <= UINT16_MAX;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I32:
      return value >= INT32_MIN && value <= INT32_MAX;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U32:
      return value >= 0 && value <= UINT32_MAX;
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I64:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_testbench_reference_store_float_element_at(
    loom_testbench_reference_numeric_kind_t kind, double value, uint8_t* data,
    iree_host_size_t element_offset) {
  uint8_t* element_data =
      data +
      element_offset * loom_testbench_reference_numeric_kind_byte_count(kind);
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F16: {
      uint16_t bits = iree_math_f32_to_f16((float)value);
      memcpy(element_data, &bits, sizeof(bits));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_BF16: {
      uint16_t bits = iree_math_f32_to_bf16((float)value);
      memcpy(element_data, &bits, sizeof(bits));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F32: {
      float stored_value = (float)value;
      memcpy(element_data, &stored_value, sizeof(stored_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_F64: {
      memcpy(element_data, &value, sizeof(value));
      return iree_ok_status();
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reference matmul cannot store floating result as %s",
          loom_testbench_reference_numeric_kind_name(kind));
  }
}

static iree_status_t loom_testbench_reference_store_integer_element_at(
    loom_testbench_reference_numeric_kind_t kind, int64_t value, uint8_t* data,
    iree_host_size_t element_offset) {
  uint8_t* element_data =
      data +
      element_offset * loom_testbench_reference_numeric_kind_byte_count(kind);
  switch (kind) {
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I8: {
      if (value < INT8_MIN || value > INT8_MAX) break;
      int8_t i8_value = (int8_t)value;
      memcpy(element_data, &i8_value, sizeof(i8_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U8: {
      if (value < 0 || value > UINT8_MAX) break;
      uint8_t u8_value = (uint8_t)value;
      memcpy(element_data, &u8_value, sizeof(u8_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I16: {
      if (value < INT16_MIN || value > INT16_MAX) break;
      int16_t i16_value = (int16_t)value;
      memcpy(element_data, &i16_value, sizeof(i16_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U16: {
      if (value < 0 || value > UINT16_MAX) break;
      uint16_t u16_value = (uint16_t)value;
      memcpy(element_data, &u16_value, sizeof(u16_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I32: {
      if (value < INT32_MIN || value > INT32_MAX) break;
      int32_t i32_value = (int32_t)value;
      memcpy(element_data, &i32_value, sizeof(i32_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_U32: {
      if (value < 0 || value > UINT32_MAX) break;
      uint32_t u32_value = (uint32_t)value;
      memcpy(element_data, &u32_value, sizeof(u32_value));
      return iree_ok_status();
    }
    case LOOM_TESTBENCH_REFERENCE_NUMERIC_I64: {
      memcpy(element_data, &value, sizeof(value));
      return iree_ok_status();
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "reference matmul cannot store integer result as %s",
          loom_testbench_reference_numeric_kind_name(kind));
  }
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                          "reference matmul result does not fit in %s",
                          loom_testbench_reference_numeric_kind_name(kind));
}

static iree_status_t loom_testbench_reference_validate_tiled_matmul_shape(
    iree_string_view_t oracle_name,
    const loom_testbench_reference_rank4_view_t* lhs,
    const loom_testbench_reference_rank4_view_t* rhs,
    const loom_testbench_reference_rank4_view_t* init) {
  if (lhs->outer_columns != rhs->outer_rows ||
      lhs->outer_rows != init->outer_rows ||
      rhs->outer_columns != init->outer_columns ||
      lhs->inner_columns != rhs->inner_rows ||
      lhs->inner_rows != init->inner_rows ||
      rhs->inner_columns != init->inner_columns) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s shape mismatch: lhs[%" PRIhsz ", %" PRIhsz ", %" PRIhsz
        ", %" PRIhsz "], rhs[%" PRIhsz ", %" PRIhsz ", %" PRIhsz ", %" PRIhsz
        "], init[%" PRIhsz ", %" PRIhsz ", %" PRIhsz ", %" PRIhsz "]",
        (int)oracle_name.size, oracle_name.data, lhs->outer_rows,
        lhs->outer_columns, lhs->inner_rows, lhs->inner_columns,
        rhs->outer_rows, rhs->outer_columns, rhs->inner_rows,
        rhs->inner_columns, init->outer_rows, init->outer_columns,
        init->inner_rows, init->inner_columns);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_validate_matmul_shape(
    iree_string_view_t oracle_name,
    const loom_testbench_reference_rank2_view_t* lhs,
    const loom_testbench_reference_rank2_view_t* rhs,
    const loom_testbench_reference_rank2_view_t* init) {
  if (lhs->columns != rhs->rows || lhs->rows != init->rows ||
      rhs->columns != init->columns) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s shape mismatch: lhs[%" PRIhsz ", %" PRIhsz "], rhs[%" PRIhsz
        ", %" PRIhsz "], init[%" PRIhsz ", %" PRIhsz "]",
        (int)oracle_name.size, oracle_name.data, lhs->rows, lhs->columns,
        rhs->rows, rhs->columns, init->rows, init->columns);
  }
  return iree_ok_status();
}

static const loom_named_attr_t* loom_testbench_reference_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_testbench_reference_module_string(module, attr->name_id),
            name)) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_testbench_reference_read_numeric_contract_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name,
    loom_testbench_reference_numeric_kind_t* out_kind) {
  const loom_named_attr_t* attr =
      loom_testbench_reference_find_attr(module, attrs, name);
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference matmul contract attr '%.*s' is required",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference matmul contract attr '%.*s' must be a string",
        (int)name.size, name.data);
  }
  iree_string_view_t value = loom_testbench_reference_module_string(
      module, loom_attr_as_string_id(attr->value));
  if (!loom_testbench_reference_numeric_kind_parse(value, out_kind)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported reference matmul numeric kind '%.*s'",
                            (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_default_matmul_contract(
    iree_string_view_t oracle_name, loom_testbench_reference_numeric_kind_t lhs,
    loom_testbench_reference_numeric_kind_t rhs,
    loom_testbench_reference_numeric_kind_t init,
    loom_testbench_reference_matmul_contract_t* out_contract) {
  if (!loom_testbench_reference_numeric_kind_is_float(lhs) ||
      !loom_testbench_reference_numeric_kind_is_float(rhs) ||
      !loom_testbench_reference_numeric_kind_is_float(init)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s requires explicit {lhs, rhs, accumulator, result} contract attrs "
        "for non-floating inputs",
        (int)oracle_name.size, oracle_name.data);
  }
  *out_contract = (loom_testbench_reference_matmul_contract_t){
      .lhs = lhs,
      .rhs = rhs,
      .init = init,
      .accumulator = LOOM_TESTBENCH_REFERENCE_NUMERIC_F64,
      .result = LOOM_TESTBENCH_REFERENCE_NUMERIC_F32,
  };
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_explicit_matmul_contract(
    const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_reference_matmul_contract_t* out_contract) {
  if (invocation->attrs.count != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference matmul contract requires exactly lhs, rhs, accumulator, and "
        "result attrs");
  }
  IREE_RETURN_IF_ERROR(loom_testbench_reference_read_numeric_contract_attr(
      invocation->module, invocation->attrs, IREE_SV("lhs"),
      &out_contract->lhs));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_read_numeric_contract_attr(
      invocation->module, invocation->attrs, IREE_SV("rhs"),
      &out_contract->rhs));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_read_numeric_contract_attr(
      invocation->module, invocation->attrs, IREE_SV("accumulator"),
      &out_contract->accumulator));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_read_numeric_contract_attr(
      invocation->module, invocation->attrs, IREE_SV("result"),
      &out_contract->result));
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_validate_matmul_contract_domains(
    iree_string_view_t oracle_name,
    const loom_testbench_reference_matmul_contract_t* contract) {
  const bool floating_contract =
      loom_testbench_reference_numeric_kind_is_float(contract->lhs) &&
      loom_testbench_reference_numeric_kind_is_float(contract->rhs) &&
      loom_testbench_reference_numeric_kind_is_float(contract->init) &&
      loom_testbench_reference_numeric_kind_is_float(contract->accumulator) &&
      loom_testbench_reference_numeric_kind_is_float(contract->result);
  const bool integer_contract =
      loom_testbench_reference_numeric_kind_is_integer(contract->lhs) &&
      loom_testbench_reference_numeric_kind_is_integer(contract->rhs) &&
      loom_testbench_reference_numeric_kind_is_integer(contract->init) &&
      loom_testbench_reference_numeric_kind_is_integer(contract->accumulator) &&
      loom_testbench_reference_numeric_kind_is_integer(contract->result);
  const bool quantized_float_contract =
      loom_testbench_reference_numeric_kind_is_integer(contract->lhs) &&
      loom_testbench_reference_numeric_kind_is_integer(contract->rhs) &&
      loom_testbench_reference_numeric_kind_is_float(contract->init) &&
      loom_testbench_reference_numeric_kind_is_float(contract->accumulator) &&
      loom_testbench_reference_numeric_kind_is_float(contract->result);
  if (!floating_contract && !integer_contract && !quantized_float_contract) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s contract must be all-floating, all-integer, or integer inputs "
        "with floating init/accumulator/result",
        (int)oracle_name.size, oracle_name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_validate_contract_storage(
    iree_string_view_t oracle_name, iree_string_view_t operand_name,
    loom_testbench_reference_numeric_kind_t kind,
    iree_hal_element_type_t element_type) {
  if (!loom_testbench_reference_numeric_kind_matches_storage(kind,
                                                             element_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s %.*s storage does not match contract kind %s",
                            (int)oracle_name.size, oracle_name.data,
                            (int)operand_name.size, operand_name.data,
                            loom_testbench_reference_numeric_kind_name(kind));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_derive_matmul_contract(
    iree_string_view_t oracle_name,
    const loom_testbench_invocation_plan_t* invocation,
    iree_hal_element_type_t lhs_element_type,
    iree_hal_element_type_t rhs_element_type,
    iree_hal_element_type_t init_element_type,
    loom_testbench_reference_matmul_contract_t* out_contract) {
  loom_testbench_reference_numeric_kind_t lhs =
      LOOM_TESTBENCH_REFERENCE_NUMERIC_NONE;
  loom_testbench_reference_numeric_kind_t rhs =
      LOOM_TESTBENCH_REFERENCE_NUMERIC_NONE;
  loom_testbench_reference_numeric_kind_t init =
      LOOM_TESTBENCH_REFERENCE_NUMERIC_NONE;
  if (!loom_testbench_reference_numeric_kind_from_element_type(lhs_element_type,
                                                               &lhs) ||
      !loom_testbench_reference_numeric_kind_from_element_type(rhs_element_type,
                                                               &rhs) ||
      !loom_testbench_reference_numeric_kind_from_element_type(
          init_element_type, &init)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s has unsupported buffer element type",
                            (int)oracle_name.size, oracle_name.data);
  }
  if (invocation->attrs.count == 0) {
    IREE_RETURN_IF_ERROR(loom_testbench_reference_default_matmul_contract(
        oracle_name, lhs, rhs, init, out_contract));
  } else {
    IREE_RETURN_IF_ERROR(loom_testbench_reference_explicit_matmul_contract(
        invocation, out_contract));
    out_contract->init = init;
  }
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_validate_matmul_contract_domains(oracle_name,
                                                                out_contract));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_validate_contract_storage(
      oracle_name, IREE_SV("lhs"), out_contract->lhs, lhs_element_type));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_validate_contract_storage(
      oracle_name, IREE_SV("rhs"), out_contract->rhs, rhs_element_type));
  return loom_testbench_reference_validate_contract_storage(
      oracle_name, IREE_SV("init"), out_contract->init, init_element_type);
}

static iree_status_t loom_testbench_reference_map_view_read(
    iree_hal_buffer_view_t* view, iree_hal_buffer_mapping_t* out_mapping) {
  return iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(view), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, 0, IREE_HAL_WHOLE_BUFFER, out_mapping);
}

static iree_status_t loom_testbench_reference_store_accumulator(
    const loom_testbench_reference_matmul_contract_t* contract,
    double floating_accumulator, int64_t integer_accumulator, uint8_t* result,
    iree_host_size_t result_offset) {
  if (loom_testbench_reference_numeric_kind_is_float(contract->result)) {
    return loom_testbench_reference_store_float_element_at(
        contract->result, floating_accumulator, result, result_offset);
  }
  return loom_testbench_reference_store_integer_element_at(
      contract->result, integer_accumulator, result, result_offset);
}

static iree_status_t loom_testbench_reference_add_integer_product(
    loom_testbench_reference_numeric_kind_t accumulator_kind, int64_t lhs,
    int64_t rhs, int64_t* inout_accumulator) {
  int64_t product = 0;
  int64_t next = 0;
  if (!iree_checked_mul_i64(lhs, rhs, &product) ||
      !iree_checked_add_i64(*inout_accumulator, product, &next)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference matmul integer accumulator is out of i64 range");
  }
  if (!loom_testbench_reference_integer_kind_contains(accumulator_kind, next)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference matmul integer accumulator does not fit in %s",
        loom_testbench_reference_numeric_kind_name(accumulator_kind));
  }
  *inout_accumulator = next;
  return iree_ok_status();
}

static iree_host_size_t loom_testbench_reference_rank4_element_offset(
    const loom_testbench_reference_rank4_view_t* view,
    iree_host_size_t outer_row, iree_host_size_t outer_column,
    iree_host_size_t inner_row, iree_host_size_t inner_column) {
  return (((outer_row * view->outer_columns + outer_column) * view->inner_rows +
           inner_row) *
              view->inner_columns +
          inner_column);
}

static iree_status_t loom_testbench_reference_compute_matmul(
    const loom_testbench_reference_rank2_view_t* lhs,
    const loom_testbench_reference_rank2_view_t* rhs,
    const loom_testbench_reference_rank2_view_t* init,
    const loom_testbench_reference_matmul_contract_t* contract,
    uint8_t* result) {
  iree_hal_buffer_mapping_t lhs_mapping = {0};
  iree_hal_buffer_mapping_t rhs_mapping = {0};
  iree_hal_buffer_mapping_t init_mapping = {0};
  bool lhs_mapped = false;
  bool rhs_mapped = false;
  bool init_mapped = false;
  iree_status_t status =
      loom_testbench_reference_map_view_read(lhs->view, &lhs_mapping);
  if (iree_status_is_ok(status)) {
    lhs_mapped = true;
    status = loom_testbench_reference_map_view_read(rhs->view, &rhs_mapping);
  }
  if (iree_status_is_ok(status)) {
    rhs_mapped = true;
    status = loom_testbench_reference_map_view_read(init->view, &init_mapping);
  }
  if (iree_status_is_ok(status)) {
    init_mapped = true;
    const bool floating_domain =
        loom_testbench_reference_numeric_kind_is_float(contract->accumulator);
    for (iree_host_size_t row = 0; row < lhs->rows; ++row) {
      for (iree_host_size_t column = 0; column < rhs->columns; ++column) {
        const iree_host_size_t init_offset = row * init->columns + column;
        double floating_accumulator = 0.0;
        int64_t integer_accumulator = 0;
        if (floating_domain) {
          floating_accumulator =
              loom_testbench_reference_load_numeric_element_as_float_at(
                  contract->init, init_mapping.contents.data, init_offset);
        } else {
          integer_accumulator =
              loom_testbench_reference_load_integer_element_at(
                  contract->init, init_mapping.contents.data, init_offset);
          if (!loom_testbench_reference_integer_kind_contains(
                  contract->accumulator, integer_accumulator)) {
            status = iree_make_status(
                IREE_STATUS_OUT_OF_RANGE,
                "reference matmul init does not fit in accumulator %s",
                loom_testbench_reference_numeric_kind_name(
                    contract->accumulator));
            break;
          }
        }
        for (iree_host_size_t k = 0; k < lhs->columns; ++k) {
          const iree_host_size_t lhs_offset = row * lhs->columns + k;
          const iree_host_size_t rhs_offset = k * rhs->columns + column;
          if (floating_domain) {
            const double lhs_value =
                loom_testbench_reference_load_numeric_element_as_float_at(
                    contract->lhs, lhs_mapping.contents.data, lhs_offset);
            const double rhs_value =
                loom_testbench_reference_load_numeric_element_as_float_at(
                    contract->rhs, rhs_mapping.contents.data, rhs_offset);
            floating_accumulator += lhs_value * rhs_value;
          } else {
            const int64_t lhs_value =
                loom_testbench_reference_load_integer_element_at(
                    contract->lhs, lhs_mapping.contents.data, lhs_offset);
            const int64_t rhs_value =
                loom_testbench_reference_load_integer_element_at(
                    contract->rhs, rhs_mapping.contents.data, rhs_offset);
            status = loom_testbench_reference_add_integer_product(
                contract->accumulator, lhs_value, rhs_value,
                &integer_accumulator);
            if (!iree_status_is_ok(status)) {
              break;
            }
          }
        }
        if (!iree_status_is_ok(status)) {
          break;
        }
        status = loom_testbench_reference_store_accumulator(
            contract, floating_accumulator, integer_accumulator, result,
            row * rhs->columns + column);
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  if (init_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&init_mapping));
  }
  if (rhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&rhs_mapping));
  }
  if (lhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&lhs_mapping));
  }
  return status;
}

static iree_status_t loom_testbench_reference_compute_tiled_matmul(
    const loom_testbench_reference_rank4_view_t* lhs,
    const loom_testbench_reference_rank4_view_t* rhs,
    const loom_testbench_reference_rank4_view_t* init,
    const loom_testbench_reference_matmul_contract_t* contract,
    uint8_t* result) {
  iree_hal_buffer_mapping_t lhs_mapping = {0};
  iree_hal_buffer_mapping_t rhs_mapping = {0};
  iree_hal_buffer_mapping_t init_mapping = {0};
  bool lhs_mapped = false;
  bool rhs_mapped = false;
  bool init_mapped = false;
  iree_status_t status =
      loom_testbench_reference_map_view_read(lhs->view, &lhs_mapping);
  if (iree_status_is_ok(status)) {
    lhs_mapped = true;
    status = loom_testbench_reference_map_view_read(rhs->view, &rhs_mapping);
  }
  if (iree_status_is_ok(status)) {
    rhs_mapped = true;
    status = loom_testbench_reference_map_view_read(init->view, &init_mapping);
  }
  if (iree_status_is_ok(status)) {
    init_mapped = true;
    const bool floating_domain =
        loom_testbench_reference_numeric_kind_is_float(contract->accumulator);
    for (iree_host_size_t outer_row = 0; outer_row < lhs->outer_rows;
         ++outer_row) {
      for (iree_host_size_t outer_column = 0; outer_column < rhs->outer_columns;
           ++outer_column) {
        for (iree_host_size_t inner_row = 0; inner_row < lhs->inner_rows;
             ++inner_row) {
          for (iree_host_size_t inner_column = 0;
               inner_column < rhs->inner_columns; ++inner_column) {
            const iree_host_size_t init_offset =
                loom_testbench_reference_rank4_element_offset(
                    init, outer_row, outer_column, inner_row, inner_column);
            double floating_accumulator = 0.0;
            int64_t integer_accumulator = 0;
            if (floating_domain) {
              floating_accumulator =
                  loom_testbench_reference_load_numeric_element_as_float_at(
                      contract->init, init_mapping.contents.data, init_offset);
            } else {
              integer_accumulator =
                  loom_testbench_reference_load_integer_element_at(
                      contract->init, init_mapping.contents.data, init_offset);
              if (!loom_testbench_reference_integer_kind_contains(
                      contract->accumulator, integer_accumulator)) {
                status = iree_make_status(
                    IREE_STATUS_OUT_OF_RANGE,
                    "reference matmul init does not fit in accumulator %s",
                    loom_testbench_reference_numeric_kind_name(
                        contract->accumulator));
                break;
              }
            }
            for (iree_host_size_t outer_k = 0; outer_k < lhs->outer_columns;
                 ++outer_k) {
              for (iree_host_size_t inner_k = 0; inner_k < lhs->inner_columns;
                   ++inner_k) {
                const iree_host_size_t lhs_offset =
                    loom_testbench_reference_rank4_element_offset(
                        lhs, outer_row, outer_k, inner_row, inner_k);
                const iree_host_size_t rhs_offset =
                    loom_testbench_reference_rank4_element_offset(
                        rhs, outer_k, outer_column, inner_k, inner_column);
                if (floating_domain) {
                  const double lhs_value =
                      loom_testbench_reference_load_numeric_element_as_float_at(
                          contract->lhs, lhs_mapping.contents.data, lhs_offset);
                  const double rhs_value =
                      loom_testbench_reference_load_numeric_element_as_float_at(
                          contract->rhs, rhs_mapping.contents.data, rhs_offset);
                  floating_accumulator += lhs_value * rhs_value;
                } else {
                  const int64_t lhs_value =
                      loom_testbench_reference_load_integer_element_at(
                          contract->lhs, lhs_mapping.contents.data, lhs_offset);
                  const int64_t rhs_value =
                      loom_testbench_reference_load_integer_element_at(
                          contract->rhs, rhs_mapping.contents.data, rhs_offset);
                  status = loom_testbench_reference_add_integer_product(
                      contract->accumulator, lhs_value, rhs_value,
                      &integer_accumulator);
                  if (!iree_status_is_ok(status)) {
                    break;
                  }
                }
              }
              if (!iree_status_is_ok(status)) {
                break;
              }
            }
            if (!iree_status_is_ok(status)) {
              break;
            }
            const iree_host_size_t result_offset =
                (((outer_row * rhs->outer_columns + outer_column) *
                      lhs->inner_rows +
                  inner_row) *
                     rhs->inner_columns +
                 inner_column);
            status = loom_testbench_reference_store_accumulator(
                contract, floating_accumulator, integer_accumulator, result,
                result_offset);
            if (!iree_status_is_ok(status)) {
              break;
            }
          }
          if (!iree_status_is_ok(status)) {
            break;
          }
        }
        if (!iree_status_is_ok(status)) {
          break;
        }
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }
  if (init_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&init_mapping));
  }
  if (rhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&rhs_mapping));
  }
  if (lhs_mapped) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&lhs_mapping));
  }
  return status;
}

static iree_status_t loom_testbench_reference_fill_matmul_result(
    iree_hal_buffer_mapping_t* mapping, void* user_data) {
  const loom_testbench_reference_matmul_result_t* result =
      (const loom_testbench_reference_matmul_result_t*)user_data;
  if (mapping->contents.data_length < result->data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reference.matmul result mapping is too small");
  }
  memcpy(mapping->contents.data, result->data, result->data_length);
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_allocate_matmul_result(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    iree_host_size_t rows, iree_host_size_t columns,
    iree_hal_element_type_t element_type, const uint8_t* data,
    iree_host_size_t data_length, iree_hal_buffer_view_t** out_buffer_view) {
  const iree_hal_dim_t shape[2] = {
      (iree_hal_dim_t)rows,
      (iree_hal_dim_t)columns,
  };
  loom_testbench_reference_matmul_result_t result = {
      .data = data,
      .data_length = data_length,
  };
  return iree_hal_buffer_view_generate_buffer(
      options->device, options->device_allocator, IREE_ARRAYSIZE(shape), shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      options->result_buffer_params,
      loom_testbench_reference_fill_matmul_result, &result, out_buffer_view);
}

static iree_status_t loom_testbench_reference_allocate_tiled_matmul_result(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    iree_host_size_t outer_rows, iree_host_size_t outer_columns,
    iree_host_size_t inner_rows, iree_host_size_t inner_columns,
    iree_hal_element_type_t element_type, const uint8_t* data,
    iree_host_size_t data_length, iree_hal_buffer_view_t** out_buffer_view) {
  const iree_hal_dim_t shape[4] = {
      (iree_hal_dim_t)outer_rows,
      (iree_hal_dim_t)outer_columns,
      (iree_hal_dim_t)inner_rows,
      (iree_hal_dim_t)inner_columns,
  };
  loom_testbench_reference_matmul_result_t result = {
      .data = data,
      .data_length = data_length,
  };
  return iree_hal_buffer_view_generate_buffer(
      options->device, options->device_allocator, IREE_ARRAYSIZE(shape), shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      options->result_buffer_params,
      loom_testbench_reference_fill_matmul_result, &result, out_buffer_view);
}

static iree_status_t loom_testbench_reference_tiled_matmul_element_count(
    const loom_testbench_reference_rank4_view_t* lhs,
    const loom_testbench_reference_rank4_view_t* rhs,
    iree_host_size_t* out_element_count) {
  iree_host_size_t outer_count = 0;
  if (!iree_host_size_checked_mul(lhs->outer_rows, rhs->outer_columns,
                                  &outer_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result outer element count overflow");
  }
  iree_host_size_t inner_count = 0;
  if (!iree_host_size_checked_mul(lhs->inner_rows, rhs->inner_columns,
                                  &inner_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result inner element count overflow");
  }
  if (!iree_host_size_checked_mul(outer_count, inner_count,
                                  out_element_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result element count overflow");
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_matmul_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const iree_vm_variant_t* inputs,
    iree_host_size_t result_count, iree_vm_variant_t* out_results) {
  const loom_testbench_reference_matmul_oracle_options_t* options =
      (const loom_testbench_reference_matmul_oracle_options_t*)user_data;
  if (input_count != 3 || result_count != 1) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reference.matmul expects 3 inputs and 1 result");
  }
  if (options->device_allocator == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "reference.matmul requires a configured HAL allocator");
  }

  iree_hal_buffer_view_t* lhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[0], &lhs_view));
  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[1], &rhs_view));
  iree_hal_buffer_view_t* init_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[2], &init_view));

  loom_testbench_reference_rank2_view_t lhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank2_view_initialize(lhs_view, &lhs));
  loom_testbench_reference_rank2_view_t rhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank2_view_initialize(rhs_view, &rhs));
  loom_testbench_reference_rank2_view_t init = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank2_view_initialize(init_view, &init));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_validate_matmul_shape(
      IREE_SV("reference.matmul"), &lhs, &rhs, &init));
  loom_testbench_reference_matmul_contract_t contract = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_reference_derive_matmul_contract(
      IREE_SV("reference.matmul"), invocation, lhs.element_type,
      rhs.element_type, init.element_type, &contract));

  iree_host_size_t result_count_checked = 0;
  if (!iree_host_size_checked_mul(lhs.rows, rhs.columns,
                                  &result_count_checked)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reference.matmul result element count overflow");
  }

  iree_allocator_t host_allocator = options->host_allocator;
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  const iree_host_size_t result_element_size =
      loom_testbench_reference_numeric_kind_byte_count(contract.result);
  iree_host_size_t result_byte_count = 0;
  if (!iree_host_size_checked_mul(result_count_checked, result_element_size,
                                  &result_byte_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reference.matmul result byte count overflow");
  }
  uint8_t* result_values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, result_byte_count, sizeof(*result_values),
      (void**)&result_values));
  iree_status_t status = loom_testbench_reference_compute_matmul(
      &lhs, &rhs, &init, &contract, result_values);
  iree_hal_buffer_view_t* result_view = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_reference_allocate_matmul_result(
        options, lhs.rows, rhs.columns,
        loom_testbench_reference_numeric_kind_hal_element_type(contract.result),
        result_values, result_byte_count, &result_view);
  }
  iree_allocator_free(host_allocator, result_values);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_vm_ref_t result_ref = iree_hal_buffer_view_move_ref(result_view);
  out_results[0] = iree_vm_make_variant_ref_assign(result_ref);
  return iree_ok_status();
}

static iree_status_t loom_testbench_reference_tiled_matmul_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const iree_vm_variant_t* inputs,
    iree_host_size_t result_count, iree_vm_variant_t* out_results) {
  const loom_testbench_reference_matmul_oracle_options_t* options =
      (const loom_testbench_reference_matmul_oracle_options_t*)user_data;
  if (input_count != 3 || result_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "reference.tiled_matmul expects 3 inputs and 1 result");
  }
  if (options->device_allocator == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "reference.tiled_matmul requires a configured HAL allocator");
  }

  iree_hal_buffer_view_t* lhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[0], &lhs_view));
  iree_hal_buffer_view_t* rhs_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[1], &rhs_view));
  iree_hal_buffer_view_t* init_view = NULL;
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_variant_buffer_view(&inputs[2], &init_view));

  loom_testbench_reference_rank4_view_t lhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank4_view_initialize(lhs_view, &lhs));
  loom_testbench_reference_rank4_view_t rhs = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank4_view_initialize(rhs_view, &rhs));
  loom_testbench_reference_rank4_view_t init = {0};
  IREE_RETURN_IF_ERROR(
      loom_testbench_reference_rank4_view_initialize(init_view, &init));
  IREE_RETURN_IF_ERROR(loom_testbench_reference_validate_tiled_matmul_shape(
      IREE_SV("reference.tiled_matmul"), &lhs, &rhs, &init));
  loom_testbench_reference_matmul_contract_t contract = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_reference_derive_matmul_contract(
      IREE_SV("reference.tiled_matmul"), invocation, lhs.element_type,
      rhs.element_type, init.element_type, &contract));

  iree_host_size_t result_count_checked = 0;
  IREE_RETURN_IF_ERROR(loom_testbench_reference_tiled_matmul_element_count(
      &lhs, &rhs, &result_count_checked));

  iree_allocator_t host_allocator = options->host_allocator;
  if (iree_allocator_is_null(host_allocator)) {
    host_allocator = iree_allocator_system();
  }
  const iree_host_size_t result_element_size =
      loom_testbench_reference_numeric_kind_byte_count(contract.result);
  iree_host_size_t result_byte_count = 0;
  if (!iree_host_size_checked_mul(result_count_checked, result_element_size,
                                  &result_byte_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "reference.tiled_matmul result byte count overflow");
  }
  uint8_t* result_values = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, result_byte_count, sizeof(*result_values),
      (void**)&result_values));
  iree_status_t status = loom_testbench_reference_compute_tiled_matmul(
      &lhs, &rhs, &init, &contract, result_values);
  iree_hal_buffer_view_t* result_view = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_testbench_reference_allocate_tiled_matmul_result(
        options, lhs.outer_rows, rhs.outer_columns, lhs.inner_rows,
        rhs.inner_columns,
        loom_testbench_reference_numeric_kind_hal_element_type(contract.result),
        result_values, result_byte_count, &result_view);
  }
  iree_allocator_free(host_allocator, result_values);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_vm_ref_t result_ref = iree_hal_buffer_view_move_ref(result_view);
  out_results[0] = iree_vm_make_variant_ref_assign(result_ref);
  return iree_ok_status();
}

void loom_testbench_reference_matmul_oracle_provider_initialize(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    loom_testbench_oracle_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_oracle_provider_t){
      .name = IREE_SV("reference.matmul"),
      .invoke =
          {
              .fn = loom_testbench_reference_matmul_invoke,
              .user_data = (void*)options,
          },
  };
}

void loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
    const loom_testbench_reference_matmul_oracle_options_t* options,
    loom_testbench_oracle_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_testbench_oracle_provider_t){
      .name = IREE_SV("reference.tiled_matmul"),
      .invoke =
          {
              .fn = loom_testbench_reference_tiled_matmul_invoke,
              .user_data = (void*)options,
          },
  };
}
