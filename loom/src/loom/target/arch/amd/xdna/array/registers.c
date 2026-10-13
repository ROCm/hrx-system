// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/registers.h"

typedef struct loom_xdna_register_dimension_t {
  // Dimension name in the shared string table.
  uint16_t name_offset;
  // Exclusive upper bound of the dimension index.
  uint16_t count;
  // Byte stride between consecutive indices.
  uint32_t stride;
} loom_xdna_register_dimension_t;

typedef struct loom_xdna_register_pattern_t {
  // Tile-relative address of the first register in the pattern.
  uint32_t base_offset;
  // Independent sources supporting this register pattern.
  loom_xdna_provenance_bits_t provenance_bits;
  // Address dimensions in declaration order.
  loom_xdna_register_dimension_t dimensions[2];
  // Register module owning this pattern.
  uint8_t module;
  // Number of populated address dimensions.
  uint8_t dimension_count;
} loom_xdna_register_pattern_t;

typedef enum loom_xdna_register_field_flag_bits_e {
  LOOM_XDNA_REGISTER_FIELD_ACCESS_MASK = 0x3u,
  LOOM_XDNA_REGISTER_FIELD_FLAG_SIGNED = 1u << 7,
} loom_xdna_register_field_flag_bits_t;

typedef struct loom_xdna_register_field_t {
  // Stable field key in the shared string table.
  uint16_t name_offset;
  // Index of the owning register pattern.
  uint8_t pattern_id;
  // Least-significant field bit in the 32-bit register.
  uint8_t least_significant_bit;
  // Encoded field width.
  uint8_t bit_width;
  // Access enum in the low two bits and signedness in the high bit.
  uint8_t access_and_flags;
} loom_xdna_register_field_t;

#include "loom/target/arch/amd/xdna/array/register_tables.inl"

static iree_status_t loom_xdna_register_field_resolve(
    loom_xdna_register_field_id_t field_id,
    const loom_xdna_register_field_t** out_field,
    const loom_xdna_register_pattern_t** out_pattern) {
  if (field_id == 0 || field_id > kLoomXdnaRegisterFieldCount) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA register field ID %u is invalid", field_id);
  }
  const loom_xdna_register_field_t* field = &kLoomXdnaRegisterFields[field_id];
  *out_field = field;
  *out_pattern = &kLoomXdnaRegisterPatterns[field->pattern_id];
  return iree_ok_status();
}

iree_host_size_t loom_xdna_register_field_count(void) {
  return kLoomXdnaRegisterFieldCount;
}

iree_status_t loom_xdna_register_field_info(
    loom_xdna_register_field_id_t field_id,
    loom_xdna_register_field_info_t* out_info) {
  IREE_ASSERT_ARGUMENT(out_info);
  *out_info = (loom_xdna_register_field_info_t){0};
  const loom_xdna_register_field_t* field = NULL;
  const loom_xdna_register_pattern_t* pattern = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_register_field_resolve(field_id, &field, &pattern));
  *out_info = (loom_xdna_register_field_info_t){
      .key =
          iree_make_cstring_view(kLoomXdnaRegisterStrings + field->name_offset),
      .module = (loom_xdna_register_module_t)pattern->module,
      .access =
          (loom_xdna_register_access_t)(field->access_and_flags &
                                        LOOM_XDNA_REGISTER_FIELD_ACCESS_MASK),
      .least_significant_bit = field->least_significant_bit,
      .bit_width = field->bit_width,
      .is_signed =
          (field->access_and_flags & LOOM_XDNA_REGISTER_FIELD_FLAG_SIGNED) != 0,
      .dimension_count = pattern->dimension_count,
      .provenance_bits = pattern->provenance_bits,
  };
  return iree_ok_status();
}

loom_xdna_register_dimension_info_t loom_xdna_register_field_dimension(
    loom_xdna_register_field_id_t field_id, iree_host_size_t ordinal) {
  const loom_xdna_register_field_t* field = &kLoomXdnaRegisterFields[field_id];
  const loom_xdna_register_pattern_t* pattern =
      &kLoomXdnaRegisterPatterns[field->pattern_id];
  const loom_xdna_register_dimension_t* dimension =
      &pattern->dimensions[ordinal];
  return (loom_xdna_register_dimension_info_t){
      .name = iree_make_cstring_view(kLoomXdnaRegisterStrings +
                                     dimension->name_offset),
      .count = dimension->count,
      .stride = dimension->stride,
  };
}

iree_status_t loom_xdna_register_field_encode(
    loom_xdna_register_field_id_t field_id, int64_t value,
    uint32_t* out_register_bits) {
  IREE_ASSERT_ARGUMENT(out_register_bits);
  *out_register_bits = 0;
  const loom_xdna_register_field_t* field = NULL;
  const loom_xdna_register_pattern_t* pattern = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_register_field_resolve(field_id, &field, &pattern));
  (void)pattern;
  const bool is_signed =
      (field->access_and_flags & LOOM_XDNA_REGISTER_FIELD_FLAG_SIGNED) != 0;
  int64_t minimum = 0;
  int64_t maximum = 0;
  if (is_signed) {
    minimum = -(INT64_C(1) << (field->bit_width - 1));
    maximum = (INT64_C(1) << (field->bit_width - 1)) - 1;
  } else {
    maximum = field->bit_width == 32 ? (int64_t)UINT32_MAX
                                     : (INT64_C(1) << field->bit_width) - 1;
  }
  if (value < minimum || value > maximum) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA register field %u value %" PRId64
                            " is outside [%" PRId64 ", %" PRId64 "]",
                            field_id, value, minimum, maximum);
  }
  *out_register_bits =
      loom_xdna_register_field_encode_admitted(field_id, value);
  return iree_ok_status();
}

uint32_t loom_xdna_register_field_encode_admitted(
    loom_xdna_register_field_id_t field_id, int64_t value) {
  const loom_xdna_register_field_t* field = &kLoomXdnaRegisterFields[field_id];
  const uint32_t value_mask = field->bit_width == 32
                                  ? UINT32_MAX
                                  : (UINT32_C(1) << field->bit_width) - 1;
  return ((uint32_t)value & value_mask) << field->least_significant_bit;
}

iree_status_t loom_xdna_register_field_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_register_field_id_t field_id,
    loom_xdna_tile_coordinate_t coordinate, iree_host_size_t index_count,
    const uint16_t* indices, uint64_t* out_address) {
  IREE_ASSERT_ARGUMENT(family);
  IREE_ASSERT_ARGUMENT(out_address);
  *out_address = 0;
  const loom_xdna_register_field_t* field = NULL;
  const loom_xdna_register_pattern_t* pattern = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_register_field_resolve(field_id, &field, &pattern));
  (void)field;
  if (index_count != pattern->dimension_count ||
      (index_count != 0 && indices == NULL)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "XDNA register field %u requires %u indices but received %" PRIhsz,
        field_id, pattern->dimension_count, index_count);
  }
  uint32_t register_offset = pattern->base_offset;
  for (iree_host_size_t i = 0; i < index_count; ++i) {
    const loom_xdna_register_dimension_t* dimension = &pattern->dimensions[i];
    if (indices[i] >= dimension->count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "XDNA register field %u index %" PRIhsz
                              " value %u exceeds %u",
                              field_id, i, indices[i], dimension->count);
    }
    register_offset += indices[i] * dimension->stride;
  }
  return loom_xdna_array_register_address(
      family, coordinate, (loom_xdna_register_module_t)pattern->module,
      register_offset, out_address);
}

uint64_t loom_xdna_register_field_address_admitted(
    const loom_xdna_array_family_t* family,
    loom_xdna_register_field_id_t field_id,
    loom_xdna_tile_coordinate_t coordinate, const uint16_t* indices) {
  const loom_xdna_register_field_t* field = &kLoomXdnaRegisterFields[field_id];
  const loom_xdna_register_pattern_t* pattern =
      &kLoomXdnaRegisterPatterns[field->pattern_id];
  uint32_t register_offset = pattern->base_offset;
  for (uint8_t i = 0; i < pattern->dimension_count; ++i) {
    register_offset += indices[i] * pattern->dimensions[i].stride;
  }
  return ((uint64_t)coordinate.column << family->column_shift) |
         ((uint64_t)coordinate.row << family->row_shift) | register_offset;
}
