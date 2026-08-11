// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/operand.h"

#include <stdint.h>

static bool loom_encoding_one_hot_fact_enum(uint64_t value,
                                            uint8_t* out_enum_value) {
  if (value == 0) {
    *out_enum_value = 0;
    return true;
  }
  if (!iree_is_power_of_two_uint64(value)) return false;
  uint8_t enum_value = 1;
  while ((value >>= 1) != 0) ++enum_value;
  *out_enum_value = enum_value;
  return true;
}

static bool loom_encoding_rounding_fact_enum(uint64_t value,
                                             uint8_t* out_enum_value) {
  if (value == (LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY |
                LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL)) {
    *out_enum_value = LOOM_ENCODING_ROUNDING_POLICY_FINITE_FLUSH_SUBNORMAL;
    return true;
  }
  return loom_encoding_one_hot_fact_enum(value, out_enum_value);
}

iree_string_view_t loom_encoding_operand_fact_name(
    loom_encoding_operand_parameter_t parameter, uint64_t value) {
  uint8_t enum_value = 0;
  bool has_enum_value = false;
  switch (parameter) {
    case LOOM_ENCODING_OPERAND_PARAMETER_ELEMENT_FORMAT:
    case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_FORMAT:
    case LOOM_ENCODING_OPERAND_PARAMETER_SECONDARY_SCALE_FORMAT:
    case LOOM_ENCODING_OPERAND_PARAMETER_PAYLOAD_PACKING:
    case LOOM_ENCODING_OPERAND_PARAMETER_SCALE_TOPOLOGY:
    case LOOM_ENCODING_OPERAND_PARAMETER_AFFINE:
    case LOOM_ENCODING_OPERAND_PARAMETER_CODEBOOK:
    case LOOM_ENCODING_OPERAND_PARAMETER_SPARSITY:
      has_enum_value = loom_encoding_one_hot_fact_enum(value, &enum_value);
      break;
    case LOOM_ENCODING_OPERAND_PARAMETER_ROUNDING:
      has_enum_value = loom_encoding_rounding_fact_enum(value, &enum_value);
      break;
    default:
      break;
  }
  if (!has_enum_value) return iree_string_view_empty();

  const loom_attr_descriptor_t* descriptor =
      &loom_encoding_operand_family_descriptor.parameter_descriptors[parameter];
  const loom_bstring_t name =
      loom_attr_descriptor_enum_case_name(descriptor, enum_value);
  return name ? loom_bstring_view(name) : iree_string_view_empty();
}
