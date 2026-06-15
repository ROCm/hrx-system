// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 packed-dot source-to-low lowering.

#include <stdint.h>

#include "loom/target/arch/x86/lower/internal.h"

static bool loom_x86_scalar_type_has_packed_dot_register_width(
    loom_scalar_type_t scalar_type, uint32_t* out_bit_width) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_I8:
      *out_bit_width = 8;
      return true;
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
      *out_bit_width = 16;
      return true;
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
      *out_bit_width = 32;
      return true;
    default:
      *out_bit_width = 0;
      return false;
  }
}

bool loom_x86_packed_dot_type_static_vector_bit_width(loom_type_t type,
                                                      uint32_t* out_bit_width) {
  *out_bit_width = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  uint32_t element_bit_width = 0;
  if (!loom_x86_scalar_type_has_packed_dot_register_width(
          loom_type_element_type(type), &element_bit_width)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count <= 0 ||
      (uint64_t)lane_count > UINT32_MAX / element_bit_width) {
    return false;
  }
  *out_bit_width = (uint32_t)lane_count * element_bit_width;
  return true;
}

iree_status_t loom_x86_map_packed_dot_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type) {
  uint32_t vector_bit_width = 0;
  if (loom_x86_packed_dot_type_static_vector_bit_width(source_type,
                                                       &vector_bit_width)) {
    loom_x86_register_class_t register_class = 0;
    if (loom_x86_register_class_for_vector_bit_width(vector_bit_width,
                                                     &register_class)) {
      return loom_x86_make_register_type(context, register_class, out_low_type);
    }
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}
