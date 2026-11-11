// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_integer_cast.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

bool loom_symbolic_integer_cast_from_op(const loom_module_t* module,
                                        const loom_op_t* op,
                                        loom_integer_cast_t* out_cast) {
  loom_scalar_type_t payload_type = LOOM_SCALAR_TYPE_NONE;
  loom_integer_cast_kind_t kind = LOOM_INTEGER_CAST_SIGNED;
  switch (op->kind) {
    case LOOM_OP_SCALAR_EXTSI:
    case LOOM_OP_SCALAR_EXTUI:
      payload_type = loom_type_element_type(
          loom_module_value_type(module, loom_op_const_operands(op)[0]));
      kind = op->kind == LOOM_OP_SCALAR_EXTSI ? LOOM_INTEGER_CAST_SIGNED
                                              : LOOM_INTEGER_CAST_UNSIGNED;
      break;
    case LOOM_OP_SCALAR_TRUNCI:
      payload_type = loom_type_element_type(
          loom_module_value_type(module, loom_op_const_results(op)[0]));
      kind = payload_type == LOOM_SCALAR_TYPE_I1 ? LOOM_INTEGER_CAST_UNSIGNED
                                                 : LOOM_INTEGER_CAST_SIGNED;
      break;
    case LOOM_OP_INDEX_CAST: {
      const loom_scalar_type_t input_type = loom_type_element_type(
          loom_module_value_type(module, loom_op_const_operands(op)[0]));
      const loom_scalar_type_t result_type = loom_type_element_type(
          loom_module_value_type(module, loom_op_const_results(op)[0]));
      const bool input_is_fixed = loom_scalar_type_set_contains(
          LOOM_SCALAR_TYPE_SET_INTEGER, input_type);
      const bool result_is_fixed = loom_scalar_type_set_contains(
          LOOM_SCALAR_TYPE_SET_INTEGER, result_type);
      if (input_is_fixed && result_type == LOOM_SCALAR_TYPE_OFFSET &&
          input_type != LOOM_SCALAR_TYPE_I64) {
        payload_type = input_type;
        kind = LOOM_INTEGER_CAST_UNSIGNED;
      } else if (input_is_fixed && result_type == LOOM_SCALAR_TYPE_INDEX) {
        payload_type = input_type;
        kind = input_type == LOOM_SCALAR_TYPE_I1 ? LOOM_INTEGER_CAST_UNSIGNED
                                                 : LOOM_INTEGER_CAST_SIGNED;
      } else if ((input_type == LOOM_SCALAR_TYPE_INDEX ||
                  input_type == LOOM_SCALAR_TYPE_OFFSET) &&
                 result_is_fixed) {
        payload_type = result_type;
        kind = result_type == LOOM_SCALAR_TYPE_I1 ? LOOM_INTEGER_CAST_UNSIGNED
                                                  : LOOM_INTEGER_CAST_SIGNED;
      } else {
        return false;
      }
      break;
    }
    default:
      return false;
  }

  const int32_t bit_count = loom_scalar_type_bitwidth(payload_type);
  if (bit_count <= 0 || bit_count > 64) {
    return false;
  }
  *out_cast = (loom_integer_cast_t){
      .source_value_id = loom_op_const_operands(op)[0],
      .bit_count = (uint8_t)bit_count,
      .kind = kind,
  };
  return true;
}
