// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_integer_cast.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

bool loom_symbolic_integer_cast_from_op(
    const loom_module_t* module, const loom_op_t* op,
    loom_symbolic_integer_cast_t* out_cast) {
  const loom_value_id_t input = loom_op_const_operands(op)[0];
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_scalar_type_t input_type =
      loom_type_element_type(loom_module_value_type(module, input));
  const loom_scalar_type_t result_type =
      loom_type_element_type(loom_module_value_type(module, result));
  const bool input_is_fixed =
      loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_INTEGER, input_type);
  const bool result_is_fixed =
      loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_INTEGER, result_type);

  loom_scalar_type_t payload_type = LOOM_SCALAR_TYPE_NONE;
  loom_symbolic_integer_cast_kind_t kind = LOOM_SYMBOLIC_INTEGER_CAST_SIGNED;
  switch (op->kind) {
    case LOOM_OP_SCALAR_EXTSI:
      payload_type = input_type;
      kind = LOOM_SYMBOLIC_INTEGER_CAST_SIGNED;
      break;
    case LOOM_OP_SCALAR_EXTUI:
      payload_type = input_type;
      kind = LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED;
      break;
    case LOOM_OP_SCALAR_TRUNCI:
      payload_type = result_type;
      kind = result_type == LOOM_SCALAR_TYPE_I1
                 ? LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED
                 : LOOM_SYMBOLIC_INTEGER_CAST_SIGNED;
      break;
    case LOOM_OP_INDEX_CAST:
      if (input_is_fixed && result_type == LOOM_SCALAR_TYPE_OFFSET &&
          input_type != LOOM_SCALAR_TYPE_I64) {
        payload_type = input_type;
        kind = LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED;
      } else if (input_is_fixed && result_type == LOOM_SCALAR_TYPE_INDEX) {
        payload_type = input_type;
        kind = input_type == LOOM_SCALAR_TYPE_I1
                   ? LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED
                   : LOOM_SYMBOLIC_INTEGER_CAST_SIGNED;
      } else if ((input_type == LOOM_SCALAR_TYPE_INDEX ||
                  input_type == LOOM_SCALAR_TYPE_OFFSET) &&
                 result_is_fixed) {
        payload_type = result_type;
        kind = result_type == LOOM_SCALAR_TYPE_I1
                   ? LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED
                   : LOOM_SYMBOLIC_INTEGER_CAST_SIGNED;
      } else {
        return false;
      }
      break;
    default:
      return false;
  }

  const int32_t bit_count = loom_scalar_type_bitwidth(payload_type);
  if (bit_count <= 0 || bit_count > 64) {
    return false;
  }
  *out_cast = (loom_symbolic_integer_cast_t){
      .source_value_id = input,
      .bit_count = (uint8_t)bit_count,
      .kind = kind,
  };
  return true;
}

bool loom_symbolic_integer_cast_is_identity(
    const loom_symbolic_integer_cast_t* cast, loom_value_facts_t source_facts) {
  if (cast->bit_count == 0 || cast->bit_count > 64 ||
      loom_value_facts_is_float(source_facts)) {
    return false;
  }
  return cast->kind == LOOM_SYMBOLIC_INTEGER_CAST_UNSIGNED
             ? loom_value_facts_fit_unsigned_bit_count(source_facts,
                                                       cast->bit_count)
             : loom_value_facts_fit_signed_bit_count(source_facts,
                                                     cast->bit_count);
}

bool loom_symbolic_integer_cast_same_interpretation(
    const loom_symbolic_integer_cast_t* left,
    const loom_symbolic_integer_cast_t* right) {
  return left->bit_count == right->bit_count && left->kind == right->kind;
}
