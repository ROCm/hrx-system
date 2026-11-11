// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/integer_cast.h"

loom_integer_cast_t loom_integer_cast_compose(loom_integer_cast_t input,
                                              loom_integer_cast_kind_t kind,
                                              uint8_t bit_count) {
  if (bit_count <= input.bit_count) {
    input.bit_count = bit_count;
    input.kind = kind;
    input.unsigned_bit_count = 0;
  } else if (input.kind != LOOM_INTEGER_CAST_UNSIGNED &&
             (input.kind == LOOM_INTEGER_CAST_SIGNED ||
              bit_count <= input.unsigned_bit_count)) {
    // The outer interpretation either preserves the signed low payload or
    // treats its sign-extension bits as an unsigned part of the wider value.
    input.kind = kind == LOOM_INTEGER_CAST_SIGNED
                     ? LOOM_INTEGER_CAST_SIGNED
                     : LOOM_INTEGER_CAST_SIGNED_THEN_UNSIGNED;
    input.unsigned_bit_count =
        kind == LOOM_INTEGER_CAST_UNSIGNED ? bit_count : 0;
  }
  return input;
}

loom_value_facts_t loom_integer_cast_apply(const loom_integer_cast_t* cast,
                                           loom_value_facts_t source_facts) {
  if (cast->kind == LOOM_INTEGER_CAST_UNSIGNED) {
    return loom_value_facts_wrap_unsigned_integer(source_facts,
                                                  cast->bit_count);
  }
  loom_value_facts_t result =
      loom_value_facts_wrap_integer(source_facts, cast->bit_count);
  if (cast->bit_count == 1) {
    result = loom_value_facts_sign_extend(result, 1);
  }
  if (cast->kind == LOOM_INTEGER_CAST_SIGNED_THEN_UNSIGNED) {
    result = loom_value_facts_wrap_unsigned_integer(result,
                                                    cast->unsigned_bit_count);
  }
  return result;
}

bool loom_integer_cast_is_identity(const loom_integer_cast_t* cast,
                                   loom_value_facts_t source_facts) {
  if (cast->bit_count == 0 || cast->bit_count > 64 ||
      loom_value_facts_is_float(source_facts)) {
    return false;
  }
  if (cast->kind == LOOM_INTEGER_CAST_SIGNED_THEN_UNSIGNED &&
      source_facts.range_lo < 0) {
    return false;
  }
  return cast->kind == LOOM_INTEGER_CAST_UNSIGNED
             ? loom_value_facts_fit_unsigned_bit_count(source_facts,
                                                       cast->bit_count)
             : loom_value_facts_fit_signed_bit_count(source_facts,
                                                     cast->bit_count);
}

bool loom_integer_cast_same_interpretation(const loom_integer_cast_t* left,
                                           const loom_integer_cast_t* right) {
  return left->bit_count == right->bit_count && left->kind == right->kind &&
         left->unsigned_bit_count == right->unsigned_bit_count;
}
