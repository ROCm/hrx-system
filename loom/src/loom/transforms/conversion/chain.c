// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/conversion/chain.h"

// Rows are the outer conversion, columns its immediate producer. Unlisted
// pairs retain observable intermediate rounding or integer truncation.
static const loom_conversion_chain_match_t
    kConversionChains[LOOM_CONVERSION_COUNT_][LOOM_CONVERSION_COUNT_] = {
        [LOOM_CONVERSION_EXTF] =
            {
                [LOOM_CONVERSION_EXTF] = {.candidate = LOOM_CONVERSION_EXTF},
            },
        [LOOM_CONVERSION_FPTRUNC] =
            {
                [LOOM_CONVERSION_EXTF] = {.candidate = LOOM_CONVERSION_EXTF},
            },
        [LOOM_CONVERSION_EXTSI] =
            {
                [LOOM_CONVERSION_EXTSI] = {.candidate = LOOM_CONVERSION_EXTSI},
                [LOOM_CONVERSION_EXTUI] = {.candidate = LOOM_CONVERSION_EXTUI},
            },
        [LOOM_CONVERSION_EXTUI] =
            {
                [LOOM_CONVERSION_EXTSI] =
                    {
                        .candidate = LOOM_CONVERSION_EXTUI,
                        .flags = LOOM_CONVERSION_CHAIN_FLAG_NON_NEGATIVE,
                    },
                [LOOM_CONVERSION_EXTUI] = {.candidate = LOOM_CONVERSION_EXTUI},
            },
        [LOOM_CONVERSION_TRUNCI] =
            {
                [LOOM_CONVERSION_EXTSI] = {.candidate = LOOM_CONVERSION_EXTSI},
                [LOOM_CONVERSION_EXTUI] = {.candidate = LOOM_CONVERSION_EXTUI},
                [LOOM_CONVERSION_TRUNCI] = {.candidate =
                                                LOOM_CONVERSION_TRUNCI},
            },
};

loom_conversion_chain_match_t loom_conversion_chain_match(
    loom_conversion_kind_t outer_kind, loom_conversion_kind_t inner_kind) {
  return kConversionChains[outer_kind][inner_kind];
}

loom_conversion_kind_t loom_conversion_chain_resolve(
    loom_conversion_kind_t candidate, loom_type_t input_type,
    loom_type_t result_type) {
  if (loom_type_equal(input_type, result_type)) {
    return LOOM_CONVERSION_IDENTITY;
  }
  const int32_t input_bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(input_type));
  const int32_t result_bitwidth =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (candidate == LOOM_CONVERSION_EXTF) {
    // Equal-width formats can differ in precision, range, and special values.
    // Neither extf nor fptrunc represents such a direct conversion.
    if (input_bitwidth == result_bitwidth) {
      return LOOM_CONVERSION_NONE;
    }
    return result_bitwidth > input_bitwidth ? LOOM_CONVERSION_EXTF
                                            : LOOM_CONVERSION_FPTRUNC;
  }
  return result_bitwidth > input_bitwidth ? candidate : LOOM_CONVERSION_TRUNCI;
}
