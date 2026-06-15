// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Matrix-operand encoding vocabulary.
//
// The text form of #matrix_operand uses symbolic names such as `f4e2m1` and
// `block_1d` so authored IR is readable and stable. The compiler immediately
// resolves those names into fact-table bitfields for contract matching and
// target selection; hot paths should never compare symbolic strings.

#ifndef LOOM_OPS_ENCODING_MATRIX_OPERAND_H_
#define LOOM_OPS_ENCODING_MATRIX_OPERAND_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_encoding_matrix_operand_symbol_set_e {
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT = 0,
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_PAYLOAD_PACKING,
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SCALE_TOPOLOGY,
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_AFFINE_POLICY,
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY,
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_CODEBOOK_POLICY,
  LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SPARSITY_POLICY,
} loom_encoding_matrix_operand_symbol_set_t;

// Resolves one symbolic #matrix_operand value into its target-independent
// bitfield value. Returns false when |symbol| is not part of |symbol_set|.
bool loom_encoding_matrix_operand_lookup_symbol(
    loom_encoding_matrix_operand_symbol_set_t symbol_set,
    iree_string_view_t symbol, uint64_t* out_value);

// Returns a compact diagnostic spelling of the symbols accepted by
// loom_encoding_matrix_operand_lookup_symbol().
iree_string_view_t loom_encoding_matrix_operand_expected_symbols(
    loom_encoding_matrix_operand_symbol_set_t symbol_set);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_ENCODING_MATRIX_OPERAND_H_
