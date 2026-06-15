// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/matrix_operand.h"

#include <stdint.h>

#include "loom/util/fact_table.h"
#include "loom/util/stable_id.h"

typedef struct loom_encoding_matrix_operand_symbol_t {
  // Authored symbolic spelling accepted by #matrix_operand.
  iree_string_view_t spelling;

  // Target-independent fact bit represented by spelling.
  uint64_t value;
} loom_encoding_matrix_operand_symbol_t;

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_numeric_format_symbols[] = {
        {IREE_SVL("none"), LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE},
        {IREE_SVL("f64"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F64},
        {IREE_SVL("f32"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F32},
        {IREE_SVL("tf32"), LOOM_VALUE_FACT_NUMERIC_FORMAT_TF32},
        {IREE_SVL("f16"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F16},
        {IREE_SVL("bf16"), LOOM_VALUE_FACT_NUMERIC_FORMAT_BF16},
        {IREE_SVL("i32"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I32},
        {IREE_SVL("u32"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U32},
        {IREE_SVL("i16"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I16},
        {IREE_SVL("u16"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U16},
        {IREE_SVL("i8"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I8},
        {IREE_SVL("u8"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U8},
        {IREE_SVL("i6"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I6},
        {IREE_SVL("u6"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U6},
        {IREE_SVL("i5"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I5},
        {IREE_SVL("u5"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U5},
        {IREE_SVL("i4"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I4},
        {IREE_SVL("u4"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U4},
        {IREE_SVL("i3"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I3},
        {IREE_SVL("u3"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U3},
        {IREE_SVL("i2"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I2},
        {IREE_SVL("u2"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U2},
        {IREE_SVL("i1"), LOOM_VALUE_FACT_NUMERIC_FORMAT_I1},
        {IREE_SVL("u1"), LOOM_VALUE_FACT_NUMERIC_FORMAT_U1},
        {IREE_SVL("f8e4m3"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3},
        {IREE_SVL("f8e5m2"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E5M2},
        {IREE_SVL("f8e4m3fn"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN},
        {IREE_SVL("e8m0"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E8M0},
        {IREE_SVL("bf8"), LOOM_VALUE_FACT_NUMERIC_FORMAT_BF8},
        {IREE_SVL("f6e3m2"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E3M2},
        {IREE_SVL("f6e2m3"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F6_E2M3},
        {IREE_SVL("bf6"), LOOM_VALUE_FACT_NUMERIC_FORMAT_BF6},
        {IREE_SVL("f4e2m1"), LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1},
        {IREE_SVL("ternary"), LOOM_VALUE_FACT_NUMERIC_FORMAT_TERNARY},
        {IREE_SVL("sign_bit"), LOOM_VALUE_FACT_NUMERIC_FORMAT_SIGN_BIT},
        {IREE_SVL("codebook_index"),
         LOOM_VALUE_FACT_NUMERIC_FORMAT_CODEBOOK_INDEX},
        {IREE_SVL("quant_i8"), LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8},
        {IREE_SVL("quant_i6"), LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I6},
        {IREE_SVL("quant_i4"), LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I4},
};

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_payload_packing_symbols[] = {
        {IREE_SVL("dense_lanes"), LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES},
        {IREE_SVL("little_endian_nibbles"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES},
        {IREE_SVL("big_endian_nibbles"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_BIG_ENDIAN_NIBBLES},
        {IREE_SVL("bitfield_stream"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_BITFIELD_STREAM},
        {IREE_SVL("bitplane_stream"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_BITPLANE_STREAM},
        {IREE_SVL("multi_stream"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM},
        {IREE_SVL("base_n_packed"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_BASE_N_PACKED},
        {IREE_SVL("codebook_indices"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_CODEBOOK_INDICES},
        {IREE_SVL("target_fragment"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT},
        {IREE_SVL("interleaved_scale_payload"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_INTERLEAVED_SCALE_PAYLOAD},
        {IREE_SVL("separate_scale_payload"),
         LOOM_VALUE_FACT_PAYLOAD_PACKING_SEPARATE_SCALE_PAYLOAD},
};

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_scale_topology_symbols[] = {
        {IREE_SVL("none"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_NONE},
        {IREE_SVL("tensor_global"),
         LOOM_VALUE_FACT_SCALE_TOPOLOGY_TENSOR_GLOBAL},
        {IREE_SVL("row"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_ROW},
        {IREE_SVL("column"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_COLUMN},
        {IREE_SVL("channel"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_CHANNEL},
        {IREE_SVL("group_1d"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_GROUP_1D},
        {IREE_SVL("block_1d"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D},
        {IREE_SVL("block_2d"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D},
        {IREE_SVL("subblock_in_superblock"),
         LOOM_VALUE_FACT_SCALE_TOPOLOGY_SUBBLOCK_IN_SUPERBLOCK},
        {IREE_SVL("hierarchical"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL},
        {IREE_SVL("per_token"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_TOKEN},
        {IREE_SVL("per_head"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_HEAD},
        {IREE_SVL("per_page"), LOOM_VALUE_FACT_SCALE_TOPOLOGY_PER_PAGE},
        {IREE_SVL("runtime_amax_derived"),
         LOOM_VALUE_FACT_SCALE_TOPOLOGY_RUNTIME_AMAX_DERIVED},
};

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_affine_policy_symbols[] = {
        {IREE_SVL("none"), LOOM_VALUE_FACT_AFFINE_POLICY_NONE},
        {IREE_SVL("scale_only"), LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY},
        {IREE_SVL("scale_plus_min"),
         LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN},
        {IREE_SVL("scale_plus_zero_point"),
         LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_ZERO_POINT},
        {IREE_SVL("scale_plus_bias"),
         LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_BIAS},
        {IREE_SVL("super_scale_times_subscale"),
         LOOM_VALUE_FACT_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE},
        {IREE_SVL("sum_correction"),
         LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION},
};

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_rounding_policy_symbols[] = {
        {IREE_SVL("none"), LOOM_VALUE_FACT_ROUNDING_POLICY_NONE},
        {IREE_SVL("nearest_even"),
         LOOM_VALUE_FACT_ROUNDING_POLICY_NEAREST_EVEN},
        {IREE_SVL("nearest_away"),
         LOOM_VALUE_FACT_ROUNDING_POLICY_NEAREST_AWAY},
        {IREE_SVL("toward_zero"), LOOM_VALUE_FACT_ROUNDING_POLICY_TOWARD_ZERO},
        {IREE_SVL("down"), LOOM_VALUE_FACT_ROUNDING_POLICY_DOWN},
        {IREE_SVL("up"), LOOM_VALUE_FACT_ROUNDING_POLICY_UP},
        {IREE_SVL("stochastic"), LOOM_VALUE_FACT_ROUNDING_POLICY_STOCHASTIC},
        {IREE_SVL("satfinite"), LOOM_VALUE_FACT_ROUNDING_POLICY_SATFINITE},
        {IREE_SVL("overflow_to_inf"),
         LOOM_VALUE_FACT_ROUNDING_POLICY_OVERFLOW_TO_INF},
        {IREE_SVL("overflow_to_nan"),
         LOOM_VALUE_FACT_ROUNDING_POLICY_OVERFLOW_TO_NAN},
        {IREE_SVL("flush_subnormal"),
         LOOM_VALUE_FACT_ROUNDING_POLICY_FLUSH_SUBNORMAL},
        {IREE_SVL("preserve_subnormal"),
         LOOM_VALUE_FACT_ROUNDING_POLICY_PRESERVE_SUBNORMAL},
        {IREE_SVL("relu_clamp"), LOOM_VALUE_FACT_ROUNDING_POLICY_RELU_CLAMP},
        {IREE_SVL("finite_only"), LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY},
};

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_codebook_policy_symbols[] = {
        {IREE_SVL("none"), LOOM_VALUE_FACT_CODEBOOK_POLICY_NONE},
        {IREE_SVL("static_builtin_table"),
         LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_BUILTIN_TABLE},
        {IREE_SVL("static_symbol_table"),
         LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_SYMBOL_TABLE},
        {IREE_SVL("global_data_table"),
         LOOM_VALUE_FACT_CODEBOOK_POLICY_GLOBAL_DATA_TABLE},
        {IREE_SVL("dynamic_table_operand"),
         LOOM_VALUE_FACT_CODEBOOK_POLICY_DYNAMIC_TABLE_OPERAND},
        {IREE_SVL("per_superblock_table"),
         LOOM_VALUE_FACT_CODEBOOK_POLICY_PER_SUPERBLOCK_TABLE},
};

static const loom_encoding_matrix_operand_symbol_t
    loom_encoding_matrix_operand_sparsity_policy_symbols[] = {
        {IREE_SVL("none"), LOOM_VALUE_FACT_SPARSITY_POLICY_NONE},
        {IREE_SVL("mask"), LOOM_VALUE_FACT_SPARSITY_POLICY_MASK},
        {IREE_SVL("n_m_structured"),
         LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED},
        {IREE_SVL("block_sparse"),
         LOOM_VALUE_FACT_SPARSITY_POLICY_BLOCK_SPARSE},
        {IREE_SVL("bsr"), LOOM_VALUE_FACT_SPARSITY_POLICY_BSR},
        {IREE_SVL("csr"), LOOM_VALUE_FACT_SPARSITY_POLICY_CSR},
        {IREE_SVL("coo"), LOOM_VALUE_FACT_SPARSITY_POLICY_COO},
        {IREE_SVL("page_table"), LOOM_VALUE_FACT_SPARSITY_POLICY_PAGE_TABLE},
        {IREE_SVL("moe_routing"), LOOM_VALUE_FACT_SPARSITY_POLICY_MOE_ROUTING},
        {IREE_SVL("outlier_side_stream"),
         LOOM_VALUE_FACT_SPARSITY_POLICY_OUTLIER_SIDE_STREAM},
};

static bool loom_encoding_matrix_operand_lookup_in_table(
    const loom_encoding_matrix_operand_symbol_t* symbols,
    iree_host_size_t symbol_count, iree_string_view_t symbol,
    uint64_t* out_value) {
  *out_value = 0;
  uint64_t symbol_id = loom_stable_id_from_string(symbol);
  for (iree_host_size_t i = 0; i < symbol_count; ++i) {
    if (loom_stable_id_from_string(symbols[i].spelling) == symbol_id) {
      *out_value = symbols[i].value;
      return true;
    }
  }
  return false;
}

bool loom_encoding_matrix_operand_lookup_symbol(
    loom_encoding_matrix_operand_symbol_set_t symbol_set,
    iree_string_view_t symbol, uint64_t* out_value) {
  if (!out_value) return false;
  switch (symbol_set) {
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_numeric_format_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_numeric_format_symbols),
          symbol, out_value);
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_PAYLOAD_PACKING:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_payload_packing_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_payload_packing_symbols),
          symbol, out_value);
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SCALE_TOPOLOGY:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_scale_topology_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_scale_topology_symbols),
          symbol, out_value);
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_AFFINE_POLICY:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_affine_policy_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_affine_policy_symbols),
          symbol, out_value);
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_rounding_policy_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_rounding_policy_symbols),
          symbol, out_value);
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_CODEBOOK_POLICY:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_codebook_policy_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_codebook_policy_symbols),
          symbol, out_value);
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SPARSITY_POLICY:
      return loom_encoding_matrix_operand_lookup_in_table(
          loom_encoding_matrix_operand_sparsity_policy_symbols,
          IREE_ARRAYSIZE(loom_encoding_matrix_operand_sparsity_policy_symbols),
          symbol, out_value);
    default:
      *out_value = 0;
      return false;
  }
}

iree_string_view_t loom_encoding_matrix_operand_expected_symbols(
    loom_encoding_matrix_operand_symbol_set_t symbol_set) {
  switch (symbol_set) {
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_NUMERIC_FORMAT:
      return IREE_SV("numeric format symbol");
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_PAYLOAD_PACKING:
      return IREE_SV("payload packing symbol");
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SCALE_TOPOLOGY:
      return IREE_SV("scale topology symbol");
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_AFFINE_POLICY:
      return IREE_SV("affine policy symbol");
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_ROUNDING_POLICY:
      return IREE_SV("rounding policy symbol");
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_CODEBOOK_POLICY:
      return IREE_SV("codebook policy symbol");
    case LOOM_ENCODING_MATRIX_OPERAND_SYMBOL_SET_SPARSITY_POLICY:
      return IREE_SV("sparsity policy symbol");
    default:
      return IREE_SV("matrix operand symbol");
  }
}
