// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/scalar_conversion.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float64_narrow.h"
#include "loom/target/arch/amdgpu/lower/encoding/float_to_integer64.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_encode.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/materializers.h"
#include "loom/target/arch/amdgpu/lower/source_integer_representation.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/lower/value/integer64.h"

static loom_scalar_type_t loom_amdgpu_scalar_type_or_none(loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return LOOM_SCALAR_TYPE_NONE;
  }
  return loom_type_element_type(type);
}

typedef enum loom_amdgpu_scalar_conversion_rule_index_e {
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_NONE = 0,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_TO_I1,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E4M3_TO_BF16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E5M2_TO_BF16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I32_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I32_TO_I64,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_SITOFP_I8_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_SITOFP_I16_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F8E4M3_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F8E5M2_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F16_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_BF16_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_,
} loom_amdgpu_scalar_conversion_rule_index_t;
static_assert(LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_ <= UINT8_MAX,
              "conversion rule indexes must fit in the byte selector table");

typedef struct loom_amdgpu_scalar_conversion_rule_t {
  // Lowering strategy selected when the rule matches.
  loom_amdgpu_scalar_conversion_kind_t kind;
  // Descriptor emitted by strategies that perform a conversion packet.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Auxiliary descriptor refs that must be present before the rule can select.
  loom_amdgpu_descriptor_ref_t required_descriptor_refs[3];
} loom_amdgpu_scalar_conversion_rule_t;
static_assert(sizeof(loom_amdgpu_scalar_conversion_rule_t) == 12,
              "scalar conversion rules must stay compact");

// Semantic scalar conversion plan shared by operations with the same target,
// types, and relevant value facts. Per-operation identities and later physical
// representation decisions are deliberately excluded.
typedef struct loom_amdgpu_scalar_conversion_plan_template_t {
  // Lowering strategy selected for the source/result type pair.
  uint8_t kind;
  // Static source integer payload bit count, or zero for non-integer sources.
  uint8_t source_bit_count;
  // Static result integer payload bit count, or zero for non-integer results.
  uint8_t result_bit_count;
  // Descriptor selected for conversion packets used by the strategy.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  union {
    // Exceptional-value simplifications for scalar FP8 decoding.
    struct {
      // Selected simplifications for the decoded value.
      loom_amdgpu_fp8_decode_value_flags_t value_flags;
    } fp8_decode;
    // Native packed FP8 encode strategy for an FP8-result truncation.
    loom_amdgpu_fp8_encode_plan_t fp8_encode;
    // Exact split-word strategy for narrowing an F64 source.
    loom_amdgpu_f64_narrow_plan_t f64_narrow;
    // Exact split-word strategy for producing an I64 payload.
    loom_amdgpu_float_to_integer64_plan_t float_to_integer64;
  };
} loom_amdgpu_scalar_conversion_plan_template_t;
static_assert(sizeof(loom_amdgpu_scalar_conversion_plan_template_t) == 36,
              "scalar conversion plan templates must stay compact");

typedef struct loom_amdgpu_scalar_conversion_query_state_t {
  // One-based template index by result value ordinal; zero means unretained.
  uint16_t* template_indices_by_value_ordinal;
  // Interned semantic plans selected by contract queries in this function.
  loom_amdgpu_scalar_conversion_plan_template_t* templates;
  // Number of allocated entries in |templates|.
  iree_host_size_t template_capacity;
  // Number of entries in |template_indices_by_value_ordinal|.
  loom_value_ordinal_t template_index_count;
  // Number of populated entries in |templates|.
  uint16_t template_count;
} loom_amdgpu_scalar_conversion_query_state_t;
static_assert(sizeof(loom_amdgpu_scalar_conversion_query_state_t) == 32,
              "scalar conversion query state must stay compact");

// At most one plan can be selected for an operation group, source/result type
// pair, and FP8 fact simplification state. This bound keeps one-based template
// indexes representable without coupling their width to function size.
#define LOOM_AMDGPU_SCALAR_CONVERSION_FACT_VARIANT_COUNT 16u
#define LOOM_AMDGPU_SCALAR_CONVERSION_TEMPLATE_COUNT_MAXIMUM           \
  (LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_ * LOOM_SCALAR_TYPE_COUNT_ * \
   LOOM_SCALAR_TYPE_COUNT_ * LOOM_AMDGPU_SCALAR_CONVERSION_FACT_VARIANT_COUNT)
static_assert(LOOM_AMDGPU_SCALAR_CONVERSION_TEMPLATE_COUNT_MAXIMUM < UINT16_MAX,
              "scalar conversion templates must fit in one-based indexes");

static int loom_amdgpu_scalar_conversion_query_state_key;

#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0() \
  {                                            \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(ref0) \
  {                                                \
      (ref0),                                      \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(ref0, ref1) \
  {                                                      \
      (ref0),                                            \
      (ref1),                                            \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                   \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(ref0, ref1, ref2) \
  {                                                            \
      (ref0),                                                  \
      (ref1),                                                  \
      (ref2),                                                  \
  }

static const uint8_t kLoomAmdgpuScalarConversionRuleIndexes
    [LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_][LOOM_SCALAR_TYPE_COUNT_]
    [LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I1] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_TO_I1,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I1] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_TO_I1,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I1] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_TO_I1,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I1] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_TO_I1,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16,
                [LOOM_SCALAR_TYPE_I64][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I32,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTF] =
            {
                [LOOM_SCALAR_TYPE_F8E4M3][LOOM_SCALAR_TYPE_BF16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E4M3_TO_BF16,
                [LOOM_SCALAR_TYPE_F8E5M2][LOOM_SCALAR_TYPE_BF16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E5M2_TO_BF16,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTSI] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I16,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I32_TO_I64,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTUI] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I16,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I32,
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I64,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I64,
                [LOOM_SCALAR_TYPE_I32][LOOM_SCALAR_TYPE_I64] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I32_TO_I64,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_SITOFP] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_F32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_SITOFP_I8_TO_F32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_F32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_SITOFP_I16_TO_F32,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_UITOFP] =
            {
                [LOOM_SCALAR_TYPE_I8][LOOM_SCALAR_TYPE_F32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32,
                [LOOM_SCALAR_TYPE_I16][LOOM_SCALAR_TYPE_F32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI] =
            {
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI] =
            {
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I32] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8,
                [LOOM_SCALAR_TYPE_F32][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16,
            },
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_BITCAST] =
            {
                [LOOM_SCALAR_TYPE_F8E4M3][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F8E4M3_TO_I8,
                [LOOM_SCALAR_TYPE_F8E5M2][LOOM_SCALAR_TYPE_I8] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F8E5M2_TO_I8,
                [LOOM_SCALAR_TYPE_F16][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F16_TO_I16,
                [LOOM_SCALAR_TYPE_BF16][LOOM_SCALAR_TYPE_I16] =
                    LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_BF16_TO_I16,
            },
};

static const loom_amdgpu_scalar_conversion_rule_t
    kLoomAmdgpuScalarConversionRules
        [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_] = {
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_TO_I1] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_INTEGER_TO_PREDICATE,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_TRUNCATE_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E4M3_TO_BF16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTF_F8E5M2_TO_BF16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I32_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32_RHS_INLINE)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I8_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I16_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTUI_I32_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_SITOFP_I8_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_I32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_SITOFP_I16_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_I32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F8E4M3_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F8E5M2_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_F16_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_BITCAST_BF16_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT)},
};

#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0
#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1
#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2
#undef LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3

#define LOOM_AMDGPU_SCALAR_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))
#define LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_CODE(op_group) \
  ((uint8_t)((op_group) + 1u))
#define LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(op, group) \
  [LOOM_AMDGPU_SCALAR_OP_INDEX(LOOM_OP_SCALAR_##op)] =        \
      LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_CODE(            \
          LOOM_AMDGPU_SCALAR_CONVERSION_OP_##group)

static_assert(LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_ < UINT8_MAX,
              "scalar conversion op-group codes must fit in uint8_t");

static const uint8_t
    kAmdgpuScalarConversionOpGroupCodeByScalarOp[LOOM_OP_SCALAR_COUNT_] = {
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(TRUNCI, TRUNCI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTF, EXTF),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTRUNC, FPTRUNC),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTSI, EXTSI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTUI, EXTUI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(SITOFP, SITOFP),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(UITOFP, UITOFP),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTOSI, FPTOSI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTOUI, FPTOUI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(BITCAST, BITCAST),
};

#undef LOOM_AMDGPU_SCALAR_OP_INDEX
#undef LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_CODE
#undef LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW

loom_amdgpu_scalar_conversion_op_group_t loom_amdgpu_scalar_conversion_op_group(
    loom_op_kind_t op_kind) {
  if (loom_op_dialect_id(op_kind) != LOOM_DIALECT_SCALAR) {
    return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
  const uint8_t op_index = loom_op_dialect_index(op_kind);
  if (op_index >=
      IREE_ARRAYSIZE(kAmdgpuScalarConversionOpGroupCodeByScalarOp)) {
    return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
  const uint8_t op_group_code =
      kAmdgpuScalarConversionOpGroupCodeByScalarOp[op_index];
  if (op_group_code == 0) {
    return LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_;
  }
  return (loom_amdgpu_scalar_conversion_op_group_t)(op_group_code - 1u);
}

static const loom_amdgpu_scalar_conversion_rule_t*
loom_amdgpu_scalar_conversion_rule_for(
    loom_amdgpu_scalar_conversion_op_group_t op_group,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type) {
  if (op_group >= LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_ ||
      !loom_scalar_type_is_valid(source_type) ||
      !loom_scalar_type_is_valid(result_type)) {
    return NULL;
  }
  const uint8_t rule_index =
      kLoomAmdgpuScalarConversionRuleIndexes[op_group][source_type]
                                            [result_type];
  if (rule_index == LOOM_AMDGPU_SCALAR_CONVERSION_RULE_NONE) {
    return NULL;
  }
  IREE_ASSERT(rule_index < IREE_ARRAYSIZE(kLoomAmdgpuScalarConversionRules));
  return &kLoomAmdgpuScalarConversionRules[rule_index];
}

IREE_ATTRIBUTE_NOINLINE static bool
loom_amdgpu_select_scalar_float_to_integer64_plan(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_conversion_op_group_t op_group,
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_value_id_t source, loom_value_id_t result,
    loom_amdgpu_scalar_conversion_plan_t* out_plan) {
  if (result_type != LOOM_SCALAR_TYPE_I64) {
    return false;
  }
  loom_amdgpu_float_to_integer64_signedness_t integer_signedness = 0;
  if (op_group == LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI) {
    integer_signedness = LOOM_AMDGPU_FLOAT_TO_INTEGER64_SIGNEDNESS_SIGNED;
  } else if (op_group == LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI) {
    integer_signedness = LOOM_AMDGPU_FLOAT_TO_INTEGER64_SIGNEDNESS_UNSIGNED;
  } else {
    return false;
  }
  loom_amdgpu_float_to_integer64_plan_t float_to_integer64 = {0};
  if (!loom_amdgpu_select_float_to_integer64_plan(descriptor_set, source_type,
                                                  integer_signedness,
                                                  &float_to_integer64)) {
    return false;
  }
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){
      .kind = LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FLOAT_TO_INTEGER64,
      .source = source,
      .result = result,
      .float_to_integer64 = float_to_integer64,
  };
  return true;
}

static bool loom_amdgpu_select_scalar_conversion_plan_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){0};
  if (source_op->operand_count != 1 || source_op->result_count != 1) {
    return false;
  }

  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_scalar_type_t source_type =
      loom_amdgpu_scalar_type_or_none(loom_module_value_type(module, source));
  const loom_scalar_type_t result_type =
      loom_amdgpu_scalar_type_or_none(loom_module_value_type(module, result));
  if (source_type == LOOM_SCALAR_TYPE_NONE ||
      result_type == LOOM_SCALAR_TYPE_NONE) {
    return false;
  }

  loom_amdgpu_f64_narrow_plan_t f64_narrow = {0};
  if (source_op->kind == LOOM_OP_SCALAR_FPTRUNC &&
      loom_amdgpu_select_f64_narrow_plan(descriptor_set, source_type,
                                         result_type, &f64_narrow)) {
    *out_plan = (loom_amdgpu_scalar_conversion_plan_t){
        .kind = LOOM_AMDGPU_SCALAR_CONVERSION_KIND_F64_NARROW,
        .source = source,
        .result = result,
        .f64_narrow = f64_narrow,
    };
    return true;
  }

  loom_amdgpu_fp8_encode_plan_t fp8_encode = {0};
  const loom_value_facts_t source_facts =
      fact_table != NULL ? loom_value_fact_table_lookup(fact_table, source)
                         : loom_value_facts_unknown();
  if (source_op->kind == LOOM_OP_SCALAR_FPTRUNC &&
      loom_amdgpu_select_fp8_encode_plan(
          descriptor_set, source_type, result_type, source_facts,
          loom_numeric_format_from_scalar_type(result_type), &fp8_encode)) {
    *out_plan = (loom_amdgpu_scalar_conversion_plan_t){
        .kind = LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE,
        .source = source,
        .result = result,
        .fp8_encode = fp8_encode,
    };
    return true;
  }

  const loom_amdgpu_scalar_conversion_op_group_t op_group =
      loom_amdgpu_scalar_conversion_op_group(source_op->kind);
  const loom_amdgpu_scalar_conversion_rule_t* rule =
      loom_amdgpu_scalar_conversion_rule_for(op_group, source_type,
                                             result_type);
  if (rule == NULL) {
    return loom_amdgpu_select_scalar_float_to_integer64_plan(
        descriptor_set, op_group, source_type, result_type, source, result,
        out_plan);
  }
  if ((rule->convert_descriptor_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
       !loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                           rule->convert_descriptor_ref)) ||
      !loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, rule->required_descriptor_refs,
          IREE_ARRAYSIZE(rule->required_descriptor_refs))) {
    return false;
  }
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){
      .kind = rule->kind,
      .source = source,
      .result = result,
      .source_bit_count =
          loom_amdgpu_integer_scalar_type_bit_count(source_type),
      .result_bit_count =
          loom_amdgpu_integer_scalar_type_bit_count(result_type),
      .convert_descriptor_ref = rule->convert_descriptor_ref,
  };
  if (rule->kind == LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16) {
    out_plan->fp8_decode.source_format =
        loom_numeric_format_from_scalar_type(source_type);
    out_plan->fp8_decode.value_flags =
        loom_amdgpu_fp8_decode_value_flags_from_facts(source_facts);
  }
  return true;
}

static bool loom_amdgpu_fp8_encode_plans_equal(
    const loom_amdgpu_fp8_encode_plan_t* lhs,
    const loom_amdgpu_fp8_encode_plan_t* rhs) {
  return lhs->kind == rhs->kind && lhs->flags == rhs->flags &&
         lhs->format.exponent_bits == rhs->format.exponent_bits &&
         lhs->format.mantissa_bits == rhs->format.mantissa_bits &&
         lhs->format.exponent_bias == rhs->format.exponent_bias &&
         lhs->format.special_policy == rhs->format.special_policy &&
         lhs->low_descriptor_ref == rhs->low_descriptor_ref &&
         lhs->high_descriptor_ref == rhs->high_descriptor_ref &&
         lhs->round_add_descriptor_ref == rhs->round_add_descriptor_ref &&
         lhs->sign_insert_descriptor_ref == rhs->sign_insert_descriptor_ref &&
         lhs->packed_i8_permute.kind == rhs->packed_i8_permute.kind &&
         lhs->packed_i8_permute.descriptor_ref ==
             rhs->packed_i8_permute.descriptor_ref;
}

static bool loom_amdgpu_scalar_conversion_plan_templates_equal(
    const loom_amdgpu_scalar_conversion_plan_template_t* lhs,
    const loom_amdgpu_scalar_conversion_plan_template_t* rhs) {
  if (lhs->kind != rhs->kind ||
      lhs->source_bit_count != rhs->source_bit_count ||
      lhs->result_bit_count != rhs->result_bit_count ||
      lhs->convert_descriptor_ref != rhs->convert_descriptor_ref) {
    return false;
  }
  switch (lhs->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16:
      return lhs->fp8_decode.value_flags == rhs->fp8_decode.value_flags;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE:
      return loom_amdgpu_fp8_encode_plans_equal(&lhs->fp8_encode,
                                                &rhs->fp8_encode);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_F64_NARROW:
      return lhs->f64_narrow.kind == rhs->f64_narrow.kind;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FLOAT_TO_INTEGER64:
      return lhs->float_to_integer64.kind == rhs->float_to_integer64.kind;
    default:
      return true;
  }
}

static loom_amdgpu_scalar_conversion_plan_template_t
loom_amdgpu_scalar_conversion_plan_template(
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  loom_amdgpu_scalar_conversion_plan_template_t plan_template = {
      .kind = plan->kind,
      .source_bit_count = plan->source_bit_count,
      .result_bit_count = plan->result_bit_count,
      .convert_descriptor_ref = plan->convert_descriptor_ref,
  };
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16:
      plan_template.fp8_decode.value_flags = plan->fp8_decode.value_flags;
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE:
      plan_template.fp8_encode = plan->fp8_encode;
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_F64_NARROW:
      plan_template.f64_narrow = plan->f64_narrow;
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FLOAT_TO_INTEGER64:
      plan_template.float_to_integer64 = plan->float_to_integer64;
      break;
    default:
      break;
  }
  return plan_template;
}

static loom_amdgpu_scalar_conversion_plan_t
loom_amdgpu_scalar_conversion_plan_from_template(
    const loom_amdgpu_scalar_conversion_plan_template_t* plan_template,
    const loom_module_t* module, loom_value_id_t source,
    loom_value_id_t result) {
  loom_amdgpu_scalar_conversion_plan_t plan = {
      .source = source,
      .result = result,
      .kind = (loom_amdgpu_scalar_conversion_kind_t)plan_template->kind,
      .source_bit_count = plan_template->source_bit_count,
      .result_bit_count = plan_template->result_bit_count,
      .convert_descriptor_ref = plan_template->convert_descriptor_ref,
  };
  switch (plan_template->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16:
      plan.fp8_decode.source_format =
          loom_numeric_format_from_scalar_type(loom_amdgpu_scalar_type_or_none(
              loom_module_value_type(module, source)));
      plan.fp8_decode.value_flags = plan_template->fp8_decode.value_flags;
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE:
      plan.fp8_encode = plan_template->fp8_encode;
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_F64_NARROW:
      plan.f64_narrow = plan_template->f64_narrow;
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FLOAT_TO_INTEGER64:
      plan.float_to_integer64 = plan_template->float_to_integer64;
      break;
    default:
      break;
  }
  return plan;
}

static const loom_amdgpu_scalar_conversion_plan_template_t*
loom_amdgpu_scalar_conversion_query_state_lookup(
    const loom_amdgpu_scalar_conversion_query_state_t* state,
    const loom_local_value_domain_t* value_domain, const loom_op_t* source_op) {
  if (state == NULL || state->template_indices_by_value_ordinal == NULL ||
      value_domain == NULL || source_op->result_count != 1) {
    return NULL;
  }
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_value_ordinal_t result_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, result);
  if (result_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
      result_ordinal >= state->template_index_count) {
    return NULL;
  }
  const uint16_t template_index_plus_one =
      state->template_indices_by_value_ordinal[result_ordinal];
  if (template_index_plus_one == 0) {
    return NULL;
  }
  const uint16_t template_index = template_index_plus_one - 1u;
  IREE_ASSERT_LT(template_index, state->template_count);
  return &state->templates[template_index];
}

static iree_status_t loom_amdgpu_scalar_conversion_retain_query_plan(
    const loom_target_contract_query_environment_t* environment,
    loom_amdgpu_scalar_conversion_query_state_t* state,
    const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  if (state == NULL) {
    return iree_ok_status();
  }
  IREE_ASSERT(environment->value_domain != NULL);
  const loom_local_value_domain_t* value_domain = environment->value_domain;
  if (state->template_indices_by_value_ordinal == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        environment->arena, value_domain->value_count,
        sizeof(*state->template_indices_by_value_ordinal),
        (void**)&state->template_indices_by_value_ordinal));
    memset(state->template_indices_by_value_ordinal, 0,
           value_domain->value_count *
               sizeof(*state->template_indices_by_value_ordinal));
    state->template_index_count = value_domain->value_count;
  }
  IREE_ASSERT_EQ(state->template_index_count, value_domain->value_count);

  const loom_amdgpu_scalar_conversion_plan_template_t plan_template =
      loom_amdgpu_scalar_conversion_plan_template(plan);
  uint16_t template_index = 0;
  for (; template_index < state->template_count; ++template_index) {
    if (loom_amdgpu_scalar_conversion_plan_templates_equal(
            &state->templates[template_index], &plan_template)) {
      break;
    }
  }
  if (template_index == state->template_count) {
    IREE_ASSERT_LT(state->template_count,
                   LOOM_AMDGPU_SCALAR_CONVERSION_TEMPLATE_COUNT_MAXIMUM);
    const iree_host_size_t minimum_capacity =
        state->template_capacity == 0
            ? 4
            : (iree_host_size_t)state->template_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        environment->arena, state->template_count, minimum_capacity,
        sizeof(*state->templates), &state->template_capacity,
        (void**)&state->templates));
    state->templates[template_index] = plan_template;
    ++state->template_count;
  }

  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  const loom_value_ordinal_t result_ordinal =
      loom_local_value_domain_try_ordinal(value_domain, result);
  IREE_ASSERT_NE(result_ordinal, LOOM_VALUE_ORDINAL_INVALID);
  uint16_t* retained_template_index_plus_one =
      &state->template_indices_by_value_ordinal[result_ordinal];
  IREE_ASSERT(*retained_template_index_plus_one == 0 ||
              *retained_template_index_plus_one == template_index + 1u);
  *retained_template_index_plus_one = template_index + 1u;
  return iree_ok_status();
}

static bool loom_amdgpu_scalar_conversion_op_is_callback_owned(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_op_group_t op_group) {
  if ((op_group != LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI &&
       op_group != LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI) ||
      source_op->result_count != 1) {
    return true;
  }
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  return loom_amdgpu_scalar_type_or_none(
             loom_module_value_type(module, result)) != LOOM_SCALAR_TYPE_I32;
}

iree_status_t loom_amdgpu_query_scalar_conversion_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  const loom_amdgpu_scalar_conversion_op_group_t op_group =
      loom_amdgpu_scalar_conversion_op_group(source_op->kind);
  if (op_group == LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_ ||
      environment->vector_lane_projection.source_lane_count != 0 ||
      !loom_amdgpu_scalar_conversion_op_is_callback_owned(
          environment->module, source_op, op_group)) {
    return iree_ok_status();
  }

  loom_amdgpu_scalar_conversion_plan_t plan = {0};
  if (!loom_amdgpu_select_scalar_conversion_plan_impl(
          environment->module, environment->fact_table,
          environment->descriptor_set, source_op, &plan)) {
    return iree_ok_status();
  }

  loom_target_contract_query_environment_t analyzed_environment = *environment;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_resolve_function_analysis(
      &analyzed_environment));
  environment = &analyzed_environment;
  loom_amdgpu_scalar_conversion_query_state_t* state = NULL;
  if (environment->value_domain != NULL &&
      environment->target_state_allocator.fn != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_target_contract_query_get_or_allocate_target_state(
            environment, &loom_amdgpu_scalar_conversion_query_state_key,
            sizeof(*state), (void**)&state));
    if (loom_amdgpu_scalar_conversion_query_state_lookup(
            state, environment->value_domain, source_op) != NULL) {
      out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_conversion_retain_query_plan(
      environment, state, source_op, &plan));
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_conversion_plan_t){0};
  *out_selected = false;
  const loom_amdgpu_scalar_conversion_query_state_t* state =
      (const loom_amdgpu_scalar_conversion_query_state_t*)
          loom_low_lower_lookup_target_state(
              context, &loom_amdgpu_scalar_conversion_query_state_key,
              sizeof(*state));
  const loom_amdgpu_scalar_conversion_plan_template_t* plan_template =
      loom_amdgpu_scalar_conversion_query_state_lookup(
          state, loom_low_lower_context_value_domain(context), source_op);
  if (plan_template == NULL) {
    return iree_ok_status();
  }
  const loom_value_id_t source = loom_op_const_operands(source_op)[0];
  const loom_value_id_t result = loom_op_const_results(source_op)[0];
  *out_plan = loom_amdgpu_scalar_conversion_plan_from_template(
      plan_template, loom_low_lower_context_module(context), source, result);
  *out_selected = true;
  const loom_amdgpu_scalar_conversion_op_group_t op_group =
      loom_amdgpu_scalar_conversion_op_group(source_op->kind);
  switch (op_group) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOSI:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTOUI:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_BITCAST:
      if (out_plan->result_bit_count == 8 || out_plan->result_bit_count == 16) {
        out_plan->narrow_representation =
            loom_amdgpu_source_integer_representation_lookup(context,
                                                             out_plan->result);
      }
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTSI:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTUI:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_SITOFP:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_UITOFP:
      if (out_plan->source_bit_count == 8 || out_plan->source_bit_count == 16) {
        out_plan->narrow_representation =
            loom_amdgpu_source_integer_representation_lookup(context,
                                                             out_plan->source);
      }
      break;
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_EXTF:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_FPTRUNC:
    case LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_:
      break;
  }
  if (out_plan->kind == LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT &&
      out_plan->narrow_representation ==
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS) {
    // Consumers require only the low payload bits, so no normalization or
    // register-bank conversion is necessary.
    out_plan->kind = LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS;
  }
  if (out_plan->kind == LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS) {
    return iree_ok_status();
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
      context, source_op, out_plan->result, &result_type));
  return loom_low_lower_plan_value_type(context, out_plan->result, result_type);
}

iree_status_t loom_amdgpu_finalize_scalar_conversion_plan(
    loom_low_lower_context_t* context,
    loom_amdgpu_scalar_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS:
      return loom_low_lower_plan_value_type(
          context, plan->result,
          loom_low_lower_value_binding_type(context, plan->source));
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT:
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE:
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FLOAT_TO_INTEGER64:
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32:
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW:
      return loom_amdgpu_prepare_vgpr_literal(context, plan->source,
                                              &plan->source_materialization);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64:
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND:
      if (plan->source_bit_count == 32 && plan->result_bit_count == 64 &&
          loom_amdgpu_low_type_is_register_class(
              context, loom_low_lower_value_binding_type(context, plan->result),
              LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
        return loom_amdgpu_prepare_vgpr_literal(context, plan->source,
                                                &plan->source_materialization);
      }
      break;
    default:
      break;
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }

  loom_amdgpu_scalar_conversion_plan_t plan = {0};
  if (!loom_amdgpu_select_scalar_conversion_plan_impl(
          loom_target_low_legality_module(context),
          loom_target_low_legality_fact_table(context),
          loom_target_low_legality_descriptor_set(context), op, &plan)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr_zero_extend(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint32_t source_bit_count,
    loom_value_id_t* out_low_result) {
  *out_low_result = low_source;
  IREE_ASSERT(source_bit_count == 8 || source_bit_count == 16 ||
              source_bit_count == 32);
  if (source_bit_count >= 32) {
    return iree_ok_status();
  }
  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT, low_source,
      iree_math_mask_low_bits_u32(UINT32_MAX, (int32_t)source_bit_count),
      lane_type, out_low_result);
}

static iree_status_t loom_amdgpu_lookup_and_normalize_vgpr_narrow_integer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, uint32_t source_bit_count,
    loom_low_representation_id_t source_representation,
    loom_low_representation_id_t required_representation,
    loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT(source_bit_count == 8 || source_bit_count == 16);
  IREE_ASSERT(required_representation ==
                  LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED ||
              required_representation ==
                  LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED);
  loom_value_id_t low_source = loom_low_lower_lookup_value(context, source);

  const loom_type_t low_type = loom_module_value_type(
      loom_low_lower_context_module(context), low_source);
  const bool is_partial_sgpr =
      loom_amdgpu_low_type_is_register_class(context, low_type,
                                             LOOM_AMDGPU_REG_CLASS_ID_SGPR) &&
      loom_amdgpu_low_value_defines_register_low16(context, low_source);
  if (is_partial_sgpr) {
    loom_type_t vgpr_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
    const loom_amdgpu_bitfield_extract_mode_t mode =
        required_representation ==
                LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED
            ? LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND
            : LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_ZERO_EXTEND;
    return loom_amdgpu_extract_register_bitfield(
        context, source_op, low_source, /*bit_offset=*/0, source_bit_count,
        mode, vgpr_type, out_low_source);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_registers(
      context, source_op, source, &low_source));
  return loom_amdgpu_normalize_narrow_integer(
      context, source_op, low_source, source_bit_count, source_representation,
      required_representation,
      loom_module_value_type(loom_low_lower_context_module(context),
                             low_source),
      out_low_source);
}

static iree_status_t loom_amdgpu_lookup_scalar_conversion_integer_source(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan, loom_type_t result_type,
    loom_low_representation_id_t required_representation,
    loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  const bool result_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (result_is_vgpr) {
    if (plan->source_bit_count < 32) {
      return loom_amdgpu_lookup_and_normalize_vgpr_narrow_integer(
          context, source_op, plan->source, plan->source_bit_count,
          plan->narrow_representation, required_representation, out_low_source);
    }
    return loom_amdgpu_emit_prepared_vgpr_value(
        context, source_op, plan->source, plan->source_materialization,
        out_low_source);
  }

  const bool result_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!result_is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected an unsupported register class");
    IREE_BUILTIN_UNREACHABLE();
  }

  *out_low_source = loom_low_lower_lookup_value(context, plan->source);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, *out_low_source);
  const bool source_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!source_is_sgpr || loom_low_register_type_unit_count(source_type) != 1) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected an SGPR result from a "
        "non-SGPR source");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (plan->source_bit_count < 32) {
    return loom_amdgpu_normalize_narrow_integer(
        context, source_op, *out_low_source, plan->source_bit_count,
        plan->narrow_representation, required_representation, source_type,
        out_low_source);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_bind_register64_lanes(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_bits,
    loom_value_id_t high_bits) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_bits);
  const loom_type_t high_type = loom_module_value_type(module, high_bits);
  if (!loom_low_type_is_register(low_type) ||
      loom_low_register_type_unit_count(low_type) != 1 ||
      !loom_type_equal(low_type, high_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion produced incompatible i64 lanes");
    IREE_BUILTIN_UNREACHABLE();
  }
  const loom_type_t result_type =
      loom_low_register_carrier_type_with_unit_count(low_type, 2);
  const loom_value_id_t lanes[] = {
      low_bits,
      high_bits,
  };
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, lanes, IREE_ARRAYSIZE(lanes), result_type,
      &low_result));
  return loom_low_lower_bind_value(context, source_result, low_result);
}

static iree_status_t loom_amdgpu_bind_zero_extended_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_source) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lane_type = loom_module_value_type(module, low_source);
  const bool lane_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  loom_amdgpu_descriptor_ref_t zero_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
  if (!lane_is_vgpr) {
    const bool lane_is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (!lane_is_sgpr) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU scalar conversion zero-extended a non-register source");
      IREE_BUILTIN_UNREACHABLE();
    }
    zero_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32;
  }
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, zero_descriptor_ref, 0, lane_type, &high_bits));
  return loom_amdgpu_bind_register64_lanes(context, source_op, source_result,
                                           low_source, high_bits);
}

static iree_status_t loom_amdgpu_emit_scalar_fp8_to_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_registers(
      context, source_op, plan->source, &low_source));
  loom_value_id_t low_byte = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_zero_extend(
      context, source_op, low_source, 8, &low_byte));

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
      context, plan->fp8_decode.source_format, plan->fp8_decode.source_format,
      &decode_plan));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
      context, source_op, decode_plan, low_byte, plan->fp8_decode.value_flags,
      vgpr_type, sgpr_type, mask_type, &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_emit_scalar_fp8_encode(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_scalar_type_t source_type = loom_amdgpu_scalar_type_or_none(
      loom_module_value_type(module, plan->source));
  IREE_ASSERT(source_type == LOOM_SCALAR_TYPE_F16 ||
              source_type == LOOM_SCALAR_TYPE_BF16 ||
              source_type == LOOM_SCALAR_TYPE_F32);

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_vgpr_value(
      context, source_op, plan->source, plan->source_materialization,
      &low_source));

  loom_type_t lane_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
  loom_amdgpu_fp8_encode_emission_state_t emission_state = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_fp8_encode_emission(
      context, source_op, &plan->fp8_encode, /*encoded_lane_count=*/1,
      lane_type, &emission_state));

  loom_value_id_t encoded_source = low_source;
  loom_value_id_t high_source = low_source;
  if (plan->fp8_encode.kind == LOOM_AMDGPU_FP8_ENCODE_KIND_F16_PAIR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_duplicate_f16_lane(
        context, source_op, low_source, lane_type, &encoded_source));
    high_source = LOOM_VALUE_ID_INVALID;
  } else if (source_type == LOOM_SCALAR_TYPE_F16 &&
             plan->fp8_encode.kind !=
                 LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16,
        low_source, lane_type, &encoded_source));
    high_source = encoded_source;
  } else if (source_type == LOOM_SCALAR_TYPE_BF16) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        low_source, lane_type, &encoded_source));
    high_source = encoded_source;
  }

  if (loom_amdgpu_fp8_encode_plan_is_software(&plan->fp8_encode)) {
    loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
    if (plan->fp8_encode.kind ==
        LOOM_AMDGPU_FP8_ENCODE_KIND_F16_SOFTWARE_E5M2) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f16_e5m2_lane(
          context, source_op, &plan->fp8_encode, &emission_state,
          encoded_source, &low_result));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_software_f32_lane(
          context, source_op, &plan->fp8_encode, &emission_state,
          encoded_source, &low_result));
    }
    return loom_low_lower_bind_value(context, plan->result, low_result);
  }
  if (loom_amdgpu_fp8_encode_plan_is_fnuz_bridge(&plan->fp8_encode)) {
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_fnuz_f32_lanes(
        context, source_op, &plan->fp8_encode, &emission_state, &encoded_source,
        /*source_lane_count=*/1, &packed));
    return loom_low_lower_bind_value(context, plan->result, packed);
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_encode_low_pair(
      context, source_op, &plan->fp8_encode, &emission_state, encoded_source,
      high_source, &low_result));
  if (loom_amdgpu_fp8_encode_plan_canonicalizes_native_nan(&plan->fp8_encode)) {
    const loom_value_id_t source_lanes[2] = {encoded_source, high_source};
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fp8_encode_native_nan_canonicalization(
            context, source_op, &plan->fp8_encode, &emission_state,
            source_lanes, IREE_ARRAYSIZE(source_lanes), low_result,
            &low_result));
  }
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

static iree_status_t loom_amdgpu_emit_scalar_float_to_integer64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_scalar_type_t source_type = loom_amdgpu_scalar_type_or_none(
      loom_module_value_type(module, plan->source));
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_vgpr_value(
      context, source_op, plan->source, plan->source_materialization,
      &low_source));

  if (source_type == LOOM_SCALAR_TYPE_BF16) {
    loom_type_t word_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &word_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
        low_source, word_type, &low_source));
  }

  loom_amdgpu_float_to_integer64_emission_t emission = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_initialize_float_to_integer64_emission(
      context, source_op, &plan->float_to_integer64, &emission));
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_float_to_integer64(
      context, source_op, &plan->float_to_integer64, &emission, low_source,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}

iree_status_t loom_amdgpu_lower_integer_to_predicate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t result, loom_type_t result_type) {
  const bool is_mask = loom_low_register_type_unit_count(result_type) == 2;
  loom_type_t word_type = loom_type_none();
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_bit = LOOM_VALUE_ID_INVALID;
  if (is_mask) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &word_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
        context, source_op, source, &low_source));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_source, 1, word_type, &low_bit));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &word_type));
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_sgpr_address(
        context, source_op, source, &low_source));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32, low_source, 1,
        word_type, &low_bit));
  }
  loom_value_id_t low_result = low_bit;
  if (is_mask || loom_low_register_type_class_id(result_type) ==
                     LOOM_AMDGPU_REG_CLASS_ID_SCC) {
    loom_value_id_t low_true = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op,
        is_mask ? LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32
                : LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        1, word_type, &low_true));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_binary(
        context, source_op,
        is_mask ? LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32
                : LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
        low_bit, low_true, result_type, &low_result));
  }
  return loom_low_lower_bind_value(context, result, low_result);
}

iree_status_t loom_amdgpu_lower_scalar_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  if (plan->kind == LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS) {
    return loom_low_lower_bind_value_alias(context, plan->source, plan->result);
  }
  const loom_type_t result_type =
      loom_low_lower_value_binding_type(context, plan->result);
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_INTEGER_TO_PREDICATE:
      return loom_amdgpu_lower_integer_to_predicate(
          context, source_op, plan->source, plan->result, result_type);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_TRUNCATE_LOW_32: {
      loom_value_id_t low_source =
          loom_low_lower_lookup_value(context, plan->source);
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, /*lane_offset=*/0, result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT: {
      loom_value_id_t low_source =
          loom_low_lower_lookup_value(context, plan->source);
      if (!loom_amdgpu_low_value_defines_register_low16(context, low_source)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_vgpr_value(
            context, source_op, plan->source, plan->source_materialization,
            &low_source));
      }
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_narrow_integer(
          context, source_op, low_source, plan->result_bit_count,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS,
          plan->narrow_representation, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NARROW_RESULT_LOW_32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->source, &low_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_narrow_integer(
          context, source_op, low_source, plan->result_bit_count,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS,
          plan->narrow_representation, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND: {
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_scalar_conversion_integer_source(
          context, source_op, plan, result_type,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_scalar_conversion_integer_source(
          context, source_op, plan, result_type,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED,
          &low_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_from_i32(
          context, source_op, low_source, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_scalar_conversion_integer_source(
          context, source_op, plan, result_type,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
          &low_source));
      if (plan->result_bit_count == 64) {
        return loom_amdgpu_bind_zero_extended_i64(context, source_op,
                                                  plan->result, low_source);
      }
      return loom_low_lower_bind_value(context, plan->result, low_source);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SITOFP_NARROW_TO_F32: {
      loom_value_id_t sign_extended_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_and_normalize_vgpr_narrow_integer(
          context, source_op, plan->source, plan->source_bit_count,
          plan->narrow_representation,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED,
          &sign_extended_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref,
          sign_extended_source, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32: {
      loom_value_id_t zero_extended_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_and_normalize_vgpr_narrow_integer(
          context, source_op, plan->source, plan->source_bit_count,
          plan->narrow_representation,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED,
          &zero_extended_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref,
          zero_extended_source, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_TO_BF16:
      return loom_amdgpu_emit_scalar_fp8_to_bf16(context, source_op, plan);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FP8_ENCODE:
      return loom_amdgpu_emit_scalar_fp8_encode(context, source_op, plan);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_F64_NARROW:
      return loom_amdgpu_emit_f64_narrow(context, source_op, plan->source,
                                         plan->result, &plan->f64_narrow);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FLOAT_TO_INTEGER64:
      return loom_amdgpu_emit_scalar_float_to_integer64(context, source_op,
                                                        plan);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_vgpr_value(
          context, source_op, plan->source, plan->source_materialization,
          &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref, low_source,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_prepared_vgpr_value(
          context, source_op, plan->source, plan->source_materialization,
          &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t converted_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref, low_source,
          lane_type, &converted_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_normalize_narrow_integer(
          context, source_op, converted_source, plan->result_bit_count,
          LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_LOW_BITS,
          plan->narrow_representation, result_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS:
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU scalar conversion plan kind");
  return iree_ok_status();
}
