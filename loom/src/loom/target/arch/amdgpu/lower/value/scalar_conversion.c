// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/scalar_conversion.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/lower/bitpack.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8_encode.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
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
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16,
  LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_,
} loom_amdgpu_scalar_conversion_rule_index_t;
static_assert(LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_ <= UINT8_MAX,
              "conversion rule indexes must fit in the byte selector table");

typedef struct loom_amdgpu_scalar_conversion_rule_t {
  // Lowering strategy selected when the rule matches.
  loom_amdgpu_scalar_conversion_kind_t kind;
  // Descriptor emitted by strategies that perform a conversion packet.
  loom_amdgpu_descriptor_ref_t convert_descriptor_ref;
  // Descriptor refs that must be present before the rule can select.
  loom_amdgpu_descriptor_ref_t required_descriptor_refs[4];
} loom_amdgpu_scalar_conversion_rule_t;

#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0() \
  {                                            \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,         \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(ref0) \
  {                                                \
      (ref0),                                      \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,             \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(ref0, ref1) \
  {                                                      \
      (ref0),                                            \
      (ref1),                                            \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                   \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                   \
  }
#define LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(ref0, ref1, ref2) \
  {                                                            \
      (ref0),                                                  \
      (ref1),                                                  \
      (ref2),                                                  \
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE,                         \
  }

static const uint8_t kLoomAmdgpuScalarConversionRuleIndexes
    [LOOM_AMDGPU_SCALAR_CONVERSION_OP_COUNT_][LOOM_SCALAR_TYPE_COUNT_]
    [LOOM_SCALAR_TYPE_COUNT_] = {
        [LOOM_AMDGPU_SCALAR_CONVERSION_OP_TRUNCI] =
            {
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
};

static const loom_amdgpu_scalar_conversion_rule_t
    kLoomAmdgpuScalarConversionRules
        [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_COUNT_] = {
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I16_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_TRUNCI_I64_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
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
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I8_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_0()},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_EXTSI_I16_TO_I64] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64,
                 LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
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

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I8_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_UITOFP_I16_TO_F32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_2(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_U32)},

            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOSI_F32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_I32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I32] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_1(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I8] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
            [LOOM_AMDGPU_SCALAR_CONVERSION_RULE_FPTOUI_F32_TO_I16] =
                {LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_NARROW,
                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                 LOOM_AMDGPU_SCALAR_CONVERSION_REFS_3(
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_U32_F32,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
                     LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT)},
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
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTSI, EXTSI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(EXTUI, EXTUI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(SITOFP, SITOFP),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(UITOFP, UITOFP),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTOSI, FPTOSI),
        LOOM_AMDGPU_SCALAR_CONVERSION_OP_GROUP_ROW(FPTOUI, FPTOUI),
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
      source_type >= LOOM_SCALAR_TYPE_COUNT_ ||
      result_type >= LOOM_SCALAR_TYPE_COUNT_) {
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

static bool loom_amdgpu_select_scalar_conversion_plan_impl(
    const loom_module_t* module,
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

  loom_amdgpu_fp8_encode_plan_t fp8_encode = {0};
  if (source_op->kind == LOOM_OP_SCALAR_FPTRUNC &&
      loom_amdgpu_select_fp8_encode_plan(
          descriptor_set, source_type, result_type,
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
  if (rule == NULL || !loom_amdgpu_descriptor_set_has_all_refs(
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
  return true;
}

iree_status_t loom_amdgpu_select_scalar_conversion_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_conversion_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_select_scalar_conversion_plan_impl(
      loom_low_lower_context_module(context),
      loom_low_lower_context_descriptor_set(context), source_op, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_conversion(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  loom_amdgpu_scalar_conversion_plan_t plan = {0};
  if (!loom_amdgpu_select_scalar_conversion_plan_impl(
          loom_target_low_legality_module(context),
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

static iree_status_t loom_amdgpu_lookup_scalar_conversion_source_for_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t result,
    loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_low_result_type(context, source_op, result, &result_type));
  if (!loom_low_type_is_register(result_type) ||
      loom_low_register_type_unit_count(result_type) != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected a non-register i64 result");
    IREE_BUILTIN_UNREACHABLE();
  }
  const bool result_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (result_is_vgpr) {
    return loom_amdgpu_lookup_or_materialize_vgpr_i32(context, source_op,
                                                      source, out_low_source);
  }

  const bool result_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!result_is_sgpr) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected an unsupported i64 register class");
    IREE_BUILTIN_UNREACHABLE();
  }

  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, out_low_source));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type =
      loom_module_value_type(module, *out_low_source);
  const bool source_is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, source_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (!source_is_sgpr || loom_low_register_type_unit_count(source_type) != 1) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar conversion selected an SGPR i64 result from a "
        "non-SGPR source");
    IREE_BUILTIN_UNREACHABLE();
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

static iree_status_t loom_amdgpu_bind_sign_extended_i64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, loom_value_id_t low_source) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t lane_type = loom_module_value_type(module, low_source);
  const bool lane_is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  loom_value_id_t high_bits = LOOM_VALUE_ID_INVALID;
  if (lane_is_vgpr) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
        /*shift=*/31, low_source, lane_type, &high_bits));
  } else {
    const bool lane_is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, lane_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (!lane_is_sgpr) {
      IREE_ASSERT_UNREACHABLE(
          "AMDGPU scalar conversion sign-extended a non-register source");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32, low_source,
        /*immediate=*/31, lane_type, &high_bits));
  }
  return loom_amdgpu_bind_register64_lanes(context, source_op, source_result,
                                           low_source, high_bits);
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

static loom_amdgpu_fp8_decode_value_flags_t
loom_amdgpu_scalar_fp8_decode_value_flags(loom_low_lower_context_t* context,
                                          loom_value_id_t source) {
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table == NULL) {
    return LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  }
  return loom_amdgpu_fp8_decode_value_flags_from_facts(
      loom_value_fact_table_lookup(fact_table, source));
}

static iree_status_t loom_amdgpu_emit_scalar_fp8_to_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_scalar_type_t source_type = loom_amdgpu_scalar_type_or_none(
      loom_module_value_type(module, plan->source));
  IREE_ASSERT(source_type == LOOM_SCALAR_TYPE_F8E4M3 ||
              source_type == LOOM_SCALAR_TYPE_F8E5M2);

  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
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
  const loom_value_fact_numeric_format_flags_t source_format =
      loom_numeric_format_from_scalar_type(source_type);
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
      context, source_format, source_format, &decode_plan));

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
      context, source_op, decode_plan, low_byte,
      loom_amdgpu_scalar_fp8_decode_value_flags(context, plan->source),
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
  if (source_type == LOOM_SCALAR_TYPE_F32) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_f32(
        context, source_op, plan->source, &low_source));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
        context, source_op, plan->source, &low_source));
  }

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

iree_status_t loom_amdgpu_lower_scalar_conversion(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_conversion_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_TRUNCATE_LOW_32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, /*lane_offset=*/0, result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vgpr_bitfield(
          context, source_op, low_source, /*bit_offset=*/0,
          plan->result_bit_count, LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_NARROW_LOW_32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vgpr_bitfield(
          context, source_op, low_source, /*bit_offset=*/0,
          plan->result_bit_count, LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_SIGN_EXTEND_I64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_scalar_conversion_source_for_result(
              context, source_op, plan->source, plan->result, &low_source));
      return loom_amdgpu_bind_sign_extended_i64(context, source_op,
                                                plan->result, low_source);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_ZERO_EXTEND: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      if (plan->result_bit_count == 64) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_lookup_scalar_conversion_source_for_result(
                context, source_op, plan->source, plan->result, &low_source));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
            context, source_op, plan->source, &low_source));
      }
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_zero_extend(
          context, source_op, low_source, plan->source_bit_count, &low_result));
      if (plan->result_bit_count == 64) {
        return loom_amdgpu_bind_zero_extended_i64(context, source_op,
                                                  plan->result, low_result);
      }
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_UITOFP_NARROW_TO_F32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i32(
          context, source_op, plan->source, &low_source));
      loom_value_id_t zero_extended_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_zero_extend(
          context, source_op, low_source, plan->source_bit_count,
          &zero_extended_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
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
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_FPTOI_F32_TO_I32: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_f32(
          context, source_op, plan->source, &low_source));
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
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_f32(
          context, source_op, plan->source, &low_source));
      loom_type_t lane_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &lane_type));
      loom_value_id_t converted_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op, plan->convert_descriptor_ref, low_source,
          lane_type, &converted_source));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_vgpr_bitfield(
          context, source_op, converted_source, /*bit_offset=*/0,
          plan->result_bit_count, LOOM_AMDGPU_BITFIELD_EXTRACT_MODE_SIGN_EXTEND,
          lane_type, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_CONVERSION_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("invalid AMDGPU scalar conversion plan kind");
  return iree_ok_status();
}
