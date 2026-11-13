// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/control_operands.h"

#include "iree/testing/gtest.h"
#include "loom/target/arch/amdgpu/lower/types.h"

namespace {

loom_type_t Carrier(uint16_t register_class, uint32_t words) {
  return loom_low_register_type(0, register_class, words);
}

loom_value_facts_t UniformFacts() {
  auto facts = loom_value_facts_unknown();
  loom_value_facts_mark_subgroup_uniform(&facts);
  return facts;
}

TEST(AmdgpuControlOperandTest, NumericBankTransferPreservesEveryPayloadWord) {
  for (loom_scalar_type_t element = LOOM_SCALAR_TYPE_I8;
       element < LOOM_SCALAR_TYPE_COUNT_; ++element) {
    for (uint32_t count : {0u, 1u, 2u, 3u, 8u}) {
      SCOPED_TRACE(::testing::Message()
                   << "element=" << int(element) << " count=" << count);
      const auto source =
          count == 0 ? loom_type_scalar(element)
                     : loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                                           loom_dim_pack_static(count), 0);
      const uint32_t words =
          (loom_scalar_type_bitwidth(element) * (count ? count : 1) + 31) / 32;
      const auto scalar = Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, words);
      const auto vector = Carrier(LOOM_AMDGPU_REG_CLASS_ID_VGPR, words);
      auto varying = loom_value_facts_unknown();
      loom_value_facts_mark_lane_varying(&varying);
      for (auto facts : {loom_value_facts_unknown(), varying}) {
        EXPECT_TRUE(
            loom_amdgpu_control_operand_accepts(source, facts, scalar, vector));
        EXPECT_FALSE(
            loom_amdgpu_control_operand_accepts(source, facts, vector, scalar));
      }
      EXPECT_TRUE(loom_amdgpu_control_operand_accepts(source, UniformFacts(),
                                                      vector, scalar));
      for (auto register_class :
           {LOOM_AMDGPU_REG_CLASS_ID_SGPR, LOOM_AMDGPU_REG_CLASS_ID_VGPR}) {
        const auto wrong_width = Carrier(register_class, words + 1);
        EXPECT_FALSE(loom_amdgpu_control_operand_accepts(source, UniformFacts(),
                                                         scalar, wrong_width));
        EXPECT_FALSE(loom_amdgpu_control_operand_accepts(source, UniformFacts(),
                                                         vector, wrong_width));
        EXPECT_FALSE(loom_amdgpu_control_operand_accepts(source, UniformFacts(),
                                                         wrong_width, scalar));
      }
      EXPECT_FALSE(loom_amdgpu_control_operand_accepts(
          source, UniformFacts(), vector,
          Carrier(LOOM_AMDGPU_REG_CLASS_ID_SCC, 1)));
    }
  }
}

TEST(AmdgpuControlOperandTest, PredicateMaskCollapseRequiresAgreement) {
  const auto source = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
  const auto condition = Carrier(LOOM_AMDGPU_REG_CLASS_ID_SCC, 1);
  const auto truth = Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1);
  const auto mask = Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  const auto unknown = loom_value_facts_unknown();
  EXPECT_TRUE(
      loom_amdgpu_control_operand_accepts(source, unknown, condition, truth));
  EXPECT_TRUE(
      loom_amdgpu_control_operand_accepts(source, unknown, condition, mask));
  EXPECT_TRUE(
      loom_amdgpu_control_operand_accepts(source, unknown, truth, mask));
  EXPECT_FALSE(
      loom_amdgpu_control_operand_accepts(source, unknown, mask, truth));
  EXPECT_TRUE(
      loom_amdgpu_control_operand_accepts(source, UniformFacts(), mask, truth));
  EXPECT_TRUE(
      loom_amdgpu_control_operand_accepts(source, unknown, truth, condition));
  EXPECT_FALSE(
      loom_amdgpu_control_operand_accepts(source, unknown, mask, condition));
  EXPECT_TRUE(loom_amdgpu_control_operand_accepts(source, UniformFacts(), mask,
                                                  condition));
  EXPECT_FALSE(loom_amdgpu_control_operand_accepts(
      source, UniformFacts(), mask, Carrier(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 1)));
  for (uint32_t count : {1u, 3u, 8u}) {
    const auto predicates = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_dim_pack_static(count), 0);
    EXPECT_FALSE(loom_amdgpu_control_operand_accepts(
        predicates, UniformFacts(),
        Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, count * 2),
        Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, count)));
  }
}

TEST(AmdgpuControlOperandTest, AddressNarrowingRequiresRepresentableRange) {
  for (auto element : {LOOM_SCALAR_TYPE_INDEX, LOOM_SCALAR_TYPE_OFFSET}) {
    const auto source = loom_type_scalar(element);
    for (auto register_class :
         {LOOM_AMDGPU_REG_CLASS_ID_SGPR, LOOM_AMDGPU_REG_CLASS_ID_VGPR}) {
      const auto word = Carrier(register_class, 1);
      const auto pair = Carrier(register_class, 2);
      EXPECT_TRUE(loom_amdgpu_control_operand_accepts(
          source, loom_value_facts_unknown(), word, pair));
      EXPECT_FALSE(loom_amdgpu_control_operand_accepts(
          source, loom_value_facts_unknown(), pair, word));
      auto facts = loom_value_facts_unknown();
      facts.range_lo = 0;
      facts.range_hi = UINT32_MAX;
      EXPECT_TRUE(
          loom_amdgpu_control_operand_accepts(source, facts, pair, word));
      facts.range_hi = int64_t{UINT32_MAX} + 1;
      EXPECT_FALSE(
          loom_amdgpu_control_operand_accepts(source, facts, pair, word));
      facts.range_lo = INT32_MIN;
      facts.range_hi = INT32_MAX;
      EXPECT_EQ(loom_amdgpu_control_operand_accepts(source, facts, pair, word),
                element == LOOM_SCALAR_TYPE_INDEX);
    }
    auto facts = UniformFacts();
    facts.range_lo = 0;
    facts.range_hi = 31;
    EXPECT_TRUE(loom_amdgpu_control_operand_accepts(
        source, facts, Carrier(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2),
        Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1)));
    loom_value_facts_mark_lane_varying(&facts);
    EXPECT_FALSE(loom_amdgpu_control_operand_accepts(
        source, facts, Carrier(LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2),
        Carrier(LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1)));
  }
}

}  // namespace
