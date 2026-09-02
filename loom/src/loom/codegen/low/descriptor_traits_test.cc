// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_traits.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static loom_trait_flags_t ProjectEffects(const loom_low_effect_t* effects,
                                         uint16_t effect_count) {
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.effects = effects;
  descriptor_set.effect_count = effect_count;
  loom_low_descriptor_t descriptor = {};
  descriptor.effect_count = effect_count;
  return loom_low_descriptor_effective_traits(&descriptor_set, &descriptor);
}

TEST(LowDescriptorTraitsTest, EffectFreeDescriptorIsPure) {
  const loom_trait_flags_t traits = ProjectEffects(nullptr, 0);

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, PreciseEffectsCompose) {
  loom_low_effect_t effects[4] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[2].kind = LOOM_LOW_EFFECT_KIND_CONTROL;
  effects[3].kind = LOOM_LOW_EFFECT_KIND_CONVERGENT;

  const loom_trait_flags_t traits =
      ProjectEffects(effects, IREE_ARRAYSIZE(effects));

  EXPECT_TRUE(iree_all_bits_set(
      traits, LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_WRITES_MEMORY |
                  LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_CONVERGENT));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, OpaqueEffectsRemainUnknown) {
  loom_low_effect_t effects[2] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_CALL;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_COUNTER;

  const loom_trait_flags_t traits =
      ProjectEffects(effects, IREE_ARRAYSIZE(effects));

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, BarrierIsMemoryFence) {
  loom_low_effect_t effect = {};
  effect.kind = LOOM_LOW_EFFECT_KIND_BARRIER;

  const loom_trait_flags_t traits = ProjectEffects(&effect, 1);

  EXPECT_TRUE(loom_traits_order_memory(traits));
  EXPECT_TRUE(loom_traits_may_read(traits));
  EXPECT_TRUE(loom_traits_may_write(traits));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, BarrierPreservesPreciseMemoryEffects) {
  loom_low_effect_t effects[2] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_BARRIER;

  const loom_trait_flags_t traits =
      ProjectEffects(effects, IREE_ARRAYSIZE(effects));

  EXPECT_TRUE(iree_all_bits_set(
      traits, LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_MEMORY_FENCE));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
}

TEST(LowDescriptorTraitsTest, SideEffectingTerminatorFlagsCompose) {
  const loom_low_descriptor_set_t descriptor_set = {};
  loom_low_descriptor_t descriptor = {};
  descriptor.flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
                     LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR;

  const loom_trait_flags_t traits =
      loom_low_descriptor_effective_traits(&descriptor_set, &descriptor);

  EXPECT_TRUE(iree_all_bits_set(
      traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, ImplicitStateResultIsNonDeterministic) {
  loom_low_operand_t operand = {};
  operand.role = LOOM_LOW_OPERAND_ROLE_RESULT;
  operand.flags =
      LOOM_LOW_OPERAND_FLAG_IMPLICIT | LOOM_LOW_OPERAND_FLAG_STATE_WRITE;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.operands = &operand;
  descriptor_set.operand_count = 1;
  loom_low_descriptor_t descriptor = {};
  descriptor.operand_count = 1;
  descriptor.result_count = 1;

  const loom_trait_flags_t traits =
      loom_low_descriptor_effective_traits(&descriptor_set, &descriptor);

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_NON_DETERMINISTIC));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, RematerializationIsPerResult) {
  loom_low_operand_t operands[2] = {};
  operands[0].flags = LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE;
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.operands = operands;
  descriptor_set.operand_count = IREE_ARRAYSIZE(operands);
  loom_low_descriptor_t descriptor = {};
  descriptor.operand_count = IREE_ARRAYSIZE(operands);
  descriptor.result_count = IREE_ARRAYSIZE(operands);

  EXPECT_TRUE(loom_low_descriptor_result_can_rematerialize(&descriptor_set,
                                                           &descriptor, 0));
  EXPECT_FALSE(loom_low_descriptor_result_can_rematerialize(&descriptor_set,
                                                            &descriptor, 1));
  EXPECT_FALSE(loom_low_descriptor_result_can_rematerialize(&descriptor_set,
                                                            &descriptor, 2));
}

}  // namespace
}  // namespace loom
