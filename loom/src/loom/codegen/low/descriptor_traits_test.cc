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
                                         uint8_t effect_count) {
  const loom_low_descriptor_layout_t layout = {.effect_count = effect_count};
  loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
      .effects = effects,
  };
  descriptor_set.effect_count = effect_count;
  loom_low_descriptor_t descriptor = {};
  return loom_low_descriptor_effective_traits(&descriptor_set, &descriptor,
                                              nullptr, nullptr);
}

TEST(LowDescriptorTraitsTest, EffectFreeDescriptorIsPure) {
  const loom_trait_flags_t traits = ProjectEffects(nullptr, 0);

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, RetainsIntrinsicAccessSpaceFamily) {
  struct Case {
    // Native descriptor memory domain.
    loom_low_memory_space_t native_space;
    // Expected source memory-space set.
    loom_memory_space_mask_t source_spaces;
  };
  const Case cases[] = {
      {LOOM_LOW_MEMORY_SPACE_NONE, 0},
      {LOOM_LOW_MEMORY_SPACE_GENERIC,
       1u << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN},
      {LOOM_LOW_MEMORY_SPACE_GLOBAL, 1u << LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL},
      {LOOM_LOW_MEMORY_SPACE_WORKGROUP,
       1u << LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP},
      {LOOM_LOW_MEMORY_SPACE_STACK, 1u << LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE},
      {LOOM_LOW_MEMORY_SPACE_WASM_MEMORY,
       1u << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN},
  };
  const loom_low_descriptor_layout_t layout = {.effect_count = 1};
  const loom_low_descriptor_t descriptor = {};
  for (const auto& test : cases) {
    loom_low_effect_t effect = {
        .kind = LOOM_LOW_EFFECT_KIND_WRITE,
        .memory_space = test.native_space,
    };
    const loom_low_descriptor_set_t set = {
        .descriptor_layouts = &layout,
        .descriptor_layout_count = 1,
        .effects = &effect,
        .effect_count = 1,
    };
    loom_op_t op = {};
    op.traits = loom_low_descriptor_effective_traits(
        &set, &descriptor, &op.intrinsic_read_spaces,
        &op.intrinsic_write_spaces);
    EXPECT_EQ(loom_op_read_memory_spaces(&op, op.traits), 0);
    EXPECT_EQ(loom_op_written_memory_spaces(&op, op.traits),
              test.source_spaces);
    effect.kind = LOOM_LOW_EFFECT_KIND_READ;
    op.traits = loom_low_descriptor_effective_traits(
        &set, &descriptor, &op.intrinsic_read_spaces,
        &op.intrinsic_write_spaces);
    EXPECT_EQ(loom_op_read_memory_spaces(&op, op.traits), test.source_spaces);
    EXPECT_EQ(loom_op_written_memory_spaces(&op, op.traits), 0);
  }
}

TEST(LowDescriptorTraitsTest, WriteBoundsDoNotHideFencesOrUnknownEffects) {
  loom_low_effect_t effects[] = {
      {
          .kind = LOOM_LOW_EFFECT_KIND_WRITE,
          .memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL,
      },
      {
          .kind = LOOM_LOW_EFFECT_KIND_WRITE,
          .memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP,
      },
  };
  const loom_low_descriptor_layout_t layout = {.effect_count = 2};
  const loom_low_descriptor_t descriptor = {};
  const loom_low_descriptor_set_t set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
      .effects = effects,
      .effect_count = 2,
  };
  loom_op_t op = {};
  op.traits = loom_low_descriptor_effective_traits(
      &set, &descriptor, &op.intrinsic_read_spaces, &op.intrinsic_write_spaces);
  EXPECT_EQ(loom_op_written_memory_spaces(&op, op.traits),
            (1u << LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) |
                (1u << LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP));
  for (auto kind :
       {LOOM_LOW_EFFECT_KIND_UNKNOWN, LOOM_LOW_EFFECT_KIND_CALL,
        LOOM_LOW_EFFECT_KIND_BARRIER, LOOM_LOW_EFFECT_KIND_COUNTER}) {
    effects[1].kind = kind;
    op.traits = loom_low_descriptor_effective_traits(
        &set, &descriptor, &op.intrinsic_read_spaces,
        &op.intrinsic_write_spaces);
    EXPECT_EQ(loom_op_written_memory_spaces(&op, op.traits),
              1u << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN);
  }
}

TEST(LowDescriptorTraitsTest, PreciseEffectsCompose) {
  loom_low_effect_t effects[4] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[0].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[1].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
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
  effects[1].kind = LOOM_LOW_EFFECT_KIND_UNKNOWN;

  const loom_trait_flags_t traits =
      ProjectEffects(effects, IREE_ARRAYSIZE(effects));

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, BarrierIsMemoryFence) {
  loom_low_effect_t effect = {
      .kind = LOOM_LOW_EFFECT_KIND_BARRIER,
  };

  const loom_trait_flags_t traits = ProjectEffects(&effect, 1);

  EXPECT_TRUE(loom_traits_order_memory(traits));
  EXPECT_TRUE(loom_traits_may_read(traits));
  EXPECT_TRUE(loom_traits_may_write(traits));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
  EXPECT_FALSE(loom_traits_may_access_memory(traits));
}

TEST(LowDescriptorTraitsTest, CounterOrdersWithoutAccessingMemory) {
  loom_low_effect_t effect = {
      .kind = LOOM_LOW_EFFECT_KIND_COUNTER,
  };
  const loom_trait_flags_t traits = ProjectEffects(&effect, 1);
  EXPECT_TRUE(loom_traits_order_memory(traits));
  EXPECT_TRUE(loom_traits_may_read(traits));
  EXPECT_TRUE(loom_traits_may_write(traits));
  EXPECT_FALSE(loom_traits_may_access_memory(traits));
}

TEST(LowDescriptorTraitsTest, ExternalResourcesRemainEffectful) {
  loom_low_effect_t effect = {
      .kind = LOOM_LOW_EFFECT_KIND_READ,
  };
  const loom_trait_flags_t read_traits = ProjectEffects(&effect, 1);
  EXPECT_TRUE(iree_all_bits_set(read_traits, LOOM_TRAIT_NON_DETERMINISTIC));
  EXPECT_FALSE(loom_traits_may_access_memory(read_traits));
  EXPECT_FALSE(iree_any_bit_set(read_traits, LOOM_TRAIT_PURE));

  effect.kind = LOOM_LOW_EFFECT_KIND_WRITE;
  const loom_trait_flags_t write_traits = ProjectEffects(&effect, 1);
  EXPECT_TRUE(iree_all_bits_set(write_traits, LOOM_TRAIT_OBSERVABLE_EFFECT));
  EXPECT_FALSE(loom_traits_may_access_memory(write_traits));
  EXPECT_FALSE(iree_any_bit_set(write_traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, BarrierPreservesPreciseMemoryEffects) {
  loom_low_effect_t effects[2] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[0].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_BARRIER;

  const loom_trait_flags_t traits =
      ProjectEffects(effects, IREE_ARRAYSIZE(effects));

  EXPECT_TRUE(iree_all_bits_set(
      traits, LOOM_TRAIT_READS_MEMORY | LOOM_TRAIT_MEMORY_FENCE));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS));
}

TEST(LowDescriptorTraitsTest, OpaqueEffectsSubsumePreciseMemoryTraits) {
  for (auto kind : {LOOM_LOW_EFFECT_KIND_CALL, LOOM_LOW_EFFECT_KIND_UNKNOWN}) {
    loom_low_effect_t effects[5] = {};
    effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
    effects[0].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
    effects[1].kind = kind;
    effects[2].kind = LOOM_LOW_EFFECT_KIND_WRITE;
    effects[2].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
    effects[3].kind = LOOM_LOW_EFFECT_KIND_CONTROL;
    effects[4].kind = LOOM_LOW_EFFECT_KIND_CONVERGENT;
    const loom_trait_flags_t traits =
        ProjectEffects(effects, IREE_ARRAYSIZE(effects));

    EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_UNKNOWN_EFFECTS |
                                              LOOM_TRAIT_TERMINATOR |
                                              LOOM_TRAIT_CONVERGENT));
    EXPECT_TRUE(loom_traits_may_read(traits));
    EXPECT_TRUE(loom_traits_may_write(traits));
    EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE |
                                              LOOM_TRAIT_READS_MEMORY |
                                              LOOM_TRAIT_WRITES_MEMORY));
  }
}

TEST(LowDescriptorTraitsTest, SideEffectingTerminatorFlagsCompose) {
  const loom_low_descriptor_layout_t layout = {};
  const loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
  };
  loom_low_descriptor_t descriptor = {
      .flags = LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING |
               LOOM_LOW_DESCRIPTOR_FLAG_TERMINATOR,
  };

  const loom_trait_flags_t traits = loom_low_descriptor_effective_traits(
      &descriptor_set, &descriptor, nullptr, nullptr);

  EXPECT_TRUE(iree_all_bits_set(
      traits, LOOM_TRAIT_TERMINATOR | LOOM_TRAIT_UNKNOWN_EFFECTS));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, StateAssignmentDoesNotInvalidateStorage) {
  const loom_low_descriptor_layout_t layout = {};
  const loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
  };
  for (uint16_t flags : {LOOM_LOW_DESCRIPTOR_FLAG_SIDE_EFFECTING,
                         LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE}) {
    const loom_low_descriptor_t descriptor = {
        .flags = static_cast<uint16_t>(
            flags | LOOM_LOW_DESCRIPTOR_FLAG_STATE_ASSIGNMENT),
    };
    const loom_trait_flags_t traits = loom_low_descriptor_effective_traits(
        &descriptor_set, &descriptor, nullptr, nullptr);
    EXPECT_TRUE(loom_traits_have_observable_effects(traits));
    EXPECT_FALSE(loom_traits_may_write(traits));
    EXPECT_FALSE(loom_traits_may_access_memory(traits));
    EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
  }
}

TEST(LowDescriptorTraitsTest, UniqueIdentityIsNotPureAndCannotBeCse) {
  const loom_low_descriptor_layout_t layout = {};
  const loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
  };
  loom_low_descriptor_t descriptor = {
      .flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE |
               LOOM_LOW_DESCRIPTOR_FLAG_UNIQUE_IDENTITY,
  };

  const loom_trait_flags_t traits = loom_low_descriptor_effective_traits(
      &descriptor_set, &descriptor, nullptr, nullptr);

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_UNIQUE_IDENTITY));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
  EXPECT_FALSE(loom_traits_has_side_effects(traits));
}

TEST(LowDescriptorTraitsTest, ImplicitStateResultOwnsLocalRegisterLifetime) {
  const loom_low_descriptor_layout_t layout = {};
  loom_low_operand_t operand = {
      .role = LOOM_LOW_OPERAND_ROLE_RESULT,
      .flags =
          LOOM_LOW_OPERAND_FLAG_IMPLICIT | LOOM_LOW_OPERAND_FLAG_STATE_WRITE,
  };
  loom_low_descriptor_set_t descriptor_set = {
      .descriptor_layouts = &layout,
      .descriptor_layout_count = 1,
      .operands = &operand,
      .operand_count = 1,
  };
  const uint16_t operand_indices[] = {0};
  descriptor_set.operand_indices = operand_indices;
  descriptor_set.operand_index_count = IREE_ARRAYSIZE(operand_indices);
  loom_low_descriptor_t descriptor = {
      .operand_count = 1,
      .result_count = 1,
  };

  const loom_trait_flags_t traits = loom_low_descriptor_effective_traits(
      &descriptor_set, &descriptor, nullptr, nullptr);

  EXPECT_TRUE(loom_traits_has_unique_identity(traits));
  EXPECT_TRUE(loom_traits_have_storage_relation(traits));
  EXPECT_FALSE(loom_traits_may_create_resources(traits));
  EXPECT_FALSE(loom_traits_may_read(traits));
  EXPECT_FALSE(loom_traits_may_write(traits));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_NON_DETERMINISTIC));
  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
}

TEST(LowDescriptorTraitsTest, RematerializationIsPerResult) {
  loom_low_operand_t operands[2] = {};
  operands[0].flags = LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE;
  loom_low_descriptor_set_t descriptor_set = {
      .operands = operands,
      .operand_count = IREE_ARRAYSIZE(operands),
  };
  const uint16_t operand_indices[] = {0, 1};
  descriptor_set.operand_indices = operand_indices;
  descriptor_set.operand_index_count = IREE_ARRAYSIZE(operand_indices);
  loom_low_descriptor_t descriptor = {
      .operand_count = IREE_ARRAYSIZE(operands),
      .result_count = IREE_ARRAYSIZE(operands),
  };

  EXPECT_TRUE(loom_low_descriptor_result_can_rematerialize(&descriptor_set,
                                                           &descriptor, 0));
  EXPECT_FALSE(loom_low_descriptor_result_can_rematerialize(&descriptor_set,
                                                            &descriptor, 1));
}

TEST(LowDescriptorTraitsTest, ProjectsRematerializationRecipeSource) {
  loom_low_operand_t operands[3] = {};
  operands[0].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  operands[0].flags = LOOM_LOW_OPERAND_FLAG_REMATERIALIZABLE;
  operands[0].rematerialization_source_operand_plus_one = 1;
  operands[1].role = LOOM_LOW_OPERAND_ROLE_RESULT;
  operands[2].role = LOOM_LOW_OPERAND_ROLE_OPERAND;
  operands[2].source_value_index = 0;
  const uint16_t operand_indices[] = {0, 1, 2};
  const loom_low_descriptor_set_t descriptor_set = {
      .operands = operands,
      .operand_count = IREE_ARRAYSIZE(operands),
      .operand_indices = operand_indices,
      .operand_index_count = IREE_ARRAYSIZE(operand_indices),
  };
  const loom_low_descriptor_t descriptor = {
      .operand_count = IREE_ARRAYSIZE(operands),
      .result_count = 2,
  };

  EXPECT_EQ(loom_low_descriptor_result_rematerialization_source_operand(
                &descriptor_set, &descriptor, 0),
            0u);
  EXPECT_EQ(loom_low_descriptor_result_rematerialization_source_operand(
                &descriptor_set, &descriptor, 1),
            LOOM_LOW_ID_NONE);
}

}  // namespace
}  // namespace loom
