// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptors.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowDescriptorsTest, SummarizesOnlyMemoryAttachedEffects) {
  loom_low_effect_t effects[6] = {};
  effects[0].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[1].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[1].width_bits = 64;
  effects[2].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[2].memory_space = LOOM_LOW_MEMORY_SPACE_GENERIC;
  effects[3].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[3].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  effects[3].width_bits = 7;
  effects[4].kind = LOOM_LOW_EFFECT_KIND_READ;
  effects[4].memory_space = LOOM_LOW_MEMORY_SPACE_GLOBAL;
  effects[4].width_bits = 32;
  effects[5].kind = LOOM_LOW_EFFECT_KIND_WRITE;
  effects[5].memory_space = LOOM_LOW_MEMORY_SPACE_WORKGROUP;
  effects[5].width_bits = 64;
  loom_low_descriptor_t descriptor = {};
  descriptor.effect_count = IREE_ARRAYSIZE(effects);
  loom_low_descriptor_set_t descriptor_set = {};
  descriptor_set.effects = effects;
  descriptor_set.effect_count = IREE_ARRAYSIZE(effects);

  const loom_low_descriptor_memory_effect_summary_t summary =
      loom_low_descriptor_memory_effect_summary(&descriptor_set, &descriptor);

  EXPECT_EQ(summary.read_unknown_width_count, 1u);
  EXPECT_EQ(summary.write_unknown_width_count, 1u);
  EXPECT_EQ(summary.read_byte_count, 4u);
  EXPECT_EQ(summary.write_byte_count, 8u);
}

}  // namespace
}  // namespace loom
