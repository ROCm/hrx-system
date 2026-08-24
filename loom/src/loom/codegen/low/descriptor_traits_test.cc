// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/descriptor_traits.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

TEST(LowDescriptorTraitsTest, OrdinaryEffectFreeDescriptorIsPure) {
  const loom_low_descriptor_set_t descriptor_set = {};
  const loom_low_descriptor_t descriptor = {};

  const loom_trait_flags_t traits =
      loom_low_descriptor_effective_traits(&descriptor_set, &descriptor);

  EXPECT_TRUE(iree_all_bits_set(traits, LOOM_TRAIT_PURE));
  EXPECT_FALSE(loom_traits_has_unique_identity(traits));
}

TEST(LowDescriptorTraitsTest, UniqueIdentityDescriptorIsNotPure) {
  const loom_low_descriptor_set_t descriptor_set = {};
  loom_low_descriptor_t descriptor = {};
  descriptor.flags = LOOM_LOW_DESCRIPTOR_FLAG_DEAD_REMOVABLE |
                     LOOM_LOW_DESCRIPTOR_FLAG_UNIQUE_IDENTITY;

  const loom_trait_flags_t traits =
      loom_low_descriptor_effective_traits(&descriptor_set, &descriptor);

  EXPECT_FALSE(iree_any_bit_set(traits, LOOM_TRAIT_PURE));
  EXPECT_TRUE(loom_traits_has_unique_identity(traits));
}

}  // namespace
}  // namespace loom
