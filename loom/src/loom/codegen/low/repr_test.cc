// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/repr.h"

#include "iree/testing/gtest.h"
#include "loom/target/test/descriptors.h"

namespace loom {
namespace {

TEST(LowReprTest, StableKeysRoundTripContractLocalOrdinals) {
  static const loom_low_descriptor_set_provider_t kDescriptorSetProviders[] = {
      loom_test_low_core_descriptor_set,
  };
  loom_low_descriptor_registry_t registry = {};
  registry.descriptor_set_providers = kDescriptorSetProviders;
  registry.descriptor_set_provider_count =
      IREE_ARRAYSIZE(kDescriptorSetProviders);
  loom_low_repr_environment_t environment = {};
  loom_low_repr_environment_initialize(&registry, &environment);

  const loom_low_repr_descriptor_set_t* descriptor_set =
      loom_low_repr_lookup_descriptor_set(&environment,
                                          IREE_SV("test.low.core"));
  ASSERT_NE(descriptor_set, nullptr);
  EXPECT_EQ(loom_low_repr_lookup_descriptor_set(&environment,
                                                IREE_SV("test.low.missing")),
            nullptr);

  loom_low_repr_descriptor_value_t value = {};
  ASSERT_TRUE(loom_low_repr_resolve_descriptor(
      &environment, descriptor_set, IREE_SV("test.add.i32"), &value));
  EXPECT_TRUE(iree_any_bit_set(value.effective_traits, LOOM_TRAIT_PURE));
  EXPECT_TRUE(iree_string_view_equal(
      loom_low_repr_descriptor_key(&environment, descriptor_set, value.ordinal),
      IREE_SV("test.add.i32")));

  EXPECT_FALSE(loom_low_repr_resolve_descriptor(
      &environment, descriptor_set, IREE_SV("test.missing"), &value));
  EXPECT_TRUE(iree_string_view_is_empty(
      loom_low_repr_descriptor_key(&environment, descriptor_set, UINT32_MAX)));
}

}  // namespace
}  // namespace loom
