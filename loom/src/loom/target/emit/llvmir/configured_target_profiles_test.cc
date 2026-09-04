// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/configured_target_profiles.h"

#include "iree/testing/gtest.h"

#ifndef LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
#define LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES 0
#endif  // LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
#ifndef LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES
#define LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES 0
#endif  // LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES

namespace loom {
namespace {

TEST(ConfiguredTargetProfilesTest, ContainsConfiguredProvidersExactlyOnce) {
  const loom_llvmir_target_profile_registry_t* registry =
      &loom_llvmir_configured_target_profile_registry;
  const iree_host_size_t expected_provider_count =
      LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES +
      LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES;
  ASSERT_EQ(registry->provider_count, expected_provider_count);
  EXPECT_EQ(registry->providers == nullptr, expected_provider_count == 0);

  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    const loom_llvmir_target_profile_provider_t* provider =
        registry->providers[i];
    ASSERT_NE(provider, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(provider->name));
    for (iree_host_size_t j = 0; j < i; ++j) {
      EXPECT_FALSE(
          iree_string_view_equal(provider->name, registry->providers[j]->name));
    }
  }
}

}  // namespace
}  // namespace loom
