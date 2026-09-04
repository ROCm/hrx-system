// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured.h"

#include "iree/testing/gtest.h"

#ifndef LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS

namespace loom {
namespace {

TEST(ConfiguredCompileTest, ReturnsStableCompleteEnvironment) {
  const loom_tooling_compile_environment_t* environment =
      loom_tooling_configured_compile_environment();
  ASSERT_NE(environment, nullptr);
  EXPECT_EQ(environment, loom_tooling_configured_compile_environment());
  ASSERT_NE(environment->target_environment, nullptr);
  ASSERT_NE(environment->target_environment->provider_set, nullptr);
  EXPECT_GT(environment->target_environment->provider_set->provider_count, 0u);

  const loom_artifact_provider_registry_t* artifact_registry =
      environment->artifact_provider_registry;
  ASSERT_NE(artifact_registry, nullptr);
  const iree_host_size_t expected_artifact_provider_count =
      LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS +
      LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS;
  ASSERT_EQ(artifact_registry->provider_count,
            expected_artifact_provider_count);
  EXPECT_EQ(artifact_registry->providers == nullptr,
            expected_artifact_provider_count == 0);

  for (iree_host_size_t i = 0; i < artifact_registry->provider_count; ++i) {
    const loom_artifact_provider_t* provider = artifact_registry->providers[i];
    ASSERT_NE(provider, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(provider->name));
    EXPECT_FALSE(iree_string_view_is_empty(provider->public_artifact_format));
    ASSERT_NE(provider->target_profile_type, nullptr);
    for (iree_host_size_t j = 0; j < i; ++j) {
      EXPECT_FALSE(iree_string_view_equal(
          provider->public_artifact_format,
          artifact_registry->providers[j]->public_artifact_format));
    }
  }
}

}  // namespace
}  // namespace loom
