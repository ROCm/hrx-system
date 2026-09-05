// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/configured.h"

#include "iree/testing/gtest.h"

#ifndef LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#define LOOM_CONFIG_EXECUTION_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#ifndef LOOM_CONFIG_EXECUTION_HAVE_SPIRV
#define LOOM_CONFIG_EXECUTION_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV

namespace loom {
namespace {

TEST(ConfiguredExecutionTest, ReturnsMatchingProviderTables) {
  const loom_tooling_execution_providers_t* providers =
      loom_tooling_configured_execution_providers();
  ASSERT_NE(providers, nullptr);
  EXPECT_EQ(providers, loom_tooling_configured_execution_providers());

  const iree_host_size_t expected_provider_count =
      LOOM_CONFIG_EXECUTION_HAVE_AMDGPU + LOOM_CONFIG_EXECUTION_HAVE_SPIRV;
  const loom_run_execution_provider_set_t* execution_providers =
      &providers->execution_provider_set;
  const loom_device_provider_registry_t* device_providers =
      &providers->device_provider_registry;
  ASSERT_EQ(execution_providers->provider_count, expected_provider_count);
  ASSERT_EQ(device_providers->provider_count, expected_provider_count);
  EXPECT_EQ(execution_providers->providers == nullptr,
            expected_provider_count == 0);
  EXPECT_EQ(device_providers->providers == nullptr,
            expected_provider_count == 0);

  for (iree_host_size_t i = 0; i < expected_provider_count; ++i) {
    const loom_run_execution_provider_t* execution_provider =
        execution_providers->providers[i];
    const loom_device_provider_t* device_provider =
        device_providers->providers[i];
    ASSERT_NE(execution_provider, nullptr);
    ASSERT_NE(device_provider, nullptr);
    ASSERT_NE(execution_provider->target_provider, nullptr);
    ASSERT_NE(device_provider->artifact_provider, nullptr);
    EXPECT_FALSE(iree_string_view_is_empty(execution_provider->name));
    EXPECT_FALSE(iree_string_view_is_empty(device_provider->driver_name));
    bool found_matching_backend = false;
    for (iree_host_size_t j = 0;
         j < execution_provider->execution_backend_count; ++j) {
      const loom_run_execution_backend_t* backend =
          execution_provider->execution_backends[j];
      ASSERT_NE(backend, nullptr);
      found_matching_backend |= iree_string_view_equal(
          backend->name, device_provider->artifact_provider->name);
    }
    EXPECT_TRUE(found_matching_backend);
  }
}

}  // namespace
}  // namespace loom
