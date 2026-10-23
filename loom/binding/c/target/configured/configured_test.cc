// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/configured.h"

#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/binding/c/src/target.h"
#include "loom/pass/testing/registry_verify.h"
#include "loom/target/configured/compiler_provider_set.h"

namespace {

struct TargetEnvironmentDeleter {
  void operator()(loomc_target_environment_t* environment) const {
    loomc_target_environment_release(environment);
  }
};

TEST(ConfiguredTargetTest, BorrowsValidConfiguredCompilerProviders) {
  loomc_target_environment_t* raw_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_configured(
      loomc_allocator_system(), &raw_environment);
  ASSERT_TRUE(loomc_status_is_ok(status));
  std::unique_ptr<loomc_target_environment_t, TargetEnvironmentDeleter>
      environment(raw_environment);

  const loom_target_environment_t* internal_environment =
      loomc_target_environment_loom_target_environment(environment.get());
  ASSERT_NE(internal_environment, nullptr);
  EXPECT_EQ(internal_environment->provider_set,
            loom_configured_compiler_provider_set());
  const auto* provider_set = internal_environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const auto* registry = provider_set->providers[i]->pass_registry;
    if (registry != nullptr) {
      IREE_EXPECT_OK(loom_pass_registry_verify(registry));
    }
  }
}

}  // namespace
