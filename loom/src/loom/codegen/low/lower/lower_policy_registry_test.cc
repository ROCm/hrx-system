// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/test/lower.h"

namespace loom {
namespace {

static loom_low_lower_policy_registry_t MakeTestPolicyRegistry() {
  loom_low_lower_policy_registry_t registry = {};
  loom_test_low_lower_policy_registry_initialize(&registry);
  return registry;
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyByContractKey) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();

  const loom_low_lower_policy_t* policy = loom_low_lower_policy_registry_lookup(
      &registry, IREE_SV("test.low.core"));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyForTargetBundle) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();
  loom_target_config_t config = {};
  config.contract_set_key = IREE_SV("test.low.core");
  loom_target_bundle_t bundle = {};
  bundle.name = IREE_SV("test-low");
  bundle.config = &config;

  const loom_low_lower_policy_t* policy =
      loom_low_lower_policy_registry_lookup_for_bundle(&registry, &bundle);
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, RejectsMissingContractKey) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();

  EXPECT_EQ(loom_low_lower_policy_registry_lookup(&registry,
                                                  IREE_SV("missing-target")),
            nullptr);
}

TEST(LowLowerPolicyRegistryTest, ReturnsFirstDuplicateContractKey) {
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          /*.contract_set_key=*/IREE_SVL("test.low.core"),
          /*.policy=*/loom_test_low_lower_policy(),
      },
      {
          /*.contract_set_key=*/IREE_SVL("test.low.core"),
          /*.policy=*/loom_test_low_lower_policy(),
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, entries, IREE_ARRAYSIZE(entries));

  const loom_low_lower_policy_t* policy = loom_low_lower_policy_registry_lookup(
      &registry, IREE_SV("test.low.core"));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

}  // namespace
}  // namespace loom
