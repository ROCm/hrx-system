// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/test/lower/registry.h"

namespace loom {
namespace {

static loom_low_lower_policy_registry_t MakeTestPolicyRegistry() {
  loom_low_lower_policy_registry_t registry = {};
  loom_test_low_lower_policy_registry_initialize(&registry);
  return registry;
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyByContractIdentity) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();

  const loom_low_lower_policy_t* policy = loom_low_lower_policy_registry_lookup(
      &registry, UINT64_C(0x235c515ddcc8fd92));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, LooksUpPolicyForTargetBundle) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();
  loom_target_config_t config = {
      .contract_set =
          {
              .stable_id = UINT64_C(0x235c515ddcc8fd92),
              .spelling = IREE_SV("test.low.core"),
          },
  };
  loom_target_bundle_t bundle = {
      .name = IREE_SV("test-low"),
      .config = &config,
  };

  const loom_low_lower_policy_t* policy =
      loom_low_lower_policy_registry_lookup_for_bundle(&registry, &bundle);
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

TEST(LowLowerPolicyRegistryTest, RejectsMissingContractIdentity) {
  loom_low_lower_policy_registry_t registry = MakeTestPolicyRegistry();

  EXPECT_EQ(loom_low_lower_policy_registry_lookup(&registry,
                                                  UINT64_C(0x54c0f4e693d256cf)),
            nullptr);
}

TEST(LowLowerPolicyRegistryTest, ReturnsFirstDuplicateContractIdentity) {
  const loom_low_lower_policy_registry_entry_t entries[] = {
      {
          .contract_set_stable_id = UINT64_C(0x235c515ddcc8fd92),
          .policy = loom_test_low_lower_policy(),
      },
      {
          .contract_set_stable_id = UINT64_C(0x235c515ddcc8fd92),
          .policy = loom_test_low_lower_policy(),
      },
  };
  loom_low_lower_policy_registry_t registry = {};
  loom_low_lower_policy_registry_initialize_from_entries(
      entries, IREE_ARRAYSIZE(entries), &registry);

  const loom_low_lower_policy_t* policy = loom_low_lower_policy_registry_lookup(
      &registry, UINT64_C(0x235c515ddcc8fd92));
  EXPECT_EQ(policy, loom_test_low_lower_policy());
}

}  // namespace
}  // namespace loom
