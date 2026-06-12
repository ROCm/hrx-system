// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/testing/low_descriptor_registry_verify.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/requirements.h"
#include "loom/target/low_descriptor_registry_core_test.h"

namespace loom {
namespace {

TEST(LowDescriptorRegistryTest, RegistryVerifiesSelectedTargetPackages) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  EXPECT_NE(registry.registry.descriptor_set_providers, nullptr);
  IREE_EXPECT_OK(loom_target_low_descriptor_registry_verify(
      &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, AppendToTablesBuildsBorrowedSubset) {
  loom_target_low_descriptor_registry_t source_registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&source_registry);

  loom_low_descriptor_set_provider_t descriptor_set_providers[8] = {};
  ASSERT_LE(source_registry.descriptor_set_provider_count,
            IREE_ARRAYSIZE(descriptor_set_providers));

  iree_host_size_t descriptor_set_provider_count = 0;
  IREE_ASSERT_OK(loom_target_low_descriptor_registry_append_to_tables(
      &source_registry, descriptor_set_providers,
      IREE_ARRAYSIZE(descriptor_set_providers),
      &descriptor_set_provider_count));

  EXPECT_EQ(descriptor_set_provider_count,
            source_registry.descriptor_set_provider_count);
  for (iree_host_size_t i = 0; i < descriptor_set_provider_count; ++i) {
    EXPECT_EQ(descriptor_set_providers[i],
              source_registry.descriptor_set_providers[i]);
  }

  loom_target_low_descriptor_registry_t joined_registry = {};
  loom_target_low_descriptor_registry_initialize_from_tables(
      &joined_registry, descriptor_set_providers,
      descriptor_set_provider_count);
  IREE_EXPECT_OK(loom_target_low_descriptor_registry_verify(
      &joined_registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, AppendToTablesRejectsOverflow) {
  loom_target_low_descriptor_registry_t source_registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&source_registry);
  ASSERT_GT(source_registry.descriptor_set_provider_count, 0u);

  loom_low_descriptor_set_provider_t descriptor_set_providers[8] = {};
  iree_host_size_t descriptor_set_provider_count =
      IREE_ARRAYSIZE(descriptor_set_providers);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_target_low_descriptor_registry_append_to_tables(
                            &source_registry, descriptor_set_providers,
                            IREE_ARRAYSIZE(descriptor_set_providers),
                            &descriptor_set_provider_count));
  EXPECT_EQ(descriptor_set_provider_count,
            IREE_ARRAYSIZE(descriptor_set_providers));
}

TEST(LowDescriptorRegistryTest, RegistrySatisfiesTargetLowFoundation) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  IREE_ASSERT_OK(loom_low_descriptor_registry_verify_requirements(
      &registry.registry,
      LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, MissingKeyReturnsNullDescriptorSet) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_target_core_test_low_descriptor_set_lookup(
          IREE_SV("target.missing"));
  EXPECT_EQ(descriptor_set, nullptr);
}

TEST(LowDescriptorRegistryTest, RegistryRejectsMismatchedDescriptorView) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);
  registry.registry.descriptor_set_provider_count = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_low_descriptor_registry_verify(
          &registry, LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION));
}

TEST(LowDescriptorRegistryTest, BundleSelectionFailsWhenLinkedSetIsMissing) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  const loom_target_config_t missing_config = {
      /*.name=*/IREE_SVL("missing"),
      /*.contract_set_key=*/IREE_SVL("target.missing"),
  };
  const loom_target_bundle_t missing_bundle = {
      /*.name=*/IREE_SVL("missing-bundle"),
      /*.snapshot=*/{},
      /*.export_plan=*/{},
      /*.config=*/&missing_config,
  };
  const loom_low_descriptor_set_t* descriptor_set = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_target_low_descriptor_set_select_for_bundle(
          &registry.registry, &missing_bundle, &descriptor_set));
  EXPECT_EQ(descriptor_set, nullptr);
}

}  // namespace
}  // namespace loom
