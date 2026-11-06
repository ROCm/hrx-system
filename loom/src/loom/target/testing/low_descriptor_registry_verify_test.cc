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

  ASSERT_EQ(
      loom_low_descriptor_registry_descriptor_set_count(&registry.registry),
      1u);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_descriptor_set_at(&registry.registry, 0);
  ASSERT_NE(descriptor_set, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      loom_low_descriptor_set_string(descriptor_set,
                                     descriptor_set->key_string_ref),
      IREE_SV("test.low.core")));
  EXPECT_EQ(loom_low_descriptor_registry_lookup(&registry.registry,
                                                IREE_SV("test.low.core")),
            descriptor_set);
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
      descriptor_set_providers, descriptor_set_provider_count,
      &joined_registry);
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

TEST(LowDescriptorRegistryTest, BundleSelectionBindsExactScheduleModel) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  const loom_target_config_t config = {
      .name = IREE_SVL("test.low.core"),
      .schedule_model_stable_id =
          loom_low_descriptor_stable_id_from_key(IREE_SV("test.low.core")),
      .contract_set =
          {
              .stable_id = loom_low_descriptor_stable_id_from_key(
                  IREE_SV("test.low.core")),
              .spelling = IREE_SVL("test.low.core"),
          },
  };
  const loom_target_bundle_t bundle = {
      .name = IREE_SVL("test-low"),
      .snapshot = {},
      .export_plan = {},
      .config = &config,
  };
  loom_target_low_source_binding_t binding = {};
  IREE_ASSERT_OK(loom_target_low_source_binding_select(&registry.registry,
                                                       &bundle, &binding));
  ASSERT_NE(binding.descriptor_set, nullptr);
  EXPECT_EQ(binding.schedule_model,
            loom_low_descriptor_set_lookup_schedule_model_by_id(
                binding.descriptor_set, config.schedule_model_stable_id));
}

TEST(LowDescriptorRegistryTest, BundleSelectionFailsWhenLinkedSetIsMissing) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  const loom_target_config_t missing_config = {
      .name = IREE_SVL("missing"),
      .schedule_model_stable_id =
          loom_low_descriptor_stable_id_from_key(IREE_SV("test.low.core")),
      .contract_set =
          {
              .stable_id = loom_low_descriptor_stable_id_from_key(
                  IREE_SV("target.missing")),
              .spelling = IREE_SVL("target.missing"),
          },
  };
  const loom_target_bundle_t missing_bundle = {
      .name = IREE_SVL("missing-bundle"),
      .snapshot = {},
      .export_plan = {},
      .config = &missing_config,
  };
  loom_target_low_source_binding_t binding = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_target_low_source_binding_select(
                            &registry.registry, &missing_bundle, &binding));
  EXPECT_EQ(binding.descriptor_set, nullptr);
  EXPECT_EQ(binding.schedule_model, nullptr);
}

TEST(LowDescriptorRegistryTest,
     BundleSelectionFailsWithoutAnExactScheduleModel) {
  loom_target_low_descriptor_registry_t registry = {};
  loom_target_core_test_low_descriptor_registry_initialize(&registry);

  for (uint64_t schedule_model_stable_id :
       {UINT64_C(0), loom_low_descriptor_stable_id_from_key(
                         IREE_SV("test.model.missing"))}) {
    const loom_target_config_t config = {
        .name = IREE_SVL("missing-model"),
        .schedule_model_stable_id = schedule_model_stable_id,
        .contract_set =
            {
                .stable_id = loom_low_descriptor_stable_id_from_key(
                    IREE_SV("test.low.core")),
                .spelling = IREE_SVL("test.low.core"),
            },
    };
    const loom_target_bundle_t bundle = {
        .name = IREE_SVL("missing-model-bundle"),
        .snapshot = {},
        .export_plan = {},
        .config = &config,
    };
    loom_target_low_source_binding_t binding = {};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                          loom_target_low_source_binding_select(
                              &registry.registry, &bundle, &binding));
    EXPECT_EQ(binding.descriptor_set, nullptr);
    EXPECT_EQ(binding.schedule_model, nullptr);
  }
}

}  // namespace
}  // namespace loom
