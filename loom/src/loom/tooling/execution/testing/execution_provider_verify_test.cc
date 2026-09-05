// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/testing/execution_provider_verify.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/tooling/execution/execution_provider.h"

namespace loom {
namespace {

iree_status_t FakeSelectTarget(const loom_device_provider_t* provider,
                               const loom_run_hal_runtime_t* runtime,
                               loom_device_target_t* out_target) {
  (void)provider;
  (void)runtime;
  (void)out_target;
  return iree_ok_status();
}

iree_status_t FakeSelectCompatibleTarget(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement,
    loom_device_target_t* out_target) {
  (void)provider;
  (void)runtime;
  (void)target_requirement;
  (void)out_target;
  return iree_ok_status();
}

iree_status_t FakeProjectTargetFacts(const loom_device_provider_t* provider,
                                     const loom_run_hal_runtime_t* runtime,
                                     const loom_device_target_t* target,
                                     iree_arena_allocator_t* arena,
                                     loom_target_facts_t* out_facts) {
  (void)provider;
  (void)runtime;
  (void)target;
  (void)arena;
  (void)out_facts;
  return iree_ok_status();
}

const loom_artifact_provider_t kFakeArtifactProvider = {
    /*.name=*/IREE_SVL("fake-artifact"),
};

const loom_device_provider_t kFakeDeviceProvider = {
    /*.artifact_provider=*/&kFakeArtifactProvider,
    /*.driver_name=*/IREE_SVL("fake"),
    /*.select_target=*/FakeSelectTarget,
    /*.select_compatible_target=*/FakeSelectCompatibleTarget,
    /*.project_target_facts=*/FakeProjectTargetFacts,
};

const loom_device_provider_t kDuplicateFakeDeviceProvider = {
    /*.artifact_provider=*/&kFakeArtifactProvider,
    /*.driver_name=*/IREE_SVL("fake"),
    /*.select_target=*/FakeSelectTarget,
    /*.select_compatible_target=*/FakeSelectCompatibleTarget,
    /*.project_target_facts=*/FakeProjectTargetFacts,
};

const loom_target_provider_t kCoreTestTargetProvider = {
    /*.profile_type=*/{},
    /*.materialize_definition=*/{},
    /*.register_context=*/{},
    /*.initialize_low_descriptor_registry=*/
    loom_target_core_test_low_descriptor_registry_initialize,
};

const loom_run_execution_provider_t kCoreTestProvider = {
    /*.name=*/IREE_SVL("core-test"),
    /*.target_provider=*/&kCoreTestTargetProvider,
    /*.device_provider=*/&kFakeDeviceProvider,
};

const loom_run_execution_provider_t kDuplicateCoreTestProvider = {
    /*.name=*/IREE_SVL("core-test"),
    /*.target_provider=*/&kCoreTestTargetProvider,
};

const loom_run_execution_provider_t kDuplicateExecutionProvider = {
    /*.name=*/IREE_SVL("duplicate-execution"),
    /*.target_provider=*/&kCoreTestTargetProvider,
    /*.device_provider=*/&kDuplicateFakeDeviceProvider,
};

TEST(ExecutionProviderTest, ComposesDescriptorAndDeviceProviderRegistries) {
  const loom_run_execution_provider_t* providers[] = {
      &kCoreTestProvider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  IREE_ASSERT_OK(loom_run_execution_provider_set_verify(&provider_set));

  loom_run_execution_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_run_execution_environment_initialize(&provider_set, &environment));

  const loom_device_provider_registry_t* device_provider_registry =
      loom_run_execution_environment_device_provider_registry(&environment);
  ASSERT_NE(device_provider_registry, nullptr);
  ASSERT_EQ(device_provider_registry->provider_count, 1u);
  EXPECT_EQ(device_provider_registry->providers[0], &kFakeDeviceProvider);

  loom_target_low_descriptor_registry_t low_registry = {};
  const loom_run_initialize_low_descriptor_registry_callback_t callback =
      loom_run_execution_environment_low_descriptor_registry_callback(
          &environment);
  IREE_ASSERT_OK(callback.fn(callback.user_data, &low_registry));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(&low_registry.registry,
                                          IREE_SV("test.low.core"));
  EXPECT_NE(descriptor_set, nullptr);

  loom_run_execution_environment_deinitialize(&environment);
}

TEST(ExecutionProviderTest, RejectsDuplicateProviderNames) {
  const loom_run_execution_provider_t* providers[] = {
      &kCoreTestProvider,
      &kDuplicateCoreTestProvider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_execution_provider_set_verify(&provider_set));
}

TEST(ExecutionProviderTest, RejectsDuplicateDeviceDriverNames) {
  const loom_run_execution_provider_t* providers[] = {
      &kCoreTestProvider,
      &kDuplicateExecutionProvider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_execution_provider_set_verify(&provider_set));
}

}  // namespace
}  // namespace loom
