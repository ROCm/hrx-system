// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/device_provider.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

const loom_artifact_provider_t kFakeArtifactProvider = {
    /*.name=*/IREE_SVL("fake-artifact"),
};

const loom_artifact_provider_t kOtherArtifactProvider = {
    /*.name=*/IREE_SVL("other-artifact"),
};

const loom_device_provider_t kFakeDeviceProvider = {
    /*.artifact_provider=*/&kFakeArtifactProvider,
    /*.driver_name=*/IREE_SVL("fake"),
};

const loom_device_provider_t kOtherDeviceProvider = {
    /*.artifact_provider=*/&kOtherArtifactProvider,
    /*.driver_name=*/IREE_SVL("other"),
};

TEST(DeviceProviderRegistryTest, CanonicalizesEmptyRegistry) {
  const loom_device_provider_t* providers[] = {
      &kFakeDeviceProvider,
  };
  loom_device_provider_registry_t registry = {};
  loom_device_provider_registry_initialize_from_entries(
      providers, /*provider_count=*/0, &registry);

  EXPECT_EQ(registry.providers, nullptr);
  EXPECT_EQ(registry.provider_count, 0u);
}

TEST(DeviceProviderRegistryTest, SelectsDeviceUriDriverWithoutAllocation) {
  const loom_device_provider_t* providers[] = {
      &kFakeDeviceProvider,
      &kOtherDeviceProvider,
  };
  loom_device_provider_registry_t registry = {};
  loom_device_provider_registry_initialize_from_entries(
      providers, IREE_ARRAYSIZE(providers), &registry);
  const iree_string_view_t device_uris[] = {
      IREE_SV("other://adapter-0"),
  };

  const loom_device_provider_t* selected_provider = nullptr;
  IREE_ASSERT_OK(loom_device_provider_registry_select(
      &registry,
      (iree_string_view_list_t){
          /*.count=*/IREE_ARRAYSIZE(device_uris),
          /*.values=*/device_uris,
      },
      iree_allocator_null(), &selected_provider));

  EXPECT_EQ(selected_provider, &kOtherDeviceProvider);
}

TEST(DeviceProviderRegistryTest, RequiresExactlyOneDeviceUri) {
  const loom_device_provider_t* providers[] = {
      &kFakeDeviceProvider,
  };
  loom_device_provider_registry_t registry = {};
  loom_device_provider_registry_initialize_from_entries(
      providers, IREE_ARRAYSIZE(providers), &registry);

  const loom_device_provider_t* selected_provider = &kFakeDeviceProvider;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_device_provider_registry_select(
                            &registry, (iree_string_view_list_t){},
                            iree_allocator_system(), &selected_provider));
  EXPECT_EQ(selected_provider, nullptr);

  const iree_string_view_t device_uris[] = {
      IREE_SV("fake://adapter-0"),
      IREE_SV("other://adapter-1"),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_device_provider_registry_select(
                            &registry,
                            (iree_string_view_list_t){
                                /*.count=*/IREE_ARRAYSIZE(device_uris),
                                /*.values=*/device_uris,
                            },
                            iree_allocator_system(), &selected_provider));
  EXPECT_EQ(selected_provider, nullptr);
}

TEST(DeviceProviderRegistryTest, RejectsUnavailableDeviceUriDriver) {
  const loom_device_provider_t* providers[] = {
      &kFakeDeviceProvider,
  };
  loom_device_provider_registry_t registry = {};
  loom_device_provider_registry_initialize_from_entries(
      providers, IREE_ARRAYSIZE(providers), &registry);
  const iree_string_view_t device_uris[] = {
      IREE_SV("missing://adapter-0"),
  };

  const loom_device_provider_t* selected_provider = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_device_provider_registry_select(
                            &registry,
                            (iree_string_view_list_t){
                                /*.count=*/IREE_ARRAYSIZE(device_uris),
                                /*.values=*/device_uris,
                            },
                            iree_allocator_system(), &selected_provider));
  EXPECT_EQ(selected_provider, nullptr);
}

TEST(DeviceProviderRegistryTest, ReportsBuildWithoutExecutionProviders) {
  const iree_string_view_t device_uris[] = {
      IREE_SV("fake://adapter-0"),
  };
  const loom_device_provider_t* selected_provider = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_device_provider_registry_select(
                            /*registry=*/nullptr,
                            (iree_string_view_list_t){
                                /*.count=*/IREE_ARRAYSIZE(device_uris),
                                /*.values=*/device_uris,
                            },
                            iree_allocator_system(), &selected_provider));
  EXPECT_EQ(selected_provider, nullptr);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_device_provider_registry_select(
                            /*registry=*/nullptr, (iree_string_view_list_t){},
                            iree_allocator_system(), &selected_provider));
  EXPECT_EQ(selected_provider, nullptr);
}

}  // namespace
}  // namespace loom
