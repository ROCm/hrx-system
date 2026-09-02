// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "loomc/diagnostic.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/iree_hal.h"
#include "loomc/target/spirv/profile.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using DeviceSpecPtr =
    HandlePtr<iree_hal_device_spec_t, iree_hal_device_spec_release>;

constexpr uint32_t kVulkanApiVersion13 =
    (1u << 22) | (3u << 12) | static_cast<uint32_t>(0);

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Immutable device facts borrowed from the test.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

static const iree_hal_device_spec_t* FakeHalDeviceSpec(
    iree_hal_device_t* base_device) {
  FakeHalDevice* device = reinterpret_cast<FakeHalDevice*>(base_device);
  return device->device_spec;
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.device_spec = FakeHalDeviceSpec;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

void InitializeFakeDevice(const iree_hal_device_spec_t* device_spec,
                          FakeHalDevice* out_device) {
  out_device->device_spec = device_spec;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &out_device->resource);
}

iree_hal_vulkan_features_t RequiredVulkanFeatures() {
  return {/*.general=*/
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64,
          /*.atomics=*/0};
}

iree_status_t CreateVulkanDeviceSpec(
    iree_hal_vulkan_features_t enabled_features, bool include_dispatch,
    bool include_target, DeviceSpecPtr* out_device_spec) {
  out_device_spec->reset();
  iree_hal_vulkan_device_spec_t vulkan_spec = {
      /*.api_version=*/kVulkanApiVersion13,
      /*.driver_version=*/1,
      /*.physical_device_type=*/2,
      /*.enabled_features=*/enabled_features,
      /*.flags=*/IREE_HAL_VULKAN_DEVICE_SPEC_FLAG_NONE,
  };
  iree_host_size_t vulkan_payload_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_calculate_payload_size(
      /*property_count=*/0, &vulkan_payload_size));
  std::vector<uint8_t> vulkan_payload_storage(vulkan_payload_size);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_device_spec_encode(
      &vulkan_spec, /*property_count=*/0, /*properties=*/nullptr,
      iree_make_byte_span(vulkan_payload_storage.data(),
                          vulkan_payload_storage.size())));
  iree_hal_device_spec_facet_t vulkan_facet = {
      /*.schema_id=*/
      iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID),
      /*.schema_version=*/IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION,
      /*.payload=*/
      iree_make_const_byte_span(vulkan_payload_storage.data(),
                                vulkan_payload_storage.size()),
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_ok_status();
  if (include_dispatch) {
    iree_hal_device_dispatch_spec_t dispatch = {
        /*.launch=*/
        {
            /*.maximum_workgroup_invocations=*/256,
            /*.maximum_workgroup_size=*/{256, 128, 64},
            /*.maximum_workgroup_count=*/{65535, 65535, 65535},
        },
        /*.subgroup=*/
        {
            /*.default_size=*/32,
            /*.minimum_size=*/32,
            /*.maximum_size=*/32,
            /*.supported_size_mask=*/1ull << 32,
        },
        /*.execution=*/
        {
            /*.unit_count=*/1,
            /*.group_count=*/1,
        },
        /*.addressing=*/
        {
            /*.pointer_size_bits=*/64,
            /*.address_space_bits=*/64,
        },
        /*.flags=*/IREE_HAL_DEVICE_DISPATCH_SPEC_FLAG_NONE,
    };
    status = iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  }
  const iree_hal_executable_target_t executable_target = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/include_target ? 1u : 0u,
      /*.targets=*/include_target ? &executable_target : nullptr,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status) && include_target) {
    status =
        iree_hal_device_spec_builder_set_executables(&builder, &executables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_add_facet(&builder, &vulkan_facet);
  }
  iree_hal_device_spec_t* device_spec = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, &device_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    out_device_spec->reset(device_spec);
  }
  return status;
}

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

void ExpectFailedSpirvIreeHalResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_GE(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(diagnostic->severity, LOOMC_DIAGNOSTIC_SEVERITY_ERROR);
  EXPECT_EQ(ToString(diagnostic->code), "SPIRV/IREE_HAL");
}

TargetProfilePtr CreateProfileFromHal(
    loomc_target_environment_t* target_environment, FakeHalDevice* device,
    loomc_result_t** out_result) {
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(device);
  loomc_spirv_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("fake-vulkan"),
      /*.device=*/hal_device,
      /*.physical_device_affinity=*/0,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_iree_hal(
      target_environment, &options, loomc_allocator_system(), &profile,
      out_result);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TEST(LoomcSpirvIreeHalTargetTest, CreatesProfileFromHalFacts) {
  iree_hal_vulkan_features_t enabled_features = RequiredVulkanFeatures();
  enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS;
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(enabled_features,
                                        /*include_dispatch=*/true,
                                        /*include_target=*/true, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      CreateProfileFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
  loomc_spirv_limit_value_t limit = {};
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_limit(
      profile.get(), LOOMC_SPIRV_LIMIT_MAX_WORKGROUP_SIZE_X, &limit));
  EXPECT_EQ(limit.state, LOOMC_TARGET_FACT_STATE_TRUE);
  EXPECT_EQ(limit.value, 256u);
  loomc_target_fact_state_t feature_state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_FLOAT16, &feature_state));
  EXPECT_EQ(feature_state, LOOMC_TARGET_FACT_STATE_TRUE);
  LOOMC_EXPECT_OK(loomc_spirv_target_profile_query_feature(
      profile.get(), LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
      &feature_state));
  EXPECT_EQ(feature_state, LOOMC_TARGET_FACT_STATE_TRUE);
}

TEST(LoomcSpirvIreeHalTargetTest, MissingExecutableTargetFailsResult) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(RequiredVulkanFeatures(),
                                        /*include_dispatch=*/true,
                                        /*include_target=*/false,
                                        &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      CreateProfileFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, MissingRequiredHalFactFailsResult) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(RequiredVulkanFeatures(),
                                        /*include_dispatch=*/false,
                                        /*include_target=*/true, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      CreateProfileFromHal(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, ProviderRoutesThroughGenericHalRouter) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateVulkanDeviceSpec(RequiredVulkanFeatures(),
                                        /*include_dispatch=*/true,
                                        /*include_target=*/true, &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_spirv_iree_hal_profile_provider(),
  };
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("router"),
      /*.device=*/hal_device,
      /*.physical_device_affinity=*/0,
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  ASSERT_NE(profile_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, ProviderMissLetsRouterReportUnsupported) {
  FakeHalDevice device = {};
  InitializeFakeDevice(nullptr, &device);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_spirv_iree_hal_profile_provider(),
  };
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("miss"),
      /*.device=*/hal_device,
      /*.physical_device_affinity=*/0,
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loomc_result_t* result = nullptr;
  loomc_target_profile_t* profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &options, loomc_allocator_system(), &profile,
      &result));
  TargetProfilePtr profile_ptr(profile);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile_ptr.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
}

}  // namespace
