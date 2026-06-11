// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/iree_hal.h"

#include <cstdint>
#include <memory>
#include <string>

#include "iree/hal/api.h"
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

constexpr uint32_t kVulkanApiVersion13 =
    (1u << 22) | (3u << 12) | static_cast<uint32_t>(0);

typedef struct FakeQueryRow {
  // HAL query category.
  iree_string_view_t category;

  // HAL query key within category.
  iree_string_view_t key;

  // Scalar query value returned to the caller.
  int64_t value;
} FakeQueryRow;

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Query rows exposed by this fake device.
  const FakeQueryRow* rows;

  // Number of entries in rows.
  iree_host_size_t row_count;
} FakeHalDevice;

typedef struct FakeExecutableCache {
  // HAL resource header used by executable-cache vtable dispatch.
  iree_hal_resource_t resource;

  // Whether this cache accepts the raw Vulkan BDA executable format.
  bool raw_bda_supported;
} FakeExecutableCache;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

static iree_status_t FakeHalDeviceQueryI64(iree_hal_device_t* base_device,
                                           iree_string_view_t category,
                                           iree_string_view_t key,
                                           int64_t* out_value) {
  FakeHalDevice* device = reinterpret_cast<FakeHalDevice*>(base_device);
  for (iree_host_size_t i = 0; i < device->row_count; ++i) {
    if (iree_string_view_equal(category, device->rows[i].category) &&
        iree_string_view_equal(key, device->rows[i].key)) {
      *out_value = device->rows[i].value;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

static bool FakeExecutableCacheCanPrepareFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  (void)caching_mode;
  FakeExecutableCache* executable_cache =
      reinterpret_cast<FakeExecutableCache*>(base_executable_cache);
  return executable_cache->raw_bda_supported &&
         iree_string_view_equal(executable_format, IREE_SV("vulkan-spirv-bda"));
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable = {
    /*.destroy=*/nullptr,
    /*.id=*/nullptr,
    /*.host_allocator=*/nullptr,
    /*.device_allocator=*/nullptr,
    /*.replace_device_allocator=*/nullptr,
    /*.replace_channel_provider=*/nullptr,
    /*.trim=*/nullptr,
    /*.query_i64=*/FakeHalDeviceQueryI64,
};

static const iree_hal_executable_cache_vtable_t kFakeExecutableCacheVtable = {
    /*.destroy=*/nullptr,
    /*.infer_format=*/nullptr,
    /*.can_prepare_format=*/FakeExecutableCacheCanPrepareFormat,
};

FakeHalDevice MakeFakeDevice(const FakeQueryRow* rows,
                             iree_host_size_t row_count) {
  FakeHalDevice device = {
      /*.resource=*/{},
      /*.rows=*/rows,
      /*.row_count=*/row_count,
  };
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &device.resource);
  return device;
}

FakeExecutableCache MakeFakeExecutableCache(bool raw_bda_supported) {
  FakeExecutableCache executable_cache = {
      /*.resource=*/{},
      /*.raw_bda_supported=*/raw_bda_supported,
  };
  iree_hal_resource_initialize(&kFakeExecutableCacheVtable,
                               &executable_cache.resource);
  return executable_cache;
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
    FakeExecutableCache* executable_cache, loomc_result_t** out_result) {
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(device);
  iree_hal_executable_cache_t* hal_executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache);
  loomc_spirv_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("fake-vulkan"),
      /*.device=*/hal_device,
      /*.executable_cache=*/hal_executable_cache,
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_spirv_iree_hal(
      target_environment, &options, loomc_allocator_system(), &profile,
      out_result);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TEST(LoomcSpirvIreeHalTargetTest, CreatesProfileFromHalFacts) {
  const FakeQueryRow rows[] = {
      {IREE_SVL("vulkan.device"), IREE_SVL("api_version"), kVulkanApiVersion13},
      {IREE_SVL("vulkan.device"), IREE_SVL("subgroup_size"), 32},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_invocations"),
       256},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_x"),
       256},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_y"),
       128},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_z"), 64},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_x"),
       65535},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_y"),
       65535},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_z"),
       65535},
      {IREE_SVL("vulkan.feature"), IREE_SVL("buffer_device_address"), 1},
      {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int64"), 1},
      {IREE_SVL("vulkan.feature"), IREE_SVL("shader_float16"), 1},
      {IREE_SVL("vulkan.feature"), IREE_SVL("storage_buffer_8bit_access"), 1},
  };
  FakeHalDevice device = MakeFakeDevice(rows, IREE_ARRAYSIZE(rows));
  FakeExecutableCache executable_cache = MakeFakeExecutableCache(true);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = CreateProfileFromHal(
      target_environment.get(), &device, &executable_cache, &result);
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

TEST(LoomcSpirvIreeHalTargetTest, MissingExecutableFormatFailsResult) {
  const FakeQueryRow rows[] = {
      {IREE_SVL("vulkan.device"), IREE_SVL("api_version"), kVulkanApiVersion13},
  };
  FakeHalDevice device = MakeFakeDevice(rows, IREE_ARRAYSIZE(rows));
  FakeExecutableCache executable_cache = MakeFakeExecutableCache(false);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = CreateProfileFromHal(
      target_environment.get(), &device, &executable_cache, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, MissingRequiredHalFactFailsResult) {
  const FakeQueryRow rows[] = {
      {IREE_SVL("vulkan.device"), IREE_SVL("api_version"), kVulkanApiVersion13},
      {IREE_SVL("vulkan.feature"), IREE_SVL("buffer_device_address"), 1},
      {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int64"), 1},
  };
  FakeHalDevice device = MakeFakeDevice(rows, IREE_ARRAYSIZE(rows));
  FakeExecutableCache executable_cache = MakeFakeExecutableCache(true);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = CreateProfileFromHal(
      target_environment.get(), &device, &executable_cache, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ExpectFailedSpirvIreeHalResult(result_ptr.get());
}

TEST(LoomcSpirvIreeHalTargetTest, ProviderRoutesThroughGenericHalRouter) {
  const FakeQueryRow rows[] = {
      {IREE_SVL("vulkan.device"), IREE_SVL("api_version"), kVulkanApiVersion13},
      {IREE_SVL("vulkan.device"), IREE_SVL("subgroup_size"), 32},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_invocations"),
       256},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_x"),
       256},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_y"),
       128},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_size_z"), 64},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_x"),
       65535},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_y"),
       65535},
      {IREE_SVL("vulkan.device"), IREE_SVL("max_compute_workgroup_count_z"),
       65535},
      {IREE_SVL("vulkan.feature"), IREE_SVL("buffer_device_address"), 1},
      {IREE_SVL("vulkan.feature"), IREE_SVL("shader_int64"), 1},
  };
  FakeHalDevice device = MakeFakeDevice(rows, IREE_ARRAYSIZE(rows));
  FakeExecutableCache executable_cache = MakeFakeExecutableCache(true);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  iree_hal_executable_cache_t* hal_executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(&executable_cache);
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_spirv_iree_hal_profile_provider(),
  };
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("router"),
      /*.device=*/hal_device,
      /*.executable_cache=*/hal_executable_cache,
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
  FakeHalDevice device = MakeFakeDevice(nullptr, 0);
  FakeExecutableCache executable_cache = MakeFakeExecutableCache(true);
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  iree_hal_device_t* hal_device = reinterpret_cast<iree_hal_device_t*>(&device);
  iree_hal_executable_cache_t* hal_executable_cache =
      reinterpret_cast<iree_hal_executable_cache_t*>(&executable_cache);
  const loomc_iree_hal_profile_provider_t* providers[] = {
      loomc_spirv_iree_hal_profile_provider(),
  };
  loomc_iree_hal_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("miss"),
      /*.device=*/hal_device,
      /*.executable_cache=*/hal_executable_cache,
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
