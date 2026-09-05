// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/spirv/device_provider.h"

#include <memory>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/device_spec.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/target/spirv/vulkan_profile.h"

namespace loom {
namespace {

struct DeviceSpecDeleter {
  void operator()(iree_hal_device_spec_t* device_spec) const {
    iree_hal_device_spec_release(device_spec);
  }
};
using DeviceSpecPtr =
    std::unique_ptr<iree_hal_device_spec_t, DeviceSpecDeleter>;

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Immutable device facts borrowed from the test fixture.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

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

static iree_hal_vulkan_features_t BaselineVulkanFeatures() {
  return {/*.general=*/
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL |
              IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS |
              IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16 |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8 |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16 |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64 |
              IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT |
              IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL,
          /*.atomics=*/0};
}

static iree_status_t CreateBaselineDeviceSpec(DeviceSpecPtr* out_device_spec) {
  out_device_spec->reset();
  const iree_hal_device_dispatch_spec_t dispatch = {
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
  const iree_hal_vulkan_device_spec_t vulkan_spec = {
      /*.api_version=*/LOOM_SPIRV_VULKAN_API_VERSION_1_3,
      /*.driver_version=*/1,
      /*.physical_device_type=*/2,
      /*.enabled_features=*/BaselineVulkanFeatures(),
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
  const iree_hal_device_spec_facet_t vulkan_facet = {
      /*.schema_id=*/
      iree_make_cstring_view(IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_ID),
      /*.schema_version=*/IREE_HAL_VULKAN_DEVICE_SPEC_SCHEMA_VERSION,
      /*.payload=*/
      iree_make_const_byte_span(vulkan_payload_storage.data(),
                                vulkan_payload_storage.size()),
  };

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status =
      iree_hal_device_spec_builder_set_dispatch(&builder, &dispatch);
  const iree_hal_executable_target_t executable_target = {
      /*.family=*/IREE_SV("spirv"),
      /*.target_key=*/IREE_SV("vulkan1.3+bda"),
      /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
      /*.priority=*/100,
      /*.physical_device_affinity=*/1,
      /*.flags=*/IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
  };
  const iree_hal_device_executable_spec_t executables = {
      /*.target_count=*/1,
      /*.targets=*/&executable_target,
      /*.flags=*/IREE_HAL_DEVICE_EXECUTABLE_SPEC_FLAG_NONE,
  };
  if (iree_status_is_ok(status)) {
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

class SpirvDeviceProviderTest : public ::testing::Test {
 protected:
  iree_status_t SelectBaselineTarget() {
    IREE_RETURN_IF_ERROR(CreateBaselineDeviceSpec(&device_spec_));
    device_.device_spec = device_spec_.get();
    iree_hal_resource_initialize(&kFakeHalDeviceVtable, &device_.resource);
    runtime_.device = reinterpret_cast<iree_hal_device_t*>(&device_);
    return loom_device_provider_select_target(
        &loom_spirv_vulkan_device_provider, &runtime_, &target_);
  }

  // Immutable device-spec storage used by |device_|.
  DeviceSpecPtr device_spec_;

  // Stack HAL device exposing |device_spec_| through the real device API.
  FakeHalDevice device_ = {};

  // Runtime view passed through the production device-provider contract.
  loom_run_hal_runtime_t runtime_ = {};

  // Self-contained device target selected by the provider.
  loom_device_target_t target_ = {};
};

TEST_F(SpirvDeviceProviderTest, SelectsRawBdaTarget) {
  IREE_ASSERT_OK(SelectBaselineTarget());

  const loom_target_bundle_t* target_bundle =
      loom_device_target_bundle(&target_);
  ASSERT_NE(target_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(loom_device_target_key(&target_),
                                     IREE_SV("vulkan1.3+bda")));
  EXPECT_EQ(target_bundle->snapshot->codegen_format,
            LOOM_TARGET_CODEGEN_FORMAT_SPIRV);
  EXPECT_EQ(target_bundle->snapshot->artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY);
  EXPECT_EQ(target_bundle->snapshot->default_pointer_bitwidth, 64u);
  EXPECT_EQ(target_bundle->snapshot->offset_bitwidth, 64u);
  EXPECT_EQ(target_bundle->export_plan->abi_kind, LOOM_TARGET_ABI_HAL_KERNEL);

  loom_device_target_profile_t device_profile = {};
  IREE_ASSERT_OK(loom_device_target_profile_initialize(
      &loom_spirv_vulkan_device_provider, &runtime_, &target_,
      &device_profile));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_facts_t* base_facts = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(&device_profile.base, &arena,
                                                   &base_facts));
  const loom_spirv_target_facts_t* facts =
      loom_spirv_target_facts_cast(base_facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->base.selector, LOOM_SPIRV_TARGET_KIND_VULKAN1_3);
  EXPECT_EQ(facts->cooperative_properties.matrix_property_count, 0u);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
