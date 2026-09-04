// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/device_plan.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

class PhysicalDeviceSnapshotBuilder {
 public:
  PhysicalDeviceSnapshotBuilder() {
    std::memset(&snapshot_, 0, sizeof(snapshot_));
    snapshot_.properties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    snapshot_.properties2.properties.apiVersion = VK_API_VERSION_1_3;
    snapshot_.features12.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    snapshot_.features12.bufferDeviceAddress = VK_TRUE;
    snapshot_.features12.timelineSemaphore = VK_TRUE;
    snapshot_.features12.scalarBlockLayout = VK_TRUE;
    snapshot_.features11.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    snapshot_.features13.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    snapshot_.features13.synchronization2 = VK_TRUE;
  }

  void AddQueueFamily(VkQueueFlags flags, uint32_t queue_count,
                      uint32_t timestamp_valid_bits = 64) {
    VkQueueFamilyProperties2 queue_family;
    std::memset(&queue_family, 0, sizeof(queue_family));
    queue_family.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
    queue_family.queueFamilyProperties.queueFlags = flags;
    queue_family.queueFamilyProperties.queueCount = queue_count;
    queue_family.queueFamilyProperties.timestampValidBits =
        timestamp_valid_bits;
    queue_families_.push_back(queue_family);
    snapshot_.queue_family_count =
        static_cast<uint32_t>(queue_families_.size());
    snapshot_.queue_families = queue_families_.data();
  }

  void EnableSparseBinding() {
    snapshot_.features2.features.sparseBinding = VK_TRUE;
  }

  void EnableSparseResidencyAliased() {
    snapshot_.features2.features.sparseResidencyBuffer = VK_TRUE;
    snapshot_.features2.features.sparseResidencyAliased = VK_TRUE;
  }

  void EnableBufferDeviceAddress() {
    snapshot_.features12.bufferDeviceAddress = VK_TRUE;
  }

  void DisableBufferDeviceAddress() {
    snapshot_.features12.bufferDeviceAddress = VK_FALSE;
  }

  void EnableScalarShaderFeatures() {
    snapshot_.features12.storageBuffer8BitAccess = VK_TRUE;
    snapshot_.features11.storageBuffer16BitAccess = VK_TRUE;
    snapshot_.features12.shaderFloat16 = VK_TRUE;
    snapshot_.features2.features.shaderFloat64 = VK_TRUE;
    snapshot_.features12.shaderInt8 = VK_TRUE;
    snapshot_.features2.features.shaderInt16 = VK_TRUE;
    snapshot_.features2.features.shaderInt64 = VK_TRUE;
    snapshot_.features12.shaderBufferInt64Atomics = VK_TRUE;
    snapshot_.features12.shaderSharedInt64Atomics = VK_TRUE;
    snapshot_.features13.shaderIntegerDotProduct = VK_TRUE;
    snapshot_.features12.vulkanMemoryModel = VK_TRUE;
    snapshot_.features12.vulkanMemoryModelDeviceScope = VK_TRUE;
  }

  void EnableCooperativeMatrixExtension() {
    snapshot_.available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX;
  }

  void EnableCooperativeMatrix() {
    EnableCooperativeMatrixExtension();
    snapshot_.cooperative_matrix_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    snapshot_.cooperative_matrix_features.cooperativeMatrix = VK_TRUE;
    snapshot_.cooperative_matrix_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    snapshot_.cooperative_matrix_properties.cooperativeMatrixSupportedStages =
        VK_SHADER_STAGE_COMPUTE_BIT;
  }

  void EnableShaderBfloat16Extension() {
    snapshot_.available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16;
  }

  void EnableMemoryBudgetExtension() {
    snapshot_.available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_MEMORY_BUDGET;
  }

  void EnableShaderBfloat16Features() {
    EnableShaderBfloat16Extension();
    snapshot_.shader_bfloat16_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    snapshot_.shader_bfloat16_features.shaderBFloat16Type = VK_TRUE;
    snapshot_.shader_bfloat16_features.shaderBFloat16DotProduct = VK_TRUE;
    snapshot_.shader_bfloat16_features.shaderBFloat16CooperativeMatrix =
        VK_TRUE;
  }

  void EnableShaderAtomicFloatExtension() {
    snapshot_.available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT;
  }

  void EnableShaderAtomicFloat2Extension() {
    snapshot_.available_extensions |=
        IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2;
  }

  void EnableShaderAtomicFloatExtensions() {
    EnableShaderAtomicFloatExtension();
    EnableShaderAtomicFloat2Extension();
  }

  void EnableShaderAtomicFloatFeatures() {
    EnableShaderAtomicFloatExtensions();
    snapshot_.shader_atomic_float_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    snapshot_.shader_atomic_float_features.shaderBufferFloat32Atomics = VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderBufferFloat32AtomicAdd =
        VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderBufferFloat64Atomics = VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderBufferFloat64AtomicAdd =
        VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderSharedFloat32Atomics = VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderSharedFloat32AtomicAdd =
        VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderSharedFloat64Atomics = VK_TRUE;
    snapshot_.shader_atomic_float_features.shaderSharedFloat64AtomicAdd =
        VK_TRUE;
    EnableShaderAtomicFloat2Features();
  }

  void EnableShaderAtomicFloat2Features() {
    EnableShaderAtomicFloat2Extension();
    snapshot_.shader_atomic_float2_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    snapshot_.shader_atomic_float2_features.shaderBufferFloat16Atomics =
        VK_TRUE;
    snapshot_.shader_atomic_float2_features.shaderBufferFloat16AtomicAdd =
        VK_TRUE;
    snapshot_.shader_atomic_float2_features.shaderSharedFloat16Atomics =
        VK_TRUE;
    snapshot_.shader_atomic_float2_features.shaderSharedFloat16AtomicAdd =
        VK_TRUE;
  }

  const iree_hal_vulkan_physical_device_snapshot_t* snapshot() const {
    return &snapshot_;
  }

 private:
  // Snapshot whose pointer fields reference this builder's storage.
  iree_hal_vulkan_physical_device_snapshot_t snapshot_;

  // Backing storage for snapshot_.queue_families.
  std::vector<VkQueueFamilyProperties2> queue_families_;
};

static iree_hal_vulkan_device_options_t DefaultDeviceOptions() {
  iree_hal_vulkan_device_options_t options;
  iree_hal_vulkan_device_options_initialize(&options);
  return options;
}

static iree_hal_vulkan_features_t NoFeatures() { return {}; }

static iree_hal_vulkan_features_t GeneralFeatures(
    iree_hal_vulkan_general_features_t general) {
  return {/*.general=*/general, /*.atomics=*/0};
}

static iree_hal_vulkan_features_t AtomicFeatures(
    iree_hal_vulkan_shader_atomic_features_t atomics) {
  return {/*.general=*/0, /*.atomics=*/atomics};
}

static iree_hal_vulkan_external_device_params_t DefaultExternalParams() {
  iree_hal_vulkan_external_device_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.enabled_features.general = IREE_HAL_VULKAN_FEATURE_REQUIRED_BASELINE;
  return params;
}

static iree_hal_vulkan_shader_atomic_features_t ShaderAtomicFloat2Features() {
  return IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16 |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16_ADD |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16 |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT16_ADD;
}

static iree_hal_vulkan_shader_atomic_features_t ShaderAtomicFloatFeatures() {
  return ShaderAtomicFloat2Features() |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32 |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32 |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT32_ADD |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64 |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64 |
         IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_FLOAT64_ADD;
}

class ScopedDevicePlan : public iree_hal_vulkan_device_plan_t {
 public:
  ScopedDevicePlan() : iree_hal_vulkan_device_plan_t{} {}

  ~ScopedDevicePlan() {
    if (initialized_) {
      iree_hal_vulkan_device_plan_deinitialize(this);
    }
  }

  iree_hal_vulkan_device_plan_t* Initialize() {
    initialized_ = true;
    return this;
  }

 private:
  bool initialized_ = false;
};

static iree_status_t iree_hal_vulkan_device_plan_initialize_for_create(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    iree_hal_vulkan_request_flags_t request_flags,
    iree_hal_vulkan_features_t required_features, ScopedDevicePlan* out_plan) {
  return ::iree_hal_vulkan_device_plan_initialize_for_create(
      snapshot, device_options, request_flags, required_features,
      iree_allocator_system(), out_plan->Initialize());
}

static iree_status_t iree_hal_vulkan_device_plan_initialize_for_wrap(
    const iree_hal_vulkan_physical_device_snapshot_t* snapshot,
    const iree_hal_vulkan_device_options_t* device_options,
    const iree_hal_vulkan_external_device_params_t* external_device_params,
    ScopedDevicePlan* out_plan) {
  return ::iree_hal_vulkan_device_plan_initialize_for_wrap(
      snapshot, device_options, external_device_params, iree_allocator_system(),
      out_plan->Initialize());
}

static bool PlanContainsExtension(const iree_hal_vulkan_device_plan_t& plan,
                                  const char* extension_name) {
  for (uint32_t i = 0; i < plan.enabled_extension_count; ++i) {
    if (std::strcmp(plan.enabled_extension_names[i], extension_name) == 0) {
      return true;
    }
  }
  return false;
}

TEST(DevicePlanTest, OwnedCreateInventoriesEverySelectedFamilyQueue) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 2);
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 3);
  builder.AddQueueFamily(VK_QUEUE_TRANSFER_BIT, 4);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  options.flags = IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_EQ(IREE_HAL_VULKAN_REQUEST_FLAG_NONE, plan.request_flags);
  EXPECT_EQ(1u, plan.queue_assignment.compute.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(2u, plan.queue_assignment.transfer.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.family_ordinal);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_ordinal);
  EXPECT_EQ(1u, plan.queue_assignment.transfer.family_ordinal);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.queue_ordinal);
  ASSERT_EQ(2u, plan.queue_inventory.family_count);
  EXPECT_EQ(1u, plan.queue_inventory.families[0].native_family_index);
  EXPECT_EQ(3u, plan.queue_inventory.families[0].queue_count);
  EXPECT_EQ(0u, plan.queue_inventory.families[0].queue_offset);
  EXPECT_EQ(2u, plan.queue_inventory.families[1].native_family_index);
  EXPECT_EQ(4u, plan.queue_inventory.families[1].queue_count);
  EXPECT_EQ(3u, plan.queue_inventory.families[1].queue_offset);
  ASSERT_EQ(7u, plan.queue_inventory.queue_count);
  const uint32_t expected_queue_indices[] = {0, 1, 2, 0, 1, 2, 3};
  EXPECT_EQ(
      0, std::memcmp(plan.queue_inventory.queue_indices, expected_queue_indices,
                     sizeof(expected_queue_indices)));
  EXPECT_EQ(2u, plan.queue_create_info_count);
  EXPECT_EQ(1u, plan.queue_create_infos[0].queueFamilyIndex);
  EXPECT_EQ(3u, plan.queue_create_infos[0].queueCount);
  EXPECT_EQ(2u, plan.queue_create_infos[1].queueFamilyIndex);
  EXPECT_EQ(4u, plan.queue_create_infos[1].queueCount);
}

TEST(DevicePlanTest,
     OwnedCreateUsesSecondQueueWhenTransferSharesComputeFamily) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT, 2);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_EQ(0u, plan.queue_assignment.compute.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.family_index);
  EXPECT_EQ(1u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_ordinal);
  EXPECT_EQ(1u, plan.queue_assignment.transfer.queue_ordinal);
  ASSERT_EQ(1u, plan.queue_inventory.family_count);
  EXPECT_TRUE(iree_all_bits_set(plan.queue_inventory.families[0].flags,
                                VK_QUEUE_TRANSFER_BIT));
  EXPECT_EQ(2u, plan.queue_inventory.families[0].queue_count);
  EXPECT_EQ(2u, plan.queue_inventory.queue_count);
  EXPECT_EQ(1u, plan.queue_create_info_count);
  EXPECT_EQ(2u, plan.queue_create_infos[0].queueCount);
}

TEST(DevicePlanTest, OwnedCreateKeepsSparseBindingSeparateFromSparseResidency) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
                             VK_QUEUE_SPARSE_BINDING_BIT,
                         2);
  builder.EnableSparseBinding();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      GeneralFeatures(IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING), &plan));

  EXPECT_TRUE(plan.enabled_features2.features.sparseBinding);
  EXPECT_FALSE(plan.enabled_features2.features.sparseResidencyBuffer);
  EXPECT_FALSE(plan.enabled_features2.features.sparseResidencyAliased);
  EXPECT_TRUE(iree_hal_vulkan_queue_assignment_has_sparse_binding(
      &plan.queue_assignment));
}

TEST(DevicePlanTest, OwnedCreateRequiresCompleteSparseResidencyRequest) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
                             VK_QUEUE_SPARSE_BINDING_BIT,
                         2);
  builder.EnableSparseBinding();
  builder.EnableSparseResidencyAliased();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  const iree_hal_vulkan_features_t sparse_residency_without_sparse_binding =
      GeneralFeatures(IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED &
                      ~IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING);

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          sparse_residency_without_sparse_binding, &plan));
}

TEST(DevicePlanTest, OwnedCreateRequiresBufferDeviceAddressBaseline) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.DisableBufferDeviceAddress();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          NoFeatures(), &plan));
}

TEST(DevicePlanTest, OwnedCreateEnablesBaselineBufferDeviceAddress) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_TRUE(plan.enabled_features12.bufferDeviceAddress);
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES));
}

TEST(DevicePlanTest, OwnedCreateReportsAvailableScalarShaderFeatures) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableScalarShaderFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_8BIT_ACCESS));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features.general,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features.general,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT64));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features.general,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT8));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features.general,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT16));
  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features.general,
                                IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INT64));
  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.atomics,
                        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_INT64));
  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.atomics,
                        IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_SHARED_INT64));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_INTEGER_DOT_PRODUCT));
  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_VULKAN_MEMORY_MODEL_DEVICE_SCOPE));
  EXPECT_TRUE(plan.enabled_features12.storageBuffer8BitAccess);
  EXPECT_TRUE(plan.enabled_features11.storageBuffer16BitAccess);
  EXPECT_TRUE(plan.enabled_features12.shaderFloat16);
  EXPECT_TRUE(plan.enabled_features2.features.shaderFloat64);
  EXPECT_TRUE(plan.enabled_features12.shaderInt8);
  EXPECT_TRUE(plan.enabled_features2.features.shaderInt16);
  EXPECT_TRUE(plan.enabled_features2.features.shaderInt64);
  EXPECT_TRUE(plan.enabled_features12.shaderBufferInt64Atomics);
  EXPECT_TRUE(plan.enabled_features12.shaderSharedInt64Atomics);
  EXPECT_TRUE(plan.enabled_features13.shaderIntegerDotProduct);
  EXPECT_TRUE(plan.enabled_features12.vulkanMemoryModel);
  EXPECT_TRUE(plan.enabled_features12.vulkanMemoryModelDeviceScope);
}

TEST(DevicePlanTest, OwnedCreateRejectsRequiredUnavailableReportedFeature) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          GeneralFeatures(IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_FLOAT16),
          &plan));
}

TEST(DevicePlanTest,
     OwnedCreateRejectsRequiredUnavailableStorageBuffer16BitAccess) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          GeneralFeatures(
              IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS),
          &plan));
}

TEST(DevicePlanTest, OwnedCreateEnablesCooperativeMatrixWhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrix();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_extensions,
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX));
  EXPECT_TRUE(
      PlanContainsExtension(plan, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME));
  EXPECT_TRUE(plan.enabled_cooperative_matrix_features.cooperativeMatrix);

  iree_hal_vulkan_device_plan_t copied_plan = plan;
  VkDeviceCreateInfo create_info;
  iree_hal_vulkan_device_plan_make_create_info(&copied_plan, &create_info);

  EXPECT_EQ(&copied_plan.enabled_features2, create_info.pNext);
  EXPECT_EQ(&copied_plan.enabled_features11,
            copied_plan.enabled_features2.pNext);
  EXPECT_EQ(&copied_plan.enabled_features12,
            copied_plan.enabled_features11.pNext);
  EXPECT_EQ(&copied_plan.enabled_features13,
            copied_plan.enabled_features12.pNext);
  EXPECT_EQ(&copied_plan.enabled_cooperative_matrix_features,
            copied_plan.enabled_features13.pNext);
}

TEST(DevicePlanTest, OwnedCreateEnablesMemoryBudgetWhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableMemoryBudgetExtension();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_extensions,
                        IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_MEMORY_BUDGET));
  EXPECT_TRUE(PlanContainsExtension(
      plan, IREE_HAL_VULKAN_EXT_MEMORY_BUDGET_EXTENSION_NAME));
}

TEST(DevicePlanTest, OwnedCreateEnablesShaderBfloat16WhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderBfloat16Features();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX));
  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_extensions,
                        IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16));
  EXPECT_TRUE(PlanContainsExtension(
      plan, IREE_HAL_VULKAN_KHR_SHADER_BFLOAT16_EXTENSION_NAME));
  EXPECT_TRUE(plan.enabled_shader_bfloat16_features.shaderBFloat16Type);
  EXPECT_TRUE(plan.enabled_shader_bfloat16_features.shaderBFloat16DotProduct);
  EXPECT_TRUE(
      plan.enabled_shader_bfloat16_features.shaderBFloat16CooperativeMatrix);

  iree_hal_vulkan_device_plan_t copied_plan = plan;
  VkDeviceCreateInfo create_info;
  iree_hal_vulkan_device_plan_make_create_info(&copied_plan, &create_info);

  EXPECT_EQ(&copied_plan.enabled_features2, create_info.pNext);
  EXPECT_EQ(&copied_plan.enabled_features11,
            copied_plan.enabled_features2.pNext);
  EXPECT_EQ(&copied_plan.enabled_features12,
            copied_plan.enabled_features11.pNext);
  EXPECT_EQ(&copied_plan.enabled_features13,
            copied_plan.enabled_features12.pNext);
  EXPECT_EQ(&copied_plan.enabled_shader_bfloat16_features,
            copied_plan.enabled_features13.pNext);
  EXPECT_EQ(nullptr, copied_plan.enabled_shader_bfloat16_features.pNext);
}

TEST(DevicePlanTest, OwnedCreateRejectsRequiredShaderBfloat16WhenUnavailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderBfloat16Extension();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          GeneralFeatures(IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE),
          &plan));
}

TEST(DevicePlanTest, OwnedCreateEnablesShaderAtomicFloatWhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloatFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  const iree_hal_vulkan_shader_atomic_features_t expected_features =
      ShaderAtomicFloatFeatures();
  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.atomics, expected_features));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_extensions,
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT |
          IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2));
  EXPECT_TRUE(
      PlanContainsExtension(plan, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME));
  EXPECT_TRUE(
      PlanContainsExtension(plan, VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME));

  iree_hal_vulkan_device_plan_t copied_plan = plan;
  VkDeviceCreateInfo create_info;
  iree_hal_vulkan_device_plan_make_create_info(&copied_plan, &create_info);

  EXPECT_EQ(&copied_plan.enabled_features2, create_info.pNext);
  EXPECT_EQ(&copied_plan.enabled_features11,
            copied_plan.enabled_features2.pNext);
  EXPECT_EQ(&copied_plan.enabled_features12,
            copied_plan.enabled_features11.pNext);
  EXPECT_EQ(&copied_plan.enabled_features13,
            copied_plan.enabled_features12.pNext);
  EXPECT_EQ(&copied_plan.enabled_shader_atomic_float_features,
            copied_plan.enabled_features13.pNext);
  EXPECT_EQ(&copied_plan.enabled_shader_atomic_float2_features,
            copied_plan.enabled_shader_atomic_float_features.pNext);
  EXPECT_EQ(nullptr, copied_plan.enabled_shader_atomic_float2_features.pNext);
}

TEST(DevicePlanTest,
     OwnedCreateDoesNotEnableShaderAtomicFloat2WithoutBaseExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloat2Features();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  EXPECT_FALSE(iree_any_bit_set(
      plan.enabled_extensions,
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2));
  EXPECT_FALSE(iree_any_bit_set(plan.enabled_features.atomics,
                                ShaderAtomicFloat2Features()));
}

TEST(DevicePlanTest,
     OwnedCreateRejectsRequiredShaderAtomicFloat2WithoutBaseExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloat2Features();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          AtomicFeatures(IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16),
          &plan));
}

TEST(DevicePlanTest,
     OwnedCreateRejectsRequiredShaderAtomicFloatWhenUnavailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kUnavailable,
      iree_hal_vulkan_device_plan_initialize_for_create(
          builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
          AtomicFeatures(
              IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32_ADD),
          &plan));
}

TEST(DevicePlanTest,
     OwnedCreateChainsShaderBfloat16BeforeCooperativeMatrixFeatures) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrix();
  builder.EnableShaderBfloat16Features();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_NONE,
      NoFeatures(), &plan));

  iree_hal_vulkan_device_plan_t copied_plan = plan;
  VkDeviceCreateInfo create_info;
  iree_hal_vulkan_device_plan_make_create_info(&copied_plan, &create_info);

  EXPECT_EQ(&copied_plan.enabled_shader_bfloat16_features,
            copied_plan.enabled_features13.pNext);
  EXPECT_EQ(&copied_plan.enabled_cooperative_matrix_features,
            copied_plan.enabled_shader_bfloat16_features.pNext);
  EXPECT_EQ(nullptr, copied_plan.enabled_cooperative_matrix_features.pNext);
}

TEST(DevicePlanTest, OwnedCreateCarriesRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_create(
      builder.snapshot(), &options, IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS,
      NoFeatures(), &plan));

  EXPECT_EQ(IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, plan.request_flags);
}

TEST(DevicePlanTest, OwnedCreateRejectsUnknownRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_plan_initialize_for_create(
                            builder.snapshot(), &options,
                            IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED + 1,
                            NoFeatures(), &plan));
}

TEST(DevicePlanTest, WrapRejectsRequestFlagsInEnabledFeatures) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_REQUEST_FLAG_VALIDATION_LAYERS;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapInfersImplicitTransferFromCompute) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_EQ(0u, plan.queue_assignment.compute.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.family_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(1ull << 1, plan.queue_assignment.transfer.affinity);
}

TEST(DevicePlanTest, WrapCarriesRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_EQ(IREE_HAL_VULKAN_REQUEST_FLAG_DEBUG_UTILS, plan.request_flags);
}

TEST(DevicePlanTest, WrapCarriesStorageBuffer16BitAccessWhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableScalarShaderFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS));
}

TEST(DevicePlanTest,
     WrapRejectsStorageBuffer16BitAccessWhenFeatureIsUnavailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_STORAGE_BUFFER_16BIT_ACCESS;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsCooperativeMatrixWithoutEnabledExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrix();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsCooperativeMatrixWhenFeatureIsUnavailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableCooperativeMatrixExtension();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_COOPERATIVE_MATRIX;
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_COOPERATIVE_MATRIX;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapCarriesShaderBfloat16WhenAvailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderBfloat16Features();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT |
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX;
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_TRUE(
      iree_all_bits_set(plan.enabled_features.general,
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_DOT_PRODUCT));
  EXPECT_TRUE(iree_all_bits_set(
      plan.enabled_features.general,
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_COOPERATIVE_MATRIX));
}

TEST(DevicePlanTest, WrapRejectsShaderBfloat16WithoutEnabledExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderBfloat16Features();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsShaderBfloat16WhenFeatureIsUnavailable) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderBfloat16Extension();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SHADER_BFLOAT16_TYPE;
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_KHR_SHADER_BFLOAT16;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapCarriesShaderAtomicFloatFeatures) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloatFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.atomics |= ShaderAtomicFloatFeatures();
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT |
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_TRUE(iree_all_bits_set(plan.enabled_features.atomics,
                                ShaderAtomicFloatFeatures()));
}

TEST(DevicePlanTest, WrapRejectsShaderAtomicFloatWithoutEnabledExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloatFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.atomics |=
      IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT32;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsShaderAtomicFloat2WithoutEnabledExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloatFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.atomics |=
      IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsShaderAtomicFloat2WithoutBaseExtension) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloatFeatures();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.atomics |=
      IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT16;
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT2;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsUnavailableShaderAtomicFloatFeature) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);
  builder.EnableShaderAtomicFloatExtensions();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.atomics |=
      IREE_HAL_VULKAN_SHADER_ATOMIC_FEATURE_BUFFER_FLOAT64_ADD;
  params.enabled_extensions |=
      IREE_HAL_VULKAN_DEVICE_EXTENSION_EXT_SHADER_ATOMIC_FLOAT;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsUnknownRequestFlags) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.request_flags = IREE_HAL_VULKAN_REQUEST_FLAG_ALL_RECOGNIZED + 1;
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 0;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapRejectsQueuesOutsideTheNativeFamily) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 2);

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.compute_queue_set.queue_family_index = 0;
  params.compute_queue_set.queue_indices = 1ull << 2;

  ScopedDevicePlan plan;
  IREE_EXPECT_STATUS_IS(StatusCode::kOutOfRange,
                        iree_hal_vulkan_device_plan_initialize_for_wrap(
                            builder.snapshot(), &options, &params, &plan));
}

TEST(DevicePlanTest, WrapInventoriesEverySuppliedQueueWithoutDuplicates) {
  PhysicalDeviceSnapshotBuilder builder;
  builder.AddQueueFamily(VK_QUEUE_TRANSFER_BIT, 5);
  builder.AddQueueFamily(VK_QUEUE_GRAPHICS_BIT, 1);
  builder.AddQueueFamily(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
                             VK_QUEUE_SPARSE_BINDING_BIT,
                         6);
  builder.EnableSparseBinding();

  iree_hal_vulkan_device_options_t options = DefaultDeviceOptions();
  iree_hal_vulkan_external_device_params_t params = DefaultExternalParams();
  params.enabled_features.general |=
      IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING;
  params.compute_queue_set.queue_family_index = 2;
  params.compute_queue_set.queue_indices = (1ull << 1) | (1ull << 5);
  params.transfer_queue_set.queue_family_index = 0;
  params.transfer_queue_set.queue_indices = (1ull << 2) | (1ull << 4);
  params.sparse_binding_queue_set.queue_family_index = 2;
  params.sparse_binding_queue_set.queue_indices = 1ull << 5;

  ScopedDevicePlan plan;
  IREE_ASSERT_OK(iree_hal_vulkan_device_plan_initialize_for_wrap(
      builder.snapshot(), &options, &params, &plan));

  EXPECT_EQ(1u, plan.queue_assignment.compute.queue_index);
  EXPECT_EQ(1u, plan.queue_assignment.compute.family_ordinal);
  EXPECT_EQ(0u, plan.queue_assignment.compute.queue_ordinal);
  EXPECT_EQ(2u, plan.queue_assignment.transfer.queue_index);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.family_ordinal);
  EXPECT_EQ(0u, plan.queue_assignment.transfer.queue_ordinal);
  EXPECT_EQ(5u, plan.queue_assignment.sparse_binding.queue_index);
  EXPECT_EQ(1u, plan.queue_assignment.sparse_binding.family_ordinal);
  EXPECT_EQ(1u, plan.queue_assignment.sparse_binding.queue_ordinal);
  ASSERT_EQ(2u, plan.queue_inventory.family_count);
  EXPECT_EQ(0u, plan.queue_inventory.families[0].native_family_index);
  EXPECT_EQ(2u, plan.queue_inventory.families[0].queue_count);
  EXPECT_EQ(2u, plan.queue_inventory.families[1].native_family_index);
  EXPECT_EQ(2u, plan.queue_inventory.families[1].queue_count);
  ASSERT_EQ(4u, plan.queue_inventory.queue_count);
  const uint32_t expected_queue_indices[] = {2, 4, 1, 5};
  EXPECT_EQ(
      0, std::memcmp(plan.queue_inventory.queue_indices, expected_queue_indices,
                     sizeof(expected_queue_indices)));
}

}  // namespace
}  // namespace iree::hal::vulkan
