// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/execution_resource_descriptor.h"

#include "binding/hip/execution_resource.h"
#include "common/device.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"

namespace iree::hip {
namespace {

static constexpr iree_hal_queue_priority_t kQueuePriorities[] = {
    IREE_HAL_QUEUE_PRIORITY_NORMAL,
};
static constexpr iree_hal_queue_execution_resource_group_spec_t
    kExecutionResourceGroups[] = {
        {.minimum_selected_resource_count = 1},
};
static constexpr iree_hal_queue_execution_resource_spec_t
    kExecutionResources[] = {
        {
            .group_ordinal = 0,
            .first_execution_unit_ordinal = 0,
            .execution_unit_count = 2,
        },
        {
            .group_ordinal = 0,
            .first_execution_unit_ordinal = 2,
            .execution_unit_count = 2,
        },
        {
            .group_ordinal = 0,
            .first_execution_unit_ordinal = 4,
            .execution_unit_count = 2,
        },
        {
            .group_ordinal = 0,
            .first_execution_unit_ordinal = 6,
            .execution_unit_count = 2,
        },
};

static iree_hal_device_spec_t* CreateDeviceSpec() {
  const iree_hal_physical_device_spec_t physical_device = {
      .identity =
          {
              .display_name = IREE_SV("Test GPU"),
              .backend_path = IREE_SV("test://gpu"),
          },
      .physical_ordinal = 0,
      .partition_ordinal = 0,
      .partition_count = 1,
      .physical_device_affinity = 1,
  };
  const iree_hal_device_identity_spec_t identity = {
      .logical_device_id = IREE_SV("hip-resource-descriptor-test"),
      .display_name = IREE_SV("HIP resource descriptor test device"),
      .driver_id = IREE_SV("mock"),
      .driver_version = IREE_SV("test"),
      .backend_id = IREE_SV("mock"),
      .device_path = IREE_SV("test://device"),
      .vendor_name = IREE_SV("Test"),
      .vendor_id = 0,
      .device_id = 0,
      .revision_id = 0,
      .logical_ordinal = 0,
      .physical_device_count = 1,
      .physical_devices = &physical_device,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  const iree_hal_queue_family_spec_t queue_families[] = {
      {
          .name = IREE_SV("dispatch-a"),
          .provisioned_queue_count = 0,
          .priority_count = IREE_ARRAYSIZE(kQueuePriorities),
          .priorities = kQueuePriorities,
          .execution_unit_count = 8,
          .execution_resource_group_count =
              IREE_ARRAYSIZE(kExecutionResourceGroups),
          .execution_resource_groups = kExecutionResourceGroups,
          .execution_resource_count = IREE_ARRAYSIZE(kExecutionResources),
          .execution_resources = kExecutionResources,
          .supported_queue_features = IREE_HAL_QUEUE_FEATURE_FLAG_NONE,
          .timestamp_valid_bits = 0,
          .timestamp_frequency_hz = 0,
          .physical_device_affinity = 1,
          .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
          .atomic_capabilities = {},
          .zero_compute_atomic_capabilities = {},
          .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION,
      },
      {
          .name = IREE_SV("dispatch-b"),
          .provisioned_queue_count = 0,
          .priority_count = IREE_ARRAYSIZE(kQueuePriorities),
          .priorities = kQueuePriorities,
          .execution_unit_count = 8,
          .execution_resource_group_count =
              IREE_ARRAYSIZE(kExecutionResourceGroups),
          .execution_resource_groups = kExecutionResourceGroups,
          .execution_resource_count = IREE_ARRAYSIZE(kExecutionResources),
          .execution_resources = kExecutionResources,
          .supported_queue_features = IREE_HAL_QUEUE_FEATURE_FLAG_NONE,
          .timestamp_valid_bits = 0,
          .timestamp_frequency_hz = 0,
          .physical_device_affinity = 1,
          .role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
          .atomic_capabilities = {},
          .zero_compute_atomic_capabilities = {},
          .flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION,
      },
  };
  const iree_hal_device_queue_spec_t queues = {
      .family_count = IREE_ARRAYSIZE(queue_families),
      .families = queue_families,
  };
  const iree_hal_device_spec_params_t params = {
      .identity = &identity,
      .memory = nullptr,
      .virtual_memory = nullptr,
      .queues = &queues,
  };
  iree_hal_device_spec_t* device_spec = nullptr;
  IREE_CHECK_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                            &device_spec));
  return device_spec;
}

class ExecutionResourceDescriptorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_device_spec_t* device_spec = CreateDeviceSpec();
    iree_hal_mock_device_options_t options;
    iree_hal_mock_device_options_initialize(&options);
    options.identifier = IREE_SV("hip-resource-descriptor-test");
    options.device_spec = device_spec;
    IREE_CHECK_OK(iree_hal_mock_device_create(&options, iree_allocator_system(),
                                              &hal_device_));
    iree_hal_device_spec_release(device_spec);

    device_.ordinal = 11;
    device_.hal_device = hal_device_;
    IREE_CHECK_OK(iree_hal_streaming_execution_resource_table_initialize(
        hal_device_, iree_allocator_system(),
        &device_.execution_resource_table));
  }

  void TearDown() override {
    iree_hal_streaming_execution_resource_table_deinitialize(
        &device_.execution_resource_table);
    iree_hal_device_release(hal_device_);
  }

  hipDevResource CreateResource(
      iree_hal_queue_family_ordinal_t family_ordinal,
      iree_hal_queue_execution_resource_list_t resources) {
    hipDevResource resource;
    IREE_CHECK_OK(iree_hip_execution_resource_create_sm(
        &device_, iree_hal_device_queue_family(hal_device_, family_ordinal),
        resources, hipDevSmResourceGroupDefault, &resource));
    return resource;
  }

  iree_hal_device_t* hal_device_ = nullptr;
  iree_hal_streaming_device_t device_ = {};
};

TEST_F(ExecutionResourceDescriptorTest, UnionsDisjointExactSets) {
  const iree_hal_queue_execution_resource_ordinal_t even_ordinals[] = {0, 2};
  const iree_hal_queue_execution_resource_ordinal_t odd_ordinals[] = {1, 3};
  hipDevResource resources[] = {
      CreateResource(
          /*family_ordinal=*/0,
          {.count = IREE_ARRAYSIZE(even_ordinals), .ordinals = even_ordinals}),
      CreateResource(
          /*family_ordinal=*/0,
          {.count = IREE_ARRAYSIZE(odd_ordinals), .ordinals = odd_ordinals}),
  };
  resources[0].nextResource = &resources[1];

  hipDevResourceDesc_t handle = nullptr;
  ASSERT_EQ(iree_hip_execution_resource_descriptor_create(
                &device_, resources, IREE_ARRAYSIZE(resources), &handle),
            hipSuccess);
  ASSERT_NE(handle, nullptr);

  iree_hip_execution_resource_descriptor_t* descriptor = nullptr;
  ASSERT_TRUE(iree_hip_execution_resource_descriptor_take(handle, &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->device_ordinal, device_.ordinal);
  EXPECT_EQ(descriptor->table_incarnation,
            iree_hal_streaming_execution_resource_table_incarnation(
                &device_.execution_resource_table));
  EXPECT_EQ(descriptor->queue_family_ordinal, 0u);
  EXPECT_EQ(descriptor->resource_count, IREE_ARRAYSIZE(resources));
  EXPECT_EQ(descriptor->resources[0].nextResource, nullptr);
  EXPECT_EQ(descriptor->resources[1].nextResource, nullptr);

  const iree_hal_streaming_execution_resource_set_t* union_set =
      iree_hal_streaming_execution_resource_table_resolve(
          &device_.execution_resource_table, descriptor->sm_resource_set_id);
  ASSERT_NE(union_set, nullptr);
  ASSERT_EQ(union_set->resources.count, IREE_ARRAYSIZE(kExecutionResources));
  for (iree_host_size_t i = 0; i < union_set->resources.count; ++i) {
    EXPECT_EQ(union_set->resources.ordinals[i],
              (iree_hal_queue_execution_resource_ordinal_t)i);
  }

  iree_hip_execution_resource_descriptor_t* unchanged_descriptor = descriptor;
  EXPECT_FALSE(iree_hip_execution_resource_descriptor_take(
      handle, &unchanged_descriptor));
  EXPECT_EQ(unchanged_descriptor, descriptor);
  iree_hip_execution_resource_descriptor_release(descriptor);
}

TEST_F(ExecutionResourceDescriptorTest,
       RejectsOverlapAndCrossFamilyWithoutPublishing) {
  const iree_hal_queue_execution_resource_ordinal_t first_ordinals[] = {0, 1};
  const iree_hal_queue_execution_resource_ordinal_t overlap_ordinals[] = {1, 2};
  const iree_hal_queue_execution_resource_ordinal_t disjoint_ordinals[] = {2,
                                                                           3};
  const hipDevResource first_resource = CreateResource(
      /*family_ordinal=*/0,
      {.count = IREE_ARRAYSIZE(first_ordinals), .ordinals = first_ordinals});
  const hipDevResource overlap_resource = CreateResource(
      /*family_ordinal=*/0, {
                                .count = IREE_ARRAYSIZE(overlap_ordinals),
                                .ordinals = overlap_ordinals,
                            });
  const hipDevResource other_family_resource = CreateResource(
      /*family_ordinal=*/1, {
                                .count = IREE_ARRAYSIZE(disjoint_ordinals),
                                .ordinals = disjoint_ordinals,
                            });

  auto expect_invalid_without_publication =
      [&](const hipDevResource& second_resource) {
        hipDevResource resources[] = {first_resource, second_resource};
        hipDevResourceDesc_t descriptor =
            reinterpret_cast<hipDevResourceDesc_t>(1);
        EXPECT_EQ(
            iree_hip_execution_resource_descriptor_create(
                &device_, resources, IREE_ARRAYSIZE(resources), &descriptor),
            hipErrorInvalidResourceConfiguration);
        EXPECT_EQ(descriptor, reinterpret_cast<hipDevResourceDesc_t>(1));
      };

  expect_invalid_without_publication(overlap_resource);
  expect_invalid_without_publication(other_family_resource);
}

}  // namespace
}  // namespace iree::hip
