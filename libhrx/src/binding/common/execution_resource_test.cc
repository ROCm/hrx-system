// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/execution_resource.h"

#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::streaming {
namespace {

static constexpr iree_hal_queue_priority_t kQueuePriorities[] = {
    IREE_HAL_QUEUE_PRIORITY_NORMAL,
};
static constexpr iree_hal_queue_execution_resource_group_spec_t
    kExecutionResourceGroups[] = {
        {.minimum_selected_resource_count = 1},
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
            .group_ordinal = 1,
            .first_execution_unit_ordinal = 4,
            .execution_unit_count = 2,
        },
        {
            .group_ordinal = 1,
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
      .logical_device_id = IREE_SV("execution-resource-test"),
      .display_name = IREE_SV("Execution resource test device"),
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
  const iree_hal_queue_family_spec_t queue_family = {
      .name = IREE_SV("dispatch"),
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
  };
  const iree_hal_device_queue_spec_t queues = {
      .family_count = 1,
      .families = &queue_family,
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

static iree_hal_device_t* CreateMockDevice(
    iree_hal_device_spec_t* device_spec) {
  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = IREE_SV("execution-resource-test");
  options.device_spec = device_spec;
  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  return device;
}

class ExecutionResourceTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_device_spec_t* device_spec = CreateDeviceSpec();
    device_ = CreateMockDevice(device_spec);
    iree_hal_device_spec_release(device_spec);
    IREE_CHECK_OK(iree_hal_streaming_execution_resource_table_initialize(
        device_, iree_allocator_system(), &table_));
  }

  void TearDown() override {
    iree_hal_streaming_execution_resource_table_deinitialize(&table_);
    iree_hal_device_release(device_);
  }

  const iree_hal_queue_family_t* queue_family() const {
    return iree_hal_device_queue_family(device_, 0);
  }

  iree_hal_device_t* device_ = nullptr;
  iree_hal_streaming_execution_resource_table_t table_ = {};
};

TEST_F(ExecutionResourceTableTest, InternsCanonicalResourceSets) {
  iree_hal_streaming_execution_resource_set_id_t full_set_id =
      IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_execution_resource_table_intern(
      &table_, queue_family(), {.count = 0, .ordinals = nullptr},
      &full_set_id));

  const iree_hal_queue_execution_resource_ordinal_t full_ordinals[] = {0, 1, 2,
                                                                       3};
  iree_hal_streaming_execution_resource_set_id_t explicit_full_set_id =
      IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_execution_resource_table_intern(
      &table_, queue_family(),
      {.count = IREE_ARRAYSIZE(full_ordinals), .ordinals = full_ordinals},
      &explicit_full_set_id));
  EXPECT_EQ(explicit_full_set_id, full_set_id);

  const iree_hal_queue_execution_resource_ordinal_t partition_ordinals[] = {0,
                                                                            2};
  iree_hal_streaming_execution_resource_set_id_t partition_set_id =
      IREE_HAL_STREAMING_EXECUTION_RESOURCE_SET_ID_INVALID;
  IREE_ASSERT_OK(iree_hal_streaming_execution_resource_table_intern(
      &table_, queue_family(),
      {
          .count = IREE_ARRAYSIZE(partition_ordinals),
          .ordinals = partition_ordinals,
      },
      &partition_set_id));
  EXPECT_NE(partition_set_id, full_set_id);

  const iree_hal_streaming_execution_resource_set_t* partition_set =
      iree_hal_streaming_execution_resource_table_resolve(&table_,
                                                          partition_set_id);
  ASSERT_NE(partition_set, nullptr);
  EXPECT_EQ(partition_set->queue_family_ordinal, 0u);
  EXPECT_EQ(partition_set->execution_unit_count, 4u);
  ASSERT_EQ(partition_set->resources.count, IREE_ARRAYSIZE(partition_ordinals));
  EXPECT_EQ(partition_set->resources.ordinals[0], 0u);
  EXPECT_EQ(partition_set->resources.ordinals[1], 2u);
}

TEST_F(ExecutionResourceTableTest, RejectsSetsThatCannotNameAnExactQueue) {
  const iree_hal_queue_execution_resource_ordinal_t missing_group[] = {0, 1};
  iree_hal_streaming_execution_resource_set_id_t set_id = 42;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_streaming_execution_resource_table_intern(
          &table_, queue_family(),
          {.count = IREE_ARRAYSIZE(missing_group), .ordinals = missing_group},
          &set_id));
  EXPECT_EQ(set_id, 42u);

  const iree_hal_queue_execution_resource_ordinal_t unsorted[] = {2, 0};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_streaming_execution_resource_table_intern(
          &table_, queue_family(),
          {.count = IREE_ARRAYSIZE(unsorted), .ordinals = unsorted}, &set_id));
  EXPECT_EQ(set_id, 42u);

  iree_hal_device_spec_t* foreign_device_spec = CreateDeviceSpec();
  iree_hal_device_t* foreign_device = CreateMockDevice(foreign_device_spec);
  iree_hal_device_spec_release(foreign_device_spec);
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_streaming_execution_resource_table_intern(
          &table_, iree_hal_device_queue_family(foreign_device, 0),
          {.count = 0, .ordinals = nullptr}, &set_id));
  EXPECT_EQ(set_id, 42u);
  iree_hal_device_release(foreign_device);
}

}  // namespace
}  // namespace iree::hal::streaming
