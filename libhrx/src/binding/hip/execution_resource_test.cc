// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/execution_resource.h"

#include <cstring>

#include "common/device.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hip {
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
    kVariableExecutionResources[] = {
        {.group_ordinal = 0,
         .first_execution_unit_ordinal = 0,
         .execution_unit_count = 2},
        {.group_ordinal = 0,
         .first_execution_unit_ordinal = 2,
         .execution_unit_count = 4},
        {.group_ordinal = 1,
         .first_execution_unit_ordinal = 6,
         .execution_unit_count = 2},
        {.group_ordinal = 1,
         .first_execution_unit_ordinal = 8,
         .execution_unit_count = 6},
};
static constexpr iree_hal_queue_execution_resource_spec_t
    kUniformExecutionResources[] = {
        {.group_ordinal = 0,
         .first_execution_unit_ordinal = 0,
         .execution_unit_count = 2},
        {.group_ordinal = 1,
         .first_execution_unit_ordinal = 2,
         .execution_unit_count = 2},
        {.group_ordinal = 0,
         .first_execution_unit_ordinal = 4,
         .execution_unit_count = 2},
        {.group_ordinal = 1,
         .first_execution_unit_ordinal = 6,
         .execution_unit_count = 2},
        {.group_ordinal = 0,
         .first_execution_unit_ordinal = 8,
         .execution_unit_count = 2},
        {.group_ordinal = 1,
         .first_execution_unit_ordinal = 10,
         .execution_unit_count = 2},
        {.group_ordinal = 0,
         .first_execution_unit_ordinal = 12,
         .execution_unit_count = 2},
        {.group_ordinal = 1,
         .first_execution_unit_ordinal = 14,
         .execution_unit_count = 2},
};

static iree_hal_device_spec_t* CreateDeviceSpec() {
  const iree_hal_physical_device_spec_t physical_device = {
      .identity = {
          /*.display_name=*/IREE_SV("Test GPU"),
          /*.backend_path=*/IREE_SV("test://gpu"),
      },
      .physical_ordinal = 0,
      .partition_ordinal = 0,
      .partition_count = 1,
      .physical_device_affinity = 1,
  };
  const iree_hal_device_identity_spec_t identity = {
      /*.logical_device_id=*/IREE_SV("hip-execution-resource-test"),
      /*.display_name=*/IREE_SV("HIP execution resource test device"),
      /*.driver_id=*/IREE_SV("mock"),
      /*.driver_version=*/IREE_SV("test"),
      /*.backend_id=*/IREE_SV("mock"),
      /*.device_path=*/IREE_SV("test://device"),
      /*.vendor_name=*/IREE_SV("Test"),
      /*.vendor_id=*/0,
      /*.device_id=*/0,
      /*.revision_id=*/0,
      /*.logical_ordinal=*/0,
      /*.physical_device_count=*/1,
      /*.physical_devices=*/&physical_device,
      /*.flags=*/IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  const iree_hal_queue_family_spec_t queue_families[] = {
      {
          /*.name=*/IREE_SV("variable-dispatch"),
          /*.provisioned_queue_count=*/0,
          /*.priority_count=*/IREE_ARRAYSIZE(kQueuePriorities),
          /*.priorities=*/kQueuePriorities,
          /*.execution_unit_count=*/14,
          /*.execution_resource_group_count=*/
          IREE_ARRAYSIZE(kExecutionResourceGroups),
          /*.execution_resource_groups=*/kExecutionResourceGroups,
          /*.execution_resource_count=*/
          IREE_ARRAYSIZE(kVariableExecutionResources),
          /*.execution_resources=*/kVariableExecutionResources,
          /*.supported_queue_features=*/IREE_HAL_QUEUE_FEATURE_FLAG_NONE,
          /*.timestamp_valid_bits=*/0,
          /*.timestamp_frequency_hz=*/0,
          /*.physical_device_affinity=*/1,
          /*.role_flags=*/IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
          /*.atomic_capabilities=*/{},
          /*.zero_compute_atomic_capabilities=*/{},
          /*.flags=*/IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION,
      },
      {
          /*.name=*/IREE_SV("uniform-dispatch"),
          /*.provisioned_queue_count=*/0,
          /*.priority_count=*/IREE_ARRAYSIZE(kQueuePriorities),
          /*.priorities=*/kQueuePriorities,
          /*.execution_unit_count=*/16,
          /*.execution_resource_group_count=*/
          IREE_ARRAYSIZE(kExecutionResourceGroups),
          /*.execution_resource_groups=*/kExecutionResourceGroups,
          /*.execution_resource_count=*/
          IREE_ARRAYSIZE(kUniformExecutionResources),
          /*.execution_resources=*/kUniformExecutionResources,
          /*.supported_queue_features=*/IREE_HAL_QUEUE_FEATURE_FLAG_NONE,
          /*.timestamp_valid_bits=*/0,
          /*.timestamp_frequency_hz=*/0,
          /*.physical_device_affinity=*/1,
          /*.role_flags=*/IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH,
          /*.atomic_capabilities=*/{},
          /*.zero_compute_atomic_capabilities=*/{},
          /*.flags=*/IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION,
      },
  };
  const iree_hal_device_queue_spec_t queues = {
      /*.family_count=*/IREE_ARRAYSIZE(queue_families),
      /*.families=*/queue_families,
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

class ExecutionResourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_hal_device_spec_t* device_spec = CreateDeviceSpec();
    iree_hal_mock_device_options_t options;
    iree_hal_mock_device_options_initialize(&options);
    options.identifier = IREE_SV("hip-execution-resource-test");
    options.device_spec = device_spec;
    IREE_CHECK_OK(iree_hal_mock_device_create(&options, iree_allocator_system(),
                                              &hal_device_));
    iree_hal_device_spec_release(device_spec);

    device_.ordinal = 7;
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

  hipError_t SplitSmByCount(hipDevResource* out_resources,
                            unsigned int* inout_group_count,
                            const hipDevResource* input,
                            hipDevResource* out_remainder, unsigned int flags,
                            unsigned int minimum_count) {
    const iree_hal_streaming_execution_resource_set_t* input_set = nullptr;
    hipError_t result = iree_hip_execution_resource_resolve_sm_for_device(
        input, &device_, &input_set);
    if (result != hipSuccess) {
      return result;
    }
    return iree_hip_execution_resource_split_sm_by_count(
        &device_, input_set, input, flags, minimum_count, out_resources,
        inout_group_count, out_remainder);
  }

  hipError_t SplitSm(hipDevResource* out_resources, unsigned int group_count,
                     const hipDevResource* input, hipDevResource* out_remainder,
                     unsigned int flags,
                     hipDevSmResourceGroupParams* group_parameters) {
    const iree_hal_streaming_execution_resource_set_t* input_set = nullptr;
    hipError_t result = iree_hip_execution_resource_resolve_sm_for_device(
        input, &device_, &input_set);
    if (result != hipSuccess) {
      return result;
    }
    return iree_hip_execution_resource_split_sm(
        &device_, input_set, input, group_count, flags, group_parameters,
        out_resources, out_remainder);
  }

  const iree_hal_queue_family_t* variable_queue_family() const {
    return iree_hal_device_queue_family(hal_device_, 0);
  }

  const iree_hal_queue_family_t* uniform_queue_family() const {
    return iree_hal_device_queue_family(hal_device_, 1);
  }

  iree_hal_device_t* hal_device_ = nullptr;
  iree_hal_streaming_device_t device_ = {};
};

TEST_F(ExecutionResourceTest, CreatesCopyableResourcesFromExactSets) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, variable_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));
  EXPECT_EQ(full_resource.type, hipDevResourceTypeSm);
  EXPECT_EQ(full_resource.sm.smCount, 14u);
  EXPECT_EQ(full_resource.sm.minSmPartitionSize, 4u);
  EXPECT_EQ(full_resource.sm.smCoscheduledAlignment, 2u);
  EXPECT_EQ(full_resource.sm.flags, hipDevSmResourceGroupDefault);
  EXPECT_EQ(full_resource.nextResource, nullptr);

  hipDevResource copied_resource = full_resource;
  const iree_hal_streaming_execution_resource_set_t* full_set = nullptr;
  EXPECT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &copied_resource, &device_, &full_set),
            hipSuccess);
  ASSERT_NE(full_set, nullptr);
  EXPECT_EQ(full_set->resources.count,
            IREE_ARRAYSIZE(kVariableExecutionResources));

  const iree_hal_queue_execution_resource_ordinal_t partition_ordinals[] = {1,
                                                                            3};
  hipDevResource partition_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, variable_queue_family(),
      {/*.count=*/IREE_ARRAYSIZE(partition_ordinals),
       /*.ordinals=*/partition_ordinals},
      hipDevSmResourceGroupBackfill, &partition_resource));
  EXPECT_EQ(partition_resource.sm.smCount, 10u);
  EXPECT_EQ(partition_resource.sm.minSmPartitionSize, 10u);
  EXPECT_EQ(partition_resource.sm.smCoscheduledAlignment, 2u);
  EXPECT_EQ(partition_resource.sm.flags, hipDevSmResourceGroupBackfill);
  const iree_hal_streaming_execution_resource_set_t* partition_set = nullptr;
  EXPECT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &partition_resource, &device_, &partition_set),
            hipSuccess);
}

TEST_F(ExecutionResourceTest, TranslatesCuMasksIntoExactResourceSets) {
  const uint32_t exact_mask[] = {0x00000F0Fu, UINT32_MAX};
  const iree_hal_streaming_execution_resource_set_t* exact_set = nullptr;
  IREE_ASSERT_OK(iree_hip_execution_resource_intern_sm_cu_mask(
      &device_, uniform_queue_family(), IREE_ARRAYSIZE(exact_mask), exact_mask,
      &exact_set));
  ASSERT_NE(exact_set, nullptr);
  const iree_hal_queue_execution_resource_ordinal_t expected_ordinals[] = {
      0, 1, 4, 5};
  ASSERT_EQ(exact_set->resources.count, IREE_ARRAYSIZE(expected_ordinals));
  EXPECT_EQ(std::memcmp(exact_set->resources.ordinals, expected_ordinals,
                        sizeof(expected_ordinals)),
            0);

  uint32_t round_trip_mask[] = {0xA5A5A5A5u, 0xA5A5A5A5u};
  IREE_ASSERT_OK(iree_hip_execution_resource_write_sm_cu_mask(
      uniform_queue_family(), exact_set->resources,
      IREE_ARRAYSIZE(round_trip_mask), round_trip_mask));
  EXPECT_EQ(round_trip_mask[0], exact_mask[0]);
  EXPECT_EQ(round_trip_mask[1], 0u);

  const uint32_t effectively_empty_mask[] = {0u, UINT32_MAX};
  const iree_hal_streaming_execution_resource_set_t* full_set = nullptr;
  IREE_ASSERT_OK(iree_hip_execution_resource_intern_sm_cu_mask(
      &device_, uniform_queue_family(), IREE_ARRAYSIZE(effectively_empty_mask),
      effectively_empty_mask, &full_set));
  ASSERT_NE(full_set, nullptr);
  EXPECT_EQ(full_set->resources.count,
            IREE_ARRAYSIZE(kUniformExecutionResources));
}

TEST_F(ExecutionResourceTest, RejectsInexactCuMasksWithoutPublishing) {
  const iree_hal_streaming_execution_resource_set_t* const sentinel =
      reinterpret_cast<const iree_hal_streaming_execution_resource_set_t*>(
          uintptr_t{1});
  const iree_hal_streaming_execution_resource_set_t* output = sentinel;

  const uint32_t partial_resource_mask[] = {0x00000001u};
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hip_execution_resource_intern_sm_cu_mask(
                            &device_, uniform_queue_family(),
                            IREE_ARRAYSIZE(partial_resource_mask),
                            partial_resource_mask, &output));
  EXPECT_EQ(output, sentinel);

  const uint32_t missing_group_mask[] = {0x00000033u};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hip_execution_resource_intern_sm_cu_mask(
          &device_, uniform_queue_family(), IREE_ARRAYSIZE(missing_group_mask),
          missing_group_mask, &output));
  EXPECT_EQ(output, sentinel);

  uint32_t untouched_mask = 0xA5A5A5A5u;
  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hip_execution_resource_write_sm_cu_mask(
          uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
          /*mask_word_count=*/0, &untouched_mask));
  EXPECT_EQ(untouched_mask, 0xA5A5A5A5u);
}

TEST_F(ExecutionResourceTest, RejectsStaleAndTamperedCopies) {
  hipDevResource unchanged_resource;
  std::memset(&unchanged_resource, 0xA5, sizeof(unchanged_resource));
  const hipDevResource expected_unchanged_resource = unchanged_resource;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hip_execution_resource_create_sm(
                            &device_, variable_queue_family(),
                            {/*.count=*/0, /*.ordinals=*/nullptr}, /*flags=*/2,
                            &unchanged_resource));
  EXPECT_EQ(std::memcmp(&unchanged_resource, &expected_unchanged_resource,
                        sizeof(unchanged_resource)),
            0);

  hipDevResource stale_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, variable_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &stale_resource));

  iree_hal_streaming_execution_resource_table_deinitialize(
      &device_.execution_resource_table);
  IREE_ASSERT_OK(iree_hal_streaming_execution_resource_table_initialize(
      hal_device_, iree_allocator_system(), &device_.execution_resource_table));
  const iree_hal_streaming_execution_resource_set_t* resolved_set =
      reinterpret_cast<const iree_hal_streaming_execution_resource_set_t*>(1);
  EXPECT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &stale_resource, &device_, &resolved_set),
            hipErrorInvalidResourceConfiguration);
  EXPECT_EQ(
      resolved_set,
      reinterpret_cast<const iree_hal_streaming_execution_resource_set_t*>(1));

  hipDevResource tampered_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, variable_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &tampered_resource));
  ++tampered_resource.sm.smCount;
  EXPECT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &tampered_resource, &device_, &resolved_set),
            hipErrorInvalidResourceConfiguration);
  EXPECT_EQ(
      resolved_set,
      reinterpret_cast<const iree_hal_streaming_execution_resource_set_t*>(1));

  --tampered_resource.sm.smCount;
  tampered_resource.sm.flags = hipDevSmResourceGroupBackfill;
  EXPECT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &tampered_resource, &device_, &resolved_set),
            hipErrorInvalidResourceConfiguration);
  EXPECT_EQ(
      resolved_set,
      reinterpret_cast<const iree_hal_streaming_execution_resource_set_t*>(1));
}

TEST_F(ExecutionResourceTest, SplitsUniformResourcesIntoExactDisjointSets) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));

  hipDevResource untouched_remainder = {};
  std::memset(&untouched_remainder, 0xA5, sizeof(untouched_remainder));
  const hipDevResource expected_untouched_remainder = untouched_remainder;
  unsigned int partition_count = 1;
  EXPECT_EQ(SplitSmByCount(
                /*out_resources=*/nullptr, &partition_count, &full_resource,
                &untouched_remainder, /*flags=*/0, /*minimum_count=*/5),
            hipSuccess);
  EXPECT_EQ(partition_count, 2u);
  EXPECT_EQ(std::memcmp(&untouched_remainder, &expected_untouched_remainder,
                        sizeof(untouched_remainder)),
            0);

  hipDevResource requested_partition;
  unsigned int requested_partition_count = 1;
  ASSERT_EQ(
      SplitSmByCount(&requested_partition, &requested_partition_count,
                     &full_resource, /*out_remainder=*/nullptr, /*flags=*/0,
                     /*minimum_count=*/5),
      hipSuccess);
  EXPECT_EQ(requested_partition_count, 1u);
  EXPECT_EQ(requested_partition.sm.smCount, 6u);

  hipDevResource partitions[4];
  hipDevResource remainder;
  partition_count = IREE_ARRAYSIZE(partitions);
  ASSERT_EQ(
      SplitSmByCount(partitions, &partition_count, &full_resource, &remainder,
                     /*flags=*/0, /*minimum_count=*/5),
      hipSuccess);
  ASSERT_EQ(partition_count, 2u);
  for (iree_host_size_t i = 0; i < partition_count; ++i) {
    EXPECT_EQ(partitions[i].type, hipDevResourceTypeSm);
    EXPECT_EQ(partitions[i].sm.smCount, 6u);
    EXPECT_EQ(partitions[i].sm.minSmPartitionSize, 4u);
    EXPECT_EQ(partitions[i].sm.smCoscheduledAlignment, 2u);
    EXPECT_EQ(partitions[i].sm.flags, hipDevSmResourceGroupDefault);
  }
  EXPECT_EQ(remainder.type, hipDevResourceTypeSm);
  EXPECT_EQ(remainder.sm.smCount, 4u);
  EXPECT_EQ(remainder.sm.minSmPartitionSize, 4u);
  EXPECT_EQ(remainder.sm.smCoscheduledAlignment, 2u);

  bool seen_ordinals[IREE_ARRAYSIZE(kUniformExecutionResources)] = {};
  auto record_exact_set = [&](const hipDevResource* resource) {
    const iree_hal_streaming_execution_resource_set_t* set = nullptr;
    EXPECT_EQ(iree_hip_execution_resource_resolve_sm_for_device(resource,
                                                                &device_, &set),
              hipSuccess);
    ASSERT_NE(set, nullptr);
    iree_host_size_t group_counts[IREE_ARRAYSIZE(kExecutionResourceGroups)] =
        {};
    for (iree_host_size_t i = 0; i < set->resources.count; ++i) {
      const iree_hal_queue_execution_resource_ordinal_t ordinal =
          set->resources.ordinals[i];
      ASSERT_LT(ordinal, IREE_ARRAYSIZE(kUniformExecutionResources));
      EXPECT_FALSE(seen_ordinals[ordinal]);
      seen_ordinals[ordinal] = true;
      ++group_counts[kUniformExecutionResources[ordinal].group_ordinal];
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kExecutionResourceGroups);
         ++i) {
      EXPECT_GE(group_counts[i],
                kExecutionResourceGroups[i].minimum_selected_resource_count);
    }
  };
  for (iree_host_size_t i = 0; i < partition_count; ++i) {
    record_exact_set(&partitions[i]);
  }
  record_exact_set(&remainder);
  for (bool seen_ordinal : seen_ordinals) {
    EXPECT_TRUE(seen_ordinal);
  }
}

TEST_F(ExecutionResourceTest, SplitsStructuredUnevenExactSets) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));

  hipDevSmResourceGroupParams group_parameters[2] = {};
  group_parameters[0].smCount = 6;
  group_parameters[1].smCount = 4;
  hipDevResource partitions[2];
  hipDevResource remainder;
  ASSERT_EQ(SplitSm(partitions, IREE_ARRAYSIZE(partitions), &full_resource,
                    &remainder, /*flags=*/0, group_parameters),
            hipSuccess);

  EXPECT_EQ(group_parameters[0].smCount, 6u);
  EXPECT_EQ(group_parameters[1].smCount, 4u);
  for (const auto& parameter : group_parameters) {
    EXPECT_EQ(parameter.coscheduledSmCount, 2u);
    EXPECT_EQ(parameter.preferredCoscheduledSmCount, 2u);
  }
  EXPECT_EQ(partitions[0].sm.smCount, 6u);
  EXPECT_EQ(partitions[1].sm.smCount, 4u);
  EXPECT_EQ(remainder.sm.smCount, 6u);

  bool seen_ordinals[IREE_ARRAYSIZE(kUniformExecutionResources)] = {};
  const hipDevResource* resources[] = {&partitions[0], &partitions[1],
                                       &remainder};
  for (const hipDevResource* resource : resources) {
    const iree_hal_streaming_execution_resource_set_t* set = nullptr;
    ASSERT_EQ(iree_hip_execution_resource_resolve_sm_for_device(resource,
                                                                &device_, &set),
              hipSuccess);
    iree_host_size_t group_counts[IREE_ARRAYSIZE(kExecutionResourceGroups)] =
        {};
    for (iree_host_size_t i = 0; i < set->resources.count; ++i) {
      const iree_hal_queue_execution_resource_ordinal_t ordinal =
          set->resources.ordinals[i];
      ASSERT_LT(ordinal, IREE_ARRAYSIZE(kUniformExecutionResources));
      EXPECT_FALSE(seen_ordinals[ordinal]);
      seen_ordinals[ordinal] = true;
      ++group_counts[kUniformExecutionResources[ordinal].group_ordinal];
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kExecutionResourceGroups);
         ++i) {
      EXPECT_GE(group_counts[i],
                kExecutionResourceGroups[i].minimum_selected_resource_count);
    }
  }
  for (bool seen_ordinal : seen_ordinals) {
    EXPECT_TRUE(seen_ordinal);
  }
}

TEST_F(ExecutionResourceTest, BackfillsAutomaticallySizedStructuredGroup) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));

  hipDevSmResourceGroupParams group_parameters[2] = {};
  group_parameters[0].smCount = 4;
  group_parameters[1].flags = hipDevSmResourceGroupBackfill;
  hipDevResource partitions[2];
  hipDevResource remainder;
  ASSERT_EQ(SplitSm(partitions, IREE_ARRAYSIZE(partitions), &full_resource,
                    &remainder, /*flags=*/0, group_parameters),
            hipSuccess);

  EXPECT_EQ(group_parameters[0].smCount, 4u);
  EXPECT_EQ(partitions[0].sm.smCount, 4u);
  EXPECT_EQ(group_parameters[1].smCount, 12u);
  EXPECT_EQ(partitions[1].sm.smCount, 12u);
  EXPECT_EQ(partitions[1].sm.flags, hipDevSmResourceGroupBackfill);
  EXPECT_EQ(remainder.type, hipDevResourceTypeInvalid);
}

TEST_F(ExecutionResourceTest, DiscoversStructuredGroupsInOrder) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));

  hipDevSmResourceGroupParams group_parameters[2] = {};
  ASSERT_EQ(SplitSm(/*out_resources=*/nullptr, IREE_ARRAYSIZE(group_parameters),
                    &full_resource,
                    /*out_remainder=*/nullptr, /*flags=*/0, group_parameters),
            hipSuccess);

  EXPECT_EQ(group_parameters[0].smCount, full_resource.sm.smCount);
  EXPECT_EQ(group_parameters[1].smCount, 0u);
  for (const auto& parameter : group_parameters) {
    EXPECT_EQ(parameter.coscheduledSmCount, 2u);
    EXPECT_EQ(parameter.preferredCoscheduledSmCount, 2u);
  }
}

TEST_F(ExecutionResourceTest, DryRunPublishesStructuredRemainder) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));

  hipDevSmResourceGroupParams group_parameter = {};
  group_parameter.smCount = 4;
  hipDevResource remainder;
  ASSERT_EQ(SplitSm(/*out_resources=*/nullptr, /*group_count=*/1,
                    &full_resource, &remainder, /*flags=*/0, &group_parameter),
            hipSuccess);

  EXPECT_EQ(group_parameter.smCount, 4u);
  ASSERT_EQ(remainder.type, hipDevResourceTypeSm);
  EXPECT_EQ(remainder.sm.smCount, 12u);
  const iree_hal_streaming_execution_resource_set_t* remainder_set = nullptr;
  ASSERT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &remainder, &device_, &remainder_set),
            hipSuccess);
  ASSERT_NE(remainder_set, nullptr);
  EXPECT_EQ(remainder_set->resources.count, 6u);
}

TEST_F(ExecutionResourceTest,
       RejectsUnprovenStructuredCoschedulingWithoutPublishing) {
  hipDevResource full_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &full_resource));

  hipDevSmResourceGroupParams group_parameter = {};
  group_parameter.smCount = 4;
  group_parameter.coscheduledSmCount = 4;
  const hipDevSmResourceGroupParams expected_parameter = group_parameter;
  hipDevResource partition;
  std::memset(&partition, 0xA5, sizeof(partition));
  const hipDevResource expected_partition = partition;

  EXPECT_EQ(SplitSm(&partition, /*group_count=*/1, &full_resource,
                    /*out_remainder=*/nullptr, /*flags=*/0, &group_parameter),
            hipErrorNotSupported);
  EXPECT_EQ(std::memcmp(&group_parameter, &expected_parameter,
                        sizeof(group_parameter)),
            0);
  EXPECT_EQ(std::memcmp(&partition, &expected_partition, sizeof(partition)), 0);
}

TEST_F(ExecutionResourceTest,
       RejectsImpossibleStructuredSplitWithoutPublishing) {
  const iree_hal_queue_execution_resource_ordinal_t input_ordinals[] = {0, 1, 2,
                                                                        4};
  hipDevResource input_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(),
      {/*.count=*/IREE_ARRAYSIZE(input_ordinals),
       /*.ordinals=*/input_ordinals},
      hipDevSmResourceGroupDefault, &input_resource));

  hipDevSmResourceGroupParams group_parameters[2] = {};
  group_parameters[0].smCount = 4;
  group_parameters[1].smCount = 4;
  const hipDevSmResourceGroupParams expected_parameters[2] = {
      group_parameters[0], group_parameters[1]};
  hipDevResource partitions[2];
  std::memset(partitions, 0xA5, sizeof(partitions));
  hipDevResource expected_partitions[2];
  std::memcpy(expected_partitions, partitions, sizeof(partitions));
  hipDevResource remainder;
  std::memset(&remainder, 0x5A, sizeof(remainder));
  const hipDevResource expected_remainder = remainder;

  EXPECT_EQ(SplitSm(partitions, IREE_ARRAYSIZE(partitions), &input_resource,
                    &remainder, /*flags=*/0, group_parameters),
            hipErrorInvalidResourceConfiguration);
  EXPECT_EQ(std::memcmp(group_parameters, expected_parameters,
                        sizeof(group_parameters)),
            0);
  EXPECT_EQ(std::memcmp(partitions, expected_partitions, sizeof(partitions)),
            0);
  EXPECT_EQ(std::memcmp(&remainder, &expected_remainder, sizeof(remainder)), 0);
}

TEST_F(ExecutionResourceTest, PreservesConstraintGroupsInRemainder) {
  const iree_hal_queue_execution_resource_ordinal_t input_ordinals[] = {0, 1, 2,
                                                                        3, 4};
  hipDevResource input_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(),
      {/*.count=*/IREE_ARRAYSIZE(input_ordinals),
       /*.ordinals=*/input_ordinals},
      hipDevSmResourceGroupDefault, &input_resource));

  hipDevResource untouched_remainder = {};
  unsigned int discovered_partition_count = 99;
  ASSERT_EQ(SplitSmByCount(
                /*out_resources=*/nullptr, &discovered_partition_count,
                &input_resource, &untouched_remainder, /*flags=*/0,
                /*minimum_count=*/4),
            hipSuccess);
  EXPECT_EQ(discovered_partition_count, 1u);

  hipDevResource partitions[2];
  std::memset(partitions, 0xA5, sizeof(partitions));
  const hipDevResource expected_unused_partition = partitions[1];
  hipDevResource remainder;
  unsigned int partition_count = IREE_ARRAYSIZE(partitions);
  ASSERT_EQ(SplitSmByCount(partitions, &partition_count, &input_resource,
                           &remainder, /*flags=*/0, /*minimum_count=*/4),
            hipSuccess);
  ASSERT_EQ(partition_count, 1u);
  EXPECT_EQ(partitions[0].sm.smCount, 4u);
  EXPECT_EQ(remainder.sm.smCount, 6u);
  EXPECT_EQ(std::memcmp(&partitions[1], &expected_unused_partition,
                        sizeof(partitions[1])),
            0);

  const iree_hal_streaming_execution_resource_set_t* partition_set = nullptr;
  ASSERT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &partitions[0], &device_, &partition_set),
            hipSuccess);
  const iree_hal_streaming_execution_resource_set_t* remainder_set = nullptr;
  ASSERT_EQ(iree_hip_execution_resource_resolve_sm_for_device(
                &remainder, &device_, &remainder_set),
            hipSuccess);
  ASSERT_NE(partition_set, nullptr);
  ASSERT_NE(remainder_set, nullptr);
  EXPECT_EQ(partition_set->resources.count + remainder_set->resources.count,
            IREE_ARRAYSIZE(input_ordinals));
}

TEST_F(ExecutionResourceTest, RejectsUnsupportedSplitWithoutPublishing) {
  hipDevResource uniform_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, uniform_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &uniform_resource));
  hipDevResource variable_resource;
  IREE_ASSERT_OK(iree_hip_execution_resource_create_sm(
      &device_, variable_queue_family(), {/*.count=*/0, /*.ordinals=*/nullptr},
      hipDevSmResourceGroupDefault, &variable_resource));

  auto expect_failure_without_publication = [&](const hipDevResource* input,
                                                unsigned int flags,
                                                unsigned int minimum_count,
                                                hipError_t expected_result) {
    hipDevResource partitions[2];
    std::memset(partitions, 0xA5, sizeof(partitions));
    hipDevResource expected_partitions[2];
    std::memcpy(expected_partitions, partitions, sizeof(partitions));
    hipDevResource remainder;
    std::memset(&remainder, 0x5A, sizeof(remainder));
    const hipDevResource expected_remainder = remainder;
    unsigned int partition_count = IREE_ARRAYSIZE(partitions);
    const unsigned int expected_partition_count = partition_count;

    EXPECT_EQ(SplitSmByCount(partitions, &partition_count, input, &remainder,
                             flags, minimum_count),
              expected_result);
    EXPECT_EQ(partition_count, expected_partition_count);
    EXPECT_EQ(std::memcmp(partitions, expected_partitions, sizeof(partitions)),
              0);
    EXPECT_EQ(std::memcmp(&remainder, &expected_remainder, sizeof(remainder)),
              0);
  };

  expect_failure_without_publication(&uniform_resource,
                                     hipDevSmResourceSplitIgnoreSmCoscheduling,
                                     /*minimum_count=*/4, hipErrorNotSupported);
  expect_failure_without_publication(
      &uniform_resource, hipDevSmResourceSplitMaxPotentialClusterSize,
      /*minimum_count=*/4, hipErrorNotSupported);
  expect_failure_without_publication(&uniform_resource, /*flags=*/4,
                                     /*minimum_count=*/4, hipErrorInvalidValue);
  expect_failure_without_publication(
      &uniform_resource, /*flags=*/0,
      /*minimum_count=*/uniform_resource.sm.smCount + 1, hipErrorInvalidValue);
  expect_failure_without_publication(&variable_resource, /*flags=*/0,
                                     /*minimum_count=*/4, hipErrorNotSupported);
}

}  // namespace
}  // namespace iree::hip
