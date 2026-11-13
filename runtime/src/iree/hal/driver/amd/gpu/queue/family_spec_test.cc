// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/family_spec.h"

#include <tuple>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class FamilySpecTest
    : public ::testing::TestWithParam<std::tuple<
          amdf_queue_command_type_t, amdf_queue_publication_modes_t>> {
 protected:
  void SetUp() override {
    context_.api = &api_;
    context_.gpu = &gpu_;
    context_.families = families_;
    context_.target_info.compute.compute_unit_count = 40;
    context_.target_info.execution_resources = {
        .compute_units_per_resource = 2,
        .group_count = 2,
        .minimum_selected_resources_per_group = 1,
    };
    families_[1].command_type = std::get<0>(GetParam());
    families_[1].user_queue_capabilities =
        AMDF_USER_QUEUE_CAPABILITY_EXECUTION_RESOURCE_SELECTION;
    families_[1].priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_LOW |
                                         AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL |
                                         AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH;
    gpu_.extension_version = AMDF_GPU_EXTENSION_VERSION_2;
    gpu_.device_query_queue_clock_info =
        [](amdf_device_t*, uint32_t ordinal,
           amdf_observation_clock_info_t* out) {
          EXPECT_EQ(ordinal, 1u);
          out->frequency_hz = 100000000;
          return AMDF_STATUS_OK;
        };
    api_.device_query_queue_family_placement_info =
        [](amdf_device_t*, uint32_t ordinal,
           amdf_queue_family_placement_info_t* out) {
          EXPECT_EQ(ordinal, 1u);
          out->placement_capabilities =
              AMDF_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL |
              AMDF_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL;
          return AMDF_STATUS_OK;
        };
  }

  void TearDown() override {
    iree_hal_amd_gpu_queue_family_spec_deinitialize(&storage_,
                                                    iree_allocator_system());
  }

  iree_status_t Initialize() {
    return iree_hal_amd_gpu_queue_family_spec_initialize(
        &context_, {11, 0, 0}, 1, std::get<1>(GetParam()), 0,
        iree_allocator_system(), &storage_, &spec_);
  }

  // Native facts are dependencies; the real translator owns public metadata.
  iree_hal_amd_gpu_context_t context_ = {};
  // Placement query table for this native device.
  amdf_api_t api_ = {};
  // Native clock query table.
  amdf_gpu_api_t gpu_ = {};
  // A nonzero native ordinal detects accidental compute-family substitution.
  amdf_queue_family_info_t families_[2] = {};
  // Temporary metadata retained until the public builder copies it.
  iree_hal_amd_gpu_queue_family_spec_storage_t storage_ = {};
  // Actual translated capabilities, inspected before releasing storage.
  iree_hal_queue_family_spec_t spec_ = {};
};

TEST_P(FamilySpecTest, ExposesOnlyImplementedOperationsAndResourceSelections) {
  IREE_ASSERT_OK(Initialize());
  const bool compute =
      std::get<0>(GetParam()) != AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA;
  const bool user = std::get<1>(GetParam()) == AMDF_QUEUE_PUBLICATION_MODE_USER;
  EXPECT_EQ(spec_.provisioned_queue_count, 0u);
  EXPECT_NE(spec_.flags & IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION,
            0u);
  EXPECT_EQ(spec_.role_flags,
            IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL |
                (compute ? IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH |
                               IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC
                         : 0));
  EXPECT_EQ(spec_.execution_unit_count, compute ? 40u : 0u);
  EXPECT_EQ(spec_.execution_resource_count, compute ? (user ? 20u : 1u) : 0u);
  EXPECT_EQ(spec_.execution_resource_group_count,
            compute ? (user ? 2u : 1u) : 0u);
  EXPECT_EQ(spec_.atomic_capabilities.operations.device_scope_32,
            compute ? IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL : 0u);
  EXPECT_EQ(spec_.priority_count, user ? 3u : 1u);
  EXPECT_EQ(spec_.priorities[0], user ? -1 : 0);
  EXPECT_EQ(spec_.placement_capabilities,
            user ? IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL |
                       IREE_HAL_QUEUE_PLACEMENT_CAPABILITY_DEVICE_LOCAL
                 : 0u);
  EXPECT_EQ(spec_.timestamp_frequency_hz, 100000000u);
  EXPECT_EQ(spec_.timestamp_valid_bits, 64u);
}

TEST_P(FamilySpecTest, CooperativeAdmissionRequiresIndependentGridState) {
  families_[1].user_queue_capabilities |=
      AMDF_USER_QUEUE_CAPABILITY_COOPERATIVE_DISPATCH;
  IREE_ASSERT_OK(Initialize());
  const bool supported =
      std::get<0>(GetParam()) == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
      std::get<1>(GetParam()) == AMDF_QUEUE_PUBLICATION_MODE_USER;
  EXPECT_EQ(spec_.supported_queue_features,
            supported ? IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH : 0u);
}

TEST_P(FamilySpecTest, MissingNativeClockDoesNotAdvertiseTimestamps) {
  gpu_.device_query_queue_clock_info = [](amdf_device_t*, uint32_t,
                                          amdf_observation_clock_info_t*) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  };
  IREE_ASSERT_OK(Initialize());
  EXPECT_EQ(spec_.timestamp_frequency_hz, 0u);
  EXPECT_EQ(spec_.timestamp_valid_bits, 0u);
  EXPECT_EQ(spec_.flags &
                (IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS |
                 IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS),
            0u);
}

INSTANTIATE_TEST_SUITE_P(
    Protocols, FamilySpecTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA),
                       ::testing::Values(AMDF_QUEUE_PUBLICATION_MODE_USER,
                                         AMDF_QUEUE_PUBLICATION_MODE_KERNEL)));

}  // namespace
