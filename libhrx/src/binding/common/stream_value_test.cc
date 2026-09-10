// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/stream_value.h"

#include "iree/testing/gtest.h"

namespace {

iree_hal_queue_family_spec_t MakeValueWaitFamily(uint32_t queue_count = 1) {
  iree_hal_queue_family_spec_t family = {};
  family.provisioned_queue_count = queue_count;
  family.physical_device_affinity = UINT64_C(1) << 0;
  family.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  family.zero_compute_atomic_capabilities.operations.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.operations.device_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.operations.system_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.operations.system_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT;
  family.zero_compute_atomic_capabilities.wait_conditions.device_scope_32 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.zero_compute_atomic_capabilities.wait_conditions.device_scope_64 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.zero_compute_atomic_capabilities.wait_conditions.system_scope_32 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.zero_compute_atomic_capabilities.wait_conditions.system_scope_64 =
      IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL;
  family.atomic_capabilities = family.zero_compute_atomic_capabilities;
  family.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_DYNAMIC_ACQUISITION;
  return family;
}

TEST(StreamQueueCapabilitiesTest, AcceptsDynamicZeroComputeWaitQueue) {
  const iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  EXPECT_TRUE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresDynamicWaitQueueAcquisition) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily(2);
  family.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresEveryWaitCapability) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();

  family.zero_compute_atomic_capabilities.operations.system_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));

  family = MakeValueWaitFamily();
  family.zero_compute_atomic_capabilities.wait_conditions.system_scope_32 &=
      ~IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NOT_EQUAL;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));

  family = MakeValueWaitFamily();
  family.zero_compute_atomic_capabilities.operations.device_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresOnePhysicalDevicePerFamily) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.physical_device_affinity = (UINT64_C(1) << 0) | (UINT64_C(1) << 1);
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RejectsComputeBackedWaits) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.zero_compute_atomic_capabilities = {};
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, DoesNotRequireHostCallRole) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.role_flags |= IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_HOST_CALL;
  EXPECT_TRUE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresAtomicRole) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily();
  family.role_flags &= ~IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_ATOMIC;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RequiresIndependentProducerLane) {
  iree_hal_queue_family_spec_t family = MakeValueWaitFamily(1);
  family.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_NONE;
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(&family));
}

TEST(StreamQueueCapabilitiesTest, RejectsMissingFamilySpecification) {
  EXPECT_FALSE(iree_hal_streaming_queue_family_supports_value_waits(nullptr));
}

}  // namespace
