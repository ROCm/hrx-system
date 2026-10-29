// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/stream_value.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(StreamValueTest, DeviceScopeDoesNotDependOnPrivateHostVisibility) {
  iree_hal_streaming_buffer_t allocation = {
      .memory_kind = IREE_HAL_STREAMING_MEMORY_KIND_DEVICE,
  };
  iree_hal_streaming_retained_buffer_ref_t target = {
      .owner_wrapper = &allocation,
      .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_NONE,
            iree_hip_stream_value_target_scope(&target));

  // The same device allocation can have a CPU view for internal transfers,
  // including unified backing, without promising concurrent CPU atomics.
  target.memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                        IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                        IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_NONE,
            iree_hip_stream_value_target_scope(&target));
}

TEST(StreamValueTest, HostAndManagedAllocationsRequireSystemScope) {
  iree_hal_streaming_buffer_t allocation = {
      .memory_kind = IREE_HAL_STREAMING_MEMORY_KIND_HOST,
  };
  iree_hal_streaming_retained_buffer_ref_t target = {
      .owner_wrapper = &allocation,
      .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                     IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                     IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                     IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
  };
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
            iree_hip_stream_value_target_scope(&target));
  allocation.memory_kind = IREE_HAL_STREAMING_MEMORY_KIND_MANAGED;
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
            iree_hip_stream_value_target_scope(&target));
}

TEST(StreamValueTest, CrossContextDeviceAllocationsRequireSystemScope) {
  iree_hal_streaming_buffer_t allocation = {
      .memory_kind = IREE_HAL_STREAMING_MEMORY_KIND_DEVICE,
  };
  iree_hal_streaming_retained_buffer_ref_t target = {
      .owner_wrapper = &allocation,
      .memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .is_cross_context = true,
  };
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
            iree_hip_stream_value_target_scope(&target));
}

TEST(StreamValueTest, TranslatesGreaterEqualAndEqual) {
  iree_hal_atomic_wait_params_t params;
  IREE_ASSERT_OK(iree_hip_stream_wait_value_params_initialize(
      UINT64_C(0x100000002), UINT64_C(0x10000000F),
      IREE_HIP_STREAM_WAIT_VALUE_GTE, IREE_HAL_ATOMIC_WIDTH_32, &params));
  EXPECT_EQ(2u, params.value);
  EXPECT_EQ(15u, params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
            params.condition);
  EXPECT_EQ(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
            params.target_error_mode);

  IREE_ASSERT_OK(iree_hip_stream_wait_value_params_initialize(
      7, 15, IREE_HIP_STREAM_WAIT_VALUE_EQ, IREE_HAL_ATOMIC_WIDTH_64, &params));
  EXPECT_EQ(7u, params.value);
  EXPECT_EQ(15u, params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL, params.condition);
}

TEST(StreamValueTest, TranslatesAndPredicate) {
  iree_hal_atomic_wait_params_t params;
  IREE_ASSERT_OK(iree_hip_stream_wait_value_params_initialize(
      UINT64_C(0x35), UINT64_C(0x0F), IREE_HIP_STREAM_WAIT_VALUE_AND,
      IREE_HAL_ATOMIC_WIDTH_64, &params));
  EXPECT_EQ(0u, params.value);
  EXPECT_EQ(5u, params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL, params.condition);
}

TEST(StreamValueTest, TranslatesNorPredicateWithPartialMask) {
  iree_hal_atomic_wait_params_t params;
  IREE_ASSERT_OK(iree_hip_stream_wait_value_params_initialize(
      UINT64_C(0x05), UINT64_C(0x0F), IREE_HIP_STREAM_WAIT_VALUE_NOR,
      IREE_HAL_ATOMIC_WIDTH_64, &params));
  EXPECT_EQ(10u, params.value);
  EXPECT_EQ(10u, params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL, params.condition);
}

TEST(StreamValueTest, RejectsUnknownConditionAndWidth) {
  iree_hal_atomic_wait_params_t params;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_stream_wait_value_params_initialize(
                            0, 0, 4, IREE_HAL_ATOMIC_WIDTH_32, &params));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_stream_wait_value_params_initialize(
                            0, 0, IREE_HIP_STREAM_WAIT_VALUE_EQ,
                            (iree_hal_atomic_width_t)16, &params));
}

TEST(StreamValueTest, TranslatesWriteOperations) {
  iree_hip_stream_value_write_params_t params;
  IREE_ASSERT_OK(iree_hip_stream_write_value_params_initialize(
      UINT64_C(0x100000002), IREE_HIP_STREAM_WRITE_VALUE_DEFAULT,
      IREE_HAL_ATOMIC_WIDTH_32, &params));
  EXPECT_EQ(IREE_HIP_STREAM_VALUE_WRITE_OPERATION_STORE, params.operation);
  EXPECT_EQ(2u, params.store.value);
  EXPECT_EQ(IREE_HAL_ATOMIC_WIDTH_32, params.store.width);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_RELEASE, params.store.flags);
  EXPECT_EQ(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
            params.store.target_error_mode);

  IREE_ASSERT_OK(iree_hip_stream_write_value_params_initialize(
      7, IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT, IREE_HAL_ATOMIC_WIDTH_64,
      &params));
  EXPECT_EQ(IREE_HIP_STREAM_VALUE_WRITE_OPERATION_ADD, params.operation);
  EXPECT_EQ(7u, params.update.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_RMW_OPERATION_ADD, params.update.operation);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
            params.update.flags);
  EXPECT_EQ(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
            params.update.target_error_mode);

  IREE_ASSERT_OK(iree_hip_stream_write_value_params_initialize(
      11, IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT, IREE_HAL_ATOMIC_WIDTH_64,
      &params));
  EXPECT_EQ(IREE_HIP_STREAM_VALUE_WRITE_OPERATION_SUBTRACT, params.operation);
  EXPECT_EQ(11u, params.update.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT, params.update.operation);
  EXPECT_EQ(IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE,
            params.update.target_error_mode);
}

TEST(StreamValueTest, RejectsUnknownWriteOperationAndWidth) {
  iree_hip_stream_value_write_params_t params;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_stream_write_value_params_initialize(
                            0, 1, IREE_HAL_ATOMIC_WIDTH_32, &params));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hip_stream_write_value_params_initialize(
                            0, IREE_HIP_STREAM_WRITE_VALUE_DEFAULT,
                            (iree_hal_atomic_width_t)16, &params));
}

}  // namespace
