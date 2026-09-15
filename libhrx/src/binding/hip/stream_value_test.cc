// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/stream_value.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(StreamValueTest, TranslatesGreaterEqualAndEqual) {
  iree_hal_atomic_wait_params_t params;
  IREE_ASSERT_OK(iree_hip_stream_wait_value_params_initialize(
      UINT64_C(0x100000002), UINT64_C(0x10000000F),
      IREE_HIP_STREAM_WAIT_VALUE_GTE, IREE_HAL_ATOMIC_WIDTH_32, &params));
  EXPECT_EQ(2u, params.value);
  EXPECT_EQ(15u, params.mask);
  EXPECT_EQ(IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
            params.condition);

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

  IREE_ASSERT_OK(iree_hip_stream_write_value_params_initialize(
      7, IREE_HIP_EXT_STREAM_WRITE_VALUE_INCREMENT, IREE_HAL_ATOMIC_WIDTH_64,
      &params));
  EXPECT_EQ(IREE_HIP_STREAM_VALUE_WRITE_OPERATION_ADD, params.operation);
  EXPECT_EQ(7u, params.update.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_RMW_OPERATION_ADD, params.update.operation);
  EXPECT_EQ(IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE,
            params.update.flags);

  IREE_ASSERT_OK(iree_hip_stream_write_value_params_initialize(
      11, IREE_HIP_EXT_STREAM_WRITE_VALUE_DECREMENT, IREE_HAL_ATOMIC_WIDTH_64,
      &params));
  EXPECT_EQ(IREE_HIP_STREAM_VALUE_WRITE_OPERATION_SUBTRACT, params.operation);
  EXPECT_EQ(11u, params.update.operand);
  EXPECT_EQ(IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT, params.update.operation);
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
