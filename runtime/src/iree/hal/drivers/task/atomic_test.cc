// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/task/atomic.h"

#include <thread>

#include "iree/base/internal/atomics.h"
#include "iree/testing/gtest.h"

namespace {

TEST(LocalAtomicTest, CapabilitiesFollowLockFreeWidths) {
  const iree_hal_atomic_operation_flags_t allowed_operations =
      IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD;
  const iree_hal_atomic_capabilities_t capabilities =
      iree_hal_task_atomic_capabilities(allowed_operations);

  const iree_hal_atomic_operation_flags_t expected_32 =
      iree_atomic_int32_is_lock_free() ? allowed_operations : 0;
  EXPECT_EQ(capabilities.operations.device_scope_32, expected_32);
  EXPECT_EQ(capabilities.operations.system_scope_32, expected_32);

  const iree_hal_atomic_operation_flags_t expected_64 =
      iree_atomic_int64_is_lock_free() ? allowed_operations : 0;
  EXPECT_EQ(capabilities.operations.device_scope_64, expected_64);
  EXPECT_EQ(capabilities.operations.system_scope_64, expected_64);

  EXPECT_EQ(capabilities.wait_conditions.device_scope_32, 0u);
  EXPECT_EQ(capabilities.wait_conditions.device_scope_64, 0u);
  EXPECT_EQ(capabilities.wait_conditions.system_scope_32, 0u);
  EXPECT_EQ(capabilities.wait_conditions.system_scope_64, 0u);

  const iree_hal_atomic_capabilities_t wait_capabilities =
      iree_hal_task_atomic_capabilities(IREE_HAL_ATOMIC_OPERATION_FLAGS_ALL);
  const iree_hal_atomic_wait_condition_flags_t expected_wait_32 =
      iree_atomic_int32_is_lock_free()
          ? IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL
          : 0;
  EXPECT_EQ(wait_capabilities.wait_conditions.device_scope_32,
            expected_wait_32);
  EXPECT_EQ(wait_capabilities.wait_conditions.system_scope_32,
            expected_wait_32);
  const iree_hal_atomic_wait_condition_flags_t expected_wait_64 =
      iree_atomic_int64_is_lock_free()
          ? IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL
          : 0;
  EXPECT_EQ(wait_capabilities.wait_conditions.device_scope_64,
            expected_wait_64);
  EXPECT_EQ(wait_capabilities.wait_conditions.system_scope_64,
            expected_wait_64);

  EXPECT_FALSE(iree_hal_task_atomic_width_is_lock_free(
      static_cast<iree_hal_atomic_width_t>(16)));
}

template <typename AtomicType, typename ValueType>
static void TestStoreAndRmw(iree_hal_atomic_width_t width) {
  AtomicType target = IREE_ATOMIC_VAR_INIT(0);

  iree_hal_atomic_store_params_t store_params = {};
  store_params.value = 10;
  store_params.flags = IREE_HAL_ATOMIC_FLAG_RELEASE;
  store_params.width = width;
  iree_hal_task_atomic_store(&target, store_params);
  EXPECT_EQ(iree_atomic_load(&target, iree_memory_order_acquire),
            static_cast<ValueType>(10));

  iree_hal_atomic_rmw_params_t rmw_params = {};
  rmw_params.flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                     IREE_HAL_ATOMIC_FLAG_RELEASE |
                     IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE;
  rmw_params.width = width;

  rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_ADD;
  rmw_params.operand = 5;
  iree_hal_task_atomic_rmw(&target, rmw_params);
  EXPECT_EQ(iree_atomic_load(&target, iree_memory_order_acquire),
            static_cast<ValueType>(15));

  rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT;
  rmw_params.operand = 3;
  iree_hal_task_atomic_rmw(&target, rmw_params);
  EXPECT_EQ(iree_atomic_load(&target, iree_memory_order_acquire),
            static_cast<ValueType>(12));

  rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_AND;
  rmw_params.operand = 10;
  iree_hal_task_atomic_rmw(&target, rmw_params);
  EXPECT_EQ(iree_atomic_load(&target, iree_memory_order_acquire),
            static_cast<ValueType>(8));

  rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_OR;
  rmw_params.operand = 3;
  iree_hal_task_atomic_rmw(&target, rmw_params);
  EXPECT_EQ(iree_atomic_load(&target, iree_memory_order_acquire),
            static_cast<ValueType>(11));

  rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_XOR;
  rmw_params.operand = 15;
  iree_hal_task_atomic_rmw(&target, rmw_params);
  EXPECT_EQ(iree_atomic_load(&target, iree_memory_order_acquire),
            static_cast<ValueType>(4));
}

template <typename AtomicType, typename ValueType>
static void TestWait(iree_hal_atomic_width_t width) {
  AtomicType target = IREE_ATOMIC_VAR_INIT(0x12);

  iree_hal_atomic_wait_params_t wait_params = {};
  wait_params.value = 0x2;
  wait_params.mask = 0xF;
  wait_params.flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE;
  wait_params.width = width;
  wait_params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL;
  iree_hal_task_atomic_wait(&target, wait_params);

  wait_params.value = 0x3;
  wait_params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
  iree_hal_task_atomic_wait(&target, wait_params);

  iree_atomic_store(&target, static_cast<ValueType>(0),
                    iree_memory_order_relaxed);
  wait_params.value = 9;
  wait_params.mask =
      width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
  wait_params.flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                      IREE_HAL_ATOMIC_FLAG_RELEASE |
                      IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE;
  wait_params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
  std::thread producer([&target]() {
    iree_atomic_store(&target, static_cast<ValueType>(9),
                      iree_memory_order_release);
  });
  iree_hal_task_atomic_wait(&target, wait_params);
  producer.join();
}

TEST(LocalAtomicTest, Operations32) {
  if (!iree_atomic_int32_is_lock_free()) {
    GTEST_SKIP();
  }
  TestStoreAndRmw<iree_atomic_uint32_t, uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
  TestWait<iree_atomic_uint32_t, uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
}

TEST(LocalAtomicTest, Operations64) {
  if (!iree_atomic_int64_is_lock_free()) {
    GTEST_SKIP();
  }
  TestStoreAndRmw<iree_atomic_uint64_t, uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
  TestWait<iree_atomic_uint64_t, uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
}

}  // namespace
