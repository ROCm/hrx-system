// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>

#include "iree/hal/cts/util/atomic_test_util.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class QueueAtomicTest : public CtsTestBase<> {
 protected:
  static constexpr iree_device_size_t kBufferSize = 32;
  static constexpr iree_device_size_t kTargetOffset = 8;

  bool SelectConfiguration(
      iree_hal_atomic_width_t width,
      iree_hal_atomic_operation_flags_t operation_flags,
      iree_hal_atomic_wait_condition_flags_t wait_condition_flags,
      AtomicTestConfiguration* out_configuration) {
    const AtomicTestRequirements requirements = {
        /*.operation_flags=*/operation_flags,
        /*.wait_condition_flags=*/wait_condition_flags,
        /*.memory_type=*/IREE_HAL_MEMORY_TYPE_NONE,
        /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_STORAGE |
            IREE_HAL_BUFFER_USAGE_TRANSFER,
        /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_READ |
            IREE_HAL_MEMORY_ACCESS_WRITE,
        /*.width=*/width,
        /*.atomic_flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE |
            IREE_HAL_ATOMIC_FLAG_RELEASE,
    };
    return SelectAtomicTestConfiguration(iree_hal_device_spec(device_),
                                         requirements, out_configuration);
  }

  iree_status_t AllocateAtomicBuffer(
      const AtomicTestConfiguration& configuration,
      iree_hal_buffer_t** out_buffer) {
    return iree_hal_allocator_allocate_buffer(device_allocator_,
                                              configuration.buffer_params,
                                              kBufferSize, out_buffer);
  }

  iree_status_t QueueStoreAndWait(const AtomicTestConfiguration& configuration,
                                  iree_hal_buffer_t* buffer,
                                  iree_hal_atomic_store_params_t params) {
    SemaphoreList empty_wait;
    SemaphoreList signal(device_, {0}, {1});
    iree_status_t status = iree_hal_device_queue_atomic_store(
        device_, configuration.queue_affinity, empty_wait, signal, buffer,
        kTargetOffset, params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    return status;
  }

  iree_status_t QueueRmwAndWait(const AtomicTestConfiguration& configuration,
                                iree_hal_buffer_t* buffer,
                                iree_hal_atomic_rmw_params_t params) {
    SemaphoreList empty_wait;
    SemaphoreList signal(device_, {0}, {1});
    iree_status_t status = iree_hal_device_queue_atomic_rmw(
        device_, configuration.queue_affinity, empty_wait, signal, buffer,
        kTargetOffset, params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    return status;
  }

  iree_status_t QueueWaitAndWait(const AtomicTestConfiguration& configuration,
                                 iree_hal_buffer_t* buffer,
                                 iree_hal_atomic_wait_params_t params) {
    SemaphoreList empty_wait;
    SemaphoreList signal(device_, {0}, {1});
    iree_status_t status = iree_hal_device_queue_atomic_wait(
        device_, configuration.queue_affinity, empty_wait, signal, buffer,
        kTargetOffset, params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(signal, iree_infinite_timeout(),
                                            IREE_ASYNC_WAIT_FLAG_NONE);
    }
    return status;
  }

  template <typename ValueType>
  ValueType ReadAtomicValue(const AtomicTestConfiguration& configuration,
                            iree_hal_buffer_t* buffer) {
    const std::vector<uint8_t> bytes = ReadBufferBytes(
        buffer, kTargetOffset, sizeof(ValueType), configuration.queue_affinity);
    ValueType value = 0;
    EXPECT_EQ(bytes.size(), sizeof(value));
    if (bytes.size() == sizeof(value)) {
      memcpy(&value, bytes.data(), sizeof(value));
    }
    return value;
  }

  template <typename ValueType>
  void RunStoreAndRmwTest(iree_hal_atomic_width_t width) {
    const iree_hal_atomic_operation_flags_t operation_flags =
        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_ADD |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_SUBTRACT |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_AND |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR |
        IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_XOR;
    AtomicTestConfiguration configuration;
    if (!SelectConfiguration(width, operation_flags,
                             IREE_HAL_ATOMIC_WAIT_CONDITION_FLAG_NONE,
                             &configuration)) {
      GTEST_SKIP() << "Device does not advertise the tested "
                      "queue/memory atomic operation set";
    }

    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, buffer.out()));

    const iree_hal_atomic_flags_t atomic_flags =
        IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE;
    const iree_hal_atomic_store_params_t store_params = {
        /*.value=*/10,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
    };
    IREE_ASSERT_OK(QueueStoreAndWait(configuration, buffer, store_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(10));

    iree_hal_atomic_rmw_params_t rmw_params = {
        /*.operand=*/5,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
        /*.operation=*/IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
    };
    IREE_ASSERT_OK(QueueRmwAndWait(configuration, buffer, rmw_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(15));

    rmw_params.operand = 3;
    rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT;
    IREE_ASSERT_OK(QueueRmwAndWait(configuration, buffer, rmw_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(12));

    rmw_params.operand = 10;
    rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_AND;
    IREE_ASSERT_OK(QueueRmwAndWait(configuration, buffer, rmw_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(8));

    rmw_params.operand = 3;
    rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_OR;
    IREE_ASSERT_OK(QueueRmwAndWait(configuration, buffer, rmw_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(11));

    rmw_params.operand = 15;
    rmw_params.operation = IREE_HAL_ATOMIC_RMW_OPERATION_XOR;
    IREE_ASSERT_OK(QueueRmwAndWait(configuration, buffer, rmw_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(4));
  }

  template <typename ValueType>
  void RunWaitTest(iree_hal_atomic_width_t width) {
    const iree_hal_atomic_operation_flags_t operation_flags =
        IREE_HAL_ATOMIC_OPERATION_FLAG_WAIT |
        IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
    AtomicTestConfiguration configuration;
    if (!SelectConfiguration(width, operation_flags,
                             IREE_HAL_ATOMIC_WAIT_CONDITION_FLAGS_ALL,
                             &configuration)) {
      GTEST_SKIP() << "Device does not advertise the tested "
                      "queue/memory atomic wait set";
    }

    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(AllocateAtomicBuffer(configuration, buffer.out()));
    const iree_hal_atomic_flags_t atomic_flags =
        IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE;
    const iree_hal_atomic_store_params_t store_params = {
        /*.value=*/0x12,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
    };
    IREE_ASSERT_OK(QueueStoreAndWait(configuration, buffer, store_params));

    iree_hal_atomic_wait_params_t wait_params = {
        /*.value=*/0x2,
        /*.mask=*/0xF,
        /*.flags=*/atomic_flags,
        /*.width=*/width,
        /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
    };
    IREE_ASSERT_OK(QueueWaitAndWait(configuration, buffer, wait_params));

    wait_params.value = 0x3;
    wait_params.condition = IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL;
    IREE_ASSERT_OK(QueueWaitAndWait(configuration, buffer, wait_params));

    wait_params.value = 0x10;
    wait_params.mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    wait_params.condition =
        IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
    IREE_ASSERT_OK(QueueWaitAndWait(configuration, buffer, wait_params));
    EXPECT_EQ(ReadAtomicValue<ValueType>(configuration, buffer),
              static_cast<ValueType>(0x12));
  }
};

TEST_P(QueueAtomicTest, StoreAndRmw32) {
  RunStoreAndRmwTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
}

TEST_P(QueueAtomicTest, StoreAndRmw64) {
  RunStoreAndRmwTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
}

TEST_P(QueueAtomicTest, Wait32) {
  RunWaitTest<uint32_t>(IREE_HAL_ATOMIC_WIDTH_32);
}

TEST_P(QueueAtomicTest, Wait64) {
  RunWaitTest<uint64_t>(IREE_HAL_ATOMIC_WIDTH_64);
}

CTS_REGISTER_TEST_SUITE(QueueAtomicTest);

}  // namespace iree::hal::cts
