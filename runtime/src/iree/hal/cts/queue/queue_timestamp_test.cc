// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class QueueTimestampTest : public CtsTestBase<> {
 protected:
  bool SelectTimestampQueue(iree_hal_queue_affinity_t* out_queue_affinity,
                            uint32_t* out_timestamp_valid_bits) {
    const iree_hal_device_queue_spec_t* queue_spec =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    for (iree_host_size_t i = 0; i < queue_spec->family_count; ++i) {
      const iree_hal_queue_family_spec_t* family = &queue_spec->families[i];
      if (!iree_all_bits_set(family->role_flags,
                             IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_PROFILING) ||
          family->timestamp_valid_bits == 0 ||
          family->timestamp_frequency_hz == 0 ||
          iree_hal_queue_affinity_is_empty(family->queue_affinity)) {
        continue;
      }
      *out_queue_affinity = 1ull << iree_hal_queue_affinity_find_first_set(
                                family->queue_affinity);
      *out_timestamp_valid_bits = family->timestamp_valid_bits;
      return true;
    }
    return false;
  }
};

TEST_P(QueueTimestampTest, RejectsInvalidRequestBeforeBackendDispatch) {
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  buffer_params.access = IREE_HAL_MEMORY_ACCESS_WRITE;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.min_alignment = sizeof(uint64_t);
  Ref<iree_hal_buffer_t> timestamp_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, buffer_params,
      /*allocation_size=*/2 * sizeof(uint64_t), timestamp_buffer.out()));

  SemaphoreList empty_list;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_device_queue_timestamp(
                            device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_list,
                            empty_list, timestamp_buffer, /*target_offset=*/0,
                            static_cast<iree_hal_timestamp_flags_t>(1)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_timestamp(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, empty_list, empty_list,
          timestamp_buffer, /*target_offset=*/4, IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_device_queue_timestamp(device_, IREE_HAL_QUEUE_AFFINITY_ANY,
                                      empty_list, empty_list, timestamp_buffer,
                                      /*target_offset=*/2 * sizeof(uint64_t),
                                      IREE_HAL_TIMESTAMP_FLAG_NONE));
}

TEST_P(QueueTimestampTest, CapturesIncreasingDeviceTicks) {
  iree_hal_queue_affinity_t queue_affinity = 0;
  uint32_t timestamp_valid_bits = 0;
  if (!SelectTimestampQueue(&queue_affinity, &timestamp_valid_bits)) {
    GTEST_SKIP() << "Device does not advertise queue timestamp capture";
  }

  Ref<iree_hal_buffer_t> timestamp_buffer;
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.access =
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  buffer_params.queue_affinity = queue_affinity;
  buffer_params.min_alignment = sizeof(uint64_t);
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_, buffer_params,
      /*allocation_size=*/2 * sizeof(uint64_t), timestamp_buffer.out()));

  SemaphoreList empty_wait;
  SemaphoreList fill_signal(device_, {0}, {1});
  const uint32_t zero = 0;
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, queue_affinity, empty_wait, fill_signal, timestamp_buffer,
      /*target_offset=*/0, /*length=*/2 * sizeof(uint64_t), &zero, sizeof(zero),
      IREE_HAL_FILL_FLAG_NONE));

  SemaphoreList first_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device_, queue_affinity, fill_signal, first_signal, timestamp_buffer,
      /*target_offset=*/0, IREE_HAL_TIMESTAMP_FLAG_NONE));

  SemaphoreList second_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_timestamp(
      device_, queue_affinity, first_signal, second_signal, timestamp_buffer,
      /*target_offset=*/sizeof(uint64_t), IREE_HAL_TIMESTAMP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      second_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  const std::vector<uint64_t> ticks =
      ReadBufferData<uint64_t>(timestamp_buffer);
  ASSERT_EQ(ticks.size(), 2u);
  const uint64_t valid_mask = timestamp_valid_bits >= 64
                                  ? UINT64_MAX
                                  : (UINT64_C(1) << timestamp_valid_bits) - 1;
  const uint64_t elapsed_ticks = (ticks[1] - ticks[0]) & valid_mask;
  EXPECT_GT(elapsed_ticks, 0u);
}

CTS_REGISTER_TEST_SUITE(QueueTimestampTest);

}  // namespace iree::hal::cts
