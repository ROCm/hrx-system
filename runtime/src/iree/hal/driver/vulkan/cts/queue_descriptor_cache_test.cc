// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"

namespace iree::hal::cts {

class VulkanQueueDescriptorCacheTest : public CtsTestBase<> {};

TEST_P(VulkanQueueDescriptorCacheTest, DeferredUploadsOverlapCallerRecording) {
  constexpr iree_host_size_t kUploadCount = 128;
  std::vector<uint32_t> source(kUploadCount);
  for (iree_host_size_t i = 0; i < source.size(); ++i) {
    source[i] = 0xABCD0000u + static_cast<uint32_t>(i);
  }
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(
      2 * kUploadCount * sizeof(uint32_t), target.out()));
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList signals(device_, std::vector<uint64_t>(2 * kUploadCount, 0),
                        std::vector<uint64_t>(2 * kUploadCount, 1));

  // Releasing the first batch permits completion callbacks to record staged
  // copies while the second batch records on this thread, reusing native pools.
  iree_status_t status = iree_ok_status();
  iree_host_size_t submitted_count = 0;
  for (iree_host_size_t batch = 0; batch < 2 && iree_status_is_ok(status);
       ++batch) {
    for (iree_host_size_t i = 0; i < kUploadCount && iree_status_is_ok(status);
         ++i) {
      const iree_host_size_t ordinal = batch * kUploadCount + i;
      const iree_hal_semaphore_list_t signal = {
          .count = 1,
          .semaphores = &signals.semaphores[ordinal],
          .payload_values = &signals.payload_values[ordinal],
      };
      status = iree_hal_queue_upload(
          transfer_queue_,
          batch == 0 ? static_cast<iree_hal_semaphore_list_t>(gate)
                     : iree_hal_semaphore_list_empty(),
          signal, &source[i], target, ordinal * sizeof(uint32_t),
          sizeof(uint32_t), /*barriers=*/nullptr);
      if (iree_status_is_ok(status)) {
        ++submitted_count;
      }
    }
    if (batch == 0) {
      // Admitted uploads must complete before their source storage is released,
      // including when a later submission in this batch fails.
      status = iree_status_join(status,
                                iree_hal_semaphore_list_signal(gate, nullptr));
    }
  }
  const iree_hal_semaphore_list_t submitted = {
      .count = submitted_count,
      .semaphores = signals.semaphores.data(),
      .payload_values = signals.payload_values.data(),
  };
  status = iree_status_join(
      status, iree_hal_semaphore_list_wait(submitted, iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(status);
  std::vector<uint32_t> expected = source;
  expected.insert(expected.end(), source.begin(), source.end());
  EXPECT_EQ(ReadBufferData<uint32_t>(target), expected);
}

TEST_P(VulkanQueueDescriptorCacheTest, DeferredUnalignedFillsExceedOneBlock) {
  constexpr iree_host_size_t kSubmissionCount = 5000;
  constexpr uint8_t kPattern = 0x5A;

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kSubmissionCount, target_buffer.out()));

  SemaphoreList gate(device_, {0}, {1});
  std::vector<uint64_t> initial_values(kSubmissionCount, 0);
  std::vector<uint64_t> payload_values(kSubmissionCount, 1);
  SemaphoreList signals(device_, std::move(initial_values),
                        std::move(payload_values));

  for (iree_host_size_t i = 0; i < kSubmissionCount; ++i) {
    iree_hal_semaphore_list_t signal_list = {
        .count = 1,
        .semaphores = &signals.semaphores[i],
        .payload_values = &signals.payload_values[i],
    };
    IREE_ASSERT_OK(iree_hal_queue_fill(
        transfer_queue_, gate, signal_list, target_buffer.get(), i,
        /*length=*/1, &kPattern, sizeof(kPattern), /*barriers=*/NULL,
        IREE_HAL_FILL_FLAG_NONE));
  }

  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(gate.semaphores[0], 1, /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(signals, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<uint8_t> bytes =
      ReadBufferBytes(target_buffer.get(), /*offset=*/0, kSubmissionCount);
  ASSERT_EQ(kSubmissionCount, bytes.size());
  for (iree_host_size_t i = 0; i < bytes.size(); ++i) {
    EXPECT_EQ(kPattern, bytes[i]) << "byte offset " << i;
  }
}

TEST_P(VulkanQueueDescriptorCacheTest,
       TransferOnlyFillScratchSurvivesConcurrentUseAndTrim) {
  const iree_hal_device_queue_spec_t* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device_));
  iree_hal_queue_t* transfer_only_queue = nullptr;
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    const auto& family = queues->families[i];
    if (family.provisioned_queue_count &&
        iree_any_bit_set(family.role_flags,
                         IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER) &&
        !iree_any_bit_set(family.role_flags,
                          IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH)) {
      transfer_only_queue = iree_hal_device_queue(device_, i, 0);
      break;
    }
  }
  if (!transfer_only_queue) {
    GTEST_SKIP() << "device has no provisioned transfer-only queue";
  }
  transfer_queue_ = transfer_only_queue;

  constexpr iree_host_size_t kSubmissionCount = 3 * 4 * 17;
  constexpr iree_device_size_t kStride = 128;
  constexpr uint8_t kGuard = 0xCD;
  std::vector<uint8_t> expected(kSubmissionCount * kStride, kGuard);
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateDeviceBufferWithData(expected.data(), expected.size(),
                                            target.out()));

  for (uint32_t round = 0; round < 3; ++round) {
    SCOPED_TRACE(round);
    SemaphoreList gate(device_, {0}, {1});
    SemaphoreList signals(device_, std::vector<uint64_t>(kSubmissionCount, 0),
                          std::vector<uint64_t>(kSubmissionCount, 1));
    iree_status_t status = iree_ok_status();
    iree_host_size_t submitted_count = 0;
    for (iree_host_size_t i = 0;
         i < kSubmissionCount && iree_status_is_ok(status); ++i) {
      const iree_host_size_t pattern_length = iree_host_size_t{1}
                                              << (i / (4 * 17));
      const iree_device_size_t offset =
          i * kStride + ((i / 17) % 4) * pattern_length;
      const iree_device_size_t length = (1 + i % 17) * pattern_length;
      const uint8_t pattern[] = {static_cast<uint8_t>(0x21 + i + round), 0x87,
                                 0xDA, 0x4B};
      const iree_hal_semaphore_list_t signal = {
          .count = 1,
          .semaphores = &signals.semaphores[i],
          .payload_values = &signals.payload_values[i],
      };
      status =
          iree_hal_queue_fill(transfer_only_queue, gate, signal, target, offset,
                              length, pattern, pattern_length,
                              /*barriers=*/nullptr, IREE_HAL_FILL_FLAG_NONE);
      if (iree_status_is_ok(status)) {
        ++submitted_count;
        for (iree_device_size_t j = 0; j < length; ++j) {
          expected[offset + j] = pattern[j % pattern_length];
        }
      }
    }
    // Release admitted work even when a later submission fails.
    status = iree_status_join(
        status, iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
    const iree_hal_semaphore_list_t submitted = {
        .count = submitted_count,
        .semaphores = signals.semaphores.data(),
        .payload_values = signals.payload_values.data(),
    };
    status = iree_status_join(
        status, iree_hal_semaphore_list_wait(submitted, iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_ASSERT_OK(status);
    EXPECT_EQ(ReadBufferBytes(target, 0, expected.size()), expected);
    if (round == 1) {
      IREE_ASSERT_OK(iree_hal_device_trim(device_));
    }
  }
}

CTS_REGISTER_TEST_SUITE(VulkanQueueDescriptorCacheTest);

}  // namespace iree::hal::cts
