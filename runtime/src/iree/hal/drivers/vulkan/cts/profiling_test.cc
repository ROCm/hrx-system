// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/passthrough_pool.h"

namespace iree::hal::cts {

class VulkanProfilingTest : public CtsTestBase<> {};

TEST_P(VulkanProfilingTest, QueueEventsRecordNativeTransferSubmissions) {
  constexpr iree_device_size_t kBufferSize = 128;

  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, source.out()));
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(kBufferSize, target.out()));

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  DeviceProfilingScope profiling(device_);
  IREE_ASSERT_OK(profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
                                 TestProfileSinkAsBase(&sink)));

  SemaphoreList empty_wait;
  SemaphoreList fill_signal(device_, {0}, {1});
  uint32_t pattern = 0xA5A5A5A5u;
  IREE_ASSERT_OK(iree_hal_queue_fill(transfer_queue_, empty_wait, fill_signal,
                                     source, 0, kBufferSize, &pattern,
                                     sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_copy(transfer_queue_, fill_signal, copy_signal,
                                     source, 0, target, 0, kBufferSize,
                                     IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.queue_event_count, 1);
  EXPECT_EQ(0, sink.host_execution_event_count);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_EQ(0, sink.queue_device_event_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);

  auto find_queue_event = [&](iree_hal_profile_queue_event_type_t type) {
    return std::find_if(sink.queue_events.begin(), sink.queue_events.end(),
                        [type](const iree_hal_profile_queue_event_t& event) {
                          return event.type == type;
                        });
  };

  auto fill_it = find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL);
  ASSERT_NE(sink.queue_events.end(), fill_it);
  EXPECT_EQ(kBufferSize, fill_it->payload_length);
  EXPECT_EQ(1u, fill_it->operation_count);
  EXPECT_NE(0u, fill_it->submission_id);
  EXPECT_GE(fill_it->ready_host_time_ns, fill_it->host_time_ns);

  auto copy_it = find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY);
  ASSERT_NE(sink.queue_events.end(), copy_it);
  EXPECT_EQ(kBufferSize, copy_it->payload_length);
  EXPECT_EQ(1u, copy_it->operation_count);
  EXPECT_NE(0u, copy_it->submission_id);
  EXPECT_GE(copy_it->ready_host_time_ns, copy_it->host_time_ns);
}

TEST_P(VulkanProfilingTest,
       LightweightStatisticsRecordQueueAllocaMemoryEvents) {
  constexpr iree_device_size_t kBufferSize = 1024;

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);

  iree_hal_device_profiling_options_t options = {0};
  options.flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  options.sink = TestProfileSinkAsBase(&sink);

  DeviceProfilingScope profiling(device_);
  IREE_ASSERT_OK(profiling.Begin(&options));

  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
  params.queue_family_affinity = iree_hal_make_queue_family_affinity(
      iree_hal_queue_family_ordinal(iree_hal_queue_family(transfer_queue_)));

  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_device_query_queue_pool_backend(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, &backend));
  iree_hal_passthrough_pool_options_t pool_options = {};
  pool_options.asan = backend.asan;
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
      pool_options, backend.slab_provider, backend.notification,
      iree_allocator_system(), pool.out()));
  const iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = kBufferSize,
  };

  Ref<iree_hal_buffer_t> buffer;
  SemaphoreList empty_wait;
  SemaphoreList alloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* raw_buffer = NULL;
  IREE_ASSERT_OK(
      iree_hal_queue_alloca(transfer_queue_, empty_wait, alloca_signal, pool,
                            /*request_count=*/1, &request, &raw_buffer));
  buffer.reset(raw_buffer);

  SemaphoreList dealloca_signal(device_, {0}, {1});
  iree_hal_buffer_t* dealloca_buffer = buffer.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, alloca_signal,
                                         dealloca_signal,
                                         /*buffer_count=*/1, &dealloca_buffer));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dealloca_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_EQ(1, sink.begin_count);
  EXPECT_EQ(1, sink.end_count);
  EXPECT_EQ(1, sink.device_metadata_count);
  EXPECT_EQ(1, sink.queue_metadata_count);
  EXPECT_GE(sink.queue_event_count, 1);
  EXPECT_GE(sink.memory_event_count, 1);
  EXPECT_EQ(0, sink.host_execution_event_count);
  EXPECT_EQ(0, sink.dispatch_event_count);
  EXPECT_EQ(0, sink.queue_device_event_count);
  EXPECT_TRUE(sink.saw_device_metadata);
  EXPECT_TRUE(sink.saw_queue_metadata);
  EXPECT_FALSE(sink.write_after_end);

  auto find_queue_event = [&](iree_hal_profile_queue_event_type_t type) {
    return std::find_if(sink.queue_events.begin(), sink.queue_events.end(),
                        [type](const iree_hal_profile_queue_event_t& event) {
                          return event.type == type;
                        });
  };
  auto find_memory_event = [&](iree_hal_profile_memory_event_type_t type) {
    return std::find_if(sink.memory_events.begin(), sink.memory_events.end(),
                        [type](const iree_hal_profile_memory_event_t& event) {
                          return event.type == type;
                        });
  };

  auto queue_alloca_it =
      find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA);
  ASSERT_NE(sink.queue_events.end(), queue_alloca_it);
  EXPECT_EQ(kBufferSize, queue_alloca_it->payload_length);
  EXPECT_NE(0u, queue_alloca_it->allocation_id);

  auto queue_dealloca_it =
      find_queue_event(IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA);
  ASSERT_NE(sink.queue_events.end(), queue_dealloca_it);
  EXPECT_EQ(queue_alloca_it->allocation_id, queue_dealloca_it->allocation_id);

  auto memory_alloca_it =
      find_memory_event(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA);
  ASSERT_NE(sink.memory_events.end(), memory_alloca_it);
  EXPECT_EQ(queue_alloca_it->allocation_id, memory_alloca_it->allocation_id);
  EXPECT_EQ(kBufferSize, memory_alloca_it->length);
  EXPECT_NE(0u, memory_alloca_it->submission_id);

  auto memory_dealloca_it =
      find_memory_event(IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA);
  ASSERT_NE(sink.memory_events.end(), memory_dealloca_it);
  EXPECT_EQ(queue_alloca_it->allocation_id, memory_dealloca_it->allocation_id);
  EXPECT_EQ(kBufferSize, memory_dealloca_it->length);
  EXPECT_NE(0u, memory_dealloca_it->submission_id);
}

CTS_REGISTER_TEST_SUITE(VulkanProfilingTest);

}  // namespace iree::hal::cts
