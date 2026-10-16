// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/profile.h"

#include <array>
#include <functional>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProfileSink {
  // Stack-owned sink resource, retained by each active session.
  iree_hal_resource_t resource;
  // Successful queue-event writes, copied before the callback returns.
  std::vector<iree_hal_profile_queue_event_t> events;
  // Successful device metadata records.
  std::vector<iree_hal_profile_device_record_t> devices;
  // Successful queue metadata records.
  std::vector<iree_hal_profile_queue_record_t> queues;
  // Optional producer work during a sink write, exercising lock separation.
  std::function<void()> during_write;
  // Successfully reported drops across all queue streams.
  uint64_t dropped = 0;
  // Number of session starts.
  int begins = 0;
  // Number of terminal callbacks.
  int ends = 0;
  // Producer status supplied to the most recent terminal callback.
  iree_status_code_t end_status = IREE_STATUS_OK;
  // Dependency failure injected at the sink boundary.
  bool fail_write = false;

  iree_hal_profile_sink_t* base() {
    return reinterpret_cast<iree_hal_profile_sink_t*>(this);
  }

  static void Destroy(iree_hal_profile_sink_t*) {}
  static iree_status_t Begin(iree_hal_profile_sink_t* base,
                             const iree_hal_profile_chunk_metadata_t*) {
    ++reinterpret_cast<ProfileSink*>(base)->begins;
    return iree_ok_status();
  }
  static iree_status_t End(iree_hal_profile_sink_t* base,
                           const iree_hal_profile_chunk_metadata_t*,
                           iree_status_code_t code) {
    auto& sink = *reinterpret_cast<ProfileSink*>(base);
    ++sink.ends;
    sink.end_status = code;
    return iree_ok_status();
  }
  static iree_status_t Write(iree_hal_profile_sink_t* base,
                             const iree_hal_profile_chunk_metadata_t* metadata,
                             iree_host_size_t count,
                             const iree_const_byte_span_t* spans) {
    auto& sink = *reinterpret_cast<ProfileSink*>(base);
    if (sink.fail_write) {
      return iree_status_from_code(IREE_STATUS_DATA_LOSS);
    }
    if (sink.during_write) {
      auto callback = std::move(sink.during_write);
      callback();
    }
    EXPECT_EQ(sink.begins, sink.ends + 1);
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (iree_string_view_equal(metadata->content_type,
                                 IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
        auto* records = reinterpret_cast<const iree_hal_profile_queue_event_t*>(
            spans[i].data);
        for (size_t j = 0; j < spans[i].data_length / sizeof(*records); ++j) {
          EXPECT_EQ(records[j].queue_ordinal, metadata->queue_ordinal);
          EXPECT_EQ(records[j].stream_id, metadata->stream_id);
          sink.events.push_back(records[j]);
        }
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
        auto* record =
            reinterpret_cast<const iree_hal_profile_device_record_t*>(
                spans[i].data);
        sink.devices.push_back(*record);
      } else if (iree_string_view_equal(metadata->content_type,
                                        IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES)) {
        auto* record = reinterpret_cast<const iree_hal_profile_queue_record_t*>(
            spans[i].data);
        sink.queues.push_back(*record);
      }
    }
    sink.dropped += metadata->dropped_record_count;
    EXPECT_EQ(metadata->dropped_record_count != 0,
              (metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED) != 0);
    return iree_ok_status();
  }
  ProfileSink() {
    static const iree_hal_profile_sink_vtable_t vtable = {Destroy, Begin, Write,
                                                          End};
    iree_hal_resource_initialize(&vtable, &resource);
  }
};

class GpuProfileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_.data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS;
    options_.sink = sink_.base();
  }
  void TearDown() override {
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_end(profile_));
  }
  void Begin() {
    IREE_ASSERT_OK(iree_hal_amd_gpu_profile_begin(
        &options_, 37, 2, iree_allocator_system(), &profile_));
  }
  void Append(uint32_t queue = 0) {
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        profile_, queue, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 1, 1,
        &operation);
    iree_hal_amd_gpu_profile_operation_record(profile_, &operation, true, 0);
  }
  // Sink outlives the retained session, including failure cleanup.
  ProfileSink sink_;
  // Validated caller request for the internal session owner.
  iree_hal_device_profiling_options_t options_ = {};
  // Unique session consumed by teardown unless a test explicitly ends it.
  iree_hal_amd_gpu_profile_t* profile_ = nullptr;
};

TEST_F(GpuProfileTest, IdentitiesAndObservedAdmission) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  ASSERT_EQ(sink_.devices.size(), 1u);
  EXPECT_EQ(sink_.devices[0].queue_count, 2u);
  EXPECT_EQ(sink_.devices[0].timestamp_frequency_hz, 0u);
  ASSERT_EQ(sink_.queues.size(), 2u);
  EXPECT_NE(sink_.queues[0].stream_id, sink_.queues[1].stream_id);
  iree_hal_amd_gpu_profile_operation_t operation;
  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 1, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE, 2, 3, 1000,
      &operation);
  iree_hal_amd_gpu_profile_operation_defer(&operation);
  iree_hal_amd_gpu_profile_operation_record(profile_, &operation, true, 2);
  iree_hal_amd_gpu_profile_operation_record(profile_, &operation, true, 0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 1u);
  const auto& event = sink_.events[0];
  EXPECT_EQ(event.record_length, sizeof(event));
  EXPECT_EQ(event.queue_ordinal, 1u);
  EXPECT_EQ(event.stream_id, sink_.queues[1].stream_id);
  EXPECT_EQ(event.wait_count, 2u);
  EXPECT_EQ(event.signal_count, 3u);
  EXPECT_EQ(event.barrier_count, 2u);
  EXPECT_EQ(event.operation_count, 1000u);
  EXPECT_EQ(event.dependency_strategy,
            IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER);
  EXPECT_GE(event.ready_host_time_ns, event.host_time_ns);

  iree_hal_amd_gpu_profile_operation_initialize(
      profile_, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER, 1, 1, 0,
      &operation);
  iree_hal_amd_gpu_profile_operation_record(profile_, &operation, false, 0);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.events.back().ready_host_time_ns, 0);
}

TEST_F(GpuProfileTest, FailedSnapshotRetriesWithoutBlockingProducer) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  Append();
  sink_.fail_write = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_flush(profile_));
  sink_.fail_write = false;
  sink_.during_write = [&] { Append(); };
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 1u);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  ASSERT_EQ(sink_.events.size(), 2u);
  EXPECT_EQ(sink_.events[0].event_id + 1, sink_.events[1].event_id);
  EXPECT_EQ(sink_.events[0].submission_id + 1, sink_.events[1].submission_id);
}

TEST_F(GpuProfileTest,
       BoundedConcurrentStreamsAccountEveryDropAndReuseStorage) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  constexpr size_t kOperations = 8193;
  std::array<std::thread, 2> threads;
  for (uint32_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (size_t j = 0; j < kOperations; ++j) {
        Append(i);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_GT(sink_.dropped, 0u);
  EXPECT_EQ(sink_.events.size() + sink_.dropped, 2 * kOperations);
  const size_t first_count = sink_.events.size();
  const uint64_t first_dropped = sink_.dropped;
  Append(0);
  Append(1);
  IREE_ASSERT_OK(iree_hal_amd_gpu_profile_flush(profile_));
  EXPECT_EQ(sink_.events.size(), first_count + 2);
  EXPECT_EQ(sink_.dropped, first_dropped);
}

TEST_F(GpuProfileTest, FailedEndStillClosesSessionAndReleasesSink) {
  ASSERT_NO_FATAL_FAILURE(Begin());
  Append();
  sink_.fail_write = true;
  auto* profile = profile_;
  profile_ = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amd_gpu_profile_end(profile));
  EXPECT_EQ(sink_.ends, 1);
  EXPECT_EQ(sink_.end_status, IREE_STATUS_DATA_LOSS);
  sink_.fail_write = false;
  ASSERT_NO_FATAL_FAILURE(Begin());
}

TEST_F(GpuProfileTest, FailedMetadataClosesBegunSession) {
  sink_.fail_write = true;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DATA_LOSS,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, iree_allocator_system(),
                                     &profile_));
  EXPECT_EQ(profile_, nullptr);
  EXPECT_EQ(sink_.begins, 1);
  EXPECT_EQ(sink_.ends, 1);
}

TEST_F(GpuProfileTest, UnsupportedDataDoesNotStartSinkSession) {
  options_.data_families |= IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_hal_amd_gpu_profile_begin(&options_, 37, 2, iree_allocator_system(),
                                     &profile_));
  EXPECT_EQ(profile_, nullptr);
  EXPECT_EQ(sink_.begins, 0);
}

}  // namespace
