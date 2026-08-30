// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/profile.h"

#include <cstdint>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

using ::iree::hal::cts::TestProfileSink;
using ::iree::hal::cts::TestProfileSinkAsBase;
using ::iree::hal::cts::TestProfileSinkInitialize;

// Profile sink wrapper that injects one selected write failure before
// forwarding all other callbacks to a recording sink.
struct FaultInjectingProfileSink {
  // HAL resource header for the sink.
  iree_hal_resource_t resource;

  // Borrowed sink receiving callbacks that are not faulted.
  iree_hal_profile_sink_t* delegate = nullptr;

  // Content type whose write callback should fail, or empty when disabled.
  iree_string_view_t fail_write_content_type = iree_string_view_empty();

  // Number of matching write callbacks that should fail.
  int fail_write_remaining = 0;

  // Status code returned from matching write callbacks.
  iree_status_code_t fail_write_status_code = IREE_STATUS_OK;
};

static FaultInjectingProfileSink* FaultInjectingProfileSinkCast(
    iree_hal_profile_sink_t* sink) {
  return reinterpret_cast<FaultInjectingProfileSink*>(sink);
}

static void FaultInjectingProfileSinkDestroy(iree_hal_profile_sink_t* sink) {
  (void)sink;
}

static iree_status_t FaultInjectingProfileSinkBeginSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  FaultInjectingProfileSink* fault_sink = FaultInjectingProfileSinkCast(sink);
  return iree_hal_profile_sink_begin_session(fault_sink->delegate, metadata);
}

static iree_status_t FaultInjectingProfileSinkWrite(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  FaultInjectingProfileSink* fault_sink = FaultInjectingProfileSinkCast(sink);
  if (fault_sink->fail_write_remaining != 0 &&
      iree_string_view_equal(metadata->content_type,
                             fault_sink->fail_write_content_type)) {
    --fault_sink->fail_write_remaining;
    return iree_make_status(fault_sink->fail_write_status_code,
                            "injected profile sink write failure");
  }
  return iree_hal_profile_sink_write(fault_sink->delegate, metadata,
                                     iovec_count, iovecs);
}

static iree_status_t FaultInjectingProfileSinkEndSession(
    iree_hal_profile_sink_t* sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  FaultInjectingProfileSink* fault_sink = FaultInjectingProfileSinkCast(sink);
  return iree_hal_profile_sink_end_session(fault_sink->delegate, metadata,
                                           session_status_code);
}

static const iree_hal_profile_sink_vtable_t kFaultInjectingProfileSinkVTable = {
    /*.destroy=*/FaultInjectingProfileSinkDestroy,
    /*.begin_session=*/FaultInjectingProfileSinkBeginSession,
    /*.write=*/FaultInjectingProfileSinkWrite,
    /*.end_session=*/FaultInjectingProfileSinkEndSession,
};

static void FaultInjectingProfileSinkInitialize(
    iree_hal_profile_sink_t* delegate, FaultInjectingProfileSink* out_sink) {
  iree_hal_resource_initialize(&kFaultInjectingProfileSinkVTable,
                               &out_sink->resource);
  out_sink->delegate = delegate;
}

static iree_hal_profile_sink_t* FaultInjectingProfileSinkAsBase(
    FaultInjectingProfileSink* sink) {
  return reinterpret_cast<iree_hal_profile_sink_t*>(sink);
}

class VulkanProfileRecorderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TestProfileSinkInitialize(&recording_sink_);
    FaultInjectingProfileSinkInitialize(TestProfileSinkAsBase(&recording_sink_),
                                        &fault_sink_);

    device_record_ = iree_hal_profile_device_record_default();
    device_record_.physical_device_ordinal = 0;
    device_record_.queue_count = 1;
    device_record_.flags = IREE_HAL_PROFILE_DEVICE_FLAG_TIMESTAMP_FREQUENCY;
    device_record_.timestamp_frequency_hz = 1000000000ull;

    queue_record_ = iree_hal_profile_queue_record_default();
    queue_record_.physical_device_ordinal = 0;
    queue_record_.queue_ordinal = 0;
    queue_record_.stream_id = 1;

    recorder_options_.name = IREE_SV("vulkan-test");
    recorder_options_.session_id = 42;
    recorder_options_.device_record_count = 1;
    recorder_options_.device_records = &device_record_;
    recorder_options_.queue_record_count = 1;
    recorder_options_.queue_records = &queue_record_;
    recorder_options_.dispatch_event_capacity = 4;
    recorder_options_.queue_event_capacity = 4;
    recorder_options_.queue_device_event_capacity = 4;
    recorder_options_.memory_event_capacity = 4;
  }

  void TearDown() override {
    if (!recorder_) return;
    IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_end(recorder_));
    iree_hal_vulkan_profile_recorder_destroy(recorder_);
  }

  iree_hal_device_profiling_options_t MakeProfilingOptions(
      iree_hal_device_profiling_data_families_t data_families) {
    iree_hal_device_profiling_options_t options = {0};
    options.data_families = data_families;
    options.sink = FaultInjectingProfileSinkAsBase(&fault_sink_);
    return options;
  }

  iree_status_t Create(
      iree_hal_device_profiling_data_families_t data_families) {
    iree_hal_device_profiling_options_t options =
        MakeProfilingOptions(data_families);
    return iree_hal_vulkan_profile_recorder_create(
        &recorder_options_, &options, iree_allocator_system(), &recorder_);
  }

  iree_hal_vulkan_profile_queue_scope_t QueueScope() {
    iree_hal_vulkan_profile_queue_scope_t scope =
        iree_hal_vulkan_profile_queue_scope_default();
    scope.physical_device_ordinal = queue_record_.physical_device_ordinal;
    scope.queue_ordinal = queue_record_.queue_ordinal;
    scope.stream_id = queue_record_.stream_id;
    return scope;
  }

  iree_hal_vulkan_profile_dispatch_event_info_t DispatchEventInfo() {
    iree_hal_vulkan_profile_dispatch_event_info_t event_info =
        iree_hal_vulkan_profile_dispatch_event_info_default();
    event_info.scope = QueueScope();
    event_info.submission_id = 7;
    event_info.executable_id = 5;
    event_info.function_ordinal = 0;
    event_info.workgroup_count[0] = 1;
    event_info.workgroup_count[1] = 1;
    event_info.workgroup_count[2] = 1;
    event_info.workgroup_size[0] = 1;
    event_info.workgroup_size[1] = 1;
    event_info.workgroup_size[2] = 1;
    event_info.start_tick = 1000;
    event_info.end_tick = 1200;
    return event_info;
  }

  TestProfileSink recording_sink_;
  FaultInjectingProfileSink fault_sink_;
  iree_hal_profile_device_record_t device_record_;
  iree_hal_profile_queue_record_t queue_record_;
  iree_hal_vulkan_profile_recorder_options_t recorder_options_ = {};
  iree_hal_vulkan_profile_recorder_t* recorder_ = nullptr;
};

TEST_F(VulkanProfileRecorderTest, AppendsVulkanEventStreams) {
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
                        IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS));

  iree_hal_vulkan_profile_queue_event_info_t queue_info =
      iree_hal_vulkan_profile_queue_event_info_default();
  queue_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
  queue_info.scope = QueueScope();
  queue_info.submission_id = 7;
  queue_info.operation_count = 1;
  iree_hal_vulkan_profile_recorder_append_queue_event(recorder_, &queue_info,
                                                      nullptr);

  iree_hal_vulkan_profile_dispatch_event_info_t dispatch_info =
      DispatchEventInfo();
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_append_dispatch_event(
      recorder_, &dispatch_info, nullptr));

  iree_hal_vulkan_profile_queue_device_event_info_t queue_device_info =
      iree_hal_vulkan_profile_queue_device_event_info_default();
  queue_device_info.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE;
  queue_device_info.scope = QueueScope();
  queue_device_info.submission_id = 7;
  queue_device_info.operation_count = 1;
  queue_device_info.start_tick = 900;
  queue_device_info.end_tick = 1300;
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_append_queue_device_event(
      recorder_, &queue_device_info, nullptr));

  iree_hal_profile_memory_event_t memory_event =
      iree_hal_profile_memory_event_default();
  memory_event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA;
  memory_event.flags = IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
  memory_event.result = IREE_STATUS_OK;
  memory_event.allocation_id = 9;
  memory_event.submission_id = 7;
  memory_event.physical_device_ordinal = 0;
  memory_event.queue_ordinal = 0;
  memory_event.length = 64;
  iree_hal_vulkan_profile_recorder_append_memory_event(recorder_, &memory_event,
                                                       nullptr);

  iree_hal_profile_clock_correlation_record_t correlation =
      iree_hal_profile_clock_correlation_record_default();
  correlation.flags =
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_DEVICE_TICK |
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_CPU_TIMESTAMP |
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_SYSTEM_TIMESTAMP |
      IREE_HAL_PROFILE_CLOCK_CORRELATION_FLAG_HOST_TIME_BRACKET;
  correlation.physical_device_ordinal = 0;
  correlation.sample_id = 1;
  correlation.device_tick = 1000;
  correlation.host_cpu_timestamp_ns = 5000;
  correlation.host_system_timestamp = 6000;
  correlation.host_system_frequency_hz = 1000000000ull;
  correlation.host_time_begin_ns = 4900;
  correlation.host_time_end_ns = 5100;
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_write_clock_correlations(
      recorder_, 1, &correlation));

  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_flush(recorder_));
  EXPECT_EQ(1u, recording_sink_.queue_events.size());
  EXPECT_EQ(1u, recording_sink_.dispatch_events.size());
  EXPECT_EQ(1u, recording_sink_.queue_device_events.size());
  EXPECT_EQ(1u, recording_sink_.memory_events.size());
  EXPECT_EQ(1u, recording_sink_.clock_correlations.size());
}

TEST_F(VulkanProfileRecorderTest, DispatchAutoFlushFailurePreservesRecords) {
  recorder_options_.dispatch_event_capacity = 1;
  IREE_EXPECT_OK(Create(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS));

  iree_hal_vulkan_profile_dispatch_event_info_t first_info =
      DispatchEventInfo();
  uint64_t first_event_id = 0;
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_append_dispatch_event(
      recorder_, &first_info, &first_event_id));

  fault_sink_.fail_write_content_type =
      IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS;
  fault_sink_.fail_write_remaining = 1;
  fault_sink_.fail_write_status_code = IREE_STATUS_DATA_LOSS;

  iree_hal_vulkan_profile_dispatch_event_info_t second_info =
      DispatchEventInfo();
  second_info.submission_id = 8;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_vulkan_profile_recorder_append_dispatch_event(
                            recorder_, &second_info, nullptr));
  EXPECT_TRUE(recording_sink_.dispatch_events.empty());

  uint64_t second_event_id = 0;
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_append_dispatch_event(
      recorder_, &second_info, &second_event_id));
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_flush(recorder_));
  ASSERT_EQ(2u, recording_sink_.dispatch_events.size());
  EXPECT_EQ(first_event_id, recording_sink_.dispatch_events[0].event_id);
  EXPECT_EQ(second_event_id, recording_sink_.dispatch_events[1].event_id);
}

TEST_F(VulkanProfileRecorderTest, AcceptsDispatchCaptureFilter) {
  iree_hal_device_profiling_options_t options =
      MakeProfilingOptions(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS);
  options.capture_filter.flags =
      IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN;
  options.capture_filter.executable_function_pattern = IREE_SV("dispatch_*");
  IREE_EXPECT_OK(iree_hal_vulkan_profile_recorder_create(
      &recorder_options_, &options, iree_allocator_system(), &recorder_));
}

TEST_F(VulkanProfileRecorderTest, RejectsTaskExecutionData) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      Create(IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS));
}

}  // namespace
}  // namespace iree::hal::vulkan
