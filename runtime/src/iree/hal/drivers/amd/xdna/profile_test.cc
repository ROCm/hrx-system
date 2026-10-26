// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <functional>

#include "iree/hal/drivers/amd/xdna/testing/queue_test_util.h"

namespace iree::hal::amd::xdna::testing {
namespace {

struct ProfileSink {
  // HAL sink resource, kept on the test stack through session end.
  iree_hal_resource_t resource;
  // Copied queue records in successful write order.
  std::vector<iree_hal_profile_queue_event_t> queue_events;
  // Copied completed host spans in successful write order.
  std::vector<iree_hal_profile_host_execution_event_t> host_events;
  // Complete typed sample records and their one-element cycle values.
  std::vector<iree_hal_profile_counter_sample_record_t> samples;
  // Derived partition-shim cycle deltas in sample order.
  std::vector<uint64_t> cycles;
  // Function names proving retained executable metadata survives completion.
  std::vector<std::string> names;
  // Work invoked during a sink callback to exercise append/snapshot separation.
  std::function<void()> during_write;
  // Successfully reported dropped queue records.
  uint64_t dropped = 0;
  // Loss metadata reported independently of retained counter sample payloads.
  struct {
    // Accepted selected invocations with no retained counter sample.
    uint64_t dropped_record_count = 0;
    // Counter chunks marking a discontinuity in capture coverage.
    uint64_t truncated_chunk_count = 0;
  } counter_loss;
  // True when omitted attribution leaves executable metadata incomplete.
  bool executable_metadata_truncated = false;
  // Number of observed session begin callbacks.
  int begins = 0;
  // Number of observed session end callbacks.
  int ends = 0;
  // Terminal session status supplied by the producer.
  iree_status_code_t end_status = IREE_STATUS_OK;
  // Injected sink write failure, separate from the queue implementation.
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
                           iree_status_code_t status) {
    auto& sink = *reinterpret_cast<ProfileSink*>(base);
    ++sink.ends;
    sink.end_status = status;
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
    EXPECT_EQ(sink.ends + 1, sink.begins);
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (iree_string_view_equal(
              metadata->content_type,
              IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
        iree_hal_profile_counter_sample_record_t record;
        EXPECT_EQ(spans[i].data_length, sizeof(record) + sizeof(uint64_t));
        memcpy(&record, spans[i].data, sizeof(record));
        sink.samples.push_back(record);
        uint64_t value;
        memcpy(&value, spans[i].data + sizeof(record), sizeof(value));
        sink.cycles.push_back(value);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_FUNCTIONS)) {
        iree_hal_profile_executable_function_record_t record;
        memcpy(&record, spans[i].data, sizeof(record));
        EXPECT_EQ(record.record_length, spans[i].data_length);
        sink.names.emplace_back((const char*)spans[i].data + sizeof(record),
                                record.name_length);
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
        const auto* records =
            reinterpret_cast<const iree_hal_profile_queue_event_t*>(
                spans[i].data);
        sink.queue_events.insert(
            sink.queue_events.end(), records,
            records + spans[i].data_length / sizeof(*records));
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS)) {
        const auto* records =
            reinterpret_cast<const iree_hal_profile_host_execution_event_t*>(
                spans[i].data);
        sink.host_events.insert(
            sink.host_events.end(), records,
            records + spans[i].data_length / sizeof(*records));
      } else if (iree_string_view_equal(
                     metadata->content_type,
                     IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES)) {
        const auto* record =
            reinterpret_cast<const iree_hal_profile_device_record_t*>(
                spans[i].data);
        EXPECT_EQ(record->timestamp_frequency_hz, 0u);
        EXPECT_FALSE(record->flags &
                     IREE_HAL_PROFILE_DEVICE_FLAG_TIMESTAMP_FREQUENCY);
      }
    }
    if (iree_string_view_equal(metadata->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
      sink.dropped += metadata->dropped_record_count;
      EXPECT_EQ(metadata->dropped_record_count != 0,
                (metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED) != 0);
    }
    if (iree_string_view_equal(metadata->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_COUNTER_SAMPLES)) {
      sink.counter_loss.dropped_record_count += metadata->dropped_record_count;
      sink.counter_loss.truncated_chunk_count +=
          (metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED) != 0;
    }
    if (iree_string_view_equal(metadata->content_type,
                               IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES) &&
        (metadata->flags & IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED)) {
      sink.executable_metadata_truncated = true;
      EXPECT_EQ(metadata->dropped_record_count, 0u);
    }
    return iree_ok_status();
  }
  ProfileSink() {
    static const iree_hal_profile_sink_vtable_t vtable = {Destroy, Begin, Write,
                                                          End};
    iree_hal_resource_initialize(&vtable, &resource);
  }
};

TEST(XdnaProfileTest, DeferredReadinessAndHostPublicationExcludeNativeWait) {
  ProfileSink sink;
  QueueHarness harness;
  harness.native.HoldRetirement();
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  const iree_hal_device_profiling_options_t options = {
      .flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_hal_device_profiling_begin(harness.device, &options));
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &gate));
  ASSERT_NO_FATAL_FAILURE(harness.SubmitAfter(gate, 1));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
  EXPECT_TRUE(sink.queue_events.empty());
  EXPECT_TRUE(sink.host_events.empty());
  const iree_time_t release_time = iree_time_now();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  // Native notification is armed after the publisher's completed host span.
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilNotificationCount(1));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
  ASSERT_EQ(sink.queue_events.size(), 1u);
  ASSERT_EQ(sink.host_events.size(), 1u);
  const auto& queue = sink.queue_events[0];
  const auto& host = sink.host_events[0];
  EXPECT_GE(queue.ready_host_time_ns, release_time);
  EXPECT_GE(queue.ready_host_time_ns, queue.host_time_ns);
  EXPECT_EQ(queue.dependency_strategy,
            IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER);
  EXPECT_EQ(queue.wait_count, 1u);
  EXPECT_EQ(queue.signal_count, 1u);
  EXPECT_EQ(host.submission_id, queue.submission_id);
  EXPECT_EQ(host.operation_count, 1u);
  EXPECT_EQ(host.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(host.flags & IREE_HAL_PROFILE_HOST_EXECUTION_EVENT_FLAG_DEFERRED);
  EXPECT_GE(host.start_host_time_ns, queue.ready_host_time_ns);
  EXPECT_GE(host.end_host_time_ns, host.start_host_time_ns);
  uint64_t done = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(harness.done, &done));
  EXPECT_EQ(done, 0u);
  harness.native.retirement_limit = UINT64_MAX;
  harness.native.Wake();
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(sink.host_events.size(), 1u);
  EXPECT_EQ(sink.ends, 1);
  iree_hal_semaphore_release(gate);
}

TEST(XdnaProfileTest, HostCallRecordsCallbackBodyAfterDependencies) {
  ProfileSink sink;
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  const iree_hal_device_profiling_options_t options = {
      .flags = IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  iree_hal_semaphore_t* gate = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      harness.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, 0, &gate));
  struct CallbackState {
    // Result observed after the callback's completion signal.
    uint64_t result = 0;
    // Timestamp inside the callback body, bounded by its profile span.
    iree_time_t host_time = 0;
  } state;
  const auto call = iree_hal_make_host_call(
      +[](void* user_data, const uint64_t args[4],
          iree_hal_host_call_context_t*) -> iree_status_t {
        auto& state = *static_cast<CallbackState*>(user_data);
        state.result = args[0] + args[1];
        state.host_time = iree_time_now();
        return iree_ok_status();
      },
      &state);
  uint64_t value = 1;
  const uint64_t args[4] = {17, 25, 0, 0};
  IREE_ASSERT_OK(iree_hal_queue_host_call(harness.queue, {1, &gate, &value},
                                          {1, &harness.done, &value}, call,
                                          args, IREE_HAL_HOST_CALL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
  EXPECT_TRUE(sink.queue_events.empty());
  EXPECT_TRUE(sink.host_events.empty());
  const iree_time_t release_time = iree_time_now();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, value, nullptr));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(state.result, 42u);
  ASSERT_EQ(sink.queue_events.size(), 1u);
  ASSERT_EQ(sink.host_events.size(), 1u);
  const auto& queue = sink.queue_events[0];
  const auto& host = sink.host_events[0];
  EXPECT_EQ(queue.type, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL);
  EXPECT_EQ(queue.operation_count, 1u);
  EXPECT_GE(queue.ready_host_time_ns, release_time);
  EXPECT_EQ(host.submission_id, queue.submission_id);
  EXPECT_EQ(host.operation_count, 1u);
  EXPECT_EQ(host.status_code, IREE_STATUS_OK);
  EXPECT_GE(host.start_host_time_ns, queue.ready_host_time_ns);
  EXPECT_LE(host.start_host_time_ns, state.host_time);
  EXPECT_GE(host.end_host_time_ns, state.host_time);
  iree_hal_semaphore_release(gate);
}

TEST(XdnaProfileTest, BoundedRingRetriesWritesAndAllowsProgressInsideSink) {
  ProfileSink sink;
  HostAllocationCounters counters;
  QueueHarness harness;
  harness.host_allocator = TrackingAllocator(&counters);
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  uint64_t value = 0;
  const auto submit_barrier = [&] {
    ++value;
    IREE_ASSERT_OK(iree_hal_queue_barrier(
        harness.queue, {}, {1, &harness.done, &value}, nullptr, 0));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  };
  ASSERT_NO_FATAL_FAILURE(submit_barrier());
  // A sink callback can drive another queue operation without taking a lock
  // held by the snapshot writer; its event belongs to the next snapshot.
  sink.during_write = submit_barrier;
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
  EXPECT_EQ(sink.queue_events.size(), 1u);
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
  EXPECT_EQ(sink.queue_events.size(), 2u);
  const auto warmed = SnapshotHostAllocations(counters);
  for (int i = 0; i < 4096 + 5; ++i) {
    ASSERT_NO_FATAL_FAILURE(submit_barrier());
  }
  const auto captured = SnapshotHostAllocations(counters);
  EXPECT_EQ(warmed.allocations, captured.allocations);
  EXPECT_EQ(warmed.reallocations, captured.reallocations);
  EXPECT_EQ(warmed.frees, captured.frees);
  sink.fail_write = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_device_profiling_flush(harness.device));
  EXPECT_EQ(sink.queue_events.size(), 2u);
  sink.fail_write = false;
  IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
  EXPECT_EQ(sink.queue_events.size(), 4098u);
  EXPECT_EQ(sink.dropped, 5u);
  for (size_t i = 1; i < sink.queue_events.size(); ++i) {
    EXPECT_GT(sink.queue_events[i].event_id, sink.queue_events[i - 1].event_id);
  }
  ASSERT_NO_FATAL_FAILURE(submit_barrier());
  sink.fail_write = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(sink.end_status, IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(sink.ends, 1);
  // Failed end still released the session and its retained sink ownership.
  sink.fail_write = false;
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(sink.ends, 2);
}

// Fixed native transaction layout for the controlled provider. The compiler's
// actual instruction bytes and numeric output are exercised by HAL hardware
// CTS.
static ImageFixture MeasuredFixture() {
  ImageFixture fixture;
  fixture.allocations[0].byte_length = 32768 + 48;
  fixture.relocations[0].byte_offset = 16;
  fixture.relocations[1].byte_offset = 24;
  for (auto& invocation : fixture.invocations) {
    invocation.byte_length = 48;
    invocation.measurement.begin = 16;
    invocation.measurement.end = 48;
  }
  for (size_t i = 0; i < 2; ++i) {
    fixture.loads[i].file_range.length = fixture.loads[i].memory_size = 48;
    fixture.payloads[i].assign(48, 0);
    fixture.payloads[i][1] = 1;
    iree_unaligned_store_le_u32(fixture.payloads[i].data() + 8, 1);
    iree_unaligned_store_le_u32(fixture.payloads[i].data() + 12, 48);
  }
  fixture.loads[2].file_range.offset = 4096 + 48;
  return fixture;
}

static void RecordMeasuredDispatches(QueueHarness& harness, uint32_t count,
                                     iree_hal_command_buffer_t** out_buffer) {
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(MeasuredFixture(), &executable, &function));
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      iree_hal_queue_family(harness.queue),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      1, out_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(*out_buffer));
  const auto binding = iree_hal_make_indirect_buffer_ref(0, 0, 64);
  for (uint32_t i = 0; i < count; ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        *out_buffer, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding},
        IREE_HAL_DISPATCH_FLAG_NONE));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(*out_buffer));
  iree_hal_executable_release(executable);
}

TEST(XdnaProfileTest, MeasuredArraysReuseCommandsAndRetainAttribution) {
  ProfileSink sink;
  HostAllocationCounters allocations;
  QueueHarness harness;
  harness.native.pending_capacity = harness.native.maximum_command_count = 8;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize(TrackingAllocator(&allocations)));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      RecordMeasuredDispatches(harness, 8, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  const iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  HostAllocationSnapshot warm = {};
  size_t flushes = 0;
  for (uint64_t iteration = 1; iteration <= 3; ++iteration) {
    IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                          {1, &harness.done, &iteration},
                                          command_buffer, {1, &binding}, 0));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, iteration));
    IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
    EXPECT_EQ(harness.native.result.collects, 0u);
    if (iteration == 1) {
      warm = SnapshotHostAllocations(allocations);
      flushes = harness.native.FlushCount();
    } else {
      ExpectHostAllocationsUnchanged(warm, allocations);
      EXPECT_EQ(harness.native.FlushCount(), flushes);
    }
  }
  EXPECT_EQ(harness.native.submission_count, 3u);
  const uint64_t command_buffer_id =
      iree_hal_command_buffer_profile_id(command_buffer);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(harness.native.result.collects, 1u);
  EXPECT_EQ(harness.native.result.detaches, 1u);
  EXPECT_EQ(sink.names, std::vector<std::string>{"main"});
  ASSERT_EQ(sink.samples.size(), 24u);
  for (uint32_t i = 0; i < 24; ++i) {
    EXPECT_EQ(sink.samples[i].command_buffer_id, command_buffer_id);
    EXPECT_EQ(sink.samples[i].command_index, i % 8);
    EXPECT_EQ(sink.samples[i].submission_id, i / 8 + 1);
    EXPECT_EQ(sink.samples[i].function_ordinal, 0u);
    EXPECT_EQ(sink.samples[i].flags, 0u);
    EXPECT_EQ(sink.samples[i].start_tick, 0u);
    EXPECT_EQ(sink.cycles[i], 100u + i);
  }
}

TEST(XdnaProfileTest,
     BusyArrayConsumesBudgetOnceAndRejectedSuffixHasNoSamples) {
  for (bool reject : {false, true}) {
    ProfileSink sink;
    QueueHarness harness;
    harness.native.pending_capacity = 24;
    harness.native.maximum_command_count = 8;
    harness.native.HoldRetirement();
    // The second array crosses the capacity boundary before its BUSY retry.
    harness.native.result.capacity = 10 * 24;
    if (reject) {
      harness.native.reject_submission_attempt = 3;
    } else {
      harness.native.busy_submission_attempt = 2;
    }
    ASSERT_NO_FATAL_FAILURE(harness.Initialize());
    ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
    iree_hal_command_buffer_t* command_buffer = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        RecordMeasuredDispatches(harness, 24, &command_buffer));
    iree_hal_buffer_t* buffer = nullptr;
    ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
    const iree_hal_device_profiling_options_t options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
        .sink = sink.base(),
    };
    IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
    const iree_hal_buffer_binding_t binding = {buffer, 0, 64};
    uint64_t value = 1;
    IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                          {1, &harness.done, &value},
                                          command_buffer, {1, &binding}, 0));
    const size_t flushes = harness.native.FlushCount();
    harness.native.retirement_limit = UINT64_MAX;
    harness.native.Wake();
    if (reject) {
      uint64_t observed = 0;
      iree_status_t status = iree_hal_semaphore_query(harness.done, &observed);
      while (iree_status_is_ok(status)) {
        IREE_ASSERT_OK(iree_async_proactor_poll(
            harness.proactor, iree_infinite_timeout(), nullptr));
        status = iree_hal_semaphore_query(harness.done, &observed);
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    } else {
      ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
    }
    EXPECT_EQ(harness.native.FlushCount(), flushes);
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(buffer);
    IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
    const uint32_t accepted = reject ? 16 : 24;
    ASSERT_EQ(sink.samples.size(), 10u);
    EXPECT_EQ(sink.counter_loss.dropped_record_count, accepted - 10);
    EXPECT_EQ(sink.counter_loss.truncated_chunk_count, 1u);
    EXPECT_EQ(sink.end_status, IREE_STATUS_OK);
    for (uint32_t i = 0; i < 10; ++i) {
      EXPECT_EQ(sink.samples[i].command_index, i);
      EXPECT_EQ(sink.cycles[i], 100u + i);
    }
  }
}

TEST(XdnaProfileTest, FiniteBudgetPreservesSamplesAcrossLaterSubmissions) {
  ProfileSink sink;
  QueueHarness harness;
  harness.native.pending_capacity = harness.native.maximum_command_count = 3;
  harness.native.result.capacity = 48;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      RecordMeasuredDispatches(harness, 3, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  const iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  for (uint64_t value = 1; value <= 2; ++value) {
    IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                          {1, &harness.done, &value},
                                          command_buffer, {1, &binding}, 0));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
    IREE_ASSERT_OK(iree_hal_device_profiling_flush(harness.device));
    EXPECT_EQ(harness.native.result.collects, 0u);
  }
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  // Short recorded runs retain their ordinary singleton publication policy.
  EXPECT_EQ(harness.native.submission_count, 6u);
  EXPECT_EQ(sink.end_status, IREE_STATUS_OK);
  EXPECT_EQ(sink.names, std::vector<std::string>{"main"});
  ASSERT_EQ(sink.samples.size(), 2u);
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(sink.samples[i].command_index, i);
    EXPECT_EQ(sink.samples[i].submission_id, 1u);
    EXPECT_EQ(sink.cycles[i], 100u + i);
  }
  EXPECT_EQ(sink.counter_loss.dropped_record_count, 4u);
  EXPECT_EQ(sink.counter_loss.truncated_chunk_count, 1u);
  EXPECT_EQ(harness.native.result.collects, 1u);
  EXPECT_EQ(harness.native.result.detaches, 1u);
  EXPECT_FALSE(harness.native.result.binding.memory);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
}

TEST(XdnaProfileTest, MalformedResultsFailCaptureAfterWorkCompletes) {
  ProfileSink sink;
  QueueHarness harness;
  harness.native.result.corrupt = true;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      RecordMeasuredDispatches(harness, 3, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  const iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  EXPECT_EQ(harness.native.submission_count, 3u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_device_profiling_end(harness.device));
  EXPECT_TRUE(sink.samples.empty());
  EXPECT_EQ(harness.native.result.detaches, 1u);
  EXPECT_FALSE(harness.native.result.binding.memory);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
}

TEST(XdnaProfileTest, FilteredCaptureCopiesSelectionAndBudgetsOnlyMatches) {
  ProfileSink sink;
  QueueHarness harness;
  harness.native.pending_capacity = harness.native.maximum_command_count = 8;
  harness.native.result.capacity = 24;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      RecordMeasuredDispatches(harness, 8, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  char pattern[] = "ma*";
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
      .sink = sink.base(),
      .capture_filter =
          {
              .flags =
                  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN |
                  IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_COMMAND_INDEX,
              .executable_function_pattern = iree_make_cstring_view(pattern),
              .command_index = 3,
          },
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  pattern[0] = 'x';
  const iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  for (uint64_t value = 1; value <= 2; ++value) {
    IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                          {1, &harness.done, &value},
                                          command_buffer, {1, &binding}, 0));
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  }
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  ASSERT_EQ(sink.samples.size(), 1u);
  EXPECT_EQ(sink.samples[0].command_index, 3u);
  EXPECT_EQ(harness.native.submission_count, 2u);
  EXPECT_EQ(sink.counter_loss.dropped_record_count, 1u);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
}

TEST(XdnaProfileTest,
     DirectDispatchRebindsMeasuredStorageAndRearmsAfterSinkFailure) {
  ProfileSink sink;
  HostAllocationCounters allocations;
  QueueHarness harness;
  harness.native.enforce_single_command_memory = false;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize(TrackingAllocator(&allocations)));
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function;
  ASSERT_NO_FATAL_FAILURE(
      harness.LoadExecutable(MeasuredFixture(), &executable, &function));
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  for (auto& buffer : buffers) {
    ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  }
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
      .sink = sink.base(),
  };
  std::vector<uint8_t> first_command;
  HostAllocationSnapshot warm = {};
  for (uint64_t iteration = 1; iteration <= 3; ++iteration) {
    IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
    if (iteration > 1) {
      warm = SnapshotHostAllocations(allocations);
    }
    harness.native.HoldRetirement();
    const auto binding =
        iree_hal_make_buffer_ref(buffers[iteration % 2], 0, 64);
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        harness.queue, {}, {1, &harness.done, &iteration}, executable, function,
        iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, {},
        0));
    {
      std::lock_guard<std::mutex> lock(harness.native.mutex);
      ASSERT_EQ(harness.native.pending_commands.size(), 1u);
      const auto& bytes = harness.native.pending_commands[0].bytes;
      if (iteration == 1) {
        first_command = bytes;
      } else if (iteration == 2) {
        EXPECT_NE(bytes, first_command);
      } else {
        EXPECT_EQ(bytes, first_command);
      }
    }
    harness.native.retirement_limit = UINT64_MAX;
    harness.native.Wake();
    ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, iteration));
    if (iteration > 1) {
      ExpectHostAllocationsUnchanged(warm, allocations);
    }
    sink.fail_write = iteration == 2;
    if (sink.fail_write) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                            iree_hal_device_profiling_end(harness.device));
    } else {
      IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
    }
    sink.fail_write = false;
    EXPECT_FALSE(harness.native.result.binding.memory);
  }
  ASSERT_EQ(sink.samples.size(), 2u);
  EXPECT_EQ(sink.samples[0].command_buffer_id, 0u);
  EXPECT_EQ(sink.samples[0].command_index, UINT32_MAX);
  EXPECT_EQ(harness.native.result.attaches, 3u);
  EXPECT_EQ(harness.native.result.detaches, 3u);
  iree_hal_executable_release(executable);
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
}

TEST(XdnaProfileTest,
     MetadataAloneDoesNotRequireTimerBoundariesOrNativeControls) {
  ProfileSink sink;
  QueueHarness harness;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  ASSERT_NO_FATAL_FAILURE(harness.Submit());
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilDone(IREE_STATUS_OK));
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(sink.names, std::vector<std::string>{"main"});
  EXPECT_EQ(harness.native.result.attaches, 0u);
  EXPECT_EQ(harness.native.result.collects, 0u);
}

TEST(XdnaProfileTest, MetadataOverflowPreservesRetainedNames) {
  ProfileSink sink;
  QueueHarness harness;
  harness.native.pending_capacity = harness.native.maximum_command_count = 8;
  ASSERT_NO_FATAL_FAILURE(harness.Initialize());
  ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
  iree_hal_command_buffer_t* command_buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      RecordMeasuredDispatches(harness, 4097, &command_buffer));
  iree_hal_buffer_t* buffer = nullptr;
  ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
  const iree_hal_buffer_binding_t binding = {buffer, 0, 64};
  const iree_hal_device_profiling_options_t options = {
      .data_families = IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA,
      .sink = sink.base(),
  };
  IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
  uint64_t value = 1;
  IREE_ASSERT_OK(iree_hal_queue_execute(harness.queue, {},
                                        {1, &harness.done, &value},
                                        command_buffer, {1, &binding}, 0));
  ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(buffer);
  IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
  EXPECT_EQ(sink.end_status, IREE_STATUS_OK);
  EXPECT_EQ(sink.names, std::vector<std::string>{"main"});
  EXPECT_TRUE(sink.executable_metadata_truncated);
  EXPECT_EQ(harness.native.result.attaches, 0u);
  EXPECT_EQ(harness.native.result.collects, 0u);
}

TEST(XdnaProfileTest,
     UnsupportedAttachmentReleasesStorageAndKeepsDeviceUsable) {
  for (bool single_aperture : {false, true}) {
    SCOPED_TRACE(single_aperture);
    HostAllocationCounters allocations;
    {
      ProfileSink sink;
      QueueHarness harness;
      harness.native.enforce_single_command_memory = single_aperture;
      harness.native.result.unsupported_attachment = true;
      ASSERT_NO_FATAL_FAILURE(
          harness.Initialize(TrackingAllocator(&allocations)));
      ASSERT_NO_FATAL_FAILURE(harness.RegisterNativeObserver());
      const iree_hal_device_profiling_options_t options = {
          .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
          .sink = sink.base(),
      };
      int64_t live_allocations = 0;
      size_t live_memories = 0;
      for (uint32_t attempt = 0; attempt < 3; ++attempt) {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_UNIMPLEMENTED,
            iree_hal_device_profiling_begin(harness.device, &options));
        EXPECT_FALSE(harness.context->result_buffer_retained);
        EXPECT_FALSE(harness.native.result.binding.memory);
        if (attempt == 0) {
          // Range mode can warm the reusable command arena on first use.
          live_allocations = allocations.live_allocations.load();
          live_memories = harness.native.live_memories;
        } else {
          EXPECT_EQ(allocations.live_allocations.load(), live_allocations);
          EXPECT_EQ(harness.native.live_memories, live_memories);
        }
      }
      EXPECT_EQ(harness.native.result.collects, 0u);
      EXPECT_EQ(harness.native.result.detaches, 0u);
      EXPECT_EQ(harness.diagnostic_count, 0u);
      iree_hal_executable_t* executable = nullptr;
      iree_hal_executable_function_t function;
      ASSERT_NO_FATAL_FAILURE(
          harness.LoadExecutable(MeasuredFixture(), &executable, &function));
      iree_hal_buffer_t* buffer = nullptr;
      ASSERT_NO_FATAL_FAILURE(harness.MakeBuffer(&buffer));
      const auto binding = iree_hal_make_buffer_ref(buffer, 0, 64);
      uint64_t value = 1;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          harness.queue, {}, {1, &harness.done, &value}, executable, function,
          iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, {},
          0));
      ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, 1));

      harness.native.result.unsupported_attachment = false;
      IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
      value = 2;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          harness.queue, {}, {1, &harness.done, &value}, executable, function,
          iree_hal_make_static_dispatch_config(1, 1, 1), {}, {1, &binding}, {},
          0));
      iree_hal_executable_release(executable);
      iree_hal_buffer_release(buffer);
      ASSERT_NO_FATAL_FAILURE(harness.PollUntilValue(harness.done, value));
      IREE_ASSERT_OK(iree_hal_device_profiling_end(harness.device));
      EXPECT_EQ(sink.samples.size(), 1u);
      EXPECT_EQ(harness.native.result.attaches, 4u);
      EXPECT_EQ(harness.native.result.collects, 1u);
      EXPECT_EQ(harness.native.result.detaches, 1u);
      harness.ReleaseDevice();
      ASSERT_NO_FATAL_FAILURE(harness.PollUntilDeviceDestroyed());
      EXPECT_EQ(harness.native.live_memories, 0u);
    }
    EXPECT_EQ(allocations.live_allocations.load(), 0);
  }
}

static void RunAmbiguousResultControlChild(uint32_t control,
                                           amdf_status_t native_status,
                                           iree_status_code_t expected_status) {
  ProfileSink sink;
  QueueHarness harness;
  struct AllocationWatch {
    // Queue allocation whose release proves finalization reached its callback.
    void* pointer = nullptr;
    // Set by the allocator after the queue allocation is released.
    bool released = false;
  } watch;
  const iree_allocator_t allocator = {
      &watch,
      +[](void* self, iree_allocator_command_t command, const void* params,
          void** pointer) -> iree_status_t {
        auto& watch = *static_cast<AllocationWatch*>(self);
        const bool watched_free = command == IREE_ALLOCATOR_COMMAND_FREE &&
                                  pointer && *pointer == watch.pointer;
        auto allocator = iree_allocator_system();
        iree_status_t status =
            allocator.ctl(allocator.self, command, params, pointer);
        if (watched_free && iree_status_is_ok(status)) {
          watch.released = true;
        }
        return status;
      },
  };
  [&] {
    ASSERT_NO_FATAL_FAILURE(harness.Initialize(allocator));
    watch.pointer = harness.queue;
    harness.native.result.fail_control = control;
    harness.native.result.failure_status = native_status;
    const iree_hal_device_profiling_options_t options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES,
        .sink = sink.base(),
    };
    if (control == 1) {
      IREE_EXPECT_STATUS_IS(expected_status, iree_hal_device_profiling_begin(
                                                 harness.device, &options));
    } else {
      IREE_ASSERT_OK(iree_hal_device_profiling_begin(harness.device, &options));
      IREE_EXPECT_STATUS_IS(expected_status,
                            iree_hal_device_profiling_end(harness.device));
    }
    EXPECT_TRUE(harness.context->result_buffer_retained);
    EXPECT_EQ(harness.native.result.collects, control == 1 ? 0u : 1u);
    EXPECT_EQ(harness.native.result.detaches, control == 3 ? 1u : 0u);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_device_profiling_begin(harness.device, &options));
    const size_t memories = harness.native.live_memories;
    harness.ReleaseDevice();
    while (!watch.released) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          harness.proactor, iree_infinite_timeout(), nullptr));
    }
    EXPECT_EQ(harness.native.live_memories, memories);
    EXPECT_EQ(harness.native.context_destroy_count, 0u);
    EXPECT_EQ(harness.diagnostic_count, 1u);
  }();
  // Retention is the behavior under test. Never run ordinary harness teardown,
  // which waits for destruction that this native failure cannot authorize.
  std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
}

TEST(XdnaProfileDeathTest, AmbiguousControlsRetainBackingAndParents) {
  for (uint32_t control : {1u, 2u, 3u}) {
    EXPECT_EXIT(RunAmbiguousResultControlChild(
                    control, amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL),
                    IREE_STATUS_INTERNAL),
                ::testing::ExitedWithCode(0), "");
  }
  for (uint32_t control : {2u, 3u}) {
    EXPECT_EXIT(RunAmbiguousResultControlChild(
                    control, amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED),
                    IREE_STATUS_UNIMPLEMENTED),
                ::testing::ExitedWithCode(0), "");
  }
}

}  // namespace
}  // namespace iree::hal::amd::xdna::testing
