// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "stream.h"

#include <array>
#include <atomic>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using EventPtr =
    hrx::runtime::hrx_ptr<hrx_event_t, hrx_event_retain, hrx_event_release>;

class StreamTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  inline static hrx_device_group_t group_ = nullptr;

  static void SetUpTestSuite() {
    const hrx_device_event_sink_t sink = {
        .fn =
            [](void*, const hrx_device_event_t* event) {
              if (event->type == HRX_DEVICE_EVENT_TYPE_DRIVER_FAILURE) {
                failure_count_.fetch_add(1, std::memory_order_relaxed);
              }
            },
    };
    const hrx_runtime_options_t options = {.event_sink = sink};
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_runtime_create(&options, hrx_host_allocator_system(), &runtime)));
    const hrx_device_create_params_t params = {.uri = "task://"};
    hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group_);
    hrx_runtime_release(runtime);
    IREE_ASSERT_OK(hrx_status_to_iree(status));
    device_ = hrx_device_group_get_device(group_, 0);
  }

  static void TearDownTestSuite() {
    hrx_device_group_release(group_);
    group_ = nullptr;
    device_ = nullptr;
  }

  // Exercises the same public flow with distinct queues and intentional
  // sharing.
  void RunProducerConsumer(bool share_queue) {
    std::array<uint32_t, 16> source;
    std::array<uint32_t, 16> target = {};
    for (size_t i = 0; i < source.size(); ++i) {
      source[i] = 17 + i * 31;
    }
    hrx::runtime::queue_ptr first;
    hrx::runtime::queue_ptr second;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_acquire(device_, 0, nullptr, first.for_output())));
    if (share_queue) {
      second = first;
    } else {
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_queue_acquire(device_, 0, nullptr, second.for_output())));
    }
    hrx::runtime::stream_ptr producer;
    hrx::runtime::stream_ptr consumer;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_create(first.get(), 0, producer.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_create(second.get(), 0, consumer.for_output())));
    EXPECT_EQ(first.get(), hrx_stream_get_queue(producer.get()));
    EXPECT_EQ(second.get(), hrx_stream_get_queue(consumer.get()));
    hrx::runtime::stream_ptr retained_consumer = consumer;
    consumer.reset();
    first.reset();
    second.reset();

    // Queue ownership survives both the caller's queue references and an
    // intermediate stream release. No native queue is acquired by a stream.
    hrx_queue_params_t params;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_queue_query_params(
        hrx_stream_get_queue(retained_consumer.get()), &params)));
    hrx_device_t device = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_get_device(retained_consumer.get(), &device)));
    EXPECT_EQ(device_, device);

    hrx::runtime::buffer_ptr input;
    hrx::runtime::buffer_ptr output;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
        producer.get(), sizeof(source), HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_BUFFER_USAGE_DEFAULT, input.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
        retained_consumer.get(), sizeof(target), HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_BUFFER_USAGE_DEFAULT, output.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_copy_h2d(
        producer.get(), source.data(), input.get(), 0, sizeof(source))));

    EventPtr ready;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_create(
        device_, HRX_EVENT_FLAG_DISABLE_TIMING, ready.for_output())));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(ready.get(), producer.get())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_wait_event(retained_consumer.get(), ready.get())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_copy_buffer(retained_consumer.get(), input.get(), 0,
                               output.get(), 0, sizeof(source))));
    const uint32_t pattern = 0xAABBCCDD;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        retained_consumer.get(), output.get(), 4 * sizeof(uint32_t),
        4 * sizeof(uint32_t), &pattern, sizeof(pattern))));
    const uint32_t update = 0x11223344;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_update_buffer(
        retained_consumer.get(), &update, sizeof(update), output.get(),
        12 * sizeof(uint32_t))));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_copy_d2h(retained_consumer.get(), output.get(), 0,
                            target.data(), sizeof(target))));

    // Final stream release joins its own work before dropping the acquired
    // queue. The readback storage remains live throughout that boundary.
    retained_consumer.reset();
    producer.reset();
    for (size_t i = 0; i < target.size(); ++i) {
      EXPECT_EQ(target[i], i == 12           ? update
                           : i >= 4 && i < 8 ? pattern
                                             : source[i]);
    }
  }

  // One task device serves the entire suite; streams own their exact queues.
  static hrx_device_t device_;
  // Failures delivered through the runtime's thread-safe event channel.
  static std::atomic<uint32_t> failure_count_;
};

hrx_device_t StreamTest::device_ = nullptr;
std::atomic<uint32_t> StreamTest::failure_count_{0};

TEST_F(StreamTest, AcquiredQueuesPreserveIndependentStreamDependencies) {
  RunProducerConsumer(false);
}

TEST_F(StreamTest, SharedQueuePreservesExplicitStreamDependencies) {
  RunProducerConsumer(true);
}

TEST_F(StreamTest, InvalidCreationClearsOutput) {
  hrx_stream_t stream = reinterpret_cast<hrx_stream_t>(uintptr_t{1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_stream_create(nullptr, 0, &stream)));
  EXPECT_EQ(stream, nullptr);
  EXPECT_EQ(hrx_stream_get_queue(nullptr), nullptr);
}

TEST_F(StreamTest, GraphReplayUsesAnotherQueueInTheRecordingFamily) {
  std::array<uint32_t, 16> output = {};
  hrx::runtime::queue_ptr recording_queue;
  hrx::runtime::queue_ptr launch_queue;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_queue_acquire(device_, 0, nullptr, recording_queue.for_output())));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_queue_acquire(device_, 0, nullptr, launch_queue.for_output())));
  hrx::runtime::stream_ptr stream;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_create(launch_queue.get(), 0, stream.for_output())));
  launch_queue.reset();

  hrx::runtime::buffer_ptr buffer;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_allocate(stream.get(), 64, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, buffer.for_output())));
  using GraphPtr =
      hrx::runtime::hrx_ptr<hrx_graph_t, hrx_graph_retain, hrx_graph_release>;
  using GraphExecPtr =
      hrx::runtime::hrx_ptr<hrx_graph_exec_t, hrx_graph_exec_retain,
                            hrx_graph_exec_release>;
  GraphPtr graph;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_graph_create(device_, 0, graph.for_output())));
  const hrx_graph_fill_buffer_node_attrs_t fill = {
      .dst = {buffer.get(), 0, 64},
      .pattern = 0xAABBCCDD,
      .pattern_size = sizeof(uint32_t),
  };
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_add_fill_buffer_node(graph.get(), nullptr, 0, &fill, nullptr)));
  GraphExecPtr executable;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_graph_instantiate(
      graph.get(), recording_queue.get(), 0, executable.for_output())));
  // Instantiation borrows only the family; it does not own this native queue.
  recording_queue.reset();
  graph.reset();
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_graph_exec_launch(executable.get(), stream.get())));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_copy_d2h(
      stream.get(), buffer.get(), 0, output.data(), sizeof(output))));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(stream.get())));
  for (uint32_t value : output) {
    EXPECT_EQ(value, fill.pattern);
  }
}

TEST_F(StreamTest, FinalReleaseReportsFailedTimeline) {
  hrx::runtime::stream_ptr stream;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_create(
      hrx_device_get_queue(device_, 0, 0), 0, stream.for_output())));
  hrx_semaphore_t semaphore = nullptr;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_get_semaphore(stream.get(), &semaphore)));
  uint64_t value = 0;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_stream_advance_timeline(stream.get(), &value)));
  // The external producer fails before any native work is accepted. This is
  // an ordinary failed timeline, not a simulated loss of in-flight hardware.
  iree_hal_semaphore_fail(
      semaphore->hal_semaphore,
      iree_make_status(IREE_STATUS_ABORTED, "external producer failed"));
  const uint32_t previous_failure_count =
      failure_count_.load(std::memory_order_relaxed);
  stream.reset();
  EXPECT_EQ(failure_count_.load(std::memory_order_relaxed),
            previous_failure_count + 1);
}

}  // namespace
