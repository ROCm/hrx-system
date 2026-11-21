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

// HAL dependency recording the public HRX host-wait boundary. It performs no
// native submission; HRX owns policy forwarding and status conversion.
struct WaitSemaphore {
  WaitSemaphore() {
    static const iree_hal_semaphore_vtable_t vtable = {
        .wait = [](iree_hal_semaphore_t* base, uint64_t value,
                   iree_timeout_t timeout,
                   iree_async_wait_flags_t flags) -> iree_status_t {
          auto* self = reinterpret_cast<WaitSemaphore*>(base);
          ++self->calls;
          self->value = value;
          self->timeout = timeout;
          self->flags = flags;
          return iree_status_from_code(self->result);
        },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }
  iree_hal_semaphore_t* get() {
    return reinterpret_cast<iree_hal_semaphore_t*>(this);
  }

  // Borrowed backend resource, alive for the complete host call.
  iree_hal_resource_t resource;
  // Number of waits forwarded to this dependency.
  uint32_t calls = 0;
  // Last requested completion point.
  uint64_t value = 0;
  // Last requested timeout, before any backend-specific deadline conversion.
  iree_timeout_t timeout = {};
  // Last requested caller policy, independent of device-worker policy.
  iree_async_wait_flags_t flags = IREE_ASYNC_WAIT_FLAG_NONE;
  // Backend result returned unchanged through HRX's status representation.
  iree_status_code_t result = IREE_STATUS_OK;
};

TEST(HostWaitTest, ForwardsPolicyTimeoutAndFailure) {
  WaitSemaphore backend;
  hrx_device_s device = {};
  hrx_queue_s queue = {.device = &device};
  hrx_semaphore_s semaphore = {
      .hal_semaphore = backend.get(),
      .device = &device,
  };
  hrx_stream_s stream = {
      .queue = &queue,
      .semaphore = &semaphore,
      .timepoint = 17,
  };
  hrx_event_s event = {.semaphore = &semaphore, .signal_value = 17};
  for (hrx_wait_flags_t flags :
       {HRX_WAIT_FLAG_NONE, HRX_WAIT_FLAG_YIELD, HRX_WAIT_FLAG_ACTIVE,
        HRX_WAIT_FLAG_YIELD | HRX_WAIT_FLAG_ACTIVE}) {
    SCOPED_TRACE(flags);
    for (iree_status_code_t result : {IREE_STATUS_OK, IREE_STATUS_ABORTED}) {
      backend.result = result;
      for (uint64_t timeout_ns : {uint64_t{0}, uint64_t{1000}, UINT64_MAX}) {
        IREE_EXPECT_STATUS_IS(result, hrx_status_to_iree(hrx_semaphore_wait(
                                          &semaphore, 17, timeout_ns, flags)));
        EXPECT_EQ(backend.value, 17u);
        EXPECT_EQ(backend.flags, flags);
        if (timeout_ns == UINT64_MAX) {
          EXPECT_TRUE(iree_timeout_is_infinite(backend.timeout));
        } else if (timeout_ns == 0) {
          EXPECT_TRUE(iree_timeout_is_immediate(backend.timeout));
        } else {
          EXPECT_EQ(backend.timeout.type, IREE_TIMEOUT_RELATIVE);
          EXPECT_EQ(backend.timeout.nanos, 1000);
        }
      }
      IREE_EXPECT_STATUS_IS(
          result, hrx_status_to_iree(hrx_stream_wait(&stream, flags)));
      EXPECT_EQ(backend.flags, flags);
      EXPECT_EQ(backend.value, 17u);
      EXPECT_TRUE(iree_timeout_is_infinite(backend.timeout));
      IREE_EXPECT_STATUS_IS(
          result, hrx_status_to_iree(hrx_stream_synchronize(&stream, flags)));
      EXPECT_EQ(backend.flags, flags);
      IREE_EXPECT_STATUS_IS(
          result, hrx_status_to_iree(hrx_event_synchronize(&event, flags)));
      EXPECT_EQ(backend.flags, flags);
    }
  }
  EXPECT_EQ(backend.calls, 48u);
}

TEST(HostWaitTest, InvalidPolicyDoesNotSubmitOrWait) {
  WaitSemaphore backend;
  hrx_semaphore_s semaphore = {.hal_semaphore = backend.get()};
  hrx_stream_s stream = {.pending_full_barrier = true};
  hrx_event_s event = {};
  hrx_fence_s fence = {};
  constexpr hrx_wait_flags_t invalid = 1u << 31;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_semaphore_wait(&semaphore, 1, 0, invalid)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_stream_synchronize(&stream, invalid)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_stream_wait(&stream, invalid)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_event_synchronize(&event, invalid)));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_fence_wait(&fence, 0, invalid)));
  EXPECT_TRUE(stream.pending_full_barrier);
  EXPECT_EQ(stream.timepoint, 0u);
  EXPECT_EQ(backend.calls, 0u);
}

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
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_copy_h2d_linear(
        producer.get(), source.data(), input.get(), 0, sizeof(source))));

    EventPtr ready;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_create(
        device_, HRX_EVENT_FLAG_DISABLE_TIMING, ready.for_output())));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(ready.get(), producer.get())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_wait_event(retained_consumer.get(), ready.get())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_copy_buffer_linear(retained_consumer.get(), input.get(), 0,
                                      output.get(), 0, sizeof(source))));
    const uint32_t pattern = 0xAABBCCDD;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer_linear(
        retained_consumer.get(), output.get(), 4 * sizeof(uint32_t),
        4 * sizeof(uint32_t), &pattern, sizeof(pattern))));
    const uint32_t update = 0x11223344;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_update_buffer_linear(
        retained_consumer.get(), &update, sizeof(update), output.get(),
        12 * sizeof(uint32_t))));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_copy_d2h_linear(retained_consumer.get(), output.get(), 0,
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

TEST_F(StreamTest, FullBarrierStaysPendingUntilFlushOrFinalRelease) {
  hrx::runtime::stream_ptr stream;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_create(
      hrx_device_get_queue(device_, 0, 0), 0, stream.for_output())));
  hrx_timeline_point_t point;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(stream.get(), &point)));
  auto semaphore = hrx::runtime::semaphore_ptr::borrow(point.semaphore);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_barrier(stream.get(), nullptr)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_barrier(stream.get(), nullptr)));
  EXPECT_EQ(stream.get()->pending_cb, nullptr);
  for (hrx_wait_flags_t flags :
       {HRX_WAIT_FLAG_NONE, HRX_WAIT_FLAG_YIELD, HRX_WAIT_FLAG_ACTIVE}) {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_wait(stream.get(), flags)));
  }
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(stream.get(), &point)));
  EXPECT_EQ(point.value, 0u);

  // Neither a failed first recording nor a rejected direct submission can
  // consume the pending barrier or publish a point.
  const hrx_barrier_t invalid = {.flags = UINT64_MAX};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_stream_barrier(stream.get(), &invalid)));
  EXPECT_EQ(stream.get()->pending_cb, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_stream_flush_with_barrier(
                            stream.get(), &invalid)));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_flush(stream.get())));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_flush(stream.get())));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(stream.get(), &point)));
  EXPECT_EQ(point.value, 1u);

  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_barrier(stream.get(), nullptr)));
  stream.reset();
  uint64_t completed = 0;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_query(semaphore.get(), &completed)));
  EXPECT_EQ(completed, 2u);
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
      .region =
          {
              .row_length = 64,
              .row_count = 1,
              .slice_count = 1,
          },
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
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_copy_d2h_linear(
      stream.get(), buffer.get(), 0, output.data(), sizeof(output))));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_synchronize(stream.get(), HRX_WAIT_FLAG_NONE)));
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
