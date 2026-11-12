// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using EventPtr =
    hrx::runtime::hrx_ptr<hrx_event_t, hrx_event_retain, hrx_event_release>;

struct SignalOnExit {
  ~SignalOnExit() {
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_semaphore_signal(semaphore, 1)));
  }
  // Borrowed host-controlled dependency released even on assertion failure.
  hrx_semaphore_t semaphore;
};

class EventTest : public ::testing::Test {
 protected:
  // Owns the task device and its runtime through fixture cleanup.
  inline static hrx_device_group_t group_ = nullptr;

  static void SetUpTestSuite() {
    hrx_runtime_t runtime = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
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

  void SetUp() override {
    hrx::runtime::queue_ptr queue;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_acquire(device_, 0, nullptr, queue.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_create(queue.get(), 0, first_.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_acquire(device_, 0, nullptr, queue.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_create(queue.get(), 0, second_.for_output())));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_create(
        device_, HRX_EVENT_FLAG_DISABLE_TIMING, event_.for_output())));
  }

  // Runtime-owned Task device shared by every case.
  static hrx_device_t device_;
  // Independent exact queues with distinct logical timelines.
  hrx::runtime::stream_ptr first_;
  // Second stream, whose initial timeline is behind |first_| in the reuse test.
  hrx::runtime::stream_ptr second_;
  // Event retaining the point captured by its most recent recording.
  EventPtr event_;
};

hrx_device_t EventTest::device_ = nullptr;

TEST_F(EventTest, EmptyRecordDoesNotAdvanceTheStream) {
  bool complete = false;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_query(event_.get(), &complete)));
  EXPECT_TRUE(complete);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_.get())));

  hrx_timeline_point_t before;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(first_.get(), &before)));
  for (int i = 0; i < 8; ++i) {
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(event_.get(), first_.get())));
  }
  hrx_timeline_point_t after;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(first_.get(), &after)));
  EXPECT_EQ(after.semaphore, before.semaphore);
  EXPECT_EQ(after.value, before.value);
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_query(event_.get(), &complete)));
  EXPECT_TRUE(complete);
}

TEST_F(EventTest, ReRecordCannotReuseACompletedValueFromAnotherStream) {
  hrx::runtime::buffer_ptr buffer;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_allocate(first_.get(), 64, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, buffer.for_output())));
  for (int i = 0; i < 8; ++i) {
    const uint32_t pattern = static_cast<uint32_t>(i);
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        first_.get(), buffer.get(), 0, 64, &pattern, sizeof(pattern))));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(event_.get(), first_.get())));
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_.get())));

  hrx::runtime::semaphore_ptr gate;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_create(device_, 0, gate.for_output())));
  const uint32_t pattern = 0xAABBCCDD;
  {
    SignalOnExit signal{gate.get()};
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_wait_on(second_.get(), {gate.get(), 1})));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        second_.get(), buffer.get(), 0, 64, &pattern, sizeof(pattern))));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(event_.get(), second_.get())));
    bool complete = true;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_query(event_.get(), &complete)));
    EXPECT_FALSE(complete);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_.get())));
  std::array<uint32_t, 16> output = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(hrx_stream_get_queue(second_.get()), buffer.get(), 0,
                          output.data(), sizeof(output))));
  for (uint32_t value : output) {
    EXPECT_EQ(value, pattern);
  }
}

TEST_F(EventTest, ReplacementDoesNotOrderProducersOrChangeAnEnqueuedWait) {
  hrx::runtime::buffer_ptr buffer;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
      first_.get(), 2 * sizeof(uint32_t), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, buffer.for_output())));
  hrx::runtime::queue_ptr queue;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_queue_acquire(device_, 0, nullptr, queue.for_output())));
  hrx::runtime::stream_ptr consumer;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_create(queue.get(), 0, consumer.for_output())));
  hrx::runtime::semaphore_ptr gate;
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_semaphore_create(device_, 0, gate.for_output())));

  const uint32_t first_pattern = 0x11223344;
  const uint32_t second_pattern = 0xAABBCCDD;
  {
    SignalOnExit signal{gate.get()};
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_wait_on(first_.get(), {gate.get(), 1})));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        first_.get(), buffer.get(), 0, sizeof(first_pattern), &first_pattern,
        sizeof(first_pattern))));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(event_.get(), first_.get())));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_wait_event(consumer.get(), event_.get())));

    // B's actual work completes while A's producer is still gated. Recording
    // that completed point cannot reintroduce A as a dependency of B.
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_fill_buffer(
        second_.get(), buffer.get(), sizeof(first_pattern),
        sizeof(second_pattern), &second_pattern, sizeof(second_pattern))));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(second_.get())));
    hrx_timeline_point_t before;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_get_timeline_position(second_.get(), &before)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_record(event_.get(), second_.get())));
    hrx_timeline_point_t after;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_stream_get_timeline_position(second_.get(), &after)));
    EXPECT_EQ(after.semaphore, before.semaphore);
    EXPECT_EQ(after.value, before.value);
    bool complete = false;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_event_query(event_.get(), &complete)));
    EXPECT_TRUE(complete);

    // The accepted consumer wait still names A even after replacement and
    // release of the public event handle.
    event_.reset();
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_stream_query(consumer.get(), &complete)));
    EXPECT_FALSE(complete);
  }
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_stream_synchronize(consumer.get())));
  std::array<uint32_t, 2> output = {};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_synchronous_d2h(hrx_stream_get_queue(second_.get()), buffer.get(), 0,
                          output.data(), sizeof(output))));
  EXPECT_EQ(output[0], first_pattern);
  EXPECT_EQ(output[1], second_pattern);
}

TEST_F(EventTest, RejectedRecordPreservesTheAcceptedEventAndStreamPositions) {
  hrx::runtime::buffer_ptr buffer;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
      first_.get(), sizeof(uint32_t), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, buffer.for_output())));
  const uint32_t pattern = 0x12345678;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_fill_buffer(first_.get(), buffer.get(), 0, sizeof(pattern),
                             &pattern, sizeof(pattern))));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_event_record(event_.get(), first_.get())));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_.get())));
  hrx_timeline_point_t before;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(second_.get(), &before)));
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_fill_buffer(second_.get(), buffer.get(), 0, sizeof(pattern),
                             &pattern, sizeof(pattern))));

  // Model a queue dependency rejecting before acceptance. Real Task queues,
  // semaphores, and public stream/event ownership remain intact.
  static const iree_hal_queue_vtable_t vtable = {
      .destroy = [](iree_hal_queue_t*) {},
      .execute =
          [](iree_hal_queue_t*, iree_hal_semaphore_list_t,
             iree_hal_semaphore_list_t, iree_hal_command_buffer_t*,
             iree_hal_buffer_binding_table_t, iree_hal_queue_execute_flags_t) {
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "submission storage exhausted");
          },
  };
  iree_hal_queue_t rejected_queue;
  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  iree_hal_queue_t* native_queue = second_.get()->hal_queue;
  iree_hal_queue_initialize(iree_hal_queue_family(native_queue), &params,
                            &vtable, &rejected_queue);
  second_.get()->hal_queue = &rejected_queue;
  hrx_status_t status = hrx_event_record(event_.get(), second_.get());
  second_.get()->hal_queue = native_queue;
  iree_hal_queue_release(&rejected_queue);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        hrx_status_to_iree(status));

  hrx_timeline_point_t after;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_get_timeline_position(second_.get(), &after)));
  EXPECT_EQ(after.semaphore, before.semaphore);
  EXPECT_EQ(after.value, before.value);
  if (after.value != before.value) {
    // No work was accepted. Release a wrongly reserved wait on a regression
    // so failure diagnostics are not lost to a teardown hang.
    iree_hal_semaphore_fail(second_.get()->semaphore->hal_semaphore,
                            iree_make_status(IREE_STATUS_ABORTED));
    return;
  }
  bool complete = false;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_query(event_.get(), &complete)));
  EXPECT_TRUE(complete);
  // The rejected one-shot recording was consumed. A fresh batch can record
  // and submit without trying to append to or end the rejected recording.
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_fill_buffer(second_.get(), buffer.get(), 0, sizeof(pattern),
                             &pattern, sizeof(pattern))));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_event_record(event_.get(), second_.get())));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_.get())));
}

TEST_F(EventTest, CompletedEventSurvivesItsRecordingStream) {
  hrx::runtime::buffer_ptr buffer;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_allocate(
      first_.get(), sizeof(uint32_t), HRX_MEMORY_TYPE_DEVICE_LOCAL,
      HRX_BUFFER_USAGE_DEFAULT, buffer.for_output())));
  const uint32_t pattern = 0x12345678;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_stream_fill_buffer(first_.get(), buffer.get(), 0, sizeof(pattern),
                             &pattern, sizeof(pattern))));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_event_record(event_.get(), first_.get())));
  first_.reset();
  bool complete = false;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_query(event_.get(), &complete)));
  EXPECT_TRUE(complete);
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_event_record(event_.get(), second_.get())));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event_.get())));
}

TEST(EventLifecycleTest, UnrecordedEventRetainsItsCreationDevice) {
  hrx_runtime_t runtime = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_runtime_create(nullptr, hrx_host_allocator_system(), &runtime)));
  const hrx_device_create_params_t params = {.uri = "task://"};
  hrx_device_group_t group = nullptr;
  hrx_status_t status = hrx_device_group_create(runtime, 1, &params, &group);
  hrx_runtime_release(runtime);
  IREE_ASSERT_OK(hrx_status_to_iree(status));
  EventPtr event;
  status = hrx_event_create(hrx_device_group_get_device(group, 0),
                            HRX_EVENT_FLAG_DISABLE_TIMING, event.for_output());
  hrx_device_group_release(group);
  IREE_ASSERT_OK(hrx_status_to_iree(status));
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_event_synchronize(event.get())));
}

}  // namespace
