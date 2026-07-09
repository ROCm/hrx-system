// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for streaming event timing (iree_hal_streaming_event_*).
//
// These tests deliberately exercise the HOST-timing path: a host (local-task)
// device does not advertise a device timestamp domain, so events created on it
// stay host-timed (timestamp_buffer == NULL) and elapsed_time is computed from
// record_time_ns. The device-timestamp path requires an accelerator that
// advertises a device timestamp domain, so this host path is the realistically
// GPU-free-testable behavior of the timing change.
//
// The test brings up a host device via the public hrx CPU lifecycle and wraps
// it in a minimal iree_hal_streaming_device_t so it can drive the internal
// streaming context/stream/event APIs directly.
// iree_hal_streaming_context_create only reads hal_device + ordinal from the
// entry (and the device allocator), and the create/record/elapsed/release
// lifecycle never touches the unset fields when no pageable staging buffer is
// allocated.

#include <string.h>

#include "common/internal.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

// Fixture that owns a host (local-task) device and a streaming context built on
// top of it. The hrx CPU accelerator is a process-global singleton, so the
// fixture reference-counts init/shutdown to stay robust if gtest ever runs the
// suite alongside other hrx consumers in the same binary.
class EventTimingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Bring up the host accelerator. ALREADY_EXISTS means another fixture (or
    // test) already initialized it; treat that as success and skip our matching
    // shutdown in TearDown.
    hrx_status_t cpu_status = hrx_cpu_initialize(0);
    if (hrx_status_is_ok(cpu_status)) {
      owns_cpu_ = true;
    } else if (hrx_status_code(cpu_status) == HRX_STATUS_ALREADY_EXISTS) {
      hrx_status_ignore(cpu_status);
    } else {
      IREE_ASSERT_OK(hrx_to_iree_status(cpu_status));
    }

    hrx_device_t hrx_device = NULL;
    IREE_ASSERT_OK(hrx_to_iree_status(hrx_cpu_device_get(0, &hrx_device)));
    ASSERT_NE(hrx_device, nullptr);

    // Minimal device entry: the streaming context only needs the HAL device and
    // an ordinal. Everything else stays zeroed and is never dereferenced for
    // the create/record/elapsed/release lifecycle exercised here.
    memset(&device_entry_, 0, sizeof(device_entry_));
    device_entry_.ordinal = 0;
    device_entry_.hal_device = hrx_device_hal(hrx_device);
    ASSERT_NE(device_entry_.hal_device, nullptr);

    iree_hal_streaming_context_flags_t flags;
    memset(&flags, 0, sizeof(flags));
    IREE_ASSERT_OK(iree_hal_streaming_context_create(
        &device_entry_, flags, iree_allocator_system(), &context_));

    IREE_ASSERT_OK(iree_hal_streaming_stream_create(
        context_, IREE_HAL_STREAMING_STREAM_FLAG_NONE, /*priority=*/0,
        iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    iree_hal_streaming_stream_release(stream_);
    iree_hal_streaming_context_release(context_);
    if (owns_cpu_) {
      hrx_status_ignore(hrx_cpu_shutdown());
    }
  }

  // Creates an event and records it on the fixture stream.
  iree_hal_streaming_event_t* CreateAndRecord(
      iree_hal_streaming_event_flags_t flags) {
    iree_hal_streaming_event_t* event = NULL;
    IREE_CHECK_OK(iree_hal_streaming_event_create(
        context_, flags, iree_allocator_system(), &event));
    // This fixture's host device does not advertise a device timestamp domain,
    // so events stay host-timed (no device tick buffer) and elapsed_time uses
    // record_time_ns. Pin that here so the suite fails loudly rather than
    // silently changing paths if the host device ever advertises the domain.
    EXPECT_EQ(event->timestamp_buffer, nullptr);
    IREE_CHECK_OK(iree_hal_streaming_event_record(event, stream_));
    return event;
  }

  bool owns_cpu_ = false;
  iree_hal_streaming_device_t device_entry_;
  iree_hal_streaming_context_t* context_ = NULL;
  iree_hal_streaming_stream_t* stream_ = NULL;
};

// Records two events in program order, synchronizes, and asserts the
// host-timing path produces a finite, non-negative duration. The two records
// sample a monotonic host clock (record_time_ns via iree_time_now) in program
// order, so start <= stop; the delta can legitimately be 0ms on a fast host,
// and only a negative or non-finite result is wrong.
TEST_F(EventTimingTest, HostElapsedTimeIsNonNegative) {
  iree_hal_streaming_event_t* start =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);

  // This flush enqueues no ordered device work between the records:
  // event_record submits its barrier directly and there is no pending command
  // buffer. The record ordering comes solely from the monotonic host clock
  // sampled in each event_record.
  IREE_ASSERT_OK(iree_hal_streaming_stream_flush(stream_));

  iree_hal_streaming_event_t* stop =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);

  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(start));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(stop));

  float ms = -1.0f;
  IREE_ASSERT_OK(iree_hal_streaming_event_elapsed_time(&ms, start, stop));
  EXPECT_GE(ms, 0.0f);
  EXPECT_TRUE(ms == ms);           // not NaN
  EXPECT_LT(ms, 60.0f * 1000.0f);  // sane upper bound (< 60s)

  iree_hal_streaming_event_release(start);
  iree_hal_streaming_event_release(stop);
}

// Elapsed time across events recorded on the same stream must be monotonic:
// (start -> stop) is non-negative and (stop -> start) is its negation, matching
// CUDA/HIP signed-duration semantics on the host-timing path.
TEST_F(EventTimingTest, HostElapsedTimeIsSignedAndConsistent) {
  iree_hal_streaming_event_t* first =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);
  IREE_ASSERT_OK(iree_hal_streaming_stream_flush(stream_));
  iree_hal_streaming_event_t* second =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);

  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(first));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(second));

  float forward = 0.0f;
  float backward = 0.0f;
  IREE_ASSERT_OK(
      iree_hal_streaming_event_elapsed_time(&forward, first, second));
  IREE_ASSERT_OK(
      iree_hal_streaming_event_elapsed_time(&backward, second, first));

  EXPECT_GE(forward, 0.0f);
  EXPECT_FLOAT_EQ(backward, -forward);

  iree_hal_streaming_event_release(first);
  iree_hal_streaming_event_release(second);
}

// Elapsed time converts the host record-time delta from ns to ms exactly and is
// signed. record_time_ns is set to known values (white-box) so the scale and
// sign are verified deterministically, independent of wall-clock timing.
TEST_F(EventTimingTest, HostElapsedTimeScaleAndSignAreExact) {
  iree_hal_streaming_event_t* start =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);
  iree_hal_streaming_event_t* stop =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(start));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(stop));

  // 5,000,000 ns == 5.0 ms (both nonzero so the "recorded" guard passes).
  start->record_time_ns = 1000000;
  stop->record_time_ns = 6000000;

  float ms = 0.0f;
  IREE_ASSERT_OK(iree_hal_streaming_event_elapsed_time(&ms, start, stop));
  EXPECT_FLOAT_EQ(ms, 5.0f);
  IREE_ASSERT_OK(iree_hal_streaming_event_elapsed_time(&ms, stop, start));
  EXPECT_FLOAT_EQ(ms, -5.0f);

  iree_hal_streaming_event_release(start);
  iree_hal_streaming_event_release(stop);
}

// An event created with DISABLE_TIMING has no usable timestamp; measuring
// elapsed time with it on either side must be rejected with INVALID_ARGUMENT.
TEST_F(EventTimingTest, DisableTimingRejectsElapsedTime) {
  iree_hal_streaming_event_t* timed =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);
  iree_hal_streaming_event_t* untimed =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING);

  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(timed));
  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(untimed));

  float ms = 0.0f;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_event_elapsed_time(&ms, untimed, timed)),
      StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_event_elapsed_time(&ms, timed, untimed)),
      StatusIs(iree::StatusCode::kInvalidArgument));

  iree_hal_streaming_event_release(timed);
  iree_hal_streaming_event_release(untimed);
}

// Events that have never been recorded have no timestamp; elapsed time must be
// rejected with INVALID_ARGUMENT rather than reporting a bogus duration.
TEST_F(EventTimingTest, UnrecordedEventRejectsElapsedTime) {
  iree_hal_streaming_event_t* recorded =
      CreateAndRecord(IREE_HAL_STREAMING_EVENT_FLAG_NONE);

  iree_hal_streaming_event_t* unrecorded = NULL;
  IREE_ASSERT_OK(iree_hal_streaming_event_create(
      context_, IREE_HAL_STREAMING_EVENT_FLAG_NONE, iree_allocator_system(),
      &unrecorded));

  IREE_ASSERT_OK(iree_hal_streaming_event_synchronize(recorded));

  float ms = 0.0f;
  // Stop event never recorded.
  EXPECT_THAT(iree::Status(iree_hal_streaming_event_elapsed_time(&ms, recorded,
                                                                 unrecorded)),
              StatusIs(iree::StatusCode::kInvalidArgument));
  // Start event never recorded.
  EXPECT_THAT(iree::Status(iree_hal_streaming_event_elapsed_time(
                  &ms, unrecorded, recorded)),
              StatusIs(iree::StatusCode::kInvalidArgument));

  iree_hal_streaming_event_release(recorded);
  iree_hal_streaming_event_release(unrecorded);
}

// Directly covers the device tick->ms conversion math (the device record path
// itself needs an accelerator advertising the timestamp domain, which this
// GPU-free suite cannot provide).
TEST(EventTickConversionTest, TicksToMilliseconds) {
  // 1 GHz: 1 tick == 1 ns, so 5,000,000 ticks == 5 ms.
  EXPECT_FLOAT_EQ(
      iree_hal_streaming_event_ticks_to_ms(1000, 5001000, 1000000000ull), 5.0f);
  // Signed: stop before start is negative.
  EXPECT_FLOAT_EQ(
      iree_hal_streaming_event_ticks_to_ms(5001000, 1000, 1000000000ull),
      -5.0f);
  // 2 GHz: 10,000,000 ticks == 5 ms.
  EXPECT_FLOAT_EQ(
      iree_hal_streaming_event_ticks_to_ms(0, 10000000, 2000000000ull), 5.0f);
}

}  // namespace
