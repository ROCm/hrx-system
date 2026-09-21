// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/kernel_queue.h"

#include <drm/amdxdna_accel.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/drm/context.h"
#include "libamdf/src/xdna/umd/drm/elf_packet.h"

namespace {

// Native dependencies are replaced at link time in this Linux-only test. The
// shipping queue, packet framing, deadline handling and atomics execute intact.
struct NativeState {
  // Distinct cache-line-aligned command BOs supplied by the native dependency.
  struct Packet {
    // CPU mapping kept stable through all native operations in one case.
    alignas(64) std::array<uint32_t, 1024> words = {};
    // Whether the native allocation has been released.
    bool released = false;
  };
  // Fixed dependency storage large enough for creation and reacquisition.
  std::array<Packet, 16> packets;
  // Number of command BOs created, including those later released.
  uint32_t packet_count = 0;
  // Allocation attempt that fails, or UINT32_MAX when all attempts succeed.
  uint32_t failed_packet = UINT32_MAX;
  // Command BO handle for each accepted native sequence.
  std::vector<uint32_t> accepted_handles;
  // Number of completed native commands, including sequence zero.
  uint64_t completed_count = 0;
  // Fence captured by the first-point binary object.
  uint64_t captured_count = 0;
  // Native transfer failure before any following command is accepted.
  int capture_error = 0;
  // Number of first-point native fence transfers.
  uint32_t capture_count = 0;
  // Handle selected by the last native wait.
  uint32_t waited_handle = 0;
  // Number of binary first-fence objects released.
  uint32_t snapshot_release_count = 0;
  // Captured command payload at native acceptance.
  std::array<uint32_t, 12> submitted_packet = {};
  // Number of accepted native submissions.
  uint64_t submission_count = 0;
  // Native sequence observed by the last timeline ioctl.
  uint64_t waited_sequence = UINT64_MAX;
  // Number of timeline ioctls issued.
  uint32_t wait_count = 0;
  // Error returned by native submission, or zero for acceptance.
  int submission_error = 0;
  // Error returned by native timeline wait, or zero for completion.
  int wait_error = 0;
  // Error from GEM close after the packet's host view is unmapped.
  int packet_release_error = 0;
  // Number of final packet release attempts.
  uint32_t packet_release_count = 0;
  // Handle-zero capability response; ENOENT means the ioctl is implemented.
  int notification_probe_error = ENOENT;
  // Native notification registration error, or zero for acceptance.
  int notification_error = 0;
  // Number of actual registrations, excluding capability discovery.
  uint32_t notification_count = 0;
  // Exact syncobj and timeline point supplied to native registration.
  struct {
    // Requested syncobj handle.
    uint32_t handle = 0;
    // Requested point in the native object's domain.
    uint64_t point = UINT64_MAX;
  } notification;
  // Test synchronization protecting the controlled native wait.
  std::mutex mutex;
  // Notification when the native wait enters or may return.
  std::condition_variable changed;
  // Whether the native wait has reached the driver boundary.
  bool wait_entered = false;
  // Whether the driver may complete the current native wait.
  bool complete_wait = true;
  // Whether the first-fence transfer has reached the driver boundary.
  bool capture_entered = false;
  // Whether the driver may finish the first-fence transfer.
  bool complete_capture = true;
};

// Each case owns this dependency state through all joined native operations.
NativeState* native_state = nullptr;

class LinuxXdnaKernelQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_state = &native_;
    device_.host_allocator = amdf_allocator_system();
    device_.descriptor = 42;
    device_.page_size = 4096;
    device_.cache_line_size = 64;
    context_.device = &device_;
    context_.handle = 7;
    context_.completion_syncobj = 8;
    ASSERT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, 3, &queue_),
              AMDF_STATUS_OK);
    ASSERT_EQ(
        amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline_),
        AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (queue_ != nullptr) {
      EXPECT_EQ(amdf_xdna_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(amdf_atomic_uint32_load_acquire(&context_.queue_leased), 0u);
    if (event_.type != AMDF_NATIVE_EVENT_TYPE_NONE) {
      EXPECT_EQ(close(static_cast<int>(event_.payload.file_descriptor)), 0);
    }
    native_state = nullptr;
  }

  void CreateEvent() {
    const int descriptor = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    ASSERT_GE(descriptor, 0);
    event_.type = AMDF_NATIVE_EVENT_TYPE_EVENTFD;
    event_.payload.file_descriptor = descriptor;
  }

  void ConsumeEvent() {
    eventfd_t value = 0;
    ASSERT_EQ(
        eventfd_read(static_cast<int>(event_.payload.file_descriptor), &value),
        0);
    EXPECT_GT(value, 0u);
  }

  // Controlled native dependency state.
  NativeState native_;
  // Borrowed native device and context with no live hardware resources.
  amdf_xdna_umd_device_t device_ = {};
  // Context owning the exclusive queue lease.
  amdf_xdna_umd_context_t context_ = {};
  // Actual queue implementation under test.
  amdf_xdna_umd_kernel_queue_t* queue_ = nullptr;
  // Infinite production wait; the outer test harness catches hangs.
  amdf_wait_deadline_t deadline_ = {};
  // Caller-owned native event, live through queue teardown.
  amdf_native_event_t event_ = {};
};

TEST_F(LinuxXdnaKernelQueueTest,
       NotificationUsesExactFirstSnapshotAndLaterPoint) {
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  EXPECT_NE(amdf_xdna_umd_kernel_queue_query_notification_types(queue_) &
                AMDF_NATIVE_EVENT_TYPE_BIT_EVENTFD,
            0u);
  uint64_t first = 0, second = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  ASSERT_EQ(
      amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_),
      AMDF_STATUS_OK);
  EXPECT_EQ(native_.notification.handle, 100u);
  EXPECT_EQ(native_.notification.point, 0u);
  EXPECT_EQ(native_.captured_count, 1u);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &second),
            AMDF_STATUS_OK);
  ASSERT_EQ(
      amdf_xdna_umd_kernel_queue_request_notification(queue_, second, &event_),
      AMDF_STATUS_OK);
  EXPECT_EQ(native_.notification.handle, 8u);
  EXPECT_EQ(native_.notification.point, 1u);
  EXPECT_EQ(native_.notification_count, 2u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(native_.wait_count, 0u);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, second, &deadline_),
            AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 1);
  for (uint32_t i = 0; i < 2; ++i) {
    ASSERT_EQ(
        amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(ConsumeEvent());
  }
  EXPECT_EQ(native_.notification_count, 2u);
}

TEST_F(LinuxXdnaKernelQueueTest,
       FirstCaptureContentionProvidesImmediateRecheck) {
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  native_.complete_capture = false;
  amdf_status_t waiter_status = AMDF_STATUS_OK;
  std::thread waiter([&] {
    waiter_status = amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_);
  });
  {
    std::unique_lock<std::mutex> lock(native_.mutex);
    native_.changed.wait(lock, [&] { return native_.capture_entered; });
  }
  EXPECT_EQ(
      amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_),
      AMDF_STATUS_OK);
  ConsumeEvent();
  EXPECT_EQ(native_.notification_count, 0u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  {
    std::lock_guard<std::mutex> lock(native_.mutex);
    native_.complete_capture = true;
  }
  native_.changed.notify_all();
  waiter.join();
  EXPECT_EQ(waiter_status, AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  ASSERT_EQ(
      amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_),
      AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ConsumeEvent());
}

TEST_F(LinuxXdnaKernelQueueTest,
       NotificationCaptureProtectsFirstFenceFromPublication) {
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  native_.complete_capture = false;
  amdf_status_t notification_status = AMDF_STATUS_OK;
  std::thread notifier([&] {
    notification_status =
        amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_);
  });
  {
    std::unique_lock<std::mutex> lock(native_.mutex);
    native_.changed.wait(lock, [&] { return native_.capture_entered; });
  }
  uint64_t second = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &second),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(second, UINT64_MAX);
  EXPECT_EQ(native_.submission_count, 1u);
  {
    std::lock_guard<std::mutex> lock(native_.mutex);
    native_.complete_capture = true;
  }
  native_.changed.notify_all();
  notifier.join();
  ASSERT_EQ(notification_status, AMDF_STATUS_OK);
  EXPECT_EQ(native_.notification.handle, 100u);
  EXPECT_EQ(native_.notification.point, 0u);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &second),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.captured_count, 1u);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, second, &deadline_),
            AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 1);
}

TEST_F(LinuxXdnaKernelQueueTest, NotificationFailuresPreserveAcceptedIdentity) {
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  native_.capture_error = ENOMEM;
  EXPECT_EQ(
      amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_),
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.notification_count, 0u);
  native_.capture_error = 0;
  native_.notification_error = ENOMEM;
  EXPECT_EQ(
      amdf_xdna_umd_kernel_queue_request_notification(queue_, first, &event_),
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(native_.submission_count, 1u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_),
            AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
}

TEST_F(LinuxXdnaKernelQueueTest,
       MissingNotificationSupportDoesNotDisableExecution) {
  for (const int error : {EINVAL, ENOTTY, EOPNOTSUPP}) {
    ASSERT_EQ(amdf_xdna_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    queue_ = nullptr;
    native_.notification_probe_error = error;
    ASSERT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, 1, &queue_),
              AMDF_STATUS_OK);
    EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_notification_types(queue_), 0u);
    uint64_t submission = 0;
    ASSERT_EQ(
        amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &submission),
        AMDF_STATUS_OK);
    ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, submission, &deadline_),
              AMDF_STATUS_OK);
    amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  }
}

TEST_F(LinuxXdnaKernelQueueTest, UnexpectedProbeErrorRollsBackConstruction) {
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  native_.notification_probe_error = EIO;
  auto* output = reinterpret_cast<amdf_xdna_umd_kernel_queue_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, 1, &output),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO));
  EXPECT_EQ(output,
            reinterpret_cast<amdf_xdna_umd_kernel_queue_t*>(uintptr_t{1}));
  EXPECT_EQ(amdf_atomic_uint32_load_acquire(&context_.queue_leased), 0u);
  EXPECT_EQ(native_.packet_count, 3u);
}

TEST_F(LinuxXdnaKernelQueueTest, FullReadableEventCoalescesAnImmediateHint) {
  ASSERT_NO_FATAL_FAILURE(CreateEvent());
  ASSERT_EQ(eventfd_write(static_cast<int>(event_.payload.file_descriptor),
                          UINT64_MAX - 1),
            0);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_request_notification(queue_, 0, &event_),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(ConsumeEvent());
}

TEST_F(LinuxXdnaKernelQueueTest, FailedPacketReleaseConsumesContextLease) {
  native_.packet_release_error = EBUSY;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_destroy(queue_),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY));
  queue_ = nullptr;
  EXPECT_EQ(native_.packet_release_count, 3u);
  EXPECT_EQ(amdf_atomic_uint32_load_acquire(&context_.queue_leased), 0u);

  native_.packet_release_error = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, 3, &queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.packet_release_count, 3u);
  EXPECT_EQ(amdf_atomic_uint32_load_acquire(&context_.queue_leased), 1u);
}

TEST_F(LinuxXdnaKernelQueueTest, FramesByteLengthAndNoIndirectBufferList) {
  uint64_t sequence = UINT64_MAX;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(
                queue_, 0, UINT64_C(0x1234567887650000), 260, &sequence),
            AMDF_STATUS_OK);
  EXPECT_EQ(sequence, 1u);
  const std::array<uint32_t, 12> expected = {
      0x3B00B001, 1, 0x87650000, 0x12345678, 0, 0, 0, 0, 260, 0, 0, 0};
  EXPECT_EQ(native_.submitted_packet, expected);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, sequence, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.waited_sequence, 0u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 1u);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
}

TEST_F(LinuxXdnaKernelQueueTest,
       FirstPointObserverDoesNotBlockLaterPublication) {
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  native_.complete_wait = false;
  amdf_status_t wait_status = AMDF_STATUS_OK;
  std::thread observer([&] {
    wait_status = amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_);
  });
  {
    std::unique_lock<std::mutex> lock(native_.mutex);
    native_.changed.wait(lock, [&] { return native_.wait_entered; });
  }
  const auto packet = native_.packets[0].words;
  uint64_t second = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &second),
            AMDF_STATUS_OK);
  EXPECT_EQ(second, 2u);
  EXPECT_EQ(native_.submission_count, 2u);
  EXPECT_EQ(native_.packets[0].words, packet);
  EXPECT_EQ(native_.captured_count, 1u);
  {
    std::lock_guard<std::mutex> lock(native_.mutex);
    native_.complete_wait = true;
    native_.changed.notify_all();
  }
  observer.join();
  ASSERT_EQ(wait_status, AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 1u);
  EXPECT_EQ(native_.waited_handle, 100u);
  // A late observer of point zero must consume cached retirement, not DRM's
  // new latest fence. No second timeline ioctl is issued for the old point.
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.wait_count, 1u);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, second, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.waited_sequence, 1u);
  EXPECT_EQ(native_.waited_handle, context_.completion_syncobj);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 1);
}

TEST_F(LinuxXdnaKernelQueueTest,
       FirstFenceCaptureRejectsConcurrentReplacement) {
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  native_.complete_capture = false;
  amdf_status_t wait_status = AMDF_STATUS_OK;
  std::thread observer([&] {
    wait_status = amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_);
  });
  {
    std::unique_lock<std::mutex> lock(native_.mutex);
    native_.changed.wait(lock, [&] { return native_.capture_entered; });
  }
  uint64_t second = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &second),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(second, UINT64_MAX);
  EXPECT_EQ(native_.submission_count, 1u);
  {
    std::lock_guard<std::mutex> lock(native_.mutex);
    native_.complete_capture = true;
  }
  native_.changed.notify_all();
  observer.join();
  ASSERT_EQ(wait_status, AMDF_STATUS_OK);
  EXPECT_EQ(native_.captured_count, 1u);
  EXPECT_EQ(native_.capture_count, 1u);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &second),
            AMDF_STATUS_OK);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, second, &deadline_),
            AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 1);
}

TEST_F(LinuxXdnaKernelQueueTest, RejectionDoesNotPublishOrConsumeSequence) {
  native_.submission_error = EBUSY;
  uint64_t sequence = UINT64_MAX;
  EXPECT_NE(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &sequence),
            AMDF_STATUS_OK);
  EXPECT_EQ(sequence, UINT64_MAX);
  EXPECT_EQ(context_.last_native_sequence, 0u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
}

TEST_F(LinuxXdnaKernelQueueTest, WaitFailurePreservesPendingProgress) {
  uint64_t sequence = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &sequence),
            AMDF_STATUS_OK);
  native_.wait_error = EIO;
  EXPECT_NE(amdf_xdna_umd_kernel_queue_wait(queue_, sequence, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  native_.wait_error = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, sequence, &deadline_),
            AMDF_STATUS_OK);
  native_.packets[0].words[0] = (native_.packets[0].words[0] & ~15u) | 6u;
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_terminal_status(queue_),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 6));
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &rejected),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 6));
  EXPECT_EQ(rejected, UINT64_MAX);
}

TEST_F(LinuxXdnaKernelQueueTest, SnapshotFailureRejectsOnlyTheNextSubmission) {
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  native_.capture_error = ENOMEM;
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &rejected),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(rejected, UINT64_MAX);
  EXPECT_EQ(native_.submission_count, 1u);
  EXPECT_EQ(context_.last_native_sequence, first);
  native_.capture_error = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 1, 65536, 64, &rejected),
            AMDF_STATUS_OK);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.waited_handle, 100u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), first);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, rejected, &deadline_),
            AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 1);
}

TEST_F(LinuxXdnaKernelQueueTest, ZeroQueryRequiresExactFirstFenceProof) {
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 0, 32768, 64, &first),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_refresh_progress(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  EXPECT_EQ(native_.captured_count, 1u);
  native_.completed_count = 1;
  native_.packets[0].words[0] = (native_.packets[0].words[0] & ~15u) | 4u;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_refresh_progress(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), first);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 0);
}

TEST_F(LinuxXdnaKernelQueueTest, QueryRecoversACompletedPrefixWithoutWait) {
  uint64_t submission = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, i, 32768 + i * 64, 64,
                                                &submission),
              AMDF_STATUS_OK);
  }
  native_.completed_count = 2;
  for (uint32_t i = 0; i < 2; ++i) {
    native_.packets[i].words[0] = (native_.packets[i].words[0] & ~15u) | 4u;
  }
  amdf_wait_deadline_t immediate;
  ASSERT_EQ(amdf_wait_deadline_initialize(0, 0, &immediate), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, submission, &immediate),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  const uint32_t wait_count = native_.wait_count;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_refresh_progress(queue_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 2u);
  EXPECT_EQ(native_.wait_count, wait_count);
  for (uint32_t i = 0; i < 2; ++i) {
    amdf_xdna_umd_kernel_queue_retire_command(queue_, i);
  }
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, submission, &deadline_),
            AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_, 2);
}

TEST_F(LinuxXdnaKernelQueueTest,
       ConstructionFailureReleasesOnlyAcquiredPackets) {
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
  queue_ = nullptr;
  native_.failed_packet = native_.packet_count + 1;
  amdf_xdna_umd_kernel_queue_t* output =
      reinterpret_cast<amdf_xdna_umd_kernel_queue_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, 3, &output),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM));
  EXPECT_EQ(output,
            reinterpret_cast<amdf_xdna_umd_kernel_queue_t*>(uintptr_t{1}));
  EXPECT_EQ(native_.packet_release_count, 4u);
  EXPECT_EQ(native_.snapshot_release_count, 2u);
  EXPECT_EQ(amdf_atomic_uint32_load_acquire(&context_.queue_leased), 0u);
  native_.failed_packet = UINT32_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, 3, &queue_),
            AMDF_STATUS_OK);
}

}  // namespace

extern "C" amdf_status_t amdf_linux_xdna_buffer_create(
    int descriptor, uint32_t type, size_t byte_length,
    amdf_linux_xdna_buffer_t* out_buffer) {
  EXPECT_EQ(descriptor, 42);
  EXPECT_EQ(type, AMDXDNA_BO_CMD);
  EXPECT_EQ(byte_length, 4096u);
  if (native_state->packet_count == native_state->failed_packet) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM);
  }
  *out_buffer = {};
  out_buffer->handle = 9 + native_state->packet_count++;
  out_buffer->byte_length = byte_length;
  return AMDF_STATUS_OK;
}

extern "C" amdf_status_t amdf_linux_xdna_buffer_attach(
    int descriptor, size_t alignment, size_t page_size,
    const amdf_linux_xdna_buffer_t* heap, amdf_linux_xdna_buffer_t* buffer) {
  EXPECT_EQ(heap, nullptr);
  buffer->host_pointer = native_state->packets[buffer->handle - 9].words.data();
  return AMDF_STATUS_OK;
}

extern "C" amdf_status_t amdf_linux_xdna_buffer_deinitialize(
    int descriptor, amdf_linux_xdna_buffer_t* buffer) {
  EXPECT_FALSE(native_state->packets[buffer->handle - 9].released);
  native_state->packets[buffer->handle - 9].released = true;
  ++native_state->packet_release_count;
  buffer->host_pointer = nullptr;
  if (native_state->packet_release_error != 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO,
                            native_state->packet_release_error);
  }
  *buffer = {};
  return AMDF_STATUS_OK;
}

extern "C" int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  EXPECT_EQ(descriptor, 42);
  NativeState& native = *native_state;
  if (request == DRM_IOCTL_SYNCOBJ_EVENTFD) {
    const auto* notify = static_cast<const drm_syncobj_eventfd*>(argument);
    EXPECT_EQ(notify->flags, 0u);
    EXPECT_EQ(notify->pad, 0u);
    if (notify->handle == 0) {
      EXPECT_EQ(notify->fd, -1);
      errno = native.notification_probe_error;
      return -1;
    }
    ++native.notification_count;
    native.notification.handle = notify->handle;
    native.notification.point = notify->point;
    if (native.notification_error != 0) {
      errno = native.notification_error;
      return -1;
    }
    return 0;
  }
  if (request == DRM_IOCTL_SYNCOBJ_CREATE) {
    static_cast<drm_syncobj_create*>(argument)->handle = 100;
    return 0;
  }
  if (request == DRM_IOCTL_SYNCOBJ_DESTROY) {
    EXPECT_EQ(static_cast<drm_syncobj_destroy*>(argument)->handle, 100u);
    ++native.snapshot_release_count;
    return 0;
  }
  if (request == DRM_IOCTL_SYNCOBJ_TRANSFER) {
    std::unique_lock<std::mutex> lock(native.mutex);
    const auto* transfer = static_cast<const drm_syncobj_transfer*>(argument);
    EXPECT_EQ(transfer->src_handle, 8u);
    EXPECT_EQ(transfer->dst_handle, 100u);
    EXPECT_EQ(transfer->src_point, 0u);
    EXPECT_EQ(transfer->dst_point, 0u);
    ++native.capture_count;
    native.capture_entered = true;
    native.changed.notify_all();
    native.changed.wait(lock, [&] { return native.complete_capture; });
    if (native.capture_error) {
      errno = native.capture_error;
      return -1;
    }
    native.captured_count = native.submission_count;
    return 0;
  }
  if (request == DRM_IOCTL_SYNCOBJ_QUERY) {
    auto* query = static_cast<drm_syncobj_timeline_array*>(argument);
    EXPECT_EQ(query->flags, 0u);
    *reinterpret_cast<uint64_t*>(query->points) =
        native.completed_count ? native.completed_count - 1 : 0;
    return 0;
  }
  if (request == DRM_IOCTL_AMDXDNA_EXEC_CMD) {
    auto* submit = static_cast<amdxdna_drm_exec_cmd*>(argument);
    EXPECT_EQ(submit->hwctx, 7u);
    EXPECT_EQ(submit->cmd_count, 1u);
    EXPECT_EQ(submit->args, 0u);
    EXPECT_EQ(submit->arg_count, 0u);
    if (native.submission_error != 0) {
      errno = native.submission_error;
      return -1;
    }
    std::memcpy(native.submitted_packet.data(),
                native.packets[submit->cmd_handles - 9].words.data(),
                sizeof(amdf_linux_xdna_elf_packet_t));
    native.accepted_handles.push_back(
        static_cast<uint32_t>(submit->cmd_handles));
    submit->seq = native.submission_count++;
    return 0;
  }
  if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) {
    auto* wait = static_cast<drm_syncobj_timeline_wait*>(argument);
    std::unique_lock<std::mutex> lock(native.mutex);
    native.waited_sequence = *reinterpret_cast<uint64_t*>(wait->points);
    native.waited_handle = *reinterpret_cast<uint32_t*>(wait->handles);
    const uint64_t target = native.waited_handle == 100
                                ? native.captured_count
                                : native.waited_sequence + 1;
    ++native.wait_count;
    if (wait->timeout_nsec == 0 && native.completed_count < target) {
      errno = ETIME;
      return -1;
    }
    native.wait_entered = true;
    native.changed.notify_all();
    native.changed.wait(lock, [&] { return native.complete_wait; });
    if (native.wait_error != 0) {
      errno = native.wait_error;
      return -1;
    }
    for (uint64_t i = native.completed_count; i < target; ++i) {
      auto& packet = native.packets[native.accepted_handles[i] - 9].words;
      packet[0] = (packet[0] & ~15u) | 4u;
    }
    if (target > native.completed_count) {
      native.completed_count = target;
    }
    return 0;
  }
  ADD_FAILURE() << "Unexpected ioctl " << request;
  errno = EINVAL;
  return -1;
}
