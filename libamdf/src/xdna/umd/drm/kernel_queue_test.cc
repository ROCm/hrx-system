// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/kernel_queue.h"

#include <drm/amdxdna_accel.h>
#include <sys/ioctl.h>

#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/drm/context.h"
#include "libamdf/src/xdna/umd/drm/elf_packet.h"

namespace {

// Native dependencies are replaced at link time in this Linux-only test. The
// shipping queue, packet framing, deadline handling and atomics execute intact.
struct NativeState {
  // CPU-mapped native command BO with cache-line-aligned storage.
  alignas(64) std::array<uint32_t, 1024> packet = {};
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
  // Test synchronization protecting the controlled native wait.
  std::mutex mutex;
  // Notification when the native wait enters or may return.
  std::condition_variable changed;
  // Whether the native wait has reached the driver boundary.
  bool wait_entered = false;
  // Whether the driver may complete the current native wait.
  bool complete_wait = true;
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
    ASSERT_EQ(amdf_xdna_umd_kernel_queue_create(&context_, &queue_),
              AMDF_STATUS_OK);
    ASSERT_EQ(
        amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline_),
        AMDF_STATUS_OK);
  }

  void TearDown() override {
    EXPECT_EQ(amdf_xdna_umd_kernel_queue_destroy(queue_), AMDF_STATUS_OK);
    EXPECT_EQ(amdf_atomic_uint32_load_acquire(&context_.queue_leased), 0u);
    native_state = nullptr;
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
};

TEST_F(LinuxXdnaKernelQueueTest, FramesByteLengthAndNoIndirectBufferList) {
  uint64_t sequence = UINT64_MAX;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(
                queue_, UINT64_C(0x1234567887650000), 260, &sequence),
            AMDF_STATUS_OK);
  EXPECT_EQ(sequence, 1u);
  const std::array<uint32_t, 12> expected = {
      0x3B00B001, 1, 0x87650000, 0x12345678, 0, 0, 0, 0, 260, 0, 0, 0};
  EXPECT_EQ(native_.submitted_packet, expected);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, sequence, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.waited_sequence, 0u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 1u);
  amdf_xdna_umd_kernel_queue_retire_command(queue_);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_terminal_status(queue_),
            AMDF_STATUS_OK);
}

TEST_F(LinuxXdnaKernelQueueTest, FirstPointObserverExcludesFenceReplacement) {
  uint64_t first = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 32768, 64, &first),
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
  const auto packet = native_.packet;
  uint64_t second = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 65536, 64, &second),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(second, UINT64_MAX);
  EXPECT_EQ(native_.submission_count, 1u);
  EXPECT_EQ(native_.packet, packet);
  {
    std::lock_guard<std::mutex> lock(native_.mutex);
    native_.complete_wait = true;
    native_.changed.notify_all();
  }
  observer.join();
  ASSERT_EQ(wait_status, AMDF_STATUS_OK);
  amdf_xdna_umd_kernel_queue_retire_command(queue_);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 65536, 64, &second),
            AMDF_STATUS_OK);
  EXPECT_EQ(second, 2u);
  // A late observer of point zero must consume cached retirement, not DRM's
  // new latest fence. No second timeline ioctl is issued for the old point.
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, first, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.wait_count, 1u);
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, second, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.waited_sequence, 1u);
  amdf_xdna_umd_kernel_queue_retire_command(queue_);
}

TEST_F(LinuxXdnaKernelQueueTest, RejectionDoesNotPublishOrConsumeSequence) {
  native_.submission_error = EBUSY;
  uint64_t sequence = UINT64_MAX;
  EXPECT_NE(amdf_xdna_umd_kernel_queue_submit(queue_, 32768, 64, &sequence),
            AMDF_STATUS_OK);
  EXPECT_EQ(sequence, UINT64_MAX);
  EXPECT_EQ(context_.last_native_sequence, 0u);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
}

TEST_F(LinuxXdnaKernelQueueTest, WaitFailurePreservesPendingProgress) {
  uint64_t sequence = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 32768, 64, &sequence),
            AMDF_STATUS_OK);
  native_.wait_error = EIO;
  EXPECT_NE(amdf_xdna_umd_kernel_queue_wait(queue_, sequence, &deadline_),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_progress(queue_), 0u);
  native_.wait_error = 0;
  ASSERT_EQ(amdf_xdna_umd_kernel_queue_wait(queue_, sequence, &deadline_),
            AMDF_STATUS_OK);
  native_.packet[0] = (native_.packet[0] & ~15u) | 6u;
  amdf_xdna_umd_kernel_queue_retire_command(queue_);
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_query_terminal_status(queue_),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 6));
  uint64_t rejected = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_kernel_queue_submit(queue_, 65536, 64, &rejected),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 6));
  EXPECT_EQ(rejected, UINT64_MAX);
}

}  // namespace

extern "C" amdf_status_t amdf_linux_xdna_buffer_create(
    int descriptor, uint32_t type, size_t byte_length,
    amdf_linux_xdna_buffer_t* out_buffer) {
  EXPECT_EQ(descriptor, 42);
  EXPECT_EQ(type, AMDXDNA_BO_CMD);
  EXPECT_EQ(byte_length, 4096u);
  *out_buffer = {};
  out_buffer->handle = 9;
  out_buffer->byte_length = byte_length;
  return AMDF_STATUS_OK;
}

extern "C" amdf_status_t amdf_linux_xdna_buffer_attach(
    int descriptor, size_t alignment, size_t page_size,
    const amdf_linux_xdna_buffer_t* heap, amdf_linux_xdna_buffer_t* buffer) {
  EXPECT_EQ(heap, nullptr);
  buffer->host_pointer = native_state->packet.data();
  return AMDF_STATUS_OK;
}

extern "C" amdf_status_t amdf_linux_xdna_buffer_deinitialize(
    int descriptor, amdf_linux_xdna_buffer_t* buffer) {
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
  if (request == DRM_IOCTL_AMDXDNA_EXEC_CMD) {
    auto* submit = static_cast<amdxdna_drm_exec_cmd*>(argument);
    EXPECT_EQ(submit->hwctx, 7u);
    EXPECT_EQ(submit->cmd_handles, 9u);
    EXPECT_EQ(submit->cmd_count, 1u);
    EXPECT_EQ(submit->args, 0u);
    EXPECT_EQ(submit->arg_count, 0u);
    if (native.submission_error != 0) {
      errno = native.submission_error;
      return -1;
    }
    std::memcpy(native.submitted_packet.data(), native.packet.data(),
                sizeof(amdf_linux_xdna_elf_packet_t));
    submit->seq = native.submission_count++;
    return 0;
  }
  if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) {
    auto* wait = static_cast<drm_syncobj_timeline_wait*>(argument);
    std::unique_lock<std::mutex> lock(native.mutex);
    native.waited_sequence = *reinterpret_cast<uint64_t*>(wait->points);
    ++native.wait_count;
    native.wait_entered = true;
    native.changed.notify_all();
    native.changed.wait(lock, [&] { return native.complete_wait; });
    if (native.wait_error != 0) {
      errno = native.wait_error;
      return -1;
    }
    native.packet[0] = (native.packet[0] & ~15u) | 4u;
    return 0;
  }
  ADD_FAILURE() << "Unexpected ioctl " << request;
  errno = EINVAL;
  return -1;
}
