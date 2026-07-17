// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX-specific fence CTS tests.
//
// Tests for device fence import/export using POSIX file descriptors.
// Uses eventfd (Linux) or pipe (other POSIX) to simulate device fences
// without requiring actual GPU hardware.
//
// Device fence bridging enables GPU↔proactor synchronization:
//   - import_fence: GPU completion → semaphore signal
//   - export_fence: semaphore signal → GPU wait
//
// This allows zero-copy pipelines where network I/O and GPU work are
// synchronized without host-side round-trips.

#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <future>
#include <thread>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/primitive.h"
#include "iree/async/semaphore.h"

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#include <sys/eventfd.h>
#define IREE_CTS_HAVE_EVENTFD 1
#endif

namespace iree::async::cts {

class FencePosixTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase<>::SetUp();
    if (!iree_any_bit_set(capabilities_,
                          IREE_ASYNC_PROACTOR_CAPABILITY_DEVICE_FENCE)) {
      GTEST_SKIP() << "backend lacks device fence capability";
    }
  }

  // Creates a fence fd for testing.
  // On Linux, uses eventfd. On other POSIX, uses the read end of a pipe.
  // Returns the fence fd and optionally the write end for signaling.
  // Caller owns both fds and must close them.
  void CreateTestFence(int* out_fence_fd, int* out_signal_fd) {
#if defined(IREE_CTS_HAVE_EVENTFD)
    // eventfd is ideal: single fd that can be both signaled and polled.
    int efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    ASSERT_GE(efd, 0) << "eventfd failed: " << strerror(errno);
    *out_fence_fd = efd;
    *out_signal_fd = efd;  // Same fd for eventfd.
#else
    // Fallback: pipe pair. Write end signals, read end is the fence.
    int pipefd[2];
    ASSERT_EQ(pipe(pipefd), 0) << "pipe failed: " << strerror(errno);
    *out_fence_fd = pipefd[0];   // Read end is the fence.
    *out_signal_fd = pipefd[1];  // Write end for signaling.
#endif
  }

  // Creates a test fence with a separate signaler fd.
  // The fence_fd is suitable for import_fence (proactor takes ownership and
  // closes it). The signaler_fd is a distinct fd that the caller can use to
  // signal the fence via SignalTestFence() and must close after use.
  //
  // On Linux (eventfd): dup()s the eventfd so proactor and signaler each own
  // a separate fd to the same underlying counter.
  // On other POSIX (pipe): fence_fd is the read end, signaler_fd is the
  // write end — already distinct, no dup needed.
  void CreateTestFenceWithSignaler(int* out_fence_fd, int* out_signaler_fd) {
    int fence_fd = -1;
    int signal_fd = -1;
    CreateTestFence(&fence_fd, &signal_fd);
#if defined(IREE_CTS_HAVE_EVENTFD)
    *out_fence_fd = fence_fd;
    *out_signaler_fd = dup(signal_fd);
    ASSERT_GE(*out_signaler_fd, 0) << "dup failed: " << strerror(errno);
#else
    *out_fence_fd = fence_fd;
    *out_signaler_fd = signal_fd;
#endif
  }

  // Signals the test fence (makes it readable/pollable).
  void SignalTestFence(int signal_fd) {
#if defined(IREE_CTS_HAVE_EVENTFD)
    uint64_t value = 1;
    ssize_t written = write(signal_fd, &value, sizeof(value));
    ASSERT_EQ(written, sizeof(value)) << "eventfd write failed";
#else
    char byte = 1;
    ssize_t written = write(signal_fd, &byte, 1);
    ASSERT_EQ(written, 1) << "pipe write failed";
#endif
  }

  // Checks if an fd is readable (fence signaled).
  bool IsFdReadable(int fd) {
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int result = poll(&pfd, 1, 0);
    return result > 0 && (pfd.revents & POLLIN);
  }
};

//===----------------------------------------------------------------------===//
// Import fence tests
//===----------------------------------------------------------------------===//

// Import a fence fd and verify semaphore advances when fence signals.
TEST_P(FencePosixTest, ImportFence_SignalAdvancesSemaphore) {
  // Create test fence with a separate signaler fd. Proactor takes ownership
  // of fence_fd; signaler_fd is distinct so the signaler thread can write
  // after proactor closes the original.
  int fence_fd = -1;
  int signaler_fd = -1;
  CreateTestFenceWithSignaler(&fence_fd, &signaler_fd);

  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Import fence: when it signals, semaphore should advance to 1.
  iree_async_primitive_t fence_primitive =
      iree_async_primitive_from_fd(fence_fd);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, fence_primitive,
                                                   semaphore, 1));

  // Semaphore should still be at 0.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 0u);

  // Signal the fence from a background thread (simulates GPU completion).
  // The import is synchronous, so we can signal immediately.
  std::thread signaler([this, signaler_fd]() { SignalTestFence(signaler_fd); });

  PollUntilCondition([&] { return iree_async_semaphore_query(semaphore) >= 1; },
                     "imported fence signal");

  signaler.join();

  // Semaphore should now be at 1.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  // Close the signaler's fd (either the dup for eventfd, or the pipe write end
  // that was distinct from fence_fd which was already closed by the proactor).
  close(signaler_fd);

  iree_async_semaphore_release(semaphore);
}

// Import a fence that's already signaled: semaphore should advance immediately.
TEST_P(FencePosixTest, ImportFence_AlreadySignaled) {
  // Create test fence and signal it before import.
  int fence_fd = -1;
  int signal_fd = -1;
  CreateTestFence(&fence_fd, &signal_fd);
  SignalTestFence(signal_fd);

  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Import fence.
  iree_async_primitive_t fence_primitive =
      iree_async_primitive_from_fd(fence_fd);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, fence_primitive,
                                                   semaphore, 1));

  PollUntilCondition([&] { return iree_async_semaphore_query(semaphore) >= 1; },
                     "already-signaled imported fence");

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

#if !defined(IREE_CTS_HAVE_EVENTFD)
  close(signal_fd);
#endif

  iree_async_semaphore_release(semaphore);
}

// Import with invalid fd should fail.
TEST_P(FencePosixTest, ImportFence_InvalidFd) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_primitive_t bad_fence = iree_async_primitive_from_fd(-1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_semaphore_import_fence(proactor_, bad_fence, semaphore, 1));

  iree_async_semaphore_release(semaphore);
}

// Import with NONE primitive type should fail.
TEST_P(FencePosixTest, ImportFence_NonePrimitive) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  iree_async_primitive_t none_fence = iree_async_primitive_none();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_semaphore_import_fence(proactor_, none_fence, semaphore, 1));

  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Export fence tests
//===----------------------------------------------------------------------===//

// Export a fence and verify it becomes readable when semaphore advances.
TEST_P(FencePosixTest, ExportFence_SemaphoreAdvanceSignalsFence) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export fence that signals when semaphore reaches 1.
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Fence should not be readable yet.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd));

  // Advance semaphore to 1.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, /*frontier=*/NULL));

  // Semaphore timepoints dispatch synchronously from signal().
  EXPECT_TRUE(IsFdReadable(exported_fence.value.fd));

  // Caller owns the exported fence fd.
  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

// Export fence when semaphore is already at target value: immediate signal.
TEST_P(FencePosixTest, ExportFence_SemaphoreAlreadyReached) {
  // Create semaphore already at 5.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 5, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export fence for value 3 (already reached).
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 3,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Already-reached exports complete synchronously.
  EXPECT_TRUE(IsFdReadable(exported_fence.value.fd));

  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Round-trip tests
//===----------------------------------------------------------------------===//

// Import fence → semaphore → export fence: end-to-end pipeline.
TEST_P(FencePosixTest, ImportExportRoundTrip) {
  // Create input fence (simulates GPU completion).
  int input_fence_fd = -1;
  int input_signal_fd = -1;
  CreateTestFence(&input_fence_fd, &input_signal_fd);

  // Create semaphore as the bridge.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Import: input fence → semaphore value 1.
  iree_async_primitive_t input_primitive =
      iree_async_primitive_from_fd(input_fence_fd);
  IREE_ASSERT_OK(iree_async_semaphore_import_fence(proactor_, input_primitive,
                                                   semaphore, 1));

  // Export: semaphore value 1 → output fence.
  iree_async_primitive_t output_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &output_fence));

  ASSERT_EQ(output_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(output_fence.value.fd, 0);

  // Output fence should not be readable yet.
  EXPECT_FALSE(IsFdReadable(output_fence.value.fd));

  // Signal input fence (GPU completion).
  SignalTestFence(input_signal_fd);

  PollUntilCondition([&] { return IsFdReadable(output_fence.value.fd); },
                     "import-export fence round trip");

  // Output fence should now be readable.
  EXPECT_TRUE(IsFdReadable(output_fence.value.fd));

  // Verify semaphore reached target.
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

#if !defined(IREE_CTS_HAVE_EVENTFD)
  close(input_signal_fd);
#endif
  close(output_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Export fence error/edge-case tests
//===----------------------------------------------------------------------===//

// Export fence, then fail the semaphore: fd should stay unreadable.
// Exercises the failure branch in the export timepoint callback.
TEST_P(FencePosixTest, ExportFence_SemaphoreFailsAfterExport) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export fence for value 1 (not yet reached).
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Fence should not be readable yet.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd));

  // Fail the semaphore. The export callback fires synchronously (under the
  // semaphore's lock) and leaves the fd unreadable.
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "test failure"));

  // Fence should remain unreadable after semaphore failure.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd));

  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

// Export fence on an already-failed semaphore: callback fires synchronously
// with the failure status, fd stays unreadable from the start.
TEST_P(FencePosixTest, ExportFence_SemaphoreAlreadyFailed) {
  // Create semaphore and fail it immediately.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));
  iree_async_semaphore_fail(
      semaphore, iree_make_status(IREE_STATUS_ABORTED, "pre-failed"));

  // Export should succeed (it creates the fd and registers the timepoint).
  iree_async_primitive_t exported_fence;
  IREE_ASSERT_OK(iree_async_semaphore_export_fence(proactor_, semaphore, 1,
                                                   &exported_fence));

  ASSERT_EQ(exported_fence.type, IREE_ASYNC_PRIMITIVE_TYPE_FD);
  ASSERT_GE(exported_fence.value.fd, 0);

  // Fence should be unreadable because the semaphore was already failed.
  EXPECT_FALSE(IsFdReadable(exported_fence.value.fd));

  close(exported_fence.value.fd);
  iree_async_semaphore_release(semaphore);
}

// Multiple exports on the same semaphore with different wait values.
// Verifies that each export fires independently as its target is reached.
TEST_P(FencePosixTest, ExportFence_MultipleExportsDifferentValues) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  // Export three fences at values 1, 2, and 3.
  iree_async_primitive_t fence_at_1, fence_at_2, fence_at_3;
  IREE_ASSERT_OK(
      iree_async_semaphore_export_fence(proactor_, semaphore, 1, &fence_at_1));
  IREE_ASSERT_OK(
      iree_async_semaphore_export_fence(proactor_, semaphore, 2, &fence_at_2));
  IREE_ASSERT_OK(
      iree_async_semaphore_export_fence(proactor_, semaphore, 3, &fence_at_3));

  ASSERT_GE(fence_at_1.value.fd, 0);
  ASSERT_GE(fence_at_2.value.fd, 0);
  ASSERT_GE(fence_at_3.value.fd, 0);

  // None should be readable yet.
  EXPECT_FALSE(IsFdReadable(fence_at_1.value.fd));
  EXPECT_FALSE(IsFdReadable(fence_at_2.value.fd));
  EXPECT_FALSE(IsFdReadable(fence_at_3.value.fd));

  // Signal semaphore to 1. Only fence_at_1 should become readable.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 1, /*frontier=*/NULL));
  EXPECT_TRUE(IsFdReadable(fence_at_1.value.fd));
  EXPECT_FALSE(IsFdReadable(fence_at_2.value.fd));
  EXPECT_FALSE(IsFdReadable(fence_at_3.value.fd));

  // Signal semaphore to 3. Both fence_at_2 and fence_at_3 should fire.
  IREE_ASSERT_OK(iree_async_semaphore_signal(semaphore, 3, /*frontier=*/NULL));
  EXPECT_TRUE(IsFdReadable(fence_at_2.value.fd));
  EXPECT_TRUE(IsFdReadable(fence_at_3.value.fd));

  close(fence_at_1.value.fd);
  close(fence_at_2.value.fd);
  close(fence_at_3.value.fd);
  iree_async_semaphore_release(semaphore);
}

//===----------------------------------------------------------------------===//
// Cross-thread import fence tests
//===----------------------------------------------------------------------===//
//
// These tests exercise the thread-safety of import_fence by calling it from a
// background thread while poll() runs on the main thread. This is the real
// use case: a GPU driver completion callback imports a fence from a
// driver-internal thread.

// Import fence from a background thread while the main thread polls.
// Under TSAN, this catches data races in fd_map/event_set access (POSIX) or
// concurrent io_uring_enter calls (io_uring).
TEST_P(FencePosixTest, ImportFence_CrossThreadImportRacesWithPoll) {
  // Create semaphore starting at 0.
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
      iree_allocator_system(), &semaphore));

  int fence_fd = -1;
  int signaler_fd = -1;
  CreateTestFenceWithSignaler(&fence_fd, &signaler_fd);

  std::promise<void> importer_ready_promise;
  auto importer_ready = importer_ready_promise.get_future();
  std::promise<void> start_promise;
  auto start = start_promise.get_future();
  iree_status_code_t import_status_code = IREE_STATUS_UNKNOWN;
  std::atomic<bool> import_completed{false};
  std::thread importer([this, semaphore, fence_fd, signaler_fd,
                        &importer_ready_promise, &start, &import_status_code,
                        &import_completed]() {
    importer_ready_promise.set_value();
    start.wait();

    // Import fence from THIS thread (not the poll thread).
    iree_async_primitive_t fence_primitive =
        iree_async_primitive_from_fd(fence_fd);
    iree_status_t status = iree_async_semaphore_import_fence(
        proactor_, fence_primitive, semaphore, 1);
    import_status_code = iree_status_code(status);
    iree_status_free(status);
    if (import_status_code == IREE_STATUS_OK) {
      SignalTestFence(signaler_fd);
    }
    import_completed.store(true, std::memory_order_release);
    iree_async_proactor_wake(proactor_);
  });

  importer_ready.wait();
  start_promise.set_value();
  PollUntilCondition(
      [&] {
        if (iree_async_semaphore_query(semaphore) >= 1) return true;
        return import_completed.load(std::memory_order_acquire) &&
               import_status_code != IREE_STATUS_OK;
      },
      "cross-thread fence import");

  importer.join();
  EXPECT_EQ(import_status_code, IREE_STATUS_OK);

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 1u);

  close(signaler_fd);
  iree_async_semaphore_release(semaphore);
}

// Multiple threads importing fences concurrently while poll runs.
// Exercises concurrent import_fence calls racing with each other AND with poll.
TEST_P(FencePosixTest, ImportFence_ConcurrentImportsRaceWithPoll) {
  static constexpr int kThreadCount = 4;

  iree_async_semaphore_t* semaphores[kThreadCount];
  int fence_fds[kThreadCount];
  int signaler_fds[kThreadCount];
  for (int i = 0; i < kThreadCount; ++i) {
    IREE_ASSERT_OK(iree_async_semaphore_create(
        proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
        iree_allocator_system(), &semaphores[i]));
    CreateTestFenceWithSignaler(&fence_fds[i], &signaler_fds[i]);
  }

  std::promise<void> ready_promises[kThreadCount];
  std::future<void> ready_futures[kThreadCount];
  std::promise<void> start_promise;
  auto start = start_promise.get_future().share();
  iree_status_code_t import_status_codes[kThreadCount];
  std::atomic<bool> import_completed[kThreadCount];
  std::thread threads[kThreadCount];
  for (int i = 0; i < kThreadCount; ++i) {
    ready_futures[i] = ready_promises[i].get_future();
    import_status_codes[i] = IREE_STATUS_UNKNOWN;
    import_completed[i].store(false, std::memory_order_relaxed);
    threads[i] = std::thread([this, &semaphores, &fence_fds, &signaler_fds,
                              &ready_promises, start, &import_status_codes,
                              &import_completed, i]() {
      ready_promises[i].set_value();
      start.wait();
      iree_async_primitive_t fence_primitive =
          iree_async_primitive_from_fd(fence_fds[i]);
      iree_status_t status = iree_async_semaphore_import_fence(
          proactor_, fence_primitive, semaphores[i], 1);
      import_status_codes[i] = iree_status_code(status);
      iree_status_free(status);
      if (import_status_codes[i] == IREE_STATUS_OK) {
        SignalTestFence(signaler_fds[i]);
      }
      import_completed[i].store(true, std::memory_order_release);
      iree_async_proactor_wake(proactor_);
    });
  }

  for (auto& ready : ready_futures) ready.wait();
  start_promise.set_value();
  PollUntilCondition(
      [&] {
        bool all_completed = true;
        bool any_failed = false;
        bool all_signaled = true;
        for (int i = 0; i < kThreadCount; ++i) {
          bool completed = import_completed[i].load(std::memory_order_acquire);
          all_completed &= completed;
          if (completed) {
            any_failed |= import_status_codes[i] != IREE_STATUS_OK;
          }
          all_signaled &= iree_async_semaphore_query(semaphores[i]) >= 1;
        }
        return all_signaled || (all_completed && any_failed);
      },
      "concurrent fence imports");

  for (int i = 0; i < kThreadCount; ++i) {
    threads[i].join();
    EXPECT_EQ(import_status_codes[i], IREE_STATUS_OK);
    EXPECT_EQ(iree_async_semaphore_query(semaphores[i]), 1u);
    close(signaler_fds[i]);
    iree_async_semaphore_release(semaphores[i]);
  }
}

CTS_REGISTER_TEST_SUITE(FencePosixTest);

}  // namespace iree::async::cts
