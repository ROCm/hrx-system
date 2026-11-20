// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>

#include "iree/async/api.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/platform/posix/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

extern "C" int __real_close(int fd);

namespace {

// The file worker and the polling thread can both reach close(). Only the
// chosen test descriptor is intercepted, through completion and final release.
std::atomic<int> watched_fd{-1};
std::atomic<int> attempts{0};
int close_error = 0;

int CloseWithDelayedError(int fd) {
  if (attempts.fetch_add(1) == 0) {
    // Linux detaches the descriptor before returning a filesystem flush error.
    if (__real_close(fd) < 0) {
      return -1;
    }
    errno = close_error;
  } else {
    // Count a mistaken retry without touching the released descriptor number.
    errno = EBADF;
  }
  return -1;
}

}  // namespace

extern "C" int __wrap_close(int fd) {
  return fd == watched_fd.load() ? CloseWithDelayedError(fd) : __real_close(fd);
}

#if defined(POSIX_CLOSE_RESTART)
extern "C" int __real_posix_close(int fd, int flags);
extern "C" int __wrap_posix_close(int fd, int flags) {
  return fd == watched_fd.load() && flags == 0 ? CloseWithDelayedError(fd)
                                               : __real_posix_close(fd, flags);
}
#endif  // POSIX_CLOSE_RESTART

namespace {

using iree::async::cts::CompletionTracker;

class CloseTest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_async_proactor_create_posix(iree_async_proactor_options_default(),
                                         iree_allocator_system(), &proactor_));
    attempts.store(0);
    close_error = GetParam();
  }

  void TearDown() override {
    watched_fd.store(-1);
    iree_async_proactor_release(proactor_);
  }

  void ExpectCloseError(iree_async_operation_t* operation,
                        iree_async_operation_type_t type, int fd) {
    watched_fd.store(fd);
    CompletionTracker tracker;
    iree_async_operation_initialize(
        operation, type, IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        CompletionTracker::Callback, &tracker);
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
    while (!tracker.call_count) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
    // Formatting the saved status can open symbol files on the released
    // descriptor number. Native retirement is complete at the callback.
    watched_fd.store(-1);
    EXPECT_EQ(tracker.call_count, 1);
    EXPECT_EQ(tracker.last_flags, IREE_ASYNC_COMPLETION_FLAG_NONE);
    EXPECT_EQ(attempts.load(), 1);
#if IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_ANNOTATIONS
    EXPECT_THAT(iree::Status::ToString(tracker.last_status),
                ::testing::HasSubstr(strerror(close_error)));
#endif  // IREE_STATUS_FEATURE_ANNOTATIONS
    IREE_EXPECT_STATUS_IS(iree_status_code_from_errno(close_error),
                          tracker.ConsumeStatus());
  }

  // Owns the submitting, worker, and polling paths under test.
  iree_async_proactor_t* proactor_ = nullptr;
};

TEST_P(CloseTest, FileReportsErrorAndRetiresOnce) {
  int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);
  iree_async_file_close_operation_t operation = {};
  IREE_ASSERT_OK(
      iree_async_file_import(proactor_, iree_async_primitive_from_fd(fd),
                             IREE_ASYNC_FILE_FLAG_NONE, &operation.file));
  ExpectCloseError(&operation.base, IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE, fd);
}

TEST_P(CloseTest, SocketReportsErrorAndRetiresOnce) {
  iree_async_socket_close_operation_t operation = {};
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &operation.socket));
  ExpectCloseError(&operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                   operation.socket->primitive.value.fd);
}

// posix_close(fd, 0) can report EINPROGRESS for an interrupted close.
#if defined(POSIX_CLOSE_RESTART)
constexpr int kInterruptedClose = EINPROGRESS;
#else
constexpr int kInterruptedClose = EINTR;
#endif  // POSIX_CLOSE_RESTART
INSTANTIATE_TEST_SUITE_P(NativeErrors, CloseTest,
                         ::testing::Values(EIO, ENOSPC, ECANCELED,
                                           kInterruptedClose));

}  // namespace
