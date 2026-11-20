// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <string>

#include "iree/async/api.h"
#include "iree/async/cts/util/test_base.h"
#include "iree/async/platform/io_uring/api.h"
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
thread_local bool is_poll_owner = false;
std::atomic<bool> closed_on_poll_owner{false};

int CloseWithDelayedError(int fd) {
  if (attempts.fetch_add(1) == 0) {
    closed_on_poll_owner.store(is_poll_owner);
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

using ProactorFactory = decltype(&iree_async_proactor_create_posix);

class CloseTest : public ::testing::TestWithParam<ProactorFactory> {
 protected:
  void SetUp() override {
    iree_status_t status = GetParam()(iree_async_proactor_options_default(),
                                      iree_allocator_system(), &proactor_);
    if (iree_status_is_unavailable(status)) {
      std::string message = iree::Status::ToString(status);
      iree_status_free(status);
      GTEST_SKIP() << message;
    }
    IREE_ASSERT_OK(status);
    is_poll_owner = true;
  }

  void TearDown() override {
    watched_fd.store(-1);
    is_poll_owner = false;
    iree_async_proactor_release(proactor_);
  }

  void ExpectCloseError(iree_async_operation_t* operation,
                        iree_async_operation_type_t type, int fd) {
    attempts.store(0);
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
    EXPECT_FALSE(closed_on_poll_owner.load());
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

// A checked filesystem close can return any flush error after releasing the fd.
// POSIX also issues its socket close through libc. Linux ring sockets have a
// native kernel close and do not participate in the libc error injection.
TEST_P(CloseTest, IssuedCloseReportsErrorsAndRetiresOnce) {
#if defined(POSIX_CLOSE_RESTART)
  constexpr int kInterruptedClose = EINPROGRESS;
#else
  constexpr int kInterruptedClose = EINTR;
#endif  // POSIX_CLOSE_RESTART
  for (int error : {EIO, ENOSPC, ECANCELED, kInterruptedClose}) {
    SCOPED_TRACE(strerror(error));
    close_error = error;
    int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(fd, 0);
    iree_async_file_close_operation_t file_close = {};
    IREE_ASSERT_OK(
        iree_async_file_import(proactor_, iree_async_primitive_from_fd(fd),
                               IREE_ASYNC_FILE_FLAG_NONE, &file_close.file));
    ASSERT_NO_FATAL_FAILURE(ExpectCloseError(
        &file_close.base, IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE, fd));
    if (GetParam() == iree_async_proactor_create_posix) {
      iree_async_socket_close_operation_t socket_close = {};
      IREE_ASSERT_OK(iree_async_socket_create(
          proactor_, IREE_ASYNC_SOCKET_TYPE_TCP, IREE_ASYNC_SOCKET_OPTION_NONE,
          &socket_close.socket));
      ASSERT_NO_FATAL_FAILURE(ExpectCloseError(
          &socket_close.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
          socket_close.socket->primitive.value.fd));
    }
  }
}

// Failed open skips both successors. Retirement of the only file reference
// must complete before either skipped callback can reuse its operation.
TEST_P(CloseTest, SkippedLastCloseRetiresOnWorkerBeforeSuccessor) {
  iree_async_file_open_operation_t opening = {};
  CompletionTracker opened, closed;
  iree_async_operation_initialize(
      &opening.base, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN,
      IREE_ASYNC_OPERATION_FLAG_LINKED, CompletionTracker::Callback, &opened);
  opening.path = "/dev/null/not-a-directory";
  opening.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
  int fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);
  iree_async_file_close_operation_t closing = {};
  IREE_ASSERT_OK(
      iree_async_file_import(proactor_, iree_async_primitive_from_fd(fd),
                             IREE_ASYNC_FILE_FLAG_NONE, &closing.file));
  iree_async_operation_initialize(
      &closing.base, IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE,
      IREE_ASYNC_OPERATION_FLAG_LINKED, CompletionTracker::Callback, &closed);
  iree_async_operation_t tail = {};
  iree_async_operation_initialize(
      &tail, IREE_ASYNC_OPERATION_TYPE_NOP, 0,
      +[](void* user_data, iree_async_operation_t*, iree_status_t status,
          iree_async_completion_flags_t) {
        auto* closed = static_cast<CompletionTracker*>(user_data);
        EXPECT_EQ(closed->call_count, 1);
        EXPECT_EQ(attempts.load(), 1);
        IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
      },
      &closed);
  iree_async_operation_t* operations[] = {&opening.base, &closing.base, &tail};
  close_error = EIO;
  attempts.store(0);
  watched_fd.store(fd);
  IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, {operations, 3}));
  iree_host_size_t callbacks = 0;
  while (callbacks < 3) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), &count));
    callbacks += count;
  }
  watched_fd.store(-1);
  EXPECT_FALSE(closed_on_poll_owner.load());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        opened.ConsumeStatus());
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, closed.ConsumeStatus());
}

// Keeping a second owner across a skipped close leaves it usable. With only
// the transferred owner the peer sees EOF by the time the callback runs.
TEST_P(CloseTest, SkippedSocketRetiresOnlyTransferredReference) {
  for (bool retain : {false, true}) {
    SCOPED_TRACE(retain);
    int pair[2];
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, pair), 0);
    iree_async_socket_close_operation_t closing = {};
    IREE_ASSERT_OK(iree_async_socket_import(
        proactor_, iree_async_primitive_from_fd(pair[0]),
        IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM, IREE_ASYNC_SOCKET_OPTION_NONE,
        &closing.socket));
    if (retain) {
      iree_async_socket_retain(closing.socket);
    }
    CompletionTracker opened, closed;
    iree_async_file_open_operation_t opening = {};
    iree_async_operation_initialize(
        &opening.base, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN,
        IREE_ASYNC_OPERATION_FLAG_LINKED, CompletionTracker::Callback, &opened);
    opening.path = "/dev/null/not-a-directory";
    opening.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;
    iree_async_operation_initialize(
        &closing.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
        IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS,
        CompletionTracker::Callback, &closed);
    iree_async_operation_t* chain[] = {&opening.base, &closing.base};
    IREE_ASSERT_OK(iree_async_proactor_submit(proactor_, {chain, 2}));
    while (!closed.call_count) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), nullptr));
    }
    EXPECT_EQ(opened.call_count, 1);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                          opened.ConsumeStatus());
    IREE_EXPECT_OK(closed.ConsumeStatus());
    EXPECT_EQ(closed.last_flags, IREE_ASYNC_COMPLETION_FLAG_CANCELLED);
    char byte = 0;
    if (retain) {
      ASSERT_EQ(send(closing.socket->primitive.value.fd, "a", 1, MSG_NOSIGNAL),
                1);
      EXPECT_EQ(recv(pair[1], &byte, 1, MSG_DONTWAIT), 1);
      EXPECT_EQ(byte, 'a');
      iree_async_socket_release(closing.socket);
    }
    EXPECT_EQ(recv(pair[1], &byte, 1, MSG_DONTWAIT), 0);
    close(pair[1]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Backends, CloseTest,
    ::testing::Values(iree_async_proactor_create_posix,
                      iree_async_proactor_create_io_uring),
    [](const ::testing::TestParamInfo<ProactorFactory>& info) {
      return info.param == iree_async_proactor_create_posix ? "Posix"
                                                            : "IoUring";
    });

}  // namespace
