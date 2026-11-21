// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/proactor.h"
#include "iree/async/posix_wait.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_APPLE)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

TEST(PosixWaitTest, NativeTimespecPreservesFiniteRangeAndFraction) {
  struct timespec storage = {};
  EXPECT_EQ(iree_async_posix_timeout_timespec(IREE_DURATION_INFINITE, &storage),
            nullptr);
  EXPECT_EQ(iree_async_posix_timeout_timespec(0, &storage), &storage);
  EXPECT_EQ(storage.tv_sec, 0);
  EXPECT_EQ(storage.tv_nsec, 0);
  for (iree_duration_t duration_ns :
       {999999999LL, 1000000001LL, INT64_MAX - 1LL}) {
    ASSERT_EQ(iree_async_posix_timeout_timespec(duration_ns, &storage),
              &storage);
    int64_t expected_seconds = duration_ns / 1000000000;
    if ((sizeof(time_t) == 4 || sizeof(long) == 4) &&
        expected_seconds > INT32_MAX) {
      EXPECT_EQ(storage.tv_sec, INT32_MAX);
      EXPECT_EQ(storage.tv_nsec, 999999999);
    } else {
      EXPECT_EQ(storage.tv_sec, expected_seconds);
      EXPECT_EQ(storage.tv_nsec, duration_ns % 1000000000);
    }
  }
}

#if defined(IREE_PLATFORM_APPLE)
TEST(PosixWaitTest, DarwinPollLeavesItsTimerEmptyAfterExpiry) {
  int timer_fd = iree_async_posix_kqueue();
  ASSERT_GE(timer_fd, 0);
  int flags = fcntl(timer_fd, F_GETFD);
  EXPECT_GE(flags, 0);
  EXPECT_NE(flags & FD_CLOEXEC, 0);
  struct pollfd timer = {.fd = timer_fd, .events = POLLIN};
  struct pollfd spare = {};
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(iree_async_darwin_poll_until(&spare, 0, iree_time_now() + 100000,
                                           timer_fd),
              0);
    // An unconsumed timer would make the next poll wake immediately.
    EXPECT_EQ(poll(&timer, 1, 0), 0);
  }
  EXPECT_EQ(close(timer_fd), 0);
}
#endif  // IREE_PLATFORM_APPLE

// Records the real poll owner's choice of deadline and injects the native
// dependency's EINTR outcome. Every other event-set operation is unchanged.
struct InterruptedWait : iree_async_posix_event_set_vtable_t {
  // Most recent absolute deadline supplied by the poll owner.
  mutable iree_time_t observed_deadline_ns;
};

TEST(PosixProactorPollTest, InterruptedWaitPreservesCallerAndTimerDeadlines) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_posix(iree_async_proactor_options_default(),
                                       iree_allocator_system(), &proactor));
  auto* event_set = iree_async_proactor_posix_cast(proactor)->event_set;
  const auto* original_vtable = event_set->vtable;
  InterruptedWait interrupted{*original_vtable, 0};
  interrupted.wait =
      +[](iree_async_posix_event_set_t* event_set, iree_time_t deadline_ns,
          iree_host_size_t* ready_count, bool* timed_out) -> iree_status_t {
    auto* observer = static_cast<const InterruptedWait*>(event_set->vtable);
    observer->observed_deadline_ns = deadline_ns;
    *ready_count = 0;
    *timed_out = false;
    return iree_ok_status();
  };
  event_set->vtable = &interrupted;
  auto poll = [&](iree_time_t caller, iree_time_t expected, size_t count) {
    interrupted.observed_deadline_ns = 0;
    iree_host_size_t completed = 99;
    IREE_EXPECT_OK(iree_async_proactor_poll(
        proactor, iree_make_deadline(caller), &completed));
    EXPECT_EQ(interrupted.observed_deadline_ns, expected);
    EXPECT_EQ(completed, count);
  };

  // Keep the timer in the future under any scheduler load. The values are one
  // nanosecond apart so an early conversion to even microseconds loses the
  // distinction between the two owners.
  constexpr iree_time_t kTimer = IREE_TIME_INFINITE_FUTURE - 123;
  poll(IREE_TIME_INFINITE_FUTURE, IREE_TIME_INFINITE_FUTURE, 0);
  poll(kTimer - 1, kTimer - 1, 0);

  iree_async_timer_operation_t timer = {};
  timer.base.type = IREE_ASYNC_OPERATION_TYPE_TIMER;
  timer.base.completion_fn =
      +[](void*, iree_async_operation_t*, iree_status_t status,
          iree_async_completion_flags_t) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
      };
  timer.deadline_ns = kTimer;
  IREE_EXPECT_OK(iree_async_proactor_submit_one(proactor, &timer.base));
  poll(kTimer - 1, kTimer - 1, 0);
  poll(kTimer + 1, kTimer, 0);
  poll(IREE_TIME_INFINITE_FUTURE, kTimer, 0);

  IREE_EXPECT_OK(iree_async_proactor_cancel(proactor, &timer.base));
  poll(IREE_TIME_INFINITE_FUTURE, IREE_TIME_INFINITE_PAST, 1);
  event_set->vtable = original_vtable;
  iree_async_proactor_release(proactor);
}

}  // namespace
