// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/tracing.h"
#include "iree/testing/gtest.h"

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_WASM)
#include <pthread.h>
#include <signal.h>

#include "client/TracyThread.hpp"

namespace {

TEST(TracyTracingTest, WorkerPreservesApplicationSignalOwnership) {
  sigset_t creator_mask;
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, nullptr, &creator_mask), 0);
  struct {
    // Signal mask observed by the worker before doing any other work.
    sigset_t mask;
    // Result from querying the worker's signal mask.
    int query_result;
  } worker_state = {};
  {
    tracy::Thread worker(
        [](void* user_data) {
          auto* state = static_cast<decltype(worker_state)*>(user_data);
          state->query_result =
              pthread_sigmask(SIG_SETMASK, nullptr, &state->mask);
        },
        &worker_state);
  }
  ASSERT_EQ(worker_state.query_result, 0);
  sigset_t restored_mask;
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, nullptr, &restored_mask), 0);
  for (int signal = 1; signal < NSIG; ++signal) {
    EXPECT_EQ(sigismember(&restored_mask, signal),
              sigismember(&creator_mask, signal))
        << signal;
  }
#if defined(IREE_PLATFORM_LINUX) && !defined(TRACY_NO_CRASH_HANDLER)
#if defined(TRACY_CRASH_SIGNAL)
  const int crash_signal = TRACY_CRASH_SIGNAL;
#else
  const int crash_signal = SIGPWR;
#endif
#else
  const int crash_signal = 0;
#endif
  for (int signal : {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2}) {
    if (signal != crash_signal) {
      EXPECT_EQ(sigismember(&worker_state.mask, signal), 1) << signal;
    }
  }
  for (int signal :
       {SIGILL, SIGFPE, SIGSEGV, SIGBUS, SIGABRT, SIGTRAP, SIGSYS}) {
    EXPECT_EQ(sigismember(&worker_state.mask, signal),
              sigismember(&creator_mask, signal))
        << signal;
  }
  if (crash_signal) {
    EXPECT_EQ(sigismember(&worker_state.mask, crash_signal),
              sigismember(&creator_mask, crash_signal));
  }
}

}  // namespace
#endif  // !IREE_PLATFORM_WINDOWS && !IREE_PLATFORM_WASM
