// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_STALLED_STDERR_H_
#define LOOM_SERVE_TESTING_STALLED_STDERR_H_

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <utility>

#include "iree/testing/gtest.h"

namespace loom_serve::testing {

// A genuinely full inherited stderr pipe reproduces the serving boundary: any
// synchronous diagnostic write on the only worker prevents inference progress.
// Restoration drains accepted writes before closing the read end, including on
// a failed assertion. The outer test harness owns the deadlock guard.
class StalledStderr {
 public:
  void Begin() {
    ASSERT_EQ(pipe2(pipe_, O_CLOEXEC | O_NONBLOCK), 0);
    saved_ = dup(STDERR_FILENO);
    ASSERT_GE(saved_, 0);
    const char bytes[4096] = {};
    ssize_t result;
    do {
      result = write(pipe_[1], bytes, sizeof(bytes));
    } while (result > 0);
    ASSERT_EQ(result, -1);
    ASSERT_EQ(errno, EAGAIN);
    ASSERT_EQ(fcntl(pipe_[1], F_SETFL, 0), 0);
    ASSERT_EQ(fcntl(pipe_[0], F_SETFL, 0), 0);
    ASSERT_EQ(dup2(pipe_[1], STDERR_FILENO), STDERR_FILENO);
    ASSERT_EQ(close(std::exchange(pipe_[1], -1)), 0);
  }
  ~StalledStderr() {
    if (saved_ >= 0) {
      EXPECT_EQ(dup2(saved_, STDERR_FILENO), STDERR_FILENO);
      EXPECT_EQ(close(saved_), 0);
    }
    if (pipe_[1] >= 0) {
      EXPECT_EQ(close(pipe_[1]), 0);
    }
    if (pipe_[0] >= 0) {
      char bytes[4096];
      ssize_t result;
      do {
        result = read(pipe_[0], bytes, sizeof(bytes));
      } while (result > 0 || (result < 0 && errno == EINTR));
      EXPECT_EQ(result, 0);
      EXPECT_EQ(close(pipe_[0]), 0);
    }
  }

 private:
  // Owned saved process stderr, restored before draining the full pipe.
  int saved_ = -1;
  // Owned read and write handles; the write handle moves into stderr.
  int pipe_[2] = {-1, -1};
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_STALLED_STDERR_H_
