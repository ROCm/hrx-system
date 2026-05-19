// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/pollable_fd.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class OwnedFileDescriptor {
 public:
  OwnedFileDescriptor() = default;
  explicit OwnedFileDescriptor(int value) : value_(value) {}
  OwnedFileDescriptor(const OwnedFileDescriptor&) = delete;
  OwnedFileDescriptor& operator=(const OwnedFileDescriptor&) = delete;
  ~OwnedFileDescriptor() { reset(); }

  int get() const { return value_; }

  void reset(int value = -1) {
    if (value_ >= 0) close(value_);
    value_ = value;
  }

 private:
  int value_ = -1;
};

TEST(PollableFdTest, SetsNonblockingAndCloseOnExec) {
  int pipe_descriptors[2] = {-1, -1};
  ASSERT_EQ(0, pipe(pipe_descriptors));
  OwnedFileDescriptor read_descriptor(pipe_descriptors[0]);
  OwnedFileDescriptor write_descriptor(pipe_descriptors[1]);

  IREE_ASSERT_OK(iree_net_rdma_pollable_fd_initialize(read_descriptor.get()));

  int flags = fcntl(read_descriptor.get(), F_GETFL);
  ASSERT_NE(-1, flags);
  EXPECT_NE(0, flags & O_NONBLOCK);

  int descriptor_flags = fcntl(read_descriptor.get(), F_GETFD);
  ASSERT_NE(-1, descriptor_flags);
  EXPECT_NE(0, descriptor_flags & FD_CLOEXEC);
}

TEST(PollableFdTest, RejectsInvalidFileDescriptor) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_rdma_pollable_fd_initialize(-1));
}

TEST(PollableFdTest, MapsMissingDeviceToUnavailable) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_net_rdma_pollable_fd_status_from_errno(
                            __FILE__, __LINE__, ENODEV, "test"));
}

TEST(PollableFdTest, AcceptsZeroErrno) {
  IREE_EXPECT_OK(iree_net_rdma_pollable_fd_status_from_errno(__FILE__, __LINE__,
                                                             0, "test"));
}

}  // namespace
