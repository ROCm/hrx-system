// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/buffer_lease.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

TEST(BufferLeaseTest, AllocatesWritableHostStorage) {
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(
      iree_net_buffer_lease_allocate(32, iree_allocator_system(), &lease));
  EXPECT_EQ(lease.span.length, 32u);
  EXPECT_EQ(lease.span.region, nullptr);
  memset(iree_async_span_ptr(lease.span), 0xCD, lease.span.length);
  iree_async_buffer_lease_release(&lease);
  iree_async_buffer_lease_release(&lease);
}

TEST(BufferLeaseTest, FailedAllocationClearsOutput) {
  iree_async_buffer_lease_t lease;
  memset(&lease, 0xCD, sizeof(lease));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_buffer_lease_allocate(32, iree_allocator_null(), &lease));
  EXPECT_EQ(lease.span.region, nullptr);
  EXPECT_EQ(lease.span.offset, 0u);
  EXPECT_EQ(lease.span.length, 0u);
  EXPECT_EQ(lease.release.fn, nullptr);
}

}  // namespace
}  // namespace net
}  // namespace iree
