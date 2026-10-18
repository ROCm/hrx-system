// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/wire.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class WireTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_realtime_wire_initialize(1, nullptr, nullptr,
                                        iree_allocator_system(), &wire_);
  }
  void TearDown() override { loom_serve_realtime_wire_deinitialize(&wire_); }

  iree_status_t Append(iree_host_size_t padding = 0) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(&wire_, "test"));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&wire_.scratch, ",\"text\":\""));
    const std::string text(padding, 'a');
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
        &wire_.scratch, iree_make_string_view(text.data(), text.size())));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(&wire_.scratch, "\"}"));
    return loom_serve_realtime_wire_commit(&wire_, 0, false);
  }

  // Real queue owner; these tests exercise construction without transport
  // flush.
  loom_serve_realtime_wire_t wire_;
};

TEST_F(WireTest, RewindPreservesEarlierQueuedEvents) {
  IREE_ASSERT_OK(Append());
  auto* tail = wire_.last;
  const iree_host_size_t retained_bytes = wire_.bytes;
  IREE_ASSERT_OK(Append(100));
  IREE_ASSERT_OK(Append(200));
  ASSERT_EQ(wire_.count, 3);
  loom_serve_realtime_wire_rewind(&wire_, tail);
  EXPECT_EQ(wire_.count, 1);
  EXPECT_EQ(wire_.bytes, retained_bytes);
  EXPECT_EQ(wire_.first, tail);
  EXPECT_EQ(wire_.last, tail);
  IREE_ASSERT_OK(Append());
  EXPECT_EQ(wire_.count, 2);
  EXPECT_EQ(wire_.first, tail);
  EXPECT_EQ(wire_.sequence, 4);
}

TEST_F(WireTest, OversizedEventReleasesScratchForTheError) {
  IREE_ASSERT_OK(Append());
  auto* tail = wire_.last;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Append(512 * 1024));
  EXPECT_EQ(wire_.count, 1);
  EXPECT_EQ(wire_.last, tail);
  EXPECT_EQ(wire_.scratch.capacity, 0);
  IREE_ASSERT_OK(Append());
  EXPECT_EQ(wire_.count, 2);
}

TEST_F(WireTest, AggregatePressureRetainsTheQueueUntilRollback) {
  for (int i = 0; i < 64; ++i) {
    IREE_ASSERT_OK(Append());
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Append());
  EXPECT_EQ(wire_.count, 64);
  loom_serve_realtime_wire_rewind(&wire_, nullptr);
  EXPECT_EQ(wire_.count, 0);
  EXPECT_EQ(wire_.bytes, 0);
  EXPECT_EQ(wire_.first, nullptr);
  EXPECT_EQ(wire_.last, nullptr);
  IREE_ASSERT_OK(Append());
  EXPECT_EQ(wire_.count, 1);
}

}  // namespace
