// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/internal.h"
#include "common/streaming_query_test_util.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::hal::stream::FailedTimelineValue;
using ::iree::hal::stream::QueryableStream;

TEST(StreamingStreamQueryTest, UnreachedPendingValueReportsNotComplete) {
  QueryableStream stream(/*timeline_value=*/4, /*pending_value=*/5);

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(stream.get(), &status));
  EXPECT_EQ(status, 1);
}

// The timeline reaching the pending value reports complete, and so does a
// timeline that has run past it: the pending value is a lower bound, not an
// equality. A complete answer also advances the stream's completed value to
// what the timeline reported.
TEST(StreamingStreamQueryTest, ReachedPendingValueReportsComplete) {
  QueryableStream stream(/*timeline_value=*/5, /*pending_value=*/5);

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(stream.get(), &status));
  EXPECT_EQ(status, 0);
  EXPECT_EQ(stream.get()->completed_value, 5u);

  stream.set_timeline_value(6);
  status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_stream_query(stream.get(), &status));
  EXPECT_EQ(status, 0);
  EXPECT_EQ(stream.get()->completed_value, 6u);
}

// A failed timeline is returned, not folded into a completion answer, which is
// the answer iree_hal_streaming_event_query gives as well. The pending value
// sits below the failure sentinel the HAL reports alongside the status, so a
// query that dropped the status would read the timeline as reached and answer
// complete.
TEST(StreamingStreamQueryTest, FailedTimelineIsPropagated) {
  QueryableStream stream(FailedTimelineValue(IREE_STATUS_ABORTED),
                         /*pending_value=*/5);

  // |status| is meaningful only on success, so only the returned status is
  // checked.
  int status = -1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_streaming_stream_query(stream.get(), &status));
}

}  // namespace
