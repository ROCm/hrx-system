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
using ::iree::hal::stream::QueryableEvent;

TEST(StreamingEventQueryTest, UnreachedRecordedPointReportsNotComplete) {
  QueryableEvent event(/*timeline_value=*/4, /*record_value=*/5);

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event.get(), &status));
  EXPECT_EQ(status, 1);
}

// The timeline reaching the recorded point reports complete, and so does a
// timeline that has run past it: the point is a lower bound, not an equality.
TEST(StreamingEventQueryTest, ReachedRecordedPointReportsComplete) {
  QueryableEvent event(/*timeline_value=*/5, /*record_value=*/5);

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event.get(), &status));
  EXPECT_EQ(status, 0);

  event.set_timeline_value(6);
  status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event.get(), &status));
  EXPECT_EQ(status, 0);
}

// An event names no timeline until a record of it is submitted: a record taken
// during stream capture produces no submission and leaves the point alone, and
// an event never passed to a record has never held one. A point naming no
// timeline reads as reached, so the query reports complete.
TEST(StreamingEventQueryTest, EventWithNoRecordReportsComplete) {
  QueryableEvent event;

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_event_query(event.get(), &status));
  EXPECT_EQ(status, 0);
}

// A failed timeline is returned, not folded into a completion answer. The
// recorded value sits below the failure sentinel the HAL reports alongside the
// status, so a query that dropped the status would read the timeline as reached
// and answer complete - the worst of the three possible answers.
TEST(StreamingEventQueryTest, FailedTimelineIsPropagated) {
  QueryableEvent event(FailedTimelineValue(IREE_STATUS_ABORTED),
                       /*record_value=*/5);

  // |status| is meaningful only on success, so only the returned status is
  // checked. The spy's destructor checks that the query released the reference
  // it took even on this path.
  int status = -1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_streaming_event_query(event.get(), &status));
}

}  // namespace
