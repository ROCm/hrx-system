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
using ::iree::hal::stream::QueryableContext;
using ::iree::hal::stream::QueryableStream;

TEST(StreamingContextQueryTest, AllStreamsCompleteReportsComplete) {
  QueryableStream first(/*timeline_value=*/5, /*pending_value=*/5);
  QueryableStream second(/*timeline_value=*/7, /*pending_value=*/7);
  QueryableContext context({&first, &second});

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_context_query(context.get(), &status));
  EXPECT_EQ(status, 0);
  EXPECT_EQ(second.get()->completed_value, 7u);
}

// The first stream the list holds that is busy is the answer, and the streams
// the list holds after it are left alone. A complete stream advances its
// completed value when it is queried, so the third stream's completed value
// staying 0 is what shows the walk stopped at the second.
TEST(StreamingContextQueryTest, BusyStreamReportsNotCompleteAndEndsTheWalk) {
  QueryableStream complete(/*timeline_value=*/5, /*pending_value=*/5);
  QueryableStream busy(/*timeline_value=*/4, /*pending_value=*/5);
  QueryableStream behind(/*timeline_value=*/7, /*pending_value=*/7);
  QueryableContext context({&complete, &busy, &behind});

  int status = -1;
  IREE_ASSERT_OK(iree_hal_streaming_context_query(context.get(), &status));
  EXPECT_EQ(status, 1);
  EXPECT_EQ(complete.get()->completed_value, 5u);
  EXPECT_EQ(behind.get()->completed_value, 0u);
}

// A stream whose timeline reports a failure ends the walk and hands that
// failure back, which is the answer iree_hal_streaming_stream_query gives for
// the stream on its own. |status| is meaningful only on success, so only the
// returned status and the untouched stream the list holds after the failure
// are checked.
TEST(StreamingContextQueryTest, FailedStreamTimelineIsPropagated) {
  QueryableStream complete(/*timeline_value=*/5, /*pending_value=*/5);
  QueryableStream failed(FailedTimelineValue(IREE_STATUS_ABORTED),
                         /*pending_value=*/5);
  QueryableStream behind(/*timeline_value=*/7, /*pending_value=*/7);
  QueryableContext context({&complete, &failed, &behind});

  int status = -1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, iree_hal_streaming_context_query(
                                                 context.get(), &status));
  EXPECT_EQ(behind.get()->completed_value, 0u);
}

}  // namespace
