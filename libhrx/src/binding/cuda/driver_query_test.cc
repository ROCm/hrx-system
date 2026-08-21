// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/cuda/driver.h"
#include "common/streaming_query_test_util.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace {

using ::iree::hal::stream::FailedTimelineValue;
using ::iree::hal::stream::QueryableEvent;
using ::iree::hal::stream::QueryableStream;

// cuEventQuery runs no initialization and consults no current context, so a
// hand-built event drives it directly.
TEST(CudaDriverQueryTest, EventQueryIsNotReadyBeforeTheRecordIsReached) {
  QueryableEvent event(/*timeline_value=*/4, /*record_value=*/5);

  EXPECT_EQ(cuEventQuery(reinterpret_cast<CUevent>(event.get())),
            CUDA_ERROR_NOT_READY);
}

TEST(CudaDriverQueryTest, EventQuerySucceedsOnceTheRecordIsReached) {
  QueryableEvent event(/*timeline_value=*/5, /*record_value=*/5);

  EXPECT_EQ(cuEventQuery(reinterpret_cast<CUevent>(event.get())), CUDA_SUCCESS);
}

// A failed query is an error, not a completion answer. The entry point starts
// its out-parameter at 0 and reads 0 as complete, so the iree_status_is_ok
// guard ahead of that read is what keeps a failure from returning CUDA_SUCCESS
// off the initializer. IREE_STATUS_ABORTED is the code these two tests use
// because iree_status_to_cu_result maps it through its default arm to
// CUDA_ERROR_UNKNOWN, distinct from both CUDA_SUCCESS and CUDA_ERROR_NOT_READY;
// IREE_STATUS_UNAVAILABLE maps to CUDA_ERROR_NOT_READY, which would not tell a
// propagated failure apart from a not-complete answer. What these two pin is
// that the answer is an error at all; that the timeline's own code reaches the
// caller intact is pinned a layer down, where the encoding is decoded.
TEST(CudaDriverQueryTest, EventQueryReportsAFailedTimelineAsAnError) {
  QueryableEvent event(FailedTimelineValue(IREE_STATUS_ABORTED),
                       /*record_value=*/5);

  EXPECT_EQ(cuEventQuery(reinterpret_cast<CUevent>(event.get())),
            CUDA_ERROR_UNKNOWN);
}

// cuStreamQuery consults the current context only to resolve a null handle,
// which no test here passes, so a hand-built stream drives it directly.
TEST(CudaDriverQueryTest, StreamQueryIsNotReadyWhileWorkIsPending) {
  QueryableStream stream(/*timeline_value=*/4, /*pending_value=*/5);

  EXPECT_EQ(cuStreamQuery(reinterpret_cast<CUstream>(stream.get())),
            CUDA_ERROR_NOT_READY);
}

TEST(CudaDriverQueryTest, StreamQuerySucceedsOnceThePendingValueIsReached) {
  QueryableStream stream(/*timeline_value=*/5, /*pending_value=*/5);

  EXPECT_EQ(cuStreamQuery(reinterpret_cast<CUstream>(stream.get())),
            CUDA_SUCCESS);
}

// The same guard, in cuStreamQuery.
TEST(CudaDriverQueryTest, StreamQueryReportsAFailedTimelineAsAnError) {
  QueryableStream stream(FailedTimelineValue(IREE_STATUS_ABORTED),
                         /*pending_value=*/5);

  EXPECT_EQ(cuStreamQuery(reinterpret_cast<CUstream>(stream.get())),
            CUDA_ERROR_UNKNOWN);
}

}  // namespace
