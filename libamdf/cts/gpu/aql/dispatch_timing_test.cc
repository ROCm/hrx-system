// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/dispatch_timing_fixture.h"

namespace {

TEST_F(AqlDispatchTimingTest, DisabledQueueLeavesTimestampFieldsUntouched) {
  Exercise({false}, 17);
}

TEST_F(AqlDispatchTimingTest, FiniteDispatchesPublishOwnSignalTimestamps) {
  Exercise({true}, 17);
}

TEST_F(AqlDispatchTimingTest, EnablesAndDisablesCaptureOnTheSameUsedQueue) {
  Exercise({false, true, true, false, false, true, false}, 17);
}

TEST_F(AqlDispatchTimingTest,
       SelectsCaptureBeforeFirstPublicationAndChangesIt) {
  Exercise({true, false, false, true}, 17);
}

}  // namespace
