// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/schedule.h"

#include "iree/testing/gtest.h"

namespace {

TEST(TextScheduleTest, CompletionReservationCoversLegalWriteHighWater) {
  for (iree_host_size_t context = 1; context <= 260; ++context) {
    for (iree_host_size_t input = 1; input <= context; ++input) {
      for (iree_host_size_t outputs = 1;
           outputs <= iree_min(17, context - input + 1); ++outputs) {
        for (iree_host_size_t depth = 0; depth <= 7; ++depth) {
          // Every selected count is reachable by accepting only the anchor.
          // Enumerate the service's actual launch rule instead of reproducing
          // the closed-form reservation calculation.
          iree_host_size_t high_water = input;
          for (iree_host_size_t selected = 1; selected < outputs; ++selected) {
            const iree_host_size_t position = input + selected - 1;
            const bool verify = depth && outputs - selected > 1 &&
                                context - position >= depth + 1;
            high_water =
                iree_max(high_water, position + (verify ? depth + 1 : 1));
          }
          const iree_host_size_t expected = ((high_water + 63) / 64) * 64;
          ASSERT_EQ(loom_serve_text_request_reservation(context, input, outputs,
                                                        depth, 64),
                    expected)
              << "context=" << context << " input=" << input
              << " outputs=" << outputs << " depth=" << depth;
        }
      }
    }
  }
}

TEST(TextScheduleTest, CompletionReservationHandlesWideSourcePages) {
  for (iree_host_size_t page :
       {iree_host_size_t{1}, iree_host_size_t{3}, iree_host_size_t{262143},
        iree_host_size_t{UINT32_MAX}}) {
    for (iree_host_size_t extent : {1u, 3u, 262143u, 262144u}) {
      SCOPED_TRACE(page);
      SCOPED_TRACE(extent);
      const uint64_t expected = ((uint64_t{extent} + page - 1) / page) * page;
      EXPECT_EQ(loom_serve_text_request_reservation(262144, extent, 1, 0, page),
                expected);
    }
  }
}

}  // namespace
