// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch.h"

#include "iree/testing/gtest.h"

namespace {

constexpr auto kSelect = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT;
constexpr auto kPropose = LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE;

TEST(TextEpochTest, KnownInputsAndDecodePreserveRowIdentityAndPhase) {
  const loom_serve_packing_shape_t shape = {16, 4};
  const loom_serve_text_epoch_options_t options = {
      .context_capacity = 96,
      .chunk_size = 8,
      .continuation_epochs = 1,
      .shapes = &shape,
      .shape_count = 1,
  };
  const int32_t tokens[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  loom_serve_text_epoch_row_t rows[16] = {};
  rows[7] = {
      .input_tokens = tokens,
      .input_count = 10,
      .output_count = 7,
      .position = 5,
  };
  rows[2] = {.output_count = 1, .position = 94, .token = 42};
  rows[11] = {
      .input_tokens = tokens + 9,
      .input_count = 1,
      .output_count = 2,
      .position = 3,
  };
  const loom_serve_packed_span_t scheduled[] = {{7, 6}, {2, 1}, {11, 1}};
  loom_serve_text_epoch_t epoch;
  loom_serve_text_epoch_continuation_t continuation;
  loom_serve_text_epoch_prepare(&options, rows, 0, 3, scheduled, &epoch,
                                &continuation);
  EXPECT_EQ(epoch.count, 3);
  EXPECT_EQ(epoch.input_count, 8);
  EXPECT_EQ(epoch.device_epochs, 1);
  EXPECT_EQ(epoch.token_slots, 16);
  EXPECT_EQ(epoch.proposal_count, 0);
  EXPECT_EQ(continuation.continuation.span_count, 0);
  EXPECT_EQ(epoch.spans[0].row_index, 7);
  EXPECT_EQ(epoch.spans[0].token_ids, tokens);
  EXPECT_EQ(epoch.spans[0].flags, 0);
  EXPECT_EQ(epoch.positions[0], 5);
  EXPECT_TRUE(epoch.prefill[0]);
  EXPECT_EQ(epoch.spans[1].row_index, 2);
  EXPECT_EQ(epoch.spans[1].token_ids, &epoch.decode_tokens[1]);
  EXPECT_EQ(*epoch.spans[1].token_ids, 42);
  EXPECT_EQ(epoch.spans[1].flags, kSelect);
  EXPECT_FALSE(epoch.prefill[1]);
  EXPECT_EQ(epoch.spans[2].row_index, 11);
  EXPECT_EQ(epoch.spans[2].token_ids, tokens + 9);
  EXPECT_EQ(epoch.spans[2].flags, kSelect);
  EXPECT_TRUE(epoch.prefill[2]);
}

TEST(TextEpochTest, ContinuationMixesPromptTailsAndBoundedVerifiers) {
  const loom_serve_packing_shape_t shape = {16, 4};
  loom_serve_text_epoch_options_t options = {
      .context_capacity = 64,
      .chunk_size = 8,
      .continuation_epochs = 2,
      .draft_depth = 3,
      .shapes = &shape,
      .shape_count = 1,
  };
  const int32_t tokens[] = {10, 11, 12, 13};
  loom_serve_text_epoch_row_t rows[16] = {};
  rows[5] = {
      .input_tokens = tokens,
      .input_count = 4,
      .output_count = 9,
      .position = 10,
  };
  rows[3] = {.output_count = 9, .position = 8, .token = 20};
  rows[9] = {
      .input_tokens = tokens,
      .input_count = 1,
      .output_count = 3,
      .position = 20,
  };
  rows[12] = {.output_count = 1, .position = 40, .token = 21};
  // Ready work outside the leased cohort is not a continuation candidate.
  rows[0] = {.output_count = 9, .position = 1, .token = 22};
  const loom_serve_packed_span_t scheduled[] = {
      {5, 2}, {3, 4}, {9, 1}, {12, 1}};
  loom_serve_text_epoch_t epoch;
  loom_serve_text_epoch_continuation_t next;
  loom_serve_text_epoch_prepare(&options, rows, 0, 4, scheduled, &epoch, &next);
  EXPECT_EQ(epoch.device_epochs, 2);
  EXPECT_EQ(epoch.token_slots, 32);
  EXPECT_EQ(epoch.input_count, 8);
  EXPECT_EQ(epoch.proposal_count, 1);
  EXPECT_EQ(epoch.continued_prefill_count, 2);
  EXPECT_EQ(epoch.output_limits[1], 8);
  EXPECT_EQ(epoch.spans[1].flags, kSelect | kPropose);
  EXPECT_EQ(epoch.spans[2].flags, kSelect);
  ASSERT_EQ(next.continuation.span_count, 3);
  EXPECT_EQ(next.continuation.spans, next.spans);
  EXPECT_EQ(next.continuation.output_limits, next.output_limits);
  EXPECT_EQ(next.spans[0].row_index, 3);
  EXPECT_EQ(next.spans[0].token_ids, nullptr);
  EXPECT_EQ(next.spans[0].token_count, 4);
  EXPECT_EQ(next.spans[0].flags, kSelect | kPropose);
  EXPECT_EQ(next.output_limits[0], 8);
  EXPECT_EQ(next.spans[1].row_index, 9);
  EXPECT_EQ(next.spans[1].token_ids, nullptr);
  EXPECT_EQ(next.spans[1].token_count, 4);
  EXPECT_EQ(next.output_limits[1], 3);
  EXPECT_EQ(next.spans[2].row_index, 5);
  EXPECT_EQ(next.spans[2].token_ids, tokens + 2);
  EXPECT_EQ(next.spans[2].token_count, 2);
  EXPECT_EQ(next.spans[2].flags, kSelect);
  EXPECT_EQ(next.output_limits[2], 0);

  options.packing_mode = LOOM_SERVE_TEXT_PACKING_SEPARATE;
  loom_serve_text_epoch_prepare(&options, rows, 0, 4, scheduled, &epoch, &next);
  ASSERT_EQ(next.continuation.span_count, 2);
  EXPECT_EQ(next.spans[0].row_index, 3);
  EXPECT_EQ(next.spans[1].row_index, 9);
  EXPECT_EQ(epoch.continued_prefill_count, 0);
}

TEST(TextEpochTest, NoContinuationReducesCreditToOneVerifier) {
  const loom_serve_packing_shape_t shape = {16, 4};
  const loom_serve_text_epoch_options_t options = {
      .context_capacity = 64,
      .chunk_size = 8,
      .continuation_epochs = 2,
      .draft_depth = 3,
      .shapes = &shape,
      .shape_count = 1,
  };
  loom_serve_text_epoch_row_t rows[16] = {};
  rows[4] = {.output_count = 9, .position = 60, .token = 21};
  rows[7] = {.output_count = 1, .position = 10, .token = 22};
  const loom_serve_packed_span_t scheduled[] = {{4, 4}, {7, 1}};
  loom_serve_text_epoch_t epoch;
  loom_serve_text_epoch_continuation_t continuation;
  loom_serve_text_epoch_prepare(&options, rows, 0, 2, scheduled, &epoch,
                                &continuation);
  EXPECT_EQ(epoch.device_epochs, 1);
  EXPECT_EQ(epoch.token_slots, 16);
  EXPECT_EQ(epoch.output_limits[0], 4);
  EXPECT_EQ(epoch.output_limits[1], 0);
  EXPECT_EQ(continuation.continuation.span_count, 0);
}

TEST(TextEpochTest, ContinuationCapacityTracksItsIndependentlySelectedBucket) {
  const loom_serve_packing_shape_t shapes[] = {{4, 1}, {16, 1}};
  const loom_serve_text_epoch_options_t options = {
      .context_capacity = 64,
      .chunk_size = 16,
      .continuation_epochs = 2,
      .draft_depth = 3,
      .shapes = shapes,
      .shape_count = IREE_ARRAYSIZE(shapes),
  };
  const int32_t tokens[16] = {};
  const loom_serve_text_epoch_row_t rows[] = {
      {.input_tokens = tokens, .input_count = 16, .output_count = 9}};
  for (iree_host_size_t first : {0u, 1u}) {
    SCOPED_TRACE(first);
    const loom_serve_packed_span_t scheduled[] = {
        {0, shapes[first].token_capacity}};
    loom_serve_text_epoch_t epoch;
    loom_serve_text_epoch_continuation_t next;
    loom_serve_text_epoch_prepare(&options, rows, first, 1, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.device_epochs, 2);
    EXPECT_EQ(next.continuation.shape_index, 1 - first);
    EXPECT_EQ(epoch.token_slots, 20);
  }
}

TEST(TextEpochTest, IsolatedStagesDoNotClaimPackedCapacity) {
  const loom_serve_packing_shape_t shape = {16, 4};
  for (auto mode :
       {LOOM_SERVE_TEXT_SCHEDULE_ISOLATED, LOOM_SERVE_TEXT_SCHEDULE_MATCHED}) {
    const loom_serve_text_epoch_options_t options = {
        .context_capacity = 64,
        .chunk_size = 8,
        .continuation_epochs = 1,
        .schedule_mode = mode,
        .shapes = &shape,
        .shape_count = 1,
    };
    const loom_serve_text_epoch_row_t rows[] = {
        {.output_count = 9, .position = 3, .token = 21}};
    const loom_serve_packed_span_t scheduled[] = {{0, 1}};
    loom_serve_text_epoch_t epoch;
    loom_serve_text_epoch_continuation_t next;
    loom_serve_text_epoch_prepare(&options, rows, 0, 1, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.device_epochs, 1);
    EXPECT_EQ(epoch.token_slots, 0);
  }
}

TEST(TextEpochTest, RowCapacityFamilyPreservesBorrowedInputsAndOutputCredit) {
  const loom_serve_packing_shape_t shape = {128, 16};
  const int32_t tokens[10] = {};
  for (iree_host_size_t depth = 1; depth <= 7; ++depth) {
    for (iree_host_size_t count = 1; count <= 16; ++count) {
      for (iree_host_size_t epochs : {1u, 2u}) {
        SCOPED_TRACE(count);
        SCOPED_TRACE(epochs);
        const loom_serve_text_epoch_options_t options = {
            .context_capacity = 64,
            .chunk_size = 8,
            .continuation_epochs = epochs,
            .draft_depth = depth,
            .shapes = &shape,
            .shape_count = 1,
        };
        loom_serve_text_epoch_row_t rows[16] = {};
        loom_serve_packed_span_t scheduled[16];
        for (iree_host_size_t i = 0; i < count; ++i) {
          const iree_host_size_t row = 15 - i;
          rows[row] = {
              .input_tokens = i % 2 ? nullptr : tokens,
              .input_count = i % 2 ? 0u : 10u,
              .output_count = 3 + i,
              .position = i,
              .token = static_cast<int32_t>(i),
          };
          scheduled[i] = {row, i % 2 ? depth + 1 : 2u};
        }
        loom_serve_text_epoch_t epoch;
        loom_serve_text_epoch_continuation_t next;
        loom_serve_text_epoch_prepare(&options, rows, 0, count, scheduled,
                                      &epoch, &next);
        EXPECT_EQ(epoch.count, count);
        for (iree_host_size_t i = 0; i < count; ++i) {
          EXPECT_EQ(epoch.spans[i].row_index, scheduled[i].row_index);
          EXPECT_EQ(epoch.positions[i], i);
          EXPECT_EQ(epoch.prefill[i], i % 2 == 0);
          EXPECT_LE(epoch.output_limits[i], rows[15 - i].output_count);
          EXPECT_EQ(epoch.results[i].output_count, 0);
          if (i % 2) {
            EXPECT_EQ(epoch.spans[i].token_ids, &epoch.decode_tokens[i]);
            EXPECT_EQ(*epoch.spans[i].token_ids, i);
          } else {
            EXPECT_EQ(epoch.spans[i].token_ids, tokens);
          }
        }
        iree_host_size_t continued_tokens = 0;
        for (iree_host_size_t i = 0; i < next.continuation.span_count; ++i) {
          const auto& span = next.spans[i];
          EXPECT_GE(span.row_index, 16 - count);
          EXPECT_LE(next.output_limits[i], rows[span.row_index].output_count);
          continued_tokens += span.token_count;
        }
        EXPECT_LE(continued_tokens, shape.token_capacity);
        EXPECT_EQ(epoch.device_epochs, next.continuation.span_count ? 2 : 1);
        EXPECT_EQ(epoch.token_slots,
                  shape.token_capacity * epoch.device_epochs);
      }
    }
  }
}

TEST(TextEpochTest, DraftWidthControlsContinuationCreditAndContextFrontier) {
  const loom_serve_packing_shape_t shape = {128, 16};
  for (iree_host_size_t depth = 1; depth <= 7; ++depth) {
    const iree_host_size_t width = depth + 1;
    const loom_serve_text_epoch_options_t options = {
        .context_capacity = 64,
        .chunk_size = 8,
        .continuation_epochs = 2,
        .draft_depth = depth,
        .shapes = &shape,
        .shape_count = 1,
    };
    loom_serve_text_epoch_row_t rows[] = {
        {.output_count = 16, .position = 64 - width - 1, .token = 21}};
    const loom_serve_packed_span_t scheduled[] = {{0, width}};
    loom_serve_text_epoch_t epoch;
    loom_serve_text_epoch_continuation_t next;
    loom_serve_text_epoch_prepare(&options, rows, 0, 1, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.spans[0].flags, kSelect | kPropose);
    EXPECT_EQ(epoch.output_limits[0], 2 * width);
    ASSERT_EQ(next.continuation.span_count, 1);
    EXPECT_EQ(next.spans[0].token_count, width);
    EXPECT_EQ(next.output_limits[0], 2 * width);
    ++rows[0].position;
    loom_serve_text_epoch_prepare(&options, rows, 0, 1, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.output_limits[0], width);
    EXPECT_EQ(next.continuation.span_count, 0);
  }
}

}  // namespace
