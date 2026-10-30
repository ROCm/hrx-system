// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch.h"

#include <vector>

#include "iree/testing/gtest.h"

namespace {

constexpr auto kSelect = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT;
constexpr auto kPropose = LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE;

class Epoch : public loom_serve_text_epoch_t {
 public:
  explicit Epoch(iree_host_size_t capacity) {
    IREE_CHECK_OK(loom_serve_text_epoch_initialize(
        capacity, iree_allocator_system(), this));
  }
  ~Epoch() { loom_serve_text_epoch_deinitialize(this); }
  Epoch(const Epoch&) = delete;
  Epoch& operator=(const Epoch&) = delete;
};

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
  const loom_serve_text_media_input_t media = {};
  loom_serve_text_epoch_row_t rows[3] = {};
  rows[0] = {
      .input_tokens = tokens,
      .input_count = 10,
      .input_offset = 37,
      .media = &media,
      .output_count = 7,
      .position = 5,
  };
  rows[1] = {.output_count = 1, .position = 94, .token = 42};
  rows[2] = {
      .input_tokens = tokens + 9,
      .input_count = 1,
      .output_count = 2,
      .position = 3,
  };
  const loom_serve_packed_span_t scheduled[] = {{7, 6}, {2, 1}, {11, 1}};
  const loom_serve_packing_plan_t plan = {
      .span_count = 3,
      .token_count = 8,
      .next_row = 8,
      .next_span = 1,
  };
  Epoch epoch(IREE_ARRAYSIZE(scheduled));
  loom_serve_text_continuation_t continuation;
  loom_serve_text_epoch_prepare(&options, rows, 12, &plan, scheduled, &epoch,
                                &continuation);
  EXPECT_EQ(epoch.count, 3);
  EXPECT_EQ(epoch.input_count, 8);
  EXPECT_EQ(epoch.device_epochs, 1);
  EXPECT_EQ(epoch.token_slots, 16);
  EXPECT_EQ(epoch.proposal_count, 0);
  EXPECT_EQ(continuation.span_count, 0);
  EXPECT_EQ(epoch.spans[0].row_index, 7);
  EXPECT_EQ(epoch.spans[0].token_ids, tokens);
  EXPECT_EQ(epoch.spans[0].flags, 0);
  EXPECT_EQ(epoch.spans[0].input_offset, 37u);
  EXPECT_EQ(epoch.spans[0].media, &media);
  EXPECT_EQ(epoch.positions[0], 5);
  EXPECT_TRUE(epoch.prefill[0]);
  EXPECT_EQ(epoch.spans[1].row_index, 2);
  EXPECT_EQ(epoch.spans[1].token_ids, &epoch.decode_tokens[1]);
  EXPECT_EQ(*epoch.spans[1].token_ids, 42);
  EXPECT_EQ(epoch.spans[1].flags, kSelect);
  EXPECT_EQ(epoch.spans[1].input_offset, 0u);
  EXPECT_EQ(epoch.spans[1].media, nullptr);
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
  const loom_serve_text_media_input_t media = {};
  loom_serve_text_epoch_row_t rows[4] = {};
  rows[0] = {
      .input_tokens = tokens,
      .input_count = 4,
      .input_offset = 30,
      .media = &media,
      .output_count = 9,
      .position = 10,
  };
  rows[1] = {.output_count = 9, .position = 8, .token = 20};
  rows[2] = {
      .input_tokens = tokens,
      .input_count = 1,
      .output_count = 3,
      .position = 20,
  };
  rows[3] = {.output_count = 1, .position = 40, .token = 21};
  // Only participant snapshots are present; resident IDs are not array indices.
  const loom_serve_packed_span_t scheduled[] = {
      {5, 2}, {3, 4}, {9, 1}, {12, 1}};
  const loom_serve_packing_plan_t plan = {
      .span_count = 4,
      .token_count = 8,
      .next_row = 6,
      .next_span = 1,
  };
  Epoch epoch(IREE_ARRAYSIZE(scheduled));
  loom_serve_text_continuation_t next;
  loom_serve_text_epoch_prepare(&options, rows, 13, &plan, scheduled, &epoch,
                                &next);
  EXPECT_EQ(epoch.device_epochs, 2);
  EXPECT_EQ(epoch.token_slots, 32);
  EXPECT_EQ(epoch.input_count, 8);
  EXPECT_EQ(epoch.proposal_count, 1);
  EXPECT_EQ(epoch.continued_prefill_count, 2);
  EXPECT_EQ(epoch.output_limits[1], 8);
  EXPECT_EQ(epoch.spans[1].flags, kSelect | kPropose);
  EXPECT_EQ(epoch.spans[2].flags, kSelect);
  EXPECT_EQ(epoch.spans[0].input_offset, 30u);
  EXPECT_EQ(epoch.spans[0].media, &media);
  ASSERT_EQ(next.span_count, 3);
  EXPECT_EQ(next.spans, epoch.planning.spans);
  EXPECT_EQ(next.output_limits, epoch.planning.output_limits);
  EXPECT_EQ(next.spans[0].row_index, 3);
  EXPECT_EQ(next.spans[0].token_ids, nullptr);
  EXPECT_EQ(next.spans[0].token_count, 4);
  EXPECT_EQ(next.spans[0].flags, kSelect | kPropose);
  EXPECT_EQ(next.spans[0].input_offset, 0u);
  EXPECT_EQ(next.spans[0].media, nullptr);
  EXPECT_EQ(next.output_limits[0], 8);
  EXPECT_EQ(next.spans[1].row_index, 9);
  EXPECT_EQ(next.spans[1].token_ids, nullptr);
  EXPECT_EQ(next.spans[1].token_count, 4);
  EXPECT_EQ(next.output_limits[1], 3);
  EXPECT_EQ(next.spans[2].row_index, 5);
  EXPECT_EQ(next.spans[2].token_ids, tokens + 2);
  EXPECT_EQ(next.spans[2].token_count, 2);
  EXPECT_EQ(next.spans[2].flags, kSelect);
  EXPECT_EQ(next.spans[2].input_offset, 32u);
  EXPECT_EQ(next.spans[2].media, &media);
  EXPECT_EQ(next.output_limits[2], 0);

  options.packing_mode = LOOM_SERVE_TEXT_PACKING_SEPARATE;
  loom_serve_text_epoch_prepare(&options, rows, 13, &plan, scheduled, &epoch,
                                &next);
  ASSERT_EQ(next.span_count, 2);
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
  loom_serve_text_epoch_row_t rows[] = {
      {.output_count = 9, .position = 60, .token = 21},
      {.output_count = 1, .position = 10, .token = 22}};
  const loom_serve_packed_span_t scheduled[] = {{4, 4}, {7, 1}};
  const loom_serve_packing_plan_t plan = {
      .span_count = 2,
      .token_count = 5,
      .next_row = 5,
      .next_span = 1,
  };
  Epoch epoch(IREE_ARRAYSIZE(scheduled));
  loom_serve_text_continuation_t continuation;
  loom_serve_text_epoch_prepare(&options, rows, 8, &plan, scheduled, &epoch,
                                &continuation);
  EXPECT_EQ(epoch.device_epochs, 1);
  EXPECT_EQ(epoch.token_slots, 16);
  EXPECT_EQ(epoch.output_limits[0], 4);
  EXPECT_EQ(epoch.output_limits[1], 0);
  EXPECT_EQ(continuation.span_count, 0);
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
    const loom_serve_packing_plan_t plan = {
        .span_count = 1,
        .token_count = shapes[first].token_capacity,
        .shape_index = first,
    };
    Epoch epoch(IREE_ARRAYSIZE(scheduled));
    loom_serve_text_continuation_t next;
    loom_serve_text_epoch_prepare(&options, rows, 1, &plan, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.device_epochs, 2);
    EXPECT_EQ(next.shape_index, 1 - first);
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
    const loom_serve_packing_plan_t plan = {.span_count = 1, .token_count = 1};
    Epoch epoch(IREE_ARRAYSIZE(scheduled));
    loom_serve_text_continuation_t next;
    loom_serve_text_epoch_prepare(&options, rows, 1, &plan, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.device_epochs, 1);
    EXPECT_EQ(epoch.token_slots, 0);
  }
}

TEST(TextEpochTest, CohortCapacityIsIndependentOfResidentIdentity) {
  const int32_t tokens[10] = {};
  for (iree_host_size_t depth = 1; depth <= 7; ++depth) {
    for (iree_host_size_t count :
         {1u, 2u, 3u, 15u, 16u, 17u, 31u, 32u, 33u, 64u, 100u, 129u}) {
      const loom_serve_packing_shape_t shape = {count * 8, count};
      Epoch epoch(count);
      for (iree_host_size_t epochs : {1u, 2u}) {
        SCOPED_TRACE(count);
        SCOPED_TRACE(epochs);
        const loom_serve_text_epoch_options_t options = {
            .context_capacity = 4096,
            .chunk_size = 8,
            .continuation_epochs = epochs,
            .draft_depth = depth,
            .shapes = &shape,
            .shape_count = 1,
        };
        std::vector<loom_serve_text_epoch_row_t> rows(count);
        std::vector<loom_serve_packed_span_t> scheduled(count);
        loom_serve_packing_plan_t plan = {
            .span_count = count,
            .next_span = count > 1 ? 1u : 0u,
        };
        for (iree_host_size_t i = 0; i < count; ++i) {
          const iree_host_size_t row = 1000000 - i * 13;
          rows[i] = {
              .input_tokens = i % 2 ? nullptr : tokens,
              .input_count = i % 2 ? 0u : 10u,
              .output_count = 3 + i,
              .position = i,
              .token = static_cast<int32_t>(i),
          };
          scheduled[i] = {row, i % 2 ? depth + 1 : 2u};
          plan.token_count += scheduled[i].token_count;
        }
        loom_serve_text_continuation_t next;
        loom_serve_text_epoch_prepare(&options, rows.data(), 1000001, &plan,
                                      scheduled.data(), &epoch, &next);
        EXPECT_EQ(epoch.count, count);
        for (iree_host_size_t i = 0; i < count; ++i) {
          EXPECT_EQ(epoch.spans[i].row_index, scheduled[i].row_index);
          EXPECT_EQ(epoch.positions[i], i);
          EXPECT_EQ(epoch.prefill[i], i % 2 == 0);
          EXPECT_LE(epoch.output_limits[i], rows[i].output_count);
          EXPECT_EQ(epoch.results[i].output_count, 0);
          if (i % 2) {
            EXPECT_EQ(epoch.spans[i].token_ids, &epoch.decode_tokens[i]);
            EXPECT_EQ(*epoch.spans[i].token_ids, i);
          } else {
            EXPECT_EQ(epoch.spans[i].token_ids, tokens);
          }
        }
        iree_host_size_t continued_tokens = 0;
        for (iree_host_size_t i = 0; i < next.span_count; ++i) {
          const auto& span = next.spans[i];
          const iree_host_size_t first = (1000000 - span.row_index) / 13;
          ASSERT_LT(first, count);
          EXPECT_EQ(span.row_index, scheduled[first].row_index);
          EXPECT_LE(next.output_limits[i], rows[first].output_count);
          continued_tokens += span.token_count;
        }
        EXPECT_LE(continued_tokens, shape.token_capacity);
        EXPECT_EQ(epoch.device_epochs, next.span_count ? 2 : 1);
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
    const loom_serve_packing_plan_t plan = {
        .span_count = 1,
        .token_count = width,
    };
    Epoch epoch(IREE_ARRAYSIZE(scheduled));
    loom_serve_text_continuation_t next;
    loom_serve_text_epoch_prepare(&options, rows, 1, &plan, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.spans[0].flags, kSelect | kPropose);
    EXPECT_EQ(epoch.output_limits[0], 2 * width);
    ASSERT_EQ(next.span_count, 1);
    EXPECT_EQ(next.spans[0].token_count, width);
    EXPECT_EQ(next.output_limits[0], 2 * width);
    ++rows[0].position;
    loom_serve_text_epoch_prepare(&options, rows, 1, &plan, scheduled, &epoch,
                                  &next);
    EXPECT_EQ(epoch.output_limits[0], width);
    EXPECT_EQ(next.span_count, 0);
  }
}

TEST(TextEpochTest, ContinuationLeavesFinalOutputToOrdinaryDecode) {
  const loom_serve_packing_shape_t shape = {16, 1};
  const int32_t input[] = {20, 21, 22};
  for (iree_host_size_t depth = 1; depth <= 7; ++depth) {
    SCOPED_TRACE(depth);
    const iree_host_size_t width = depth + 1;
    const loom_serve_text_epoch_options_t options = {
        .context_capacity = 256,
        .chunk_size = 8,
        .continuation_epochs = 2,
        .draft_depth = depth,
        .shapes = &shape,
        .shape_count = 1,
    };
    for (bool known : {false, true}) {
      SCOPED_TRACE(known);
      for (iree_host_size_t credit = 2; credit <= 2 * width; ++credit) {
        SCOPED_TRACE(credit);
        loom_serve_text_epoch_row_t rows[] = {{
            .input_tokens = known ? input : nullptr,
            .input_count = known ? IREE_ARRAYSIZE(input) : 0,
            .output_count = credit,
            .position = 60,
            .token = 21,
        }};
        const loom_serve_packed_span_t scheduled[] = {
            {0, known ? IREE_ARRAYSIZE(input) : width}};
        const loom_serve_packing_plan_t plan = {
            .span_count = 1,
            .token_count = scheduled[0].token_count,
        };
        Epoch epoch(IREE_ARRAYSIZE(scheduled));
        loom_serve_text_continuation_t next;
        loom_serve_text_epoch_prepare(&options, rows, 1, &plan, scheduled,
                                      &epoch, &next);
        const bool continues = credit >= 3;
        EXPECT_EQ(epoch.device_epochs, continues ? 2 : 1);
        ASSERT_EQ(next.span_count, continues ? 1 : 0);
        if (continues) {
          EXPECT_EQ(next.spans[0].token_count, width);
          EXPECT_EQ(next.output_limits[0],
                    iree_min(credit, known ? width + 1 : 2 * width));
        }
      }
    }
  }
}

TEST(TextEpochTest, ContinuationPreservesActualExpansionPriority) {
  const loom_serve_packing_shape_t shape = {16, 4};
  const loom_serve_text_epoch_options_t options = {
      .context_capacity = 64,
      .chunk_size = 8,
      .continuation_epochs = 2,
      .draft_depth = 3,
      .shapes = &shape,
      .shape_count = 1,
  };
  const loom_serve_ready_span_t ready[] = {{4, 4}, {4, 4}, {20, 1}, {20, 1}};
  loom_serve_packed_span_t scheduled[4], scratch[4];
  const auto plan =
      loom_serve_pack_shapes(4, ready, 1, &shape, 8, 0, scheduled, scratch);
  ASSERT_EQ(plan.span_count, 4);
  EXPECT_EQ(plan.next_row, 3);
  EXPECT_EQ(plan.next_span, 3);
  EXPECT_FALSE(plan.deferred);
  const int32_t tokens[20] = {};
  loom_serve_text_epoch_row_t rows[4] = {};
  for (size_t i = 0; i < plan.span_count; ++i) {
    const bool prefill = scheduled[i].row_index >= 2;
    rows[i] = {
        .input_tokens = prefill ? tokens : nullptr,
        .input_count = prefill ? 20u : 0u,
        .output_count = 16,
    };
  }
  ASSERT_EQ(scheduled[2].token_count, 7);
  ASSERT_EQ(scheduled[3].token_count, 1);
  Epoch epoch(4);
  loom_serve_text_continuation_t next;
  EXPECT_EQ(loom_serve_text_epoch_prepare(&options, rows, 4, &plan, scheduled,
                                          &epoch, &next),
            0);
  ASSERT_EQ(next.span_count, 4);
  size_t second[4] = {};
  for (size_t i = 0; i < next.span_count; ++i) {
    second[next.spans[i].row_index] = next.spans[i].token_count;
  }
  EXPECT_EQ(second[0], 4);
  EXPECT_EQ(second[1], 4);
  EXPECT_EQ(second[2], 1);
  EXPECT_EQ(second[3], 7);
}

TEST(TextEpochTest, CompactContinuationMatchesResidentDomainRotation) {
  // Compare compact continuation planning with the same leased work in the
  // full resident domain. The latter preserves idle gaps without ordinal
  // translation. The layouts cover idle rows, verifiers, finishing prompt
  // tails, and long prompts; some verifiers reach their context boundary.
  Epoch epoch(5);
  const int32_t tokens[64] = {};
  for (size_t width : {2u, 4u, 8u}) {
    for (uint32_t layout = 0; layout < 1024; ++layout) {
      const loom_serve_ready_span_t states[] = {
          {0, 0}, {width, width}, {1, 1}, {64, 1}};
      for (size_t start = 0; start < 5; ++start) {
        for (bool pressured : {false, true}) {
          for (auto mode : {LOOM_SERVE_TEXT_PACKING_MIXED,
                            LOOM_SERVE_TEXT_PACKING_SEPARATE}) {
            SCOPED_TRACE(::testing::Message()
                         << width << "/" << layout << "/" << start << "/"
                         << pressured << "/" << mode);
            loom_serve_ready_span_t ready[5];
            for (size_t row = 0; row < 5; ++row) {
              ready[row] = states[(layout >> (row * 2)) & 3];
            }
            auto separate = [](auto* values, size_t cursor) {
              for (size_t i = 0; i < 5; ++i) {
                const auto minimum = values[(cursor + i) % 5].minimum_count;
                if (!minimum) {
                  continue;
                }
                for (size_t row = 0; row < 5; ++row) {
                  if (values[row].minimum_count != minimum) {
                    values[row] = {};
                  }
                }
                break;
              }
            };
            if (mode == LOOM_SERVE_TEXT_PACKING_SEPARATE) {
              separate(ready, start);
            }
            size_t minima = 0;
            for (auto value : ready) {
              minima += value.minimum_count;
            }
            const loom_serve_packing_shape_t shape = {
                pressured ? width + 1 : minima + 4, pressured ? 2u : 5u};
            loom_serve_packed_span_t scheduled[5], scratch[5];
            const auto plan = loom_serve_pack_shapes(5, ready, 1, &shape, 64,
                                                     start, scheduled, scratch);
            if (!plan.span_count) {
              continue;
            }
            // Independent expected rotation from this test's original
            // readiness, not the newly retained producer metadata.
            size_t rotation_row = scheduled[0].row_index;
            for (size_t i = 0; i < plan.span_count; ++i) {
              if (scheduled[i].token_count >
                  ready[scheduled[i].row_index].minimum_count) {
                rotation_row = scheduled[i].row_index;
                break;
              }
            }
            const size_t continuation_start = (rotation_row + 1) % 5;
            loom_serve_text_epoch_row_t rows[5] = {};
            loom_serve_ready_span_t following[5] = {};
            for (size_t i = 0; i < plan.span_count; ++i) {
              const auto span = scheduled[i];
              const auto kind = (layout >> (span.row_index * 2)) & 3;
              const bool prefill = kind >= 2;
              const size_t position =
                  kind == 1 && !(span.row_index % 2) ? 128 - width : 0;
              rows[i] = {
                  .input_tokens = prefill ? tokens : nullptr,
                  .input_count =
                      prefill ? ready[span.row_index].token_count : 0,
                  .output_count = kind == 2 ? 2u : 16u,
                  .position = position,
              };
              if (kind == 3) {
                following[span.row_index] = {64 - span.token_count, 1};
              } else if (kind == 1 && !position) {
                following[span.row_index] = {width, width};
              }
            }
            if (mode == LOOM_SERVE_TEXT_PACKING_SEPARATE) {
              separate(following, continuation_start);
            }
            loom_serve_packed_span_t expected[5];
            const auto reference =
                loom_serve_pack_shapes(5, following, 1, &shape, 64,
                                       continuation_start, expected, scratch);
            const loom_serve_text_epoch_options_t options = {
                .context_capacity = 128,
                .chunk_size = 64,
                .continuation_epochs = 2,
                .draft_depth = width - 1,
                .packing_mode = mode,
                .shapes = &shape,
                .shape_count = 1,
            };
            loom_serve_text_continuation_t next;
            const auto cursor = loom_serve_text_epoch_prepare(
                &options, rows, 5, &plan, scheduled, &epoch, &next);
            ASSERT_EQ(next.span_count, reference.span_count);
            EXPECT_EQ(cursor, plan.deferred || !reference.span_count
                                  ? plan.next_row
                                  : reference.next_row);
            EXPECT_EQ(epoch.input_count, plan.token_count);
            EXPECT_EQ(epoch.device_epochs, next.span_count ? 2 : 1);
            for (size_t i = 0; i < next.span_count; ++i) {
              EXPECT_EQ(next.spans[i].row_index, expected[i].row_index);
              EXPECT_EQ(next.spans[i].token_count, expected[i].token_count);
              const size_t row = expected[i].row_index;
              const bool proposes = following[row].minimum_count > 1;
              EXPECT_EQ(next.spans[i].flags & kPropose,
                        proposes ? kPropose : 0);
              EXPECT_LE(next.output_limits[i], 16);
            }
          }
        }
      }
    }
  }
}

}  // namespace
