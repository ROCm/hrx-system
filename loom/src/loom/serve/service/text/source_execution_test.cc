// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/service_test_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_F(SourceTextServiceTest, ChunkingRowsAndForksMatchDenseOracle) {
  const std::string prompt = "abxabbaaxbabx";
  std::vector<int32_t> tokens;
  for (char character : prompt) {
    tokens.push_back(Oracle::Id(character));
  }
  Oracle reference;
  reference.Consume(prompt);
  auto* full = loom_serve_text_model_row(model_, 0);
  auto* chunks = loom_serve_text_model_row(model_, 1);
  IREE_ASSERT_OK(
      loom_serve_text_row_prefill(full, tokens.size(), tokens.data()));
  for (size_t offset = 0; offset < tokens.size(); offset += 3) {
    IREE_ASSERT_OK(loom_serve_text_row_prefill(
        chunks, std::min<size_t>(3, tokens.size() - offset),
        tokens.data() + offset));
  }
  EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(chunks), reference.Predict());
  loom_serve_text_checkpoint_t* checkpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(full, &checkpoint));
  ASSERT_NE(checkpoint, nullptr);
  auto* fork = loom_serve_text_model_row(model_, 2);
  bool restored = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_restore(fork, checkpoint, &restored));
  ASSERT_TRUE(restored);
  const auto before = loom_serve_text_model_pool_usage(model_);
  const int32_t a = 0;
  const int32_t b = 1;
  const loom_serve_text_span_t spans[] = {
      {0, 1, &a, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {2, 1, &b, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  IREE_ASSERT_OK(
      loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(spans), spans));
  Oracle branch = reference;
  reference.Consume(a);
  branch.Consume(b);
  EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
  EXPECT_EQ(loom_serve_text_row_token(fork), branch.Predict());
  // Both writers detach the pinned partial page; the immutable endpoint stays.
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available + 8,
            before.available);
  IREE_ASSERT_OK(
      loom_serve_text_row_try_restore(chunks, checkpoint, &restored));
  ASSERT_TRUE(restored);
  IREE_ASSERT_OK(loom_serve_text_row_prefill(chunks, 1, &b));
  EXPECT_EQ(loom_serve_text_row_token(chunks), branch.Predict());
  loom_serve_text_checkpoint_release(checkpoint);
  for (size_t i = 0; i < 4; ++i) {
    IREE_ASSERT_OK(
        loom_serve_text_row_reset(loom_serve_text_model_row(model_, i)));
  }
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
}

TEST_F(SourceTextServiceTest, WideSourceEpochsRetainExactHttpHistory) {
  for (size_t capacity : {513u, 1024u, 4097u}) {
    SCOPED_TRACE(capacity);
    IREE_ASSERT_OK(loom_serve_text_model_free(model_));
    model_ = nullptr;
    const loom_serve_packing_shape_t shapes[] = {{32, 4}, {capacity, 4}};
    auto options = ModelOptions();
    options.prefill_capacity = capacity;
    options.context_capacity = capacity + 64;
    options.pool_capacity = 4 * ((options.context_capacity + 3) / 4 * 4);
    options.epoch_count = IREE_ARRAYSIZE(shapes);
    options.epoch_shapes = shapes;
    IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                  host_allocator_, &model_));
    EXPECT_EQ(loom_serve_text_model_prefill_capacity(model_), capacity);
    std::vector<int32_t> tokens(capacity - 9);
    Oracle reference;
    for (size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = i % 3;
      reference.Consume(tokens[i]);
    }
    auto* full = loom_serve_text_model_row(model_, 0);
    auto* chunks = loom_serve_text_model_row(model_, 1);
    IREE_ASSERT_OK(
        loom_serve_text_row_prefill(full, tokens.size(), tokens.data()));
    for (size_t offset = 0; offset < tokens.size(); offset += 31) {
      IREE_ASSERT_OK(loom_serve_text_row_prefill(
          chunks, std::min<size_t>(31, tokens.size() - offset),
          tokens.data() + offset));
    }
    EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
    EXPECT_EQ(loom_serve_text_row_token(chunks), reference.Predict());

    // The decode row and a fresh long prefill fill the same wide epoch.
    const int32_t pending = loom_serve_text_row_token(full);
    tokens.resize(capacity - 1, 1);
    const loom_serve_text_span_t spans[] = {
        {0, 1, &pending, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
        {2, tokens.size(), tokens.data(), LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
    };
    IREE_ASSERT_OK(
        loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(spans), spans));
    reference.Consume(pending);
    EXPECT_EQ(loom_serve_text_row_token(full), reference.Predict());
    Oracle fresh;
    for (int32_t token : tokens) {
      fresh.Consume(token);
    }
    EXPECT_EQ(loom_serve_text_row_token(loom_serve_text_model_row(model_, 2)),
              fresh.Predict());
    for (size_t row = 0; row < 4; ++row) {
      IREE_ASSERT_OK(
          loom_serve_text_row_reset(loom_serve_text_model_row(model_, row)));
    }
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available,
              options.pool_capacity);
  }

  // Same server binary and source package, now with a 4097-token host shape.
  Start(4097);
  std::string prompt;
  for (size_t i = 0; i < 4088; ++i) {
    prompt.push_back("abx"[i % 3]);
  }
  const int retained =
      Send(Request(Message("user", prompt), 2), "X-Loom-Session: wide\r\n");
  const int other = Send(Request(Message("user", std::string(769, 'x'))));
  const std::string first = Content(Read(retained));
  ASSERT_EQ(first, Oracle::Complete(prompt, 2));
  EXPECT_EQ(Content(Read(other)), Oracle::Complete(std::string(769, 'x'), 16));
  const std::string request =
      Request(Message("user", prompt) + "," + Message("assistant", first) +
              "," + Message("user", "bax"));
  const std::string resumed = Read(Send(request, "X-Loom-Session: wide\r\n"));
  EXPECT_EQ(Content(resumed), Oracle::Complete(prompt + first + "#bax", 16));
  EXPECT_NE(resumed.find("\"cached_tokens\":4089"), std::string::npos)
      << resumed;
  EXPECT_EQ(Content(Read(Send(request))), Content(resumed));
}

TEST_F(SourceTextServiceTest,
       SourceRejectsUnsupportedConfigurationAndUsesWeights) {
  auto options = ModelOptions();
  options.draft_depth = 3;
  loom_serve_text_model_t* unsupported = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  options.draft_depth = 0;
  options.options = IREE_SV("{\"unknown\":true}");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  options.options = IREE_SV("{}");
  options.pool_capacity = 0;
  options.checkpoint_capacity = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_text_model_allocate(
                            device_, &options, host_allocator_, &unsupported));
  EXPECT_EQ(unsupported, nullptr);
  IREE_ASSERT_OK(loom_serve_text_model_free(model_));
  model_ = nullptr;
  ASSERT_NO_FATAL_FAILURE(WriteWeights(7));
  options = ModelOptions();
  IREE_ASSERT_OK(loom_serve_text_model_allocate(device_, &options,
                                                host_allocator_, &model_));
  Start();
  const std::string prompt = "axb";
  const std::string actual =
      Content(Read(Send(Request(Message("user", prompt)))));
  EXPECT_EQ(actual, Oracle::Complete(prompt, 16, 7));
  EXPECT_NE(actual, Oracle::Complete(prompt, 16, 2));
}

TEST_F(SourceTextServiceTest, OrderedImageContentReachesRetainedExecution) {
  Start();
  const std::string message = ImageMessage("a", "image/png", kPng, "low", "b");
  EXPECT_EQ(Content(Read(Send(Request(message, 2)))),
            Oracle::Complete("axb", 2));
}

TEST_F(SourceTextServiceTest,
       ProjectedMediaRowsReplacePlaceholderIdsAcrossChunks) {
  const int32_t projected[] = {7, 11};
  const iree_const_byte_span_t rows[] = {
      iree_make_const_byte_span(&projected[0], sizeof(projected[0])),
      iree_make_const_byte_span(&projected[1], sizeof(projected[1])),
  };
  const loom_serve_text_media_input_span_t media_spans[] = {{
      .input_offset = 1,
      .input_count = 2,
      .row_offset = 0,
  }};
  const loom_serve_text_media_input_t media = {
      .spans = media_spans,
      .span_count = IREE_ARRAYSIZE(media_spans),
      .rows = rows,
      .row_count = IREE_ARRAYSIZE(rows),
  };
  const int32_t tokens[] = {Oracle::Id('a'), Oracle::Id('x'), Oracle::Id('x'),
                            Oracle::Id('b')};
  const loom_serve_text_span_t first[] = {
      {
          .row_index = 0,
          .token_count = 2,
          .token_ids = tokens,
          .input_offset = 0,
          .media = &media,
      },
      {
          .row_index = 1,
          .token_count = IREE_ARRAYSIZE(tokens),
          .token_ids = tokens,
          .flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
      },
  };
  IREE_ASSERT_OK(
      loom_serve_text_model_epoch(model_, 1, IREE_ARRAYSIZE(first), first));
  const loom_serve_text_span_t second = {
      .row_index = 0,
      .token_count = 2,
      .token_ids = tokens + 2,
      .flags = LOOM_SERVE_TEXT_SPAN_FLAG_SELECT,
      .input_offset = 2,
      .media = &media,
  };
  IREE_ASSERT_OK(loom_serve_text_model_epoch(model_, 1, 1, &second));

  Oracle projected_oracle;
  projected_oracle.Consume(tokens[0]);
  projected_oracle.Consume(projected[0]);
  projected_oracle.Consume(projected[1]);
  projected_oracle.Consume(tokens[3]);
  Oracle placeholder_oracle;
  for (int32_t token : tokens) {
    placeholder_oracle.Consume(token);
  }
  auto* projected_row = loom_serve_text_model_row(model_, 0);
  auto* placeholder_row = loom_serve_text_model_row(model_, 1);
  EXPECT_EQ(loom_serve_text_row_token(projected_row),
            projected_oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(placeholder_row),
            placeholder_oracle.Predict());

  const int32_t projected_prediction = loom_serve_text_row_token(projected_row);
  const int32_t placeholder_prediction =
      loom_serve_text_row_token(placeholder_row);
  const loom_serve_text_span_t follow_up[] = {
      {0, 1, &projected_prediction, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {1, 1, &placeholder_prediction, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  IREE_ASSERT_OK(loom_serve_text_model_epoch(
      model_, 1, IREE_ARRAYSIZE(follow_up), follow_up));
  projected_oracle.Consume(projected_prediction);
  placeholder_oracle.Consume(placeholder_prediction);
  EXPECT_EQ(loom_serve_text_row_token(projected_row),
            projected_oracle.Predict());
  EXPECT_EQ(loom_serve_text_row_token(placeholder_row),
            placeholder_oracle.Predict());

  IREE_ASSERT_OK(loom_serve_text_row_reset(projected_row));
  IREE_ASSERT_OK(loom_serve_text_row_reset(placeholder_row));
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
}

}  // namespace
