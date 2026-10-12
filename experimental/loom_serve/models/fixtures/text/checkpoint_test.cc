// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/fixtures/text/service_fixture.h"

namespace {

using loom_serve::testing::text::Oracle;
using loom_serve::testing::text::SourceTextServiceTest;

TEST_P(SourceTextServiceTest,
       ActivePrefixPinsPreserveReservationAndIndependentContinuation) {
  for (size_t length : {3u, 4u, 5u, 8u}) {
    SCOPED_TRACE(length);
    auto* source = loom_serve_text_model_row(model_, 0);
    auto* branch = loom_serve_text_model_row(model_, 1);
    bool admitted = false;
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        source, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 64, &admitted));
    ASSERT_TRUE(admitted);
    std::vector<int32_t> tokens(length);
    Oracle prefix;
    for (size_t i = 0; i < length; ++i) {
      tokens[i] = i % 3;
      prefix.Consume(tokens[i]);
    }
    // An intermediate prompt chunk has no selected output token. Its consumed
    // frontier, not a retokenized response, is the shared prefix identity.
    const loom_serve_text_span_t span = {0, length, tokens.data(), 0};
    IREE_ASSERT_OK(loom_serve_text_model_epoch(model_, 1, 1, &span));
    const auto before = loom_serve_text_model_pool_usage(model_);
    loom_serve_text_checkpoint_t* endpoint = nullptr;
    IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &endpoint));
    ASSERT_NE(endpoint, nullptr);
    const auto pinned = loom_serve_text_model_pool_usage(model_);
    EXPECT_EQ(pinned.available + (length % 4 ? 4 : 0), before.available);
    EXPECT_EQ(pinned.pending, before.pending);

    // A second pin before the reserved writer advances must capture the
    // authoritative reader, not the still-uninitialized writer.
    loom_serve_text_checkpoint_t* alias = nullptr;
    IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &alias));
    ASSERT_NE(alias, nullptr);
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        branch, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, alias, 32, &admitted));
    ASSERT_TRUE(admitted);
    loom_serve_text_checkpoint_release(alias);

    Oracle source_oracle = prefix;
    Oracle branch_oracle = prefix;
    const int32_t source_token = 0;
    const int32_t branch_token = 2;
    const loom_serve_text_span_t branches[] = {
        {0, 1, &source_token, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
        {1, 1, &branch_token, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
    };
    IREE_ASSERT_OK(loom_serve_text_model_epoch(
        model_, 1, IREE_ARRAYSIZE(branches), branches));
    source_oracle.Consume(source_token);
    branch_oracle.Consume(branch_token);
    EXPECT_EQ(loom_serve_text_row_token(source), source_oracle.Predict());
    EXPECT_EQ(loom_serve_text_row_token(branch), branch_oracle.Predict());

    // Releasing the producing request returns unused completion pages. The
    // endpoint owns only its consumed pages, even after its source disappears.
    IREE_ASSERT_OK(loom_serve_text_row_reset(source));
    IREE_ASSERT_OK(loom_serve_text_row_reset(branch));
    const size_t prefix_pages = (length + 3) / 4 * 4;
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available,
              256u - prefix_pages);
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, 0u);
    IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
        branch, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, endpoint, 32, &admitted));
    ASSERT_TRUE(admitted);
    IREE_ASSERT_OK(loom_serve_text_row_prefill(branch, 1, &branch_token));
    EXPECT_EQ(loom_serve_text_row_token(branch), branch_oracle.Predict());
    loom_serve_text_checkpoint_release(endpoint);
    IREE_ASSERT_OK(loom_serve_text_row_reset(branch));
    EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 256u);
  }
}

TEST_P(SourceTextServiceTest, ActivePrefixPinDenialPreservesFullPoolRequest) {
  IREE_ASSERT_OK(loom_serve_text_model_destroy(model_));
  model_ = nullptr;
  auto options = ModelOptions();
  options.pool_capacity = 16;
  IREE_ASSERT_OK(
      loom_serve_text_model_create(device_, &options, &model_, allocator_));
  auto* source = loom_serve_text_model_row(model_, 0);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_text_row_try_reserve(
      source, LOOM_SERVE_TEXT_RESERVE_FRESH, nullptr, 16, &admitted));
  ASSERT_TRUE(admitted);
  const int32_t tokens[] = {0, 1, 2};
  Oracle oracle;
  for (auto token : tokens) {
    oracle.Consume(token);
  }
  const loom_serve_text_span_t first = {0, IREE_ARRAYSIZE(tokens), tokens, 0};
  IREE_ASSERT_OK(loom_serve_text_model_epoch(model_, 0, 1, &first));
  const auto before = loom_serve_text_model_pool_usage(model_);
  EXPECT_EQ(before.available, 0u);
  loom_serve_text_checkpoint_t* denied = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &denied));
  EXPECT_EQ(denied, nullptr);
  EXPECT_EQ(loom_serve_text_row_position(source), 3u);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available,
            before.available);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).pending, before.pending);

  // Aligning the same request makes the pin possible without a free KV page.
  // The reserved suffix is reused instead of duplicating the full reservation.
  const int32_t token = 1;
  IREE_ASSERT_OK(loom_serve_text_row_prefill(source, 1, &token));
  oracle.Consume(token);
  EXPECT_EQ(loom_serve_text_row_token(source), oracle.Predict());
  loom_serve_text_checkpoint_t* endpoint = nullptr;
  IREE_ASSERT_OK(loom_serve_text_row_try_pin(source, &endpoint));
  ASSERT_NE(endpoint, nullptr);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 0u);
  for (size_t i = 4; i < 16; ++i) {
    IREE_ASSERT_OK(loom_serve_text_row_prefill(source, 1, &token));
    oracle.Consume(token);
    EXPECT_EQ(loom_serve_text_row_token(source), oracle.Predict());
  }
  EXPECT_EQ(loom_serve_text_row_position(source), 16u);
  IREE_ASSERT_OK(loom_serve_text_row_reset(source));
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 12u);
  loom_serve_text_checkpoint_release(endpoint);
  EXPECT_EQ(loom_serve_text_model_pool_usage(model_).available, 16u);
}

}  // namespace
