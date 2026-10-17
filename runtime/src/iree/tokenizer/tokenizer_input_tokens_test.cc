// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

namespace {

class TokenizerInputTokensTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static constexpr char kJson[] = R"({
      "normalizer":{"type":"Lowercase"},
      "added_tokens":[
        {"id":10,"content":"<R>","single_word":false,"lstrip":false,
         "rstrip":false,"normalized":false,"special":true},
        {"id":11,"content":"<S>","single_word":false,"lstrip":false,
         "rstrip":false,"normalized":false,"special":true},
        {"id":12,"content":"<n>","single_word":false,"lstrip":false,
         "rstrip":false,"normalized":true,"special":true},
        {"id":13,"content":"<t>","single_word":false,"lstrip":false,
         "rstrip":false,"normalized":true,"special":true}],
      "model":{"type":"BPE","vocab":{
        "a":1,"b":2,"c":3,"<":4,">":5,"q":6,"é":7,"中":8,"🙂":9},
        "merges":[]}})";
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        IREE_SV(kJson), iree_allocator_system(), &tokenizer_));
    iree_host_size_t state_size = 0;
    IREE_ASSERT_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));
    state_storage_.resize(state_size);
    transform_storage_.resize(IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE);
    IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
        tokenizer_,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        iree_make_byte_span(transform_storage_.data(),
                            transform_storage_.size()),
        iree_tokenizer_offset_run_list_empty(), IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_allocator_system(), &state_));
  }

  void TearDown() override {
    iree_tokenizer_encode_state_deinitialize(state_);
    iree_tokenizer_free(tokenizer_);
  }

  void Finalize(std::vector<int32_t>* tokens) {
    do {
      int32_t token = -1;
      iree_host_size_t count = 0;
      IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
          state_, iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &count));
      ASSERT_TRUE(count || !iree_tokenizer_encode_state_has_pending(state_));
      if (count) {
        tokens->push_back(token);
      }
    } while (iree_tokenizer_encode_state_has_pending(state_));
    EXPECT_FALSE(iree_tokenizer_encode_state_has_pending(state_));
  }

  // Immutable parsed tokenizer, owned until fixture teardown.
  iree_tokenizer_t* tokenizer_ = nullptr;
  // Stream state borrowing the storage below.
  iree_tokenizer_encode_state_t* state_ = nullptr;
  // Caller-owned pipeline state storage.
  std::vector<uint8_t> state_storage_;
  // Caller-owned normalization and segmentation ring storage.
  std::vector<uint8_t> transform_storage_;
};

TEST_F(TokenizerInputTokensTest, AcceptedMatchWaitsForOutputCredit) {
  for (const auto& item :
       {std::make_pair("<R>", 10), std::make_pair("<N>", 12)}) {
    SCOPED_TRACE(item.first);
    IREE_ASSERT_OK(iree_tokenizer_encode_state_reset(
        state_, IREE_TOKENIZER_ENCODE_FLAG_NONE));
    int32_t output = -1;
    iree_host_size_t consumed = 0;
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state_, iree_make_cstring_view(item.first),
        iree_tokenizer_make_token_output(&output, nullptr, nullptr, 0),
        &consumed, &count));
    EXPECT_EQ(consumed, 3u);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(output, -1);
    EXPECT_TRUE(iree_tokenizer_encode_state_has_pending(state_));

    // Finalization supplies the first output slot; no new input is necessary.
    std::vector<int32_t> tokens;
    Finalize(&tokens);
    EXPECT_EQ(tokens, (std::vector<int32_t>{item.second}));
  }
}

TEST_F(TokenizerInputTokensTest,
       PartitionedMatchesPreserveOrderUnderShortCredit) {
  struct TestCase {
    // Full input, partitioned at every possible byte boundary below.
    const char* input;
    // Exact expected model and extracted-token IDs in input order.
    std::vector<int32_t> expected;
  } cases[] = {
      {"<R>", {10}},
      {"<R><S>", {10, 11}},
      {"<S><R>", {11, 10}},
      {"<R><S><S><R>", {10, 11, 11, 10}},
      {"a<R>b<S>c", {1, 10, 2, 11, 3}},
      {"<Q><R>", {4, 6, 5, 10}},
      {"<R><", {10, 4}},
      {"<N>", {12}},
      {"<N><T>", {12, 13}},
      {"<T><N>", {13, 12}},
      {"a<N>b<T>c", {1, 12, 2, 13, 3}},
      {"<N><R><T><S>", {12, 10, 13, 11}},
      {"É", {7}},
      {"中", {8}},
      {"🙂", {9}},
      {"aÉ<N>中<R>🙂b", {1, 7, 12, 8, 10, 9, 2}},
  };
  for (const auto& item : cases) {
    std::string input = item.input;
    SCOPED_TRACE(input);
    for (size_t split = 0; split <= input.size(); ++split) {
      SCOPED_TRACE(split);
      IREE_ASSERT_OK(iree_tokenizer_encode_state_reset(
          state_, IREE_TOKENIZER_ENCODE_FLAG_NONE));
      std::vector<int32_t> tokens;
      size_t position = 0;
      for (size_t end : {split, input.size()}) {
        size_t capacity = 0;
        bool made_progress = true;
        while (made_progress) {
          int32_t output = 0;
          iree_host_size_t consumed = 0;
          iree_host_size_t count = 0;
          IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
              state_,
              iree_make_string_view(input.data() + position, end - position),
              iree_tokenizer_make_token_output(&output, nullptr, nullptr,
                                               capacity),
              &consumed, &count));
          position += consumed;
          if (count > 0) {
            tokens.push_back(output);
          }
          made_progress = consumed > 0 || count > 0 || capacity == 0;
          capacity = 1;
        }
      }
      EXPECT_EQ(position, input.size());
      Finalize(&tokens);
      EXPECT_EQ(tokens, item.expected);
    }
  }
}

TEST_F(TokenizerInputTokensTest, CompleteInputRejectsTruncatedUTF8) {
  std::vector<uint8_t> batch_state(state_storage_.size());
  std::vector<uint8_t> batch_transform(transform_storage_.size());
  for (const std::string codepoint : {"é", "中", "🙂"}) {
    for (size_t length = 1; length < codepoint.size(); ++length) {
      SCOPED_TRACE(codepoint);
      SCOPED_TRACE(length);
      int32_t output[16] = {};
      iree_tokenizer_token_output_t token_output =
          iree_tokenizer_make_token_output(output, nullptr, nullptr,
                                           IREE_ARRAYSIZE(output));
      iree_string_view_t input =
          iree_make_string_view(codepoint.data(), length);
      iree_host_size_t count = 0;
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_tokenizer_encode(tokenizer_, input,
                                IREE_TOKENIZER_ENCODE_FLAG_NONE, token_output,
                                iree_allocator_system(), &count));
      iree_tokenizer_encode_batch_item_t item = {
          .text = input,
          .output = token_output,
      };
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          iree_tokenizer_encode_batch(
              tokenizer_, &item, 1, IREE_TOKENIZER_ENCODE_FLAG_NONE,
              iree_make_byte_span(batch_state.data(), batch_state.size()),
              iree_make_byte_span(batch_transform.data(),
                                  batch_transform.size()),
              iree_tokenizer_offset_run_list_empty(), iree_allocator_system()));
    }
  }
}

}  // namespace
