// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/segmenter/whitespace.h"
#include "iree/tokenizer/tokenizer.h"
#include "iree/tokenizer/tokenizer_test_util.h"

namespace iree::tokenizer {
namespace {

class TokenizerBufferTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* test = static_cast<TokenizerBufferTest*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++test->allocations_;
      if (test->fail_allocations_) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
    }
    auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, params, inout_pointer);
  }

  void SetUp() override {
    testing::ScopedVocabBuilder vocabulary;
    vocabulary.AddToken(0, "a");
    vocabulary.AddToken(1, "b");
    vocabulary.AddToken(2, "aa");
    vocabulary.AddToken(3, "ab");
    vocabulary.AddMerge(0, 0);
    vocabulary.AddMerge(0, 1);
    auto vocab = vocabulary.Build();
    iree_allocator_t allocator = {this, Allocate};
    iree_tokenizer_builder_t builder;
    iree_tokenizer_builder_initialize(allocator, &builder);
    iree_tokenizer_model_t* model = nullptr;
    IREE_CHECK_OK(iree_tokenizer_bpe_model_allocate(
        vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, allocator, &model));
    iree_tokenizer_builder_set_model(&builder, model);
    iree_tokenizer_builder_set_vocab(&builder, vocab.release());
    iree_tokenizer_segmenter_t* segmenter = nullptr;
    IREE_CHECK_OK(
        iree_tokenizer_segmenter_whitespace_allocate(allocator, &segmenter));
    iree_tokenizer_builder_set_segmenter(&builder, segmenter);
    IREE_CHECK_OK(iree_tokenizer_builder_build(&builder, &tokenizer_));
    iree_tokenizer_builder_deinitialize(&builder);
    size_t size = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &size));
    storage_.resize(size);
    IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
        tokenizer_, iree_make_byte_span(storage_.data(), storage_.size()),
        iree_make_byte_span(ring_, sizeof(ring_)),
        iree_tokenizer_offset_run_list_empty(), IREE_TOKENIZER_ENCODE_FLAG_NONE,
        &state_));
  }

  void TearDown() override {
    iree_tokenizer_encode_state_deinitialize(state_);
    iree_tokenizer_free(tokenizer_);
  }

  void Encode(const std::string& input, const std::vector<int32_t>& expected) {
    std::vector<int32_t> actual;
    size_t position = 0;
    while (position < input.size()) {
      size_t consumed = 0, written = 0;
      int32_t token;
      IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
          state_,
          iree_make_string_view(input.data() + position,
                                std::min(size_t{13}, input.size() - position)),
          iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &consumed, &written));
      ASSERT_GT(consumed + written, 0u);
      position += consumed;
      if (written) {
        actual.push_back(token);
      }
    }
    do {
      size_t written = 0;
      int32_t token;
      IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
          state_, iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &written));
      if (written) {
        actual.push_back(token);
      }
      ASSERT_TRUE(written || !iree_tokenizer_encode_state_has_pending(state_));
    } while (iree_tokenizer_encode_state_has_pending(state_));
    EXPECT_EQ(actual, expected);
  }

  // Calls into the allocator after model construction, including growth.
  size_t allocations_ = 0;
  // Injects failure only at the external allocation boundary.
  bool fail_allocations_ = false;
  // Owned immutable tokenizer, retired after its streaming state.
  iree_tokenizer_t* tokenizer_ = nullptr;
  // Caller-owned state bytes.
  std::vector<uint8_t> storage_;
  // Deliberately small caller ring to exercise real wrap and growth.
  uint8_t ring_[16];
  // Streaming state in storage_, owning any overflow buffers.
  iree_tokenizer_encode_state_t* state_ = nullptr;
};

TEST_F(TokenizerBufferTest, OrdinaryWordsDoNotAllocate) {
  fail_allocations_ = true;
  Encode("ab ab ab", {3, 3, 3});
  IREE_ASSERT_OK(iree_tokenizer_encode_state_reset(state_, 0));
  Encode("a b ab", {0, 1, 3});
}

TEST_F(TokenizerBufferTest, GrowthPreservesLaterBoundariesAndIsReused) {
  std::string input = "ab " + std::string(5000, 'a') + " b ab";
  std::vector<int32_t> expected = {3};
  expected.insert(expected.end(), 2500, 2);
  expected.insert(expected.end(), {1, 3});
  Encode(input, expected);
  size_t allocations = allocations_;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_reset(state_, 0));
  fail_allocations_ = true;
  Encode(input, expected);
  EXPECT_EQ(allocations_, allocations);
}

TEST_F(TokenizerBufferTest, GrowthFailurePreservesAcceptedInput) {
  std::string input = std::string(64, 'a') + " b";
  fail_allocations_ = true;
  int32_t tokens[64];
  size_t consumed = 0, written = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_tokenizer_encode_state_feed(
          state_, iree_make_string_view(input.data(), input.size()),
          iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 64),
          &consumed, &written));
  EXPECT_EQ(written, 0u);
  ASSERT_GT(consumed, 0u);
  EXPECT_TRUE(iree_tokenizer_encode_state_has_pending(state_));
  fail_allocations_ = false;
  std::vector<int32_t> expected(32, 2);
  expected.push_back(1);
  Encode(input.substr(consumed), expected);
}

TEST_F(TokenizerBufferTest, UnresolvedInputHasAnExplicitCeiling) {
  std::string input(IREE_TOKENIZER_TRANSFORM_BUFFER_OVERFLOW_MAX_SIZE / 2 + 1,
                    'a');
  int32_t token;
  size_t consumed = 0, written = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_tokenizer_encode_state_feed(
          state_, iree_make_string_view(input.data(), input.size()),
          iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &consumed, &written));
  EXPECT_EQ(written, 0u);
  EXPECT_EQ(consumed, IREE_TOKENIZER_TRANSFORM_BUFFER_OVERFLOW_MAX_SIZE / 2);
  EXPECT_TRUE(iree_tokenizer_encode_state_has_pending(state_));
}

}  // namespace
}  // namespace iree::tokenizer
