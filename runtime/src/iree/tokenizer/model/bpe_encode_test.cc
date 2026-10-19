// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/model_test_util.h"

namespace iree::tokenizer {
namespace {

TEST(BPEEncodeTest, PartialSpanRemainsWithCallerUntilComplete) {
  testing::ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  testing::ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  testing::ScopedModel model(raw_model);
  testing::ScopedModelState state(model.get());

  // Even an unknown-only prefix cannot establish a complete segment. The
  // caller retains it until a true boundary arrives.
  const uint8_t input[] = {'?', 'a', 'b'};
  iree_tokenizer_segment_t segment = {0, 1};
  auto segments = iree_tokenizer_make_segment_list(&segment, 1);
  segments.last_is_partial = true;
  int32_t tokens[2];
  size_t consumed = 0, written = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), iree_make_const_byte_span(input, sizeof(input)), segments,
      iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 2), &consumed,
      &written));
  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(written, 0u);
  EXPECT_EQ(iree_tokenizer_model_state_reclaim(state.get()), 0u);

  // Complete the original span before supplying the next segment.
  segment = {0, 1};
  segments.last_is_partial = false;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), iree_make_const_byte_span(input, sizeof(input)), segments,
      iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 2), &consumed,
      &written));
  EXPECT_EQ(consumed, 1u);
  EXPECT_EQ(written, 0u);

  segment = {1, 3};
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), iree_make_const_byte_span(input, sizeof(input)), segments,
      iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 2), &consumed,
      &written));
  EXPECT_EQ(consumed, 1u);
  ASSERT_EQ(written, 1u);
  EXPECT_EQ(tokens[0], 2);
}

TEST(BPEEncodeTest, LaterPriorityChainChangesTheFirstToken) {
  testing::ScopedVocabBuilder builder;
  std::string input = "abcdefghijklmnopqrstuvwx";
  for (size_t i = 0; i < input.size(); ++i) {
    builder.AddToken(static_cast<int32_t>(i), input.substr(i, 1).c_str());
  }
  for (size_t i = 0; i + 1 < input.size(); ++i) {
    builder.AddToken(static_cast<int32_t>(input.size() + i),
                     input.substr(i, 2).c_str());
  }
  for (size_t i = input.size() - 1; i > 0; --i) {
    builder.AddMerge(static_cast<uint32_t>(i - 1), static_cast<uint32_t>(i));
  }
  testing::ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), 0, iree_allocator_system(), &raw_model));
  testing::ScopedModel model(raw_model);
  testing::ScopedModelState state(model.get());
  int32_t tokens[24];
  size_t consumed = 0, written = 0;
  iree_tokenizer_segment_t segment = {0, input.size() - 1};
  auto segments = iree_tokenizer_make_segment_list(&segment, 1);
  segments.last_is_partial = true;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), iree_make_const_byte_span(input.data(), input.size()),
      segments, iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 24),
      &consumed, &written));
  EXPECT_EQ(written, 0u);
  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(iree_tokenizer_model_state_reclaim(state.get()), 0u);
  // Even after eleven longest-token lengths, one more character changes the
  // first token from "a" to "ab" by reversing every adjacent priority choice.
  segment.end = input.size();
  segments.last_is_partial = false;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), iree_make_const_byte_span(input.data(), input.size()),
      segments, iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 24),
      &consumed, &written));
  ASSERT_EQ(written, 12u);
  for (size_t i = 0; i < written; ++i) {
    EXPECT_EQ(tokens[i], static_cast<int32_t>(24 + 2 * i));
  }
}

TEST(BPEEncodeTest, ExplicitSymbolsRespectScratchCeiling) {
  testing::ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  testing::ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), 0, iree_allocator_system(), &raw_model));
  testing::ScopedModel model(raw_model);
  testing::ScopedModelState state(model.get());
  std::string input(IREE_TOKENIZER_BPE_SCRATCH_MAX_SIZE, 'a');
  iree_tokenizer_segment_t segment = {0, input.size()};
  size_t consumed = 0, written = 0;
  int32_t token;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_tokenizer_model_state_encode(
          state.get(), iree_make_const_byte_span(input.data(), input.size()),
          iree_tokenizer_make_segment_list(&segment, 1),
          iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &consumed, &written));
  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(written, 0u);
}

TEST(BPEEncodeTest, CharacterAtomsDoNotSplitIntoRawByteTokens) {
  testing::ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "\xC3");
  builder.AddToken(2, "\xA9");
  builder.AddToken(3, "\xE2");
  builder.AddToken(4, "\x82");
  builder.AddToken(5, "\xAC");
  builder.AddToken(6, "\xF0");
  builder.AddToken(7, "\x9F");
  builder.AddToken(8, "\x98");
  builder.AddToken(9, "\x80");
  testing::ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  testing::ScopedModel model(raw_model);

  // These complete characters have no atom or byte-fallback spelling. Raw
  // invalid-byte tokens in the vocabulary do not change their atomization.
  testing::TestEncode(model.get(),
                      "a\xC3\xA9"
                      "a",
                      {0, 0}, false);
  testing::TestEncode(model.get(),
                      "a\xE2\x82\xAC"
                      "a",
                      {0, 0}, false);
  testing::TestEncode(model.get(),
                      "a\xF0\x9F\x98\x80"
                      "a",
                      {0, 0}, false);
  // An invalid sequence still uses individual byte atoms.
  testing::TestEncode(model.get(),
                      "a\xC3"
                      "a",
                      {0, 1, 0}, false);
}

TEST(BPEEncodeTest, BacktrackPrefixesEndAtAtomBoundaries) {
  testing::ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "\xC3\xA9");
  builder.AddToken(2, "b");
  builder.AddToken(3, "a\xC3\xA9");
  builder.AddToken(4,
                   "\xC3\xA9"
                   "b");
  builder.AddToken(5, "a\xC3");
  builder.AddToken(6, "\xC3");
  builder.AddToken(7, "\xA9");
  builder.AddMerge(1, 2);
  builder.AddMerge(0, 1);
  builder.AddMerge(0, 6);
  testing::ScopedVocab vocab(builder.Build());
  for (auto flags : {IREE_TOKENIZER_BPE_FLAG_NONE,
                     IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT}) {
    iree_tokenizer_model_t* raw_model = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
        vocab.get(), flags, iree_allocator_system(), &raw_model));
    testing::ScopedModel model(raw_model);
    // The longest initial match, aé, loses to the earlier é+b merge. Its
    // proper-prefix chain must jump to a, never the incomplete a+C3 spelling.
    testing::TestEncode(model.get(),
                        flags ? "a\xE9"
                                "b"
                              : "a\xC3\xA9"
                                "b",
                        {0, 4}, false);
  }
}

}  // namespace
}  // namespace iree::tokenizer
