// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/model_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::EncodeAndFinalize;
using testing::EncodeResult;
using testing::EncodeWithOffsetsAndFinalize;
using testing::ScopedModel;
using testing::ScopedModelState;
using testing::ScopedVocab;
using testing::ScopedVocabBuilder;
using testing::TestEncode;
using testing::TestLimitedOutputCapacity;
using testing::TestMultipleEncodeCalls;

TEST(BPEEndOfWordSuffixTest, VocabContainingAngleBrackets) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  auto tokens = EncodeAndFinalize(model.get(), "cat</w>",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);
}

TEST(BPEEndOfWordSuffixTest, BasicMatch) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t</w>");
  builder.AddToken(4, "ca");
  builder.AddMerge(1, 2);
  builder.AddMerge(4, 3);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), /*flags=*/0, iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);
}

TEST(BPEEndOfWordSuffixTest, MissingFinalAtomIsNotReplacedByBareCharacter) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "dog</w>");
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(tokens[1], 2);
}

TEST(BPEEndOfWordSuffixTest, WithOffsets) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hi</w>");
  builder.AddToken(1, "h");
  builder.AddToken(2, "i</w>");
  builder.AddMerge(1, 2);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), /*flags=*/0, iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto result =
      EncodeWithOffsetsAndFinalize(model.get(), "hi",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 1u);
  EXPECT_EQ(result.tokens[0], 0);

  ASSERT_EQ(result.offsets.size(), 1u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
}

TEST(BPEEndOfWordSuffixTest, EmptySuffixIsNoOp) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_string_view_empty()));

  auto tokens = EncodeAndFinalize(model.get(), "cat", false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);
}

TEST(BPEEndOfWordSuffixTest, PartialSegmentNoSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  const char* text = "cat";
  iree_tokenizer_segment_t segment = {0, 3};
  iree_tokenizer_segment_list_t segments = {1, &segment,
                                            /*last_is_partial=*/true};
  iree_const_byte_span_t buffer =
      iree_make_const_byte_span((const uint8_t*)text, 3);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segments,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  EXPECT_EQ(token_count, 0u);
  EXPECT_EQ(segments_consumed, 0u);
}

TEST(BPEEndOfWordSuffixTest, CompleteSegmentGetsSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t</w>");
  builder.AddToken(4, "ca");
  builder.AddMerge(1, 2);
  builder.AddMerge(4, 3);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), /*flags=*/0, iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  const char* text = "cat";
  iree_tokenizer_segment_t segment = {0, 3};
  iree_tokenizer_segment_list_t segments = {1, &segment,
                                            /*last_is_partial=*/false};
  iree_const_byte_span_t buffer =
      iree_make_const_byte_span((const uint8_t*)text, 3);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segments,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  ASSERT_EQ(token_count, 1u);
  EXPECT_EQ(tokens[0], 0);
}

TEST(BPEEndOfWordSuffixTest, StreamingThenFinalize) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "hello</w>");
  builder.AddToken(1, "h");
  builder.AddToken(2, "e");
  builder.AddToken(3, "l");
  builder.AddToken(4, "o</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  const char* text1 = "hel";
  iree_tokenizer_segment_t segment1 = {0, 3};
  iree_tokenizer_segment_list_t segments1 = {1, &segment1,
                                             /*last_is_partial=*/true};
  iree_const_byte_span_t buffer1 =
      iree_make_const_byte_span((const uint8_t*)text1, 3);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer1, segments1,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  EXPECT_EQ(token_count, 0u);
  EXPECT_EQ(segments_consumed, 0u);
  const char* complete_text = "hello";
  iree_tokenizer_segment_t complete_segment = {0, 5};
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), iree_make_const_byte_span(complete_text, 5),
      iree_tokenizer_make_segment_list(&complete_segment, 1),
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr, 1),
      &segments_consumed, &token_count));
  ASSERT_EQ(token_count, 1u);
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(segments_consumed, 0u);
  EXPECT_TRUE(iree_tokenizer_model_state_has_pending(state.get()));
  IREE_ASSERT_OK(iree_tokenizer_model_state_finalize(
      state.get(),
      iree_tokenizer_make_token_output(tokens.data() + 1, nullptr, nullptr,
                                       tokens.size() - 1),
      &token_count));
  ASSERT_EQ(token_count, 4u);
  tokens.resize(5);
  EXPECT_EQ(tokens, (std::vector<int32_t>{1, 2, 3, 3, 4}));
  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()));
}

TEST(BPEEndOfWordSuffixTest, SuffixedSpellingWithoutMergeRules) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "cat</w>");
  builder.AddToken(1, "c");
  builder.AddToken(2, "a");
  builder.AddToken(3, "t");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  EXPECT_EQ(tokens, (std::vector<int32_t>{1, 2}));
}

TEST(BPEEndOfWordSuffixTest, SuffixAppliesBeforeFinalAtomMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "aaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);
  EXPECT_EQ(tokens[1], 1);
}

TEST(BPEEndOfWordSuffixTest, MissingFinalAtomCannotSuffixAnEarlierMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "aa");
  builder.AddToken(2, "aa</w>");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "aaaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(tokens[1], 0);
}

TEST(BPEEndOfWordSuffixTest, MissingFinalAtomIsOmitted) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "aa");

  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "aaaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(tokens[1], 0);
}

TEST(BPEEndOfWordSuffixTest, SplitMergeWhenRightComponentHasSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddToken(3, "b</w>");

  builder.AddMerge(0, 1);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "ab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 0);
  EXPECT_EQ(tokens[1], 3);
}

TEST(BPEEndOfWordSuffixTest, SplitMergeEnablesNewMergeWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "ab");
  builder.AddToken(3, "b</w>");
  builder.AddToken(4, "ab</w>");

  builder.AddMerge(0, 1);
  builder.AddMerge(0, 3);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "ab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 4);
}

TEST(BPEEndOfWordSuffixTest, RecursiveSplitForSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "b");
  builder.AddToken(2, "c");
  builder.AddToken(3, "bc");
  builder.AddToken(4, "abc");
  builder.AddToken(5, "c</w>");

  builder.AddMerge(1, 2);
  builder.AddMerge(0, 3);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "abc",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);
  EXPECT_EQ(tokens[1], 1);
  EXPECT_EQ(tokens[2], 5);
}

TEST(BPEEndOfWordSuffixTest, OffsetsExcludeVirtualSuffixBytes) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto result =
      EncodeWithOffsetsAndFinalize(model.get(), "aaa",
                                   /*expect_pending_after_encode=*/false);
  ASSERT_EQ(result.tokens.size(), 2u);
  EXPECT_EQ(result.tokens[0], 2);
  EXPECT_EQ(result.tokens[1], 1);

  ASSERT_EQ(result.offsets.size(), 2u);
  EXPECT_EQ(result.offsets[0].start, 0u);
  EXPECT_EQ(result.offsets[0].end, 2u);
  EXPECT_EQ(result.offsets[1].start, 2u);
  EXPECT_EQ(result.offsets[1].end, 3u);
}

TEST(BPEEndOfWordSuffixTest, SingleCharSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a_");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa_");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("_")));

  auto tokens = EncodeAndFinalize(model.get(), "aaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);
  EXPECT_EQ(tokens[1], 1);
}

TEST(BPEEndOfWordSuffixTest, LongerSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a[ENDWD]");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa[ENDWD]");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("[ENDWD]")));

  auto tokens = EncodeAndFinalize(model.get(), "aaa",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);
  EXPECT_EQ(tokens[1], 1);
}

TEST(BPEEndOfWordSuffixTest, OnlyLastTokenGetsSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "b");
  builder.AddToken(3, "b</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "aab",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], 0);
  EXPECT_EQ(tokens[1], 0);
  EXPECT_EQ(tokens[2], 3);
}

TEST(BPEEndOfWordSuffixTest, SingleByteInput) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "x");
  builder.AddToken(1, "x</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "x",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 1);
}

TEST(BPEEndOfWordSuffixTest, LongerSequenceMultipleMerges) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "b");
  builder.AddToken(1, "b</w>");
  builder.AddToken(2, "bb");
  builder.AddToken(3, "bb</w>");
  builder.AddToken(4, "bbb");
  builder.AddToken(5, "bbb</w>");
  builder.AddMerge(0, 0);
  builder.AddMerge(2, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "bbbb",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 4);
  EXPECT_EQ(tokens[1], 1);
}

TEST(BPEEndOfWordSuffixTest, PartialSegmentNoSuffixWithBacktracking) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  ScopedModelState state(model.get());

  const char* text = "aa";
  iree_tokenizer_segment_t segment = {0, 2};
  iree_tokenizer_segment_list_t segments = {1, &segment,
                                            /*last_is_partial=*/true};
  iree_const_byte_span_t buffer =
      iree_make_const_byte_span((const uint8_t*)text, 2);

  std::vector<iree_tokenizer_token_id_t> tokens(16);
  iree_host_size_t segments_consumed = 0, token_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer, segments,
      iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                       tokens.size()),
      &segments_consumed, &token_count));

  bool found_suffixed = false;
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    if (tokens[i] == 1 || tokens[i] == 3) {
      found_suffixed = true;
    }
  }
  EXPECT_FALSE(found_suffixed)
      << "Partial segment should NOT match suffixed tokens";
}

TEST(BPEEndOfWordSuffixTest, SuffixedWordDoesNotBypassMergeRules) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "c");
  builder.AddToken(1, "a");
  builder.AddToken(2, "t");
  builder.AddToken(3, "ca");
  builder.AddToken(4, "cat");
  builder.AddToken(5, "cat</w>");
  builder.AddMerge(0, 1);
  builder.AddMerge(3, 2);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "cat",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 3);
}

TEST(BPEEndOfWordSuffixTest, LongSegmentGetsSuffixApplied) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");
  builder.AddToken(2, "aa");
  builder.AddToken(3, "aa</w>");
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  std::string input;
  input.reserve(2100);
  for (int i = 0; i < 1050; ++i) {
    input += "aa";
  }
  ASSERT_EQ(input.size(), 2100u);

  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 1051u);
  for (size_t i = 0; i < 1049; ++i) {
    EXPECT_EQ(tokens[i], 2) << "Token " << i << " should be 'aa'";
  }
  EXPECT_EQ(tokens[1049], 0);
  EXPECT_EQ(tokens[1050], 1);
}

TEST(BPEEndOfWordSuffixTest, LongSegmentSuffixEnablesMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "x");
  builder.AddToken(1, "x</w>");
  builder.AddToken(2, "xx");
  builder.AddToken(3, "xx</w>");
  builder.AddToken(4, "xxx</w>");
  builder.AddMerge(0, 0);
  builder.AddMerge(2, 1);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  std::string input(2103, 'x');
  ASSERT_EQ(input.size(), 2103u);

  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 1051u);
  for (size_t i = 0; i < 1050; ++i) {
    EXPECT_EQ(tokens[i], 2) << "Token " << i << " should be 'xx'";
  }
  EXPECT_EQ(tokens[1050], 4) << "Last token should be 'xxx</w>' (suffix merge)";
}

TEST(BPEByteLevelSuffixTest, NonAsciiBytesWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "c");
  builder.AddToken(1, "a");
  builder.AddToken(2, "f");

  builder.AddToken(3, "Ã");
  builder.AddToken(4, "©");
  builder.AddToken(5, "cafÃ©");
  builder.AddToken(6, "cafÃ©</w>");
  builder.AddToken(7, "ca");
  builder.AddToken(8, "caf");
  builder.AddToken(9, "cafÃ");
  builder.AddToken(10, "©</w>");
  builder.AddMerge(7, 2);
  builder.AddMerge(8, 3);
  builder.AddMerge(9, 10);
  builder.AddMerge(0, 1);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "café",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 6);
}

TEST(BPEByteLevelSuffixTest, MissingFinalByteLevelAtom) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "c");
  builder.AddToken(1, "a");
  builder.AddToken(2, "f");
  builder.AddToken(3, "Ã");
  builder.AddToken(4, "©");
  builder.AddToken(5, "ca");
  builder.AddToken(6, "caf");
  builder.AddToken(7, "cafÃ");
  builder.AddToken(8, "cafÃ©");
  builder.AddToken(9, "cafÃ©</w>");

  builder.AddMerge(0, 1);
  builder.AddMerge(5, 2);
  builder.AddMerge(6, 3);
  builder.AddMerge(7, 4);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "café",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 7);
}

TEST(BPEByteLevelSuffixTest, RawByteFallbackDoesNotMatchEncodedSymbols) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "<0xE1>");
  builder.AddToken(1, "<0x9E>");
  builder.AddToken(2, "<0x9F>");
  builder.AddToken(3, "</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);

  std::string khmer = "ស";
  auto tokens = EncodeAndFinalize(model.get(), khmer,
                                  /*expect_pending_after_encode=*/false);
  EXPECT_TRUE(tokens.empty());
}

TEST(BPESuffixBlockedPairValidationTest, RepeatedTokenWithSuffixMerge) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "A");
  builder.AddToken(1, "AA");
  builder.AddToken(2, "A</w>");
  builder.AddToken(3, "AA</w>");
  builder.AddToken(4, "AAA</w>");

  builder.AddMerge(0, 3);
  builder.AddMerge(0, 2);
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "AAAA",
                                  /*expect_pending_after_encode=*/false);

  EXPECT_EQ(tokens, (std::vector<int32_t>{0, 4}));
}

TEST(BPESuffixBlockedPairValidationTest, LongRepeatedPattern) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "X");
  builder.AddToken(1, "XX");
  builder.AddToken(2, "X</w>");
  builder.AddToken(3, "XX</w>");
  builder.AddToken(4, "XXX</w>");

  builder.AddMerge(0, 3);
  builder.AddMerge(0, 2);
  builder.AddMerge(0, 0);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "XXXXXXXXXX",
                                  /*expect_pending_after_encode=*/false);

  EXPECT_EQ(tokens, (std::vector<int32_t>{1, 1, 1, 0, 4}));
}

TEST(BPESuffixBlockedPairValidationTest, MissingFinalAtomAfterOtherMergeRules) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "A");
  builder.AddToken(1, "B");
  builder.AddToken(2, "AB");
  builder.AddToken(3, "AA");
  builder.AddToken(4, "A</w>");
  builder.AddToken(5, "AA</w>");

  builder.AddMerge(0, 4);
  builder.AddMerge(0, 0);
  builder.AddMerge(0, 1);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "AB",
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 0);
}

TEST(BPEByteLevelSuffixFastPathTest, AsciiSegmentWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "a");
  builder.AddToken(1, "a</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "a",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 1);
}

TEST(BPEByteLevelSuffixFastPathTest, NonAsciiByteWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ģ");
  builder.AddToken(1, "Ģ</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  std::string input(1, '\x80');
  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected single token 'Ģ</w>'";
  EXPECT_EQ(tokens[0], 1);
}

TEST(BPEByteLevelSuffixFastPathTest, SpaceByteWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ġ");
  builder.AddToken(1, "Ġ</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;

  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), " ",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected single token 'Ġ</w>'";
  EXPECT_EQ(tokens[0], 1);
}

TEST(BPEByteLevelSuffixFastPathTest, MultiByteAsciiWordWithSuffix) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "h");
  builder.AddToken(1, "i");
  builder.AddToken(2, "hi");
  builder.AddToken(3, "hi</w>");
  builder.AddToken(4, "h</w>");
  builder.AddToken(5, "i</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(),
      IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT |
          IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "hi",
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], 2);
}

TEST(BPEByteLevelSuffixFastPathTest, IgnoreMergesWithNonAscii) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ģ");
  builder.AddToken(1, "ģ");
  builder.AddToken(2, "Ģģ");
  builder.AddToken(3, "Ģ</w>");
  builder.AddToken(4, "ģ</w>");
  builder.AddToken(5, "Ģģ</w>");
  builder.AddMerge(0, 1);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(),
      IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT |
          IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  std::string input = "\x80\x81";
  auto tokens = EncodeAndFinalize(model.get(), input,
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 1u)
      << "IGNORE_MERGES does vocab lookup, expected [Ģģ]";
  EXPECT_EQ(tokens[0], 2);
}

TEST(BPEByteLevelSuffixFastPathTest, WholeSegmentSuffixMatchNonAscii) {
  ScopedVocabBuilder builder;
  builder.AddToken(0, "Ċ");
  builder.AddToken(1, "Ċ</w>");

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT,
      iree_allocator_system(), &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "\n",
                                  /*expect_pending_after_encode=*/false);
  ASSERT_EQ(tokens.size(), 1u) << "Expected single token 'Ċ</w>'";
  EXPECT_EQ(tokens[0], 1);
}

TEST(BPEEndOfWordSuffixTest, SuffixBlockingMustCheckPairValidity) {
  ScopedVocabBuilder builder;

  builder.AddToken(0, "l");
  builder.AddToken(1, "i");
  builder.AddToken(2, "s");
  builder.AddToken(3, "h");
  builder.AddToken(4, "m");
  builder.AddToken(5, "e");
  builder.AddToken(6, "n");
  builder.AddToken(7, "t");

  builder.AddToken(8, "li");
  builder.AddToken(9, "sh");
  builder.AddToken(10, "is");
  builder.AddToken(11, "en");

  builder.AddToken(12, "lish");
  builder.AddToken(13, "ent");
  builder.AddToken(14, "ment");

  builder.AddToken(15, "t</w>");
  builder.AddToken(16, "ent</w>");
  builder.AddToken(17, "ment</w>");
  builder.AddToken(18, "shment</w>");

  builder.AddMerge(0, 1);
  builder.AddMerge(2, 3);
  builder.AddMerge(1, 2);
  builder.AddMerge(5, 6);
  builder.AddMerge(11, 15);
  builder.AddMerge(4, 16);
  builder.AddMerge(9, 17);
  builder.AddMerge(8, 9);

  auto vocab = ScopedVocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
      vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
      &raw_model));
  auto model = ScopedModel(raw_model);
  IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
      model.get(), iree_make_cstring_view("</w>")));

  auto tokens = EncodeAndFinalize(model.get(), "lishment",
                                  /*expect_pending_after_encode=*/false);

  ASSERT_EQ(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 8) << "First token should be 'li'";
  EXPECT_EQ(tokens[1], 18) << "Second token should be 'shment</w>'";
}

}  // namespace
}  // namespace iree::tokenizer
