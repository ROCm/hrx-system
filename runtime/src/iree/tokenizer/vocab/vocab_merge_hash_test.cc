// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab/vocab_merge_hash.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

namespace {

class VocabMergeHashTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_allocate(
        0, iree_allocator_system(), &builder_));
  }

  void TearDown() override {
    iree_tokenizer_vocab_merge_hash_free(hash_);
    iree_tokenizer_vocab_free(vocab_);
    iree_tokenizer_vocab_builder_free(builder_);
  }

  // Build the vocab and merge hash from current builder state.
  void BuildVocabAndHash() {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
    builder_ = nullptr;  // Consumed by build.
    IREE_ASSERT_OK(iree_tokenizer_vocab_merge_hash_build(
        vocab_, iree_allocator_system(), &hash_));
  }

  iree_tokenizer_vocab_builder_t* builder_ = nullptr;
  iree_tokenizer_vocab_t* vocab_ = nullptr;
  iree_tokenizer_vocab_merge_hash_t* hash_ = nullptr;
};

TEST_F(VocabMergeHashTest, EmptyVocab) {
  BuildVocabAndHash();
  ASSERT_NE(hash_, nullptr);

  // No merges, so any lookup should return invalid.
  auto result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 0, 1);
  EXPECT_FALSE(iree_tokenizer_merge_hash_result_is_valid(result));
}

TEST_F(VocabMergeHashTest, SingleMerge) {
  // Add base tokens: a=0, b=1, ab=2
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("ab"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Add merge: (a, b) -> ab at rank 0.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 1));

  BuildVocabAndHash();

  // Lookup merge (a=0, b=1) should return rank=0, result_id=2.
  auto result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 0, 1);
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(result));
  EXPECT_EQ(result.rank, 0u);
  EXPECT_EQ(result.result_id, 2);

  // Lookup non-existent merge.
  result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 1, 0);
  EXPECT_FALSE(iree_tokenizer_merge_hash_result_is_valid(result));
}

TEST_F(VocabMergeHashTest, MergeOrderPreserved) {
  // Counter-example proving greedy is wrong:
  // Vocab: a=0, b=1, c=2, ab=3, bc=4
  // Merges: (b,c)->bc rank 0, (a,b)->ab rank 1
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("c"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("ab"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("bc"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Add merges in rank order: (b,c)->bc is higher priority (rank 0).
  // (b,c)
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 1, 2));
  // (a,b)
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 1));

  BuildVocabAndHash();

  // (b,c) should have rank 0, result bc=4.
  auto bc_result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 1, 2);
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(bc_result));
  EXPECT_EQ(bc_result.rank, 0u);
  EXPECT_EQ(bc_result.result_id, 4);

  // (a,b) should have rank 1, result ab=3.
  auto ab_result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 0, 1);
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(ab_result));
  EXPECT_EQ(ab_result.rank, 1u);
  EXPECT_EQ(ab_result.result_id, 3);

  // The key insight: bc_result.rank < ab_result.rank, so when encoding "abc"
  // the (b,c)->bc merge should be applied first, giving [a, bc] not [ab, c].
  EXPECT_LT(bc_result.rank, ab_result.rank);
}

TEST_F(VocabMergeHashTest, MultipleMerges) {
  // Tokens: a=0, b=1, c=2, d=3, ab=4, cd=5, abcd=6
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("c"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("d"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("ab"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("cd"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("abcd"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));

  // Merges in rank order.
  // (a,b)->ab rank 0
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 1));
  // (c,d)->cd rank 1
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 2, 3));
  // (ab,cd)->abcd rank 2
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 4, 5));

  BuildVocabAndHash();

  auto r0 = iree_tokenizer_vocab_merge_hash_lookup(hash_, 0, 1);
  auto r1 = iree_tokenizer_vocab_merge_hash_lookup(hash_, 2, 3);
  auto r2 = iree_tokenizer_vocab_merge_hash_lookup(hash_, 4, 5);

  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(r0));
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(r1));
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(r2));

  EXPECT_EQ(r0.rank, 0u);
  EXPECT_EQ(r0.result_id, 4);

  EXPECT_EQ(r1.rank, 1u);
  EXPECT_EQ(r1.result_id, 5);

  EXPECT_EQ(r2.rank, 2u);
  EXPECT_EQ(r2.result_id, 6);
}

TEST_F(VocabMergeHashTest, LongMergeResult) {
  const std::string left(300, 'a');
  const std::string right(300, 'b');
  const std::string result_text = left + right;
  for (const std::string& text : {left, right, result_text}) {
    IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
        builder_, iree_make_string_view(text.data(), text.size()), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE));
  }
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 1));
  BuildVocabAndHash();

  auto result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 0, 1);
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(result));
  EXPECT_EQ(result.rank, 0u);
  EXPECT_EQ(result.result_id, 2);
}

TEST_F(VocabMergeHashTest, RepeatedOperandAndRule) {
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("aa"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 0));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 0));
  BuildVocabAndHash();

  auto result = iree_tokenizer_vocab_merge_hash_lookup(hash_, 0, 0);
  ASSERT_TRUE(iree_tokenizer_merge_hash_result_is_valid(result));
  EXPECT_EQ(result.rank, 0u);
  EXPECT_EQ(result.result_id, 1);
}

TEST_F(VocabMergeHashTest, MissingMergeResult) {
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("b"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  // Ensure the absent result fits in the vocabulary's maximum token length.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("other"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 1));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
  builder_ = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_vocab_merge_hash_build(
                            vocab_, iree_allocator_system(), &hash_));
  EXPECT_EQ(hash_, nullptr);
}

TEST_F(VocabMergeHashTest, MergeResultLongerThanVocabulary) {
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token(
      builder_, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_merge(builder_, 0, 0));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
  builder_ = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_vocab_merge_hash_build(
                            vocab_, iree_allocator_system(), &hash_));
  EXPECT_EQ(hash_, nullptr);
}

class InvalidMergeOperandTest : public VocabMergeHashTest,
                                public ::testing::WithParamInterface<uint32_t> {
};

TEST_P(InvalidMergeOperandTest, RejectsInvalidOperand) {
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder_, 0, IREE_SV("a"), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  // ID 1 is an unused sparse slot; ID 2 is a real but empty token.
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_add_token_with_id(
      builder_, 2, IREE_SV(""), 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE));
  IREE_ASSERT_OK(
      iree_tokenizer_vocab_builder_add_merge(builder_, 0, GetParam()));
  IREE_ASSERT_OK(iree_tokenizer_vocab_builder_build(builder_, &vocab_));
  builder_ = nullptr;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_vocab_merge_hash_build(
                            vocab_, iree_allocator_system(), &hash_));
  EXPECT_EQ(hash_, nullptr);
}

INSTANTIATE_TEST_SUITE_P(VocabMergeHash, InvalidMergeOperandTest,
                         ::testing::Values(1u, 2u, 3u, UINT32_MAX));

}  // namespace
