// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/model_json.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/added_tokens_json.h"
#include "iree/tokenizer/format/huggingface/types.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/vocab/vocab.h"

namespace {

using ::iree::testing::status::StatusIs;

class ModelJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    allocator_ = iree_allocator_system();
    memset(&added_tokens_, 0, sizeof(added_tokens_));
  }

  void TearDown() override {
    if (model_) {
      iree_tokenizer_model_free(model_);
      model_ = nullptr;
    }
    if (vocab_) {
      iree_tokenizer_vocab_free(vocab_);
      vocab_ = nullptr;
    }
    iree_tokenizer_huggingface_added_tokens_free(&added_tokens_);
  }

  iree_allocator_t allocator_;
  iree_tokenizer_huggingface_added_tokens_t added_tokens_;
  iree_tokenizer_model_t* model_ = nullptr;
  iree_tokenizer_vocab_t* vocab_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Capacity Estimation Tests
//===----------------------------------------------------------------------===//

TEST_F(ModelJsonTest, EstimateVocabCapacitySmallFile) {
  // Small file should return minimum capacity.
  EXPECT_GE(iree_tokenizer_huggingface_estimate_vocab_capacity(100), 1000u);
}

TEST_F(ModelJsonTest, EstimateVocabCapacityLargeFile) {
  // Large file (1MB) should return proportional capacity.
  // Heuristic is json_size / 40 bytes per entry.
  iree_host_size_t capacity =
      iree_tokenizer_huggingface_estimate_vocab_capacity(1000000);
  EXPECT_GE(capacity, 20000u);  // 1M / 40 = 25k, but at least 20k.
}

//===----------------------------------------------------------------------===//
// BPE Model Parsing Tests
//===----------------------------------------------------------------------===//

TEST_F(ModelJsonTest, MinimalBPEModel) {
  // Minimal valid BPE model with vocab only.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2},
    "merges": []
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);

  // Verify vocab size.
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);

  // Verify tokens exist.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab_, iree_make_cstring_view("a")),
            0);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab_, iree_make_cstring_view("b")),
            1);
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab_, iree_make_cstring_view("c")),
            2);
}

TEST_F(ModelJsonTest, LongSpellingsSurviveScratchReuseAcrossModelFormats) {
  const std::string left_json = std::string(4096, 'a') + R"(\u00e9)";
  const std::string right_json = std::string(8193, 'b') + R"(\n)";
  const std::string left = std::string(4096, 'a') + "\xC3\xA9";
  const std::string right = std::string(8193, 'b') + "\n";
  const std::string joined = left + right;
  const std::string vocabulary = "\"vocab\":{\"" + left_json + "\":0,\"" +
                                 right_json + "\":1,\"" + left_json +
                                 right_json + "\":2}";
  const std::string unknown = ",\"unk_token\":\"" + left_json + "\"";
  for (const std::string& json :
       {"{\"type\":\"BPE\"," + vocabulary + unknown + ",\"merges\":[\"" +
            left_json + " " + right_json + "\"]}",
        "{\"type\":\"BPE\"," + vocabulary + unknown + ",\"merges\":[[\"" +
            left_json + "\",\"" + right_json + "\"]]}",
        "{\"type\":\"WordPiece\"," + vocabulary + unknown + "}",
        "{\"type\":\"Unigram\",\"unk_id\":0,\"vocab\":[[\"" + left_json +
            "\",-2],[\"" + right_json + "\",-3],[\"" + left_json + right_json +
            "\",-1]]}"}) {
    IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_model(
        iree_make_string_view(json.data(), json.size()), &added_tokens_,
        IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE, allocator_, &model_,
        &vocab_));
    EXPECT_EQ(iree_tokenizer_vocab_lookup(
                  vocab_, iree_make_string_view(left.data(), left.size())),
              0);
    EXPECT_EQ(iree_tokenizer_vocab_lookup(
                  vocab_, iree_make_string_view(right.data(), right.size())),
              1);
    EXPECT_EQ(iree_tokenizer_vocab_lookup(
                  vocab_, iree_make_string_view(joined.data(), joined.size())),
              2);
    EXPECT_EQ(iree_tokenizer_vocab_special_ids(vocab_).unk, 0);
    if (json.find("merges") != std::string::npos) {
      ASSERT_EQ(iree_tokenizer_vocab_merge_count(vocab_), 1u);
      const auto merge = iree_tokenizer_vocab_merge(vocab_, 0);
      EXPECT_EQ(merge.left_id, 0u);
      EXPECT_EQ(merge.right_id, 1u);
    } else {
      EXPECT_EQ(iree_tokenizer_vocab_merge_count(vocab_), 0u);
    }
    iree_tokenizer_model_free(model_);
    model_ = nullptr;
    iree_tokenizer_vocab_free(vocab_);
    vocab_ = nullptr;
  }
}

TEST_F(ModelJsonTest,
       SpellingGrowthPropagatesAllocationFailuresAndCancellation) {
  struct AllocationState {
    // Index of the allocation request to fail.
    size_t fail_at = 0;
    // Number of allocation requests observed.
    size_t attempts = 0;
    // Number of allocations that still need retirement.
    size_t live = 0;
    // External failure returned at the selected allocation request.
    iree_status_code_t failure = IREE_STATUS_RESOURCE_EXHAUSTED;
  };
  auto allocate = [](void* self, iree_allocator_command_t command,
                     const void* params,
                     void** inout_pointer) -> iree_status_t {
    auto& state = *static_cast<AllocationState*>(self);
    const bool had_storage = *inout_pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        state.attempts++ == state.fail_at) {
      return iree_status_from_code(state.failure);
    }
    auto allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, params, inout_pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        state.live -= had_storage;
      } else if (!had_storage) {
        ++state.live;
      }
    }
    return status;
  };
  const std::string spelling = std::string(4096, 'a') + R"(\u00e9)";
  for (const std::string& json :
       {"{\"type\":\"BPE\",\"vocab\":{\"" + spelling + "\":0,\"b\":1,\"" +
            spelling + "b\":2},\"merges\":[[\"" + spelling + "\",\"b\"]]}",
        "{\"type\":\"WordPiece\",\"vocab\":{\"" + spelling +
            "\":0},\"unk_token\":\"" + spelling + "\"}",
        "{\"type\":\"Unigram\",\"vocab\":[[\"" + spelling + "\",-1]]}"}) {
    for (iree_status_code_t failure :
         {IREE_STATUS_RESOURCE_EXHAUSTED, IREE_STATUS_CANCELLED}) {
      for (size_t fail_at = 0;; ++fail_at) {
        SCOPED_TRACE(fail_at);
        AllocationState state{.fail_at = fail_at, .failure = failure};
        iree_status_t status = iree_tokenizer_huggingface_parse_model(
            iree_make_string_view(json.data(), json.size()), &added_tokens_,
            IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE,
            iree_allocator_t{&state, allocate}, &model_, &vocab_);
        iree_tokenizer_model_free(model_);
        model_ = nullptr;
        iree_tokenizer_vocab_free(vocab_);
        vocab_ = nullptr;
        EXPECT_EQ(state.live, 0u);
        if (iree_status_is_ok(status)) {
          EXPECT_EQ(state.attempts, fail_at);
          break;
        }
        IREE_EXPECT_STATUS_IS(failure, status);
      }
    }
  }
}

TEST_F(ModelJsonTest, BPEModelWithMergesStringFormat) {
  // BPE model with merges in string format ("a b").
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": ["a b"]
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);
}

TEST_F(ModelJsonTest, BPEModelWithMergesTupleFormat) {
  // BPE model with merges in tuple format (["a", "b"]).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "ab": 2},
    "merges": [["a", "b"]]
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);
}

TEST_F(ModelJsonTest, BPEModelWithMultipleMerges) {
  // BPE model with multiple merges in order.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
    "merges": ["a b", "ab c"]
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 5u);
}

TEST_F(ModelJsonTest, BPEModelWithFlags) {
  // BPE model with all optional flags set.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": [],
    "byte_fallback": true,
    "fuse_unk": true,
    "ignore_merges": true
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, BPEModelContinuingSubwordPrefixRejected) {
  // Non-null continuing_subword_prefix must be rejected (we don't implement
  // prefix stripping during merge computation).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "##b": 1},
    "merges": [],
    "continuing_subword_prefix": "##"
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
}

TEST_F(ModelJsonTest, BPEModelEndOfWordSuffixAccepted) {
  // end_of_word_suffix is used by CLIP-style tokenizers to append a suffix
  // (like "</w>") to each word segment before vocab lookup.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b</w>": 1},
    "merges": [],
    "end_of_word_suffix": "</w>"
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, BPEModelNullAffixesAllowed) {
  // Null affixes are allowed (equivalent to not present).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1},
    "merges": [],
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, BPEModelWithDropoutIgnored) {
  // BPE model with dropout (training-only field, should be ignored).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": [],
    "dropout": 0.1
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, BPEModelWithNullDropout) {
  // BPE model with null dropout (common in real files).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": [],
    "dropout": null
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, BPEModelWithUnkToken) {
  // unk_token should be resolved to its vocab ID and set as special_ids.unk.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "<unk>": 1, "b": 2},
    "merges": [],
    "unk_token": "<unk>"
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab_);
  EXPECT_EQ(ids.unk, 1);
}

TEST_F(ModelJsonTest, BPEModelWithNullUnkToken) {
  // Null unk_token is allowed (no UNK fallback configured).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": [],
    "unk_token": null
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab_);
  EXPECT_EQ(ids.unk, -1);
}

TEST_F(ModelJsonTest, BPEModelUnkTokenNotInVocabError) {
  // unk_token that doesn't exist in the vocab should fail.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1},
    "merges": [],
    "unk_token": "<unk>"
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(ModelJsonTest, BPEModelWithEscapedTokens) {
  // BPE model with tokens that need JSON unescaping.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"hello\"world": 0, "tab\there": 1, "new\nline": 2},
    "merges": []
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_bpe_model(
      iree_make_cstring_view(json), &added_tokens_,
      IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);

  // Verify unescaped tokens exist.
  EXPECT_EQ(iree_tokenizer_vocab_lookup(vocab_,
                                        iree_make_cstring_view("hello\"world")),
            0);
  EXPECT_EQ(
      iree_tokenizer_vocab_lookup(vocab_, iree_make_cstring_view("tab\there")),
      1);
  EXPECT_EQ(
      iree_tokenizer_vocab_lookup(vocab_, iree_make_cstring_view("new\nline")),
      2);
}

//===----------------------------------------------------------------------===//
// WordPiece Model Parsing Tests
//===----------------------------------------------------------------------===//

TEST_F(ModelJsonTest, MinimalWordPieceModel) {
  // Minimal valid WordPiece model with vocab and defaults.
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0, "a": 1, "b": 2}
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);

  // Default unk_token "[UNK]" should be resolved.
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab_);
  EXPECT_EQ(ids.unk, 0);
}

TEST_F(ModelJsonTest, WordPieceModelWithAllFields) {
  // WordPiece model with all fields explicitly set.
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"<unk>": 0, "hello": 1, "##lo": 2, "wor": 3, "##ld": 4},
    "unk_token": "<unk>",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 50
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 5u);

  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab_);
  EXPECT_EQ(ids.unk, 0);
}

TEST_F(ModelJsonTest, WordPieceModelDefaultUnkToken) {
  // When unk_token is not specified, defaults to "[UNK]".
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 5, "a": 0, "b": 1}
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  iree_tokenizer_special_ids_t ids = iree_tokenizer_vocab_special_ids(vocab_);
  EXPECT_EQ(ids.unk, 5);
}

TEST_F(ModelJsonTest, WordPieceModelDefaultPrefix) {
  // When continuing_subword_prefix is not specified, defaults to "##".
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0, "un": 1, "##able": 2}
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, WordPieceModelDefaultMaxChars) {
  // When max_input_chars_per_word is not specified, defaults to 100.
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0, "a": 1}
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, WordPieceModelUnkTokenNotInVocab) {
  // unk_token that doesn't exist in the vocab should fail.
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"a": 0, "b": 1},
    "unk_token": "[UNK]"
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(ModelJsonTest, WordPieceModelInvalidMaxChars) {
  // max_input_chars_per_word <= 0 should fail.
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0},
    "max_input_chars_per_word": 0
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(ModelJsonTest, WordPieceModelUnknownFieldError) {
  // Unknown field should cause error (strict validation).
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0},
    "unknown_field": "value"
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(ModelJsonTest, WordPieceModelMissingVocab) {
  // Missing vocab should fail.
  const char* json = R"({
    "type": "WordPiece",
    "unk_token": "[UNK]"
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(ModelJsonTest, WordPieceModelWithEscapedTokens) {
  // WordPiece model with tokens that need JSON unescaping.
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0, "hello\"world": 1, "##tab\there": 2}
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);
}

TEST_F(ModelJsonTest, WordPieceModelNullUnkTokenError) {
  // Null unk_token disables UNK resolution, but WordPiece fundamentally
  // requires an UNK token (the algorithm emits UNK for untokenizable words).
  const char* json = R"({
    "type": "WordPiece",
    "vocab": {"a": 0, "b": 1},
    "unk_token": null
  })";

  iree_status_t status = iree_tokenizer_huggingface_parse_wordpiece_model(
      iree_make_cstring_view(json), &added_tokens_, allocator_, &model_,
      &vocab_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(ModelJsonTest, MissingVocab) {
  const char* json = R"({
    "type": "BPE",
    "merges": []
  })";

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_bpe_model(
          iree_make_cstring_view(json), &added_tokens_,
          IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
}

TEST_F(ModelJsonTest, MissingMerges) {
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0}
  })";

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_bpe_model(
          iree_make_cstring_view(json), &added_tokens_,
          IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
}

TEST_F(ModelJsonTest, InvalidMergeReferencesUnknownToken) {
  // Merge references token not in vocab.
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": ["a b"]
  })";

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_bpe_model(
          iree_make_cstring_view(json), &added_tokens_,
          IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
}

TEST_F(ModelJsonTest, UnknownFieldError) {
  // Unknown field should cause error (strict validation).
  const char* json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": [],
    "unknown_field": "value"
  })";

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_bpe_model(
          iree_make_cstring_view(json), &added_tokens_,
          IREE_TOKENIZER_BPE_FLAG_NONE, allocator_, &model_, &vocab_));
}

//===----------------------------------------------------------------------===//
// Model Type Dispatch Tests
//===----------------------------------------------------------------------===//

TEST_F(ModelJsonTest, AutoDetectBPE) {
  // Model type is auto-detected from "type" field.
  const char* model_json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": []
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_model(
      iree_make_cstring_view(model_json), &added_tokens_,
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, AutoDetectWordPiece) {
  // WordPiece model type is auto-detected from "type" field.
  const char* model_json = R"({
    "type": "WordPiece",
    "vocab": {"[UNK]": 0, "hello": 1, "##world": 2},
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_model(
      iree_make_cstring_view(model_json), &added_tokens_,
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);
}

TEST_F(ModelJsonTest, AutoDetectUnigram) {
  // Unigram model with valid vocab.
  const char* model_json = R"({
    "type": "Unigram",
    "vocab": [["<unk>", -10.0], ["hello", -2.0], ["world", -2.5]],
    "unk_id": 0
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_model(
      iree_make_cstring_view(model_json), &added_tokens_,
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE, allocator_, &model_,
      &vocab_));
  EXPECT_NE(model_, nullptr);
  EXPECT_NE(vocab_, nullptr);
  EXPECT_EQ(iree_tokenizer_vocab_token_count(vocab_), 3u);
}

TEST_F(ModelJsonTest, AutoDetectUnknownType) {
  // Unknown model type returns UNIMPLEMENTED (not INVALID_ARGUMENT) because
  // new model types could be added to the tokenizers library in the future.
  const char* model_json = R"({"type": "Unknown"})";

  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_tokenizer_huggingface_parse_model(
                            iree_make_cstring_view(model_json), &added_tokens_,
                            IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE,
                            allocator_, &model_, &vocab_));
}

//===----------------------------------------------------------------------===//
// Pre-Tokenizer Flag Propagation Tests
//===----------------------------------------------------------------------===//

TEST_F(ModelJsonTest, BPEWithByteLevelFlag) {
  // BYTE_LEVEL flag from pre_tokenizer parsing enables byte-to-unicode mapping
  // in the BPE model (GPT-2/RoBERTa/cl100k_base/Llama-3 style).
  const char* model_json = R"({
    "type": "BPE",
    "vocab": {"a": 0, "b": 1},
    "merges": []
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_model(
      iree_make_cstring_view(model_json), &added_tokens_,
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL, allocator_,
      &model_, &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

TEST_F(ModelJsonTest, BPEWithoutByteLevelFlag) {
  // No BYTE_LEVEL flag means no byte-to-unicode transformation.
  const char* model_json = R"({
    "type": "BPE",
    "vocab": {"a": 0},
    "merges": []
  })";

  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_model(
      iree_make_cstring_view(model_json), &added_tokens_,
      IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE, allocator_, &model_,
      &vocab_));

  ASSERT_NE(model_, nullptr);
  ASSERT_NE(vocab_, nullptr);
}

}  // namespace
