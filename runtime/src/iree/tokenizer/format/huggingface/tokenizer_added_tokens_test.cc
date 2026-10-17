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
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"

namespace {

// Inputs to these helpers are JSON string contents, with escapes preserved.
std::string TokenizerJson(const std::string& normalizer,
                          const std::string& content) {
  return R"({"version":"1.0","normalizer":)" + normalizer +
         R"(,"added_tokens":[
           {"id":10,"content":"[RAW]","single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":true},
           {"id":11,"content":")" +
         content +
         R"(","single_word":false,"lstrip":false,"rstrip":false,
            "normalized":true,"special":false}],
           "model":{"type":"BPE","vocab":{
             "[UNK]":0,"a":1,"b":2,"c":3,"_":4," ":5,"é":6
           },"merges":[],"unk_token":"[UNK]"}})";
}

void ExpectEncoding(iree_tokenizer_t* tokenizer, const std::string& input,
                    const std::vector<int32_t>& expected) {
  std::vector<int32_t> actual(512);
  iree_host_size_t count = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode(
      tokenizer, iree_make_string_view(input.data(), input.size()),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(actual.data(), nullptr, nullptr,
                                       actual.size()),
      iree_allocator_system(), &count));
  actual.resize(count);
  EXPECT_EQ(actual, expected);
}

void ExpectStreamingEncoding(iree_tokenizer_t* tokenizer,
                             const std::string& input,
                             const std::vector<int32_t>& expected) {
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  std::vector<uint8_t> transform_storage(
      iree_tokenizer_transform_buffer_recommended_size(input.size()));
  for (size_t split = 0; split <= input.size(); ++split) {
    SCOPED_TRACE(split);
    iree_tokenizer_encode_state_t* state = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
        tokenizer,
        iree_make_byte_span(state_storage.data(), state_storage.size()),
        iree_make_byte_span(transform_storage.data(), transform_storage.size()),
        iree_tokenizer_offset_run_list_empty(), IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_allocator_system(), &state));
    std::vector<int32_t> actual;
    size_t position = 0;
    size_t capacity = 0;
    for (size_t end : {split, input.size()}) {
      bool made_progress = true;
      while (made_progress) {
        int32_t token = 0;
        iree_host_size_t consumed = 0;
        iree_host_size_t count = 0;
        IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
            state,
            iree_make_string_view(input.data() + position, end - position),
            iree_tokenizer_make_token_output(&token, nullptr, nullptr,
                                             capacity),
            &consumed, &count));
        position += consumed;
        if (count > 0) {
          actual.push_back(token);
        }
        made_progress = consumed > 0 || count > 0 || capacity == 0;
        capacity = 1;
      }
    }
    EXPECT_EQ(position, input.size());
    do {
      int32_t token = -1;
      iree_host_size_t count = 0;
      IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
          state, iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &count));
      ASSERT_TRUE(count || !iree_tokenizer_encode_state_has_pending(state));
      if (count) {
        actual.push_back(token);
      }
    } while (iree_tokenizer_encode_state_has_pending(state));
    EXPECT_EQ(actual, expected);
    iree_tokenizer_encode_state_deinitialize(state);
  }
}

TEST(TokenizerAddedTokensTest, DeclaredNormalizerProducesMatchingSpelling) {
  // Independent tokenizers 0.23.3 oracle: the declared normalizer applies to
  // normalized added-token patterns as well as unmatched input spans. The
  // original vocabulary spelling and raw-token matching remain unchanged.
  struct TestCase {
    // Declared normalizer JSON, excluding pre-tokenizer transformations.
    const char* normalizer;
    // Original added-token content as JSON string contents.
    const char* content;
    // Input whose normalized representation matches the normalized pattern.
    const char* input;
    // Original vocabulary spelling after JSON unescaping.
    const char* original;
  } cases[] = {
      {R"({"type":"Lowercase"})", "AB", "ab", "AB"},
      {R"({"type":"Prepend","prepend":"_"})", "ab", "ab", "ab"},
      {R"({"type":"Strip","strip_left":true,"strip_right":true})", " ab ", "ab",
       " ab "},
      {R"({"type":"NFC"})", "e\\u0301", "é", "e\u0301"},
      {R"({"type":"Sequence","normalizers":[
             {"type":"NFD"},{"type":"StripAccents"},
             {"type":"Lowercase"}]})",
       "É", "é", "É"},
      {R"({"type":"Sequence","normalizers":[
             {"type":"NFC"},
             {"type":"Replace","pattern":{"String":"é"},"content":"ab"}]})",
       "e\\u0301", "ab", "e\u0301"},
  };
  for (const auto& item : cases) {
    SCOPED_TRACE(item.normalizer);
    std::string json = TokenizerJson(item.normalizer, item.content);
    iree_tokenizer_builder_t builder;
    iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(
        iree_make_string_view(json.data(), json.size()), &builder));
    auto spelling = iree_tokenizer_vocab_token_text(builder.vocab, 11);
    EXPECT_EQ(std::string(spelling.data, spelling.size), item.original);
    iree_tokenizer_t* tokenizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_builder_build(&builder, &tokenizer));
    iree_tokenizer_builder_deinitialize(&builder);
    ExpectEncoding(tokenizer, item.input, {11});
    ExpectStreamingEncoding(tokenizer, item.input, {11});
    ExpectEncoding(tokenizer, "[RAW]", {10});
    ExpectStreamingEncoding(tokenizer, "[RAW]", {10});
    iree_tokenizer_free(tokenizer);
  }
}

TEST(TokenizerAddedTokensTest, EmptyNormalizedPatternDoesNotMatch) {
  std::string json = TokenizerJson(
      R"({"type":"Strip","strip_left":true,"strip_right":true})", "   ");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(),
      &tokenizer));
  ExpectEncoding(tokenizer, "", {});
  ExpectEncoding(tokenizer, "   ", {});
  ExpectEncoding(tokenizer, "a", {1});
  ExpectEncoding(tokenizer, "[RAW]", {10});
  iree_tokenizer_free(tokenizer);
}

TEST(TokenizerAddedTokensTest, NormalizedPatternCapacity) {
  // The stored matching pattern has a one-byte length. Apply that bound to
  // the normalized spelling, including output retained until finalization.
  for (size_t count : {size_t{254}, size_t{255}}) {
    SCOPED_TRACE(count);
    std::string content(count, 'a');
    std::string json =
        TokenizerJson(R"({"type":"Prepend","prepend":"_"})", content);
    iree_tokenizer_t* tokenizer = nullptr;
    auto status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json.data(), json.size()),
        iree_allocator_system(), &tokenizer);
    if (count == 254) {
      IREE_ASSERT_OK(status);
      ExpectEncoding(tokenizer, content, {11});
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
      EXPECT_EQ(tokenizer, nullptr);
    }
    iree_tokenizer_free(tokenizer);
  }
}

TEST(TokenizerAddedTokensTest, ErasedSuffixAtPatternCapacity) {
  // The input can exceed the matching capacity when normalization erases its
  // suffix. Filling the result must not prevent consuming that final suffix.
  std::string input(255, 'a');
  std::string json =
      TokenizerJson(R"({"type":"Strip","strip_left":true,"strip_right":true})",
                    input + "   ");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(),
      &tokenizer));
  ExpectEncoding(tokenizer, input, {11});
  iree_tokenizer_free(tokenizer);
}

TEST(TokenizerAddedTokensTest, ComposedExpansionExceedsMatchingCapacity) {
  std::string normalizer =
      R"({"type":"Sequence","normalizers":[
        {"type":"Replace","pattern":{"String":"a"},"content":")" +
      std::string(32, 'b') +
      R"("},{"type":"Replace","pattern":{"String":"b"},"content":")" +
      std::string(32, 'c') + R"("}]})";
  std::string json = TokenizerJson(normalizer, "a");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_from_huggingface_json(
                            iree_make_string_view(json.data(), json.size()),
                            iree_allocator_system(), &tokenizer));
  EXPECT_EQ(tokenizer, nullptr);
}

TEST(TokenizerAddedTokensTest, ComposedExpansionThenContractionMatches) {
  std::string normalizer =
      R"({"type":"Sequence","normalizers":[
        {"type":"Replace","pattern":{"String":"a"},"content":")" +
      std::string(32, 'b') + R"("},{"type":"Replace","pattern":{"String":")" +
      std::string(32, 'b') + R"("},"content":"c"}]})";
  std::string input(20, 'a');
  std::string json = TokenizerJson(normalizer, input);
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_string_view(json.data(), json.size()), iree_allocator_system(),
      &tokenizer));
  ExpectEncoding(tokenizer, input, {11});
  ExpectStreamingEncoding(tokenizer, input, {11});
  iree_tokenizer_free(tokenizer);
}

TEST(TokenizerAddedTokensTest,
     FailedNormalizedCollectionReleasesRawCollection) {
  // The raw collection is built first. Exceed the normalized collection's
  // prefix capacity to exercise cleanup after only that first build succeeds.
  std::string json = R"({"normalizer":{"type":"Lowercase"},"added_tokens":[
    {"id":100,"content":"[RAW]","single_word":false,"lstrip":false,
     "rstrip":false,"normalized":false,"special":true})";
  for (int index = 0; index < 17; ++index) {
    json += R"(,{"id":)" + std::to_string(101 + index) + R"(,"content":")" +
            std::string(1, 'A' + index) +
            R"(","single_word":false,"lstrip":false,"rstrip":false,
                "normalized":true,"special":false})";
  }
  json += R"(],"model":{"type":"BPE","vocab":{"a":0},"merges":[]}})";
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_tokenizer_from_huggingface_json(
                            iree_make_string_view(json.data(), json.size()),
                            iree_allocator_system(), &tokenizer));
  EXPECT_EQ(tokenizer, nullptr);
}

}  // namespace
