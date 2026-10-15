// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

namespace {

// Expected IDs and type IDs are checked against tokenizers 0.23.3.
std::string Expansion() {
  return R"({"type":"Sequence","normalizers":[
    {"type":"Replace","pattern":{"String":"a"},"content":")" +
         std::string(32, 'b') + R"("},
    {"type":"Replace","pattern":{"String":"b"},"content":")" +
         std::string(32, 'c') + R"("}]})";
}
std::string Config(const std::string& normalizer,
                   const std::string& raw = "<aaaaaaaaa!>",
                   const std::string& normalized = "[N]",
                   const std::string& merges = "[]",
                   const std::string& pre_tokenizer = "null") {
  return R"({"normalizer":)" + normalizer + R"(,
  "added_tokens":[
    {"id":8,"content":")" +
         raw +
         R"(","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    {"id":9,"content":")" +
         normalized +
         R"(","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
  "post_processor":{"type":"TemplateProcessing",
    "single":[{"SpecialToken":{"id":"BOS","type_id":7}},
      {"Sequence":{"id":"A","type_id":2}},
      {"SpecialToken":{"id":"EOS","type_id":8}},
      {"SpecialToken":{"id":"TAIL","type_id":9}}],"pair":[],
    "special_tokens":{"BOS":{"id":"BOS","ids":[1],"tokens":["BOS"]},
      "EOS":{"id":"EOS","ids":[2],"tokens":["EOS"]},
      "TAIL":{"id":"TAIL","ids":[3],"tokens":["TAIL"]}}},
  "model":{"type":"BPE","vocab":{"c":0,"BOS":1,"EOS":2,"TAIL":3,
    "<":4,"?":5,"☃":6,"☂":7,"[":10,"N":11,"]":12,"cc":13,"c[":14,
    "ᄀ":15,"ᅡ":16,"ᆨ":17,"가":18,"각":19,"▁":20,")" +
         raw + R"(":8,")" + normalized + R"(":9},"merges":)" + merges +
         R"(},"pre_tokenizer":)" + pre_tokenizer + "}";
}

std::string SpanConfig(const std::string& normalizer) {
  return Config(normalizer, "<aaaaaaaaa!>", "[N]", R"([["c","c"],["c","["]])");
}
class TokenizerFinalizationTest : public ::testing::Test {
 protected:
  ~TokenizerFinalizationTest() override {
    iree_tokenizer_encode_state_deinitialize(state_);
    iree_tokenizer_free(tokenizer_);
  }
  void Check(const std::string& json, const std::string& input,
             const std::vector<int32_t>& body) {
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json.data(), json.size()),
        iree_allocator_system(), &tokenizer_));
    size_t state_size = 0;
    IREE_ASSERT_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));
    storage_.resize(state_size);
    transform_.resize(IREE_TOKENIZER_TRANSFORM_BUFFER_MIN_SIZE);
    std::vector<int32_t> expected{1};
    expected.insert(expected.end(), body.begin(), body.end());
    expected.insert(expected.end(), {2, 3});
    std::vector<int32_t> expected_types(expected.size(), 2);
    expected_types.front() = 7;
    expected_types[expected.size() - 2] = 8;
    expected_types.back() = 9;
    std::vector<int32_t> complete(expected.size());
    size_t complete_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode(
        tokenizer_, iree_make_string_view(input.data(), input.size()),
        IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
        iree_tokenizer_make_token_output(complete.data(), nullptr, nullptr,
                                         complete.size()),
        iree_allocator_system(), &complete_count));
    ASSERT_EQ(complete_count, expected.size());
    EXPECT_EQ(complete, expected);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_tokenizer_encode(
            tokenizer_, iree_make_string_view(input.data(), input.size()),
            IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
            iree_tokenizer_make_token_output(complete.data(), nullptr, nullptr,
                                             complete.size() - 1),
            iree_allocator_system(), &complete_count));
    for (size_t chunk : {1u, 3u, 17u, 1024u}) {
      for (size_t capacity : {1u, 7u, 1024u}) {
        SCOPED_TRACE(::testing::Message()
                     << "chunk=" << chunk << " capacity=" << capacity);
        IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
            tokenizer_, iree_make_byte_span(storage_.data(), storage_.size()),
            iree_make_byte_span(transform_.data(), transform_.size()),
            iree_tokenizer_offset_run_list_empty(),
            IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS, &state_));
        std::vector<int32_t> actual, types;
        int32_t output[1024];
        uint8_t type_ids[1024];
        size_t position = 0, end = 0;
        while (position < input.size()) {
          end = std::min(end + chunk, input.size());
          for (size_t credit : {size_t{0}, capacity}) {
            size_t consumed = 0, written = 0;
            IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
                state_,
                iree_make_string_view(input.data() + position, end - position),
                iree_tokenizer_make_token_output(output, nullptr, type_ids,
                                                 credit),
                &consumed, &written));
            ASSERT_LE(consumed, end - position);
            ASSERT_LE(written, credit);
            position += consumed;
            actual.insert(actual.end(), output, output + written);
            types.insert(types.end(), type_ids, type_ids + written);
            if (credit && position < input.size()) {
              ASSERT_TRUE(consumed || written || end < input.size());
            }
          }
        }
        do {
          size_t written = 0;
          IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
              state_,
              iree_tokenizer_make_token_output(output, nullptr, type_ids, 0),
              &written));
          ASSERT_EQ(written, 0u);
          IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
              state_,
              iree_tokenizer_make_token_output(output, nullptr, type_ids,
                                               capacity),
              &written));
          ASSERT_LE(written, capacity);
          actual.insert(actual.end(), output, output + written);
          types.insert(types.end(), type_ids, type_ids + written);
          ASSERT_TRUE(written ||
                      !iree_tokenizer_encode_state_has_pending(state_));
        } while (iree_tokenizer_encode_state_has_pending(state_));
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(types, expected_types);
        size_t written = 0;
        IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
            state_,
            iree_tokenizer_make_token_output(output, nullptr, type_ids,
                                             capacity),
            &written));
        EXPECT_EQ(written, 0u);
        size_t consumed = 0;
        IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                              iree_tokenizer_encode_state_feed(
                                  state_, IREE_SV("c"),
                                  iree_tokenizer_make_token_output(
                                      output, nullptr, type_ids, capacity),
                                  &consumed, &written));
        EXPECT_EQ(consumed, 0u);
        EXPECT_EQ(written, 0u);
        iree_tokenizer_encode_state_deinitialize(state_);
        state_ = nullptr;
      }
    }
  }
  // Owned immutable tokenizer loaded through the shipping format parser.
  iree_tokenizer_t* tokenizer_ = nullptr;
  // Encoder state borrowing the buffers below.
  iree_tokenizer_encode_state_t* state_ = nullptr;
  // Owned component state, retained across every output span.
  std::vector<uint8_t> storage_;
  // Minimum supported intermediate ring storage.
  std::vector<uint8_t> transform_;
};
TEST_F(TokenizerFinalizationTest, ExpandingNormalization) {
  Check(Config(Expansion()), std::string(20, 'a'),
        std::vector<int32_t>(20 * 1024, 0));
}
TEST_F(TokenizerFinalizationTest, PartialRawMatchAtEof) {
  std::vector<int32_t> body{4};
  body.insert(body.end(), 9 * 1024, 0);
  Check(Config(Expansion()), "<aaaaaaaaa", body);
}
TEST_F(TokenizerFinalizationTest, FailedRawMatch) {
  std::vector<int32_t> body{4};
  body.insert(body.end(), 9 * 1024, 0);
  body.push_back(5);
  Check(Config(Expansion()), "<aaaaaaaaa?", body);
}
TEST_F(TokenizerFinalizationTest, NormalizedMatchWaitsForNormalizerEof) {
  Check(Config(Expansion(), "[R]", std::string(32, 'c')), std::string(20, 'a'),
        std::vector<int32_t>(20 * 32, 9));
}
TEST_F(TokenizerFinalizationTest, UnicodeRawRecovery) {
  Check(Config(R"({"type":"Lowercase"})", "☃!"), "☂☃?", {7, 6, 5});
}
TEST_F(TokenizerFinalizationTest, MatchedRawToken) {
  Check(Config(Expansion()), "<aaaaaaaaa!>", {8});
}
TEST_F(TokenizerFinalizationTest, NormalizedPartialAtEof) {
  Check(Config(Expansion()), "[N", {10, 11});
}
TEST_F(TokenizerFinalizationTest, UnicodePartialAtEof) {
  Check(Config(R"({"type":"Lowercase"})", "☃!"), "☃", {6});
}
TEST_F(TokenizerFinalizationTest, EmptyInput) {
  Check(Config(Expansion()), "", {});
}

TEST_F(TokenizerFinalizationTest, ModelSpanCrossesInputChunks) {
  std::vector<int32_t> body(1024, 13);
  body.push_back(0);
  Check(SpanConfig("null"), std::string(2049, 'c'), body);
}

TEST_F(TokenizerFinalizationTest, ModelSpanCrossesNormalizerOutputChunks) {
  Check(SpanConfig(Expansion()), std::string(20, 'a'),
        std::vector<int32_t>(20 * 512, 13));
}

TEST_F(TokenizerFinalizationTest, ModelSpanIncludesFinalDecomposition) {
  Check(Config(R"({"type":"NFD"})", "<aaaaaaaaa!>", "[N]",
               R"([["ᄀ","ᅡ"],["가","ᆨ"]])"),
        "각", {19});
}

TEST_F(TokenizerFinalizationTest, RawTokenCompletesPreviousModelSpan) {
  std::vector<int32_t> body(512, 13);
  body.push_back(8);
  body.insert(body.end(), 512, 13);
  Check(SpanConfig(Expansion()), "a<aaaaaaaaa!>a", body);
}

TEST_F(TokenizerFinalizationTest, NormalizedTokenCompletesPreviousModelSpan) {
  Check(SpanConfig("null"), "ccc[N]ccc", {13, 0, 9, 13, 0});
}

TEST_F(TokenizerFinalizationTest, FailedNormalizedPrefixKeepsModelSpanOpen) {
  Check(SpanConfig("null"), "ccc[N?c", {13, 14, 11, 5, 0});
}

TEST_F(TokenizerFinalizationTest, NormalizedPrefixAtEofKeepsModelSpanOpen) {
  Check(SpanConfig("null"), "ccc[N", {13, 14, 11});
}

TEST_F(TokenizerFinalizationTest, DeclaredPrependRestartsAfterRawToken) {
  Check(Config(R"({"type":"Prepend","prepend":"c"})", "<R>"), "c<R>c<R><R>c",
        {0, 0, 8, 0, 0, 8, 8, 0, 0});
}

TEST_F(TokenizerFinalizationTest, RawTokensDoNotCreateEmptyNormalizationSpans) {
  Check(Config(R"({"type":"Prepend","prepend":"c"})", "<R>"), "<R>c<R>",
        {8, 0, 0, 8});
}

TEST_F(TokenizerFinalizationTest, NormalizedTokenDoesNotRestartNormalizer) {
  Check(Config(R"({"type":"Prepend","prepend":"c"})", "<R>"), "[N]c[N]c",
        {9, 9, 0});
}

TEST_F(TokenizerFinalizationTest, DeclaredSequenceRestartsAsOneComponent) {
  Check(Config(R"({"type":"Sequence","normalizers":[
                  {"type":"Prepend","prepend":"c"},
                  {"type":"Replace","pattern":{"String":"c"},"content":"cc"}]})",
               "<R>"),
        "c<R>c", {0, 0, 0, 0, 8, 0, 0, 0, 0});
}

TEST_F(TokenizerFinalizationTest, DeclaredPrependWithMetaspace) {
  Check(Config(R"({"type":"Prepend","prepend":"▁"})", "<R>", "[N]", "[]",
               R"({"type":"Metaspace","replacement":"▁",
                    "prepend_scheme":"always","split":true})"),
        "c<R>c<R><R>c", {20, 0, 8, 20, 0, 8, 8, 20, 0});
}

TEST_F(TokenizerFinalizationTest, PreTokenizerPrefixPolicySurvivesRawToken) {
  Check(Config(R"({"type":"Prepend","prepend":"c"})", "<R>", "[N]", "[]",
               R"({"type":"Metaspace","replacement":"▁",
                    "prepend_scheme":"first","split":true})"),
        "c<R>c", {20, 0, 0, 8, 0, 0});
}

TEST_F(TokenizerFinalizationTest, RawTokenConsumesOriginalFirstPosition) {
  Check(Config(R"({"type":"Prepend","prepend":"c"})", "<R>", "[N]", "[]",
               R"({"type":"Metaspace","replacement":"▁",
                    "prepend_scheme":"first","split":true})"),
        "<R>c<R>c", {8, 0, 0, 8, 0, 0});
}
}  // namespace
