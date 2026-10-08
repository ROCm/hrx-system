// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/http/tokenization.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

namespace {

class TokenizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(allocator_, &output_);
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(IREE_SV(R"({
          "model":{"type":"BPE","vocab":{
            "a":0,"b":1,"z":2,"zz":3,"<bos>":4,"<eos>":5,
            "é":6,"\n":7,"\"":8,"\\":9,"🦊":10,
            "abcdefghijklmnop":12},"merges":[]},
          "decoder":{"type":"ByteFallback"},
          "added_tokens":[
            {"id":3,"content":"zz","special":true,"single_word":false,
             "lstrip":false,"rstrip":false,"normalized":false},
            {"id":4,"content":"<bos>","special":true,"single_word":false,
             "lstrip":false,"rstrip":false,"normalized":false},
            {"id":5,"content":"<eos>","special":true,"single_word":false,
             "lstrip":false,"rstrip":false,"normalized":false}],
          "post_processor":{"type":"TemplateProcessing","single":[
            {"SpecialToken":{"id":"<bos>","type_id":0}},
            {"Sequence":{"id":"A","type_id":0}},
            {"SpecialToken":{"id":"<eos>","type_id":0}}],"pair":[],
            "special_tokens":{
              "<bos>":{"id":"<bos>","ids":[4],"tokens":["<bos>"]},
              "<eos>":{"id":"<eos>","ids":[5],"tokens":["<eos>"]}}}})"),
                                                        allocator_,
                                                        &tokenizer_));
  }
  void TearDown() override {
    iree_string_builder_deinitialize(&output_);
    iree_tokenizer_free(tokenizer_);
  }
  iree_status_t Encode(const std::string& body) {
    iree_string_builder_reset(&output_);
    return loom_serve_http_tokenize(
        tokenizer_, iree_make_string_view(body.data(), body.size()), &output_,
        allocator_);
  }
  iree_status_t Decode(const std::string& body) {
    iree_string_builder_reset(&output_);
    return loom_serve_http_detokenize(
        tokenizer_, iree_make_string_view(body.data(), body.size()), &output_,
        allocator_);
  }
  std::string Output() {
    const auto value = iree_string_builder_view(&output_);
    return std::string(value.data, value.size);
  }
  static std::string TokenRequest(size_t count, const char* id) {
    std::string result = "{\"model\":\"m\",\"tokens\":[";
    for (size_t i = 0; i < count; ++i) {
      if (i) {
        result += ",";
      }
      result += id;
    }
    return result + "]}";
  }
  // Allocator for real tokenizer state and response serialization.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Immutable HF tokenizer with sparse IDs and special-token processing.
  iree_tokenizer_t* tokenizer_ = nullptr;
  // Response builder reset between requests, including rejected inputs.
  iree_string_builder_t output_ = {};
};

TEST_F(TokenizationTest, DefaultsSpecialFlagsAndIndependentStates) {
  IREE_ASSERT_OK(Encode(R"({"model":"m","content":"azzb"})"));
  EXPECT_EQ(Output(), R"({"tokens":[0,3,1]})");
  IREE_ASSERT_OK(
      Encode(R"({"model":"m","content":"azzb","add_special":true})"));
  EXPECT_EQ(Output(), R"({"tokens":[4,0,3,1,5]})");
  IREE_ASSERT_OK(
      Encode(R"({"model":"m","content":"azzb","parse_special":false})"));
  EXPECT_EQ(Output(), R"({"tokens":[0,2,2,1]})");
  IREE_ASSERT_OK(Encode(R"({"model":"m","content":"azzb", "with_pieces":false,
                            "parse_special":false,"add_special":true})"));
  EXPECT_EQ(Output(), R"({"tokens":[4,0,2,2,1,5]})");
  IREE_ASSERT_OK(Encode(R"({"model":"m","content":""})"));
  EXPECT_EQ(Output(), R"({"tokens":[]})");
  IREE_ASSERT_OK(Encode(R"({"model":"m","content":"","add_special":true})"));
  EXPECT_EQ(Output(), R"({"tokens":[4,5]})");
  IREE_ASSERT_OK(Decode(R"({"model":"m","tokens":[4,0,3,1,5]})"));
  EXPECT_EQ(Output(), R"({"content":"<bos>azzb<eos>"})");
  IREE_ASSERT_OK(Decode(R"({"model":"m","tokens":[]})"));
  EXPECT_EQ(Output(), R"({"content":""})");
}

TEST_F(TokenizationTest, UnicodeEscapesAndSparseVocabulary) {
  IREE_ASSERT_OK(
      Encode(R"({"model":"m","content":"\u00e9\n\"\\\ud83e\udd8a"})"));
  EXPECT_EQ(Output(), R"({"tokens":[6,7,8,9,10]})");
  IREE_ASSERT_OK(Decode(R"({"model":"m","tokens":[6,7,8,9,10]})"));
  EXPECT_EQ(Output(), R"({"content":"é\n\"\\🦊"})");
  for (const char* ids :
       {"[11]", "[13]", "[-1]", "[2147483648]", "[18446744073709551616]",
        "[0.0]", "[0e0]", "[\"0\"]", "[null]", "[true]", "[[0]]"}) {
    SCOPED_TRACE(ids);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Decode(std::string("{\"tokens\":") + ids + "}"));
  }
  IREE_ASSERT_OK(Decode(R"({"model":"m","tokens":[12]})"));
  EXPECT_EQ(Output(), R"({"content":"abcdefghijklmnop"})");
}

TEST_F(TokenizationTest, StrictFieldsAndUnsupportedProfiles) {
  for (const char* body :
       {R"({})", R"({"content":null})", R"({"content":[]})",
        R"({"content":"a","content":"b"})",
        R"({"content":"a","add_special":0})",
        R"({"content":"a","parse_special":"false"})",
        R"({"content":"a","with_pieces":true})",
        R"({"content":"a","with_pieces":null})",
        R"({"content":"a","unknown":true})",
        R"({"prompt":"a","add_special_tokens":true})", R"({"messages":[]})",
        R"({"content":"\ud800"})", R"({"content":"a"}true)"}) {
    SCOPED_TRACE(body);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Encode(body));
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Encode(std::string("{\"content\":\"") + '\xff' + "\"}"));
  for (const char* body :
       {R"({})", R"({"tokens":"a"})", R"({"tokens":[],"tokens":[]})",
        R"({"tokens":[],"skip_special_tokens":true})", R"({"tokens":[]}[])"}) {
    SCOPED_TRACE(body);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Decode(body));
  }
  IREE_ASSERT_OK(Encode(" \n{\"content\":\"a\"} \r\n"));
  EXPECT_EQ(Output(), R"({"tokens":[0]})");
}

TEST_F(TokenizationTest, ByteLevelDecodeKeepsCrossTokenUtf8State) {
  iree_tokenizer_t* bytes = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      IREE_SV(R"({"model":{"type":"BPE","vocab":{"Ã":0,"©":1},
                  "merges":[]},"pre_tokenizer":{"type":"ByteLevel",
                  "add_prefix_space":false,"trim_offsets":false,"use_regex":false},
                  "decoder":{"type":"ByteLevel","add_prefix_space":false,
                             "trim_offsets":false,"use_regex":false}})"),
      allocator_, &bytes));
  iree_tokenizer_free(tokenizer_);
  tokenizer_ = bytes;
  IREE_ASSERT_OK(Encode(R"({"content":"é"})"));
  EXPECT_EQ(Output(), R"({"tokens":[0,1]})");
  IREE_ASSERT_OK(Decode(R"({"tokens":[0,1]})"));
  EXPECT_EQ(Output(), R"({"content":"é"})");
  IREE_ASSERT_OK(Decode(R"({"tokens":[0]})"));
  EXPECT_EQ(Output(), R"({"content":"�"})");
}

TEST_F(TokenizationTest, NormalizedTailsMatchTheCompleteTokenizer) {
  iree_tokenizer_t* normalized = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(IREE_SV(R"({"model":{"type":"BPE",
                  "vocab":{"a":0,"b":1,"z":2,"é":3," ":4},"merges":[]},
                 "normalizer":{"type":"Sequence","normalizers":[
                   {"type":"NFC"},{"type":"Lowercase"},
                   {"type":"Replace","pattern":{"String":"ab"},"content":"z"},
                   {"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "},
                   {"type":"Strip","strip_left":true,"strip_right":true}]}})"),
                                           allocator_, &normalized));
  iree_tokenizer_free(tokenizer_);
  tokenizer_ = normalized;
  for (const char* input :
       {"", "a", "ab", " a ", "a  ", "AB", "aba", "é", "é"}) {
    SCOPED_TRACE(input);
    int32_t expected[32];
    size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode(
        tokenizer_, iree_make_cstring_view(input),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(expected, nullptr, nullptr,
                                         IREE_ARRAYSIZE(expected)),
        allocator_, &count));
    std::string response = "{\"tokens\":[";
    for (size_t i = 0; i < count; ++i) {
      if (i) {
        response += ",";
      }
      response += std::to_string(expected[i]);
    }
    response += "]}";
    IREE_ASSERT_OK(Encode(std::string("{\"content\":\"") + input + "\"}"));
    EXPECT_EQ(Output(), response);
  }
}

TEST_F(TokenizationTest, ExactTokenBoundIncludesFinalSpecialTokens) {
  const size_t limit = LOOM_SERVE_HTTP_TOKENIZATION_MAX_TOKENS;
  IREE_ASSERT_OK(Encode("{\"content\":\"" + std::string(limit, 'a') + "\"}"));
  EXPECT_EQ(Output().size(), 2 * limit + 12);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      Encode("{\"content\":\"" + std::string(limit + 1, 'a') + "\"}"));
  IREE_ASSERT_OK(Encode("{\"content\":\"" + std::string(limit - 2, 'a') +
                        "\",\"add_special\":true}"));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Encode("{\"content\":\"" + std::string(limit - 1, 'a') +
                               "\",\"add_special\":true}"));
  IREE_ASSERT_OK(Decode(TokenRequest(limit, "0")));
  EXPECT_EQ(Output(), "{\"content\":\"" + std::string(limit, 'a') + "\"}");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Decode(TokenRequest(limit + 1, "0")));
}

TEST_F(TokenizationTest, DecodedTextBoundRejectsBeforeReturningAPrefix) {
  const size_t limit = LOOM_SERVE_HTTP_TOKENIZATION_MAX_TEXT_BYTES;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      Encode("{\"content\":\"" + std::string(limit + 1, 'a') + "\"}"));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      Encode("{\"content\":\"" + std::string(limit + 2, 'a') + "\"}"));
  IREE_ASSERT_OK(Decode(TokenRequest(limit / 16, "12")));
  EXPECT_EQ(Output().size(), limit + 14);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Decode(TokenRequest(limit / 16 + 1, "12")));
}

}  // namespace
