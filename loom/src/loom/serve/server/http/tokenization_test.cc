// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/tokenization.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

namespace {

class TokenizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(host_allocator_, &output_);
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
                                                        host_allocator_,
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
        host_allocator_);
  }
  iree_status_t Decode(const std::string& body) {
    iree_string_builder_reset(&output_);
    return loom_serve_http_detokenize(
        tokenizer_, iree_make_string_view(body.data(), body.size()), &output_,
        host_allocator_);
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
  const iree_allocator_t host_allocator_ = iree_allocator_system();
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

TEST_F(TokenizationTest, MillionTokenRoundTripPreservesEveryPosition) {
  const size_t count = 1024 * 1024 + 3;
  std::string text;
  std::string ids;
  text.reserve(count);
  ids.reserve(2 * count);
  for (size_t i = 0; i < count; ++i) {
    text += i % 3 ? 'a' : 'b';
    if (i) {
      ids += ',';
    }
    ids += i % 3 ? '0' : '1';
  }
  IREE_ASSERT_OK(Encode("{\"content\":\"" + text + "\",\"add_special\":true}"));
  EXPECT_EQ(Output(), "{\"tokens\":[4," + ids + ",5]}");
  IREE_ASSERT_OK(Decode("{\"tokens\":[" + ids + "]}"));
  EXPECT_EQ(Output(), "{\"content\":\"" + text + "\"}");
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
      host_allocator_, &bytes));
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
                                           host_allocator_, &normalized));
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
        host_allocator_, &count));
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

TEST_F(TokenizationTest, ComposedExpansionDrainsAllFinalTokens) {
  const std::string configuration =
      R"({"normalizer":{"type":"Sequence","normalizers":[
        {"type":"Replace","pattern":{"String":"a"},"content":")" +
      std::string(32, 'b') + R"("},
        {"type":"Replace","pattern":{"String":"b"},"content":")" +
      std::string(32, 'c') +
      R"("}]},"model":{"type":"BPE","vocab":{"c":0},"merges":[]}})";
  iree_tokenizer_t* expanding = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_string_view(configuration.data(), configuration.size()),
      host_allocator_, &expanding));
  iree_tokenizer_free(tokenizer_);
  tokenizer_ = expanding;
  for (size_t input_length : {8, 256, 1025}) {
    SCOPED_TRACE(input_length);
    const std::string request =
        "{\"content\":\"" + std::string(input_length, 'a') + "\"}";
    IREE_ASSERT_OK(Encode(request));
    std::string expected = "{\"tokens\":[";
    for (size_t i = 0; i < input_length * 1024; ++i) {
      expected += i ? ",0" : "0";
    }
    expected += "]}";
    EXPECT_EQ(Output(), expected);
  }
}

TEST_F(TokenizationTest, FinalSpecialTokensCrossOutputChunks) {
  for (size_t count : {1022, 1023, 1024, 1025}) {
    SCOPED_TRACE(count);
    IREE_ASSERT_OK(Encode("{\"content\":\"" + std::string(count, 'a') +
                          "\",\"add_special\":true}"));
    std::string expected = "{\"tokens\":[4";
    for (size_t i = 0; i < count; ++i) {
      expected += ",0";
    }
    EXPECT_EQ(Output(), expected + ",5]}");
  }
}

TEST_F(TokenizationTest, DecodedTextGrowsPastOneMiB) {
  const size_t count = 65537;
  std::string expected = "{\"content\":\"";
  for (size_t i = 0; i < count; ++i) {
    expected += "abcdefghijklmnop";
  }
  IREE_ASSERT_OK(Decode(TokenRequest(count, "12")));
  EXPECT_EQ(Output(), expected + "\"}");
}

TEST_F(TokenizationTest, IndivisibleTokenGrowsReusableDecodeScratch) {
  const std::string text(8193, 'a');
  const std::string configuration =
      "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"" + text +
      "\":0,\"b\":1},\"merges\":[]},\"decoder\":{\"type\":\"ByteFallback\"}}";
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_string_view(configuration.data(), configuration.size()),
      host_allocator_, &tokenizer));
  iree_tokenizer_free(tokenizer_);
  tokenizer_ = tokenizer;
  IREE_ASSERT_OK(Decode(R"({"tokens":[0,1,0]})"));
  EXPECT_EQ(Output(), "{\"content\":\"" + text + "b" + text + "\"}");
}

TEST_F(TokenizationTest, QueuedStatefulOutputDrainsThroughFinalization) {
  const std::string a(64, 'a');
  const std::string b(64, 'b');
  const std::string configuration =
      "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"" + a + "\":0,\"" + b +
      R"(":1},"merges":[]},"decoder":{"type":"CTC","pad_token":"<pad>",
          "word_delimiter_token":"|","cleanup":false}})";
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_string_view(configuration.data(), configuration.size()),
      host_allocator_, &tokenizer));
  iree_tokenizer_free(tokenizer_);
  tokenizer_ = tokenizer;
  // The final lookup batch is larger than one output chunk. Accepted IDs
  // still own queued strings when the utility begins decoder finalization.
  std::string ids;
  std::string expected;
  for (size_t i = 0; i < 128; ++i) {
    if (i) {
      ids += ',';
    }
    ids += i % 2 ? '1' : '0';
    expected += i % 2 ? b : a;
  }
  IREE_ASSERT_OK(Decode("{\"tokens\":[" + ids + "]}"));
  EXPECT_EQ(Output(), "{\"content\":\"" + expected + "\"}");
}

TEST_F(TokenizationTest, OutputFailureIsNotSuccessfulTruncation) {
  char storage[32];
  iree_string_builder_t bounded;
  iree_string_builder_initialize_with_storage(storage, sizeof(storage),
                                              &bounded);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_serve_http_tokenize(tokenizer_,
                               IREE_SV(R"({"content":"aaaaaaaaaaaaaaaaaaaa"})"),
                               &bounded, host_allocator_));
  iree_string_builder_reset(&bounded);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_http_detokenize(
                            tokenizer_, IREE_SV(R"({"tokens":[12,12,12]})"),
                            &bounded, host_allocator_));
  iree_string_builder_deinitialize(&bounded);
  IREE_ASSERT_OK(Encode(R"({"content":"ab"})"));
  EXPECT_EQ(Output(), R"({"tokens":[0,1]})");
}

TEST_F(TokenizationTest, AllocationFailuresRetireRequestScratch) {
  struct AllocationState {
    // Allocation request denied by this invocation.
    size_t fail_at = 0;
    // Number of allocation requests observed.
    size_t attempts = 0;
    // Allocations still owned by the utility invocation.
    size_t live = 0;
    // Total bytes requested from this invocation's scratch allocator.
    size_t requested_bytes = 0;
  };
  auto allocate = [](void* self, iree_allocator_command_t command,
                     const void* params, void** pointer) -> iree_status_t {
    auto& state = *static_cast<AllocationState*>(self);
    const bool had_storage = *pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      state.requested_bytes +=
          static_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length;
      if (state.attempts++ == state.fail_at) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
    }
    auto allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, params, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        state.live -= had_storage;
      } else if (!had_storage) {
        ++state.live;
      }
    }
    return status;
  };
  for (bool encode : {true, false}) {
    for (size_t fail_at = 0;; ++fail_at) {
      SCOPED_TRACE(encode);
      SCOPED_TRACE(fail_at);
      AllocationState state{.fail_at = fail_at};
      const iree_allocator_t allocator{&state, allocate};
      iree_string_builder_reset(&output_);
      iree_status_t status =
          encode ? loom_serve_http_tokenize(tokenizer_,
                                            IREE_SV(R"({"content":"ab"})"),
                                            &output_, allocator)
                 : loom_serve_http_detokenize(tokenizer_,
                                              IREE_SV(R"({"tokens":[0,1]})"),
                                              &output_, allocator);
      EXPECT_EQ(state.live, 0u);
      if (iree_status_is_ok(status)) {
        EXPECT_EQ(state.attempts, fail_at);
        EXPECT_EQ(Output(),
                  encode ? R"({"tokens":[0,1]})" : R"({"content":"ab"})");
        RecordProperty(encode ? "encode_scratch_requested_bytes"
                              : "decode_scratch_requested_bytes",
                       static_cast<int>(state.requested_bytes));
        RecordProperty(encode ? "encode_scratch_allocations"
                              : "decode_scratch_allocations",
                       static_cast<int>(state.attempts));
        break;
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    }
  }
}

}  // namespace
