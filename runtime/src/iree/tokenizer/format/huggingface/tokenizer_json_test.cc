// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TokenizerJsonTest : public ::testing::Test {
 protected:
  iree_allocator_t allocator_ = iree_allocator_system();

  void ExpectDecodedAcrossCredit(const std::string& json,
                                 const std::vector<int32_t>& tokens,
                                 const std::string& expected) {
    iree_tokenizer_t* tokenizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json.data(), json.size()), allocator_,
        &tokenizer));
    std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> owner(
        tokenizer, iree_tokenizer_free);
    std::vector<char> output(expected.size() + 64);
    iree_host_size_t written = 0;
    IREE_ASSERT_OK(iree_tokenizer_decode(
        tokenizer, {tokens.size(), tokens.data()},
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(output.data(), expected.size()),
        allocator_, &written));
    EXPECT_EQ(std::string(output.data(), written), expected);
    for (size_t batch = 1; batch <= std::max(tokens.size(), size_t{1});
         ++batch) {
      SCOPED_TRACE(batch);
      iree_host_size_t size = 0;
      IREE_ASSERT_OK(
          iree_tokenizer_decode_state_calculate_size(tokenizer, &size));
      std::vector<uint8_t> storage(size);
      iree_tokenizer_decode_state_t* state = nullptr;
      IREE_ASSERT_OK(iree_tokenizer_decode_state_initialize(
          tokenizer, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
          iree_make_byte_span(storage.data(), storage.size()), &state));
      std::unique_ptr<iree_tokenizer_decode_state_t,
                      decltype(&iree_tokenizer_decode_state_deinitialize)>
          state_owner(state, iree_tokenizer_decode_state_deinitialize);
      const size_t credits[] = {0, 1, 2, 3, 4, 7, output.size()};
      size_t cursor = 0;
      size_t turn = 0;
      std::string actual;
      while (cursor < tokens.size()) {
        const size_t count = std::min(batch, tokens.size() - cursor);
        const size_t capacity = credits[turn++ % IREE_ARRAYSIZE(credits)];
        iree_host_size_t consumed = 0;
        IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
            state, {count, tokens.data() + cursor},
            iree_make_mutable_string_view(output.data(), capacity), &consumed,
            &written));
        if (capacity == output.size()) {
          ASSERT_GT(consumed + written, 0u);
        }
        cursor += consumed;
        actual.append(output.data(), written);
        EXPECT_EQ(expected.substr(0, actual.size()), actual);
      }
      bool complete = false;
      while (!complete) {
        const size_t capacity = credits[turn++ % IREE_ARRAYSIZE(credits)];
        iree_status_t status = iree_tokenizer_decode_state_finalize(
            state, iree_make_mutable_string_view(output.data(), capacity),
            &written);
        complete = iree_status_is_ok(status);
        if (!complete) {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
          if (capacity == output.size()) {
            ASSERT_GT(written, 0u);
          }
        }
        actual.append(output.data(), written);
        EXPECT_EQ(expected.substr(0, actual.size()), actual);
      }
      EXPECT_EQ(actual, expected);
    }
  }
};

//===----------------------------------------------------------------------===//
// Top-Level Validation Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, EmptyInputError) {
  iree_string_view_t json = iree_string_view_empty();
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, MissingModelError) {
  // No model field.
  iree_string_view_t json = IREE_SV(R"({"version": "1.0"})");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, UnknownTopLevelKeyError) {
  // Unknown key at top level should fail.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {}, "merges": []},
    "unknown_field": true
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InvalidVersionError) {
  // Unsupported version should fail.
  iree_string_view_t json = IREE_SV(R"({
    "version": "2.0",
    "model": {"type": "BPE", "vocab": {}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, UnsupportedModelTypeError) {
  // Unsupported model type should fail.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "SomeNewType", "vocab": {}}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Model Type Detection Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, ExplicitBPEType) {
  // Explicit BPE type with minimal valid model.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  // Verify model and vocab were set.
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, ExplicitWordPieceType) {
  // Explicit WordPiece type with valid vocab.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "WordPiece", "vocab": {"[UNK]": 0, "a": 1},
              "unk_token": "[UNK]"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, ExplicitUnigramType) {
  // Explicit Unigram type with valid vocab.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "Unigram",
              "vocab": [["<unk>", -10.0], ["a", -1.0], ["b", -1.5]],
              "unk_id": 0}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InferBPEFromMerges) {
  // No type field, but has merges -> infer BPE. Includes valid vocab/merges.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  // Verify model and vocab were set.
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InferWordPieceFromPrefix) {
  // No type field, but has continuing_subword_prefix -> infer WordPiece.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": {"[UNK]": 0, "a": 1, "##b": 2},
              "continuing_subword_prefix": "##"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, InferUnigramFromUnkId) {
  // No type field, but has unk_id -> infer Unigram.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": [["<unk>", -10.0], ["hello", -2.0]], "unk_id": 0}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.model, nullptr);
  EXPECT_NE(builder.vocab, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, UnigramScorePrecision) {
  // Exact IDs from tokenizers 0.23.3. The first case loses the winning path if
  // imported scores narrow to float; the second loses it if sums use float.
  const struct {
    // Serialized model with the score boundary under test.
    const char* json;
    // Input whose competing paths differ at that boundary.
    const char* text;
    // Complete expected token sequence.
    std::vector<int32_t> expected;
  } cases[] = {
      {R"({"model":{"type":"Unigram","unk_id":0,"vocab":[
        ["<unk>",0],["a",-1],["b",-1],["ab",-2.00000001]]}})",
       "ab",
       {1, 2}},
      {R"({"model":{"type":"Unigram","unk_id":0,"vocab":[
        ["<unk>",0],["a",-16777216],["b",-1],["c",-1],["bc",-2]]}})",
       "abc",
       {1, 4}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.text);
    iree_tokenizer_t* tokenizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_cstring_view(test.json), allocator_, &tokenizer));
    std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> owner(
        tokenizer, iree_tokenizer_free);
    int32_t tokens[3];
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode(
        tokenizer, iree_make_cstring_view(test.text),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(tokens, nullptr, nullptr,
                                         IREE_ARRAYSIZE(tokens)),
        allocator_, &count));
    EXPECT_EQ(std::vector<int32_t>(tokens, tokens + count), test.expected);
  }
}

TEST_F(TokenizerJsonTest, CannotInferTypeError) {
  // No type field and cannot infer from structure.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"vocab": {}}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Version Handling Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, Version10Accepted) {
  // Version 1.0 should be accepted.
  iree_string_view_t json = IREE_SV(R"({
    "version": "1.0",
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NoVersionAccepted) {
  // Missing version field should be accepted (optional).
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_OK(status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Section Integration Tests
//
// Verify that the orchestrator correctly handles each optional section:
// finding the key, handling null/missing, and passing values to component
// parsers. Exhaustive per-component tests are in the respective *_json_test
// files.
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, NullNormalizer) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "normalizer": null
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_EQ(builder.normalizer, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NormalizerSuccess) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "normalizer": {"type": "Lowercase"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.normalizer, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NormalizerErrorPropagation) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "normalizer": {"type": "SomeUnknownNormalizer"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NullPreTokenizer) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": null
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_EQ(builder.segmenter, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, PreTokenizerSuccess) {
  // Uses Split (not Metaspace) since Metaspace requires the space→replacement
  // normalizer which is gated as UNIMPLEMENTED.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Removed", "invert": false}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.segmenter, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, MetaspaceSynthesizesNormalizer) {
  // A Metaspace pre_tokenizer synthesizes both replace (space→▁) and prepend
  // (▁ prefix) normalizers. Verify parsing succeeds and the normalizer is set.
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.segmenter, nullptr);
  EXPECT_NE(builder.normalizer, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, PreTokenizerErrorPropagation) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "pre_tokenizer": {"type": "SomeUnimplementedType"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, NullDecoder) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "decoder": null
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.decoder, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, DefaultAndExplicitJoiningDecodeAcrossCredit) {
  struct Case {
    // Optional complete decoder field, including its comma.
    const char* field;
    // Whether the format joins logical tokens with spaces.
    bool spaced;
  };
  const Case cases[] = {
      {"", true},
      {R"(,"decoder":null)", true},
      {R"(,"decoder":{"type":"Fuse"})", false},
      {R"(,"decoder":{"type":"Sequence","decoders":[]})", false},
      {R"(,"decoder":{"type":"Sequence","decoders":[{"type":"Fuse"}]})", false},
  };
  const std::array<int32_t, 7> tokens = {4, 0, 1, 4, 0, 3, 2};
  for (const auto& test : cases) {
    SCOPED_TRACE(test.field);
    std::string json =
        R"({"model":{"type":"BPE","vocab":{"":0,"a":1,"b":2,"\ud83d\udc08":3,"[skip]":4},"merges":[]},)"
        R"("added_tokens":[{"id":4,"content":"[skip]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}])";
    json += test.field;
    json += "}";
    iree_tokenizer_t* tokenizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json.data(), json.size()), allocator_,
        &tokenizer));
    std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> owner(
        tokenizer, iree_tokenizer_free);
    const std::string expected = test.spaced ? " a  \xf0\x9f\x90\x88 b"
                                             : "a\xf0\x9f\x90\x88"
                                               "b";
    char output[128];
    iree_host_size_t written = 0;
    IREE_ASSERT_OK(iree_tokenizer_decode(
        tokenizer, {tokens.size(), tokens.data()},
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(output, sizeof(output)), allocator_,
        &written));
    EXPECT_EQ(std::string(output, written), expected);

    for (size_t batch_size = 1; batch_size <= tokens.size(); ++batch_size) {
      SCOPED_TRACE(batch_size);
      iree_host_size_t state_size = 0;
      IREE_ASSERT_OK(
          iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
      std::vector<uint8_t> storage(state_size);
      iree_tokenizer_decode_state_t* state = nullptr;
      IREE_ASSERT_OK(iree_tokenizer_decode_state_initialize(
          tokenizer, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
          iree_make_byte_span(storage.data(), storage.size()), &state));
      const size_t credits[] = {0, 1, 5, 6, sizeof(output)};
      size_t cursor = 0;
      size_t turn = 0;
      std::string actual;
      while (cursor < tokens.size()) {
        const size_t count = std::min(batch_size, tokens.size() - cursor);
        const size_t capacity = credits[turn++ % IREE_ARRAYSIZE(credits)];
        iree_host_size_t consumed = 0;
        IREE_ASSERT_OK(iree_tokenizer_decode_state_feed(
            state, {count, tokens.data() + cursor},
            iree_make_mutable_string_view(output, capacity), &consumed,
            &written));
        cursor += consumed;
        actual.append(output, written);
        if (capacity == sizeof(output)) {
          ASSERT_EQ(consumed, count);
        }
      }
      IREE_ASSERT_OK(iree_tokenizer_decode_state_finalize(
          state, iree_make_mutable_string_view(output, sizeof(output)),
          &written));
      actual.append(output, written);
      iree_tokenizer_decode_state_deinitialize(state);
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(TokenizerJsonTest, FusedStripPreservesInteriorTokenSpaces) {
  const std::string json = R"({
    "model":{"type":"BPE","vocab":{
      "":0,"\u2581Good":1,"\u2581day":2,".":3,
      "<0x20>":4,"<0xC3>":5,"<0xA9>":6,"[skip]":7},"merges":[]},
    "added_tokens":[{"id":7,"content":"[skip]","single_word":false,
      "lstrip":false,"rstrip":false,"normalized":false,"special":true}],
    "decoder":{"type":"Sequence","decoders":[
      {"type":"Replace","pattern":{"String":"\u2581"},"content":" "},
      {"type":"ByteFallback"},
      {"type":"Sequence","decoders":[{"type":"Fuse"},
        {"type":"Strip","content":" ","start":1,"stop":0}]},
      {"type":"Fuse"}]}})";
  ExpectDecodedAcrossCredit(json, {7, 0, 1, 0, 7, 2, 3}, "Good day.");
  ExpectDecodedAcrossCredit(json, {0, 4, 5, 6, 2}, "é day");
  ExpectDecodedAcrossCredit(json, {0, 7}, "");
}

TEST_F(TokenizerJsonTest, FusedStripCountsAndReplacementCrossTokenBoundaries) {
  const std::string json = R"({
    "model":{"type":"BPE","vocab":{
      "":0," ":1,"Good ":2,"da":3,"y ":4},"merges":[]},
    "decoder":{"type":"Sequence","decoders":[
      {"type":"Fuse"},
      {"type":"Strip","content":" ","start":2,"stop":2},
      {"type":"Replace","pattern":{"String":"day"},"content":"sunny day"}]}})";
  ExpectDecodedAcrossCredit(json, {0, 1, 0, 1, 2, 3, 4, 1, 1},
                            "Good sunny day ");
  ExpectDecodedAcrossCredit(json, {0, 1, 1, 1}, "");
}

TEST_F(TokenizerJsonTest, FusedTransformRetainsLongTokensAndFinalUtf8Repair) {
  const std::string long_token = std::string(600, 'x');
  const std::string json =
      R"({"model":{"type":"BPE","vocab":{"<0xC3>":0,")" + long_token +
      R"(":1},"merges":[]},"decoder":{"type":"Sequence","decoders":[
        {"type":"ByteFallback"},{"type":"Fuse"},
        {"type":"Replace","pattern":{"String":"�"},"content":"repaired"}]}})";
  ExpectDecodedAcrossCredit(json, {1, 0}, long_token + "repaired");
}

TEST_F(TokenizerJsonTest, FusedDecoderFamilyUsesOneLogicalString) {
  struct Case {
    // Authored decoder program after an explicit Fuse.
    const char* decoder;
    // Token IDs split at semantic boundaries of the transform.
    std::vector<int32_t> tokens;
    // Exact final UTF-8 output from the reference composition.
    const char* expected;
  };
  const Case cases[] = {
      {R"({"type":"Metaspace","replacement":"\u2581","prepend_scheme":"always"})",
       {1, 2},
       "Goodday"},
      {R"({"type":"Metaspace","replacement":"\u2581","prepend_scheme":"first"})",
       {1, 2},
       "Goodday"},
      {R"({"type":"Metaspace","replacement":"\u2581","prepend_scheme":"never"})",
       {1, 2},
       " Good day"},
      {R"({"type":"WordPiece","prefix":"##","cleanup":false})",
       {14, 3},
       "##Good ."},
      {R"({"type":"WordPiece","prefix":"##","cleanup":true})",
       {14, 4, 5, 3, 15},
       "##Good don't.!"},
      {R"({"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":false})",
       {12, 9, 9, 13, 16, 17},
       "xx|"},
      {R"({"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":true})",
       {12, 9, 9, 13, 16, 17},
       "xx "},
      {R"({"type":"ByteFallback"})", {6, 7}, "a"},
      {R"({"type":"ByteFallback"})", {6, 8}, "�"},
      {R"({"type":"ByteFallback"})", {6, 7, 9}, "<0x61>x"},
      {R"({"type":"ByteFallback"})", {6, 7, 6, 8}, "<0x61><0xC3>"},
      {R"({"type":"ByteFallback"})", {6}, "<0x"},
      {R"({"type":"ByteFallback"})", {0}, ""},
      {R"({"type":"Sequence","decoders":[{"type":"ByteFallback"},
        {"type":"Strip","content":" ","start":1,"stop":0}]})",
       {6, 20},
       ""},
      {R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false})",
       {10, 11},
       "é"},
      {R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false})",
       {10},
       "�"},
      {R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false})",
       {18, 19},
       " Good day"},
  };
  const std::string prefix = R"({
    "model":{"type":"BPE","vocab":{
      "":0,"\u2581Good":1,"\u2581day":2," .":3," do ":4,"not":5,
      "<0x":6,"61>":7,"C3>":8,"x":9,"Ã":10,"©":11,"<pad>":12,
      "|":13,"##Good":14," !":15,"<pa":16,"d>":17,
      "ĠGood":18,"Ġday":19,"20>":20},"merges":[]},
    "decoder":{"type":"Sequence","decoders":[{"type":"Fuse"},)";
  for (const auto& test : cases) {
    SCOPED_TRACE(test.decoder);
    ExpectDecodedAcrossCredit(prefix + test.decoder + "]}}", test.tokens,
                              test.expected);
  }
}

TEST_F(TokenizerJsonTest, ByteLevelIntroducesFusionWithoutExplicitFuse) {
  ExpectDecodedAcrossCredit(R"({
    "model":{"type":"BPE","vocab":{"ĠGood":0,"Ġday":1},"merges":[]},
    "decoder":{"type":"Sequence","decoders":[
      {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false},
      {"type":"Strip","content":" ","start":1,"stop":0}]}})",
                            {0, 1}, "Good day");
}

TEST_F(TokenizerJsonTest, DecoderSuccess) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "decoder": {"type": "ByteFallback"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(json, &builder));
  EXPECT_NE(builder.decoder, nullptr);

  iree_tokenizer_builder_deinitialize(&builder);
}

TEST_F(TokenizerJsonTest, DecoderErrorPropagation) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "decoder": {"type": "UnknownDecoder"}
  })");
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(allocator_, &builder);

  iree_status_t status = iree_tokenizer_parse_huggingface_json(json, &builder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);

  iree_tokenizer_builder_deinitialize(&builder);
}

//===----------------------------------------------------------------------===//
// Added Token Flag Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, NonSpecialRawAddedTokens) {
  // Qwen reasoning markers are added tokens without either the special or
  // normalized flag. They match before segmentation but survive decode's
  // special-token filtering.
  const iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"a": 0, "b": 1, "<": 2, ">": 3, "/": 4,
                "t": 5, "h": 6, "i": 7, "n": 8, "k": 9},
      "merges": []
    },
    "added_tokens": [
      {"id": 10, "content": "<think>", "special": false,
       "normalized": false, "single_word": false,
       "lstrip": false, "rstrip": false},
      {"id": 11, "content": "</think>", "special": false,
       "normalized": false, "single_word": false,
       "lstrip": false, "rstrip": false},
      {"id": 12, "content": "<end>", "special": true,
       "normalized": false, "single_word": false,
       "lstrip": false, "rstrip": false}
    ],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false,
                      "trim_offsets": false, "use_regex": true},
    "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": false, "use_regex": true}
  })");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  int32_t tokens[32] = {};
  iree_host_size_t count = 0;
  IREE_EXPECT_OK(iree_tokenizer_encode(
      tokenizer, IREE_SV("a<think>b</think><end>"),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens, nullptr, nullptr, 32),
      allocator_, &count));
  EXPECT_EQ(count, 5u);
  const int32_t expected[] = {0, 10, 1, 11, 12};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_EQ(tokens[i], expected[i]);
  }
  char text[64] = {};
  iree_host_size_t length = 0;
  IREE_EXPECT_OK(iree_tokenizer_decode(
      tokenizer, iree_tokenizer_make_token_id_list(expected, 5),
      IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      iree_make_mutable_string_view(text, sizeof(text)), allocator_, &length));
  EXPECT_EQ(std::string(text, length), "a<think>b</think>");
  iree_tokenizer_free(tokenizer);
}

// Tests that added_token flags (lstrip, rstrip, single_word) load successfully.
// These flags control matching behavior for special tokens.

TEST_F(TokenizerJsonTest, AddedTokenLstripFlag) {
  // Minimal tokenizer JSON with an added token that has lstrip: true.
  // The lstrip flag means the token only matches when preceded by whitespace
  // or at the start of input.
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": true,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, AddedTokenRstripFlag) {
  // The rstrip flag indicates trailing whitespace should be stripped after
  // matching (post-processing). It does not affect match eligibility.
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": true,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

// Verifies that back-to-back special tokens are both emitted. Previously,
// when the pipeline had buffered content, the second match would overwrite
// pending_special_token before the first was emitted, dropping it.
TEST_F(TokenizerJsonTest, BackToBackSpecialTokens) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 1, "world": 2},
      "merges": []
    },
    "added_tokens": [{
      "id": 100,
      "content": "<|user|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }, {
      "id": 101,
      "content": "<|end|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }, {
      "id": 102,
      "content": "<|assistant|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": null
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  auto encode = [&](const char* text) -> std::vector<int32_t> {
    std::vector<int32_t> tokens(64);
    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode(tokenizer, iree_make_cstring_view(text),
                              IREE_TOKENIZER_ENCODE_FLAG_NONE,
                              iree_tokenizer_make_token_output(
                                  tokens.data(), NULL, NULL, tokens.size()),
                              allocator_, &token_count));
    tokens.resize(token_count);
    return tokens;
  };

  // Two special tokens with no text between them.
  {
    auto tokens = encode("<|end|><|assistant|>");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 101);  // <|end|>
    EXPECT_EQ(tokens[1], 102);  // <|assistant|>
  }

  // Text, then two back-to-back special tokens, then more text.
  {
    auto tokens = encode("<|user|>hello<|end|><|assistant|>world<|end|>");
    ASSERT_EQ(tokens.size(), 6u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello
    EXPECT_EQ(tokens[2], 101);  // <|end|>
    EXPECT_EQ(tokens[3], 102);  // <|assistant|>
    EXPECT_EQ(tokens[4], 2);    // world
    EXPECT_EQ(tokens[5], 101);  // <|end|>
  }

  // Three back-to-back special tokens.
  {
    auto tokens = encode("<|end|><|user|><|assistant|>");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], 101);  // <|end|>
    EXPECT_EQ(tokens[1], 100);  // <|user|>
    EXPECT_EQ(tokens[2], 102);  // <|assistant|>
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, MaxSpecialTokenCountNoPostProcessor) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
  })");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  EXPECT_EQ(iree_tokenizer_max_special_token_count(tokenizer), 0u);
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, MaxSpecialTokenCountWithPostProcessor) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
    "added_tokens": [
      {"id": 1, "content": "<|bos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 2, "content": "<|eos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|eos|>", "type_id": 0}}
      ],
      "pair": [],
      "special_tokens": {
        "<|bos|>": {"id": "<|bos|>", "ids": [1], "tokens": ["<|bos|>"]},
        "<|eos|>": {"id": "<|eos|>", "ids": [2], "tokens": ["<|eos|>"]}
      }
    }
  })");
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  EXPECT_EQ(iree_tokenizer_max_special_token_count(tokenizer), 2u);
  iree_tokenizer_free(tokenizer);
}

// Verifies end-to-end rstrip behavior: the token matches regardless of what
// follows, and trailing whitespace after the token is consumed (not passed to
// BPE). See:
// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/tokenizer/added_vocabulary.rs
TEST_F(TokenizerJsonTest, RstripTokenMatchAndWhitespaceConsumption) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"hello": 1, "world": 2},
      "merges": []
    },
    "added_tokens": [{
      "id": 100,
      "content": "<|user|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": true,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": null
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  auto encode = [&](const char* text) -> std::vector<int32_t> {
    std::vector<int32_t> tokens(64);
    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(
        iree_tokenizer_encode(tokenizer, iree_make_cstring_view(text),
                              IREE_TOKENIZER_ENCODE_FLAG_NONE,
                              iree_tokenizer_make_token_output(
                                  tokens.data(), NULL, NULL, tokens.size()),
                              allocator_, &token_count));
    tokens.resize(token_count);
    return tokens;
  };

  // rstrip token followed by text (no whitespace): token must match.
  {
    auto tokens = encode("<|user|>hello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello
  }

  // rstrip token followed by space + text: whitespace must be consumed.
  {
    auto tokens = encode("<|user|> hello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (space consumed by rstrip)
  }

  // rstrip token followed by multiple spaces + text.
  {
    auto tokens = encode("<|user|>   hello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (spaces consumed by rstrip)
  }

  // rstrip token followed by tab + text.
  {
    auto tokens = encode("<|user|>\thello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (tab consumed by rstrip)
  }

  // rstrip token followed by newline + text.
  {
    auto tokens = encode("<|user|>\nhello");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], 100);  // <|user|>
    EXPECT_EQ(tokens[1], 1);    // hello (newline consumed by rstrip)
  }

  // rstrip token followed by only whitespace (no text after).
  {
    auto tokens = encode("<|user|>   ");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], 100);  // <|user|> (whitespace consumed, nothing left)
  }

  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, AddedTokenSingleWordFlag) {
  // The single_word flag means the token only matches when surrounded by
  // word boundaries (whitespace or punctuation).
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": true,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

// Special tokens with normalized=true are matched AFTER the normalizer runs.
// This is useful for tokens that should match normalized text (e.g., a token
// "yesterday" with a lowercasing normalizer would match input "Yesterday").
TEST_F(TokenizerJsonTest, AddedTokenNormalizedFlag) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

// Tests that the default case (all flags false except special) loads fine.
TEST_F(TokenizerJsonTest, AddedTokenDefaultFlags) {
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {"<|test|>": 0},
      "merges": []
    },
    "added_tokens": [{
      "id": 0,
      "content": "<|test|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// High-Level API and End-to-End Tests
//===----------------------------------------------------------------------===//

TEST_F(TokenizerJsonTest, HighLevelApiEmptyError) {
  // High-level API should fail on empty input.
  iree_string_view_t json = iree_string_view_empty();
  iree_tokenizer_t* tokenizer = nullptr;

  iree_status_t status =
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(tokenizer, nullptr);
}

// Tests that consecutive newlines are tokenized separately when followed by
// text, matching tiktoken behavior. The GPT-2 regex `\s+(?!\S)|\s+` should
// produce separate matches for each newline when followed by non-whitespace.
TEST_F(TokenizerJsonTest, ConsecutiveNewlinesBeforeText) {
  // Minimal GPT-2 style config with ByteLevel pre_tokenizer.
  // Ċ = U+010A = ByteLevel encoding of newline (0x0A).
  // The vocab has both Ċ (single newline) and ĊĊ (double newline).
  // Merge "Ċ Ċ" exists, but it should NOT apply when newlines are in
  // separate segments (regex matches them separately before text).
  iree_string_view_t json = IREE_SV(R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "Ċ": 0,
        "ĊĊ": 1,
        "A": 2
      },
      "merges": ["Ċ Ċ"]
    },
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
  })");

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(
      iree_tokenizer_from_huggingface_json(json, allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Encode "\n\nA" - two newlines followed by 'A'.
  // Expected: [Ċ, Ċ, A] = [0, 0, 2] (separate newlines)
  // Bug:      [ĊĊ, A] = [1, 2] (merged newlines)
  const char* text = "\n\nA";
  std::vector<int32_t> tokens(16);
  iree_host_size_t token_count = 0;

  IREE_ASSERT_OK(iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      allocator_, &token_count));

  tokens.resize(token_count);

  // Should produce 3 tokens: [0, 0, 2] = [Ċ, Ċ, A]
  // NOT 2 tokens: [1, 2] = [ĊĊ, A]
  ASSERT_EQ(token_count, 3u)
      << "Expected 3 tokens [Ċ, Ċ, A], got " << token_count;
  EXPECT_EQ(tokens[0], 0);  // First Ċ (newline)
  EXPECT_EQ(tokens[1], 0);  // Second Ċ (newline)
  EXPECT_EQ(tokens[2], 2);  // A

  iree_tokenizer_free(tokenizer);
}

// Minimal GPT-2 ByteLevel BPE config shared by ring buffer edge case tests.
// Vocab: Ċ=0, ĊĊ=1, A=2, B=3. Merge: "Ċ Ċ".
const char kMinimalGPT2Config[] = R"({
  "model": {
    "type": "BPE",
    "vocab": {
      "\u010a": 0,
      "\u010a\u010a": 1,
      "A": 2,
      "B": 3
    },
    "merges": ["\u010a \u010a"]
  },
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true}
})";

// Helper: encodes text with the minimal GPT-2 config and returns token IDs.
std::vector<int32_t> EncodeWithMinimalGPT2(const char* text) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_tokenizer_t* tokenizer = nullptr;
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator, &tokenizer);
  IREE_CHECK_OK(status);

  std::vector<int32_t> tokens(64);
  iree_host_size_t token_count = 0;
  IREE_CHECK_OK(iree_tokenizer_encode(
      tokenizer, iree_make_cstring_view(text), IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(tokens.data(), NULL, NULL,
                                       tokens.size()),
      allocator, &token_count));

  tokens.resize(token_count);
  iree_tokenizer_free(tokenizer);
  return tokens;
}

// Tests single newline before text. The GPT-2 regex emits one whitespace
// segment and holds the trailing letter pending. Model consumes the segment,
// ring empties, finalize must still produce the letter.
TEST_F(TokenizerJsonTest, SingleNewlineBeforeText) {
  // Segments: ["\n"] emitted, ["A"] pending.
  // Ring empties after model consumes 1 segment → tests ring reset guard.
  auto tokens = EncodeWithMinimalGPT2("\nA");
  ASSERT_EQ(tokens.size(), 2u) << "Expected [Ċ, A]";
  EXPECT_EQ(tokens[0], 0);  // Ċ
  EXPECT_EQ(tokens[1], 2);  // A
}

// Tests triple newlines before text. The greedy \s+(?!\S) match consumes
// the first two newlines as one segment (lookahead passes at second \n),
// then the third newline falls through to \s+ (lookahead fails at 'A').
TEST_F(TokenizerJsonTest, TripleNewlinesBeforeText) {
  // Segments: ["\n\n", "\n"] emitted, ["A"] pending.
  // The "\n\n" segment gets BPE-merged: Ċ+Ċ → ĊĊ = token 1.
  auto tokens = EncodeWithMinimalGPT2("\n\n\nA");
  ASSERT_EQ(tokens.size(), 3u) << "Expected [ĊĊ, Ċ, A]";
  EXPECT_EQ(tokens[0], 1);  // ĊĊ (merged pair)
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 2);  // A
}

// Tests text-newline-text pattern. The regex matches "A" as a ?\p{L}+ segment,
// "\n" as \s+ (fallback), then holds trailing "A" pending. Model consumes
// both completed segments, ring empties.
TEST_F(TokenizerJsonTest, TextNewlineText) {
  // Segments: ["A", "\n"] emitted, ["A"] pending.
  auto tokens = EncodeWithMinimalGPT2("A\nA");
  ASSERT_EQ(tokens.size(), 3u) << "Expected [A, Ċ, A]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 2);  // A
}

// Tests text-double-newline-text. Both newlines become separate segments
// (first passes lookahead, second uses fallback). Model consumes 3 segments
// before ring empties with pending trailing text.
TEST_F(TokenizerJsonTest, TextDoubleNewlineText) {
  // Segments: ["A", "\n", "\n"] emitted, ["A"] pending.
  auto tokens = EncodeWithMinimalGPT2("A\n\nA");
  ASSERT_EQ(tokens.size(), 4u) << "Expected [A, Ċ, Ċ, A]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 0);  // Ċ
  EXPECT_EQ(tokens[3], 2);  // A
}

// Tests multiple words separated by newlines. Each word and each newline
// becomes a separate segment; the trailing word stays pending until finalize.
TEST_F(TokenizerJsonTest, MultipleWordsWithNewlines) {
  // Segments: ["A", "\n", "B", "\n"] emitted, ["A"] pending.
  auto tokens = EncodeWithMinimalGPT2("A\nB\nA");
  ASSERT_EQ(tokens.size(), 5u) << "Expected [A, Ċ, B, Ċ, A]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 0);  // Ċ
  EXPECT_EQ(tokens[2], 3);  // B
  EXPECT_EQ(tokens[3], 0);  // Ċ
  EXPECT_EQ(tokens[4], 2);  // A
}

// Tests input that produces 0 segments during process (entire input is one
// pending match). The ?\p{L}+ pattern starts matching but can't terminate
// without seeing a non-letter character. All tokens come from finalize.
TEST_F(TokenizerJsonTest, SingleLetterAllPending) {
  // Segments: [] emitted, ["A"] pending.
  // Finalize produces the only segment. Tests the chunk_base underflow path
  // where match.start(0) < chunk_base(1), wrapping correctly via adjustment.
  auto tokens = EncodeWithMinimalGPT2("A");
  ASSERT_EQ(tokens.size(), 1u) << "Expected [A]";
  EXPECT_EQ(tokens[0], 2);  // A
}

// Tests multi-letter input with 0 segments emitted (one pending match).
// The regex ?\p{L}+ greedily matches all letters without terminating.
TEST_F(TokenizerJsonTest, MultiLetterAllPending) {
  // Segments: [] emitted, ["AB"] pending.
  // Finalize produces one segment "AB". BPE encodes as [A, B] (no merge).
  auto tokens = EncodeWithMinimalGPT2("AB");
  ASSERT_EQ(tokens.size(), 2u) << "Expected [A, B]";
  EXPECT_EQ(tokens[0], 2);  // A
  EXPECT_EQ(tokens[1], 3);  // B
}

// Tests the streaming API with ring-draining input patterns. Verifies that
// the ring reset guard works correctly when tokens are collected incrementally
// via the state-based API (matching what iree_tokenizer_encode does
// internally).
TEST_F(TokenizerJsonTest, StreamingNewlinesBeforeText) {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Allocate encode state with recommended transform buffer size.
  iree_host_size_t state_size = 0;
  IREE_ASSERT_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  std::vector<uint8_t> state_storage(state_size);
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_recommended_size(3);
  std::vector<uint8_t> transform_buffer(transform_size);
  iree_tokenizer_encode_state_t* state = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(state_storage.data(), state_storage.size()),
      iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(),
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state));

  // Feed the full input, collecting tokens.
  iree_string_view_t input = IREE_SVL("\n\nA");
  iree_host_size_t total_tokens = 0;
  int32_t token_buffer[16];

  while (input.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
        state, input,
        iree_tokenizer_make_token_output(token_buffer + total_tokens, NULL,
                                         NULL, 16 - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
    input.data += bytes_consumed;
    input.size -= bytes_consumed;
    if (bytes_consumed == 0) {
      break;
    }
  }

  // Finalize to flush pending segments (single call like one-shot API).
  iree_host_size_t final_tokens = 0;
  IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
      state,
      iree_tokenizer_make_token_output(token_buffer + total_tokens, NULL, NULL,
                                       16 - total_tokens),
      &final_tokens));
  total_tokens += final_tokens;

  // Should produce [Ċ, Ċ, A] = [0, 0, 2].
  ASSERT_EQ(total_tokens, 3u) << "Expected [Ċ, Ċ, A]";
  EXPECT_EQ(token_buffer[0], 0);  // Ċ
  EXPECT_EQ(token_buffer[1], 0);  // Ċ
  EXPECT_EQ(token_buffer[2], 2);  // A

  iree_tokenizer_encode_state_deinitialize(state);
  iree_tokenizer_free(tokenizer);
}

//===----------------------------------------------------------------------===//
// Streaming Completion Tests
//===----------------------------------------------------------------------===//

// Helper: creates a streaming encode state for a tokenizer.
// Returns allocated state; caller must deinitialize + free storage vectors.
struct StreamingEncodeContext {
  // Stream state borrowing the storage below.
  iree_tokenizer_encode_state_t* state;
  // Owned storage for the encoder and component state.
  std::vector<uint8_t> state_storage;
  // Owned intermediate normalized bytes.
  std::vector<uint8_t> transform_buffer;
};

StreamingEncodeContext CreateStreamingState(
    const iree_tokenizer_t* tokenizer, iree_host_size_t text_hint,
    iree_tokenizer_encode_flags_t flags) {
  StreamingEncodeContext ctx = {};
  iree_host_size_t state_size = 0;
  IREE_CHECK_OK(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  ctx.state_storage.resize(state_size);
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_recommended_size(text_hint);
  ctx.transform_buffer.resize(transform_size);
  IREE_CHECK_OK(iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span(ctx.state_storage.data(), ctx.state_storage.size()),
      iree_make_byte_span(ctx.transform_buffer.data(),
                          ctx.transform_buffer.size()),
      iree_tokenizer_offset_run_list_empty(), flags, &ctx.state));
  return ctx;
}

// Feeds all bytes of |text| into the state, collecting emitted tokens.
// Returns the total tokens emitted during feed (not finalize).
iree_host_size_t FeedAll(iree_tokenizer_encode_state_t* state,
                         iree_string_view_t text,
                         iree_tokenizer_token_id_t* token_buffer,
                         iree_host_size_t token_capacity) {
  iree_host_size_t total_tokens = 0;
  while (text.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    IREE_CHECK_OK(iree_tokenizer_encode_state_feed(
        state, text,
        iree_tokenizer_make_token_output(token_buffer + total_tokens, NULL,
                                         NULL, token_capacity - total_tokens),
        &bytes_consumed, &token_count));
    total_tokens += token_count;
    text.data += bytes_consumed;
    text.size -= bytes_consumed;
    if (bytes_consumed == 0 && token_count == 0) {
      break;
    }
  }
  return total_tokens;
}

TEST_F(TokenizerJsonTest, StreamingCompletionWithOneTokenCredit) {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(kMinimalGPT2Config), allocator_, &tokenizer));
  struct Case {
    // Complete input through the actual loaded GPT-2 pipeline.
    const char* input;
    // Exact independently determined token sequence.
    std::vector<int32_t> expected;
  } cases[] = {{"\n\nA", {0, 0, 2}},
               {"\nA", {0, 2}},
               {"A\nB\nA", {2, 0, 3, 0, 2}},
               {"A", {2}},
               {"AB", {2, 3}}};
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.input);
    auto ctx = CreateStreamingState(tokenizer, strlen(test_case.input),
                                    IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START);
    int32_t token_buffer[256];
    iree_host_size_t feed_tokens = FeedAll(
        ctx.state, iree_make_cstring_view(test_case.input), token_buffer, 256);
    std::vector<int32_t> actual(token_buffer, token_buffer + feed_tokens);
    do {
      int32_t token = -1;
      iree_host_size_t count = 0;
      IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
          ctx.state,
          iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
          &count));
      ASSERT_TRUE(count || !iree_tokenizer_encode_state_has_pending(ctx.state));
      if (count) {
        actual.push_back(token);
      }
    } while (iree_tokenizer_encode_state_has_pending(ctx.state));
    EXPECT_EQ(actual, test_case.expected);
    iree_tokenizer_encode_state_deinitialize(ctx.state);
  }
  iree_tokenizer_free(tokenizer);
}

TEST_F(TokenizerJsonTest, StreamingCompletionPreservesSpecialTokenOrder) {
  // Build a tokenizer with BOS/EOS postprocessor. Uses the minimal GPT-2 vocab
  // (Ċ=0, ĊĊ=1, A=2, B=3) plus BOS (4) and EOS (5) special tokens.
  const char* json = R"({
    "model": {
      "type": "BPE",
      "vocab": {
        "\u010a": 0,
        "\u010a\u010a": 1,
        "A": 2,
        "B": 3,
        "<|bos|>": 4,
        "<|eos|>": 5
      },
      "merges": ["\u010a \u010a"]
    },
    "added_tokens": [
      {"id": 4, "content": "<|bos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true},
      {"id": 5, "content": "<|eos|>", "single_word": false, "lstrip": false,
       "rstrip": false, "normalized": false, "special": true}
    ],
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|bos|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|eos|>", "type_id": 0}}
      ],
      "pair": [],
      "special_tokens": {
        "<|bos|>": {"id": "<|bos|>", "ids": [4], "tokens": ["<|bos|>"]},
        "<|eos|>": {"id": "<|eos|>", "ids": [5], "tokens": ["<|eos|>"]}
      }
    },
    "pre_tokenizer": null
  })";

  iree_tokenizer_t* tokenizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
      iree_make_cstring_view(json), allocator_, &tokenizer));
  ASSERT_NE(tokenizer, nullptr);

  // Encode with add_special_tokens to activate the postprocessor.
  auto ctx =
      CreateStreamingState(tokenizer, 3,
                           IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START |
                               IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS);

  int32_t token_buffer[256];
  iree_host_size_t feed_tokens =
      FeedAll(ctx.state, iree_make_cstring_view("AB"), token_buffer, 256);

  iree_host_size_t total = feed_tokens;
  do {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(
        ctx.state,
        iree_tokenizer_make_token_output(token_buffer + total, nullptr, nullptr,
                                         1),
        &count));
    ASSERT_TRUE(count || !iree_tokenizer_encode_state_has_pending(ctx.state));
    total += count;
  } while (iree_tokenizer_encode_state_has_pending(ctx.state));
  EXPECT_EQ(std::vector<int32_t>(token_buffer, token_buffer + total),
            (std::vector<int32_t>{4, 2, 3, 5}));
  EXPECT_EQ(iree_tokenizer_max_special_token_count(tokenizer), 2u);

  iree_tokenizer_encode_state_deinitialize(ctx.state);
  iree_tokenizer_free(tokenizer);
}

}  // namespace
