// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/tokenizer.h"

namespace {

struct ModelCase {
  // Serialized model; every case uses identity normalization.
  const char* model;
  // Expected IDs for the two letters in "aa" and the scalar in "é".
  std::array<int32_t, 3> ids;
};

const ModelCase kModels[] = {
    {R"({"type":"BPE","vocab":{"a":0,"é":1},"merges":[]})", {0, 0, 1}},
    {R"({"type":"Unigram","unk_id":0,"vocab":[["<unk>",0],["a",-1],["é",-1]]})",
     {1, 1, 2}},
    {R"({"type":"WordPiece","unk_token":"[UNK]","vocab":{"[UNK]":0,"a":1,"##a":2,"é":3}})",
     {1, 2, 3}},
};

// Collects the actual stream while input arrivals and output credit vary
// independently. An incomplete UTF-8 suffix stays offered until more arrives.
void ExpectStreamingOffsets(iree_tokenizer_encode_state_t* state,
                            const ModelCase& model, size_t input_width,
                            size_t output_capacity) {
  std::string text;
  std::vector<int32_t> expected_ids;
  std::vector<iree_tokenizer_offset_t> expected_offsets;
  for (int i = 0; i < 48; ++i) {
    size_t start = text.size();
    text += "aa é ";
    expected_ids.insert(expected_ids.end(), model.ids.begin(), model.ids.end());
    expected_offsets.insert(
        expected_offsets.end(),
        {{start, start + 1}, {start + 1, start + 2}, {start + 3, start + 5}});
  }
  std::array<int32_t, 8> tokens;
  std::array<iree_tokenizer_offset_t, 8> offsets;
  auto output = iree_tokenizer_make_token_output(tokens.data(), offsets.data(),
                                                 nullptr, output_capacity);
  auto no_credit_output = output;
  no_credit_output.capacity = 0;
  std::vector<int32_t> actual_ids;
  std::vector<iree_tokenizer_offset_t> actual_offsets;
  size_t position = 0;
  size_t available = 0;
  while (position < text.size()) {
    available = std::min(text.size(), available + input_width);
    while (position < available) {
      auto input =
          iree_make_string_view(text.data() + position, available - position);
      // With no output credit, accepted input stays owned by the encoder.
      iree_host_size_t consumed = 0, count = 0;
      IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(
          state, input, no_credit_output, &consumed, &count));
      ASSERT_EQ(count, 0u);
      ASSERT_LE(consumed, input.size);
      size_t buffered = consumed;
      position += consumed;
      input =
          iree_make_string_view(text.data() + position, available - position);
      IREE_ASSERT_OK(iree_tokenizer_encode_state_feed(state, input, output,
                                                      &consumed, &count));
      actual_ids.insert(actual_ids.end(), tokens.begin(),
                        tokens.begin() + count);
      actual_offsets.insert(actual_offsets.end(), offsets.begin(),
                            offsets.begin() + count);
      position += consumed;
      if (!buffered && !consumed && !count) {
        ASSERT_LT(available, text.size());
        break;
      }
    }
  }
  do {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(
        iree_tokenizer_encode_state_finalize(state, no_credit_output, &count));
    ASSERT_EQ(count, 0u);
    IREE_ASSERT_OK(iree_tokenizer_encode_state_finalize(state, output, &count));
    actual_ids.insert(actual_ids.end(), tokens.begin(), tokens.begin() + count);
    actual_offsets.insert(actual_offsets.end(), offsets.begin(),
                          offsets.begin() + count);
  } while (iree_tokenizer_encode_state_has_pending(state));
  EXPECT_EQ(actual_ids, expected_ids);
  ASSERT_EQ(actual_offsets.size(), expected_offsets.size());
  for (size_t i = 0; i < expected_offsets.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual_offsets[i].start, expected_offsets[i].start);
    EXPECT_EQ(actual_offsets[i].end, expected_offsets[i].end);
  }
}

TEST(TokenizerOffsetsTest, LogicalPositionsSurviveWrapAndReset) {
  for (const auto& model : kModels) {
    SCOPED_TRACE(model.model);
    std::string json = std::string("{\"model\":") + model.model +
                       R"(,"pre_tokenizer":{"type":"WhitespaceSplit"}})";
    iree_tokenizer_t* tokenizer = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_string_view(json.data(), json.size()),
        iree_allocator_system(), &tokenizer));
    std::unique_ptr<iree_tokenizer_t, decltype(&iree_tokenizer_free)> owner(
        tokenizer, iree_tokenizer_free);
    iree_host_size_t state_size = 0;
    IREE_ASSERT_OK(
        iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
    std::vector<uint8_t> storage(state_size);
    uint8_t transform[32];
    iree_tokenizer_offset_run_t runs[64];
    iree_tokenizer_encode_state_t* state = nullptr;
    auto flags = IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
    IREE_ASSERT_OK(iree_tokenizer_encode_state_initialize(
        tokenizer, iree_make_byte_span(storage.data(), storage.size()),
        iree_make_byte_span(transform, sizeof(transform)),
        {IREE_ARRAYSIZE(runs), runs}, flags, &state));
    std::unique_ptr<iree_tokenizer_encode_state_t,
                    decltype(&iree_tokenizer_encode_state_deinitialize)>
        state_owner(state, iree_tokenizer_encode_state_deinitialize);
    for (size_t input_width : {1, 7, 31}) {
      for (size_t output_capacity : {1, 3, 8}) {
        SCOPED_TRACE(input_width);
        SCOPED_TRACE(output_capacity);
        IREE_ASSERT_OK(iree_tokenizer_encode_state_reset(state, flags));
        ExpectStreamingOffsets(state, model, input_width, output_capacity);
      }
    }
  }
}

TEST(TokenizerOffsetsTest, ModelRetainsOffsetsAcrossBufferRelocation) {
  for (const auto& model : kModels) {
    SCOPED_TRACE(model.model);
    std::string json = std::string("{\"model\":") + model.model + "}";
    iree_tokenizer_builder_t builder;
    iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);
    IREE_ASSERT_OK(iree_tokenizer_parse_huggingface_json(
        iree_make_string_view(json.data(), json.size()), &builder));
    std::unique_ptr<iree_tokenizer_builder_t,
                    decltype(&iree_tokenizer_builder_deinitialize)>
        builder_owner(&builder, iree_tokenizer_builder_deinitialize);
    std::vector<uint8_t> storage(
        iree_tokenizer_model_state_size(builder.model));
    iree_tokenizer_model_state_t* state = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_model_state_initialize(
        builder.model, storage.data(), &state));
    std::unique_ptr<iree_tokenizer_model_state_t,
                    decltype(&iree_tokenizer_model_state_deinitialize)>
        state_owner(state, iree_tokenizer_model_state_deinitialize);
    iree_tokenizer_segment_t segment = {4096, 4098};
    auto segments = iree_tokenizer_make_segment_list(&segment, 1);
    const char* buffers[] = {"xaa", "zzzaa"};
    const size_t origins[] = {4095, 4093};
    for (size_t i = 0; i < 2; ++i) {
      segments.buffer_offset = origins[i];
      auto buffer = iree_make_cstring_view(buffers[i]);
      int32_t token = -1;
      iree_tokenizer_offset_t offset = {};
      iree_host_size_t consumed = 0, count = 0;
      IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
          state, iree_make_const_byte_span(buffer.data, buffer.size), segments,
          iree_tokenizer_make_token_output(&token, &offset, nullptr, 1),
          &consumed, &count));
      ASSERT_EQ(count, 1u);
      EXPECT_EQ(token, model.ids[i]);
      EXPECT_EQ(offset.start, 4096u + i);
      EXPECT_EQ(offset.end, 4097u + i);
      EXPECT_EQ(consumed, i == 1 ? 1u : 0u);
    }
  }
}

}  // namespace
