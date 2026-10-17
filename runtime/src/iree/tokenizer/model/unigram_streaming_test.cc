// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/model_test_util.h"
#include "iree/tokenizer/model/unigram.h"

namespace iree::tokenizer::testing {
namespace {

// Every arrival replaces the borrowed allocation. Only unreclaimed bytes remain
// available, so retained paths and output cannot borrow an earlier allocation.
void ExpectStream(iree_tokenizer_model_t* model, const std::string& text,
                  const std::vector<int32_t>& expected_ids,
                  const std::vector<iree_tokenizer_offset_t>& expected_offsets,
                  size_t input_width, size_t credit, bool ambiguous) {
  ScopedModelState owner(model);
  auto* state = owner.get();
  const size_t origin = 4096;
  size_t accepted = 0, available = 0;
  std::array<int32_t, 3> tokens;
  std::array<iree_tokenizer_offset_t, 3> offsets;
  std::vector<int32_t> actual_ids;
  std::vector<iree_tokenizer_offset_t> actual_offsets;
  auto output = iree_tokenizer_make_token_output(tokens.data(), offsets.data(),
                                                 nullptr, credit);
  auto collect = [&](size_t count) {
    actual_ids.insert(actual_ids.end(), tokens.begin(), tokens.begin() + count);
    actual_offsets.insert(actual_offsets.end(), offsets.begin(),
                          offsets.begin() + count);
    ASSERT_LE(actual_ids.size(), expected_ids.size());
    for (size_t i = 0; i < actual_ids.size(); ++i) {
      EXPECT_EQ(actual_ids[i], expected_ids[i]);
      EXPECT_EQ(actual_offsets[i].start, origin + expected_offsets[i].start);
      EXPECT_EQ(actual_offsets[i].end, origin + expected_offsets[i].end);
    }
  };
  while (accepted < text.size()) {
    available = std::min(text.size(), available + input_width);
    std::string borrowed = text.substr(accepted, available - accepted);
    iree_tokenizer_segment_t segment = {origin + accepted, origin + available};
    auto segments = iree_tokenizer_make_segment_list(&segment, 1);
    segments.buffer_offset = segment.start;
    segments.last_is_partial = true;
    size_t consumed = 0, count = 0;
    IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
        state, iree_make_const_byte_span(borrowed.data(), borrowed.size()),
        segments, output, &consumed, &count));
    EXPECT_EQ(consumed, 0u);
    if (ambiguous) {
      EXPECT_EQ(count, 0u);
    }
    collect(count);
    size_t reclaimed = iree_tokenizer_model_state_reclaim(state);
    ASSERT_LE(reclaimed, borrowed.size());
    ASSERT_GT(reclaimed + count, 0u);
    accepted += reclaimed;
  }
  size_t consumed = 0;
  do {
    iree_tokenizer_segment_t segment = {origin + accepted, origin + accepted};
    auto segments = iree_tokenizer_make_segment_list(&segment, 1);
    segments.buffer_offset = segment.start;
    size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
        state, iree_make_const_byte_span("", 0), segments, output, &consumed,
        &count));
    collect(count);
    ASSERT_GT(consumed + count, 0u);
  } while (!consumed);
  EXPECT_EQ(actual_ids, expected_ids);
  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state));
}

TEST(UnigramStreamingTest, AmbiguousPathsSurviveInputReclamation) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "<unk>", 0.0);
  builder.AddTokenWithScore(1, "a", -1.0);
  builder.AddTokenWithScore(2, "aa", -1.5);
  ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), 0, -11.5, IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK,
      iree_allocator_system(), &raw_model));
  ScopedModel model(raw_model);
  for (size_t length : {257, 258}) {
    std::vector<int32_t> ids;
    std::vector<iree_tokenizer_offset_t> offsets;
    size_t position = 0;
    if (length % 2) {
      ids.push_back(1);
      offsets.push_back({0, 1});
      position = 1;
    }
    for (; position < length; position += 2) {
      ids.push_back(2);
      offsets.push_back({position, position + 2});
    }
    for (size_t width : {1, 7, 31}) {
      for (size_t credit : {1, 3}) {
        SCOPED_TRACE(length);
        SCOPED_TRACE(width);
        SCOPED_TRACE(credit);
        ExpectStream(model.get(), std::string(length, 'a'), ids, offsets, width,
                     credit, /*ambiguous=*/true);
      }
    }
  }
}

TEST(UnigramStreamingTest, UnknownExpansionRetainsWholePieceSpans) {
  for (bool complete_bytes : {false, true}) {
    ScopedVocabBuilder builder;
    builder.AddTokenWithScore(0, "<unk>", 0.0);
    builder.AddTokenWithScore(1, "!", -1.0);
    builder.AddTokenWithScore(2, "<0xC3>", 0.0);
    builder.AddTokenWithScore(3, "<0xA9>", 0.0);
    if (complete_bytes) {
      builder.AddTokenWithScore(4, "<0xF0>", 0.0);
      builder.AddTokenWithScore(5, "<0x90>", 0.0);
      builder.AddTokenWithScore(6, "<0x80>", 0.0);
    }
    ScopedVocab vocab(builder.Build());
    for (bool fuse : {false, true}) {
      for (bool fallback : {false, true}) {
        iree_tokenizer_unigram_flags_t flags = 0;
        if (!fuse) {
          flags |= IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK;
        }
        if (!fallback) {
          flags |= IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK;
        }
        iree_tokenizer_model_t* raw_model = nullptr;
        IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
            vocab.get(), 0, -11.0, flags, iree_allocator_system(), &raw_model));
        ScopedModel model(raw_model);
        std::vector<int32_t> ids;
        std::vector<iree_tokenizer_offset_t> offsets;
        auto append = [&](std::initializer_list<int32_t> values, size_t start,
                          size_t end) {
          ids.insert(ids.end(), values);
          offsets.insert(offsets.end(), values.size(), {start, end});
        };
        if (fuse) {
          if (fallback && complete_bytes) {
            append({2, 3, 4, 5, 5, 6}, 0, 6);
          } else {
            append({0}, 0, 6);
          }
        } else {
          append(fallback ? std::initializer_list<int32_t>{2, 3}
                          : std::initializer_list<int32_t>{0},
                 0, 2);
          append(fallback && complete_bytes
                     ? std::initializer_list<int32_t>{4, 5, 5, 6}
                     : std::initializer_list<int32_t>{0},
                 2, 6);
        }
        append({1}, 6, 7);
        for (size_t width : {1, 3, 7}) {
          ExpectStream(model.get(), "é𐐀!", ids, offsets, width, 1,
                       /*ambiguous=*/false);
        }
      }
    }
  }
}

struct Workspace {
  // Allocation attempt to reject, or zero when growth is allowed.
  size_t fail_at = 0;
  // Allocation/reallocation attempts since construction.
  size_t attempts = 0;
  // Live workspace allocations; fixed state storage is separately owned.
  size_t live = 0;

  iree_allocator_t allocator() {
    return {this,
            [](void* self, iree_allocator_command_t command, const void* params,
               void** pointer) -> iree_status_t {
              auto& workspace = *static_cast<Workspace*>(self);
              bool freeing = command == IREE_ALLOCATOR_COMMAND_FREE;
              if (!freeing && ++workspace.attempts == workspace.fail_at) {
                return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
              }
              bool had_storage = *pointer != nullptr;
              auto system = iree_allocator_system();
              iree_status_t status =
                  system.ctl(system.self, command, params, pointer);
              if (iree_status_is_ok(status)) {
                if (freeing && had_storage) {
                  --workspace.live;
                }
                if (!freeing && !had_storage) {
                  ++workspace.live;
                }
              }
              return status;
            }};
  }
};

TEST(UnigramStreamingTest, GrowthFailureResetAndCancellationReleaseWorkspace) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "<unk>", 0.0);
  builder.AddTokenWithScore(1, "a", -1.0);
  builder.AddTokenWithScore(2, "aa", -1.5);
  ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), 0, -11.5, IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK,
      iree_allocator_system(), &raw_model));
  ScopedModel model(raw_model);
  for (char character : {'a', 'x'}) {
    for (size_t fail_at : {1, 2, 3}) {
      Workspace workspace;
      {
        ScopedModelState state(model.get(), workspace.allocator());
        std::string text(257, character);
        iree_tokenizer_segment_t segment = {0, text.size()};
        auto segments = iree_tokenizer_make_segment_list(&segment, 1);
        std::array<int32_t, 258> tokens;
        auto encode = [&]() {
          size_t consumed = 0, count = 0;
          return iree_tokenizer_model_state_encode(
              state.get(), iree_make_const_byte_span(text.data(), text.size()),
              segments,
              iree_tokenizer_make_token_output(tokens.data(), nullptr, nullptr,
                                               tokens.size()),
              &consumed, &count);
        };
        workspace.fail_at = fail_at;
        IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, encode());
        workspace.fail_at = 0;
        state.Reset();
        IREE_ASSERT_OK(encode());
        size_t attempts = workspace.attempts;
        state.Reset();
        IREE_ASSERT_OK(encode());
        EXPECT_EQ(workspace.attempts, attempts);
        // Cancel with unresolved paths/output after reclaiming all input.
        state.Reset();
        segments.last_is_partial = true;
        IREE_ASSERT_OK(encode());
        EXPECT_EQ(iree_tokenizer_model_state_reclaim(state.get()), text.size());
      }
      EXPECT_EQ(workspace.live, 0u);
    }
  }
}

TEST(UnigramStreamingTest, MalformedUtf8IsAnExplicitInputError) {
  ScopedVocabBuilder builder;
  builder.AddTokenWithScore(0, "<unk>", 0.0);
  ScopedVocab vocab(builder.Build());
  iree_tokenizer_model_t* raw_model = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_unigram_model_allocate(
      vocab.get(), 0, -10.0, 0, iree_allocator_system(), &raw_model));
  ScopedModel model(raw_model);
  ScopedModelState state(model.get());
  for (const char* text :
       {"\x80", "\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xc3"}) {
    state.Reset();
    auto input = iree_make_cstring_view(text);
    iree_tokenizer_segment_t segment = {0, input.size};
    int32_t token = -1;
    size_t consumed = 0, count = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_tokenizer_model_state_encode(
            state.get(), iree_make_const_byte_span(input.data, input.size),
            iree_tokenizer_make_segment_list(&segment, 1),
            iree_tokenizer_make_token_output(&token, nullptr, nullptr, 1),
            &consumed, &count));
    EXPECT_EQ(count, 0u);
  }
}

}  // namespace
}  // namespace iree::tokenizer::testing
