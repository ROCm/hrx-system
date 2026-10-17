// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/input.h"

#include <algorithm>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::string Read(const loom_serve_speech_input_block_t& block) {
  std::string result;
  for (const auto& span : block.spans) {
    if (span.data_length) {
      result.append(reinterpret_cast<const char*>(span.data), span.data_length);
    }
  }
  return result;
}

class SpeechInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &text_);
  }
  void TearDown() override {
    loom_serve_speech_input_release(input_);
    iree_string_builder_deinitialize(&text_);
  }
  iree_status_t Allocate(size_t capacity, uint64_t limit, bool complete) {
    return loom_serve_speech_input_allocate(capacity, limit, &text_, complete,
                                            iree_allocator_system(), &input_);
  }
  iree_status_t Write(iree_string_view_t text, size_t* accepted) {
    return loom_serve_speech_input_write(
        input_, iree_make_const_byte_span(text.data, text.size), accepted);
  }
  // Request builder, transferred only after successful finite admission.
  iree_string_builder_t text_ = {};
  // Serving owner of the input object under test.
  loom_serve_speech_input_t* input_ = nullptr;
};

TEST_F(SpeechInputTest, FiniteStorageTransfersAndWindowsEndAtScalars) {
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&text_, "abc🦜日z"));
  const char* storage = iree_string_builder_buffer(&text_);
  IREE_ASSERT_OK(Allocate(4, 64, true));
  EXPECT_EQ(iree_string_builder_size(&text_), 0u);
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 0u);
  auto block = loom_serve_speech_input_peek(input_);
  EXPECT_EQ(block.spans[0].data, reinterpret_cast<const uint8_t*>(storage));
  uint64_t position = 0;
  for (const char* expected : {"abc", "🦜", "日z"}) {
    block = loom_serve_speech_input_peek(input_);
    EXPECT_EQ(block.position, position);
    EXPECT_EQ(Read(block), expected);
    EXPECT_EQ(block.complete, std::string(expected) == "日z");
    position += block.byte_count;
    loom_serve_speech_input_consume(input_, block.byte_count);
  }
  block = loom_serve_speech_input_peek(input_);
  EXPECT_EQ(block.position, position);
  EXPECT_EQ(block.byte_count, 0u);
  EXPECT_TRUE(block.complete);
  loom_serve_speech_input_retain(input_);
  loom_serve_speech_input_release(input_);
  EXPECT_TRUE(loom_serve_speech_input_peek(input_).complete);
}

TEST_F(SpeechInputTest, EmptyFiniteInputHasACompleteWindow) {
  IREE_ASSERT_OK(Allocate(4, 64, true));
  const auto block = loom_serve_speech_input_peek(input_);
  EXPECT_EQ(block.byte_count, 0u);
  EXPECT_TRUE(block.complete);
  EXPECT_EQ(block.spans[0].data, nullptr);
}

TEST_F(SpeechInputTest, AppendsDoNotOverwriteBorrowedSnapshots) {
  IREE_ASSERT_OK(Allocate(7, 64, false));
  size_t accepted = 0;
  IREE_ASSERT_OK(Write(IREE_SV("abc"), &accepted));
  EXPECT_EQ(accepted, 3u);
  const auto snapshot = loom_serve_speech_input_peek(input_);
  IREE_ASSERT_OK(Write(IREE_SV("🦜tail"), &accepted));
  EXPECT_EQ(accepted, 4u);
  EXPECT_EQ(Read(snapshot), "abc");
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 0u);
  IREE_ASSERT_OK(Write(IREE_SV("tail"), &accepted));
  EXPECT_EQ(accepted, 0u);
  loom_serve_speech_input_consume(input_, snapshot.byte_count);
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 3u);
  IREE_ASSERT_OK(Write(IREE_SV("日"), &accepted));
  EXPECT_EQ(accepted, 3u);
  const auto wrapped = loom_serve_speech_input_peek(input_);
  EXPECT_EQ(wrapped.position, 3u);
  EXPECT_EQ(wrapped.spans[0].data_length, 4u);
  EXPECT_EQ(wrapped.spans[1].data_length, 3u);
  EXPECT_EQ(Read(wrapped), "🦜日");
  IREE_ASSERT_OK(loom_serve_speech_input_finish(input_));
  EXPECT_TRUE(loom_serve_speech_input_peek(input_).complete);
  EXPECT_EQ(Read(wrapped), "🦜日");
}

TEST_F(SpeechInputTest, PartialScalarAndRejectedWritesPreserveProgress) {
  IREE_ASSERT_OK(Allocate(7, 64, false));
  size_t accepted = 0;
  IREE_ASSERT_OK(Write(IREE_SV("a\xf0"), &accepted));
  EXPECT_EQ(accepted, 2u);
  auto block = loom_serve_speech_input_peek(input_);
  EXPECT_EQ(Read(block), "a");
  loom_serve_speech_input_consume(input_, block.byte_count);
  EXPECT_EQ(loom_serve_speech_input_peek(input_).byte_count, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_serve_speech_input_finish(input_));
  EXPECT_FALSE(loom_serve_speech_input_peek(input_).complete);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Write(IREE_SV("\x8f"), &accepted));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 6u);
  IREE_ASSERT_OK(Write(IREE_SV("\x9f\xa6\x9c"), &accepted));
  EXPECT_EQ(Read(loom_serve_speech_input_peek(input_)), "🦜");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Write(IREE_SV("ok\xff"), &accepted));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(Read(loom_serve_speech_input_peek(input_)), "🦜");
  IREE_ASSERT_OK(loom_serve_speech_input_finish(input_));
  IREE_ASSERT_OK(loom_serve_speech_input_finish(input_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        Write(IREE_SV("x"), &accepted));
  EXPECT_EQ(accepted, 0u);
}

TEST_F(SpeechInputTest, ValidatesOnlyTheAcceptedPrefix) {
  IREE_ASSERT_OK(Allocate(4, 64, false));
  size_t accepted = 0;
  IREE_ASSERT_OK(Write(IREE_SV("abc"), &accepted));
  IREE_ASSERT_OK(Write(IREE_SV("d\xff"), &accepted));
  EXPECT_EQ(accepted, 1u);
  auto block = loom_serve_speech_input_peek(input_);
  EXPECT_EQ(Read(block), "abcd");
  loom_serve_speech_input_consume(input_, block.byte_count);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Write(IREE_SV("\xff"), &accepted));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 4u);
}

TEST_F(SpeechInputTest, CoversScalarBoundariesAcrossFragmentationAndWrap) {
  // Extremes of each UTF-8 leading-byte family, including NUL and U+10FFFF.
  const std::string text =
      std::string("\0", 1) +
      "\x7f\xc2\x80\xdf\xbf\xe0\xa0\x80\xe1\x80\x80\xec\xbf\xbf"
      "\xed\x9f\xbf\xee\x80\x80\xef\xbf\xbf\xf0\x90\x80\x80"
      "\xf1\x80\x80\x80\xf3\xbf\xbf\xbf\xf4\x8f\xbf\xbf";
  for (size_t fragment : {1u, 3u, 7u}) {
    SCOPED_TRACE(fragment);
    IREE_ASSERT_OK(Allocate(7, text.size(), false));
    size_t written = 0;
    std::string received;
    while (written < text.size()) {
      size_t accepted = 0;
      IREE_ASSERT_OK(Write(
          iree_make_string_view(text.data() + written,
                                std::min(fragment, text.size() - written)),
          &accepted));
      written += accepted;
      auto block = loom_serve_speech_input_peek(input_);
      EXPECT_EQ(block.position, received.size());
      received += Read(block);
      loom_serve_speech_input_consume(input_, block.byte_count);
    }
    IREE_ASSERT_OK(loom_serve_speech_input_finish(input_));
    EXPECT_TRUE(loom_serve_speech_input_peek(input_).complete);
    EXPECT_EQ(received, text);
    loom_serve_speech_input_release(input_);
    input_ = nullptr;
  }
}

TEST_F(SpeechInputTest, RejectsInvalidLeadingAndContinuationFamilies) {
  IREE_ASSERT_OK(Allocate(7, 64, false));
  for (auto invalid :
       {IREE_SV("\x80"), IREE_SV("\xbf"), IREE_SV("\xc0"), IREE_SV("\xc1"),
        IREE_SV("\xf5"), IREE_SV("\xff"), IREE_SV("\xe0\x9f"),
        IREE_SV("\xed\xa0"), IREE_SV("\xf0\x8f"), IREE_SV("\xf4\x90"),
        IREE_SV("\xc2\x7f"), IREE_SV("\xc2\xc0")}) {
    size_t accepted = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Write(invalid, &accepted));
    EXPECT_EQ(accepted, 0u);
    EXPECT_EQ(loom_serve_speech_input_credit(input_), 7u);
    EXPECT_EQ(loom_serve_speech_input_peek(input_).byte_count, 0u);
  }
}

TEST_F(SpeechInputTest, TotalLimitFailureRetainsAcceptedInput) {
  IREE_ASSERT_OK(Allocate(4, 5, false));
  size_t accepted = 0;
  IREE_ASSERT_OK(Write(IREE_SV("abc"), &accepted));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Write(IREE_SV("def"), &accepted));
  EXPECT_EQ(accepted, 0u);
  EXPECT_EQ(Read(loom_serve_speech_input_peek(input_)), "abc");
  loom_serve_speech_input_consume(input_, 3);
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 2u);
  IREE_ASSERT_OK(Write(IREE_SV("de"), &accepted));
  EXPECT_EQ(accepted, 2u);
  EXPECT_EQ(loom_serve_speech_input_credit(input_), 0u);
  IREE_ASSERT_OK(loom_serve_speech_input_finish(input_));
}

TEST_F(SpeechInputTest, FailedAllocationDoesNotTransferRequestStorage) {
  IREE_ASSERT_OK(iree_string_builder_append_cstring(&text_, "retained"));
  const char* storage = iree_string_builder_buffer(&text_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Allocate(4, 3, true));
  iree_allocator_t failing_allocator = {
      nullptr, +[](void*, iree_allocator_command_t, const void*, void**) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_serve_speech_input_allocate(
                            4, 64, &text_, true, failing_allocator, &input_));
  EXPECT_EQ(input_, nullptr);
  EXPECT_EQ(iree_string_builder_buffer(&text_), storage);
  EXPECT_EQ(iree_string_builder_size(&text_), 8u);
}

}  // namespace
