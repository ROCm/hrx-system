// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/span.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"
#include "iree/tokenizer/normalizer/prepend.h"
#include "iree/tokenizer/normalizer/replace.h"
#include "iree/tokenizer/normalizer/sequence.h"

namespace iree::tokenizer::testing {
namespace {

ScopedNormalizer Prepend(iree_string_view_t prefix) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_normalizer_prepend_allocate(
      prefix, false, iree_allocator_system(), &normalizer));
  return ScopedNormalizer(normalizer);
}

ScopedNormalizer Span(ScopedNormalizer child) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_normalizer_span_allocate(
      child.get(), iree_allocator_system(), &normalizer));
  child.release();
  return ScopedNormalizer(normalizer);
}

ScopedNormalizer Sequence(ScopedNormalizer first, ScopedNormalizer second) {
  iree_tokenizer_normalizer_t* children[] = {first.get(), second.get()};
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_normalizer_sequence_allocate(
      children, IREE_ARRAYSIZE(children), iree_allocator_system(),
      &normalizer));
  first.release();
  second.release();
  return ScopedNormalizer(normalizer);
}

// Exercises the same state across finalized spans, with exhausted output credit
// before every process and finalize call. Input is ASCII so one-byte credit is
// sufficient for every normalization unit used here.
void CheckSpan(iree_tokenizer_normalizer_state_t* state,
               iree_string_view_t input, const std::string& expected,
               size_t capacity) {
  std::string actual;
  char buffer[7];
  while (input.size > 0) {
    size_t consumed = 0, written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state, input, iree_make_mutable_string_view(buffer, 0), 0, &consumed,
        &written));
    ASSERT_LE(consumed, input.size);
    ASSERT_EQ(written, 0u);
    input = iree_string_view_substr(input, consumed, IREE_HOST_SIZE_MAX);
    size_t accepted = consumed;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state, input, iree_make_mutable_string_view(buffer, capacity), 0,
        &consumed, &written));
    ASSERT_LE(consumed, input.size);
    ASSERT_LE(written, capacity);
    ASSERT_TRUE(accepted || consumed || written);
    input = iree_string_view_substr(input, consumed, IREE_HOST_SIZE_MAX);
    actual.append(buffer, written);
  }
  do {
    size_t written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state, iree_make_mutable_string_view(buffer, 0), &written));
    ASSERT_EQ(written, 0u);
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state, iree_make_mutable_string_view(buffer, capacity), &written));
    actual.append(buffer, written);
    ASSERT_TRUE(written || !iree_tokenizer_normalizer_state_has_pending(state));
  } while (iree_tokenizer_normalizer_state_has_pending(state));
  EXPECT_EQ(actual, expected);

  // A completed span stays complete until actual input arrives.
  size_t consumed = 0, written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state, iree_string_view_empty(),
      iree_make_mutable_string_view(buffer, capacity), 0, &consumed, &written));
  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(written, 0u);
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state, iree_make_mutable_string_view(buffer, capacity), &written));
  EXPECT_EQ(written, 0u);
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state));
}

TEST(SpanNormalizerTest, RestartsOnlyScopedComponent) {
  auto normalizer =
      Sequence(Span(Prepend(IREE_SV("_"))), Prepend(IREE_SV("^")));
  for (size_t capacity : {1u, 7u}) {
    ScopedNormalizerState state(normalizer.get());
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("a"), "^_a", capacity));
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("b"), "_b", capacity));
    ASSERT_NO_FATAL_FAILURE(CheckSpan(state.get(), IREE_SV(""), "", capacity));
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("c"), "_c", capacity));
    state.Reset();
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("d"), "^_d", capacity));
  }
}

TEST(SpanNormalizerTest, DrainsDelayedAndExpandingChildBeforeRestart) {
  std::string expansion(IREE_TOKENIZER_REPLACE_MAX_CONTENT, 'x');
  iree_tokenizer_normalizer_t* replace = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_replace_allocate(
      IREE_SV("ab"), iree_make_string_view(expansion.data(), expansion.size()),
      iree_allocator_system(), &replace));
  auto normalizer =
      Span(Sequence(Prepend(IREE_SV("_")), ScopedNormalizer(replace)));
  for (size_t capacity : {1u, 7u}) {
    ScopedNormalizerState state(normalizer.get());
    ASSERT_NO_FATAL_FAILURE(CheckSpan(state.get(), IREE_SV(""), "", capacity));
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("ab"), "_" + expansion, capacity));
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("a"), "_a", capacity));
    ASSERT_NO_FATAL_FAILURE(
        CheckSpan(state.get(), IREE_SV("b"), "_b", capacity));
  }
}

TEST(SpanNormalizerTest, FailedAllocationPreservesChildOwnership) {
  auto child = Prepend(IREE_SV("_"));
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_normalizer_span_allocate(
                            child.get(), iree_allocator_null(), &normalizer));
  EXPECT_EQ(normalizer, nullptr);
  EXPECT_EQ(ProcessAndFinalize(child.get(), "a", false), "_a");
}

}  // namespace
}  // namespace iree::tokenizer::testing
