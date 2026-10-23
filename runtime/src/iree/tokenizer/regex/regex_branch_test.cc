// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/unicode.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/compile.h"

namespace {

class RegexBranches : public ::testing::Test {
 protected:
  void Compile(const char* pattern) {
    iree_tokenizer_regex_compile_error_t error = {0};
    IREE_ASSERT_OK(iree_tokenizer_regex_compile_and_load(
        iree_make_cstring_view(pattern), 0, iree_allocator_system(), &dfa_,
        &storage_, &error));
    IREE_ASSERT_OK(iree_tokenizer_regex_stride_allocate(
        &dfa_, iree_allocator_system(), &stride_));
  }

  ~RegexBranches() override {
    iree_tokenizer_regex_stride_free(stride_, iree_allocator_system());
    iree_allocator_free(iree_allocator_system(), storage_);
  }

  void Check(const std::string& text, const std::vector<std::string>& expected,
             size_t chunk_size, bool use_stride) {
    SCOPED_TRACE(::testing::Message()
                 << "text=" << ::testing::PrintToString(text)
                 << " chunk=" << chunk_size << " stride=" << use_stride);
    Capture capture = {text, {}};
    if (chunk_size == 0) {
      IREE_ASSERT_OK(iree_tokenizer_regex_exec(
          &dfa_, iree_make_string_view(text.data(), text.size()), Collect,
          &capture));
      IREE_ASSERT_OK(iree_tokenizer_regex_exec(
          &dfa_, iree_make_string_view(text.data(), text.size()), nullptr,
          nullptr));
    } else {
      iree_tokenizer_regex_exec_state_t state;
      iree_tokenizer_regex_exec_initialize(&state, &dfa_);
      for (size_t offset = 0; offset < text.size();) {
        size_t count = std::min(chunk_size, text.size() - offset);
        count -= iree_unicode_utf8_incomplete_tail_length(text.data() + offset,
                                                          count);
        if (count == 0) {
          count = iree_unicode_utf8_sequence_length(
              static_cast<uint8_t>(text[offset]));
        }
        IREE_ASSERT_OK(iree_tokenizer_regex_exec_feed(
            &dfa_, &state, iree_make_string_view(text.data() + offset, count),
            offset, use_stride ? stride_ : nullptr, Collect, &capture));
        offset += count;
      }
      IREE_ASSERT_OK(iree_tokenizer_regex_exec_finalize(
          &dfa_, &state, text.size(), iree_string_view_empty(), Collect,
          &capture));
    }
    EXPECT_EQ(capture.matches, expected);
  }

  void CheckAllBoundaries(const std::string& text,
                          const std::vector<std::string>& expected) {
    Check(text, expected, 0, false);
    for (size_t chunk = 1; chunk <= text.size(); ++chunk) {
      Check(text, expected, chunk, false);
      Check(text, expected, chunk, true);
    }
  }

 private:
  struct Capture {
    // Original text indexed by the executor's absolute match offsets.
    const std::string& text;
    // Complete match sequence in callback order.
    std::vector<std::string> matches;
  };

  static iree_status_t Collect(void* user_data,
                               iree_tokenizer_regex_match_t match) {
    auto& capture = *static_cast<Capture*>(user_data);
    capture.matches.push_back(
        capture.text.substr(match.start, match.end - match.start));
    return iree_ok_status();
  }

  // Loaded views into the owned serialized DFA.
  iree_tokenizer_regex_dfa_t dfa_ = {};
  // Serialized DFA storage owned by this test instance.
  uint8_t* storage_ = nullptr;
  // Optional ASCII accelerator built from the same DFA.
  iree_tokenizer_regex_stride_t* stride_ = nullptr;
};

TEST_F(RegexBranches, GreedyContinuationThroughNonAcceptingStates) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(\s*[\r\n]+|\s+(?!\S)|\s+)"));
  CheckAllBoundaries(" \n    \n    x", {" \n    \n", "   ", " "});
  CheckAllBoundaries("\n    x", {"\n", "   ", " "});
  CheckAllBoundaries("\n   ", {"\n", "   "});
  CheckAllBoundaries("\u00a0\n\u2003\n  x", {"\u00a0\n\u2003\n", " ", " "});
}

TEST_F(RegexBranches, EarlierLookaheadWinsWhenItPasses) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(\s+(?!\S)|\s*[\r\n]+|\s+)"));
  CheckAllBoundaries(" \n    \n    x", {" \n    \n   ", " "});
  CheckAllBoundaries("\n   ", {"\n   "});
}

TEST_F(RegexBranches, EarlierUnconditionalBranchWins) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(\s+|\s+(?!\S)|\s*[\r\n]+)"));
  CheckAllBoundaries(" \n    \n    x", {" \n    \n    "});
  CheckAllBoundaries("\n   ", {"\n   "});
}

TEST_F(RegexBranches, BoundedGreedyContinuation) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(a(?:ba){1,3}|a+(?!b)|a+)"));
  CheckAllBoundaries("ababababa", {"abababa", "a"});
}

TEST_F(RegexBranches, ReplaysSuffixAfterAcceptedPrefix) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"((?:ab)+|a+(?!b)|a+)"));
  CheckAllBoundaries("ababa", {"abab", "a"});
}

TEST_F(RegexBranches, ReplaysSuffixAfterUnfinishedPrefix) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(abx|b(?!q)|b)"));
  CheckAllBoundaries("ab", {"b"});
}

TEST_F(RegexBranches, EmitsMatchesInsideRewindBuffer) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(a[bc]*z|a(?!x)|b)"));
  CheckAllBoundaries("abbbcX", {"a", "b", "b", "b"});
}

TEST_F(RegexBranches, EndAnchorCanSupersedeEarlierAcceptance) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(a*b$|a+(?!c)|a+)"));
  CheckAllBoundaries("aaab", {"aaab"});
  CheckAllBoundaries("abab", {"a", "ab"});
}

TEST_F(RegexBranches, OneShotRetainsItsCompleteInput) {
  ASSERT_NO_FATAL_FAILURE(Compile(R"(\s*[\r\n]+|\s+(?!\S)|\s+)"));
  std::string spaces(IREE_TOKENIZER_UTIL_REGEX_REWIND_BUFFER_SIZE * 2, ' ');
  Check("\n" + spaces, {"\n", spaces}, 0, false);
}

}  // namespace
