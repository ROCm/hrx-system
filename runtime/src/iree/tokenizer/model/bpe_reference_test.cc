// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/model_test_util.h"

namespace iree::tokenizer {
namespace {

// Independent, literal rank-ordered adjacent merge reference. It shares no
// trie, reachability, pair validation, or merge lookup code with the encoder.
struct Dictionary {
  // Token spellings indexed by ID.
  std::vector<std::string> vocabulary;
  // Adjacent token pairs in priority order.
  std::vector<std::pair<int32_t, int32_t>> rules;
  // Virtual text appended to the final initial symbol.
  std::string suffix;

  int32_t Lookup(const std::string& text) const {
    auto it = std::find(vocabulary.begin(), vocabulary.end(), text);
    return it == vocabulary.end() ? -1 : it - vocabulary.begin();
  }

  std::vector<int32_t> Reference(const std::string& text) const {
    std::vector<int32_t> tokens;
    for (size_t i = 0; i < text.size(); ++i) {
      std::string atom(1, text[i]);
      if (i + 1 == text.size()) {
        atom += suffix;
      }
      int32_t token = Lookup(atom);
      if (token >= 0) {
        tokens.push_back(token);
      }
    }
    while (tokens.size() > 1) {
      size_t rank = rules.size(), position = 0;
      for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        auto pair = std::make_pair(tokens[i], tokens[i + 1]);
        auto it = std::find(rules.begin(), rules.end(), pair);
        size_t candidate_rank = it - rules.begin();
        if (candidate_rank < rank) {
          rank = candidate_rank;
          position = i;
        }
      }
      if (rank == rules.size()) {
        break;
      }
      tokens[position] = Lookup(vocabulary[tokens[position]] +
                                vocabulary[tokens[position + 1]]);
      tokens.erase(tokens.begin() + position + 1);
    }
    return tokens;
  }

  void Check(const std::vector<std::string>& inputs) const {
    testing::ScopedVocabBuilder builder;
    for (const auto& token : vocabulary) {
      builder.AddToken(token.c_str());
    }
    for (auto pair : rules) {
      builder.AddMerge(pair.first, pair.second);
    }
    testing::ScopedVocab vocab(builder.Build());
    iree_tokenizer_model_t* raw_model = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_bpe_model_allocate(
        vocab.get(), IREE_TOKENIZER_BPE_FLAG_NONE, iree_allocator_system(),
        &raw_model));
    testing::ScopedModel model(raw_model);
    IREE_ASSERT_OK(iree_tokenizer_bpe_model_set_end_of_word_suffix(
        model.get(), iree_make_string_view(suffix.data(), suffix.size())));
    testing::ScopedModelState state(model.get());
    for (const auto& input : inputs) {
      SCOPED_TRACE(input);
      std::vector<int32_t> actual(input.size() + suffix.size());
      iree_tokenizer_segment_t segment = {0, input.size()};
      size_t consumed = 0, count = 0;
      IREE_ASSERT_OK(iree_tokenizer_model_state_encode(
          state.get(), iree_make_const_byte_span(input.data(), input.size()),
          iree_tokenizer_make_segment_list(&segment, 1),
          iree_tokenizer_make_token_output(actual.data(), nullptr, nullptr,
                                           actual.size()),
          &consumed, &count));
      ASSERT_EQ(consumed, 1u);
      actual.resize(count);
      ASSERT_EQ(actual, Reference(input));
    }
  }
};

TEST(BpeReference, CompleteMergeOrderFamily) {
  Dictionary dictionary{{"a", "b", "aa", "ab", "ba", "bb", "aaa", "aab", "aba",
                         "abb", "baa", "bab", "bba", "bbb"},
                        {{0, 0}, {0, 1}, {1, 0}, {2, 0}, {0, 2}},
                        ""};
  std::vector<std::string> inputs;
  for (size_t length = 1; length <= 7; ++length) {
    for (size_t pattern = 0; pattern < (1u << length); ++pattern) {
      std::string text(length, 'a');
      for (size_t i = 0; i < length; ++i) {
        text[i] += (pattern >> i) & 1;
      }
      inputs.push_back(text);
    }
  }
  std::sort(dictionary.rules.begin(), dictionary.rules.end());
  do {
    dictionary.Check(inputs);
  } while (
      std::next_permutation(dictionary.rules.begin(), dictionary.rules.end()));
}

TEST(BpeReference, MissingAtomicTokens) {
  Dictionary{{"a", "b", "ab"}, {{0, 1}}, ""}.Check({"a?b", "?ab?", "a??b"});
}

TEST(BpeReference, NestedCompoundPriorityAtPathBoundary) {
  Dictionary{{"/", "v", "i", "s", "o", "n", "on", "is", "ion", "io", "ision",
              "vis", "/v", "si", "vi", "vision", "isi"},
             {{4, 5},
              {2, 3},
              {2, 6},
              {2, 4},
              {7, 8},
              {1, 7},
              {0, 1},
              {3, 2},
              {1, 2},
              {1, 10},
              {7, 2}},
             ""}
      .Check({"/vision"});
}

TEST(BpeReference, SuffixAtomsBeforeMerging) {
  Dictionary{{"a", "aa", "aa</w>"}, {{0, 0}}, "</w>"}.Check(
      {"aaaa", "aa", "a", "aaa"});
}

TEST(BpeReference, UnreachableWholeWord) {
  Dictionary{{"cat</w>"}, {}, "</w>"}.Check({"cat"});
}

TEST(BpeReference, SuffixWithLiteralSpelling) {
  Dictionary{{"a", "b", "<", "/", "w", ">", "a</w>", "b</w>"}, {}, "</w>"}
      .Check({"a</w>b", "a</w>", "a</w>a"});
}

}  // namespace
}  // namespace iree::tokenizer
