// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/request.h"

#include <cstring>
#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/testdata/streaming_testdata.h"

namespace {

static iree_string_view_t GetEmbeddedTokenizerJson() {
  const iree_file_toc_t* toc = iree_tokenizer_streaming_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_streaming_testdata_size(); ++i) {
    if (std::strcmp(toc[i].name, "bpe_bytelevel_minimal.json") == 0) {
      return iree_make_string_view(toc[i].data, toc[i].size);
    }
  }
  return iree_string_view_empty();
}

struct TokenizerDeleter {
  void operator()(iree_tokenizer_t* tokenizer) const {
    iree_tokenizer_free(tokenizer);
  }
};

using TokenizerPtr = std::unique_ptr<iree_tokenizer_t, TokenizerDeleter>;

struct ScopedRequest {
  ~ScopedRequest() {
    id4_ideogram4_request_deinitialize(&value, iree_allocator_system());
  }

  id4_ideogram4_request_t value = {};
};

struct ScopedQwenInputs {
  ~ScopedQwenInputs() {
    id4_ideogram4_qwen_inputs_deinitialize(&value, iree_allocator_system());
  }

  id4_ideogram4_qwen_inputs_t value = {};
};

static TokenizerPtr LoadTokenizer() {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_from_huggingface_json(
      GetEmbeddedTokenizerJson(), iree_allocator_system(), &tokenizer));
  return TokenizerPtr(tokenizer);
}

TEST(Ideogram4RequestTest, ParsesStrictPromptJson) {
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":\"hello world\"}"), iree_allocator_system(),
      &request.value));
  EXPECT_TRUE(
      iree_string_view_equal(request.value.prompt, IREE_SV("hello world")));

  id4_ideogram4_request_t rejected_request = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_request_parse_json(
                            IREE_SV("{\"prompt\":\"hello\",\"seed\":1}"),
                            iree_allocator_system(), &rejected_request));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_request_parse_json(
                            IREE_SV("{\"prompt\":\"\"}"),
                            iree_allocator_system(), &rejected_request));
}

TEST(Ideogram4RequestTest, LowersPromptToQwenInputTensors) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":\"hello world\"}"), iree_allocator_system(),
      &request.value));

  id4_ideogram4_qwen_lowering_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.tokenizer = tokenizer.get();
  options.request = &request.value;
  options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  options.max_token_count = 16;

  ScopedQwenInputs inputs;
  IREE_ASSERT_OK(id4_ideogram4_request_lower_qwen_inputs(
      &options, iree_allocator_system(), &inputs.value));
  ASSERT_EQ(inputs.value.token_count, 3u);
  EXPECT_EQ(inputs.value.token_ids[0], 98);
  EXPECT_EQ(inputs.value.token_ids[1], 105);
  EXPECT_EQ(inputs.value.token_ids[2], 110);
  for (uint32_t i = 0; i < inputs.value.token_count; ++i) {
    EXPECT_EQ(inputs.value.token_weights[i], 1.0f);
  }
  for (uint32_t i = 0; i < inputs.value.token_count * inputs.value.token_count;
       ++i) {
    EXPECT_EQ(inputs.value.attention_mask[i], 0.0f);
  }
}

TEST(Ideogram4RequestTest, RejectsTokenOverflow) {
  TokenizerPtr tokenizer = LoadTokenizer();
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":\"hello world\"}"), iree_allocator_system(),
      &request.value));

  id4_ideogram4_qwen_lowering_options_t options;
  memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.tokenizer = tokenizer.get();
  options.request = &request.value;
  options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  options.max_token_count = 1;

  ScopedQwenInputs inputs;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        id4_ideogram4_request_lower_qwen_inputs(
                            &options, iree_allocator_system(), &inputs.value));
}

}  // namespace
