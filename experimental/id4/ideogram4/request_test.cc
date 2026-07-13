// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/request.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>

#include "iree/base/internal/math.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/testdata/streaming_testdata.h"
#include "iree/tokenizer/vocab/vocab.h"

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

struct ScopedDitInputs {
  ~ScopedDitInputs() {
    id4_ideogram4_dit_inputs_deinitialize(&value, iree_allocator_system());
  }

  id4_ideogram4_dit_inputs_t value = {};
};

static TokenizerPtr LoadTokenizer() {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_from_huggingface_json(
      GetEmbeddedTokenizerJson(), iree_allocator_system(), &tokenizer));
  return TokenizerPtr(tokenizer);
}

static uint32_t TokenizerVocabSize(const iree_tokenizer_t* tokenizer) {
  return (uint32_t)iree_tokenizer_vocab_capacity(
      iree_tokenizer_vocab(tokenizer));
}

static size_t DitPositionEmbeddingOffset(
    uint32_t token_count, uint32_t half_size, uint32_t outer_ordinal,
    uint32_t inner_ordinal, uint32_t half_channel, uint32_t token_ordinal) {
  return ((((size_t)outer_ordinal * 2 + inner_ordinal) * half_size +
           half_channel) *
          token_count) +
         token_ordinal;
}

static float Ideogram4MRoPEInvFrequency(uint32_t half_channel,
                                        uint32_t attention_head_size) {
  const float inv_frequency =
      1.0f / std::pow(5000000.0f, (2.0f * (float)half_channel) /
                                      (float)attention_head_size);
  return iree_math_bf16_to_f32(iree_math_f32_to_bf16(inv_frequency));
}

TEST(Ideogram4RequestTest, ParsesStructuredPromptJson) {
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":\"hello world\"}"), iree_allocator_system(),
      &request.value));
  EXPECT_TRUE(iree_string_view_equal(request.value.prompt_payload,
                                     IREE_SV("{\"prompt\":\"hello world\"}")));
  EXPECT_TRUE(iree_string_view_equal(
      request.value.qwen_prompt,
      IREE_SV("<|im_start|>user\n{\"prompt\":\"hello world\"}<|im_end|>\n"
              "<|im_start|>assistant\n")));

  ScopedRequest structured_request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\"}}"),
      iree_allocator_system(), &structured_request.value));
  EXPECT_TRUE(iree_string_view_equal(
      structured_request.value.prompt_payload,
      IREE_SV("{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\"}}")));

  id4_ideogram4_request_t rejected_request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_parse_json(IREE_SV("{}"), iree_allocator_system(),
                                       &rejected_request));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_request_parse_json(
                            IREE_SV("[\"not an object\"]"),
                            iree_allocator_system(), &rejected_request));
}

TEST(Ideogram4RequestTest, ParsesFullGenerationJson) {
  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\"}},"
              "\"generation\":{\"latent_width\":8,\"latent_height\":16,"
              "\"sampler\":\"V4_DEFAULT_20\",\"seed\":20260625}}"),
      iree_allocator_system(), &request.value));
  EXPECT_TRUE(iree_all_bits_set(request.value.flags,
                                ID4_IDEOGRAM4_REQUEST_FLAG_HAS_GENERATION));
  EXPECT_TRUE(iree_string_view_equal(
      request.value.prompt_payload,
      IREE_SV("{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\"}}")));
  EXPECT_TRUE(iree_string_view_equal(
      request.value.qwen_prompt,
      IREE_SV("<|im_start|>user\n{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\"}}<|im_end|>\n"
              "<|im_start|>assistant\n")));
  EXPECT_EQ(request.value.generation.latent_width, 8u);
  EXPECT_EQ(request.value.generation.latent_height, 16u);
  EXPECT_EQ(request.value.generation.sampler_preset,
            ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20);
  EXPECT_EQ(request.value.generation.seed, 20260625u);
}

TEST(Ideogram4RequestTest, InitializesTextGenerationRequest) {
  id4_ideogram4_request_generation_t generation = {};
  generation.latent_width = 8;
  generation.latent_height = 16;
  generation.sampler_preset = ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20;
  generation.seed = 20260625;

  ScopedRequest request;
  IREE_ASSERT_OK(id4_ideogram4_request_initialize_text(
      IREE_SV("hello world"), &generation, iree_allocator_system(),
      &request.value));
  EXPECT_TRUE(iree_all_bits_set(request.value.flags,
                                ID4_IDEOGRAM4_REQUEST_FLAG_HAS_GENERATION));
  EXPECT_TRUE(iree_string_view_equal(request.value.prompt_payload,
                                     IREE_SV("hello world")));
  EXPECT_TRUE(
      iree_string_view_equal(request.value.qwen_prompt,
                             IREE_SV("<|im_start|>user\nhello world<|im_end|>\n"
                                     "<|im_start|>assistant\n")));
  EXPECT_EQ(request.value.generation.latent_width, 8u);
  EXPECT_EQ(request.value.generation.latent_height, 16u);
  EXPECT_EQ(request.value.generation.sampler_preset,
            ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20);
  EXPECT_EQ(request.value.generation.seed, 20260625u);
}

TEST(Ideogram4RequestTest, RejectsInvalidTextGenerationRequest) {
  id4_ideogram4_request_generation_t generation = {};
  generation.latent_width = 8;
  generation.latent_height = 8;
  generation.sampler_preset = ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20;

  id4_ideogram4_request_t request = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_request_initialize_text(
                            iree_string_view_empty(), &generation,
                            iree_allocator_system(), &request));

  generation.latent_width = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_initialize_text(IREE_SV("hello"), &generation,
                                            iree_allocator_system(), &request));
}

TEST(Ideogram4RequestTest, CanonicalizesPromptPayloadJson) {
  ScopedRequest compact_request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\"prompt\":{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\","
              "\"tags\":[\"city\",\"walk\"]}},\"generation\":{"
              "\"latent_width\":8,\"latent_height\":8,"
              "\"sampler\":\"V4_DEFAULT_20\",\"seed\":20260625}}"),
      iree_allocator_system(), &compact_request.value));

  ScopedRequest pretty_request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\n"
              "  \"prompt\": {\n"
              "    \"high_level_description\": \"hello\",\n"
              "    \"style_description\": {\n"
              "      \"medium\": \"photo\",\n"
              "      \"tags\": [\"city\", \"walk\",],\n"
              "    },\n"
              "  },\n"
              "  \"generation\": {\n"
              "    \"latent_width\": 8,\n"
              "    \"latent_height\": 8,\n"
              "    \"sampler\": \"V4_DEFAULT_20\",\n"
              "    \"seed\": 20260625,\n"
              "  },\n"
              "}"),
      iree_allocator_system(), &pretty_request.value));

  EXPECT_TRUE(iree_string_view_equal(compact_request.value.prompt_payload,
                                     pretty_request.value.prompt_payload));
  EXPECT_TRUE(iree_string_view_equal(compact_request.value.qwen_prompt,
                                     pretty_request.value.qwen_prompt));

  ScopedRequest standalone_request;
  IREE_ASSERT_OK(id4_ideogram4_request_parse_json(
      IREE_SV("{\n"
              "  \"high_level_description\": \"hello\",\n"
              "  \"style_description\": {\n"
              "    \"medium\": \"photo\",\n"
              "    \"tags\": [\"city\", \"walk\"],\n"
              "  },\n"
              "}"),
      iree_allocator_system(), &standalone_request.value));
  EXPECT_TRUE(iree_string_view_equal(
      standalone_request.value.prompt_payload,
      IREE_SV("{\"high_level_description\":\"hello\","
              "\"style_description\":{\"medium\":\"photo\","
              "\"tags\":[\"city\",\"walk\"]}}")));
}

TEST(Ideogram4RequestTest, RejectsMalformedFullGenerationJson) {
  id4_ideogram4_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_parse_json(
          IREE_SV("{\"generation\":{\"latent_width\":8,\"latent_height\":8,"
                  "\"sampler\":\"V4_DEFAULT_20\",\"seed\":1}}"),
          iree_allocator_system(), &request));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_parse_json(
          IREE_SV("{\"prompt\":\"hello\",\"generation\":{\"latent_width\":8,"
                  "\"latent_height\":8,\"sampler\":\"V4_DEFAULT_20\","
                  "\"seed\":1},"
                  "\"extra\":true}"),
          iree_allocator_system(), &request));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_parse_json(
          IREE_SV("{\"prompt\":\"hello\",\"generation\":{\"latent_width\":0,"
                  "\"latent_height\":8,\"sampler\":\"V4_DEFAULT_20\","
                  "\"seed\":1}}"),
          iree_allocator_system(), &request));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_parse_json(
          IREE_SV("{\"prompt\":\"hello\",\"generation\":{\"latent_width\":8,"
                  "\"latent_height\":8,\"sampler\":\"UNKNOWN\","
                  "\"seed\":1}}"),
          iree_allocator_system(), &request));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_parse_json(
          IREE_SV("{\"prompt\":\"hello\",\"generation\":\"\"}"),
          iree_allocator_system(), &request));
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
  options.max_token_count = 128;
  options.token_capacity = 128;
  options.vocab_size = TokenizerVocabSize(tokenizer.get());

  ScopedQwenInputs inputs;
  IREE_ASSERT_OK(id4_ideogram4_request_lower_qwen_inputs(
      &options, iree_allocator_system(), &inputs.value));
  ASSERT_GT(inputs.value.token_count, 3u);
  EXPECT_EQ(inputs.value.token_capacity, 128u);
  for (uint32_t i = 0; i < inputs.value.token_count; ++i) {
    EXPECT_EQ(inputs.value.token_weights[i], 1.0f);
  }
  const float future_token_mask = -FLT_MAX / 4.0f;
  for (uint32_t query = 0; query < inputs.value.token_count; ++query) {
    for (uint32_t key = 0; key < inputs.value.token_count; ++key) {
      EXPECT_EQ(inputs.value
                    .attention_mask[query * inputs.value.token_capacity + key],
                key <= query ? 0.0f : future_token_mask);
    }
  }
  for (uint32_t key = inputs.value.token_count;
       key < inputs.value.token_capacity; ++key) {
    EXPECT_EQ(inputs.value.attention_mask[key], future_token_mask);
  }

  uint32_t counted_token_count = 0;
  IREE_ASSERT_OK(id4_ideogram4_request_count_qwen_tokens(
      &options, iree_allocator_system(), &counted_token_count));
  EXPECT_EQ(counted_token_count, inputs.value.token_count);
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
  options.token_capacity = 1;
  options.vocab_size = TokenizerVocabSize(tokenizer.get());

  ScopedQwenInputs inputs;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        id4_ideogram4_request_lower_qwen_inputs(
                            &options, iree_allocator_system(), &inputs.value));
}

TEST(Ideogram4RequestTest, RejectsTokenIdOutsideModelVocab) {
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
  options.max_token_count = 128;
  options.token_capacity = 128;
  options.vocab_size = 1;

  ScopedQwenInputs inputs;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_ideogram4_request_lower_qwen_inputs(
                            &options, iree_allocator_system(), &inputs.value));
}

TEST(Ideogram4RequestTest, LowersDitMetadataInputs) {
  id4_ideogram4_request_generation_t generation = {};
  generation.latent_width = 3;
  generation.latent_height = 2;
  generation.sampler_preset = ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20;
  generation.seed = 1;

  id4_ideogram4_dit_lowering_options_t options = {};
  options.structure_size = sizeof(options);
  options.generation = &generation;
  options.text_token_count = 2;
  options.attention_head_size = 256;

  ScopedDitInputs inputs;
  IREE_ASSERT_OK(id4_ideogram4_request_lower_dit_inputs(
      &options, iree_allocator_system(), &inputs.value));
  EXPECT_EQ(inputs.value.text_token_count, 2u);
  EXPECT_EQ(inputs.value.image_token_count, 6u);
  ASSERT_EQ(inputs.value.conditioned.token_count, 8u);
  ASSERT_EQ(inputs.value.unconditioned.token_count, 6u);
  EXPECT_EQ(inputs.value.conditioned.image_indicator_byte_length,
            8u * sizeof(int32_t));
  EXPECT_EQ(inputs.value.unconditioned.image_indicator_byte_length,
            6u * sizeof(int32_t));
  EXPECT_EQ(inputs.value.conditioned.position_embedding_byte_length,
            2u * 2u * 128u * 8u * sizeof(float));
  EXPECT_EQ(inputs.value.unconditioned.position_embedding_byte_length,
            2u * 2u * 128u * 6u * sizeof(float));

  ASSERT_NE(inputs.value.conditioned.image_indicator, nullptr);
  EXPECT_EQ(inputs.value.conditioned.image_indicator[0], 0);
  EXPECT_EQ(inputs.value.conditioned.image_indicator[1], 0);
  for (uint32_t i = 2; i < inputs.value.conditioned.token_count; ++i) {
    EXPECT_EQ(inputs.value.conditioned.image_indicator[i], 1);
  }
  ASSERT_NE(inputs.value.unconditioned.image_indicator, nullptr);
  for (uint32_t i = 0; i < inputs.value.unconditioned.token_count; ++i) {
    EXPECT_EQ(inputs.value.unconditioned.image_indicator[i], 1);
  }

  ASSERT_NE(inputs.value.conditioned.position_embedding, nullptr);
  const uint32_t conditioned_token_count = inputs.value.conditioned.token_count;
  const uint32_t half_size = options.attention_head_size / 2;
  const float* conditioned_position =
      inputs.value.conditioned.position_embedding;
  EXPECT_FLOAT_EQ(conditioned_position[DitPositionEmbeddingOffset(
                      conditioned_token_count, half_size, 0, 0, 0, 0)],
                  1.0f);
  EXPECT_FLOAT_EQ(conditioned_position[DitPositionEmbeddingOffset(
                      conditioned_token_count, half_size, 0, 1, 0, 0)],
                  0.0f);
  EXPECT_FLOAT_EQ(conditioned_position[DitPositionEmbeddingOffset(
                      conditioned_token_count, half_size, 1, 0, 0, 0)],
                  -0.0f);
  EXPECT_FLOAT_EQ(conditioned_position[DitPositionEmbeddingOffset(
                      conditioned_token_count, half_size, 1, 1, 0, 0)],
                  1.0f);

  const float text_frequency =
      1.0f * Ideogram4MRoPEInvFrequency(0, options.attention_head_size);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 0, 0, 1)],
              std::cos(text_frequency), 1e-6f);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 1, 0, 1)],
              std::sin(text_frequency), 1e-6f);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 1, 0, 0, 1)],
              -std::sin(text_frequency), 1e-6f);

  const float first_image_frequency =
      65536.0f * Ideogram4MRoPEInvFrequency(0, options.attention_head_size);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 0, 0, 2)],
              std::cos(first_image_frequency), 1e-6f);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 1, 0, 2)],
              std::sin(first_image_frequency), 1e-6f);

  const float image_width_frequency =
      65537.0f * Ideogram4MRoPEInvFrequency(2, options.attention_head_size);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 0, 2, 3)],
              std::cos(image_width_frequency), 1e-5f);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 1, 2, 3)],
              std::sin(image_width_frequency), 1e-5f);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 1, 0, 2, 3)],
              -std::sin(image_width_frequency), 1e-5f);
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 1, 1, 2, 3)],
              std::cos(image_width_frequency), 1e-5f);
  // Official BF16 model loading rounds inv_freq before F32 position math.
  EXPECT_NEAR(conditioned_position[DitPositionEmbeddingOffset(
                  conditioned_token_count, half_size, 0, 1, 20, 2)],
              0.6094503998756409f, 1e-5f);

  ASSERT_NE(inputs.value.unconditioned.position_embedding, nullptr);
  const float* unconditioned_position =
      inputs.value.unconditioned.position_embedding;
  const float unconditioned_width_frequency =
      65537.0f * Ideogram4MRoPEInvFrequency(2, options.attention_head_size);
  EXPECT_NEAR(
      unconditioned_position[DitPositionEmbeddingOffset(
          inputs.value.unconditioned.token_count, half_size, 0, 0, 2, 1)],
      std::cos(unconditioned_width_frequency), 1e-5f);
}

TEST(Ideogram4RequestTest, RejectsInvalidDitMetadataLowering) {
  id4_ideogram4_request_generation_t generation = {};
  generation.latent_width = 1;
  generation.latent_height = 1;
  generation.sampler_preset = ID4_IDEOGRAM4_SAMPLER_PRESET_V4_DEFAULT_20;

  id4_ideogram4_dit_lowering_options_t options = {};
  options.structure_size = sizeof(options);
  options.generation = &generation;
  options.text_token_count = 1;
  options.attention_head_size = 128;

  ScopedDitInputs inputs;
  id4_ideogram4_dit_lowering_options_t invalid_options = options;
  invalid_options.structure_size = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_lower_dit_inputs(
          &invalid_options, iree_allocator_system(), &inputs.value));

  invalid_options = options;
  invalid_options.text_token_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_lower_dit_inputs(
          &invalid_options, iree_allocator_system(), &inputs.value));

  invalid_options = options;
  invalid_options.attention_head_size = 127;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_lower_dit_inputs(
          &invalid_options, iree_allocator_system(), &inputs.value));

  invalid_options = options;
  invalid_options.attention_head_size = 16;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_request_lower_dit_inputs(
          &invalid_options, iree_allocator_system(), &inputs.value));
}

}  // namespace
