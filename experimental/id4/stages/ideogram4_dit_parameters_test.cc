// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/ideogram4_dit_parameters.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static id4_ideogram4_dit_model_config_t MakeModelConfig() {
  return id4_ideogram4_dit_model_config_t{
      // Number of transformer blocks in the DiT.
      /*.layer_count=*/3,
      // Channel count of each VAE latent image token.
      /*.input_channel_count=*/4,
      // Transformer hidden-state channel count.
      /*.hidden_size=*/32,
      // Feed-forward intermediate channel count.
      /*.intermediate_size=*/64,
      // Transformer attention head count.
      /*.attention_head_count=*/2,
      // AdaLN conditioning vector channel count.
      /*.adaln_size=*/4,
      // Qwen condition feature channel count.
      /*.llm_feature_count=*/208,
      // Number of image-indicator embedding rows.
      /*.image_indicator_count=*/2,
  };
}

TEST(Ideogram4DitParameters, ParsesParameterFormatNames) {
  id4_ideogram4_dit_parameter_format_t format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_ASSERT_OK(
      id4_ideogram4_dit_parameter_format_parse(IREE_SV("bf16"), &format));
  EXPECT_EQ(format, ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16);
  EXPECT_TRUE(iree_string_view_equal(
      id4_ideogram4_dit_parameter_format_name(format), IREE_SV("bf16")));

  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_format_parse(
      IREE_SV("mixed_bf16_fp8_e4m3"), &format));
  EXPECT_EQ(format, ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3);
  EXPECT_TRUE(
      iree_string_view_equal(id4_ideogram4_dit_parameter_format_name(format),
                             IREE_SV("mixed_bf16_fp8_e4m3")));

  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_format_parse(
      IREE_SV("mixed_bf16_fp8_e4m3_all_supported"), &format));
  EXPECT_EQ(
      format,
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3_ALL_SUPPORTED);
  EXPECT_TRUE(
      iree_string_view_equal(id4_ideogram4_dit_parameter_format_name(format),
                             IREE_SV("mixed_bf16_fp8_e4m3_all_supported")));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_dit_parameter_format_parse(IREE_SV("q4"), &format));
}

TEST(Ideogram4DitParameters, Bf16FormatProducesNoSourceOverrides) {
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16, MakeModelConfig(),
      iree_string_view_empty(), iree_allocator_system(), &rules));
  EXPECT_EQ(rules.count, 0u);
  EXPECT_EQ(rules.values, nullptr);
  EXPECT_EQ(rules.key_storage, nullptr);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitParameters, MixedFp8FormatRequiresSourceScope) {
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      id4_ideogram4_dit_parameter_source_rule_list_initialize(
          ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3,
          MakeModelConfig(), iree_string_view_empty(), iree_allocator_system(),
          &rules));
}

TEST(Ideogram4DitParameters, MixedFp8FormatProducesLayerProjectionSourceRules) {
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3, MakeModelConfig(),
      IREE_SV("native_fp8"), iree_allocator_system(), &rules));

  ASSERT_EQ(rules.count, 9u);
  ASSERT_NE(rules.values, nullptr);
  ASSERT_NE(rules.key_storage, nullptr);
  const iree_string_view_t expected_keys[] = {
      IREE_SV("layers.0.attention.qkv.weight"),
      IREE_SV("layers.0.attention.o.weight"),
      IREE_SV("layers.0.feed_forward.w2.weight"),
      IREE_SV("layers.1.attention.qkv.weight"),
      IREE_SV("layers.1.attention.o.weight"),
      IREE_SV("layers.1.feed_forward.w2.weight"),
      IREE_SV("layers.2.attention.qkv.weight"),
      IREE_SV("layers.2.attention.o.weight"),
      IREE_SV("layers.2.feed_forward.w2.weight"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_keys); ++i) {
    EXPECT_TRUE(iree_string_view_equal(rules.values[i].key, expected_keys[i]));
    EXPECT_TRUE(iree_string_view_equal(rules.values[i].source_scope,
                                       IREE_SV("native_fp8")));
    EXPECT_EQ(rules.values[i].storage,
              ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED);
  }

  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

TEST(Ideogram4DitParameters,
     AllSupportedMixedFp8FormatProducesLayerProjectionSourceRules) {
  id4_ideogram4_dit_parameter_source_rule_list_t rules;
  IREE_ASSERT_OK(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_MIXED_BF16_FP8_E4M3_ALL_SUPPORTED,
      MakeModelConfig(), IREE_SV("native_fp8"), iree_allocator_system(),
      &rules));

  ASSERT_EQ(rules.count, 15u);
  ASSERT_NE(rules.values, nullptr);
  ASSERT_NE(rules.key_storage, nullptr);
  const iree_string_view_t expected_keys[] = {
      IREE_SV("layers.0.attention.qkv.weight"),
      IREE_SV("layers.0.attention.o.weight"),
      IREE_SV("layers.0.feed_forward.w1.weight"),
      IREE_SV("layers.0.feed_forward.w3.weight"),
      IREE_SV("layers.0.feed_forward.w2.weight"),
      IREE_SV("layers.1.attention.qkv.weight"),
      IREE_SV("layers.1.attention.o.weight"),
      IREE_SV("layers.1.feed_forward.w1.weight"),
      IREE_SV("layers.1.feed_forward.w3.weight"),
      IREE_SV("layers.1.feed_forward.w2.weight"),
      IREE_SV("layers.2.attention.qkv.weight"),
      IREE_SV("layers.2.attention.o.weight"),
      IREE_SV("layers.2.feed_forward.w1.weight"),
      IREE_SV("layers.2.feed_forward.w3.weight"),
      IREE_SV("layers.2.feed_forward.w2.weight"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_keys); ++i) {
    EXPECT_TRUE(iree_string_view_equal(rules.values[i].key, expected_keys[i]));
    EXPECT_TRUE(iree_string_view_equal(rules.values[i].source_scope,
                                       IREE_SV("native_fp8")));
    EXPECT_EQ(rules.values[i].storage,
              ID4_IDEOGRAM4_DIT_PARAMETER_STORAGE_FP8_E4M3_SCALED);
  }

  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &rules, iree_allocator_system());
}

}  // namespace
