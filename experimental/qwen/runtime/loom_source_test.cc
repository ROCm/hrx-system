// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/loom_source.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

struct ExpectedSource {
  const char* module_path;
  const char* source_identifier;
};

TEST(QwenLoomSourceTest, ResolvesEveryStableRuntimePath) {
  const ExpectedSource expected_sources[] = {
      {QWEN_LOOM_SOURCE_TOKEN_EMBEDDING_Q4K, "qwen_token_embedding_q4k.loom"},
      {QWEN_LOOM_SOURCE_GREEDY_ARGMAX_PARTIALS,
       "qwen_greedy_argmax_partials.loom"},
      {QWEN_LOOM_SOURCE_ATTENTION_METADATA, "qwen_attention_metadata.loom"},
      {QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED,
       "qwen3_moe_attention_prepare_quantized.loom"},
      {QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED,
       "qwen3_moe_attention_qkv_quantized.loom"},
      {QWEN_LOOM_SOURCE_ATTENTION_QKV_POSTPROCESS_FUSED,
       "qwen3_moe_attention_qkv_postprocess_fused.loom"},
      {QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16,
       "qwen3_moe_attention_postprocess_f32_f16.loom"},
      {QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16,
       "qwen3_moe_flash_attention_prefill_f32_f16.loom"},
      {QWEN_LOOM_SOURCE_FLASH_ATTENTION_DECODE_SPLIT_F32_F16,
       "qwen3_moe_flash_attention_decode_split_f32_f16.loom"},
      {QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16,
       "qwen3_moe_dense_linear_quantized_f16.loom"},
      {QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4, "qwen3_moe_quantize_q8_1_x4.loom"},
      {QWEN_LOOM_SOURCE_VOCABULARY_PROJECTION_Q6,
       "qwen3_moe_vocabulary_projection_q6.loom"},
      {QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32,
       "qwen3_moe_router_projection_f32.loom"},
      {QWEN_LOOM_SOURCE_ROUTER_PROJECTION_TOP8_FUSED_F32,
       "qwen3_moe_router_projection_top8_fused_f32.loom"},
      {QWEN_LOOM_SOURCE_ROUTER_TOP8_F32, "qwen3_moe_router_top8_f32.loom"},
      {QWEN_LOOM_SOURCE_ROUTE_TRACE, "qwen3_moe_route_trace.loom"},
      {QWEN_LOOM_SOURCE_EXPERT_TABLE_PARTITION_FUSED,
       "qwen3_moe_expert_table_partition_fused.loom"},
      {QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16,
       "qwen3_moe_routed_gate_up_f16.loom"},
      {QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8, "qwen3_moe_routed_gate_up_q8.loom"},
      {QWEN_LOOM_SOURCE_ROUTED_DOWN_F16, "qwen3_moe_routed_down_f16.loom"},
      {QWEN_LOOM_SOURCE_ROUTED_DOWN_NEXT_RMSNORM_F32,
       "qwen3_moe_routed_down_next_rmsnorm_f32.loom"},
      {QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8, "qwen3_moe_routed_down_q4_q8.loom"},
      {QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_F32,
       "qwen3_moe_routed_down_q6_f32.loom"},
  };

  for (const ExpectedSource& expected : expected_sources) {
    SCOPED_TRACE(expected.module_path);
    qwen_loom_source_module_t source_module;
    IREE_ASSERT_OK(qwen_loom_source_lookup(
        iree_make_cstring_view(expected.module_path), &source_module));
    EXPECT_TRUE(
        iree_string_view_equal(source_module.module_path,
                               iree_make_cstring_view(expected.module_path)));
    EXPECT_TRUE(iree_string_view_equal(
        source_module.source_identifier,
        iree_make_cstring_view(expected.source_identifier)));
    ASSERT_NE(source_module.source_contents.data, nullptr);
    ASSERT_GT(source_module.source_contents.data_length, 0u);
  }
}

TEST(QwenLoomSourceTest, RejectsUnknownPath) {
  qwen_loom_source_module_t source_module;
  iree::Status status = qwen_loom_source_lookup(
      IREE_SV("qwen3_moe/not_a_module"), &source_module);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kNotFound));
}

}  // namespace
