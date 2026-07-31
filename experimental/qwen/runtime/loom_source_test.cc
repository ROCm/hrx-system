// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/loom_source.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

struct ExpectedSource {
  const char* module_path;
  const char* source_identifier;
  const char* export_name;
};

TEST(QwenLoomSourceTest, ResolvesEveryStableRuntimePath) {
  const ExpectedSource expected_sources[] = {
      {
          QWEN_LOOM_SOURCE_TOKEN_EMBEDDING_Q4K_BRINGUP_WORKAROUND,
          "qwen_token_embedding_q4k_bringup_workaround.loom",
          "qwen_token_embedding_q4k_bringup_workaround",
      },
      {
          QWEN_LOOM_SOURCE_GREEDY_ARGMAX_PARTIALS_BRINGUP_WORKAROUND,
          "qwen_greedy_argmax_partials_bringup_workaround.loom",
          "qwen_greedy_argmax_partials_bringup_workaround",
      },
      {
          QWEN_LOOM_SOURCE_ATTENTION_METADATA,
          "qwen_attention_metadata.loom",
          "qwen_attention_metadata_bringup_workaround",
      },
      {
          QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED,
          "qwen3_moe_attention_prepare_quantized.loom",
          "qwen3_moe_rmsnorm_f32",
      },
      {
          QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED,
          "qwen3_moe_attention_qkv_quantized.loom",
          "qwen3_moe_attention_qkv_quantized",
      },
      {
          QWEN_LOOM_SOURCE_ATTENTION_QKV_POSTPROCESS_FUSED,
          "qwen3_moe_attention_qkv_postprocess_fused.loom",
          "qwen3_moe_attention_qkv_postprocess_fused_decode",
      },
      {
          QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16,
          "qwen3_moe_attention_postprocess_f32_f16.loom",
          "qwen3_moe_attention_postprocess_f32_f16",
      },
      {
          QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16,
          "qwen3_moe_flash_attention_prefill_f32_f16.loom",
          "qwen3_moe_flash_attention_f32_f16_wmma",
      },
      {
          QWEN_LOOM_SOURCE_FLASH_ATTENTION_DECODE_SPLIT_F32_F16,
          "qwen3_moe_flash_attention_decode_split_f32_f16.loom",
          "qwen3_moe_flash_attention_decode_split_f32_f16_wmma",
      },
      {
          QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16,
          "qwen3_moe_dense_linear_quantized_f16.loom",
          "qwen3_moe_dense_linear_q4k_f16_wmma",
      },
      {
          QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16,
          "qwen3_moe_dense_linear_quantized_f16.loom",
          "qwen3_moe_dense_linear_q4k_q8_1_x4",
      },
      {
          QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16,
          "qwen3_moe_dense_linear_quantized_f16.loom",
          "qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8",
      },
      {
          QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16,
          "qwen3_moe_dense_linear_quantized_f16.loom",
          "qwen3_moe_dense_linear_q6k_f16_wmma",
      },
      {
          QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4_BRINGUP_WORKAROUND,
          "qwen_quantize_q8_1_x4_bringup_workaround.loom",
          "ggml_quantize_q8_1_x4_f32",
      },
      {
          QWEN_LOOM_SOURCE_VOCABULARY_PROJECTION_Q6,
          "qwen3_moe_vocabulary_projection_q6.loom",
          "ggml_linear_q6k_q8_1_x4_partial_argmax",
      },
      {
          QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32,
          "qwen3_moe_router_projection_f32.loom",
          "qwen3_moe_router_projection_f32_four_row_wave32",
      },
      {
          QWEN_LOOM_SOURCE_ROUTER_PROJECTION_TOP8_FUSED_F32,
          "qwen3_moe_router_projection_top8_fused_f32.loom",
          "qwen3_moe_router_projection_top8_fused_decode_f32",
      },
      {
          QWEN_LOOM_SOURCE_ROUTER_TOP8_F32,
          "qwen3_moe_router_top8_f32.loom",
          "qwen3_moe_router_top8_f32",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16,
          "qwen3_moe_routed_gate_up_f16.loom",
          "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8,
          "qwen3_moe_routed_gate_up_q8.loom",
          "qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_DOWN_F16,
          "qwen3_moe_routed_down_f16.loom",
          "qwen3_moe_routed_down_weighted_reduce_f16_f32",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8,
          "qwen3_moe_routed_down_q4_q8.loom",
          "qwen3_moe_routed_down_q4k_q8_1_x4_next_q8",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_Q8,
          "qwen3_moe_routed_down_q6_q8.loom",
          "qwen3_moe_routed_down_q6k_q8_1_x4_next_q8",
      },
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

    std::string source_text(
        reinterpret_cast<const char*>(source_module.source_contents.data),
        source_module.source_contents.data_length);
    EXPECT_NE(source_text.find("export(\"" + std::string(expected.export_name) +
                               "\")"),
              std::string::npos);
    EXPECT_EQ(source_text.find("check.case"), std::string::npos);
  }
}

TEST(QwenLoomSourceTest, RejectsUnknownPath) {
  qwen_loom_source_module_t source_module;
  iree::Status status = qwen_loom_source_lookup(
      IREE_SV("qwen3_moe/not_a_module"), &source_module);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kNotFound));
}

TEST(QwenLoomSourceTest, EmbedsBoundedVocabularyWorkaroundSources) {
  qwen_loom_source_module_t quantize_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4_BRINGUP_WORKAROUND),
      &quantize_source));
  std::string quantize_text(
      reinterpret_cast<const char*>(quantize_source.source_contents.data),
      quantize_source.source_contents.data_length);
  EXPECT_NE(quantize_text.find(
                "%input_packet_end = index.sub %element_count, %three : index"),
            std::string::npos);
  EXPECT_NE(
      quantize_text.find("%input_index = index.assume %input_index0 "
                         "[lt(%input_index0, %input_packet_end)] : index"),
      std::string::npos);

  qwen_loom_source_module_t projection_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_VOCABULARY_PROJECTION_Q6), &projection_source));
  std::string projection_text(
      reinterpret_cast<const char*>(projection_source.source_contents.data),
      projection_source.source_contents.data_length);
  EXPECT_NE(
      projection_text.find(
          "%bounded_input_size = index.assume %input_size "
          "[range(%input_size, 256, 2048), mul(%input_size, 256)] : index"),
      std::string::npos);
  EXPECT_NE(projection_text.find(
                "%safe_channel0 = scf.select %valid_channel, %channel, "
                "%zero : index"),
            std::string::npos);
  EXPECT_NE(
      projection_text.find("%bounded_token_count = index.assume %token_count "
                           "[range(%token_count, 1, 1)] : index"),
      std::string::npos);
  EXPECT_NE(
      projection_text.find("%partial_count = index.assume %partial_count0 "
                           "[range(%partial_count0, 1, 18992)] : index"),
      std::string::npos);
  EXPECT_NE(projection_text.find(
                "%partial_index, %partial_bound = index.assume "
                "%partial_index0, %partial_count "
                "[lt(%partial_index0, %partial_count)] : index, index"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsDirectGateUpNextQ8FixedModelWorkaround) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8), &source_module));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("%bounded_token_count = index.assume %token_count "
                             "[range(%token_count, 1, 1)] : index"),
            std::string::npos);
  EXPECT_NE(source_text.find("%bounded_route_count = index.assume %route_count "
                             "[range(%route_count, 8, 8)] : index"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%bounded_route_stride = index.assume %route_stride "
                       "[range(%route_stride, 8, 8)] : index"),
      std::string::npos);
  EXPECT_NE(
      source_text.find("func.apply<qwen3_moe.routed_gate_up.q4k_q8.body>"),
      std::string::npos);
  EXPECT_NE(source_text.find("func.apply<ggml.quantize_q8_1_x4.group_body>"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsDirectDownFixedModelWorkarounds) {
  const char* module_paths[] = {
      QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8,
      QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_Q8,
  };
  for (const char* module_path : module_paths) {
    SCOPED_TRACE(module_path);
    qwen_loom_source_module_t source_module;
    IREE_ASSERT_OK(qwen_loom_source_lookup(iree_make_cstring_view(module_path),
                                           &source_module));
    std::string source_text(
        reinterpret_cast<const char*>(source_module.source_contents.data),
        source_module.source_contents.data_length);
    EXPECT_NE(
        source_text.find("%bounded_token_count = index.assume %token_count "
                         "[range(%token_count, 1, 1)] : index"),
        std::string::npos);
    EXPECT_NE(
        source_text.find("%bounded_route_count = index.assume %route_count "
                         "[range(%route_count, 8, 8)] : index"),
        std::string::npos);
    EXPECT_NE(
        source_text.find("%bounded_output_size = index.assume %output_size "
                         "[range(%output_size, 2048, 2048), "
                         "mul(%output_size, 128)] : index"),
        std::string::npos);
    EXPECT_NE(
        source_text.find("func.apply<qwen3_moe.rmsnorm_quantize_q8_1_x4.body>"),
        std::string::npos);
  }
}

TEST(QwenLoomSourceTest, EmbedsWorkaroundFlashAttentionSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_FLASH_ATTENTION_PREFILL_F32_F16),
      &source_module));

  EXPECT_TRUE(iree_string_view_equal(
      source_module.source_identifier,
      IREE_SV("qwen3_moe_flash_attention_prefill_f32_f16.loom")));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(
      source_text.find("%tail_key_value_token_count = index.rem "
                       "%bounded_key_value_token_count, %sixtyfour : index"),
      std::string::npos);
  EXPECT_NE(source_text.find(
                "%full_tile_key_value_token_count = index.assume "
                "%bounded_key_value_token_count "
                "[range(%bounded_key_value_token_count, 64, 32768)] : index"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%last_full_key_tile_start = index.sub "
                       "%full_tile_key_value_token_count, %fifteen : index"),
      std::string::npos);
  EXPECT_NE(
      source_text.find("%tail_key_count = scf.select %is_first_tail_tile, "
                       "%first_tail_key_count, %second_tail_key_count : index"),
      std::string::npos);
  EXPECT_NE(
      source_text.find("%tail_key_value_stage = buffer.alloca "
                       "%tail_key_value_stage_capacity "
                       "{base_alignment = 16, memory_space = workgroup} : "
                       "buffer"),
      std::string::npos);
  EXPECT_EQ(source_text.find("%tail_key_value_stage = buffer.alloca "
                             "%tail_key_value_stage_bytes"),
            std::string::npos);
  size_t subtraction_position = source_text.find("index.sub");
  ASSERT_NE(subtraction_position, std::string::npos);
  EXPECT_EQ(source_text.find("index.sub", subtraction_position + 1),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsWorkaroundRouterProjectionSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32), &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find(
                "export(\"qwen3_moe_router_projection_f32_four_row_wave32\") "
                "@qwen3_moe_router_projection_f32_four_row_wave32"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%vector_channel_end = index.sub %hidden_size, %three : index"),
            std::string::npos);
  EXPECT_NE(source_text.find("[%lane_channel to %vector_channel_end step "
                             "%onetwentyeight]"),
            std::string::npos);
  EXPECT_NE(source_text.find("%vector_channel = index.assume %channel "
                             "[lt(%channel, %vector_channel_end)] : index"),
            std::string::npos);
  EXPECT_EQ(
      source_text.find("[%lane_channel to %hidden_size step %onetwentyeight]"),
      std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsWorkaroundRouterTop8Source) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTER_TOP8_F32), &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("%route_id_storage_count = index.mul "
                             "%launch_token_count, %route_count : index"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%route_id_token_base = index.mul %token, %route_count : "
                "index"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%route_id_storage_count = index.mul "
                             "%launch_token_count, %bounded_route_id_stride"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%route_id_token_base = index.mul %token, "
                             "%bounded_route_id_stride"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "kernel.launch.config workgroups(%wide_workgroup_count, %one, "
                "%one) workgroup_size(%wide_workgroup_size, %one, %one)"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%token_base = index.mul %token_workgroup, %four : index"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%workgroup_count = scf.select"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%subgroup_count = scf.select"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsWorkaroundFusedRouterSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_TOP8_FUSED_F32),
      &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("%route_count = config.get "
                             "@qwen3_moe.router.route_count"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%route_count, %logits_noalias, %route_ids_noalias"),
      std::string::npos);
  EXPECT_EQ(source_text.find("%bounded_route_id_stride, %logits_noalias, "
                             "%route_ids_noalias"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsWorkaroundExpertTableSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16), &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("export(\"qwen3_moe_build_expert_table\") "
                             "@qwen3_moe_build_expert_table"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%assignment_count = index.mul %bounded_token_count, %eight : "
                "index"),
            std::string::npos);
  EXPECT_NE(source_text.find("%token0 = index.div %assignment, %eight : index"),
            std::string::npos);
  EXPECT_NE(source_text.find("%route0 = index.rem %assignment, %eight : index"),
            std::string::npos);
  EXPECT_EQ(
      source_text.find("%token0 = index.div %assignment, %bounded_route_count"),
      std::string::npos);
  EXPECT_EQ(
      source_text.find("%route0 = index.rem %assignment, %bounded_route_count"),
      std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsDecodeSplitFixedContextWorkaround) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_FLASH_ATTENTION_DECODE_SPLIT_F32_F16),
      &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(
      source_text.find("[range(%key_value_token_count, 513, 513)] : index"),
      std::string::npos);
  EXPECT_NE(
      source_text.find("kernel.launch.config workgroups(%nine, %four, %one)"),
      std::string::npos);
  EXPECT_NE(source_text.find(
                "func.apply<qwen3_moe.attention.decode_split.reduce_fused>"),
            std::string::npos);
}

}  // namespace
