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
          "qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8",
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
          QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4,
          "qwen3_moe_quantize_q8_1_x4.loom",
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
          QWEN_LOOM_SOURCE_ROUTE_TRACE,
          "qwen3_moe_route_trace.loom",
          "qwen3_moe_route_trace_capture",
      },
      {
          QWEN_LOOM_SOURCE_EXPERT_TABLE_PARTITION_FUSED,
          "qwen3_moe_expert_table_partition_fused.loom",
          "qwen3_moe_build_expert_table_partition_prefill_512",
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
          QWEN_LOOM_SOURCE_ROUTED_DOWN_NEXT_RMSNORM_F32,
          "qwen3_moe_routed_down_next_rmsnorm_f32.loom",
          "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8,
          "qwen3_moe_routed_down_q4_q8.loom",
          "qwen3_moe_routed_down_q4k_q8_1_x4_next_q8",
      },
      {
          QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_F32,
          "qwen3_moe_routed_down_q6_f32.loom",
          "qwen3_moe_routed_down_q6k_f32_wave64_next_q8",
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

TEST(QwenLoomSourceTest, EmbedsConfiguredQwenLaunchCapacities) {
  qwen_loom_source_module_t token_embedding_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_TOKEN_EMBEDDING_Q4K_BRINGUP_WORKAROUND),
      &token_embedding_source));
  std::string token_embedding_text(
      reinterpret_cast<const char*>(
          token_embedding_source.source_contents.data),
      token_embedding_source.source_contents.data_length);
  EXPECT_NE(token_embedding_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(token_embedding_text.find(
                "%workgroup_count = index.mul %token_capacity, %two"),
            std::string::npos);

  qwen_loom_source_module_t attention_metadata_source;
  IREE_ASSERT_OK(
      qwen_loom_source_lookup(IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_METADATA),
                              &attention_metadata_source));
  std::string attention_metadata_text(
      reinterpret_cast<const char*>(
          attention_metadata_source.source_contents.data),
      attention_metadata_source.source_contents.data_length);
  EXPECT_NE(
      attention_metadata_text.find("@qwen.attention.metadata_context_capacity"),
      std::string::npos);
  EXPECT_NE(attention_metadata_text.find(
                "workgroups(%key_workgroup_count, %token_capacity, %one)"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalVocabularyEndpointSources) {
  qwen_loom_source_module_t quantize_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4), &quantize_source));
  std::string quantize_text(
      reinterpret_cast<const char*>(quantize_source.source_contents.data),
      quantize_source.source_contents.data_length);
  EXPECT_NE(quantize_text.find("%group_count0 = index.div %element_count, "
                               "%elements_per_group : index"),
            std::string::npos);
  EXPECT_NE(quantize_text.find("le(%group_count0, %group_capacity)"),
            std::string::npos);
  EXPECT_NE(
      quantize_text.find("%launched_element_count = index.mul %group_count, "
                         "%onetwentyeight : index"),
      std::string::npos);
  EXPECT_EQ(quantize_text.find("%input_packet_end"), std::string::npos);

  qwen_loom_source_module_t projection_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_VOCABULARY_PROJECTION_Q6), &projection_source));
  std::string projection_text(
      reinterpret_cast<const char*>(projection_source.source_contents.data),
      projection_source.source_contents.data_length);
  EXPECT_NE(projection_text.find("func.apply<ggml.linear_q6k_q8_1_x4.body>"),
            std::string::npos);
  EXPECT_NE(projection_text.find("@ggml.linear_q6k_q8_1_x4.output_capacity"),
            std::string::npos);
  EXPECT_NE(projection_text.find("workgroups(%output_tiles, %one, %one)"),
            std::string::npos);
  EXPECT_NE(projection_text.find("%publish_output = scalar.constant true"),
            std::string::npos);
  EXPECT_NE(projection_text.find(
                "body>(%publish_output, %bounded_token_count, %token0"),
            std::string::npos);
  EXPECT_EQ(projection_text.find("%safe_channel"), std::string::npos);
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

TEST(QwenLoomSourceTest, EmbedsCapacityBoundDirectGateUpSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_Q8), &source_module));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(source_text.find("workgroups(%configured_output_size, "
                             "%configured_route_count, %token_capacity)"),
            std::string::npos);
  EXPECT_NE(source_text.find("le(%token_count, %token_capacity)"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("body>(%valid_token, %body_token_count, %safe_token"),
      std::string::npos);
  EXPECT_NE(source_text.find("export(\"qwen3_moe_routed_gate_up_swiglu_"
                             "q4k_q8\")"),
            std::string::npos);
  EXPECT_NE(source_text.find("func.apply<ggml.quantize_q8_1_x4.group_body>"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalQ4DirectDownSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_Q4_Q8), &source_module));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("@qwen3_moe.routed_down.output_size"),
            std::string::npos);
  EXPECT_NE(source_text.find("workgroups(%output_tiles, %one, %one)"),
            std::string::npos);
  EXPECT_NE(source_text.find("eq(%output_size, %configured_output_size0)"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("func.apply<qwen3_moe.routed_down.q4k_q8_1_x4.body>"),
      std::string::npos);
  EXPECT_NE(
      source_text.find("func.apply<qwen3_moe.routed_down.next_q8_completion>"),
      std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalDirectF32Q6DownSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTED_DOWN_Q6_F32), &source_module));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("@qwen3_moe.routed_down.output_size"),
            std::string::npos);
  EXPECT_NE(source_text.find("workgroups(%output_tiles, %one, %one)"),
            std::string::npos);
  EXPECT_NE(source_text.find("eq(%route_count, %configured_route_count0)"),
            std::string::npos);
  EXPECT_NE(source_text.find("func.apply<qwen3_moe.routed_down.q6k_f32.body>"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("func.apply<qwen3_moe.routed_down.next_q8_completion>"),
      std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCapacityBoundAttentionPreparationSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_PREPARE_QUANTIZED), &source_module));

  EXPECT_TRUE(iree_string_view_equal(
      source_module.source_identifier,
      IREE_SV("qwen3_moe_attention_prepare_quantized.loom")));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(source_text.find("workgroups(%token_capacity, %one, %one)"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%safe_token0 = scf.select %valid_token, %token0, %zero"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%publishes_normalized = scalar.andi %publish_normalized, "
                "%valid_token"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCapacityBoundQ8PackerSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_QUANTIZE_Q8_1_X4), &source_module));

  EXPECT_TRUE(
      iree_string_view_equal(source_module.source_identifier,
                             IREE_SV("qwen3_moe_quantize_q8_1_x4.loom")));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("@ggml.quantize_q8_1_x4.group_capacity"),
            std::string::npos);
  EXPECT_NE(source_text.find("workgroups(%group_capacity, %unit, %unit)"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%safe_group = scf.select %valid_group, %group0, %zero"),
      std::string::npos);
  EXPECT_NE(source_text.find("func.apply<ggml.quantize_q8_1_x4.group_body>"
                             "(%valid_group"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCapacityBoundAttentionProjectionSources) {
  qwen_loom_source_module_t qkv_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_QKV_QUANTIZED), &qkv_source));
  std::string qkv_text(
      reinterpret_cast<const char*>(qkv_source.source_contents.data),
      qkv_source.source_contents.data_length);
  EXPECT_NE(qkv_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(qkv_text.find("workgroups(%output_tiles, %token_capacity, %one)"),
            std::string::npos);
  EXPECT_NE(
      qkv_text.find("%safe_token = scf.select %valid_token, %token0, %zero"),
      std::string::npos);

  qwen_loom_source_module_t postprocess_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ATTENTION_POSTPROCESS_F32_F16),
      &postprocess_source));
  std::string postprocess_text(
      reinterpret_cast<const char*>(postprocess_source.source_contents.data),
      postprocess_source.source_contents.data_length);
  EXPECT_NE(postprocess_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(postprocess_text.find(
                "workgroups(%head_domain_count, %token_capacity, %one)"),
            std::string::npos);
  EXPECT_NE(postprocess_text.find(
                "%safe_token = scf.select %valid_token, %token0, %zero"),
            std::string::npos);

  qwen_loom_source_module_t dense_source;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16), &dense_source));
  std::string dense_text(
      reinterpret_cast<const char*>(dense_source.source_contents.data),
      dense_source.source_contents.data_length);
  EXPECT_NE(dense_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(dense_text.find("%padded_token_count = index.add %token_capacity, "
                            "%thirtyone"),
            std::string::npos);
  EXPECT_NE(dense_text.find("workgroups(%output_tiles, %token_capacity, %one)"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalFlashAttentionSource) {
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
  EXPECT_NE(source_text.find("%tail_key_value_token_count = index.sub "
                             "%bounded_key_value_token_count, "
                             "%full_key_value_token_count : index"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%last_full_key_tile_start = index.sub "
                       "%bounded_key_value_token_count, %fifteen : index"),
      std::string::npos);
  EXPECT_NE(
      source_text.find("%tail_key_value_stage = buffer.alloca "
                       "%tail_key_value_stage_bytes "
                       "{base_alignment = 16, memory_space = workgroup} : "
                       "buffer"),
      std::string::npos);
  EXPECT_EQ(source_text.find("%tail_key_value_stage = buffer.alloca "
                             "%tail_key_value_stage_capacity"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalRouterProjectionSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32), &source_module));

  EXPECT_TRUE(
      iree_string_view_equal(source_module.source_identifier,
                             IREE_SV("qwen3_moe_router_projection_f32.loom")));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find(
                "export(\"qwen3_moe_router_projection_f32_four_row_wave32\") "
                "@qwen3_moe_router_projection_f32_four_row_wave32"),
            std::string::npos);
  EXPECT_NE(source_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("workgroups(%expert_tiles, %token_capacity, %one)"),
      std::string::npos);
  EXPECT_NE(source_text.find("%bounded_token_count = index.assume %token_count "
                             "[range(%token_count, 1, 2048), "
                             "le(%token_count, %token_capacity)]"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%full_channel_limit = index.sub %hidden_size, %three"),
      std::string::npos);
  EXPECT_NE(source_text.find("[%lane_channel to %full_channel_limit step "
                             "%onetwentyeight]"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%vector_channel_end"), std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalRouterTop8Source) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTER_TOP8_F32), &source_module));

  EXPECT_TRUE(
      iree_string_view_equal(source_module.source_identifier,
                             IREE_SV("qwen3_moe_router_top8_f32.loom")));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("le(%route_count0, %bounded_route_id_stride0)"),
            std::string::npos);
  EXPECT_NE(source_text.find("%route_id_storage_count = index.mul "
                             "%launch_token_count, %bounded_route_id_stride"),
            std::string::npos);
  EXPECT_NE(source_text.find("%route_id_token_base = index.mul %safe_token, "
                             "%bounded_route_id_stride"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%uses_decode_geometry = index.cmp eq, %token_capacity, "
                       "%one"),
      std::string::npos);
  EXPECT_NE(source_text.find("%workgroup_count = scf.select"),
            std::string::npos);
  EXPECT_NE(source_text.find("%subgroup_count = scf.select"),
            std::string::npos);
  EXPECT_NE(source_text.find("%bounded_token_count = index.assume %token_count "
                             "[range(%token_count, 1, 2048), "
                             "le(%token_count, %token_capacity)]"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%safe_token = scf.select %valid_token, %token0, %zero"),
      std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalFusedRouterSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTER_PROJECTION_TOP8_FUSED_F32),
      &source_module));

  EXPECT_TRUE(iree_string_view_equal(
      source_module.source_identifier,
      IREE_SV("qwen3_moe_router_projection_top8_fused_f32.loom")));
  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find(
                "export(\"qwen3_moe_router_projection_top8_fused_decode_f32\") "
                "@qwen3_moe_router_projection_top8_fused_decode_f32"),
            std::string::npos);
  EXPECT_NE(source_text.find("%bounded_route_id_stride, %logits_noalias, "
                             "%route_ids_noalias"),
            std::string::npos);
  EXPECT_NE(source_text.find("le(%route_count0, %bounded_route_id_stride0)"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalExpertTableSource) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_ROUTED_GATE_UP_F16), &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("export(\"qwen3_moe_build_expert_table\") "
                             "@qwen3_moe_build_expert_table"),
            std::string::npos);
  EXPECT_NE(source_text.find("@qwen3_moe.routed_gate_up.expert_count"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("workgroups(%configured_expert_count, %one, %one)"),
      std::string::npos);
  EXPECT_NE(source_text.find("eq(%expert_count, "
                             "%configured_expert_count0)"),
            std::string::npos);
  EXPECT_NE(
      source_text.find("%assignment_count = index.mul %bounded_token_count, "
                       "%bounded_route_count : index"),
      std::string::npos);
  EXPECT_NE(source_text.find("%token0 = index.div %assignment, "
                             "%configured_route_count : index"),
            std::string::npos);
  EXPECT_NE(source_text.find("%route0 = index.rem %assignment, "
                             "%configured_route_count : index"),
            std::string::npos);
  EXPECT_NE(source_text.find("@qwen3_moe.workload.token_capacity"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "%assignment_count = index.mul %token_capacity, %route_count"),
            std::string::npos);
  EXPECT_NE(source_text.find("le(%token_count, %token_capacity)"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsCanonicalDecodeSplitCapacityContract) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_FLASH_ATTENTION_DECODE_SPLIT_F32_F16),
      &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("config.decl "
                             "@qwen3_moe.attention."
                             "key_value_token_capacity"),
            std::string::npos);
  EXPECT_NE(source_text.find("%key_value_block_count = index.div "
                             "%key_value_token_capacity, %sixtyfour"),
            std::string::npos);
  EXPECT_NE(source_text.find("le(%key_value_token_count_in_range, "
                             "%key_value_token_capacity)"),
            std::string::npos);
  EXPECT_NE(source_text.find("le(%key_value_token_capacity, "
                             "%padded_key_value_token_count)"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "func.apply<qwen3_moe.attention.decode_split.produce_partials>("
                "%bounded_key_value_token_count, %query"),
            std::string::npos);
  EXPECT_NE(source_text.find(
                "func.apply<qwen3_moe.attention.decode_split.reduce_fused>("
                "%launch_key_value_token_capacity, %publish_q8"),
            std::string::npos);
  EXPECT_NE(source_text.find("func.apply<qwen3_moe.attention.decode_split."
                             "pack_completed_q8>"),
            std::string::npos);
  EXPECT_NE(source_text.find("%next_q8_output: buffer"), std::string::npos);
  EXPECT_EQ(source_text.find("%use_direct_reducer"), std::string::npos);
  EXPECT_EQ(source_text.find("func.call inline "
                             "@qwen3_moe_flash_attention_decode_split_"
                             "reduce_fused_"),
            std::string::npos);
  EXPECT_EQ(source_text.find("func.call inline "
                             "@qwen3_moe_flash_attention_decode_pack_"
                             "completed_"),
            std::string::npos);
  EXPECT_EQ(source_text.find("@qwen.decode.key_value_capacity"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%control_view"), std::string::npos);
  EXPECT_EQ(source_text.find("%active_key_value_token_count"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsDynamicGreedyContinuationPosition) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(
      IREE_SV(QWEN_LOOM_SOURCE_GREEDY_ARGMAX_PARTIALS_BRINGUP_WORKAROUND),
      &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("%current_context_base_raw = "
                             "view.load %next_control_view"),
            std::string::npos);
  EXPECT_NE(source_text.find("%next_context_base0 = index.add "
                             "%current_context_base, "
                             "%bounded_context_increment : index"),
            std::string::npos);
  EXPECT_EQ(source_text.find("%bounded_next_context_base = index.assume"),
            std::string::npos);
}

TEST(QwenLoomSourceTest, EmbedsPositionDynamicRouteTrace) {
  qwen_loom_source_module_t source_module;
  IREE_ASSERT_OK(qwen_loom_source_lookup(IREE_SV(QWEN_LOOM_SOURCE_ROUTE_TRACE),
                                         &source_module));

  std::string source_text(
      reinterpret_cast<const char*>(source_module.source_contents.data),
      source_module.source_contents.data_length);
  EXPECT_NE(source_text.find("%context_base_raw = "
                             "view.load %control_view"),
            std::string::npos);
  EXPECT_NE(source_text.find("%logical_token_base = index.add %context_base, "
                             "%bounded_source_token_offset : index"),
            std::string::npos);
  EXPECT_NE(source_text.find("%logical_token0 = index.add "
                             "%logical_token_base, %token : index"),
            std::string::npos);
  EXPECT_EQ(source_text.find("config.get"), std::string::npos);
  EXPECT_EQ(source_text.find("qwen3_moe_route_trace_inspect"),
            std::string::npos);
}

}  // namespace
