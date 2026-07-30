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
          QWEN_LOOM_SOURCE_DENSE_LINEAR_QUANTIZED_F16,
          "qwen3_moe_dense_linear_quantized_f16.loom",
          "qwen3_moe_dense_linear_q4k_f16_wmma",
      },
      {
          QWEN_LOOM_SOURCE_ROUTER_PROJECTION_F32,
          "qwen3_moe_router_projection_f32.loom",
          "qwen3_moe_router_projection_f32_four_row_wave32",
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
          QWEN_LOOM_SOURCE_ROUTED_DOWN_F16,
          "qwen3_moe_routed_down_f16.loom",
          "qwen3_moe_routed_down_weighted_reduce_f16_f32",
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

}  // namespace
