// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/program_layout.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void ExpectOrdered(qwen_program_span_t lhs, qwen_program_span_t rhs) {
  EXPECT_GE(rhs.offset, lhs.offset + lhs.length);
}

TEST(QwenProgramLayoutTest, PacksCompletePrefill512Layer) {
  qwen_layer_program_layout_t layout;
  IREE_ASSERT_OK(qwen_layer_program_layout_calculate(
      /*token_count=*/512, &layout));

  EXPECT_EQ(layout.attention_projection_input.length,
            512u * 2048u * sizeof(float));
  EXPECT_EQ(layout.raw_query.length, 512u * 4096u * sizeof(float));
  EXPECT_EQ(layout.raw_key.length, 512u * 512u * sizeof(float));
  EXPECT_EQ(layout.raw_value.length, layout.raw_key.length);
  EXPECT_EQ(layout.feed_forward_norm.length, 512u * 2048u * sizeof(float));
  EXPECT_EQ(layout.expert_table.length, 128u * 513u * sizeof(int32_t));
  EXPECT_EQ(layout.partition_table.length, 257u * sizeof(int32_t));
  EXPECT_EQ(layout.swiglu.length, 512u * 8u * 768u * sizeof(float));
  EXPECT_EQ(layout.routed_down.length, 512u * 8u * 2048u * 2u);

  ExpectOrdered(layout.attention_projection_input, layout.raw_query);
  ExpectOrdered(layout.raw_query, layout.raw_key);
  ExpectOrdered(layout.raw_key, layout.raw_value);
  ExpectOrdered(layout.raw_value, layout.rotated_query);
  ExpectOrdered(layout.rotated_query, layout.attention_output);
  ExpectOrdered(layout.attention_output, layout.feed_forward_norm);
  ExpectOrdered(layout.feed_forward_norm, layout.router_logits);
  ExpectOrdered(layout.router_logits, layout.route_ids);
  ExpectOrdered(layout.route_ids, layout.route_weights);
  ExpectOrdered(layout.route_weights, layout.expert_table);
  ExpectOrdered(layout.expert_table, layout.partition_table);
  ExpectOrdered(layout.partition_table, layout.swiglu);
  ExpectOrdered(layout.swiglu, layout.routed_down);
  EXPECT_EQ(layout.transient_byte_length,
            layout.routed_down.offset + layout.routed_down.length);
}

TEST(QwenProgramLayoutTest, RejectsUnsupportedTokenCount) {
  qwen_layer_program_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        qwen_layer_program_layout_calculate(0, &layout));
}

TEST(QwenProgramLayoutTest, ReservesEveryExpertTailPartitionForDecode) {
  qwen_layer_program_layout_t layout;
  IREE_ASSERT_OK(
      qwen_layer_program_layout_calculate(/*token_count=*/1, &layout));

  EXPECT_EQ(layout.partition_table.length, 130u * sizeof(int32_t));
}

}  // namespace
