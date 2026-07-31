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

static void ExpectRebased(qwen_program_span_t expected,
                          qwen_program_span_t actual,
                          iree_device_size_t base_offset) {
  EXPECT_EQ(actual.offset, base_offset + expected.offset);
  EXPECT_EQ(actual.length, expected.length);
}

static void ExpectRebasedLayer(const qwen_layer_program_layout_t& expected,
                               const qwen_layer_program_layout_t& actual,
                               iree_device_size_t base_offset) {
  ExpectRebased(expected.projection_input_scratch,
                actual.projection_input_scratch, base_offset);
  ExpectRebased(expected.raw_query, actual.raw_query, base_offset);
  ExpectRebased(expected.raw_key, actual.raw_key, base_offset);
  ExpectRebased(expected.raw_value, actual.raw_value, base_offset);
  ExpectRebased(expected.rotated_query, actual.rotated_query, base_offset);
  ExpectRebased(expected.attention_output, actual.attention_output,
                base_offset);
  ExpectRebased(expected.feed_forward_norm, actual.feed_forward_norm,
                base_offset);
  ExpectRebased(expected.router_logits, actual.router_logits, base_offset);
  ExpectRebased(expected.route_ids, actual.route_ids, base_offset);
  ExpectRebased(expected.route_weights, actual.route_weights, base_offset);
  ExpectRebased(expected.expert_table, actual.expert_table, base_offset);
  ExpectRebased(expected.partition_table, actual.partition_table, base_offset);
  ExpectRebased(expected.swiglu, actual.swiglu, base_offset);
  ExpectRebased(expected.routed_projection_scratch,
                actual.routed_projection_scratch, base_offset);
  EXPECT_EQ(actual.transient_byte_length, expected.transient_byte_length);
}

TEST(QwenProgramLayoutTest, PacksCompletePrefill512Layer) {
  qwen_layer_program_layout_t layout;
  IREE_ASSERT_OK(qwen_layer_program_layout_calculate(
      /*token_count=*/512, &layout));

  EXPECT_EQ(layout.projection_input_scratch.length,
            512u * 2048u * sizeof(float));
  EXPECT_EQ(layout.raw_query.length, 512u * 4096u * sizeof(float));
  EXPECT_EQ(layout.raw_key.length, 512u * 512u * sizeof(float));
  EXPECT_EQ(layout.raw_value.length, layout.raw_key.length);
  EXPECT_EQ(layout.feed_forward_norm.length, 512u * 2048u * sizeof(float));
  EXPECT_EQ(layout.expert_table.length, 128u * 513u * sizeof(int32_t));
  EXPECT_EQ(layout.partition_table.length, 257u * sizeof(int32_t));
  EXPECT_EQ(layout.swiglu.length, 512u * 8u * 768u * sizeof(float));
  EXPECT_EQ(layout.routed_projection_scratch.length, 512u * 8u * 2048u * 2u);

  ExpectOrdered(layout.projection_input_scratch, layout.raw_query);
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
  ExpectOrdered(layout.swiglu, layout.routed_projection_scratch);
  EXPECT_EQ(layout.transient_byte_length,
            layout.routed_projection_scratch.offset +
                layout.routed_projection_scratch.length);
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

  EXPECT_EQ(layout.projection_input_scratch.length, 8192u);
  EXPECT_GE(layout.projection_input_scratch.length, 4608u);
  EXPECT_EQ(layout.partition_table.length, 130u * sizeof(int32_t));
}

TEST(QwenProgramLayoutTest, PacksCompletePrefill512FullProgram) {
  qwen_layer_program_layout_t prefill_layer;
  IREE_ASSERT_OK(qwen_layer_program_layout_calculate(
      /*token_count=*/512, &prefill_layer));
  qwen_layer_program_layout_t terminal_layer;
  IREE_ASSERT_OK(qwen_layer_program_layout_calculate(
      /*token_count=*/1, &terminal_layer));

  qwen_full_program_layout_t layout;
  IREE_ASSERT_OK(qwen_full_program_layout_calculate(
      /*token_count=*/512, /*context_count=*/512,
      QWEN_FULL_PROGRAM_LAYOUT_FLAG_NONE, &layout));

  ExpectRebasedLayer(prefill_layer, layout.layer, /*base_offset=*/0);
  EXPECT_EQ(layout.layer.transient_byte_length, 65570560u);

  const iree_device_size_t terminal_layer_base =
      layout.layer.transient_byte_length;
  ExpectRebasedLayer(terminal_layer, layout.terminal_layer,
                     terminal_layer_base);
  EXPECT_EQ(layout.terminal_layer.projection_input_scratch.length, 8192u);
  EXPECT_EQ(layout.terminal_layer.raw_query.length, 16384u);
  EXPECT_EQ(layout.terminal_layer.raw_key.length, 2048u);
  EXPECT_EQ(layout.terminal_layer.raw_value.length, 2048u);
  EXPECT_EQ(layout.terminal_layer.rotated_query.length, 16384u);
  EXPECT_EQ(layout.terminal_layer.attention_output.length, 16384u);
  EXPECT_EQ(layout.terminal_layer.feed_forward_norm.length, 8192u);
  EXPECT_EQ(layout.terminal_layer.router_logits.length, 512u);
  EXPECT_EQ(layout.terminal_layer.route_ids.length, 32u);
  EXPECT_EQ(layout.terminal_layer.route_weights.length, 32u);
  EXPECT_EQ(layout.terminal_layer.expert_table.length, 1024u);
  EXPECT_EQ(layout.terminal_layer.partition_table.length, 520u);
  EXPECT_EQ(layout.terminal_layer.swiglu.length, 24576u);
  EXPECT_EQ(layout.terminal_layer.routed_projection_scratch.length, 32768u);
  EXPECT_EQ(layout.terminal_layer.transient_byte_length, 129792u);
  EXPECT_EQ(layout.attention_partial_maximums.length, 0u);
  EXPECT_EQ(layout.attention_partial_sums.length, 0u);
  EXPECT_EQ(layout.attention_partial_outputs.length, 0u);
  EXPECT_EQ(layout.decode_completion.initialization.length, 0u);
  EXPECT_EQ(layout.decode_completion.attention.length, 0u);
  EXPECT_EQ(layout.decode_completion.router.length, 0u);

  ExpectOrdered(layout.layer.routed_projection_scratch,
                layout.terminal_layer.projection_input_scratch);
  ExpectOrdered(layout.terminal_layer.projection_input_scratch,
                layout.terminal_layer.raw_query);
  ExpectOrdered(layout.terminal_layer.raw_query, layout.terminal_layer.raw_key);
  ExpectOrdered(layout.terminal_layer.raw_key, layout.terminal_layer.raw_value);
  ExpectOrdered(layout.terminal_layer.raw_value,
                layout.terminal_layer.rotated_query);
  ExpectOrdered(layout.terminal_layer.rotated_query,
                layout.terminal_layer.attention_output);
  ExpectOrdered(layout.terminal_layer.attention_output,
                layout.terminal_layer.feed_forward_norm);
  ExpectOrdered(layout.terminal_layer.feed_forward_norm,
                layout.terminal_layer.router_logits);
  ExpectOrdered(layout.terminal_layer.router_logits,
                layout.terminal_layer.route_ids);
  ExpectOrdered(layout.terminal_layer.route_ids,
                layout.terminal_layer.route_weights);
  ExpectOrdered(layout.terminal_layer.route_weights,
                layout.terminal_layer.expert_table);
  ExpectOrdered(layout.terminal_layer.expert_table,
                layout.terminal_layer.partition_table);
  ExpectOrdered(layout.terminal_layer.partition_table,
                layout.terminal_layer.swiglu);
  ExpectOrdered(layout.terminal_layer.swiglu,
                layout.terminal_layer.routed_projection_scratch);
  ExpectOrdered(layout.terminal_layer.routed_projection_scratch,
                layout.final_quantized_hidden_state);
  ExpectOrdered(layout.final_quantized_hidden_state, layout.vocabulary_logits);

  EXPECT_EQ(layout.final_quantized_hidden_state.offset, 65700352u);
  EXPECT_EQ(layout.final_quantized_hidden_state.length, 2304u);
  EXPECT_EQ(layout.vocabulary_logits.offset, 65702656u);
  EXPECT_EQ(layout.vocabulary_logits.length, 607744u);
  EXPECT_EQ(layout.transient_byte_length, 66310400u);
}

TEST(QwenProgramLayoutTest, ReservesReusableCompletionForDecode513) {
  qwen_full_program_layout_t layout;
  IREE_ASSERT_OK(qwen_full_program_layout_calculate(
      /*token_count=*/1, /*context_count=*/513,
      QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION |
          QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_FUSED_ROUTER,
      &layout));

  EXPECT_EQ(layout.attention_partial_maximums.length, 2304u);
  EXPECT_EQ(layout.attention_partial_sums.length, 2304u);
  EXPECT_EQ(layout.attention_partial_outputs.length, 147456u);
  EXPECT_EQ(layout.decode_completion.initialization.length, 20u);
  EXPECT_EQ(layout.decode_completion.attention.length, 16u);
  EXPECT_EQ(layout.decode_completion.router.length, 4u);
  EXPECT_EQ(layout.decode_completion.attention.offset,
            layout.decode_completion.initialization.offset);
  EXPECT_EQ(layout.decode_completion.router.offset,
            layout.decode_completion.attention.offset +
                layout.decode_completion.attention.length);
  ExpectOrdered(layout.terminal_layer.routed_projection_scratch,
                layout.attention_partial_maximums);
  ExpectOrdered(layout.attention_partial_maximums,
                layout.attention_partial_sums);
  ExpectOrdered(layout.attention_partial_sums,
                layout.attention_partial_outputs);
  ExpectOrdered(layout.attention_partial_outputs,
                layout.decode_completion.initialization);
  ExpectOrdered(layout.decode_completion.initialization,
                layout.final_quantized_hidden_state);
  EXPECT_EQ(layout.transient_byte_length, 1021952u);
}

TEST(QwenProgramLayoutTest, RejectsUnsupportedFullProgramTokenCount) {
  qwen_full_program_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        qwen_full_program_layout_calculate(
                            /*token_count=*/0, /*context_count=*/1,
                            QWEN_FULL_PROGRAM_LAYOUT_FLAG_NONE, &layout));
}

TEST(QwenProgramLayoutTest, RejectsUnsupportedFullProgramContextCount) {
  qwen_full_program_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      qwen_full_program_layout_calculate(
          /*token_count=*/1, /*context_count=*/0,
          QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_SPLIT_ATTENTION |
              QWEN_FULL_PROGRAM_LAYOUT_FLAG_DECODE_FUSED_ROUTER,
          &layout));
}

TEST(QwenProgramLayoutTest, RejectsUnsupportedFullProgramLayoutFlags) {
  qwen_full_program_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      qwen_full_program_layout_calculate(
          /*token_count=*/1, /*context_count=*/1,
          (qwen_full_program_layout_flags_t)(1u << 31), &layout));
}

}  // namespace
