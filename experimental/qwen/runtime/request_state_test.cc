// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/request_state.h"

#include "experimental/qwen/runtime/model_shape.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(QwenRequestStorageTest, PacksPrefill512WithoutOverlap) {
  qwen_request_storage_layout_t layout;
  IREE_ASSERT_OK(qwen_request_storage_layout_calculate(
      /*token_count=*/512, /*context_capacity=*/512, &layout));

  EXPECT_EQ(layout.hidden_state.offset, 0u);
  EXPECT_EQ(layout.hidden_state.length, 4u * 1024u * 1024u);
  EXPECT_GE(layout.positions.offset,
            layout.hidden_state.offset + layout.hidden_state.length);
  EXPECT_GE(layout.key_cache_indices.offset,
            layout.positions.offset + layout.positions.length);
  EXPECT_GE(layout.value_cache_indices.offset,
            layout.key_cache_indices.offset + layout.key_cache_indices.length);
  EXPECT_GE(
      layout.attention_mask.offset,
      layout.value_cache_indices.offset + layout.value_cache_indices.length);
  EXPECT_EQ(layout.reset_upload_byte_length,
            layout.attention_mask.offset + layout.attention_mask.length);
  EXPECT_GE(layout.key_cache.offset, layout.reset_upload_byte_length);
  EXPECT_GE(layout.value_cache.offset,
            layout.key_cache.offset + layout.key_cache.length);
  EXPECT_EQ(layout.persistent_byte_length,
            layout.value_cache.offset + layout.value_cache.length);
  EXPECT_EQ(layout.key_cache.length,
            layout.layer_cache_byte_length * QWEN_MODEL_LAYER_COUNT);
  EXPECT_EQ(layout.value_cache.length, layout.key_cache.length);
}

TEST(QwenRequestStorageTest, RejectsTokenCountBeyondCapacity) {
  qwen_request_storage_layout_t layout;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      qwen_request_storage_layout_calculate(
          /*token_count=*/512, /*context_capacity=*/128, &layout));
}

TEST(QwenRequestStorageTest, AlignsDecodeMaskForVectorLoads) {
  qwen_request_storage_layout_t layout;
  IREE_ASSERT_OK(qwen_request_storage_layout_calculate(
      /*token_count=*/1, /*context_capacity=*/512, &layout));

  EXPECT_EQ(layout.attention_mask.offset % 16, 0u);
}

}  // namespace
