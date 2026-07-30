// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/model_shape.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(QwenModelShapeTest, ComputesPrefill512Storage) {
  iree_device_size_t hidden_state_byte_length = 0;
  IREE_ASSERT_OK(qwen_model_hidden_state_byte_length(
      /*token_count=*/512, &hidden_state_byte_length));
  EXPECT_EQ(hidden_state_byte_length, 4u * 1024u * 1024u);

  iree_device_size_t layer_cache_byte_length = 0;
  IREE_ASSERT_OK(qwen_model_layer_cache_byte_length(
      /*context_capacity=*/512, &layer_cache_byte_length));
  EXPECT_EQ(layer_cache_byte_length, 512u * 4u * 128u * 2u);

  iree_device_size_t all_cache_byte_length = 0;
  IREE_ASSERT_OK(qwen_model_all_cache_byte_length(
      /*context_capacity=*/512, &all_cache_byte_length));
  EXPECT_EQ(all_cache_byte_length, layer_cache_byte_length * 48u * 2u);
}

TEST(QwenModelShapeTest, RejectsUnsupportedDomains) {
  iree_device_size_t byte_length = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        qwen_model_hidden_state_byte_length(0, &byte_length));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      qwen_model_hidden_state_byte_length(
          QWEN_MODEL_MAX_PHYSICAL_TOKEN_COUNT + 1, &byte_length));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        qwen_model_layer_cache_byte_length(
                            QWEN_MODEL_MAX_CONTEXT_CAPACITY + 1, &byte_length));
}

}  // namespace
