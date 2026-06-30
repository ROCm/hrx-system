// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/stages/vae_parameters.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(VaeParameters, FormatsAndParsesPackedConv3x3WeightKey) {
  char key_storage[128];
  iree_string_view_t virtual_key = iree_string_view_empty();
  IREE_ASSERT_OK(id4_vae_parameter_format_packed_conv3x3_weight_key(
      IREE_SV("decoder.mid.block_1.conv1.weight"), key_storage,
      IREE_ARRAYSIZE(key_storage), &virtual_key));

  iree_string_view_t source_key = iree_string_view_empty();
  EXPECT_TRUE(id4_vae_parameter_parse_packed_conv3x3_weight_key(virtual_key,
                                                                &source_key));
  EXPECT_TRUE(iree_string_view_equal(
      source_key, IREE_SV("decoder.mid.block_1.conv1.weight")));
}

TEST(VaeParameters, FormatsAndParsesPackedUpsampleConv3x3WeightKey) {
  char key_storage[128];
  iree_string_view_t virtual_key = iree_string_view_empty();
  IREE_ASSERT_OK(id4_vae_parameter_format_packed_upsample_conv3x3_weight_key(
      IREE_SV("decoder.up.2.upsample.conv.weight"), key_storage,
      IREE_ARRAYSIZE(key_storage), &virtual_key));

  iree_string_view_t source_key = iree_string_view_empty();
  EXPECT_TRUE(id4_vae_parameter_parse_packed_upsample_conv3x3_weight_key(
      virtual_key, &source_key));
  EXPECT_TRUE(iree_string_view_equal(
      source_key, IREE_SV("decoder.up.2.upsample.conv.weight")));
}

TEST(VaeParameters, FormatsAndParsesBf16ParameterKey) {
  char key_storage[128];
  iree_string_view_t virtual_key = iree_string_view_empty();
  IREE_ASSERT_OK(id4_vae_parameter_format_bf16_key(
      IREE_SV("decoder.post_quant_conv.bias"), key_storage,
      IREE_ARRAYSIZE(key_storage), &virtual_key));

  iree_string_view_t source_key = iree_string_view_empty();
  EXPECT_TRUE(id4_vae_parameter_parse_bf16_key(virtual_key, &source_key));
  EXPECT_TRUE(iree_string_view_equal(source_key,
                                     IREE_SV("decoder.post_quant_conv.bias")));
}

}  // namespace
