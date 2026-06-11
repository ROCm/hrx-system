// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/executable.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

TEST(ExecutableTest, InfersBdaSpirvFormat) {
  static const uint32_t kSpirvHeader[] = {
      0x07230203u, 0x00010000u, 0u, 0u, 0u,
  };
  char format[64] = {0};
  iree_host_size_t inferred_size = 0;

  IREE_ASSERT_OK(iree_hal_vulkan_executable_infer_format(
      iree_make_const_byte_span(kSpirvHeader, sizeof(kSpirvHeader)),
      IREE_ARRAYSIZE(format), format, &inferred_size));

  EXPECT_STREQ("vulkan-spirv-bda", format);
  EXPECT_EQ(sizeof(kSpirvHeader), inferred_size);
}

TEST(ExecutableTest, RejectsNonSpirvFormatInference) {
  static const uint32_t kNotSpirv[] = {
      0u, 0x00010000u, 0u, 0u, 0u,
  };
  char format[64] = {0};
  iree_host_size_t inferred_size = 0;

  IREE_EXPECT_STATUS_IS(
      StatusCode::kInvalidArgument,
      iree_hal_vulkan_executable_infer_format(
          iree_make_const_byte_span(kNotSpirv, sizeof(kNotSpirv)),
          IREE_ARRAYSIZE(format), format, &inferred_size));
}

TEST(ExecutableTest, SupportsOnlyBdaSpirvWithBdaFeature) {
  const iree_hal_vulkan_features_t bda_features =
      IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES;

  EXPECT_TRUE(iree_hal_vulkan_executable_format_supported(
      bda_features, IREE_SV("vulkan-spirv-bda")));
  EXPECT_FALSE(iree_hal_vulkan_executable_format_supported(
      IREE_HAL_VULKAN_FEATURE_NONE, IREE_SV("vulkan-spirv-bda")));
  EXPECT_FALSE(iree_hal_vulkan_executable_format_supported(
      bda_features, IREE_SV("vulkan-obsolete")));
  EXPECT_FALSE(iree_hal_vulkan_executable_format_supported(
      bda_features, IREE_SV("vulkan-unknown")));
}

}  // namespace
}  // namespace iree::hal::vulkan
