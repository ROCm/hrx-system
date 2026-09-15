// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/amdgpu_architecture.h"

#include "iree/testing/gtest.h"

namespace {

TEST(AmdgpuArchitectureTest, DecodesNumericAndAlphabeticSteppings) {
  iree_hal_streaming_amdgpu_architecture_t architecture = {};
  EXPECT_TRUE(
      iree_hal_streaming_parse_amdgpu_architecture("gfx942", &architecture));
  EXPECT_EQ(9u, architecture.major);
  EXPECT_EQ(4u, architecture.minor);
  EXPECT_EQ(2u, architecture.stepping);

  EXPECT_TRUE(iree_hal_streaming_parse_amdgpu_architecture("gfx90a:xnack+",
                                                           &architecture));
  EXPECT_EQ(9u, architecture.major);
  EXPECT_EQ(0u, architecture.minor);
  EXPECT_EQ(10u, architecture.stepping);

  EXPECT_TRUE(iree_hal_streaming_parse_amdgpu_architecture(
      "gfx1100:sramecc-:xnack+", &architecture));
  EXPECT_EQ(11u, architecture.major);
  EXPECT_EQ(0u, architecture.minor);
  EXPECT_EQ(0u, architecture.stepping);
}

TEST(AmdgpuArchitectureTest, RejectsMalformedTargetsWithoutChangingOutput) {
  const char* values[] = {
      "",         "gfx",           "gfx90",          "gfx99000",
      "gfx9a0",   "gfx90z",        "gfx942:",        "gfx942:xnack",
      "gfx942:+", "gfx942:other+", "gfx942:xnack+x", "gfx942::xnack+",
  };
  for (const char* value : values) {
    SCOPED_TRACE(value);
    iree_hal_streaming_amdgpu_architecture_t architecture = {23, 29, 31};
    EXPECT_FALSE(
        iree_hal_streaming_parse_amdgpu_architecture(value, &architecture));
    EXPECT_EQ(23u, architecture.major);
    EXPECT_EQ(29u, architecture.minor);
    EXPECT_EQ(31u, architecture.stepping);
  }
  EXPECT_FALSE(iree_hal_streaming_parse_amdgpu_architecture(nullptr, nullptr));
}

}  // namespace
