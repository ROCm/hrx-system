// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/av1_context.h"

#include <algorithm>
#include <array>

#include "gtest/gtest.h"

namespace {

TEST(Av1ContextTest, InstallsAllFourNormativeCdfSets) {
  alignas(uint16_t) std::array<uint8_t, 81920 + 64> context = {};
  std::fill(context.begin() + 81920, context.end(), 0xa5);
  amdf_media_drm_av1_context_initialize(context.data());
  // Independent fingerprints of the normative distributions in each VCN4
  // quantizer context, including its reserved zero words.
  constexpr uint64_t expected[] = {
      UINT64_C(0xdbe2602fda8d98d8), UINT64_C(0xe85564747b561831),
      UINT64_C(0x7c065218fc1ea424), UINT64_C(0xcfcc529111711ed4)};
  for (uint32_t group = 0; group < 4; ++group) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t i = 0; i < 20480; ++i) {
      hash = (hash ^ context[group * 20480 + i]) * UINT64_C(1099511628211);
    }
    EXPECT_EQ(hash, expected[group]) << "quantizer context=" << group;
  }
  EXPECT_TRUE(std::all_of(context.begin() + 81920, context.end(),
                          [](uint8_t byte) { return byte == 0xa5; }));
}

TEST(Av1ContextTest, InstallsAllFourEncoderCdfSets) {
  alignas(uint16_t)
      std::array<uint8_t, AMDF_MEDIA_DRM_AV1_ENCODE_CDF_BYTES + 64>
          context = {};
  std::fill(context.begin() + AMDF_MEDIA_DRM_AV1_ENCODE_CDF_BYTES,
            context.end(), 0xa5);
  amdf_media_drm_av1_encode_cdf_initialize(context.data());
  // Independent fingerprints of Mesa's VCN4 encoder table, including every
  // normative distribution and the firmware's reserved-word sentinels.
  constexpr uint64_t expected[] = {
      UINT64_C(0xd2154c21d3a8909e), UINT64_C(0xbfc4baae73bea167),
      UINT64_C(0x670cb61626cc0c66), UINT64_C(0xa5edeb7fad07f3d2)};
  for (uint32_t group = 0; group < 4; ++group) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t i = 0; i < 20480; ++i) {
      hash = (hash ^ context[group * 20480 + i]) * UINT64_C(1099511628211);
    }
    EXPECT_EQ(hash, expected[group]) << "quantizer context=" << group;
  }
  EXPECT_TRUE(std::all_of(context.begin() + AMDF_MEDIA_DRM_AV1_ENCODE_CDF_BYTES,
                          context.end(),
                          [](uint8_t byte) { return byte == 0xa5; }));
}

}  // namespace
