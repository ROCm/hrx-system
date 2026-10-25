// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/av1_grain.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "gtest/gtest.h"

namespace {

uint64_t Hash(const uint8_t* data, size_t size) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ data[i]) * UINT64_C(1099511628211);
  }
  return hash;
}

amdf_media_av1_film_grain_t Grain() {
  amdf_media_av1_film_grain_t grain = {
      .apply = 1,
      .seed = 0x4ac3,
      .luma_point_count = 4,
      .luma_points = {{0, 32}, {64, 96}, {192, 48}, {255, 128}},
      .u_point_count = 3,
      .u_points = {{0, 24}, {128, 112}, {255, 40}},
      .v_point_count = 3,
      .v_points = {{0, 80}, {96, 32}, {255, 104}},
      .scaling_shift_minus8 = 1,
      .coefficient_lag = 2,
      .coefficient_shift_minus6 = 2,
      .scale_shift = 1,
      .chroma_multipliers = {129, 117},
      .chroma_luma_multipliers = {64, 72},
      .chroma_offsets = {257, 193},
      .overlap = 1,
      .clip_to_restricted_range = 1,
  };
  for (size_t i = 0; i < 12; ++i) {
    grain.luma_coefficients[i] = static_cast<int8_t>((i % 5) - 2);
  }
  for (size_t i = 0; i < 13; ++i) {
    grain.u_coefficients[i] = static_cast<int8_t>((i % 7) - 3);
    grain.v_coefficients[i] = static_cast<int8_t>(3 - (i % 7));
  }
  return grain;
}

TEST(Av1GrainTest, SelectsNativeBlockPackingByFirmwareProtocol) {
  constexpr size_t kGuardBytes = 64;
  const amdf_media_av1_film_grain_t grain = Grain();
  std::array<uint8_t, AMDF_MEDIA_DRM_AV1_GRAIN_PARAMETER_BYTES> parameters1 =
      {};
  std::array<uint8_t, AMDF_MEDIA_DRM_AV1_GRAIN_PARAMETER_BYTES> parameters2 =
      {};
  std::array<uint8_t, AMDF_MEDIA_DRM_AV1_GRAIN_AUXILIARY_BYTES + kGuardBytes>
      auxiliary1 = {};
  std::array<uint8_t, AMDF_MEDIA_DRM_AV1_GRAIN_AUXILIARY_BYTES + kGuardBytes>
      auxiliary2 = {};
  std::fill(auxiliary1.end() - kGuardBytes, auxiliary1.end(), 0xa5);
  std::fill(auxiliary2.end() - kGuardBytes, auxiliary2.end(), 0xa5);

  amdf_media_drm_av1_grain(AMDF_MEDIA_DRM_AV1_PROTOCOL_V1, &grain, 10,
                           parameters1.data(), auxiliary1.data());
  amdf_media_drm_av1_grain(AMDF_MEDIA_DRM_AV1_PROTOCOL_V2, &grain, 10,
                           parameters2.data(), auxiliary2.data());

  EXPECT_EQ(parameters1, parameters2);
  // Complete fingerprints pin Mesa's native parameter record and the two
  // distinct grain-table arrangements without embedding 40 KiB of goldens.
  EXPECT_EQ(Hash(parameters1.data(), parameters1.size()),
            UINT64_C(0x176c803497e28460));
  EXPECT_EQ(Hash(auxiliary1.data(), AMDF_MEDIA_DRM_AV1_GRAIN_AUXILIARY_BYTES),
            UINT64_C(0xbd3f70f656d57047));
  EXPECT_EQ(Hash(auxiliary2.data(), AMDF_MEDIA_DRM_AV1_GRAIN_AUXILIARY_BYTES),
            UINT64_C(0x9ded9d737f79e859));

  // Version 2 packs only the dense 64x64 and 32x32 generated blocks. The LUT
  // and seed region beginning after the three fixed-capacity blocks is shared.
  constexpr size_t kLutOffset = 18432;
  EXPECT_TRUE(
      std::equal(auxiliary1.begin() + kLutOffset,
                 auxiliary1.begin() + AMDF_MEDIA_DRM_AV1_GRAIN_AUXILIARY_BYTES,
                 auxiliary2.begin() + kLutOffset));
  EXPECT_TRUE(std::all_of(auxiliary2.begin() + 8192, auxiliary2.begin() + 12288,
                          [](uint8_t byte) { return byte == 0; }));
  EXPECT_TRUE(std::all_of(auxiliary2.begin() + 14336,
                          auxiliary2.begin() + 15360,
                          [](uint8_t byte) { return byte == 0; }));
  EXPECT_TRUE(std::all_of(auxiliary2.begin() + 17408,
                          auxiliary2.begin() + kLutOffset,
                          [](uint8_t byte) { return byte == 0; }));
  EXPECT_TRUE(std::all_of(auxiliary1.end() - kGuardBytes, auxiliary1.end(),
                          [](uint8_t byte) { return byte == 0xa5; }));
  EXPECT_TRUE(std::all_of(auxiliary2.end() - kGuardBytes, auxiliary2.end(),
                          [](uint8_t byte) { return byte == 0xa5; }));
}

}  // namespace
