// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4/encoding/profile.h"

#include <array>

#include "gtest/gtest.h"

namespace {

TEST(Pm4ProfileTest, CoversThePm4PhysicalTargetFamilies) {
  constexpr std::array<std::array<uint32_t, 3>, 15> targets = {{
      {11, 0, 0},
      {11, 0, 1},
      {11, 0, 2},
      {11, 0, 3},
      {11, 5, 0},
      {11, 5, 1},
      {11, 5, 2},
      {11, 5, 3},
      {11, 7, 0},
      {11, 7, 1},
      {11, 7, 2},
      {12, 0, 0},
      {12, 0, 1},
      {12, 5, 0},
      {12, 5, 1},
  }};
  for (const auto& target : targets) {
    SCOPED_TRACE(::testing::Message()
                 << target[0] << '.' << target[1] << '.' << target[2]);
    amdf_gpu_endpoint_info_t endpoint = {
        .gfx_ip = {target[0], target[1], target[2]},
    };
    const auto* profile = Pm4CommandProfile::Find(endpoint);
    ASSERT_NE(profile, nullptr);
    EXPECT_EQ(profile->supports_wave64, target[0] != 12 || target[1] != 5);
    // The gfx1250 instruction overlay does not change these PM4 registers.
    endpoint.asic_revision = 1;
    EXPECT_EQ(Pm4CommandProfile::Find(endpoint), profile);
  }
}

TEST(Pm4ProfileTest, DoesNotInferAnEncodingForUnrepresentedFamilies) {
  constexpr std::array<std::array<uint32_t, 3>, 5> targets = {{
      {0, 0, 0},
      {9, 4, 2},
      {9, 5, 0},
      {11, 1, 0},
      {12, 2, 0},
  }};
  for (const auto& target : targets) {
    amdf_gpu_endpoint_info_t endpoint = {
        .gfx_ip = {target[0], target[1], target[2]},
    };
    EXPECT_EQ(Pm4CommandProfile::Find(endpoint), nullptr);
  }
}

TEST(Pm4ProfileTest, BacksImageTailAndTheCompletePrefetchField) {
  for (const auto& version :
       std::array<std::array<uint32_t, 2>, 3>{{{11, 5}, {12, 0}, {12, 5}}}) {
    amdf_gpu_endpoint_info_t endpoint = {.gfx_ip = {version[0], version[1], 0}};
    const auto* profile = Pm4CommandProfile::Find(endpoint);
    ASSERT_NE(profile, nullptr);
    // Preserve the complete linked image, then pad three 64-byte fetch lines.
    EXPECT_EQ(profile->CodeByteLength(1025, 256, 0), 1280u);
    // Prefix bytes matter: prefetch starts at the aligned entry, not image 0.
    EXPECT_EQ(profile->CodeByteLength(1024, 768, 0x3f0), 8832u);
    // Bits 10/11 extend the GFX12+ prefetch field; other RSRC3 bits do not.
    const uint64_t maximum = version[0] == 11 ? 8320u : 32896u;
    EXPECT_EQ(profile->CodeByteLength(1024, 256, 0xfffffff0), maximum);
  }
}

}  // namespace
