// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/aie2p/npu2.h"

#include <array>
#include <cstdint>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::StatusCode;

struct DeviceProfile {
  // Canonical endpoint key under test.
  const char* target_id;
  // Image execution identity expected for that endpoint.
  uint64_t identity;
};

class Aie2pNpu2Test : public ::testing::TestWithParam<DeviceProfile> {
 protected:
  iree_hal_amd_xdna_aie2p_target_t MakeTarget(
      uint16_t context_column_count = 3) {
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(GetParam().target_id), context_column_count,
        &target));
    return target;
  }
};

INSTANTIATE_TEST_SUITE_P(
    Devices, Aie2pNpu2Test,
    ::testing::Values(
        DeviceProfile{"amd.xdna.strix.17f0_10", UINT64_C(0x5354524958000001)},
        DeviceProfile{"amd.xdna.krackan.17f0_20", UINT64_C(0x5354524958000001)},
        DeviceProfile{"amd.xdna.strix_halo.17f0_11",
                      UINT64_C(0x535848414C4F0001)}));

TEST_P(Aie2pNpu2Test, InitializesExactTargetFacts) {
  const iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  EXPECT_EQ(target.identity.device_profile_revision, 1u);
  EXPECT_EQ(target.identity.device_profile_id, GetParam().identity);
  EXPECT_EQ(target.identity.firmware_abi_id, UINT64_C(0x4E5055320006000C));
  EXPECT_EQ(target.context.column_count, 3u);
  EXPECT_EQ(target.context.row_count, 6u);
  EXPECT_EQ(target.instruction_alignment, 32768u);
}

TEST_P(Aie2pNpu2Test, RejectsInvalidContextWidths) {
  iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_amd_xdna_aie2p_npu2_target_initialize(
          iree_make_cstring_view(GetParam().target_id), 0, &target));
  EXPECT_EQ(target.context.column_count, 3u);
  IREE_EXPECT_STATUS_IS(
      StatusCode::kOutOfRange,
      iree_hal_amd_xdna_aie2p_npu2_target_initialize(
          iree_make_cstring_view(GetParam().target_id), 9, &target));
  EXPECT_EQ(target.context.column_count, 3u);
}

TEST_P(Aie2pNpu2Test, RejectsUnknownDeviceWithoutPublishing) {
  iree_hal_amd_xdna_aie2p_target_t target = MakeTarget();
  IREE_EXPECT_STATUS_IS(StatusCode::kUnimplemented,
                        iree_hal_amd_xdna_aie2p_npu2_target_initialize(
                            IREE_SV("amd.xdna.strix.17f0_ff"), 1, &target));
  EXPECT_EQ(target.identity.device_profile_id, GetParam().identity);
  EXPECT_EQ(target.context.column_count, 3u);
}

}  // namespace
