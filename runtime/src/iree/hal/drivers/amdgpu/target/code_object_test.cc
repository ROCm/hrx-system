// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/target/code_object.h"

#include <array>
#include <string>

#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static constexpr uint8_t kElfClass64 = 2;
static constexpr uint8_t kElfData2Lsb = 1;
static constexpr uint8_t kElfVersionCurrent = 1;
static constexpr uint8_t kElfOsAbiAmdgpuHsa = 64;
static constexpr uint8_t kElfAbiVersionV3 = 1;
static constexpr uint8_t kElfAbiVersionV5 = 3;
static constexpr uint8_t kElfAbiVersionV6 = 4;
static constexpr uint16_t kElfMachineAmdgpu = 224;
static constexpr uint32_t kElfMachineGfx906 = 0x02f;
static constexpr uint32_t kElfMachineGfx1100 = 0x041;
static constexpr uint32_t kElfMachineGfx942 = 0x04c;
static constexpr uint32_t kElfMachineGfx11Generic = 0x054;
static constexpr uint32_t kElfFeatureXnackUnsupportedV4 = 0x000;
static constexpr uint32_t kElfFeatureXnackOffV4 = 0x200;
static constexpr uint32_t kElfFeatureSrameccAnyV4 = 0x400;
static constexpr uint32_t kElfFeatureSrameccOnV4 = 0xc00;
static constexpr uint32_t kElfGenericVersionOffset = 24;

static std::array<uint8_t, 64> MakeElf64AmdgpuHsa(uint8_t abi_version,
                                                  uint16_t machine,
                                                  uint32_t e_flags) {
  std::array<uint8_t, 64> elf = {};
  elf[0] = 0x7f;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = kElfClass64;
  elf[5] = kElfData2Lsb;
  elf[6] = kElfVersionCurrent;
  elf[7] = kElfOsAbiAmdgpuHsa;
  elf[8] = abi_version;
  iree_unaligned_store_le_u16(&elf[18], machine);
  iree_unaligned_store_le_u32(&elf[20], kElfVersionCurrent);
  iree_unaligned_store_le_u32(&elf[48], e_flags);
  iree_unaligned_store_le_u16(&elf[52], (uint16_t)elf.size());
  return elf;
}

static iree_hal_amdgpu_target_identity_t ParseCodeObjectIdentity(
    const std::array<uint8_t, 64>& elf) {
  iree_hal_amdgpu_target_identity_t identity;
  IREE_CHECK_OK(iree_hal_amdgpu_code_object_identity_from_elf(
      iree_make_const_byte_span(elf.data(), elf.size()), &identity));
  return identity;
}

static std::string FormatIdentity(
    const iree_hal_amdgpu_target_identity_t* identity) {
  char buffer[64] = {0};
  IREE_CHECK_OK(iree_hal_amdgpu_target_identity_format_artifact_key(
      identity, sizeof(buffer), buffer, /*out_buffer_length=*/nullptr));
  return std::string(buffer);
}

TEST(CodeObjectTest, ParsesV5FeatureStates) {
  const auto elf = MakeElf64AmdgpuHsa(
      kElfAbiVersionV5, kElfMachineAmdgpu,
      kElfMachineGfx942 | kElfFeatureSrameccOnV4 | kElfFeatureXnackOffV4);
  auto identity = ParseCodeObjectIdentity(elf);
  EXPECT_EQ(identity.kind, IREE_HAL_AMDGPU_TARGET_KIND_EXACT);
  EXPECT_EQ(identity.generic_version, 0u);
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);
  EXPECT_EQ(FormatIdentity(&identity), "gfx942:sramecc+:xnack-");
}

TEST(CodeObjectTest, ParsesV5AnyAndUnsupportedFeatures) {
  const auto elf =
      MakeElf64AmdgpuHsa(kElfAbiVersionV5, kElfMachineAmdgpu,
                         kElfMachineGfx1100 | kElfFeatureSrameccAnyV4 |
                             kElfFeatureXnackUnsupportedV4);
  auto identity = ParseCodeObjectIdentity(elf);
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED);
  EXPECT_EQ(FormatIdentity(&identity), "gfx1100");
}

TEST(CodeObjectTest, ParsesV6GenericVersion) {
  const auto elf = MakeElf64AmdgpuHsa(
      kElfAbiVersionV6, kElfMachineAmdgpu,
      kElfMachineGfx11Generic | (1u << kElfGenericVersionOffset));
  auto identity = ParseCodeObjectIdentity(elf);
  EXPECT_EQ(identity.kind, IREE_HAL_AMDGPU_TARGET_KIND_GENERIC);
  EXPECT_EQ(identity.generic_version, 1u);
  EXPECT_EQ(FormatIdentity(&identity), "gfx11-generic");
}

TEST(CodeObjectTest, ParsesV3SupportedAbsentFeaturesAsOff) {
  const auto elf = MakeElf64AmdgpuHsa(kElfAbiVersionV3, kElfMachineAmdgpu,
                                      kElfMachineGfx906);
  auto identity = ParseCodeObjectIdentity(elf);
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF);
  EXPECT_EQ(FormatIdentity(&identity), "gfx906:sramecc-:xnack-");
}

TEST(CodeObjectTest, RejectsV6GenericWithoutVersion) {
  const auto elf = MakeElf64AmdgpuHsa(kElfAbiVersionV6, kElfMachineAmdgpu,
                                      kElfMachineGfx11Generic);
  iree_hal_amdgpu_target_identity_t identity;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_code_object_identity_from_elf(
          iree_make_const_byte_span(elf.data(), elf.size()), &identity));
}

TEST(CodeObjectTest, RejectsUnsupportedMachineValue) {
  const auto elf =
      MakeElf64AmdgpuHsa(kElfAbiVersionV5, kElfMachineAmdgpu, 0x027);
  iree_hal_amdgpu_target_identity_t identity;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_code_object_identity_from_elf(
          iree_make_const_byte_span(elf.data(), elf.size()), &identity));
}

TEST(CodeObjectTest, RejectsNonAmdgpuElfMachine) {
  const auto elf =
      MakeElf64AmdgpuHsa(kElfAbiVersionV5, /*machine=*/3, kElfMachineGfx1100);
  iree_hal_amdgpu_target_identity_t identity;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amdgpu_code_object_identity_from_elf(
          iree_make_const_byte_span(elf.data(), elf.size()), &identity));
}

}  // namespace
}  // namespace iree::hal::amdgpu
