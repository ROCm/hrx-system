// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable.h"

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
static constexpr uint8_t kElfAbiVersionV5 = 3;
static constexpr uint16_t kElfMachineAmdgpu = 224;
static constexpr uint32_t kElfMachineGfx942 = 0x04c;
static constexpr uint32_t kElfFeatureXnackOffV4 = 0x200;
static constexpr uint32_t kElfFeatureSrameccOnV4 = 0xc00;

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
  iree_unaligned_store_le_u16((uint16_t*)&elf[18], machine);
  iree_unaligned_store_le_u32((uint32_t*)&elf[20], kElfVersionCurrent);
  iree_unaligned_store_le_u32((uint32_t*)&elf[48], e_flags);
  iree_unaligned_store_le_u16((uint16_t*)&elf[52], (uint16_t)elf.size());
  return elf;
}

static std::string InferExecutableFormat(iree_const_byte_span_t executable_data,
                                         iree_host_size_t* out_inferred_size) {
  char executable_format[64] = {};
  IREE_CHECK_OK(iree_hal_amdgpu_executable_infer_format(
      executable_data, sizeof(executable_format), executable_format,
      iree_allocator_system(), out_inferred_size));
  return std::string(executable_format);
}

TEST(ExecutableTest, InfersRawHsacoTargetIdFromElfFlags) {
  const auto elf = MakeElf64AmdgpuHsa(
      kElfAbiVersionV5, kElfMachineAmdgpu,
      kElfMachineGfx942 | kElfFeatureSrameccOnV4 | kElfFeatureXnackOffV4);

  iree_host_size_t inferred_size = 0;
  EXPECT_EQ(
      InferExecutableFormat(iree_make_const_byte_span(elf.data(), elf.size()),
                            &inferred_size),
      "gfx942:sramecc+:xnack-");
  EXPECT_EQ(inferred_size, elf.size());
}

}  // namespace
}  // namespace iree::hal::amdgpu
