// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/executable_measurement.h"

#include <array>
#include <vector>

#include "iree/hal/driver/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {
using iree::hal::amd::xdna::testing::ImageFixture;
using iree::hal::amd::xdna::testing::MakeImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

// Header plus three 24-byte commands, with a measured middle command. Dynamic
// word pairs on both sides exercise all three translated relocation regions.
ImageFixture MeasurementFixture() {
  ImageFixture fixture;
  fixture.allocations[0].byte_length = 32768 + 88;
  for (size_t i = 0; i < 2; ++i) {
    fixture.loads[i].file_range.length = 88;
    fixture.loads[i].memory_size = 88;
    fixture.payloads[i].resize(88, 0xA5);
    fixture.payloads[i][0] = 0;
    fixture.payloads[i][1] = 1;
    iree_unaligned_store_le_u32(fixture.payloads[i].data() + 8, 3);
    iree_unaligned_store_le_u32(fixture.payloads[i].data() + 12, 88);
    fixture.invocations[i].byte_length = 88;
    fixture.invocations[i].measurement = {40, 64};
  }
  fixture.loads[2].file_range.offset = 4096 + 88;
  fixture.entries[0].static_relocation_count = 0;
  fixture.entries[0].first_dynamic_relocation = 0;
  fixture.entries[0].dynamic_relocation_count = 3;
  const auto dynamic_relocation = fixture.relocations[1];
  fixture.relocations.assign(3, dynamic_relocation);
  for (uint32_t i = 0; i < 3; ++i) {
    fixture.relocations[i].byte_offset = 16 + i * 24;
  }
  return fixture;
}

TEST(ExecutableMeasurementTest,
     CopiesOnlyDeclaredBoundariesAndRebindsAllRegions) {
  auto fixture = MeasurementFixture();
  auto source = MakeOwnedByteSequence(fixture.Build());
  const auto target = MakeImageTarget();
  iree_hal_amd_xdna_image_t* image = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_allocate(
      source.get(), &target, iree_allocator_system(), &image));
  const auto* tables = iree_hal_amd_xdna_image_tables(image);
  const auto invocation = iree_hal_amd_xdna_image_tables_invocation(tables, 0);
  const auto entry = iree_hal_amd_xdna_image_tables_entry(tables, 0);
  auto command = fixture.payloads[0];
  const auto original = command;
  std::array<uint8_t, 112> measured;
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_measurement_initialize(
      &invocation, command.data(), measured.data()));
  EXPECT_EQ(command, original);
  EXPECT_EQ(iree_unaligned_load_le_u32(measured.data() + 8), 5u);
  EXPECT_EQ(iree_unaligned_load_le_u32(measured.data() + 12), 112u);
  EXPECT_EQ(std::vector<uint8_t>(measured.begin() + 16, measured.begin() + 40),
            std::vector<uint8_t>(original.begin() + 16, original.begin() + 40));
  EXPECT_EQ(std::vector<uint8_t>(measured.begin() + 52, measured.begin() + 76),
            std::vector<uint8_t>(original.begin() + 40, original.begin() + 64));
  EXPECT_EQ(std::vector<uint8_t>(measured.begin() + 88, measured.end()),
            std::vector<uint8_t>(original.begin() + 64, original.end()));
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t offset = 40 + i * 36;
    EXPECT_EQ(iree_unaligned_load_le_u32(measured.data() + offset), 0x83u);
    EXPECT_EQ(iree_unaligned_load_le_u32(measured.data() + offset + 4), 12u);
    EXPECT_EQ(iree_unaligned_load_le_u32(measured.data() + offset + 8), i + 1);
  }
  auto expected = measured;
  for (uint32_t i = 0; i < 3; ++i) {
    const uint64_t address = UINT64_C(0xA5A5123400000000) + i * 4096;
    iree_unaligned_store_le_u64(command.data() + 16 + i * 24, address);
    iree_unaligned_store_le_u64(expected.data() + 16 + i * 36, address);
  }
  const auto changed = iree_hal_amd_xdna_executable_measurement_patch(
      tables, &entry, &invocation, command.data(), measured.data());
  EXPECT_EQ(changed.data, measured.data() + 16);
  EXPECT_EQ(changed.data_length, 80u);
  EXPECT_EQ(measured, expected);
  iree_hal_amd_xdna_image_free(image);
}

TEST(ExecutableMeasurementTest, RejectsSplitRelocationsAndNativeHeaders) {
  for (const auto interval : {std::array<uint32_t, 2>{0, 64},
                              {40, 0},
                              {44, 64},
                              {40, 68},
                              {40, 92},
                              {12, 64}}) {
    auto fixture = MeasurementFixture();
    fixture.invocations[0].measurement = {interval[0], interval[1]};
    auto source = MakeOwnedByteSequence(fixture.Build());
    const auto target = MakeImageTarget();
    iree_hal_amd_xdna_image_t* image = nullptr;
    IREE_EXPECT_STATUS_IS(
        iree::StatusCode::kInvalidArgument,
        iree_hal_amd_xdna_image_allocate(source.get(), &target,
                                         iree_allocator_system(), &image));
    EXPECT_EQ(image, nullptr);
  }
  auto fixture = MeasurementFixture();
  for (uint32_t field : {0u, 8u, 12u}) {
    auto command = fixture.payloads[0];
    iree_unaligned_store_le_u32(command.data() + field, UINT32_MAX);
    std::array<uint8_t, 112> measured = {};
    const auto before = measured;
    IREE_EXPECT_STATUS_IS(
        iree::StatusCode::kInvalidArgument,
        iree_hal_amd_xdna_executable_measurement_initialize(
            &fixture.invocations[0], command.data(), measured.data()));
    EXPECT_EQ(measured, before);
  }
}
}  // namespace
