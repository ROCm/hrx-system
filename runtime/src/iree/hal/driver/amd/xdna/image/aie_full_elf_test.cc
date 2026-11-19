// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "iree/hal/driver/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/driver/amd/xdna/image/image.h"
#include "iree/hal/driver/amd/xdna/image/testdata/aie/full_elf_42.h"
#include "iree/hal/driver/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

// Test edits use the producer's named sections. Only called on the original
// compiler output, before the individual test changes the field under study.
uint8_t* Section(std::vector<uint8_t>& bytes, const char* name) {
  uint8_t* headers =
      bytes.data() + iree_unaligned_load_le_u32(bytes.data() + 32);
  const uint16_t count = iree_unaligned_load_le_u16(bytes.data() + 48);
  const uint16_t names = iree_unaligned_load_le_u16(bytes.data() + 50);
  const char* strings = reinterpret_cast<const char*>(
      bytes.data() + iree_unaligned_load_le_u32(headers + 40 * names + 16));
  for (uint16_t i = 0; i < count; ++i) {
    uint8_t* row = headers + 40 * i;
    if (std::strcmp(strings + iree_unaligned_load_le_u32(row), name) == 0) {
      return bytes.data() + iree_unaligned_load_le_u32(row + 16);
    }
  }
  return nullptr;
}

class AieFullElfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // This producer targets the AIE2P architecture and carries no Loom
    // execution-profile ID. Exercise the loader on an unlisted endpoint.
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        IREE_SV("amd.xdna.unlisted"), 1, &target_));
    target_.supports_load_pdi = true;
    // The first file uses a single PDI. The second can reinitialize a
    // placement.
    const auto& file = iree_hal_amd_xdna_test_full_elf_42_create()[1];
    bytes_.assign(file.data, file.data + file.size);
  }
  void TearDown() override { iree_hal_amd_xdna_image_free(image_); }

  Status Admit() {
    auto source = MakeOwnedByteSequence(bytes_);
    return Status(iree_hal_amd_xdna_image_allocate(
        source.get(), &target_, iree_allocator_system(), &image_));
  }

  // Source data for the expanded compiler fixture or one test variation.
  std::vector<uint8_t> bytes_;
  // Effective queue instruction capabilities and placement geometry.
  iree_hal_amd_xdna_aie2p_target_t target_ = {};
  // Admitted image, released even after a fatal test assertion.
  iree_hal_amd_xdna_image_t* image_ = nullptr;
};

TEST_F(AieFullElfTest, RetainsSegmentedSourceAndDistinctResetImages) {
  // Split in the middle of the first PDI and in the section directory.
  const size_t table = iree_unaligned_load_le_u32(bytes_.data() + 32);
  auto first = MakeOwnedByteSequence({bytes_.begin(), bytes_.begin() + 301});
  auto middle =
      MakeOwnedByteSequence({bytes_.begin() + 301, bytes_.begin() + table + 3});
  auto last = MakeOwnedByteSequence({bytes_.begin() + table + 3, bytes_.end()});
  iree_byte_sequence_t* parts[] = {first.get(), middle.get(), last.get()};
  iree_byte_sequence_t* source = nullptr;
  IREE_ASSERT_OK(iree_byte_sequence_create_concat(
      IREE_ARRAYSIZE(parts), parts, iree_allocator_system(), &source));
  first.reset();
  middle.reset();
  last.reset();
  Status status(iree_hal_amd_xdna_image_allocate(
      source, &target_, iree_allocator_system(), &image_));
  iree_byte_sequence_release(source);
  IREE_ASSERT_OK(status);
  uint32_t sequence = UINT32_MAX, empty = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
      image_, IREE_SV("main:sequence"), &sequence));
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
      image_, IREE_SV("main:aiecc_empty"), &empty));
  EXPECT_NE(sequence, empty);
  const auto* tables = iree_hal_amd_xdna_image_tables(image_);
  const auto& entry = tables->entries[sequence];
  ASSERT_EQ(entry.binding_count, 1u);
  ASSERT_EQ(entry.static_relocation_count, 2u);
  ASSERT_EQ(entry.dynamic_relocation_count, 1u);
  ASSERT_EQ(entry.allocation_use_count, 3u);
  std::vector<uint8_t> payloads[2];
  uint32_t ordinals[2];
  for (uint32_t i = 0; i < 2; ++i) {
    const auto& patch = tables->relocations[entry.first_static_relocation + i];
    EXPECT_EQ(patch.kind, IREE_HAL_AMD_XDNA_RELOCATION_FIRMWARE_ADDRESS_64);
    ordinals[i] = tables->allocation_uses[entry.first_allocation_use +
                                          patch.source_ordinal];
    const auto& allocation = tables->allocations[ordinals[i]];
    ASSERT_NE(allocation.byte_length, 0u);
    payloads[i].resize(allocation.byte_length);
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_load_allocation(image_, ordinals[i],
                                                           payloads[i].data()));
    const uint8_t* expected = Section(bytes_, i == 0 ? ".pdi.2" : ".pdi.1");
    ASSERT_NE(expected, nullptr);
    EXPECT_EQ(std::memcmp(expected, payloads[i].data(), payloads[i].size()), 0);
  }
  EXPECT_NE(ordinals[0], ordinals[1]);
  EXPECT_EQ(payloads[0], payloads[1]);
}

TEST_F(AieFullElfTest, RetainsArgumentHolesAndBothBufferDisplacements) {
  uint8_t* symbols = Section(bytes_, ".dynsym");
  uint8_t* names = Section(bytes_, ".dynstr");
  uint8_t* relocations = Section(bytes_, ".rela.dyn");
  uint8_t* command = Section(bytes_, ".ctrltext.1");
  ASSERT_NE(symbols, nullptr);
  ASSERT_NE(names, nullptr);
  ASSERT_NE(relocations, nullptr);
  ASSERT_NE(command, nullptr);
  // The second record in this producer fixture addresses the output buffer.
  uint8_t* relocation = relocations + 12;
  const uint32_t info = iree_unaligned_load_le_u32(relocation + 4);
  ASSERT_EQ(info & 0xffu, 5u);
  uint8_t* symbol = symbols + (info >> 8) * 16;
  names[iree_unaligned_load_le_u32(symbol)] = '2';
  iree_unaligned_store_le_u32(relocation + 8, 12);
  uint8_t* field = command + iree_unaligned_load_le_u32(relocation) + 4;
  iree_unaligned_store_le_u32(field, 64);
  IREE_ASSERT_OK(Admit());
  uint32_t ordinal = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
      image_, IREE_SV("main:sequence"), &ordinal));
  const auto* tables = iree_hal_amd_xdna_image_tables(image_);
  const auto& entry = tables->entries[ordinal];
  ASSERT_EQ(entry.binding_count, 3u);
  EXPECT_EQ(tables->bindings[entry.first_binding].kind,
            IREE_XDNA_ELF_BINDING_KIND_NONE);
  EXPECT_EQ(tables->bindings[entry.first_binding + 1].kind,
            IREE_XDNA_ELF_BINDING_KIND_NONE);
  EXPECT_EQ(tables->bindings[entry.first_binding + 2].minimum_byte_length, 80u);
  const auto& patch = tables->relocations[entry.first_dynamic_relocation];
  EXPECT_EQ(patch.source_ordinal, 2u);
  EXPECT_EQ(patch.addend, 76);
}

TEST_F(AieFullElfTest, RequiresAnAdmittedProviderAndMatchingPartition) {
  target_.supports_load_pdi = false;
  EXPECT_EQ(Admit().code(), StatusCode::kIncompatible);
  EXPECT_EQ(image_, nullptr);
  target_.supports_load_pdi = true;
  target_.context.column_count = 2;
  EXPECT_EQ(Admit().code(), StatusCode::kFailedPrecondition);
  EXPECT_EQ(image_, nullptr);
}

TEST_F(AieFullElfTest, RejectsConflictingOrUnimplementedFixups) {
  uint8_t* rela = Section(bytes_, ".rela.dyn");
  ASSERT_NE(rela, nullptr);
  // Point the external shim address at the existing firmware address field.
  uint32_t saved_offset = iree_unaligned_load_le_u32(rela + 12);
  iree_unaligned_store_le_u32(rela + 12, iree_unaligned_load_le_u32(rela) - 4);
  EXPECT_EQ(Admit().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(image_, nullptr);
  iree_unaligned_store_le_u32(rela + 12, saved_offset);
  rela[12 + 4] = 99;
  EXPECT_EQ(Admit().code(), StatusCode::kUnimplemented);
  EXPECT_EQ(image_, nullptr);
}

TEST_F(AieFullElfTest, RejectsTruncatedDirectoryAndMissingGroupMember) {
  uint8_t* group = Section(bytes_, ".group.1");
  ASSERT_NE(group, nullptr);
  iree_unaligned_store_le_u32(group + 4, 0);
  EXPECT_EQ(Admit().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(image_, nullptr);
  bytes_.pop_back();
  EXPECT_EQ(Admit().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(image_, nullptr);
}
}  // namespace
