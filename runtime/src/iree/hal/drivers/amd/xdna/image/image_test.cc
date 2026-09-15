// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/image.h"

#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::Aie2pImageTargetState;
using iree::hal::amd::xdna::testing::BuildAie2pImage;
using iree::hal::amd::xdna::testing::ByteSequencePtr;
using iree::hal::amd::xdna::testing::kAie2pImageArrayProgramHeaderOrdinal;
using iree::hal::amd::xdna::testing::kAie2pImageControlProgramHeaderOrdinal;
using iree::hal::amd::xdna::testing::kAie2pImageStructuralCapabilities;
using iree::hal::amd::xdna::testing::kAie2pImageTileProgramHeaderOrdinal;
using iree::hal::amd::xdna::testing::MakeAie2pImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;
using iree::testing::status::StatusIs;

struct ImageDeleter {
  void operator()(iree_hal_amd_xdna_image_t* image) const {
    iree_hal_amd_xdna_image_destroy(image);
  }
};

using ImagePtr = std::unique_ptr<iree_hal_amd_xdna_image_t, ImageDeleter>;

static ImagePtr CreateImage(ByteSequencePtr sequence,
                            Aie2pImageTargetState* target_state) {
  const iree_hal_amd_xdna_aie2p_target_t aie2p_target =
      MakeAie2pImageTarget(target_state);
  iree_hal_amd_xdna_image_target_t target;
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      &aie2p_target, &target));
  iree_hal_amd_xdna_image_t* image = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_create(
      sequence.get(), &target, iree_allocator_system(), &image));
  return ImagePtr(image);
}

struct SourceEnumerator {
  // Bytes appended in source order.
  std::vector<uint8_t> bytes;
};

static iree_status_t AppendSourceSegment(void* user_data,
                                         iree_const_byte_span_t segment) {
  auto* enumerator = static_cast<SourceEnumerator*>(user_data);
  enumerator->bytes.insert(enumerator->bytes.end(), segment.data,
                           segment.data + segment.data_length);
  return iree_ok_status();
}

TEST(ImageTest, OwnsCompleteQualifiedImage) {
  const std::vector<uint8_t> bytes = BuildAie2pImage();
  Aie2pImageTargetState target_state;
  ImagePtr image = CreateImage(MakeOwnedByteSequence(bytes), &target_state);

  EXPECT_EQ(iree_hal_amd_xdna_image_source_length(image.get()), bytes.size());
  EXPECT_EQ(iree_hal_amd_xdna_image_target_flags(image.get()),
            IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS);
  EXPECT_EQ(iree_hal_amd_xdna_image_program_header_count(image.get()), 7u);
  EXPECT_EQ(iree_hal_amd_xdna_image_program_header(image.get(), 7), nullptr);

  const iree_hal_amd_xdna_elf_abi_note_t* abi_note =
      iree_hal_amd_xdna_image_abi_note(image.get());
  ASSERT_NE(abi_note, nullptr);
  EXPECT_EQ(abi_note->target_generation,
            IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P);
  EXPECT_EQ(abi_note->context_column_count, 3u);
  EXPECT_EQ(abi_note->context_row_count, 6u);

  ASSERT_EQ(iree_hal_amd_xdna_image_entry_count(image.get()), 1u);
  const iree_hal_amd_xdna_elf_entry_record_t* entry =
      iree_hal_amd_xdna_image_entry(image.get(), 0);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->array_program_header_ordinal,
            kAie2pImageArrayProgramHeaderOrdinal);
  EXPECT_EQ(entry->control_program_header_ordinal,
            kAie2pImageControlProgramHeaderOrdinal);
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amd_xdna_image_entry_name(image.get(), 0), IREE_SV("main")));
  EXPECT_EQ(iree_hal_amd_xdna_image_entry(image.get(), 1), nullptr);

  ASSERT_EQ(iree_hal_amd_xdna_image_binding_count(image.get()), 1u);
  EXPECT_NE(iree_hal_amd_xdna_image_binding(image.get(), 0), nullptr);
  EXPECT_EQ(iree_hal_amd_xdna_image_binding(image.get(), 1), nullptr);
  ASSERT_EQ(iree_hal_amd_xdna_image_relocation_count(image.get()), 1u);
  EXPECT_NE(iree_hal_amd_xdna_image_relocation(image.get(), 0), nullptr);
  EXPECT_EQ(iree_hal_amd_xdna_image_relocation(image.get(), 1), nullptr);

  ASSERT_EQ(iree_hal_amd_xdna_image_array_count(image.get()), 1u);
  EXPECT_NE(iree_hal_amd_xdna_image_array(image.get(), 0), nullptr);
  EXPECT_EQ(iree_hal_amd_xdna_image_array(image.get(), 1), nullptr);
  ASSERT_EQ(iree_hal_amd_xdna_image_control_count(image.get()), 1u);
  EXPECT_NE(iree_hal_amd_xdna_image_control(image.get(), 0), nullptr);
  EXPECT_EQ(iree_hal_amd_xdna_image_control(image.get(), 1), nullptr);
  ASSERT_EQ(iree_hal_amd_xdna_image_record_count(image.get()), 2u);
  EXPECT_NE(iree_hal_amd_xdna_image_record(image.get(), 0), nullptr);
  EXPECT_EQ(iree_hal_amd_xdna_image_record(image.get(), 2), nullptr);

  EXPECT_EQ(iree_hal_amd_xdna_image_structural_capabilities(image.get()),
            kAie2pImageStructuralCapabilities);
  ASSERT_EQ(iree_hal_amd_xdna_image_tile_placement_count(image.get()), 1u);
  const iree_hal_amd_xdna_image_tile_placement_t* placement =
      iree_hal_amd_xdna_image_tile_placement(image.get(), 0);
  ASSERT_NE(placement, nullptr);
  EXPECT_EQ(placement->program_header_ordinal,
            kAie2pImageTileProgramHeaderOrdinal);
  EXPECT_EQ(iree_hal_amd_xdna_image_tile_placement(image.get(), 1), nullptr);
  EXPECT_EQ(target_state.configuration_register_call_count, 1u);

  const iree_hal_amd_xdna_image_program_header_t* tile_header =
      iree_hal_amd_xdna_image_program_header(
          image.get(), kAie2pImageTileProgramHeaderOrdinal);
  ASSERT_NE(tile_header, nullptr);
  std::vector<uint8_t> tile_bytes(tile_header->file_range.length);
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_read_source_range(
      image.get(), tile_header->file_range,
      iree_make_byte_span(tile_bytes.data(), tile_bytes.size())));
  const auto expected_tile_begin =
      bytes.begin() + tile_header->file_range.offset;
  EXPECT_EQ(tile_bytes,
            std::vector<uint8_t>(
                expected_tile_begin,
                expected_tile_begin + tile_header->file_range.length));

  SourceEnumerator enumerator;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_enumerate_source_range(
      image.get(), tile_header->file_range,
      (iree_byte_sequence_segment_callback_t){
          /*.fn=*/AppendSourceSegment,
          /*.user_data=*/&enumerator,
      }));
  EXPECT_EQ(enumerator.bytes, tile_bytes);
}

TEST(ImageTest, LeavesOutputEmptyOnConstructionFailure) {
  constexpr uint32_t kElfTargetFlagsOffset = 36;
  std::vector<uint8_t> bytes = BuildAie2pImage();
  bytes[kElfTargetFlagsOffset] ^= 1;
  ByteSequencePtr sequence = MakeOwnedByteSequence(bytes);
  Aie2pImageTargetState target_state;
  const iree_hal_amd_xdna_aie2p_target_t aie2p_target =
      MakeAie2pImageTarget(&target_state);
  iree_hal_amd_xdna_image_target_t target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      &aie2p_target, &target));

  auto* image = reinterpret_cast<iree_hal_amd_xdna_image_t*>(uintptr_t{1});
  EXPECT_THAT(Status(iree_hal_amd_xdna_image_create(
                  sequence.get(), &target, iree_allocator_system(), &image)),
              StatusIs(StatusCode::kFailedPrecondition));
  EXPECT_EQ(image, nullptr);
}

}  // namespace
