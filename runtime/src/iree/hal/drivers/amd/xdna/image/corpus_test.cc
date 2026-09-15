// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/program_format.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"
#include "iree/hal/drivers/amd/xdna/image/image.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::Aie2pImageTargetState;
using iree::hal::amd::xdna::testing::ByteSequencePtr;
using iree::hal::amd::xdna::testing::MakeAie2pImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;
using testing::HasSubstr;

enum : uint32_t {
  kNoteProgramOrdinal = 0,
  kEntryProgramOrdinal = 1,
  kBindingProgramOrdinal = 2,
  kRelocationProgramOrdinal = 3,
  kFirstTileProgramOrdinal = 4,
  kArrayProgramOrdinal = 5,
  kControlProgramOrdinal = 6,
};

struct ImageDeleter {
  void operator()(iree_hal_amd_xdna_image_t* image) const {
    iree_hal_amd_xdna_image_destroy(image);
  }
};

using ImagePtr = std::unique_ptr<iree_hal_amd_xdna_image_t, ImageDeleter>;

// Complete target objects whose borrowed callback state shares one lifetime.
struct ImageTargetFixture {
  ImageTargetFixture() : aie2p_target(MakeAie2pImageTarget(&state)) {
    aie2p_target.context.column_count = 1;
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_target_initialize_image_target(
        &aie2p_target, &image_target));
  }
  ImageTargetFixture(const ImageTargetFixture&) = delete;
  ImageTargetFixture& operator=(const ImageTargetFixture&) = delete;

  // Mutable observations made by target qualification callbacks.
  Aie2pImageTargetState state = {};
  // AIE2P target facts borrowed by |image_target|.
  iree_hal_amd_xdna_aie2p_target_t aie2p_target = {};
  // Generic target contract passed to image construction.
  iree_hal_amd_xdna_image_target_t image_target = {};
};

static std::vector<uint8_t> LoadMulI32ImageBytes() {
  EXPECT_EQ(iree_hal_amd_xdna_test_mul_i32_size(), 1u);
  if (iree_hal_amd_xdna_test_mul_i32_size() != 1u) return {};
  const iree_file_toc_t* file = iree_hal_amd_xdna_test_mul_i32_create();
  EXPECT_STREQ(file->name, "mul_i32.xdna");
  const auto* data = reinterpret_cast<const uint8_t*>(file->data);
  return std::vector<uint8_t>(data, data + file->size);
}

static ImagePtr CreateImage(iree_byte_sequence_t* sequence,
                            const ImageTargetFixture& target) {
  iree_hal_amd_xdna_image_t* image = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_create(
      sequence, &target.image_target, iree_allocator_system(), &image));
  return ImagePtr(image);
}

static ImagePtr CreateOwnedImage(const std::vector<uint8_t>& bytes,
                                 const ImageTargetFixture& target) {
  ByteSequencePtr sequence = MakeOwnedByteSequence(bytes);
  return CreateImage(sequence.get(), target);
}

struct SourceCollector {
  // Source bytes appended in logical order.
  std::vector<uint8_t> bytes;
};

static iree_status_t AppendSourceSegment(void* user_data,
                                         iree_const_byte_span_t segment) {
  auto* collector = static_cast<SourceCollector*>(user_data);
  collector->bytes.insert(collector->bytes.end(), segment.data,
                          segment.data + segment.data_length);
  return iree_ok_status();
}

struct ProgramHeaderExpectation {
  // Expected ELF program type.
  uint32_t type;
  // Expected payload byte offset.
  uint64_t file_offset;
  // Expected payload byte length and memory extent.
  uint64_t byte_length;
  // Expected target physical address or placement descriptor.
  uint32_t physical_address;
  // Expected ELF permission flags.
  iree_hal_amd_xdna_elf_program_flags_t flags;
  // Expected ELF load alignment.
  uint32_t alignment;
};

static constexpr ProgramHeaderExpectation kExpectedProgramHeaders[] = {
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_NOTE, 276, 84, 0,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ, 4},
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ENTRIES, 360, 71, 0,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ, 8},
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_BINDINGS, 432, 192, 0,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ, 8},
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_RELOCATIONS, 624, 168, 0,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ, 8},
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_TILE, 800, 818, 0x00010200,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ |
         IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_EXECUTE,
     16},
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY, 1620, 1024, 0,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ, 4},
    {IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL, 2644, 252, 0,
     IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ, 4},
};

static void ExpectCanonicalMulI32Image(
    const std::vector<uint8_t>& expected_source,
    const iree_hal_amd_xdna_image_t* image) {
  EXPECT_EQ(iree_hal_amd_xdna_image_source_length(image),
            expected_source.size());
  EXPECT_EQ(iree_hal_amd_xdna_image_target_flags(image),
            IREE_HAL_AMD_XDNA_ELF_AIE2P_FLAGS);
  ASSERT_EQ(iree_hal_amd_xdna_image_program_header_count(image),
            std::size(kExpectedProgramHeaders));
  for (iree_host_size_t i = 0; i < std::size(kExpectedProgramHeaders); ++i) {
    SCOPED_TRACE(i);
    const auto* program_header =
        iree_hal_amd_xdna_image_program_header(image, i);
    ASSERT_NE(program_header, nullptr);
    EXPECT_EQ(program_header, iree_hal_amd_xdna_image_program_header(image, i));
    EXPECT_EQ(program_header->type, kExpectedProgramHeaders[i].type);
    EXPECT_EQ(program_header->file_range.offset,
              kExpectedProgramHeaders[i].file_offset);
    EXPECT_EQ(program_header->file_range.length,
              kExpectedProgramHeaders[i].byte_length);
    EXPECT_EQ(program_header->virtual_address, 0u);
    EXPECT_EQ(program_header->physical_address,
              kExpectedProgramHeaders[i].physical_address);
    EXPECT_EQ(program_header->memory_size,
              kExpectedProgramHeaders[i].byte_length);
    EXPECT_EQ(program_header->flags, kExpectedProgramHeaders[i].flags);
    EXPECT_EQ(program_header->alignment, kExpectedProgramHeaders[i].alignment);
  }
  EXPECT_EQ(iree_hal_amd_xdna_image_program_header(
                image, std::size(kExpectedProgramHeaders)),
            nullptr);

  const auto* abi_note = iree_hal_amd_xdna_image_abi_note(image);
  ASSERT_NE(abi_note, nullptr);
  EXPECT_EQ(abi_note, iree_hal_amd_xdna_image_abi_note(image));
  EXPECT_EQ(abi_note->abi_major, IREE_HAL_AMD_XDNA_ELF_ABI_MAJOR);
  EXPECT_EQ(abi_note->abi_minor, IREE_HAL_AMD_XDNA_ELF_ABI_MINOR);
  EXPECT_EQ(abi_note->target_generation,
            IREE_HAL_AMD_XDNA_TARGET_GENERATION_AIE2P);
  EXPECT_EQ(abi_note->device_profile_revision, 1u);
  EXPECT_EQ(abi_note->device_profile_id, UINT64_C(0x535848414C4F0001));
  EXPECT_EQ(abi_note->firmware_abi_id, UINT64_C(0x4E5055320006000C));
  EXPECT_EQ(abi_note->policy_id, UINT64_C(0x413250504C414E01));
  EXPECT_EQ(abi_note->required_capabilities,
            IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS |
                IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS);
  EXPECT_EQ(abi_note->context_origin_column, 0u);
  EXPECT_EQ(abi_note->context_origin_row, 0u);
  EXPECT_EQ(abi_note->context_column_count, 1u);
  EXPECT_EQ(abi_note->context_row_count, 6u);
  EXPECT_EQ(abi_note->coordinate_model,
            IREE_HAL_AMD_XDNA_ELF_COORDINATE_MODEL_CONTEXT_RELATIVE);

  ASSERT_EQ(iree_hal_amd_xdna_image_entry_count(image), 1u);
  const auto* entry = iree_hal_amd_xdna_image_entry(image, 0);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry, iree_hal_amd_xdna_image_entry(image, 0));
  EXPECT_EQ(entry->export_ordinal, 0u);
  EXPECT_EQ(entry->array_program_header_ordinal, kArrayProgramOrdinal);
  EXPECT_EQ(entry->control_program_header_ordinal, kControlProgramOrdinal);
  EXPECT_EQ(entry->first_binding_ordinal, 0u);
  EXPECT_EQ(entry->binding_count, 3u);
  EXPECT_EQ(entry->flags, IREE_HAL_AMD_XDNA_ELF_ENTRY_FLAG_DEFAULT);
  EXPECT_EQ(entry->required_capabilities,
            IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS |
                IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS);
  EXPECT_TRUE(iree_string_view_equal(
      iree_hal_amd_xdna_image_entry_name(image, 0), IREE_SV("mul_i32")));

  ASSERT_EQ(iree_hal_amd_xdna_image_binding_count(image), 3u);
  constexpr uint64_t kExpectedBindingByteLengths[] = {64, 64, 64};
  for (uint32_t i = 0; i < 3; ++i) {
    SCOPED_TRACE(i);
    const auto* binding = iree_hal_amd_xdna_image_binding(image, i);
    ASSERT_NE(binding, nullptr);
    EXPECT_EQ(binding, iree_hal_amd_xdna_image_binding(image, i));
    EXPECT_EQ(binding->binding_ordinal, i);
    EXPECT_EQ(binding->entry_ordinal, 0u);
    EXPECT_EQ(binding->kind, IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER);
    EXPECT_EQ(binding->address_space,
              IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL);
    EXPECT_EQ(binding->access, i == 2
                                   ? IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE
                                   : IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ);
    EXPECT_EQ(binding->usage,
              IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE |
                  IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_COHERENT);
    EXPECT_EQ(binding->minimum_byte_length, kExpectedBindingByteLengths[i]);
    EXPECT_EQ(binding->minimum_alignment, 4u);
    EXPECT_EQ(binding->minimum_byte_offset, 0u);
    EXPECT_EQ(binding->maximum_byte_offset, 0u);
  }

  constexpr uint32_t kExpectedRelocationOffsets[] = {44, 108, 172};
  constexpr uint32_t kExpectedRelocationBindings[] = {0, 1, 2};
  constexpr int64_t kExpectedRelocationAddends[] = {0, 0, 0};
  ASSERT_EQ(iree_hal_amd_xdna_image_relocation_count(image),
            std::size(kExpectedRelocationOffsets));
  for (iree_host_size_t i = 0; i < std::size(kExpectedRelocationOffsets); ++i) {
    SCOPED_TRACE(i);
    const auto* relocation = iree_hal_amd_xdna_image_relocation(image, i);
    ASSERT_NE(relocation, nullptr);
    EXPECT_EQ(relocation, iree_hal_amd_xdna_image_relocation(image, i));
    EXPECT_EQ(relocation->target_program_header_ordinal,
              kControlProgramOrdinal);
    EXPECT_EQ(relocation->target_byte_offset, kExpectedRelocationOffsets[i]);
    EXPECT_EQ(relocation->binding_ordinal, kExpectedRelocationBindings[i]);
    EXPECT_EQ(relocation->kind,
              IREE_HAL_AMD_XDNA_ELF_RELOCATION_KIND_BINDING_ADDRESS);
    EXPECT_EQ(relocation->field_byte_width, 8u);
    EXPECT_EQ(relocation->flags, 0u);
    EXPECT_EQ(relocation->addend, kExpectedRelocationAddends[i]);
    EXPECT_EQ(relocation->minimum_value, 0u);
    EXPECT_EQ(relocation->required_alignment, 4u);
  }

  ASSERT_EQ(iree_hal_amd_xdna_image_array_count(image), 1u);
  const auto* array = iree_hal_amd_xdna_image_array(image, 0);
  ASSERT_NE(array, nullptr);
  EXPECT_EQ(array->program_header_ordinal, kArrayProgramOrdinal);
  EXPECT_EQ(array->first_tile_program_header_ordinal, kFirstTileProgramOrdinal);
  EXPECT_EQ(array->tile_program_header_count, 1u);
  EXPECT_EQ(array->first_record_ordinal, 0u);
  EXPECT_EQ(array->record_count, 50u);

  ASSERT_EQ(iree_hal_amd_xdna_image_control_count(image), 1u);
  const auto* control = iree_hal_amd_xdna_image_control(image, 0);
  ASSERT_NE(control, nullptr);
  EXPECT_EQ(control->program_header_ordinal, kControlProgramOrdinal);
  EXPECT_EQ(control->first_record_ordinal, 50u);
  EXPECT_EQ(control->record_count, 8u);

  std::array<iree_host_size_t, 6> record_type_counts = {};
  std::vector<uint32_t> tile_program_ordinals;
  bool found_dma_task_wait = false;
  ASSERT_EQ(iree_hal_amd_xdna_image_record_count(image), 58u);
  for (iree_host_size_t i = 0; i < 58; ++i) {
    SCOPED_TRACE(i);
    const auto* record = iree_hal_amd_xdna_image_record(image, i);
    ASSERT_NE(record, nullptr);
    EXPECT_EQ(record, iree_hal_amd_xdna_image_record(image, i));
    EXPECT_EQ(record->program_header_ordinal,
              i < 50 ? kArrayProgramOrdinal : kControlProgramOrdinal);
    EXPECT_EQ(record->program_record_ordinal, i < 50 ? i : i - 50);
    std::vector<uint8_t> record_storage(record->source_range.length);
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_read_source_range(
        image, record->source_range,
        iree_make_byte_span(record_storage.data(), record_storage.size())));
    iree_hal_amd_xdna_aie2p_program_record_t decoded_record;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_decode_program_record(
        iree_make_const_byte_span(record_storage.data(), record_storage.size()),
        &decoded_record));
    ASSERT_GT(decoded_record.type, 0);
    const iree_host_size_t record_type =
        static_cast<iree_host_size_t>(decoded_record.type);
    ASSERT_LT(record_type, record_type_counts.size());
    ++record_type_counts[record_type];
    if (decoded_record.type ==
        IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD) {
      tile_program_ordinals.push_back(
          decoded_record.value.tile_program_load.program_header_ordinal);
    } else if (decoded_record.type ==
               IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_DMA_TASK_WAIT) {
      const auto& wait = decoded_record.value.dma_task_wait;
      EXPECT_FALSE(found_dma_task_wait);
      found_dma_task_wait = true;
      EXPECT_EQ(wait.column, 0u);
      EXPECT_EQ(wait.row, 0u);
      EXPECT_EQ(wait.direction,
                IREE_HAL_AMD_XDNA_AIE2P_DMA_DIRECTION_STREAM_TO_MEMORY);
      EXPECT_EQ(wait.dma_channel, 0u);
      EXPECT_EQ(wait.column_count, 1u);
      EXPECT_EQ(wait.row_count, 1u);
    }
  }
  EXPECT_EQ(record_type_counts[1], 33u);
  EXPECT_EQ(record_type_counts[2], 14u);
  EXPECT_EQ(record_type_counts[3], 9u);
  EXPECT_EQ(record_type_counts[4], 1u);
  EXPECT_EQ(record_type_counts[5], 1u);
  EXPECT_EQ(tile_program_ordinals,
            (std::vector<uint32_t>{kFirstTileProgramOrdinal}));
  EXPECT_TRUE(found_dma_task_wait);

  EXPECT_EQ(iree_hal_amd_xdna_image_structural_capabilities(image),
            IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS |
                IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS);
  constexpr uint16_t kExpectedPlacementRows[] = {2};
  constexpr uint32_t kExpectedPlacementLengths[] = {818};
  ASSERT_EQ(iree_hal_amd_xdna_image_tile_placement_count(image), 1u);
  for (iree_host_size_t i = 0; i < std::size(kExpectedPlacementRows); ++i) {
    SCOPED_TRACE(i);
    const auto* placement = iree_hal_amd_xdna_image_tile_placement(image, i);
    ASSERT_NE(placement, nullptr);
    EXPECT_EQ(placement, iree_hal_amd_xdna_image_tile_placement(image, i));
    EXPECT_EQ(placement->program_header_ordinal, kFirstTileProgramOrdinal + i);
    EXPECT_EQ(placement->owner_column, 0u);
    EXPECT_EQ(placement->owner_row, kExpectedPlacementRows[i]);
    EXPECT_EQ(placement->memory_space,
              IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM);
    EXPECT_EQ(placement->owner_offset, 0u);
    EXPECT_EQ(placement->byte_length, kExpectedPlacementLengths[i]);
    EXPECT_EQ(placement->available_capacity, 4096u);
  }

  SourceCollector collector;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_enumerate_source_range(
      image,
      {
          /*.offset=*/0,
          /*.length=*/expected_source.size(),
      },
      {
          /*.fn=*/AppendSourceSegment,
          /*.user_data=*/&collector,
      }));
  EXPECT_EQ(collector.bytes, expected_source);
}

TEST(XdnaImageCorpusTest, QualifiesExactLoomMulI32Image) {
  const std::vector<uint8_t> bytes = LoadMulI32ImageBytes();
  ASSERT_EQ(bytes.size(), 3524u);
  ImageTargetFixture target;
  ImagePtr image = CreateOwnedImage(bytes, target);

  EXPECT_EQ(target.state.configuration_register_call_count, 56u);
  ExpectCanonicalMulI32Image(bytes, image.get());
  EXPECT_EQ(target.state.configuration_register_call_count, 56u);
}

// A deliberately non-contiguous sequence whose every source byte is a segment.
struct ByteSegmentedSequence {
  // Byte sequence interface retained by the image.
  iree_byte_sequence_t base;
  // One-byte segments borrowing the test's source vector.
  std::vector<iree_const_byte_span_t> segments;
  // Number of final releases observed by the test.
  int destroy_count;
};

static void DestroyByteSegmentedSequence(iree_byte_sequence_t* base_sequence) {
  auto* sequence = reinterpret_cast<ByteSegmentedSequence*>(base_sequence);
  ++sequence->destroy_count;
}

static iree_status_t EnumerateByteSegmentedSequence(
    const iree_byte_sequence_t* base_sequence,
    iree_byte_sequence_segment_callback_t callback) {
  const auto* sequence =
      reinterpret_cast<const ByteSegmentedSequence*>(base_sequence);
  for (iree_const_byte_span_t segment : sequence->segments) {
    IREE_RETURN_IF_ERROR(callback.fn(callback.user_data, segment));
  }
  return iree_ok_status();
}

static const iree_byte_sequence_vtable_t kByteSegmentedSequenceVtable = {
    /*.destroy=*/DestroyByteSegmentedSequence,
    /*.enumerate=*/EnumerateByteSegmentedSequence,
    /*.try_get_contiguous_span=*/nullptr,
};

static void InitializeByteSegmentedSequence(
    const std::vector<uint8_t>& bytes, ByteSegmentedSequence* out_sequence) {
  out_sequence->segments.reserve(bytes.size());
  for (const uint8_t& byte : bytes) {
    out_sequence->segments.push_back(iree_make_const_byte_span(&byte, 1));
  }
  iree_byte_sequence_initialize(&kByteSegmentedSequenceVtable, bytes.size(),
                                &out_sequence->base);
}

TEST(XdnaImageCorpusTest, QualifiesEveryFieldAcrossSegmentBoundaries) {
  const std::vector<uint8_t> bytes = LoadMulI32ImageBytes();
  ByteSegmentedSequence sequence = {};
  InitializeByteSegmentedSequence(bytes, &sequence);
  ImageTargetFixture target;
  ImagePtr image = CreateImage(&sequence.base, target);
  iree_byte_sequence_release(&sequence.base);
  EXPECT_EQ(sequence.destroy_count, 0);

  ExpectCanonicalMulI32Image(bytes, image.get());
  EXPECT_EQ(target.state.configuration_register_call_count, 56u);

  image.reset();
  EXPECT_EQ(sequence.destroy_count, 1);
}

static uint8_t* ProgramHeaderStorage(std::vector<uint8_t>& bytes,
                                     uint32_t ordinal) {
  return bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE +
         ordinal * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE;
}

static uint32_t ProgramFileOffset(const std::vector<uint8_t>& bytes,
                                  uint32_t ordinal) {
  const uint8_t* program_header =
      bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE +
      ordinal * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE;
  return iree_unaligned_load_le_u32(program_header + 4);
}

static uint8_t* FindProgramRecord(std::vector<uint8_t>& bytes,
                                  uint32_t program_ordinal,
                                  uint32_t program_header_size,
                                  uint16_t record_type) {
  const uint32_t program_offset = ProgramFileOffset(bytes, program_ordinal);
  const uint32_t record_count =
      iree_unaligned_load_le_u32(bytes.data() + program_offset + 12);
  uint8_t* record = bytes.data() + program_offset + program_header_size;
  for (uint32_t i = 0; i < record_count; ++i) {
    if (iree_unaligned_load_le_u16(record) == record_type) return record;
    record += iree_unaligned_load_le_u32(record + 4);
  }
  ADD_FAILURE() << "required program record is missing from the fixture";
  return nullptr;
}

static void MutateElfIdentity(std::vector<uint8_t>& bytes) { bytes[4] = 2; }

static void MutateOverlappingTileRange(std::vector<uint8_t>& bytes) {
  uint8_t* program_header =
      ProgramHeaderStorage(bytes, kFirstTileProgramOrdinal);
  iree_unaligned_store_le_u32(
      program_header + 4,
      ProgramFileOffset(bytes, kBindingProgramOrdinal) + 16);
}

static void MutateEntryCount(std::vector<uint8_t>& bytes) {
  const uint32_t entry_offset = ProgramFileOffset(bytes, kEntryProgramOrdinal);
  iree_unaligned_store_le_u32(bytes.data() + entry_offset + 12, 2);
}

static void MutateArrayRecordFraming(std::vector<uint8_t>& bytes) {
  const uint32_t array_offset = ProgramFileOffset(bytes, kArrayProgramOrdinal);
  iree_unaligned_store_le_u32(
      bytes.data() + array_offset + IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE + 4,
      4);
}

static void MutateTargetFlags(std::vector<uint8_t>& bytes) { bytes[36] ^= 1; }

static void MutateEntryProgramReference(std::vector<uint8_t>& bytes) {
  const uint32_t entry_offset = ProgramFileOffset(bytes, kEntryProgramOrdinal);
  iree_unaligned_store_le_u32(bytes.data() + entry_offset +
                                  IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE + 12,
                              kFirstTileProgramOrdinal);
}

static void MutateTileProgramReference(std::vector<uint8_t>& bytes) {
  uint8_t* record = FindProgramRecord(
      bytes, kArrayProgramOrdinal, IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE,
      IREE_HAL_AMD_XDNA_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD);
  if (record == nullptr) return;
  iree_unaligned_store_le_u32(record + 8, kRelocationProgramOrdinal);
}

static void MutateTileDestination(std::vector<uint8_t>& bytes) {
  uint8_t* program_header =
      ProgramHeaderStorage(bytes, kFirstTileProgramOrdinal);
  iree_unaligned_store_le_u32(program_header + 12, UINT32_C(0x00010209));
}

static void MutateRelocationTarget(std::vector<uint8_t>& bytes) {
  const uint32_t relocation_offset =
      ProgramFileOffset(bytes, kRelocationProgramOrdinal);
  uint8_t* final_relocation = bytes.data() + relocation_offset +
                              IREE_HAL_AMD_XDNA_ELF_TABLE_HEADER_SIZE +
                              2 * IREE_HAL_AMD_XDNA_ELF_RELOCATION_RECORD_SIZE;
  iree_unaligned_store_le_u32(final_relocation, 99);
}

struct MalformedImageCase {
  // Diagnostic label shown when the case fails.
  const char* name;
  // Mutation applied to one fresh copy of the canonical image.
  void (*mutate)(std::vector<uint8_t>& bytes);
  // Expected terminal status code.
  StatusCode status_code;
  // Stable rejection reason identifying the construction phase reached.
  const char* message_substring;
};

TEST(XdnaImageCorpusTest, RejectsMutationsAcrossConstructionPhases) {
  static constexpr MalformedImageCase kCases[] = {
      {"ELF identity", MutateElfIdentity, StatusCode::kInvalidArgument,
       "not canonical ELF32LE"},
      {"overlapping TILE range", MutateOverlappingTileRange,
       StatusCode::kInvalidArgument, "partially overlap"},
      {"entry table count", MutateEntryCount, StatusCode::kOutOfRange,
       "fixed metadata records exceed their table"},
      {"ARRAY record framing", MutateArrayRecordFraming,
       StatusCode::kInvalidArgument, "noncanonical framing"},
      {"target flags", MutateTargetFlags, StatusCode::kFailedPrecondition,
       "ELF flags do not match the target"},
      {"entry program reference", MutateEntryProgramReference,
       StatusCode::kInvalidArgument, "invalid ARRAY realization"},
      {"TILE program reference", MutateTileProgramReference,
       StatusCode::kInvalidArgument, "invalid or duplicate TILE reference"},
      {"TILE destination", MutateTileDestination, StatusCode::kOutOfRange,
       "has an invalid destination"},
      {"relocation target", MutateRelocationTarget, StatusCode::kOutOfRange,
       "references an invalid program header"},
  };

  const std::vector<uint8_t> canonical_bytes = LoadMulI32ImageBytes();
  for (const MalformedImageCase& test_case : kCases) {
    SCOPED_TRACE(test_case.name);
    std::vector<uint8_t> bytes = canonical_bytes;
    test_case.mutate(bytes);
    ByteSequencePtr sequence = MakeOwnedByteSequence(bytes);
    ImageTargetFixture target;
    auto* image = reinterpret_cast<iree_hal_amd_xdna_image_t*>(uintptr_t{1});
    Status status(iree_hal_amd_xdna_image_create(
        sequence.get(), &target.image_target, iree_allocator_system(), &image));
    EXPECT_EQ(status.code(), test_case.status_code);
    EXPECT_THAT(status.ToString(), HasSubstr(test_case.message_substring));
    EXPECT_EQ(image, nullptr);
    if (status.ok()) iree_hal_amd_xdna_image_destroy(image);
  }
}

}  // namespace
