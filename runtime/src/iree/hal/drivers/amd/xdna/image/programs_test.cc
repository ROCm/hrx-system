// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/programs.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_builder.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::ImageBuilder;
using iree::hal::amd::xdna::testing::MakeProgramHeader;
using testing::HasSubstr;

struct ByteSequenceDeleter {
  void operator()(iree_byte_sequence_t* sequence) const {
    iree_byte_sequence_release(sequence);
  }
};

using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, ByteSequenceDeleter>;

struct DirectoryDeleter {
  void operator()(iree_hal_amd_xdna_image_directory_t* directory) const {
    iree_hal_amd_xdna_image_directory_destroy(directory);
  }
};

using DirectoryPtr =
    std::unique_ptr<iree_hal_amd_xdna_image_directory_t, DirectoryDeleter>;

struct ProgramsDeleter {
  void operator()(iree_hal_amd_xdna_image_programs_t* programs) const {
    iree_hal_amd_xdna_image_programs_destroy(programs);
  }
};

using ProgramsPtr =
    std::unique_ptr<iree_hal_amd_xdna_image_programs_t, ProgramsDeleter>;

enum : uint32_t {
  kArrayOffset = 128,
  kControlOffset = 188,
};

typedef struct validation_call_t {
  // Program-header ordinal supplied to the validator.
  uint32_t program_header_ordinal;
  // ARRAY or CONTROL program type supplied to the validator.
  iree_hal_amd_xdna_elf_program_type_t program_type;
  // Dense record ordinal within the containing program.
  uint32_t program_record_ordinal;
  // Target-owned record type.
  uint16_t record_type;
  // Target-owned record flags.
  uint16_t record_flags;
  // Complete record byte length.
  iree_host_size_t record_length;
} validation_call_t;

typedef struct validator_state_t {
  // Calls observed in source program order.
  mutable std::vector<validation_call_t> calls;
  // Flattened call ordinal to reject, or IREE_HOST_SIZE_MAX to accept all.
  iree_host_size_t rejection_call_ordinal;
  // Program reference reported for every accepted record.
  uint32_t referenced_program_header_ordinal;
} validator_state_t;

static iree_status_t validate_record(
    const void* user_data, uint32_t program_header_ordinal,
    iree_hal_amd_xdna_elf_program_type_t program_type,
    uint32_t program_record_ordinal,
    const iree_hal_amd_xdna_elf_program_record_header_t* record_header,
    iree_const_byte_span_t record_storage,
    uint32_t* out_referenced_program_header_ordinal) {
  const auto* state = static_cast<const validator_state_t*>(user_data);
  state->calls.push_back({
      /*.program_header_ordinal=*/program_header_ordinal,
      /*.program_type=*/program_type,
      /*.program_record_ordinal=*/program_record_ordinal,
      /*.record_type=*/record_header->type,
      /*.record_flags=*/record_header->flags,
      /*.record_length=*/record_storage.data_length,
  });
  if (state->calls.size() - 1 == state->rejection_call_ordinal) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "target rejected program record");
  }
  *out_referenced_program_header_ordinal =
      state->referenced_program_header_ordinal;
  return iree_ok_status();
}

static iree_hal_amd_xdna_image_program_record_validator_t make_validator(
    validator_state_t* state) {
  return {
      /*.fn=*/validate_record,
      /*.user_data=*/state,
  };
}

static ByteSequencePtr make_owned_sequence(const std::vector<uint8_t>& bytes) {
  iree_byte_span_t storage = iree_byte_span_empty();
  storage.data_length = bytes.size();
  IREE_CHECK_OK(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), storage.data_length, (void**)&storage.data));
  memcpy(storage.data, bytes.data(), bytes.size());
  iree_byte_sequence_t* sequence = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &storage, iree_allocator_system(), &sequence));
  return ByteSequencePtr(sequence);
}

static std::vector<uint8_t> encode_record(uint16_t type, uint16_t flags,
                                          uint32_t byte_length) {
  const iree_hal_amd_xdna_elf_program_record_header_t header = {
      /*.type=*/type,
      /*.flags=*/flags,
      /*.byte_length=*/byte_length,
  };
  std::vector<uint8_t> bytes(byte_length, 0xCC);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_program_record_header(
      &header,
      iree_make_byte_span(bytes.data(),
                          IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE)));
  return bytes;
}

static std::vector<uint8_t> encode_array(
    uint32_t first_tile_program_header_ordinal,
    uint32_t tile_program_header_count,
    const std::vector<std::vector<uint8_t>>& records) {
  uint32_t byte_length = IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE;
  for (const auto& record : records) byte_length += record.size();
  const iree_hal_amd_xdna_elf_array_header_t header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/(uint32_t)records.size(),
      /*.byte_length=*/byte_length,
      /*.flags=*/0,
      /*.first_tile_program_header_ordinal=*/
      first_tile_program_header_ordinal,
      /*.tile_program_header_count=*/tile_program_header_count,
  };
  std::vector<uint8_t> bytes(byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_array_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE)));
  iree_host_size_t offset = IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE;
  for (const auto& record : records) {
    memcpy(bytes.data() + offset, record.data(), record.size());
    offset += record.size();
  }
  return bytes;
}

static std::vector<uint8_t> encode_control(
    const std::vector<std::vector<uint8_t>>& records) {
  uint32_t byte_length = IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE;
  for (const auto& record : records) byte_length += record.size();
  const iree_hal_amd_xdna_elf_control_header_t header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/(uint32_t)records.size(),
      /*.byte_length=*/byte_length,
      /*.flags=*/0,
  };
  std::vector<uint8_t> bytes(byte_length);
  IREE_CHECK_OK(iree_hal_amd_xdna_elf_encode_control_header(
      &header, iree_make_byte_span(bytes.data(),
                                   IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE)));
  iree_host_size_t offset = IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE;
  for (const auto& record : records) {
    memcpy(bytes.data() + offset, record.data(), record.size());
    offset += record.size();
  }
  return bytes;
}

static std::vector<uint8_t> make_valid_image() {
  const std::vector<uint8_t> array =
      encode_array(/*first_tile_program_header_ordinal=*/7,
                   /*tile_program_header_count=*/3,
                   {encode_record(11, 1, 16), encode_record(12, 2, 12)});
  const std::vector<uint8_t> control =
      encode_control({encode_record(21, 3, 8)});
  ImageBuilder builder;
  builder
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                                    kArrayOffset, array.size()),
                  array)
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL,
                                    kControlOffset, control.size()),
                  control);
  return builder.Build();
}

static DirectoryPtr open_directory(const std::vector<uint8_t>& bytes,
                                   ByteSequencePtr* out_sequence = nullptr) {
  ByteSequencePtr sequence = make_owned_sequence(bytes);
  iree_hal_amd_xdna_image_directory_t* directory = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_directory_create(
      sequence.get(), iree_allocator_system(), &directory));
  if (out_sequence != nullptr) *out_sequence = std::move(sequence);
  return DirectoryPtr(directory);
}

static ProgramsPtr open_programs(const std::vector<uint8_t>& bytes,
                                 validator_state_t* validator_state) {
  DirectoryPtr directory = open_directory(bytes);
  iree_hal_amd_xdna_image_programs_t* programs = nullptr;
  IREE_CHECK_OK(iree_hal_amd_xdna_image_programs_create(
      directory.get(), make_validator(validator_state), iree_allocator_system(),
      &programs));
  return ProgramsPtr(programs);
}

static Status open_programs_status(const std::vector<uint8_t>& bytes,
                                   validator_state_t* validator_state) {
  DirectoryPtr directory = open_directory(bytes);
  iree_hal_amd_xdna_image_programs_t* programs = nullptr;
  Status status(iree_hal_amd_xdna_image_programs_create(
      directory.get(), make_validator(validator_state), iree_allocator_system(),
      &programs));
  iree_hal_amd_xdna_image_programs_destroy(programs);
  return status;
}

static void expect_status(const Status& status, StatusCode expected_code,
                          const char* expected_message_substring) {
  EXPECT_EQ(status.code(), expected_code);
  EXPECT_THAT(status.ToString(), HasSubstr(expected_message_substring));
}

static uint8_t* program_header(std::vector<uint8_t>& bytes, uint32_t ordinal) {
  return bytes.data() + IREE_HAL_AMD_XDNA_ELF_HEADER_SIZE +
         ordinal * IREE_HAL_AMD_XDNA_ELF_PROGRAM_HEADER_SIZE;
}

TEST(ImageProgramsTest, FramesAndValidatesPrograms) {
  validator_state_t validator_state = {
      /*.calls=*/{},
      /*.rejection_call_ordinal=*/IREE_HOST_SIZE_MAX,
      /*.referenced_program_header_ordinal=*/9,
  };
  ByteSequencePtr sequence;
  DirectoryPtr directory = open_directory(make_valid_image(), &sequence);
  iree_hal_amd_xdna_image_programs_t* raw_programs = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_programs_create(
      directory.get(), make_validator(&validator_state),
      iree_allocator_system(), &raw_programs));
  ProgramsPtr programs(raw_programs);
  directory.reset();
  sequence.reset();

  ASSERT_EQ(iree_hal_amd_xdna_image_programs_array_count(programs.get()), 1);
  const auto* array = iree_hal_amd_xdna_image_programs_array(programs.get(), 0);
  ASSERT_NE(array, nullptr);
  EXPECT_EQ(array->program_header_ordinal, 0);
  EXPECT_EQ(array->first_tile_program_header_ordinal, 7);
  EXPECT_EQ(array->tile_program_header_count, 3);
  EXPECT_EQ(array->first_record_ordinal, 0);
  EXPECT_EQ(array->record_count, 2);
  EXPECT_EQ(iree_hal_amd_xdna_image_programs_array(programs.get(), 1), nullptr);

  ASSERT_EQ(iree_hal_amd_xdna_image_programs_control_count(programs.get()), 1);
  const auto* control =
      iree_hal_amd_xdna_image_programs_control(programs.get(), 0);
  ASSERT_NE(control, nullptr);
  EXPECT_EQ(control->program_header_ordinal, 1);
  EXPECT_EQ(control->first_record_ordinal, 2);
  EXPECT_EQ(control->record_count, 1);

  ASSERT_EQ(iree_hal_amd_xdna_image_programs_record_count(programs.get()), 3);
  const auto* first_record =
      iree_hal_amd_xdna_image_programs_record(programs.get(), 0);
  const auto* second_record =
      iree_hal_amd_xdna_image_programs_record(programs.get(), 1);
  const auto* third_record =
      iree_hal_amd_xdna_image_programs_record(programs.get(), 2);
  ASSERT_NE(first_record, nullptr);
  ASSERT_NE(second_record, nullptr);
  ASSERT_NE(third_record, nullptr);
  EXPECT_EQ(first_record->type, 11);
  EXPECT_EQ(first_record->flags, 1);
  EXPECT_EQ(first_record->referenced_program_header_ordinal, 9);
  EXPECT_EQ(first_record->source_range.offset, kArrayOffset + 32);
  EXPECT_EQ(first_record->source_range.length, 16);
  EXPECT_EQ(second_record->program_record_ordinal, 1);
  EXPECT_EQ(second_record->source_range.offset, kArrayOffset + 48);
  EXPECT_EQ(second_record->source_range.length, 12);
  EXPECT_EQ(third_record->program_header_ordinal, 1);
  EXPECT_EQ(third_record->program_record_ordinal, 0);
  EXPECT_EQ(third_record->source_range.offset, kControlOffset + 24);
  EXPECT_EQ(third_record->source_range.length, 8);
  EXPECT_EQ(iree_hal_amd_xdna_image_programs_record(programs.get(), 3),
            nullptr);

  ASSERT_EQ(validator_state.calls.size(), 3);
  EXPECT_EQ(validator_state.calls[0].program_type,
            IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY);
  EXPECT_EQ(validator_state.calls[0].record_length, 16);
  EXPECT_EQ(validator_state.calls[2].program_type,
            IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL);
  EXPECT_EQ(validator_state.calls[2].record_flags, 3);
}

TEST(ImageProgramsTest, RequiresARecordValidator) {
  DirectoryPtr directory = open_directory(make_valid_image());
  iree_hal_amd_xdna_image_programs_t* programs = nullptr;
  expect_status(Status(iree_hal_amd_xdna_image_programs_create(
                    directory.get(), {/*.fn=*/nullptr, /*.user_data=*/nullptr},
                    iree_allocator_system(), &programs)),
                StatusCode::kInvalidArgument, "record validator is required");
  EXPECT_EQ(programs, nullptr);
}

TEST(ImageProgramsTest, RejectsMalformedProgramMemoryContracts) {
  validator_state_t validator_state = {
      /*.calls=*/{},
      /*.rejection_call_ordinal=*/IREE_HOST_SIZE_MAX,
      /*.referenced_program_header_ordinal=*/UINT32_MAX,
  };
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(program_header(bytes, 0) + 24,
                                IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_READ |
                                    IREE_HAL_AMD_XDNA_ELF_PROGRAM_FLAG_WRITE);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument, "invalid memory contract");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(program_header(bytes, 0) + 28, 8);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument, "noncanonical alignment");
  }
}

TEST(ImageProgramsTest, RejectsMalformedProgramHeaders) {
  validator_state_t validator_state = {
      /*.calls=*/{},
      /*.rejection_call_ordinal=*/IREE_HOST_SIZE_MAX,
      /*.referenced_program_header_ordinal=*/UINT32_MAX,
  };
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset, 0);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument, "header magic is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + kArrayOffset + 4, 2);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kUnimplemented,
                  "unsupported XDNA program ABI version");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 16, 32);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument,
                  "program header contract is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 20, 1);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument,
                  "program header contract is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 12, 0);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument,
                  "program header contract is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 28, 0);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument,
                  "program header contract is invalid");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 12, 5);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kOutOfRange,
                  "program records exceed their payload");
  }
}

TEST(ImageProgramsTest, RejectsMalformedRecordFraming) {
  validator_state_t validator_state = {
      /*.calls=*/{},
      /*.rejection_call_ordinal=*/IREE_HOST_SIZE_MAX,
      /*.referenced_program_header_ordinal=*/UINT32_MAX,
  };
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u16(bytes.data() + kArrayOffset + 32, 0);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument, "noncanonical framing");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 36, 64);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kOutOfRange,
                  "program record exceeds its payload");
  }
  {
    std::vector<uint8_t> bytes = make_valid_image();
    iree_unaligned_store_le_u32(bytes.data() + kArrayOffset + 52, 8);
    expect_status(open_programs_status(bytes, &validator_state),
                  StatusCode::kInvalidArgument,
                  "does not consume its complete payload");
  }
}

TEST(ImageProgramsTest, PropagatesTargetValidatorFailures) {
  validator_state_t validator_state = {
      /*.calls=*/{},
      /*.rejection_call_ordinal=*/1,
      /*.referenced_program_header_ordinal=*/UINT32_MAX,
  };
  expect_status(open_programs_status(make_valid_image(), &validator_state),
                StatusCode::kAborted, "target rejected program record");
  EXPECT_EQ(validator_state.calls.size(), 2);
}

TEST(ImageProgramsTest, RejectsImageWideRecordCountOverflow) {
  constexpr uint32_t kArrayRecordCount = 40000;
  constexpr uint32_t kControlRecordCount = 30000;
  std::vector<uint8_t> array(
      IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE +
      kArrayRecordCount * IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE);
  const iree_hal_amd_xdna_elf_array_header_t array_header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/kArrayRecordCount,
      /*.byte_length=*/(uint32_t)array.size(),
      /*.flags=*/0,
      /*.first_tile_program_header_ordinal=*/2,
      /*.tile_program_header_count=*/1,
  };
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_array_header(
      &array_header,
      iree_make_byte_span(array.data(),
                          IREE_HAL_AMD_XDNA_ELF_ARRAY_HEADER_SIZE)));
  std::vector<uint8_t> control(
      IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE +
      kControlRecordCount * IREE_HAL_AMD_XDNA_ELF_PROGRAM_RECORD_HEADER_SIZE);
  const iree_hal_amd_xdna_elf_control_header_t control_header = {
      /*.abi_major=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MAJOR,
      /*.abi_minor=*/IREE_HAL_AMD_XDNA_ELF_PROGRAM_ABI_MINOR,
      /*.record_count=*/kControlRecordCount,
      /*.byte_length=*/(uint32_t)control.size(),
      /*.flags=*/0,
  };
  IREE_ASSERT_OK(iree_hal_amd_xdna_elf_encode_control_header(
      &control_header,
      iree_make_byte_span(control.data(),
                          IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE)));
  const uint32_t control_offset = kArrayOffset + array.size();
  ImageBuilder builder;
  builder
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_ARRAY,
                                    kArrayOffset, array.size()),
                  array)
      .AddProgram(MakeProgramHeader(IREE_HAL_AMD_XDNA_ELF_PROGRAM_TYPE_CONTROL,
                                    control_offset, control.size()),
                  control);
  validator_state_t validator_state = {
      /*.calls=*/{},
      /*.rejection_call_ordinal=*/IREE_HOST_SIZE_MAX,
      /*.referenced_program_header_ordinal=*/UINT32_MAX,
  };
  expect_status(open_programs_status(builder.Build(), &validator_state),
                StatusCode::kOutOfRange, "too many program records");
  EXPECT_TRUE(validator_state.calls.empty());
}

}  // namespace
