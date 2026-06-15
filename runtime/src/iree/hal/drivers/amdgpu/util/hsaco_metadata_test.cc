// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/hsaco_metadata.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

static void AppendByte(std::vector<uint8_t>* output, uint8_t value) {
  output->push_back(value);
}

static void AppendU16BE(std::vector<uint8_t>* output, uint16_t value) {
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)value);
}

static void AppendU32BE(std::vector<uint8_t>* output, uint32_t value) {
  output->push_back((uint8_t)(value >> 24));
  output->push_back((uint8_t)(value >> 16));
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)value);
}

static void AppendU32LE(std::vector<uint8_t>* output, uint32_t value) {
  output->push_back((uint8_t)value);
  output->push_back((uint8_t)(value >> 8));
  output->push_back((uint8_t)(value >> 16));
  output->push_back((uint8_t)(value >> 24));
}

static void StoreU16LE(std::vector<uint8_t>* output, size_t offset,
                       uint16_t value) {
  (*output)[offset + 0] = (uint8_t)value;
  (*output)[offset + 1] = (uint8_t)(value >> 8);
}

static void StoreU32LE(std::vector<uint8_t>* output, size_t offset,
                       uint32_t value) {
  (*output)[offset + 0] = (uint8_t)value;
  (*output)[offset + 1] = (uint8_t)(value >> 8);
  (*output)[offset + 2] = (uint8_t)(value >> 16);
  (*output)[offset + 3] = (uint8_t)(value >> 24);
}

static void StoreU64LE(std::vector<uint8_t>* output, size_t offset,
                       uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    (*output)[offset + i] = (uint8_t)(value >> (i * 8));
  }
}

static void AppendAligned4Padding(std::vector<uint8_t>* output) {
  while ((output->size() & 3) != 0) output->push_back(0);
}

static void AppendMsgPackMap(std::vector<uint8_t>* output, uint32_t count) {
  if (count < 16) {
    AppendByte(output, (uint8_t)(0x80 | count));
  } else {
    AppendByte(output, 0xDE);
    AppendU16BE(output, (uint16_t)count);
  }
}

static void AppendMsgPackArray(std::vector<uint8_t>* output, uint32_t count) {
  if (count < 16) {
    AppendByte(output, (uint8_t)(0x90 | count));
  } else {
    AppendByte(output, 0xDC);
    AppendU16BE(output, (uint16_t)count);
  }
}

static void AppendMsgPackString(std::vector<uint8_t>* output,
                                iree_string_view_t value) {
  if (value.size < 32) {
    AppendByte(output, (uint8_t)(0xA0 | value.size));
  } else if (value.size <= UINT8_MAX) {
    AppendByte(output, 0xD9);
    AppendByte(output, (uint8_t)value.size);
  } else {
    AppendByte(output, 0xDA);
    AppendU16BE(output, (uint16_t)value.size);
  }
  output->insert(output->end(), value.data, value.data + value.size);
}

static void AppendMsgPackUint(std::vector<uint8_t>* output, uint32_t value) {
  if (value <= 0x7F) {
    AppendByte(output, (uint8_t)value);
  } else if (value <= UINT8_MAX) {
    AppendByte(output, 0xCC);
    AppendByte(output, (uint8_t)value);
  } else if (value <= UINT16_MAX) {
    AppendByte(output, 0xCD);
    AppendU16BE(output, (uint16_t)value);
  } else {
    AppendByte(output, 0xCE);
    AppendU32BE(output, value);
  }
}

static void AppendStringField(std::vector<uint8_t>* output,
                              iree_string_view_t key,
                              iree_string_view_t value) {
  AppendMsgPackString(output, key);
  AppendMsgPackString(output, value);
}

static void AppendUintField(std::vector<uint8_t>* output,
                            iree_string_view_t key, uint32_t value) {
  AppendMsgPackString(output, key);
  AppendMsgPackUint(output, value);
}

static std::vector<uint8_t> BuildKernelMetadata(
    bool out_of_range_arg = false, bool unknown_value_kind = false,
    bool narrow_by_value_arg = false) {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 3);

  AppendMsgPackString(&output, IREE_SV("amdhsa.version"));
  AppendMsgPackArray(&output, 2);
  AppendMsgPackUint(&output, 1);
  AppendMsgPackUint(&output, 2);

  AppendStringField(&output, IREE_SV("amdhsa.target"),
                    IREE_SV("amdgcn-amd-amdhsa--gfx1100"));

  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 8);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("vector_add"));
  AppendStringField(&output, IREE_SV(".symbol"), IREE_SV("vector_add.kd"));
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 24);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 1024);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 64);
  AppendMsgPackString(&output, IREE_SV(".reqd_workgroup_size"));
  AppendMsgPackArray(&output, 3);
  AppendMsgPackUint(&output, 16);
  AppendMsgPackUint(&output, 4);
  AppendMsgPackUint(&output, 1);

  AppendMsgPackString(&output, IREE_SV(".args"));
  AppendMsgPackArray(&output, 4);

  AppendMsgPackMap(&output, 8);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("lhs"));
  AppendUintField(&output, IREE_SV(".offset"), 0);
  AppendUintField(&output, IREE_SV(".size"), 8);
  AppendStringField(
      &output, IREE_SV(".value_kind"),
      unknown_value_kind ? IREE_SV("made_up_kind") : IREE_SV("global_buffer"));
  AppendStringField(&output, IREE_SV(".address_space"), IREE_SV("global"));
  AppendStringField(&output, IREE_SV(".access"), IREE_SV("read_write"));
  AppendStringField(&output, IREE_SV(".actual_access"), IREE_SV("read_only"));
  AppendUintField(&output, IREE_SV(".align"), 8);

  AppendMsgPackMap(&output, 7);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("rhs"));
  AppendUintField(&output, IREE_SV(".offset"), 8);
  AppendUintField(&output, IREE_SV(".size"), 8);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("global_buffer"));
  AppendStringField(&output, IREE_SV(".address_space"), IREE_SV("global"));
  AppendStringField(&output, IREE_SV(".access"), IREE_SV("write_only"));
  AppendUintField(&output, IREE_SV(".align"), 8);

  AppendMsgPackMap(&output, 5);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("n"));
  AppendUintField(&output, IREE_SV(".offset"), 16);
  AppendUintField(&output, IREE_SV(".size"), narrow_by_value_arg ? 2 : 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  AppendUintField(&output, IREE_SV(".align"), 4);

  AppendMsgPackMap(&output, 5);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("alpha"));
  AppendUintField(&output, IREE_SV(".offset"), out_of_range_arg ? 20 : 20);
  AppendUintField(&output, IREE_SV(".size"), out_of_range_arg ? 8 : 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  AppendUintField(&output, IREE_SV(".align"), 4);

  return output;
}

static std::vector<uint8_t> BuildHiddenArgumentMetadata() {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 1);
  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 6);
  AppendStringField(&output, IREE_SV(".symbol"), IREE_SV("hidden_args.kd"));
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 20);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 0);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 0);
  AppendMsgPackString(&output, IREE_SV(".args"));
  AppendMsgPackArray(&output, 3);

  AppendMsgPackMap(&output, 5);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("buffer"));
  AppendUintField(&output, IREE_SV(".offset"), 0);
  AppendUintField(&output, IREE_SV(".size"), 8);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("global_buffer"));
  AppendUintField(&output, IREE_SV(".align"), 8);

  AppendMsgPackMap(&output, 5);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("grid_x"));
  AppendUintField(&output, IREE_SV(".offset"), 8);
  AppendUintField(&output, IREE_SV(".size"), 8);
  AppendStringField(&output, IREE_SV(".value_kind"),
                    IREE_SV("hidden_global_offset_x"));
  AppendUintField(&output, IREE_SV(".align"), 8);

  AppendMsgPackMap(&output, 5);
  AppendStringField(&output, IREE_SV(".name"), IREE_SV("value"));
  AppendUintField(&output, IREE_SV(".offset"), 16);
  AppendUintField(&output, IREE_SV(".size"), 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  AppendUintField(&output, IREE_SV(".align"), 4);

  return output;
}

static std::vector<uint8_t> BuildMalformedMissingKernelFieldsMetadata() {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 1);
  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 0);
  return output;
}

static std::vector<uint8_t> BuildDuplicateArgumentFieldMetadata() {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 1);
  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 6);
  AppendStringField(&output, IREE_SV(".symbol"), IREE_SV("duplicate.kd"));
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 8);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 0);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 0);
  AppendMsgPackString(&output, IREE_SV(".args"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(&output, 4);
  AppendUintField(&output, IREE_SV(".offset"), 0);
  AppendUintField(&output, IREE_SV(".offset"), 4);
  AppendUintField(&output, IREE_SV(".size"), 4);
  AppendStringField(&output, IREE_SV(".value_kind"), IREE_SV("by_value"));
  return output;
}

static std::vector<uint8_t> BuildElfWithNote(
    const std::vector<uint8_t>& metadata, iree_string_view_t note_name,
    uint32_t note_type) {
  constexpr size_t kElfHeaderSize = 64;
  constexpr size_t kProgramHeaderOffset = 64;
  constexpr size_t kProgramHeaderSize = 56;
  constexpr size_t kNoteOffset = 128;

  std::vector<uint8_t> note;
  AppendU32LE(&note, (uint32_t)note_name.size + 1);
  AppendU32LE(&note, (uint32_t)metadata.size());
  AppendU32LE(&note, note_type);
  note.insert(note.end(), note_name.data, note_name.data + note_name.size);
  note.push_back(0);
  AppendAligned4Padding(&note);
  note.insert(note.end(), metadata.begin(), metadata.end());
  AppendAligned4Padding(&note);

  std::vector<uint8_t> elf(kNoteOffset, 0);
  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;               // ELFCLASS64.
  elf[5] = 1;               // ELFDATA2LSB.
  elf[6] = 1;               // EV_CURRENT.
  StoreU16LE(&elf, 16, 3);  // ET_DYN.
  StoreU16LE(&elf, 18, 224);
  StoreU32LE(&elf, 20, 1);
  StoreU64LE(&elf, 32, kProgramHeaderOffset);
  StoreU16LE(&elf, 52, kElfHeaderSize);
  StoreU16LE(&elf, 54, kProgramHeaderSize);
  StoreU16LE(&elf, 56, 1);

  StoreU32LE(&elf, kProgramHeaderOffset + 0, 4);  // PT_NOTE.
  StoreU64LE(&elf, kProgramHeaderOffset + 8, kNoteOffset);
  StoreU64LE(&elf, kProgramHeaderOffset + 32, note.size());
  StoreU64LE(&elf, kProgramHeaderOffset + 40, note.size());
  StoreU64LE(&elf, kProgramHeaderOffset + 48, 4);

  elf.insert(elf.end(), note.begin(), note.end());
  return elf;
}

static std::vector<uint8_t> BuildElfWithMetadata(
    const std::vector<uint8_t>& metadata) {
  return BuildElfWithNote(metadata, IREE_SV("AMDGPU"), 32);
}

static void AlignVector(std::vector<uint8_t>* output, size_t alignment) {
  while ((output->size() % alignment) != 0) output->push_back(0);
}

static void AppendElf64Symbol(std::vector<uint8_t>* output,
                              uint32_t name_offset, uint8_t info,
                              uint16_t section_index = 1) {
  size_t symbol_offset = output->size();
  output->resize(symbol_offset + 24, 0);
  StoreU32LE(output, symbol_offset + 0, name_offset);
  (*output)[symbol_offset + 4] = info;
  StoreU16LE(output, symbol_offset + 6, section_index);
}

static std::vector<uint8_t> AddSyntheticCandidateSymbolSection(
    std::vector<uint8_t> elf, uint8_t descriptor_info = 0x11,
    uint16_t function_section_index = 1) {
  constexpr uint8_t kGlobalFunction = 0x12;  // STB_GLOBAL | STT_FUNC.
  constexpr size_t kSectionHeaderSize = 64;

  const size_t string_offset = elf.size();
  elf.push_back(0);
  const uint32_t function_name_offset =
      static_cast<uint32_t>(elf.size() - string_offset);
  const char kFunctionName[] = "extra_kernel";
  elf.insert(elf.end(), kFunctionName, kFunctionName + sizeof(kFunctionName));
  const uint32_t descriptor_name_offset =
      static_cast<uint32_t>(elf.size() - string_offset);
  const char kDescriptorName[] = "extra_kernel.kd";
  elf.insert(elf.end(), kDescriptorName,
             kDescriptorName + sizeof(kDescriptorName));
  const size_t string_size = elf.size() - string_offset;

  AlignVector(&elf, 8);
  const size_t symbol_offset = elf.size();
  AppendElf64Symbol(&elf, 0, 0);
  AppendElf64Symbol(&elf, function_name_offset, kGlobalFunction,
                    function_section_index);
  AppendElf64Symbol(&elf, descriptor_name_offset, descriptor_info);
  const size_t symbol_size = elf.size() - symbol_offset;

  AlignVector(&elf, 8);
  const size_t section_offset = elf.size();
  StoreU64LE(&elf, 40, section_offset);
  StoreU16LE(&elf, 58, kSectionHeaderSize);
  StoreU16LE(&elf, 60, 3);

  elf.resize(section_offset + 3 * kSectionHeaderSize, 0);
  const size_t string_section = section_offset + kSectionHeaderSize;
  StoreU32LE(&elf, string_section + 4, 3);  // SHT_STRTAB.
  StoreU64LE(&elf, string_section + 24, string_offset);
  StoreU64LE(&elf, string_section + 32, string_size);
  StoreU64LE(&elf, string_section + 48, 1);

  const size_t symbol_section = section_offset + 2 * kSectionHeaderSize;
  StoreU32LE(&elf, symbol_section + 4, 2);  // SHT_SYMTAB.
  StoreU64LE(&elf, symbol_section + 24, symbol_offset);
  StoreU64LE(&elf, symbol_section + 32, symbol_size);
  StoreU32LE(&elf, symbol_section + 40, 1);  // sh_link: string table section.
  StoreU64LE(&elf, symbol_section + 48, 8);
  StoreU64LE(&elf, symbol_section + 56, 24);
  return elf;
}

static std::vector<uint8_t> AddMalformedSymbolSection(
    std::vector<uint8_t> elf) {
  constexpr size_t kSectionHeaderSize = 64;
  size_t section_offset = elf.size();
  StoreU64LE(&elf, 40, section_offset);
  StoreU16LE(&elf, 58, kSectionHeaderSize);
  StoreU16LE(&elf, 60, 1);

  elf.resize(section_offset + kSectionHeaderSize, 0);
  StoreU32LE(&elf, section_offset + 4, 2);  // SHT_SYMTAB.
  // Point inside the ELF but declare a section size that extends beyond EOF.
  StoreU64LE(&elf, section_offset + 24, elf.size() - 8);
  StoreU64LE(&elf, section_offset + 32, 64);
  StoreU64LE(&elf, section_offset + 56, 24);
  return elf;
}

static iree_const_byte_span_t ByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(HsacoMetadataTest, ParsesValidMetadata) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata());

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  ASSERT_EQ(metadata.kernel_count, 1);
  ASSERT_EQ(metadata.arg_count, 4);
  ASSERT_GT(metadata.message_pack_data.data_length, 0);
  EXPECT_TRUE(iree_string_view_starts_with(metadata.target,
                                           IREE_SV("amdgcn-amd-amdhsa--gfx")));
  ASSERT_NE(metadata.kernels, nullptr);
  ASSERT_NE(metadata.args, nullptr);

  const iree_hal_amdgpu_hsaco_metadata_kernel_t& kernel = metadata.kernels[0];
  EXPECT_EQ(ToString(kernel.name), "vector_add");
  EXPECT_EQ(ToString(kernel.symbol_name), "vector_add.kd");
  EXPECT_EQ(ToString(kernel.reflection_name), "vector_add");
  EXPECT_EQ(kernel.kernarg_segment_size, 24);
  EXPECT_EQ(kernel.kernarg_segment_alignment, 8);
  EXPECT_EQ(kernel.group_segment_fixed_size, 1024);
  EXPECT_EQ(kernel.private_segment_fixed_size, 64);
  ASSERT_TRUE(kernel.has_required_workgroup_size);
  EXPECT_EQ(kernel.required_workgroup_size[0], 16);
  EXPECT_EQ(kernel.required_workgroup_size[1], 4);
  EXPECT_EQ(kernel.required_workgroup_size[2], 1);
  ASSERT_EQ(kernel.arg_count, 4);
  ASSERT_EQ(kernel.args, metadata.args);
  EXPECT_EQ(kernel.arg_name_storage_size, 12);
  EXPECT_EQ(metadata.reflection_name_storage_size, 10);
  EXPECT_EQ(metadata.arg_name_storage_size, 12);

  EXPECT_EQ(ToString(kernel.args[0].name), "lhs");
  EXPECT_EQ(kernel.args[0].offset, 0);
  EXPECT_EQ(kernel.args[0].size, 8);
  EXPECT_EQ(kernel.args[0].alignment, 8);
  EXPECT_EQ(kernel.args[0].kind,
            IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER);
  EXPECT_EQ(ToString(kernel.args[0].value_kind), "global_buffer");
  EXPECT_EQ(ToString(kernel.args[0].address_space), "global");
  EXPECT_EQ(ToString(kernel.args[0].access), "read_only");

  EXPECT_EQ(ToString(kernel.args[1].name), "rhs");
  EXPECT_EQ(kernel.args[1].offset, 8);
  EXPECT_EQ(kernel.args[1].size, 8);
  EXPECT_EQ(ToString(kernel.args[1].access), "write_only");

  EXPECT_EQ(ToString(kernel.args[2].name), "n");
  EXPECT_EQ(kernel.args[2].offset, 16);
  EXPECT_EQ(kernel.args[2].size, 4);
  EXPECT_EQ(kernel.args[2].kind,
            IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE);

  EXPECT_EQ(ToString(kernel.args[3].name), "alpha");
  EXPECT_EQ(kernel.args[3].offset, 20);
  EXPECT_EQ(kernel.args[3].size, 4);
  EXPECT_EQ(kernel.args[3].kind,
            IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE);

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, FindsKernelBySymbol) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata());

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel = nullptr;
  IREE_EXPECT_OK(iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
      &metadata, IREE_SV("vector_add.kd"), &kernel));
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(ToString(kernel->name), "vector_add");

  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
                            &metadata, IREE_SV("missing.kd"), &kernel));

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, AllowsUnknownValueKindAsOpaqueMetadata) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildKernelMetadata(/*out_of_range_arg=*/false,
                                               /*unknown_value_kind=*/true));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  ASSERT_EQ(metadata.kernel_count, 1);
  ASSERT_EQ(metadata.kernels[0].arg_count, 4);
  EXPECT_EQ(metadata.kernels[0].args[0].kind,
            IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_UNKNOWN);
  EXPECT_EQ(ToString(metadata.kernels[0].args[0].value_kind), "made_up_kind");

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, RejectsOutOfRangeArgument) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildKernelMetadata(/*out_of_range_arg=*/true));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(HsacoMetadataTest, RejectsMissingMetadataNote) {
  std::vector<uint8_t> elf =
      BuildElfWithNote(BuildKernelMetadata(), IREE_SV("OTHER"), 32);

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(HsacoMetadataTest, RejectsMalformedMessagePackMetadata) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildMalformedMissingKernelFieldsMetadata());

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(HsacoMetadataTest, DiscoversElfSymbolsWithoutSynthesizingKernels) {
  std::vector<uint8_t> elf = AddSyntheticCandidateSymbolSection(
      BuildElfWithMetadata(BuildKernelMetadata()));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  ASSERT_EQ(metadata.kernel_count, 1);
  EXPECT_EQ(ToString(metadata.kernels[0].symbol_name), "vector_add.kd");
  ASSERT_EQ(metadata.elf_kernel_symbol_count, 1);
  EXPECT_EQ(ToString(metadata.elf_kernel_symbols[0].name), "extra_kernel");
  EXPECT_EQ(ToString(metadata.elf_kernel_symbols[0].symbol_name),
            "extra_kernel.kd");

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, IgnoresElfFunctionWithWrongDescriptorType) {
  constexpr uint8_t kGlobalFunction = 0x12;  // STB_GLOBAL | STT_FUNC.
  std::vector<uint8_t> elf = AddSyntheticCandidateSymbolSection(
      BuildElfWithMetadata(BuildKernelMetadata()),
      /*descriptor_info=*/kGlobalFunction);

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  ASSERT_EQ(metadata.kernel_count, 1);
  EXPECT_EQ(metadata.elf_kernel_symbol_count, 0);

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, IgnoresUndefinedElfFunctionSymbol) {
  constexpr uint16_t kUndefinedSectionIndex = 0;
  std::vector<uint8_t> elf = AddSyntheticCandidateSymbolSection(
      BuildElfWithMetadata(BuildKernelMetadata()),
      /*descriptor_info=*/0x11, kUndefinedSectionIndex);

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  ASSERT_EQ(metadata.kernel_count, 1);
  EXPECT_EQ(metadata.elf_kernel_symbol_count, 0);

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, IgnoresMalformedElfSymbolSectionBounds) {
  std::vector<uint8_t> elf =
      AddMalformedSymbolSection(BuildElfWithMetadata(BuildKernelMetadata()));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  EXPECT_EQ(metadata.kernel_count, 1);
  EXPECT_EQ(metadata.elf_kernel_symbol_count, 0);
  EXPECT_EQ(ToString(metadata.kernels[0].symbol_name), "vector_add.kd");

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, RejectsDuplicateArgumentField) {
  std::vector<uint8_t> elf =
      BuildElfWithMetadata(BuildDuplicateArgumentFieldMetadata());

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(HsacoMetadataTest, TruncatedElfPrefixesNeverSucceed) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata());
  for (size_t length = 0; length < elf.size(); ++length) {
    iree_hal_amdgpu_hsaco_metadata_t metadata;
    iree_status_t status = iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
        iree_make_const_byte_span(elf.data(), length), iree_allocator_system(),
        &metadata);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
      ADD_FAILURE() << "unexpected success for truncated ELF prefix " << length;
      return;
    }
    iree_status_free(status);
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
