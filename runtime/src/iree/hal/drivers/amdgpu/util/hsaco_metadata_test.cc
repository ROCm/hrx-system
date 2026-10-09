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
#include "iree/hal/drivers/amdgpu/abi/asan.h"
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
  while ((output->size() & 3) != 0) {
    output->push_back(0);
  }
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

enum BuildKernelMetadataFlagBits : uint32_t {
  kBuildKernelMetadataNone = 0u,
  kBuildKernelMetadataOutOfRangeArg = 1u << 0,
  kBuildKernelMetadataUnknownValueKind = 1u << 1,
  kBuildKernelMetadataClusterDimensions = 1u << 2,
  kBuildKernelMetadataUniformWorkgroups = 1u << 3,
  kBuildKernelMetadataInvalidUniformWorkgroups = 1u << 4,
  kBuildKernelMetadataOmitMaxFlatWorkgroupSize = 1u << 5,
  kBuildKernelMetadataOmitVgprCount = 1u << 6,
  kBuildKernelMetadataZeroMaxFlatWorkgroupSize = 1u << 7,
};

static std::vector<uint8_t> BuildKernelMetadata(
    uint32_t flags = kBuildKernelMetadataNone,
    iree_string_view_t name = IREE_SV("vector_add"),
    iree_string_view_t symbol = IREE_SV("vector_add.kd")) {
  const bool out_of_range_arg =
      (flags & kBuildKernelMetadataOutOfRangeArg) != 0;
  const bool unknown_value_kind =
      (flags & kBuildKernelMetadataUnknownValueKind) != 0;
  const bool has_cluster_dimensions =
      (flags & kBuildKernelMetadataClusterDimensions) != 0;
  const bool uniform_workgroups =
      (flags & kBuildKernelMetadataUniformWorkgroups) != 0;
  const bool invalid_uniform_workgroups =
      (flags & kBuildKernelMetadataInvalidUniformWorkgroups) != 0;
  const bool omit_max_flat_workgroup_size =
      (flags & kBuildKernelMetadataOmitMaxFlatWorkgroupSize) != 0;
  const bool omit_vgpr_count = (flags & kBuildKernelMetadataOmitVgprCount) != 0;
  const bool zero_max_flat_workgroup_size =
      (flags & kBuildKernelMetadataZeroMaxFlatWorkgroupSize) != 0;
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 3);

  AppendMsgPackString(&output, IREE_SV("amdhsa.version"));
  AppendMsgPackArray(&output, 2);
  AppendMsgPackUint(&output, 1);
  AppendMsgPackUint(&output, 2);

  AppendStringField(&output, IREE_SV("amdhsa.target"),
                    has_cluster_dimensions
                        ? IREE_SV("amdgcn-amd-amdhsa--gfx1250")
                        : IREE_SV("amdgcn-amd-amdhsa--gfx1100"));

  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  AppendMsgPackMap(
      &output, 10 - (omit_max_flat_workgroup_size ? 1 : 0) -
                   (omit_vgpr_count ? 1 : 0) +
                   (has_cluster_dimensions ? 1 : 0) +
                   (uniform_workgroups || invalid_uniform_workgroups ? 1 : 0));
  AppendStringField(&output, IREE_SV(".name"), name);
  AppendStringField(&output, IREE_SV(".symbol"), symbol);
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 24);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 1024);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 64);
  if (!omit_max_flat_workgroup_size) {
    AppendUintField(&output, IREE_SV(".max_flat_workgroup_size"),
                    zero_max_flat_workgroup_size ? 0 : 256);
  }
  if (!omit_vgpr_count) {
    AppendUintField(&output, IREE_SV(".vgpr_count"), 40);
  }
  AppendMsgPackString(&output, IREE_SV(".reqd_workgroup_size"));
  AppendMsgPackArray(&output, 3);
  AppendMsgPackUint(&output, 16);
  AppendMsgPackUint(&output, 4);
  AppendMsgPackUint(&output, 1);

  if (uniform_workgroups || invalid_uniform_workgroups) {
    AppendUintField(&output, IREE_SV(".uniform_work_group_size"),
                    invalid_uniform_workgroups ? 2 : 1);
  }

  if (has_cluster_dimensions) {
    AppendMsgPackString(&output, IREE_SV(".cluster_dims"));
    AppendMsgPackArray(&output, 3);
    AppendMsgPackUint(&output, 1);
    AppendMsgPackUint(&output, 2);
    AppendMsgPackUint(&output, 1);
  }

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
  AppendUintField(&output, IREE_SV(".size"), 4);
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

enum class ClusterDimensionsMetadataVariant {
  kDuplicate,
  kWrongLength,
  kWrongType,
  kZero,
  kOutOfRange,
  kTrivial,
};

static void AppendClusterDimensionsField(
    std::vector<uint8_t>* output, ClusterDimensionsMetadataVariant variant) {
  AppendMsgPackString(output, IREE_SV(".cluster_dims"));
  if (variant == ClusterDimensionsMetadataVariant::kWrongType) {
    AppendMsgPackString(output, IREE_SV("1,2,1"));
    return;
  }
  const uint32_t dimension_count =
      variant == ClusterDimensionsMetadataVariant::kWrongLength ? 2 : 3;
  AppendMsgPackArray(output, dimension_count);
  AppendMsgPackUint(output, 1);
  if (variant == ClusterDimensionsMetadataVariant::kOutOfRange) {
    AppendMsgPackUint(output, 256);
  } else if (variant == ClusterDimensionsMetadataVariant::kZero) {
    AppendMsgPackUint(output, 0);
  } else {
    AppendMsgPackUint(
        output, variant == ClusterDimensionsMetadataVariant::kTrivial ? 1 : 2);
  }
  if (dimension_count == 3) {
    AppendMsgPackUint(output, 1);
  }
}

static std::vector<uint8_t> BuildMalformedClusterDimensionsMetadata(
    ClusterDimensionsMetadataVariant variant) {
  std::vector<uint8_t> output;
  AppendMsgPackMap(&output, 1);
  AppendMsgPackString(&output, IREE_SV("amdhsa.kernels"));
  AppendMsgPackArray(&output, 1);
  const uint32_t cluster_field_count =
      variant == ClusterDimensionsMetadataVariant::kDuplicate ? 2 : 1;
  AppendMsgPackMap(&output, 6 + cluster_field_count);
  AppendStringField(&output, IREE_SV(".symbol"), IREE_SV("cluster.kd"));
  AppendUintField(&output, IREE_SV(".kernarg_segment_size"), 0);
  AppendUintField(&output, IREE_SV(".kernarg_segment_align"), 8);
  AppendUintField(&output, IREE_SV(".group_segment_fixed_size"), 0);
  AppendUintField(&output, IREE_SV(".private_segment_fixed_size"), 0);
  AppendClusterDimensionsField(&output, variant);
  if (variant == ClusterDimensionsMetadataVariant::kDuplicate) {
    AppendClusterDimensionsField(&output, variant);
  }
  AppendMsgPackString(&output, IREE_SV(".args"));
  AppendMsgPackArray(&output, 0);
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

// Models both notes linked into one PT_NOTE and notes in separate segments.
static std::vector<uint8_t> BuildElfWithMetadataNotes(
    const std::vector<uint8_t>& first, const std::vector<uint8_t>& second,
    bool separate_segments) {
  constexpr size_t kProgramHeaderOffset = 64;
  constexpr size_t kProgramHeaderSize = 56;
  constexpr size_t kNoteOffset = 128;
  std::vector<uint8_t> elf = BuildElfWithMetadata(first);
  std::vector<uint8_t> second_elf = BuildElfWithMetadata(second);
  const size_t first_note_size = elf.size() - kNoteOffset;
  const size_t second_note_size = second_elf.size() - kNoteOffset;
  if (separate_segments) {
    // Make room for another program header before the first note.
    elf.insert(elf.begin() + kNoteOffset, kProgramHeaderSize, 0);
    StoreU16LE(&elf, 56, 2);
    StoreU64LE(&elf, kProgramHeaderOffset + 8,
               kNoteOffset + kProgramHeaderSize);
    const size_t second_header = kProgramHeaderOffset + kProgramHeaderSize;
    StoreU32LE(&elf, second_header, 4);  // PT_NOTE.
    StoreU64LE(&elf, second_header + 8, elf.size());
    StoreU64LE(&elf, second_header + 32, second_note_size);
    StoreU64LE(&elf, second_header + 40, second_note_size);
    StoreU64LE(&elf, second_header + 48, 4);
  } else {
    StoreU64LE(&elf, kProgramHeaderOffset + 32,
               first_note_size + second_note_size);
    StoreU64LE(&elf, kProgramHeaderOffset + 40,
               first_note_size + second_note_size);
  }
  elf.insert(elf.end(), second_elf.begin() + kNoteOffset, second_elf.end());
  return elf;
}

static void AlignVector(std::vector<uint8_t>* output, size_t alignment) {
  while ((output->size() % alignment) != 0) {
    output->push_back(0);
  }
}

static void AppendElf64Symbol(std::vector<uint8_t>* output,
                              uint32_t name_offset, uint8_t info,
                              uint16_t section_index = 1, uint64_t value = 0,
                              uint64_t byte_length = 0) {
  size_t symbol_offset = output->size();
  output->resize(symbol_offset + 24, 0);
  StoreU32LE(output, symbol_offset + 0, name_offset);
  (*output)[symbol_offset + 4] = info;
  StoreU16LE(output, symbol_offset + 6, section_index);
  StoreU64LE(output, symbol_offset + 8, value);
  StoreU64LE(output, symbol_offset + 16, byte_length);
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

enum SyntheticDataLayoutFlagBits : uint32_t {
  kSyntheticDataLayoutNone = 0u,
  kSyntheticDataLayoutOmitMarker = 1u << 0,
  kSyntheticDataLayoutInvalidMarkerLength = 1u << 1,
  kSyntheticDataLayoutMissingTrailingRedzone = 1u << 2,
  kSyntheticDataLayoutOverlappingObjects = 1u << 3,
  kSyntheticDataLayoutDuplicateMarker = 1u << 4,
};

static std::vector<uint8_t> AddSyntheticDataLayout(
    std::vector<uint8_t> elf, uint32_t flags = kSyntheticDataLayoutNone) {
  constexpr uint8_t kGlobalObject = 0x11;  // STB_GLOBAL | STT_OBJECT.
  constexpr size_t kSectionHeaderSize = 64;
  constexpr uint64_t kDataAddress = 0x2000;

  const size_t string_offset = elf.size();
  elf.push_back(0);
  const auto append_name = [&](const char* name, size_t length) {
    const uint32_t offset = static_cast<uint32_t>(elf.size() - string_offset);
    elf.insert(elf.end(), name, name + length);
    return offset;
  };
  const char kDescriptorName[] = "vector_add.kd";
  const uint32_t descriptor_name_offset =
      append_name(kDescriptorName, sizeof(kDescriptorName));
  // A data symbol may end in the conventional kernel-descriptor suffix; only
  // symbols actually referenced by kernel metadata are descriptors.
  const char kTableName[] = "lookup_table.kd";
  const uint32_t table_name_offset =
      append_name(kTableName, sizeof(kTableName));
  const char kMarkerName[] = IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME;
  const uint32_t marker_name_offset =
      append_name(kMarkerName, sizeof(kMarkerName));
  const size_t string_size = elf.size() - string_offset;

  AlignVector(&elf, 8);
  const size_t data_offset = elf.size();
  const size_t data_size =
      (flags & kSyntheticDataLayoutMissingTrailingRedzone) ? 0x75 : 0xA0;
  elf.resize(data_offset + data_size, 0);

  AlignVector(&elf, 8);
  const size_t symbol_offset = elf.size();
  AppendElf64Symbol(&elf, 0, 0, 0);
  AppendElf64Symbol(&elf, descriptor_name_offset, kGlobalObject,
                    /*section_index=*/2, /*value=*/kDataAddress,
                    /*byte_length=*/64);
  const uint64_t table_address =
      (flags & kSyntheticDataLayoutOverlappingObjects) ? 0x2048 : 0x2060;
  // Deliberately emit the table before the lower-addressed marker so the
  // decoded result proves canonical address ordering.
  AppendElf64Symbol(&elf, table_name_offset, kGlobalObject,
                    /*section_index=*/2, table_address,
                    /*byte_length=*/12);
  if (!(flags & kSyntheticDataLayoutOmitMarker)) {
    const uint64_t marker_length =
        (flags & kSyntheticDataLayoutInvalidMarkerLength) ? 2 : 1;
    AppendElf64Symbol(&elf, marker_name_offset, kGlobalObject,
                      /*section_index=*/2, /*value=*/0x2040, marker_length);
    if (flags & kSyntheticDataLayoutDuplicateMarker) {
      AppendElf64Symbol(&elf, marker_name_offset, kGlobalObject,
                        /*section_index=*/2, /*value=*/0x2080,
                        /*byte_length=*/1);
    }
  }
  const size_t symbol_size = elf.size() - symbol_offset;

  AlignVector(&elf, 8);
  const size_t section_offset = elf.size();
  StoreU64LE(&elf, 40, section_offset);
  StoreU16LE(&elf, 58, kSectionHeaderSize);
  StoreU16LE(&elf, 60, 4);

  elf.resize(section_offset + 4 * kSectionHeaderSize, 0);
  const size_t string_section = section_offset + kSectionHeaderSize;
  StoreU32LE(&elf, string_section + 4, 3);  // SHT_STRTAB.
  StoreU64LE(&elf, string_section + 24, string_offset);
  StoreU64LE(&elf, string_section + 32, string_size);
  StoreU64LE(&elf, string_section + 48, 1);

  const size_t data_section = section_offset + 2 * kSectionHeaderSize;
  StoreU32LE(&elf, data_section + 4, 1);  // SHT_PROGBITS.
  StoreU64LE(&elf, data_section + 8, 2);  // SHF_ALLOC.
  StoreU64LE(&elf, data_section + 16, kDataAddress);
  StoreU64LE(&elf, data_section + 24, data_offset);
  StoreU64LE(&elf, data_section + 32, data_size);
  StoreU64LE(&elf, data_section + 48, 8);

  const size_t symbol_section = section_offset + 3 * kSectionHeaderSize;
  StoreU32LE(&elf, symbol_section + 4, 11);  // SHT_DYNSYM.
  StoreU64LE(&elf, symbol_section + 24, symbol_offset);
  StoreU64LE(&elf, symbol_section + 32, symbol_size);
  StoreU32LE(&elf, symbol_section + 40, 1);  // sh_link: string table section.
  StoreU64LE(&elf, symbol_section + 48, 8);
  StoreU64LE(&elf, symbol_section + 56, 24);
  return elf;
}

static iree_const_byte_span_t ByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

static std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

TEST(HsacoMetadataTest, ParsesValidMetadata) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(
      BuildKernelMetadata(kBuildKernelMetadataUniformWorkgroups));

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
  EXPECT_EQ(kernel.max_flat_workgroup_size, 256);
  EXPECT_EQ(kernel.vgpr_count, 40);
  ASSERT_TRUE(kernel.has_required_workgroup_size);
  EXPECT_EQ(kernel.required_workgroup_size[0], 16);
  EXPECT_EQ(kernel.required_workgroup_size[1], 4);
  EXPECT_EQ(kernel.required_workgroup_size[2], 1);
  EXPECT_TRUE(kernel.uniform_workgroup_size);
  EXPECT_FALSE(kernel.has_workgroup_cluster_size);
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

TEST(HsacoMetadataTest, RejectsInvalidUniformWorkgroupValue) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(
      BuildKernelMetadata(kBuildKernelMetadataInvalidUniformWorkgroups));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(HsacoMetadataTest, MergesAllMetadataNotes) {
  for (bool separate_segments : {false, true}) {
    SCOPED_TRACE(separate_segments);
    std::vector<uint8_t> elf = BuildElfWithMetadataNotes(
        BuildKernelMetadata(),
        BuildKernelMetadata(kBuildKernelMetadataUniformWorkgroups,
                            IREE_SV("extra_kernel"),
                            IREE_SV("extra_kernel.kd")),
        separate_segments);

    iree_hal_amdgpu_hsaco_metadata_t metadata;
    IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
        ByteSpan(elf), iree_allocator_system(), &metadata));
    ASSERT_EQ(metadata.kernel_count, 2);
    ASSERT_EQ(metadata.arg_count, 8);
    EXPECT_EQ(metadata.reflection_name_storage_size, 22);
    EXPECT_EQ(metadata.arg_name_storage_size, 24);
    EXPECT_EQ(ToString(metadata.kernels[0].symbol_name), "vector_add.kd");
    EXPECT_EQ(ToString(metadata.kernels[1].symbol_name), "extra_kernel.kd");
    EXPECT_FALSE(metadata.kernels[0].uniform_workgroup_size);
    EXPECT_TRUE(metadata.kernels[1].uniform_workgroup_size);
    for (size_t i = 0; i < 2; ++i) {
      const auto& kernel = metadata.kernels[i];
      ASSERT_EQ(kernel.arg_count, 4);
      ASSERT_EQ(kernel.args, metadata.args + i * 4);
      EXPECT_EQ(kernel.kernarg_segment_size, 24);
      EXPECT_EQ(kernel.arg_name_storage_size, 12);
      EXPECT_EQ(ToString(kernel.args[0].name), "lhs");
      EXPECT_EQ(kernel.args[0].offset, 0);
      EXPECT_EQ(kernel.args[0].kind,
                IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_GLOBAL_BUFFER);
      EXPECT_EQ(ToString(kernel.args[1].name), "rhs");
      EXPECT_EQ(kernel.args[1].offset, 8);
      EXPECT_EQ(ToString(kernel.args[2].name), "n");
      EXPECT_EQ(kernel.args[2].offset, 16);
      EXPECT_EQ(ToString(kernel.args[3].name), "alpha");
      EXPECT_EQ(kernel.args[3].offset, 20);
      EXPECT_EQ(kernel.args[3].kind,
                IREE_HAL_AMDGPU_HSACO_METADATA_ARG_KIND_BY_VALUE);
    }
    const iree_hal_amdgpu_hsaco_metadata_kernel_t* kernel = nullptr;
    IREE_EXPECT_OK(iree_hal_amdgpu_hsaco_metadata_find_kernel_by_symbol(
        &metadata, IREE_SV("extra_kernel.kd"), &kernel));
    EXPECT_EQ(kernel, &metadata.kernels[1]);
    iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
  }
}

TEST(HsacoMetadataTest, LaterMetadataNotePreventsElfOnlyClassification) {
  std::vector<uint8_t> elf =
      AddSyntheticCandidateSymbolSection(BuildElfWithMetadataNotes(
          BuildKernelMetadata(),
          BuildKernelMetadata(kBuildKernelMetadataNone, IREE_SV("extra_kernel"),
                              IREE_SV("extra_kernel.kd")),
          /*separate_segments=*/false));
  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  EXPECT_EQ(metadata.kernel_count, 2);
  EXPECT_EQ(metadata.elf_kernel_symbol_count, 0);
  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, RejectsMalformedLaterMetadataNote) {
  for (bool separate_segments : {false, true}) {
    SCOPED_TRACE(separate_segments);
    std::vector<uint8_t> elf = BuildElfWithMetadataNotes(
        BuildKernelMetadata(), BuildMalformedMissingKernelFieldsMetadata(),
        separate_segments);
    iree_hal_amdgpu_hsaco_metadata_t metadata;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
            ByteSpan(elf), iree_allocator_system(), &metadata));
  }
}

TEST(HsacoMetadataTest, RejectsConflictingTargetsAcrossMetadataNotes) {
  std::vector<uint8_t> elf = BuildElfWithMetadataNotes(
      BuildKernelMetadata(),
      BuildKernelMetadata(kBuildKernelMetadataClusterDimensions,
                          IREE_SV("extra_kernel"), IREE_SV("extra_kernel.kd")),
      /*separate_segments=*/false);
  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
}

TEST(HsacoMetadataTest, RejectsZeroMaximumFlatWorkgroupSize) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(
      BuildKernelMetadata(kBuildKernelMetadataZeroMaxFlatWorkgroupSize));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
                            ByteSpan(elf), iree_allocator_system(), &metadata));
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

TEST(HsacoMetadataTest, ParsesWorkgroupClusterDimensions) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(
      BuildKernelMetadata(kBuildKernelMetadataClusterDimensions));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));

  ASSERT_EQ(metadata.kernel_count, 1);
  const iree_hal_amdgpu_hsaco_metadata_kernel_t& kernel = metadata.kernels[0];
  ASSERT_TRUE(kernel.has_workgroup_cluster_size);
  EXPECT_EQ(kernel.workgroup_cluster_size[0], 1);
  EXPECT_EQ(kernel.workgroup_cluster_size[1], 2);
  EXPECT_EQ(kernel.workgroup_cluster_size[2], 1);

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, RejectsMalformedWorkgroupClusterDimensions) {
  const ClusterDimensionsMetadataVariant variants[] = {
      ClusterDimensionsMetadataVariant::kDuplicate,
      ClusterDimensionsMetadataVariant::kWrongLength,
      ClusterDimensionsMetadataVariant::kWrongType,
      ClusterDimensionsMetadataVariant::kZero,
      ClusterDimensionsMetadataVariant::kOutOfRange,
      ClusterDimensionsMetadataVariant::kTrivial,
  };
  for (const ClusterDimensionsMetadataVariant variant : variants) {
    std::vector<uint8_t> elf =
        BuildElfWithMetadata(BuildMalformedClusterDimensionsMetadata(variant));
    iree_hal_amdgpu_hsaco_metadata_t metadata;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
            ByteSpan(elf), iree_allocator_system(), &metadata));
  }
}

TEST(HsacoMetadataTest, AllowsUnknownValueKindAsOpaqueMetadata) {
  std::vector<uint8_t> elf = BuildElfWithMetadata(
      BuildKernelMetadata(kBuildKernelMetadataUnknownValueKind));

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
  std::vector<uint8_t> elf = BuildElfWithMetadata(
      BuildKernelMetadata(kBuildKernelMetadataOutOfRangeArg));

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

TEST(HsacoMetadataTest, RejectsMissingResourceMetadata) {
  const uint32_t missing_resource_flags[] = {
      kBuildKernelMetadataOmitMaxFlatWorkgroupSize,
      kBuildKernelMetadataOmitVgprCount,
  };
  for (uint32_t flags : missing_resource_flags) {
    std::vector<uint8_t> elf = BuildElfWithMetadata(BuildKernelMetadata(flags));

    iree_hal_amdgpu_hsaco_metadata_t metadata;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
            ByteSpan(elf), iree_allocator_system(), &metadata));
  }
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

TEST(HsacoMetadataTest, LeavesUnmarkedDataLayoutOpaque) {
  std::vector<uint8_t> elf =
      AddSyntheticDataLayout(BuildElfWithMetadata(BuildKernelMetadata()),
                             kSyntheticDataLayoutOmitMarker);

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  EXPECT_EQ(metadata.data_object_count, 0);
  EXPECT_EQ(metadata.data_objects, nullptr);

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, ParsesMarkedAsanDataLayout) {
  std::vector<uint8_t> elf =
      AddSyntheticDataLayout(BuildElfWithMetadata(BuildKernelMetadata()));

  iree_hal_amdgpu_hsaco_metadata_t metadata;
  IREE_ASSERT_OK(iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
      ByteSpan(elf), iree_allocator_system(), &metadata));
  ASSERT_EQ(metadata.data_object_count, 2);
  ASSERT_NE(metadata.data_objects, nullptr);
  EXPECT_EQ(ToString(metadata.data_objects[0].name),
            IREE_HAL_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_MARKER_NAME);
  EXPECT_EQ(metadata.data_objects[0].virtual_address, 0x2040);
  EXPECT_EQ(metadata.data_objects[0].byte_length, 1);
  EXPECT_EQ(ToString(metadata.data_objects[1].name), "lookup_table.kd");
  EXPECT_EQ(metadata.data_objects[1].virtual_address, 0x2060);
  EXPECT_EQ(metadata.data_objects[1].byte_length, 12);

  iree_hal_amdgpu_hsaco_metadata_deinitialize(&metadata);
}

TEST(HsacoMetadataTest, RejectsInvalidMarkedAsanDataLayout) {
  const uint32_t invalid_layouts[] = {
      kSyntheticDataLayoutInvalidMarkerLength,
      kSyntheticDataLayoutMissingTrailingRedzone,
      kSyntheticDataLayoutOverlappingObjects,
      kSyntheticDataLayoutDuplicateMarker,
  };
  for (uint32_t flags : invalid_layouts) {
    SCOPED_TRACE(flags);
    std::vector<uint8_t> elf = AddSyntheticDataLayout(
        BuildElfWithMetadata(BuildKernelMetadata()), flags);
    iree_hal_amdgpu_hsaco_metadata_t metadata;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_hal_amdgpu_hsaco_metadata_initialize_from_elf(
            ByteSpan(elf), iree_allocator_system(), &metadata));
  }
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
