// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verification.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/status_cc.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/inspection.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/wire/core/control.h"
#include "iree/vm/bytecode/wire/core/opcodes.h"
#include "iree/vm/bytecode/wire/core/selectors.h"
#include "iree/vm/bytecode/wire/hal/opcodes.h"
#include "iree/vm/bytecode/wire/module_format.h"

namespace {

using iree::vm::bytecode::testing::BuildBufferModuleImage;
using iree::vm::bytecode::testing::BuildCallModuleImage;
using iree::vm::bytecode::testing::BuildFunctionStateModuleImage;
using iree::vm::bytecode::testing::BuildHALInspectionModuleImage;
using iree::vm::bytecode::testing::BuildOwnershipModuleImage;
using iree::vm::bytecode::testing::BuildRefStateModuleImage;
using iree::vm::bytecode::testing::BuildScalarStateModuleImage;
using iree::vm::bytecode::testing::BuildSwitchInspectionModuleImage;
using iree::vm::bytecode::testing::BuildValueOverflowModuleImage;
using iree::vm::bytecode::testing::FindFunctionImage;
using iree::vm::bytecode::testing::FindSectionPayload;
using iree::vm::bytecode::testing::MutableFunctionImage;

enum class ValidationObligation {
#define IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION(token, entity_id) token,
#include "iree/vm/bytecode/module_validation_obligations.inl"
#undef IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION
};

struct ValidationObligationCase {
  // Stable generated obligation identity.
  ValidationObligation obligation;
  // Complete specification entity ID used as the test trace.
  const char* entity_id;
};

constexpr ValidationObligationCase kValidationObligations[] = {
#define IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION(token, entity_id) \
  {ValidationObligation::token, entity_id},
#include "iree/vm/bytecode/module_validation_obligations.inl"
#undef IREE_VM_BYTECODE_MODULE_VALIDATION_OBLIGATION
};

struct SectionData {
  // Architectural section identifier.
  uint16_t type;
  // Exact section flags.
  uint16_t flags;
  // Complete section payload.
  std::vector<uint8_t> payload;
};

template <typename T>
void AppendValue(const T& value, std::vector<uint8_t>* data) {
  const size_t offset = data->size();
  data->resize(offset + sizeof(value));
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void AlignData(size_t alignment, std::vector<uint8_t>* data) {
  data->resize((data->size() + alignment - 1) & ~(alignment - 1), 0);
}

std::vector<SectionData> ExtractSections(const std::vector<uint8_t>& image) {
  if (image.size() < sizeof(iree_vm_bytecode_v0_image_header_t)) {
    ADD_FAILURE() << "test fixture has no image header";
    return {};
  }
  const auto* header =
      reinterpret_cast<const iree_vm_bytecode_v0_image_header_t*>(image.data());
  const size_t directory_length =
      header->section_count_u16 *
      sizeof(iree_vm_bytecode_v0_section_directory_row_t);
  if (directory_length >
      image.size() - sizeof(iree_vm_bytecode_v0_image_header_t)) {
    ADD_FAILURE() << "test fixture has a truncated section directory";
    return {};
  }
  const auto* rows =
      reinterpret_cast<const iree_vm_bytecode_v0_section_directory_row_t*>(
          header + 1);
  size_t offset = sizeof(*header) + directory_length;
  std::vector<SectionData> sections;
  sections.reserve(header->section_count_u16);
  for (uint16_t i = 0; i < header->section_count_u16; ++i) {
    offset = (offset + IREE_VM_BYTECODE_SECTION_ALIGNMENT - 1) &
             ~(IREE_VM_BYTECODE_SECTION_ALIGNMENT - 1);
    if (rows[i].byte_length_u64 > image.size() - offset) {
      ADD_FAILURE() << "test fixture has a truncated section payload";
      return {};
    }
    const size_t length = static_cast<size_t>(rows[i].byte_length_u64);
    sections.push_back(SectionData{
        rows[i].section_type_u16,
        rows[i].section_flags_u16,
        std::vector<uint8_t>(image.begin() + offset,
                             image.begin() + offset + length),
    });
    offset += length;
  }
  if (offset != image.size()) {
    ADD_FAILURE() << "test fixture has trailing bytes";
    return {};
  }
  return sections;
}

std::vector<uint8_t> BuildImage(const std::vector<SectionData>& sections) {
  std::vector<uint8_t> image(
      sizeof(iree_vm_bytecode_v0_image_header_t) +
          sections.size() * sizeof(iree_vm_bytecode_v0_section_directory_row_t),
      0);
  auto* header =
      reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image.data());
  std::memcpy(header->magic_u8, IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_BYTES,
              IREE_VM_BYTECODE_IMAGE_HEADER_MAGIC_U8_LENGTH);
  header->core_major_u16 = IREE_VM_BYTECODE_CORE_MAJOR;
  header->core_required_minor_u16 = IREE_VM_BYTECODE_CORE_MINOR;
  header->section_count_u16 = static_cast<uint16_t>(sections.size());
  auto* rows = reinterpret_cast<iree_vm_bytecode_v0_section_directory_row_t*>(
      header + 1);
  for (size_t i = 0; i < sections.size(); ++i) {
    rows[i].section_type_u16 = sections[i].type;
    rows[i].section_flags_u16 = sections[i].flags;
    rows[i].byte_length_u64 = sections[i].payload.size();
  }
  for (const SectionData& section : sections) {
    AlignData(IREE_VM_BYTECODE_SECTION_ALIGNMENT, &image);
    image.insert(image.end(), section.payload.begin(), section.payload.end());
  }
  return image;
}

std::vector<uint8_t> ReplaceSection(std::vector<uint8_t> image,
                                    uint16_t section_type,
                                    std::vector<uint8_t> replacement) {
  std::vector<SectionData> sections = ExtractSections(image);
  auto it = std::find_if(
      sections.begin(), sections.end(),
      [&](const SectionData& section) { return section.type == section_type; });
  if (it == sections.end()) {
    ADD_FAILURE() << "test fixture is missing section " << section_type;
    return {};
  }
  it->payload = std::move(replacement);
  return BuildImage(sections);
}

std::vector<uint8_t> RemoveSection(std::vector<uint8_t> image,
                                   uint16_t section_type) {
  std::vector<SectionData> sections = ExtractSections(image);
  const size_t original_size = sections.size();
  sections.erase(std::remove_if(sections.begin(), sections.end(),
                                [&](const SectionData& section) {
                                  return section.type == section_type;
                                }),
                 sections.end());
  if (sections.size() + 1 != original_size) {
    ADD_FAILURE() << "test fixture has an unexpected section multiplicity";
    return {};
  }
  return BuildImage(sections);
}

template <typename T>
T* SectionRecord(std::vector<uint8_t>* image, uint16_t section_type,
                 size_t byte_offset = 0) {
  uint8_t* payload = FindSectionPayload(image, section_type);
  if (!payload) {
    ADD_FAILURE() << "test fixture is missing section " << section_type;
    return nullptr;
  }
  return reinterpret_cast<T*>(payload + byte_offset);
}

iree_vm_bytecode_v0_section_directory_row_t* DirectoryRows(
    std::vector<uint8_t>* image) {
  return reinterpret_cast<iree_vm_bytecode_v0_section_directory_row_t*>(
      image->data() + sizeof(iree_vm_bytecode_v0_image_header_t));
}

void ExpectStatus(iree_status_t status_value, iree_status_code_t expected_code,
                  const char* expected_message) {
  iree::Status status(std::move(status_value));
  EXPECT_FALSE(status.ok());
  if (status.ok()) return;
  EXPECT_EQ(static_cast<iree_status_code_t>(status.code()), expected_code);
  EXPECT_NE(status.ToString().find(expected_message), std::string::npos)
      << status.ToString();
}

void ExpectStructureRejected(const std::vector<uint8_t>& image,
                             const char* expected_message) {
  iree_vm_bytecode_module_plan_t plan = {};
  ExpectStatus(
      iree_vm_bytecode_module_verify_structure(
          iree_make_const_byte_span(image.data(), image.size()), &plan),
      IREE_STATUS_INVALID_ARGUMENT, expected_message);
  const iree_vm_bytecode_module_plan_t empty_plan = {};
  EXPECT_EQ(std::memcmp(&plan, &empty_plan, sizeof(plan)), 0);
}

void ExpectExecutableRejected(const std::vector<uint8_t>& image,
                              iree_status_code_t expected_code,
                              const char* expected_message) {
  iree_vm_environment_t* environment = nullptr;
  IREE_ASSERT_OK(
      iree_vm_environment_allocate(iree_allocator_system(), &environment));
  iree_vm_module_t* const sentinel =
      reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  iree_vm_module_t* module = sentinel;
  ExpectStatus(iree_vm_bytecode_module_create(
                   environment, IREE_SV("malformed"),
                   {iree_make_const_byte_span(image.data(), image.size()),
                    iree_allocator_null()},
                   iree_allocator_system(), &module),
               expected_code, expected_message);
  EXPECT_EQ(module, nullptr);
  if (module != nullptr && module != sentinel) iree_vm_module_release(module);
  iree_vm_environment_free(environment);
}

void ExpectInspectionRejected(const std::vector<uint8_t>& image,
                              iree_status_code_t expected_code,
                              const char* expected_message) {
  iree_vm_module_t* const sentinel =
      reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
  iree_vm_module_t* module = sentinel;
  ExpectStatus(iree_vm_bytecode_module_create_for_inspection(
                   IREE_SV("malformed"),
                   {iree_make_const_byte_span(image.data(), image.size()),
                    iree_allocator_null()},
                   iree_allocator_system(), &module),
               expected_code, expected_message);
  EXPECT_EQ(module, nullptr);
  if (module != nullptr && module != sentinel) iree_vm_module_release(module);
}

void ExpectInspectionAccepted(const std::vector<uint8_t>& image) {
  iree_vm_module_t* module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create_for_inspection(
      IREE_SV("inspectable"),
      {iree_make_const_byte_span(image.data(), image.size()),
       iree_allocator_null()},
      iree_allocator_system(), &module));
  ASSERT_NE(module, nullptr);
  iree_vm_module_release(module);
}

std::vector<uint8_t> BuildDuplicateRefTypeGroups() {
  std::vector<uint8_t> payload;
  AppendValue(iree_vm_bytecode_v0_ref_types_header_t{2}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_group_row_t{0, 0, 1}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_group_row_t{0, 0, 1}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0}, &payload);
  return ReplaceSection(BuildOwnershipModuleImage(),
                        IREE_VM_BYTECODE_SECTION_REF_TYPES, std::move(payload));
}

std::vector<uint8_t> BuildDuplicateRefTypeEntries() {
  std::vector<uint8_t> payload;
  AppendValue(iree_vm_bytecode_v0_ref_types_header_t{1}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_group_row_t{0, 0, 2}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0}, &payload);
  AppendValue(iree_vm_bytecode_v0_ref_type_entry_row_t{1, 0}, &payload);
  return ReplaceSection(BuildOwnershipModuleImage(),
                        IREE_VM_BYTECODE_SECTION_REF_TYPES, std::move(payload));
}

std::vector<uint8_t> BuildDuplicateImportGroups() {
  std::vector<uint8_t> payload;
  AppendValue(iree_vm_bytecode_v0_imports_header_t{2}, &payload);
  AppendValue(iree_vm_bytecode_v0_import_group_row_t{0, 0, 1}, &payload);
  AppendValue(iree_vm_bytecode_v0_import_group_row_t{0, 0, 1}, &payload);
  AppendValue(
      iree_vm_bytecode_v0_import_entry_row_t{
          1, 0, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL, 0},
      &payload);
  AppendValue(
      iree_vm_bytecode_v0_import_entry_row_t{
          1, 0, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL, 0},
      &payload);
  return ReplaceSection(iree::vm::bytecode::testing::BuildFunctionModuleImage(),
                        IREE_VM_BYTECODE_SECTION_IMPORTS, std::move(payload));
}

std::vector<uint8_t> BuildDuplicateImportEntries() {
  std::vector<uint8_t> payload;
  AppendValue(iree_vm_bytecode_v0_imports_header_t{1}, &payload);
  AppendValue(iree_vm_bytecode_v0_import_group_row_t{0, 0, 2}, &payload);
  AppendValue(
      iree_vm_bytecode_v0_import_entry_row_t{
          1, 0, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL, 0},
      &payload);
  AppendValue(
      iree_vm_bytecode_v0_import_entry_row_t{
          1, 0, IREE_VM_BYTECODE_IMPORT_FLAG_OPTIONAL, 0},
      &payload);
  return ReplaceSection(iree::vm::bytecode::testing::BuildFunctionModuleImage(),
                        IREE_VM_BYTECODE_SECTION_IMPORTS, std::move(payload));
}

std::vector<uint8_t> BuildDuplicateMetadataEntries() {
  std::vector<uint8_t> payload;
  AppendValue(iree_vm_bytecode_v0_metadata_header_t{2, 0, 0, 2}, &payload);
  AppendValue(
      iree_vm_bytecode_v0_metadata_entry_row_t{
          11, IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES},
      &payload);
  AppendValue(
      iree_vm_bytecode_v0_metadata_entry_row_t{
          11, IREE_VM_BYTECODE_METADATA_VALUE_TYPE_BYTES},
      &payload);
  AlignData(iree_alignof(iree_vm_bytecode_v0_metadata_value_offset_t),
            &payload);
  AppendValue(iree_vm_bytecode_v0_metadata_value_offset_t{0}, &payload);
  AppendValue(iree_vm_bytecode_v0_metadata_value_offset_t{1}, &payload);
  AppendValue(iree_vm_bytecode_v0_metadata_value_offset_t{2}, &payload);
  payload.push_back('a');
  payload.push_back('b');
  return ReplaceSection(BuildOwnershipModuleImage(),
                        IREE_VM_BYTECODE_SECTION_METADATA, std::move(payload));
}

void RunValidationObligation(ValidationObligation obligation) {
  switch (obligation) {
    case ValidationObligation::IMAGE_STORAGE: {
      const std::vector<uint8_t> image = BuildOwnershipModuleImage();
      std::vector<uint8_t> misaligned_storage(image.size() + 1);
      std::memcpy(misaligned_storage.data() + 1, image.data(), image.size());
      iree_vm_environment_t* environment = nullptr;
      IREE_ASSERT_OK(
          iree_vm_environment_allocate(iree_allocator_system(), &environment));
      iree_vm_module_t* module =
          reinterpret_cast<iree_vm_module_t*>(uintptr_t{1});
      ExpectStatus(iree_vm_bytecode_module_create(
                       environment, IREE_SV("misaligned"),
                       {iree_make_const_byte_span(misaligned_storage.data() + 1,
                                                  image.size()),
                        iree_allocator_null()},
                       iree_allocator_system(), &module),
                   IREE_STATUS_INVALID_ARGUMENT, "eight-byte aligned");
      EXPECT_EQ(module, nullptr);
      iree_vm_environment_free(environment);
      break;
    }
    case ValidationObligation::IMAGE_HEADER: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header =
          reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image.data());
      header->zero_padding_u16 = 1;
      ExpectStructureRejected(image, "header padding");
      break;
    }
    case ValidationObligation::IMAGE_DIRECTORY_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header =
          reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image.data());
      ++header->section_count_u16;
      ExpectStructureRejected(image, "nonzero alignment padding");
      break;
    }
    case ValidationObligation::DIRECTORY_ORDER: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* rows = DirectoryRows(&image);
      rows[1].section_type_u16 = rows[0].section_type_u16;
      ExpectStructureRejected(image, "strictly increasing");
      break;
    }
    case ValidationObligation::DIRECTORY_ROWS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      DirectoryRows(&image)[0].reserved_u32 = 1;
      ExpectStructureRejected(image, "reserved bits");
      break;
    }
    case ValidationObligation::DIRECTORY_KNOWN_SECTIONS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      DirectoryRows(&image)[0].section_flags_u16 =
          IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE;
      ExpectExecutableRejected(image, IREE_STATUS_INCOMPATIBLE,
                               "unsupported flags");
      break;
    }
    case ValidationObligation::DIRECTORY_UNKNOWN_SECTIONS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header =
          reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image.data());
      auto* rows = DirectoryRows(&image);
      rows[header->section_count_u16 - 1].section_type_u16 = 0x000E;
      rows[header->section_count_u16 - 1].section_flags_u16 = 0;
      ExpectExecutableRejected(image, IREE_STATUS_INCOMPATIBLE,
                               "unsupported flags");
      rows[header->section_count_u16 - 1].section_flags_u16 =
          IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE;
      iree_vm_environment_t* environment = nullptr;
      IREE_ASSERT_OK(
          iree_vm_environment_allocate(iree_allocator_system(), &environment));
      iree_vm_module_t* module = nullptr;
      IREE_ASSERT_OK(iree_vm_bytecode_module_create(
          environment, IREE_SV("unknown_section"),
          {iree_make_const_byte_span(image.data(), image.size()),
           iree_allocator_null()},
          iree_allocator_system(), &module));
      iree_vm_module_release(module);
      iree_vm_environment_free(environment);
      break;
    }
    case ValidationObligation::DIRECTORY_PAYLOAD_PACKING: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      image.push_back(0);
      ExpectStructureRejected(image, "trailing bytes");
      break;
    }
    case ValidationObligation::REQUIREMENTS_EXTENT: {
      std::vector<uint8_t> image = ReplaceSection(
          BuildHALInspectionModuleImage(),
          IREE_VM_BYTECODE_SECTION_REQUIREMENTS, std::vector<uint8_t>(5));
      ExpectStructureRejected(image, "Requirements rows");
      break;
    }
    case ValidationObligation::REQUIREMENTS_ORDER: {
      std::vector<uint8_t> payload;
      const iree_vm_bytecode_v0_requirement_row_t row = {
          IREE_VM_ISA_PAGE_HAL, IREE_VM_ISA_HAL_MAJOR, IREE_VM_ISA_HAL_MINOR};
      AppendValue(row, &payload);
      AppendValue(row, &payload);
      std::vector<uint8_t> image = ReplaceSection(
          BuildHALInspectionModuleImage(),
          IREE_VM_BYTECODE_SECTION_REQUIREMENTS, std::move(payload));
      ExpectStructureRejected(image, "unique sorted extension pages");
      break;
    }
    case ValidationObligation::REQUIREMENTS_CAPABILITIES: {
      std::vector<uint8_t> image = BuildHALInspectionModuleImage();
      auto* row = SectionRecord<iree_vm_bytecode_v0_requirement_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_REQUIREMENTS);
      ASSERT_NE(row, nullptr);
      ++row->major_u16;
      ExpectInspectionRejected(image, IREE_STATUS_INCOMPATIBLE, "HAL version");
      break;
    }
    case ValidationObligation::REQUIREMENTS_FEATURE_COVERAGE: {
      // Core and HAL minor zero have no older in-major feature boundary to
      // under-declare. A real HAL record plus its exact declaration is the
      // complete applicable witness until the first additive minor exists.
      ExpectInspectionAccepted(BuildHALInspectionModuleImage());
      break;
    }
    case ValidationObligation::STRINGS_INDEX_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_strings_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_STRINGS);
      ASSERT_NE(header, nullptr);
      header->string_count_u32 = UINT32_MAX;
      ExpectStructureRejected(image, "Strings count");
      break;
    }
    case ValidationObligation::STRINGS_OFFSETS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* offsets = SectionRecord<iree_vm_bytecode_v0_string_offset_t>(
          &image, IREE_VM_BYTECODE_SECTION_STRINGS,
          sizeof(iree_vm_bytecode_v0_strings_header_t));
      ASSERT_NE(offsets, nullptr);
      offsets[0] = 1;
      ExpectStructureRejected(image, "exact byte tail");
      break;
    }
    case ValidationObligation::STRINGS_TEXT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_strings_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_STRINGS);
      ASSERT_NE(header, nullptr);
      auto* data = reinterpret_cast<uint8_t*>(header + 1) +
                   (header->string_count_u32 + 1) *
                       sizeof(iree_vm_bytecode_v0_string_offset_t);
      data[0] = 0;
      ExpectStructureRejected(image, "NUL-free valid UTF-8");
      break;
    }
    case ValidationObligation::REF_TYPES_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_ref_types_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_REF_TYPES);
      ASSERT_NE(header, nullptr);
      ++header->group_count_u32;
      ExpectStructureRejected(image, "RefTypes section is truncated");
      break;
    }
    case ValidationObligation::REF_TYPES_COUNTS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* group = SectionRecord<iree_vm_bytecode_v0_ref_type_group_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_REF_TYPES,
          sizeof(iree_vm_bytecode_v0_ref_types_header_t));
      ASSERT_NE(group, nullptr);
      group->entry_count_u32 = 0;
      ExpectStructureRejected(image, "group counts");
      break;
    }
    case ValidationObligation::REF_TYPES_GROUP_ORDER:
      ExpectStructureRejected(BuildDuplicateRefTypeGroups(),
                              "namespaces must be strictly");
      break;
    case ValidationObligation::REF_TYPES_ENTRY_ORDER:
      ExpectStructureRejected(BuildDuplicateRefTypeEntries(),
                              "names must be strictly");
      break;
    case ValidationObligation::SIGNATURES_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_signatures_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_SIGNATURES);
      ASSERT_NE(header, nullptr);
      ++header->signature_count_u32;
      ExpectStructureRejected(image, "Signatures section is truncated");
      break;
    }
    case ValidationObligation::SIGNATURES_DESCRIPTORS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_signatures_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_SIGNATURES);
      ASSERT_NE(header, nullptr);
      auto* rows =
          reinterpret_cast<iree_vm_bytecode_v0_signature_row_t*>(header + 1);
      auto* descriptors =
          reinterpret_cast<iree_vm_bytecode_v0_signature_descriptor_row_t*>(
              rows + header->signature_count_u32);
      descriptors[0].type_ordinal_u16 = 1;
      ExpectStructureRejected(image, "scalar signature descriptor");
      break;
    }
    case ValidationObligation::SIGNATURES_PHYSICAL_LAYOUT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_signatures_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_SIGNATURES);
      ASSERT_NE(header, nullptr);
      auto* rows =
          reinterpret_cast<iree_vm_bytecode_v0_signature_row_t*>(header + 1);
      rows[1].descriptor_base_u32 = 1;
      ExpectStructureRejected(image, "descriptor prefix");
      break;
    }
    case ValidationObligation::CALLABLE_TYPES_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_callable_types_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES);
      ASSERT_NE(header, nullptr);
      ++header->callable_type_count_u32;
      ExpectStructureRejected(image, "truncated");
      break;
    }
    case ValidationObligation::CALLABLE_TYPES_ROWS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* row = SectionRecord<iree_vm_bytecode_v0_callable_type_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES,
          sizeof(iree_vm_bytecode_v0_callable_types_header_t));
      ASSERT_NE(row, nullptr);
      row->flags_u16 = UINT16_MAX;
      ExpectStructureRejected(image, "callable type row");
      break;
    }
    case ValidationObligation::CALLABLE_TYPES_TOPOLOGY: {
      std::vector<uint8_t> image = BuildFunctionStateModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_signatures_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_SIGNATURES);
      ASSERT_NE(header, nullptr);
      auto* rows =
          reinterpret_cast<iree_vm_bytecode_v0_signature_row_t*>(header + 1);
      auto* descriptors =
          reinterpret_cast<iree_vm_bytecode_v0_signature_descriptor_row_t*>(
              rows + header->signature_count_u32);
      descriptors[43].type_ordinal_u16 = 2;
      ExpectStructureRejected(image, "topologically ordered");
      break;
    }
    case ValidationObligation::CALLABLE_TYPES_DEPTH: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* row = SectionRecord<iree_vm_bytecode_v0_callable_type_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES,
          sizeof(iree_vm_bytecode_v0_callable_types_header_t));
      ASSERT_NE(row, nullptr);
      row->nesting_depth_u16 = 1;
      ExpectStructureRejected(image, "nesting depth is not canonical");
      break;
    }
    case ValidationObligation::CALLABLE_TYPES_ORDER: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_callable_types_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_CALLABLE_TYPES);
      ASSERT_NE(header, nullptr);
      auto* rows = reinterpret_cast<iree_vm_bytecode_v0_callable_type_row_t*>(
          header + 1);
      rows[1] = rows[0];
      ExpectStructureRejected(image, "unique and strictly ordered");
      break;
    }
    case ValidationObligation::IMPORTS_EXTENT: {
      std::vector<uint8_t> image =
          iree::vm::bytecode::testing::BuildFunctionModuleImage();
      std::vector<SectionData> sections = ExtractSections(image);
      auto it = std::find_if(
          sections.begin(), sections.end(), [](const SectionData& section) {
            return section.type == IREE_VM_BYTECODE_SECTION_IMPORTS;
          });
      ASSERT_NE(it, sections.end());
      it->payload.pop_back();
      ExpectStructureRejected(BuildImage(sections), "truncated");
      break;
    }
    case ValidationObligation::IMPORTS_COUNTS: {
      std::vector<uint8_t> image =
          iree::vm::bytecode::testing::BuildFunctionModuleImage();
      auto* group = SectionRecord<iree_vm_bytecode_v0_import_group_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_IMPORTS,
          sizeof(iree_vm_bytecode_v0_imports_header_t));
      ASSERT_NE(group, nullptr);
      group->entry_count_u32 = 0;
      ExpectStructureRejected(image, "group counts");
      break;
    }
    case ValidationObligation::IMPORTS_GROUP_ORDER:
      ExpectStructureRejected(BuildDuplicateImportGroups(),
                              "groups must be strictly");
      break;
    case ValidationObligation::IMPORTS_ENTRY_ORDER:
      ExpectStructureRejected(BuildDuplicateImportEntries(),
                              "entries must be strictly");
      break;
    case ValidationObligation::EXPORTS_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      std::vector<SectionData> sections = ExtractSections(image);
      auto it = std::find_if(
          sections.begin(), sections.end(), [](const SectionData& section) {
            return section.type == IREE_VM_BYTECODE_SECTION_EXPORTS;
          });
      ASSERT_NE(it, sections.end());
      it->payload.pop_back();
      ExpectStructureRejected(BuildImage(sections), "truncated");
      break;
    }
    case ValidationObligation::EXPORTS_ORDER: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_exports_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_EXPORTS);
      ASSERT_NE(header, nullptr);
      auto* rows =
          reinterpret_cast<iree_vm_bytecode_v0_export_row_t*>(header + 1);
      rows[1].name_string_u16 = rows[0].name_string_u16;
      ExpectStructureRejected(image, "strictly byte-sorted");
      break;
    }
    case ValidationObligation::FUNCTIONS_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_functions_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_FUNCTIONS);
      ASSERT_NE(header, nullptr);
      ++header->function_count_u32;
      ExpectStructureRejected(image, "Functions section is truncated");
      break;
    }
    case ValidationObligation::FUNCTIONS_ROWS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 0);
      ASSERT_NE(function.row, nullptr);
      function.row->reserved_u32[0] = 1;
      ExpectStructureRejected(image, "function row");
      break;
    }
    case ValidationObligation::FUNCTIONS_SIGNATURE_PREFIXES: {
      std::vector<uint8_t> image = BuildCallModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 1);
      ASSERT_NE(function.row, nullptr);
      function.row->value_register_count_u16 = 0;
      ExpectStructureRejected(image, "direct signature prefixes");
      break;
    }
    case ValidationObligation::FUNCTIONS_CALL_PACKET: {
      std::vector<uint8_t> image = BuildValueOverflowModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 0);
      ASSERT_NE(function.row, nullptr);
      const iree_vm_isa_control_call_record_t call = {
          IREE_VM_ISA_CORE_OPCODE_CONTROL_CALL,
          IREE_VM_ISA_CONTROL_CALL_TARGET_LOCAL, 1, 0, 0};
      std::memcpy(function.bytecode + 4, &call, sizeof(call));
      function.row->flags_u16 |= IREE_VM_BYTECODE_FUNCTION_FLAG_HAS_CALL;
      ExpectExecutableRejected(image, IREE_STATUS_INVALID_ARGUMENT,
                               "overflow packet");
      break;
    }
    case ValidationObligation::FUNCTIONS_DECODE: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 0);
      ASSERT_NE(function.bytecode, nullptr);
      function.bytecode[4] = 0x0F;
      ExpectExecutableRejected(image, IREE_STATUS_UNIMPLEMENTED, "opcode 0x0f");
      break;
    }
    case ValidationObligation::FUNCTIONS_BLOCK_COUNT: {
      std::vector<uint8_t> image = BuildSwitchInspectionModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 0);
      ASSERT_NE(function.row, nullptr);
      ++function.row->block_count_u32;
      ExpectExecutableRejected(image, IREE_STATUS_INVALID_ARGUMENT,
                               "block count");
      break;
    }
    case ValidationObligation::FUNCTIONS_ENTRY_BLOCK: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 0);
      ASSERT_NE(function.bytecode, nullptr);
      function.bytecode[0] = IREE_VM_ISA_CORE_OPCODE_CONTROL_RETURN;
      ExpectExecutableRejected(image, IREE_STATUS_INVALID_ARGUMENT,
                               "does not begin with control.block");
      break;
    }
    case ValidationObligation::FUNCTIONS_TARGETS: {
      std::vector<uint8_t> image = BuildSwitchInspectionModuleImage();
      MutableFunctionImage function = FindFunctionImage(&image, 0);
      ASSERT_NE(function.row, nullptr);
      auto* targets =
          reinterpret_cast<iree_vm_bytecode_v0_switch_target_entry_t*>(
              function.row + 1);
      targets[0] = 2;
      ExpectExecutableRejected(image, IREE_STATUS_INVALID_ARGUMENT,
                               "control.block");
      break;
    }
    case ValidationObligation::CONSTANTS_EXTENT: {
      std::vector<uint8_t> image = BuildScalarStateModuleImage();
      std::vector<SectionData> sections = ExtractSections(image);
      auto it = std::find_if(
          sections.begin(), sections.end(), [](const SectionData& section) {
            return section.type == IREE_VM_BYTECODE_SECTION_CONSTANTS;
          });
      ASSERT_NE(it, sections.end());
      it->payload.pop_back();
      ExpectStructureRejected(BuildImage(sections), "Constants cells");
      break;
    }
    case ValidationObligation::GLOBALS_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_globals_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_GLOBALS);
      ASSERT_NE(header, nullptr);
      header->ref_count_u32 = 1;
      ExpectStructureRejected(image, "truncated");
      break;
    }
    case ValidationObligation::GLOBALS_COUNTS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_globals_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_GLOBALS);
      ASSERT_NE(header, nullptr);
      header->immutable_value_count_u32 = header->value_count_u32 + 1;
      ExpectStructureRejected(image, "Globals counts");
      break;
    }
    case ValidationObligation::GLOBALS_REFS: {
      std::vector<uint8_t> image = BuildRefStateModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_globals_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_GLOBALS);
      ASSERT_NE(header, nullptr);
      auto* refs =
          reinterpret_cast<iree_vm_bytecode_v0_global_ref_descriptor_row_t*>(
              header + 1);
      refs[0].ref_type_ordinal_u16 = 2;
      ExpectStructureRejected(image, "ref-global descriptor");
      break;
    }
    case ValidationObligation::GLOBALS_FUNCTIONS: {
      std::vector<uint8_t> image = BuildFunctionStateModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_globals_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_GLOBALS);
      ASSERT_NE(header, nullptr);
      auto* functions = reinterpret_cast<
          iree_vm_bytecode_v0_global_function_descriptor_row_t*>(header + 1);
      functions[0].callable_type_ordinal_u16 = 7;
      ExpectStructureRejected(image, "function-global descriptor");
      break;
    }
    case ValidationObligation::RODATA_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_rodata_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_RODATA);
      ASSERT_NE(header, nullptr);
      header->block_count_u32 = UINT32_MAX;
      ExpectStructureRejected(image, "Rodata header");
      break;
    }
    case ValidationObligation::RODATA_BLOCKS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* length = SectionRecord<iree_vm_bytecode_v0_rodata_block_length_t>(
          &image, IREE_VM_BYTECODE_SECTION_RODATA,
          sizeof(iree_vm_bytecode_v0_rodata_header_t));
      ASSERT_NE(length, nullptr);
      *length = UINT64_MAX;
      ExpectStructureRejected(image, "Rodata section is truncated");
      break;
    }
    case ValidationObligation::PRESENTATION_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      std::vector<SectionData> sections = ExtractSections(image);
      auto it = std::find_if(
          sections.begin(), sections.end(), [](const SectionData& section) {
            return section.type == IREE_VM_BYTECODE_SECTION_PRESENTATION;
          });
      ASSERT_NE(it, sections.end());
      it->payload.pop_back();
      ExpectStructureRejected(BuildImage(sections), "truncated");
      break;
    }
    case ValidationObligation::PRESENTATION_ORDER: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* entry = SectionRecord<iree_vm_bytecode_v0_presentation_entry_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_PRESENTATION,
          sizeof(iree_vm_bytecode_v0_presentation_header_t));
      ASSERT_NE(entry, nullptr);
      entry->declaration_ordinal_u16 = UINT16_MAX;
      ExpectStructureRejected(image, "strictly ordered");
      break;
    }
    case ValidationObligation::PRESENTATION_NONEMPTY: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_presentation_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_PRESENTATION);
      ASSERT_NE(header, nullptr);
      auto* entry =
          reinterpret_cast<iree_vm_bytecode_v0_presentation_entry_row_t*>(
              header + 1);
      auto* fields =
          reinterpret_cast<iree_vm_bytecode_v0_presentation_field_row_t*>(
              entry + header->entry_count_u32);
      entry->documentation_string_u16 = UINT16_MAX;
      entry->authored_type_string_u16 = UINT16_MAX;
      for (int i = 0; i < 3; ++i) {
        fields[i].name_string_u16 = UINT16_MAX;
        fields[i].authored_type_string_u16 = UINT16_MAX;
      }
      ExpectStructureRejected(image, "no authored information");
      break;
    }
    case ValidationObligation::METADATA_EXTENT: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      std::vector<SectionData> sections = ExtractSections(image);
      auto it = std::find_if(
          sections.begin(), sections.end(), [](const SectionData& section) {
            return section.type == IREE_VM_BYTECODE_SECTION_METADATA;
          });
      ASSERT_NE(it, sections.end());
      it->payload.pop_back();
      ExpectStructureRejected(BuildImage(sections), "value offsets");
      break;
    }
    case ValidationObligation::METADATA_SCOPES: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* scope = SectionRecord<iree_vm_bytecode_v0_metadata_scope_row_t>(
          &image, IREE_VM_BYTECODE_SECTION_METADATA,
          sizeof(iree_vm_bytecode_v0_metadata_header_t));
      ASSERT_NE(scope, nullptr);
      scope->entry_base_u32 = 0;
      ExpectStructureRejected(image, "scopes are invalid");
      break;
    }
    case ValidationObligation::METADATA_ENTRIES:
      ExpectStructureRejected(BuildDuplicateMetadataEntries(),
                              "keys must be strictly sorted");
      break;
    case ValidationObligation::METADATA_VALUES: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header = SectionRecord<iree_vm_bytecode_v0_metadata_header_t>(
          &image, IREE_VM_BYTECODE_SECTION_METADATA);
      ASSERT_NE(header, nullptr);
      auto* scope = reinterpret_cast<iree_vm_bytecode_v0_metadata_scope_row_t*>(
          header + 1);
      auto* entry = reinterpret_cast<iree_vm_bytecode_v0_metadata_entry_row_t*>(
          scope + header->import_scope_count_u32 +
          header->export_scope_count_u32);
      entry[0].value_type_u16 = IREE_VM_BYTECODE_METADATA_VALUE_TYPE_INVALID;
      ExpectStructureRejected(image, "value type is invalid");
      break;
    }
    case ValidationObligation::CROSS_SECTION_PRESENCE: {
      std::vector<uint8_t> image = RemoveSection(
          BuildOwnershipModuleImage(), IREE_VM_BYTECODE_SECTION_STRINGS);
      ExpectStructureRejected(image, "string ordinal is out of range");
      break;
    }
    case ValidationObligation::CROSS_SECTION_VERSIONS: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      auto* header =
          reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(image.data());
      auto* rows = DirectoryRows(&image);
      rows[header->section_count_u16 - 1].section_type_u16 = 0xF001;
      rows[header->section_count_u16 - 1].section_flags_u16 =
          IREE_VM_BYTECODE_SECTION_FLAG_SKIPPABLE;
      ExpectStructureRejected(image, "require a Requirements declaration");
      break;
    }
    case ValidationObligation::CROSS_SECTION_PUBLICATION: {
      std::vector<uint8_t> image = BuildOwnershipModuleImage();
      image[0] ^= 0xFF;
      ExpectExecutableRejected(image, IREE_STATUS_INVALID_ARGUMENT,
                               "magic is invalid");
      break;
    }
  }
}

TEST(VMBytecodeVerificationTest, CoversEveryModuleValidationObligation) {
  EXPECT_EQ(IREE_ARRAYSIZE(kValidationObligations), 58u);
  for (const ValidationObligationCase& test_case : kValidationObligations) {
    SCOPED_TRACE(test_case.entity_id);
    RunValidationObligation(test_case.obligation);
  }
}

}  // namespace
