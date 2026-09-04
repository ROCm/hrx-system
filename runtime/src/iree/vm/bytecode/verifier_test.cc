// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/verifier.h"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/verifier_data.h"
#include "iree/vm/bytecode/verifier_testdata.h"
#include "iree/vm/bytecode/wire/core.h"

namespace {

template <typename T>
T* MutableRecord(std::vector<uint8_t>* bytes, iree_const_byte_span_t original,
                 const T* original_record) {
  const iree_host_size_t offset =
      reinterpret_cast<const uint8_t*>(original_record) - original.data;
  return reinterpret_cast<T*>(bytes->data() + offset);
}

uint8_t* MutableByte(std::vector<uint8_t>* bytes,
                     iree_const_byte_span_t original,
                     const uint8_t* original_byte) {
  return bytes->data() + (original_byte - original.data);
}

class ModuleVerificationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_file_toc_t* files = iree_vm_bytecode_verifier_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_verifier_testdata_size(), 1u);
    contents_ = iree_make_const_byte_span(files[0].data, files[0].size);
    ASSERT_TRUE(iree_host_ptr_has_alignment(contents_.data,
                                            IREE_VM_BYTECODE_IMAGE_ALIGNMENT));
    IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(contents_, &plan_));
  }

  std::vector<uint8_t> CopyImage() const {
    return std::vector<uint8_t>(contents_.data,
                                contents_.data + contents_.data_length);
  }

  void ExpectRejected(
      const std::vector<uint8_t>& bytes,
      iree_status_code_t expected_code = IREE_STATUS_INVALID_ARGUMENT) const {
    iree_vm_bytecode_module_plan_t output = {};
    output.layout.image.header =
        reinterpret_cast<const iree_vm_bytecode_v0_image_header_t*>(
            uintptr_t{1});
    output.process_layout.total_size = 123;
    IREE_EXPECT_STATUS_IS(
        expected_code,
        iree_vm_bytecode_verify_module_structure(
            iree_make_const_byte_span(bytes.data(), bytes.size()), &output));
    EXPECT_EQ(output.layout.image.header,
              reinterpret_cast<const iree_vm_bytecode_v0_image_header_t*>(
                  uintptr_t{1}));
    EXPECT_EQ(output.process_layout.total_size, 123u);
  }

  const iree_vm_bytecode_v0_section_directory_row_t* FindSection(
      uint16_t section_type) const {
    for (uint16_t i = 0; i < plan_.layout.image.section_count; ++i) {
      if (plan_.layout.image.sections[i].section_type_u16 == section_type) {
        return &plan_.layout.image.sections[i];
      }
    }
    return nullptr;
  }

  const uint8_t* FindInstruction(uint8_t opcode) const {
    const iree_vm_bytecode_v0_function_row_t* function =
        &plan_.layout.functions.rows[1];
    const iree_const_byte_span_t bytecode =
        iree_vm_bytecode_function_data(&plan_.layout.functions, function);
    for (iree_host_size_t offset = 0; offset < bytecode.data_length;) {
      const uint8_t* record = bytecode.data + offset;
      if (record[0] == opcode) return record;
      const uint32_t descriptor =
          iree_vm_bytecode_instruction_verification[record[0]];
      if (descriptor == 0) return nullptr;
      offset += iree_vm_bytecode_verification_byte_length(descriptor);
    }
    return nullptr;
  }

  void ExpectInstructionsRejected(std::vector<uint8_t> bytes) const {
    iree_vm_bytecode_module_plan_t plan = {};
    IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(
        iree_make_const_byte_span(bytes.data(), bytes.size()), &plan));
    std::vector<uint32_t> block_offsets(plan.maximum_block_count);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_vm_bytecode_verify_module_instructions(
                              &plan, block_offsets.data()));
  }

  iree_const_byte_span_t contents_ = iree_const_byte_span_empty();
  iree_vm_bytecode_module_plan_t plan_ = {};
};

TEST_F(ModuleVerificationTest, MapsCompleteCanonicalImage) {
  EXPECT_EQ(plan_.layout.image.section_count, 14u);
  EXPECT_EQ(plan_.layout.requirements.count, 1u);
  EXPECT_EQ(plan_.layout.strings.count, 18u);
  EXPECT_EQ(plan_.layout.ref_types.entry_count, 1u);
  EXPECT_EQ(plan_.layout.signatures.count, 3u);
  EXPECT_EQ(plan_.layout.callable_types.count, 3u);
  EXPECT_EQ(plan_.layout.imports.entry_count, 1u);
  EXPECT_EQ(plan_.layout.exports.count, 1u);
  EXPECT_EQ(plan_.layout.functions.count, 2u);
  EXPECT_EQ(plan_.layout.constants.count, 1u);
  ASSERT_NE(plan_.layout.globals.header, nullptr);
  EXPECT_EQ(plan_.layout.rodata.count, 1u);
  EXPECT_EQ(plan_.layout.presentation.entry_count, 2u);
  ASSERT_NE(plan_.layout.metadata.header, nullptr);
  EXPECT_EQ(plan_.layout.metadata.header->total_entry_count_u32, 3u);
  EXPECT_TRUE(iree_string_view_equal(
      iree_vm_bytecode_string_at(&plan_.layout.strings, 2), IREE_SV("run")));
  EXPECT_GT(plan_.process_layout.total_size,
            sizeof(iree_vm_bytecode_process_header_t));
  EXPECT_EQ(plan_.rodata_storage.copy_length, 0u);
  EXPECT_EQ(plan_.rodata_storage.copy_alignment, 0u);
  EXPECT_EQ(plan_.maximum_block_count, 1u);
}

TEST_F(ModuleVerificationTest, VerifiesEveryCoreInstruction) {
  std::vector<uint32_t> block_offsets(plan_.maximum_block_count);
  IREE_ASSERT_OK(iree_vm_bytecode_verify_module_instructions(
      &plan_, block_offsets.data()));
}

TEST_F(ModuleVerificationTest, RejectsUnknownInstruction) {
  std::vector<uint8_t> bytes = CopyImage();
  const uint8_t* original = FindInstruction(IREE_VM_BYTECODE_OPCODE_VALUE_COPY);
  ASSERT_NE(original, nullptr);
  uint8_t* record = MutableByte(&bytes, contents_, original);
  record[0] = 0;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsTruncatedInstruction) {
  std::vector<uint8_t> bytes = CopyImage();
  const iree_vm_bytecode_v0_function_row_t* function =
      &plan_.layout.functions.rows[1];
  const iree_const_byte_span_t bytecode =
      iree_vm_bytecode_function_data(&plan_.layout.functions, function);
  uint8_t* final_record =
      MutableByte(&bytes, contents_, bytecode.data + bytecode.data_length - 4);
  final_record[0] = IREE_VM_BYTECODE_OPCODE_CONTROL_CALL;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsMissingEntryBlock) {
  std::vector<uint8_t> bytes = CopyImage();
  uint8_t* entry =
      MutableByte(&bytes, contents_,
                  iree_vm_bytecode_function_data(
                      &plan_.layout.functions, &plan_.layout.functions.rows[1])
                      .data);
  entry[0] = IREE_VM_BYTECODE_OPCODE_CONTROL_RETURN;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsFunctionFallthrough) {
  std::vector<uint8_t> bytes = CopyImage();
  const iree_vm_bytecode_v0_function_row_t* function =
      &plan_.layout.functions.rows[1];
  const iree_const_byte_span_t bytecode =
      iree_vm_bytecode_function_data(&plan_.layout.functions, function);
  uint8_t* final_record =
      MutableByte(&bytes, contents_, bytecode.data + bytecode.data_length - 4);
  final_record[0] = IREE_VM_BYTECODE_OPCODE_VALUE_COPY;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsBlockCountMismatch) {
  std::vector<uint8_t> bytes = CopyImage();
  MutableRecord(&bytes, contents_, &plan_.layout.functions.rows[1])
      ->block_count_u32 = 2;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsCallSummaryMismatch) {
  std::vector<uint8_t> bytes = CopyImage();
  MutableRecord(&bytes, contents_, &plan_.layout.functions.rows[1])->flags_u16 =
      IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsInvalidSwitchTarget) {
  std::vector<uint8_t> bytes = CopyImage();
  MutableRecord(&bytes, contents_, plan_.layout.functions.switch_targets)
      ->target_word_offset_u32 = 1;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsInvalidDirectTarget) {
  std::vector<uint8_t> bytes = CopyImage();
  const uint8_t* original =
      FindInstruction(IREE_VM_BYTECODE_OPCODE_CONTROL_BRANCH_S16);
  ASSERT_NE(original, nullptr);
  auto* branch = MutableRecord(
      &bytes, contents_,
      reinterpret_cast<const iree_vm_bytecode_control_branch_s16_t*>(original));
  branch->target_word_offset_s16 = 0;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsInvalidSelector) {
  std::vector<uint8_t> bytes = CopyImage();
  const uint8_t* original =
      FindInstruction(IREE_VM_BYTECODE_OPCODE_CONTROL_FAIL);
  ASSERT_NE(original, nullptr);
  auto* fail = MutableRecord(
      &bytes, contents_,
      reinterpret_cast<const iree_vm_bytecode_control_fail_t*>(original));
  fail->status_u8 = 0;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsInvalidRegister) {
  std::vector<uint8_t> bytes = CopyImage();
  const uint8_t* original = FindInstruction(IREE_VM_BYTECODE_OPCODE_VALUE_COPY);
  ASSERT_NE(original, nullptr);
  auto* copy = MutableRecord(
      &bytes, contents_,
      reinterpret_cast<const iree_vm_bytecode_value_copy_t*>(original));
  copy->destination_v8 = 16;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, RejectsInvalidCallMoveMask) {
  std::vector<uint8_t> bytes = CopyImage();
  const uint8_t* original =
      FindInstruction(IREE_VM_BYTECODE_OPCODE_CONTROL_CALL);
  ASSERT_NE(original, nullptr);
  auto* call = MutableRecord(
      &bytes, contents_,
      reinterpret_cast<const iree_vm_bytecode_control_call_t*>(original));
  call->direct_ref_move_mask_u16 = 2;
  ExpectInstructionsRejected(std::move(bytes));
}

TEST_F(ModuleVerificationTest, PlansFallbackForActuallyMisalignedRodata) {
  std::vector<uint8_t> storage(contents_.data_length + 64);
  const uintptr_t storage_address = reinterpret_cast<uintptr_t>(storage.data());
  const iree_host_size_t offset = (8u + 64u - (storage_address & 63u)) & 63u;
  uint8_t* image = storage.data() + offset;
  ASSERT_EQ(reinterpret_cast<uintptr_t>(image) & 63u, 8u);
  std::memcpy(image, contents_.data, contents_.data_length);

  iree_vm_bytecode_module_plan_t plan = {};
  IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(
      iree_make_const_byte_span(image, contents_.data_length), &plan));
  EXPECT_EQ(plan.rodata_storage.copy_length, 5u);
  EXPECT_EQ(plan.rodata_storage.copy_alignment, 64u);
}

TEST_F(ModuleVerificationTest, RejectsEnvelopeAndDirectoryCorruption) {
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.image.header)->magic_u8[0] =
        'X';
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.image.header)
        ->core_major_u16 = 1;
    ExpectRejected(bytes, IREE_STATUS_INCOMPATIBLE);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    auto* row = MutableRecord(
        &bytes, contents_, FindSection(IREE_VM_BYTECODE_SECTION_PRESENTATION));
    row->section_flags_u16 = 0;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    auto* row = MutableRecord(&bytes, contents_,
                              FindSection(IREE_VM_BYTECODE_SECTION_STRINGS));
    row->section_type_u16 = IREE_VM_BYTECODE_SECTION_REQUIREMENTS;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    const uint8_t* strings_begin =
        reinterpret_cast<const uint8_t*>(plan_.layout.strings.offsets) -
        sizeof(iree_vm_bytecode_v0_strings_header_t);
    *MutableByte(&bytes, contents_, strings_begin - 1) = 1;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    auto* extension_row =
        MutableRecord(&bytes, contents_, FindSection(UINT16_C(0xF001)));
    extension_row->section_flags_u16 = 0;
    ExpectRejected(bytes, IREE_STATUS_INCOMPATIBLE);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    auto* requirement =
        MutableRecord(&bytes, contents_, plan_.layout.requirements.rows);
    requirement->page_id_u16 = UINT16_C(0xF1);
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    bytes.push_back(0);
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    bytes.pop_back();
    ExpectRejected(bytes);
  }
}

TEST_F(ModuleVerificationTest, RejectsDeclarationCorruption) {
  {
    std::vector<uint8_t> bytes = CopyImage();
    *MutableByte(&bytes, contents_, plan_.layout.strings.data) = 0xFF;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.ref_types.entries)
        ->required_flags_u16 = 1;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.signatures.descriptors)
        ->type_ordinal_u16 = 1;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, &plan_.layout.callable_types.rows[1])
        ->nesting_depth_u16 = 0;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.imports.entries)
        ->symbol_name_string_u16 = UINT16_MAX;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.exports.rows)
        ->function_ordinal_u16 = 1;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.functions.rows)
        ->value_register_count_u16 = 0;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.globals.header)
        ->immutable_value_count_u32 = 3;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    *MutableByte(&bytes, contents_,
                 plan_.layout.rodata.section_begin +
                     plan_.layout.rodata.blocks_offset) = 1;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, &plan_.layout.presentation.entries[1])
        ->field_base_u32 = 1;
    ExpectRejected(bytes);
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    *MutableByte(&bytes, contents_, plan_.layout.metadata.value_data + 4) = 2;
    ExpectRejected(bytes);
  }
}

TEST_F(ModuleVerificationTest, PreservesUnknownMetadataTypes) {
  std::vector<uint8_t> bytes = CopyImage();
  MutableRecord(&bytes, contents_, plan_.layout.metadata.entries)
      ->value_type_u16 = 7;
  iree_vm_bytecode_module_plan_t plan = {};
  IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &plan));
  EXPECT_EQ(plan.layout.metadata.entries[0].value_type_u16, 7u);
}

TEST_F(ModuleVerificationTest, TreatsYieldableAsPermission) {
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, &plan_.layout.callable_types.rows[1])
        ->flags_u16 = IREE_VM_BYTECODE_CALLABLE_TYPE_FLAG_MAY_YIELD;
    iree_vm_bytecode_module_plan_t plan = {};
    IREE_ASSERT_OK(iree_vm_bytecode_verify_module_structure(
        iree_make_const_byte_span(bytes.data(), bytes.size()), &plan));
  }
  {
    std::vector<uint8_t> bytes = CopyImage();
    MutableRecord(&bytes, contents_, plan_.layout.functions.rows)->flags_u16 =
        IREE_VM_BYTECODE_FUNCTION_FLAG_MAY_YIELD;
    ExpectRejected(bytes);
  }
}

}  // namespace
