// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/disassembler.h"

#include <cstdint>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/execution_testdata.h"
#include "iree/vm/bytecode/module_fixture.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/bytecode/wire/module.h"

namespace {

iree_status_t AppendText(void* user_data, iree_string_view_t fragment) {
  return iree_string_builder_append_string(
      static_cast<iree_string_builder_t*>(user_data), fragment);
}

struct CountingSink {
  // Number of fragments received.
  iree_host_size_t call_count = 0;
  // Whether to fail instead of accepting a fragment.
  bool fail = false;
};

iree_status_t CountWrites(void* user_data, iree_string_view_t) {
  auto* sink = static_cast<CountingSink*>(user_data);
  ++sink->call_count;
  if (sink->fail) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "test sink failure");
  }
  return iree_ok_status();
}

class DisassemblerModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const iree_file_toc_t* files = iree_vm_bytecode_execution_testdata_create();
    ASSERT_EQ(iree_vm_bytecode_execution_testdata_size(), 1u);
    IREE_ASSERT_OK(iree_vm_bytecode_module_fixture_initialize(
        iree_make_string_view(reinterpret_cast<const char*>(files[0].data),
                              files[0].size),
        iree_allocator_system(), &fixture_));
  }

  void TearDown() override {
    iree_vm_bytecode_module_fixture_deinitialize(&fixture_);
  }

  iree_const_byte_span_t contents() const {
    return iree_const_cast_byte_span(fixture_.contents);
  }

  iree_vm_bytecode_module_fixture_t fixture_ = {};
};

TEST(DisassemblerTest, LooksUpGeneratedInstructionNames) {
  EXPECT_TRUE(
      iree_string_view_equal(iree_vm_bytecode_disassembler_instruction_name(
                                 IREE_VM_BYTECODE_OPCODE_INTEGER_ADD_I32),
                             IREE_SV("integer.add.i32")));
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_vm_bytecode_disassembler_instruction_name(0xFF)));
}

TEST(DisassemblerTest, LooksUpGeneratedModuleRecordNames) {
  EXPECT_TRUE(
      iree_string_view_equal(iree_vm_bytecode_disassembler_module_record_name(
                                 IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_ROW),
                             IREE_SV("signature_row")));
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_vm_bytecode_disassembler_module_record_name(UINT8_MAX)));
}

TEST_F(DisassemblerModuleTest, DisassemblesVerifiedModuleToCanonicalText) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_vm_bytecode_disassembler_write_callback_t callback = {AppendText,
                                                             &builder};
  IREE_EXPECT_OK(iree_vm_bytecode_disassemble_module(contents(), callback,
                                                     iree_allocator_null()));

  const std::string text(iree_string_builder_buffer(&builder),
                         iree_string_builder_size(&builder));
  EXPECT_EQ(text.find('\0'), std::string::npos);
  EXPECT_EQ(text.find("vm.module core(0.0) {\n"), 0u);
  EXPECT_NE(text.find("    signature @signature1 (i32) -> (i32)\n"),
            std::string::npos);
  EXPECT_NE(
      text.find("    signature @signature2 (i32) -> (i32, ref(@ref_type0))\n"),
      std::string::npos);
  EXPECT_NE(text.find("      %v0 = integer.add.i32 %v0, %v1\n"),
            std::string::npos);
  EXPECT_NE(text.find("      control.yield.s32 ^bb1\n"), std::string::npos);
  EXPECT_NE(text.find("    rodata @rodata0 alignment(8) = "
                      "hex\"6C6F6F6D2D766D2D7631\"\n"),
            std::string::npos);
  EXPECT_GE(text.size(), 3u);
  if (text.size() >= 3) {
    EXPECT_EQ(text.substr(text.size() - 3), "\n}\n");
  }
  iree_string_builder_deinitialize(&builder);
}

TEST_F(DisassemblerModuleTest, VerifiesBeforeWriting) {
  const iree_const_byte_span_t fixture = contents();
  std::vector<uint8_t> bytes(fixture.data, fixture.data + fixture.data_length);
  bytes[0] ^= 0xFF;
  CountingSink sink;
  iree_vm_bytecode_disassembler_write_callback_t callback = {CountWrites,
                                                             &sink};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_bytecode_disassemble_module(
          iree_make_const_byte_span(bytes.data(), bytes.size()), callback,
          iree_allocator_system()));
  EXPECT_EQ(sink.call_count, 0u);
}

TEST_F(DisassemblerModuleTest, RejectsUnsupportedPresentationBeforeWriting) {
  const iree_const_byte_span_t fixture = contents();
  std::vector<uint8_t> bytes(fixture.data, fixture.data + fixture.data_length);
  auto* header =
      reinterpret_cast<iree_vm_bytecode_v0_image_header_t*>(bytes.data());
  auto* sections =
      reinterpret_cast<iree_vm_bytecode_v0_section_directory_row_t*>(header +
                                                                     1);
  sections[header->section_count_u16 - 1].section_type_u16 =
      IREE_VM_BYTECODE_SECTION_NATIVE_CALLS + 1;

  CountingSink sink;
  iree_vm_bytecode_disassembler_write_callback_t callback = {CountWrites,
                                                             &sink};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_vm_bytecode_disassemble_module(
          iree_make_const_byte_span(bytes.data(), bytes.size()), callback,
          iree_allocator_system()));
  EXPECT_EQ(sink.call_count, 0u);
}

TEST_F(DisassemblerModuleTest, PropagatesWriteFailure) {
  CountingSink sink;
  sink.fail = true;
  iree_vm_bytecode_disassembler_write_callback_t callback = {CountWrites,
                                                             &sink};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_vm_bytecode_disassemble_module(
                            contents(), callback, iree_allocator_system()));
  EXPECT_EQ(sink.call_count, 1u);
}

}  // namespace
