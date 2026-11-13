// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/disassembler.h"
#include "iree/vm/bytecode/execution_testdata.h"
#include "iree/vm/bytecode/launch_config_testdata.h"
#include "iree/vm/bytecode/reference_signature_testdata.h"
#include "iree/vm/bytecode/reference_transport_testdata.h"
#include "iree/vm/bytecode/verifier_testdata.h"

namespace {

iree_status_t AppendText(void* user_data, iree_string_view_t fragment) {
  return iree_string_builder_append_string(
      static_cast<iree_string_builder_t*>(user_data), fragment);
}

std::string Disassemble(iree_const_byte_span_t contents) {
  iree_string_builder_t source;
  iree_string_builder_initialize(iree_allocator_system(), &source);
  iree_vm_bytecode_disassembler_write_callback_t callback = {AppendText,
                                                             &source};
  IREE_EXPECT_OK(iree_vm_bytecode_disassemble_module(contents, callback,
                                                     iree_allocator_system()));
  std::string result(iree_string_builder_buffer(&source),
                     iree_string_builder_size(&source));
  iree_string_builder_deinitialize(&source);
  return result;
}

void VerifyCanonicalSource(iree_string_view_t source) {
  iree_byte_sequence_t* assembled_contents = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_assemble_module(
      source, iree_allocator_system(), &assembled_contents));

  iree_byte_span_t flat_contents = iree_make_byte_span(nullptr, 0);
  IREE_ASSERT_OK(iree_byte_sequence_clone(
      assembled_contents, iree_allocator_system(), &flat_contents));

  const std::string canonical_text =
      Disassemble(iree_const_cast_byte_span(flat_contents));
  EXPECT_EQ(canonical_text, std::string(source.data, source.size));

  iree_byte_sequence_t* roundtrip_contents = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_assemble_module(
      iree_make_string_view(canonical_text.data(), canonical_text.size()),
      iree_allocator_system(), &roundtrip_contents));
  iree_byte_span_t flat_roundtrip_contents = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(
      roundtrip_contents, iree_allocator_system(), &flat_roundtrip_contents));
  EXPECT_EQ(flat_roundtrip_contents.data_length, flat_contents.data_length);
  if (flat_roundtrip_contents.data_length == flat_contents.data_length) {
    EXPECT_EQ(std::memcmp(flat_roundtrip_contents.data, flat_contents.data,
                          flat_contents.data_length),
              0);
  }

  iree_allocator_free(iree_allocator_system(), flat_roundtrip_contents.data);
  iree_byte_sequence_release(roundtrip_contents);
  iree_allocator_free(iree_allocator_system(), flat_contents.data);
  iree_byte_sequence_release(assembled_contents);
}

TEST(AssemblerTest, CanonicalSourcesCloseByteForByte) {
  const iree_file_toc_t* sources[] = {
      iree_vm_bytecode_execution_testdata_create(),
      iree_vm_bytecode_launch_config_testdata_create(),
      iree_vm_bytecode_reference_signature_testdata_create(),
      iree_vm_bytecode_reference_transport_testdata_create(),
      iree_vm_bytecode_verifier_testdata_create(),
  };
  for (const iree_file_toc_t* source : sources) {
    VerifyCanonicalSource(iree_make_string_view(
        reinterpret_cast<const char*>(source[0].data), source[0].size));
  }
}

TEST(AssemblerTest, EmptyRodataClosesByteForByte) {
  const iree_file_toc_t* fixture =
      iree_vm_bytecode_launch_config_testdata_create();
  std::string source(reinterpret_cast<const char*>(fixture[0].data),
                     fixture[0].size);
  source.insert(source.rfind('}'), "  section rodata alignment(8) {\n  }\n\n");
  VerifyCanonicalSource(iree_make_string_view(source.data(), source.size()));
}

TEST(AssemblerTest, RejectsSemanticViolations) {
  const iree_file_toc_t* execution =
      iree_vm_bytecode_execution_testdata_create();
  const std::string canonical_source(
      reinterpret_cast<const char*>(execution[0].data), execution[0].size);
  struct Mutation {
    const char* before;
    const char* after;
  };
  static const Mutation kMutations[] = {
      {"integer.add.i32 %v0, %v1", "integer.add.i32 %v0, %v255"},
      {"stack.copy #v8.x8, #v0.x8", "stack.copy #v16.x8, #v0.x8"},
      {"minimum_alignment_log2 = 0", "minimum_alignment_log2 = 255"},
      {"signature @signature1 (i32) -> (i32)",
       "signature @signature1 (invalid) -> (i32)"},
      {"values count(2) immutable(1)", "values count(2) immutable(3)"},
      {"callable @callable0 flags([]) depth(0)",
       "callable @callable0 flags([]) depth(1)"},
      {"callable @callable2 flags([yieldable]) depth(0) : @signature1",
       "callable @callable2 flags([yieldable]) depth(0) : @signature9"},
      {"func @function11 : @callable2 flags([yieldable])",
       "func @function11 : @callable1 flags([yieldable])"},
      {"export @export0 name(@string0) : @callable8 = @function14",
       "export @export0 name(@string0) : @callable1 = @function14"},
      {"control.call @function3 {direct_ref_move_mask = 0x0000}",
       "control.call @function3 {direct_ref_move_mask = 0x0001}"},
      {"metadata @string31 : utf8 = \"lo\\x00om\"",
       "metadata @string31 : f64 = hex\"00\""},
      {"metadata @string31 : utf8 = \"lo\\x00om\"",
       "metadata @string31 : invalid = hex\"\""},
  };
  for (const Mutation& mutation : kMutations) {
    std::string source = canonical_source;
    const size_t position = source.find(mutation.before);
    ASSERT_NE(position, std::string::npos) << mutation.before;
    source.replace(position, std::strlen(mutation.before), mutation.after);
    iree_byte_sequence_t* contents = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_bytecode_assemble_module(
            iree_make_string_view(source.data(), source.size()),
            iree_allocator_system(), &contents));
    EXPECT_EQ(contents, nullptr);
  }
}

}  // namespace
