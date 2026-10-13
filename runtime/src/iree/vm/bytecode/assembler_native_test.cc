// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/assembler_native.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/disassembler_native.h"

namespace {

class NativeOperandTextTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_vm_bytecode_assembler_native_free(call_, iree_allocator_system());
  }

  void Encode(const std::string& text,
              const iree_vm_native_call_schema_t& schema,
              const iree_vm_bytecode_v0_function_row_t& function) {
    IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&schema, 1));
    iree_vm_bytecode_assembler_lexer_t lexer;
    iree_vm_bytecode_assembler_lexer_initialize(
        iree_make_string_view(text.data(), text.size()), &lexer);
    IREE_ASSERT_OK(iree_vm_bytecode_assembler_native_parse(
        &lexer, iree_allocator_system(), &call_));
    EXPECT_EQ(lexer.cursor, lexer.end);
    words_.resize(call_->word_count);
    IREE_ASSERT_OK(iree_vm_bytecode_assembler_native_encode(
        call_, &schema, &function, words_.data(), iree_allocator_system()));
    const iree_vm_native_callsite_t callsite = {
        .schema = schema,
        .record = words_.data(),
        .word_count = call_->word_count,
        .storage =
            {
                .value_count = function.value_register_count_u16,
                .ref_count = function.ref_register_count_u16 +
                             function.local_ref_count_u32,
                .local_byte_length = function.local_byte_length_u16,
            },
    };
    IREE_ASSERT_OK(
        iree_vm_native_call_verify(&callsite, iree_allocator_system()));
    std::string rendered;
    const iree_vm_bytecode_disassembler_write_callback_t output = {
        .fn =
            [](void* user_data, iree_string_view_t fragment) {
              static_cast<std::string*>(user_data)->append(fragment.data,
                                                           fragment.size);
              return iree_ok_status();
            },
        .user_data = &rendered,
    };
    IREE_ASSERT_OK(iree_vm_bytecode_disassembler_native_call(
        &schema, &function, words_.data(), output, iree_allocator_system()));
    EXPECT_EQ(rendered, text);
  }

  // Owned parse tree retained through encoding and printing.
  iree_vm_bytecode_assembler_native_call_t* call_ = nullptr;
  // Encoded immutable record consumed by admission and printing.
  std::vector<uint32_t> words_;
};

TEST_F(NativeOperandTextTest, IndependentNestedRangesAndMixedStorage) {
  const iree_vm_native_call_layout_t layouts[] = {
      {0, 1, 1, 0}, {1, 3, 2, 1}, {3, 6, 4, 2}};
  const iree_vm_native_call_field_t fields[] = {
      {IREE_VM_SCALAR_TYPE_I64, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
      {IREE_VM_NATIVE_CALL_FIELD_REF, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_SCALAR_TYPE_I64, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 1},
      {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_NATIVE_CALL_FIELD_REF, 0},
  };
  const iree_vm_native_call_schema_t schema = {layouts, fields, 3, 7, 2};
  const iree_vm_bytecode_v0_function_row_t function = {
      .local_byte_length_u16 = 8,
      .value_register_count_u16 = 3,
      .ref_register_count_u16 = 1,
      .local_ref_count_u32 = 2,
  };
  Encode("(out %v2, [(%v1), (#v0)], [([(%v0)], %r0), ([], #r0)], out #r1)",
         schema, function);
  EXPECT_EQ(words_, (std::vector<uint32_t>{2, 2, 6, 2, 8, 2, 1,
                                           IREE_VM_NATIVE_CALL_LOCAL_VALUE, 1,
                                           14, 0, 0, 15, 1, 0}));
}

TEST_F(NativeOperandTextTest, ZeroFieldRangesRemainCompact) {
  const iree_vm_native_call_layout_t layouts[] = {{0, 0, 0, 0}, {0, 2, 1, 1}};
  const iree_vm_native_call_field_t field = {IREE_VM_NATIVE_CALL_FIELD_RANGE,
                                             0};
  const iree_vm_native_call_schema_t schema = {layouts, &field, 2, 1, 1};
  Encode("([() * 4294967295])", schema, {});
  EXPECT_EQ(words_, (std::vector<uint32_t>{UINT32_MAX, 2}));
}

TEST_F(NativeOperandTextTest, DeepOperandTreesUseExplicitStorage) {
  constexpr uint16_t depth = 4096;
  std::vector<iree_vm_native_call_layout_t> layouts(depth + 1);
  std::vector<iree_vm_native_call_field_t> fields(depth + 1);
  layouts[0] = {0, 1, 1, 0};
  fields[0] = {IREE_VM_SCALAR_TYPE_I64, 0};
  std::string text;
  for (uint16_t i = 1; i <= depth; ++i) {
    layouts[i] = {i, 2, 1, i};
    fields[i] = {IREE_VM_NATIVE_CALL_FIELD_RANGE, static_cast<uint16_t>(i - 1)};
    text += "([";
  }
  text += "(%v0)";
  for (uint16_t i = 0; i < depth; ++i) {
    text += "])";
  }
  const iree_vm_native_call_schema_t schema = {layouts.data(), fields.data(),
                                               depth + 1, depth + 1, depth};
  const iree_vm_bytecode_v0_function_row_t function = {
      .value_register_count_u16 = 1,
  };
  Encode(text, schema, function);
  ASSERT_EQ(words_.size(), 2 * depth + 1);
  EXPECT_EQ(words_.front(), 1u);
  EXPECT_EQ(words_.back(), 0u);
  const iree_vm_bytecode_disassembler_write_callback_t failing_output = {
      .fn =
          [](void*, iree_string_view_t) {
            return iree_make_status(IREE_STATUS_ABORTED, "closed output sink");
          },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_vm_bytecode_disassembler_native_call(
                            &schema, &function, words_.data(), failing_output,
                            iree_allocator_system()));
}

}  // namespace
