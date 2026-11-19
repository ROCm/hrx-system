// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_binary.h"

#include <algorithm>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/layout.h"

namespace {

typedef struct counting_byte_sequence_t {
  // Byte sequence interface exposed to the writer.
  iree_byte_sequence_t base;
  // Immutable bytes returned by enumeration.
  iree_const_byte_span_t contents;
  // Number of enumeration requests received.
  int enumeration_count;
  // Number of times the final retained reference was released.
  int destroy_count;
} counting_byte_sequence_t;

static counting_byte_sequence_t* counting_byte_sequence_cast(
    iree_byte_sequence_t* base_sequence) {
  return (counting_byte_sequence_t*)base_sequence;
}

static const counting_byte_sequence_t* counting_byte_sequence_const_cast(
    const iree_byte_sequence_t* base_sequence) {
  return (const counting_byte_sequence_t*)base_sequence;
}

static void counting_byte_sequence_destroy(
    iree_byte_sequence_t* base_sequence) {
  ++counting_byte_sequence_cast(base_sequence)->destroy_count;
}

static iree_status_t counting_byte_sequence_enumerate(
    const iree_byte_sequence_t* base_sequence,
    iree_byte_sequence_segment_callback_t callback) {
  auto* sequence = const_cast<counting_byte_sequence_t*>(
      counting_byte_sequence_const_cast(base_sequence));
  ++sequence->enumeration_count;
  return callback.fn(callback.user_data, sequence->contents);
}

static const iree_byte_sequence_vtable_t counting_byte_sequence_vtable = {
    .destroy = counting_byte_sequence_destroy,
    .enumerate = counting_byte_sequence_enumerate,
    .try_get_contiguous_span = NULL,
};

TEST(VMModuleBinaryTest,
     RepeatedEmissionRetainsPreparedFunctionBytesWithoutCopying) {
  const uint8_t function_bytes[] = {0x11, 0x22, 0x33, 0x44};
  counting_byte_sequence_t function_bytecode = {};
  iree_byte_sequence_initialize(&counting_byte_sequence_vtable,
                                sizeof(function_bytes),
                                &function_bytecode.base);
  function_bytecode.contents =
      iree_make_const_byte_span(function_bytes, sizeof(function_bytes));
  iree_vm_bytecode_v0_function_row_t function_row = {
      .bytecode_length_u32 = sizeof(function_bytes),
      .block_count_u32 = 1,
  };
  loom_vm_program_plan_t plan = {
      .functions = &function_row,
      .function_count = 1,
      .function_bytecode = &function_bytecode.base,
      .maximum_block_count = 1,
  };

  iree_byte_sequence_t* first_binary = NULL;
  iree_byte_sequence_t* second_binary = NULL;
  IREE_ASSERT_OK(loom_vm_program_emit_binary(&plan, iree_allocator_system(),
                                             &first_binary));
  IREE_ASSERT_OK(loom_vm_program_emit_binary(&plan, iree_allocator_system(),
                                             &second_binary));
  EXPECT_EQ(function_bytecode.enumeration_count, 0);
  EXPECT_EQ(function_bytecode.destroy_count, 0);

  loom_vm_program_plan_deinitialize(&plan);
  EXPECT_EQ(function_bytecode.destroy_count, 0);

  iree_byte_span_t first_bytes = iree_byte_span_empty();
  iree_byte_span_t second_bytes = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(first_binary, iree_allocator_system(),
                                          &first_bytes));
  IREE_ASSERT_OK(iree_byte_sequence_clone(
      second_binary, iree_allocator_system(), &second_bytes));
  EXPECT_EQ(std::vector<uint8_t>(first_bytes.data,
                                 first_bytes.data + first_bytes.data_length),
            std::vector<uint8_t>(second_bytes.data,
                                 second_bytes.data + second_bytes.data_length));
  ASSERT_GE(first_bytes.data_length, sizeof(function_bytes));
  EXPECT_TRUE(std::equal(
      std::begin(function_bytes), std::end(function_bytes),
      first_bytes.data + first_bytes.data_length - sizeof(function_bytes)));
  EXPECT_EQ(function_bytecode.enumeration_count, 2);

  iree_allocator_free(iree_allocator_system(), second_bytes.data);
  iree_allocator_free(iree_allocator_system(), first_bytes.data);
  iree_byte_sequence_release(first_binary);
  EXPECT_EQ(function_bytecode.destroy_count, 0);
  iree_byte_sequence_release(second_binary);
  EXPECT_EQ(function_bytecode.destroy_count, 1);
}

TEST(VMModuleBinaryTest, ReadsDerivedTablesAcrossBatchBoundaries) {
  constexpr uint16_t kCount = 67;
  const std::string text(kCount, 'x');
  std::vector<iree_string_view_t> strings(kCount);
  std::vector<iree_vm_bytecode_v0_signature_row_t> signatures(kCount);
  std::vector<iree_vm_bytecode_v0_signature_descriptor_row_t> descriptors;
  std::vector<loom_vm_rodata_plan_t> blocks(kCount);
  for (uint16_t i = 0; i < kCount; ++i) {
    strings[i] = iree_make_string_view(text.data(), i);
    signatures[i].descriptor_base_u32 = (uint32_t)descriptors.size();
    signatures[i].argument_count_u16 = i;
    signatures[i].argument_value_count_u16 = i;
    descriptors.insert(descriptors.end(), i,
                       {.kind_u16 = IREE_VM_BYTECODE_SIGNATURE_KIND_I32});
    blocks[i].contents = iree_make_const_byte_span(text.data(), i + 1);
    blocks[i].alignment = 1u << (i % 4);
  }
  const loom_vm_program_plan_t program = {
      .strings = strings.data(),
      .string_count = kCount,
      .signatures = signatures.data(),
      .signature_count = kCount,
      .signature_descriptors = descriptors.data(),
      .signature_descriptor_count = (uint32_t)descriptors.size(),
      .rodata = blocks.data(),
      .rodata_count = kCount,
      .rodata_alignment = 8,
      .has_rodata_section = true,
  };
  iree_byte_sequence_t* binary = nullptr;
  IREE_ASSERT_OK(
      loom_vm_program_emit_binary(&program, iree_allocator_system(), &binary));
  std::vector<uint8_t> bytes((size_t)iree_byte_sequence_length(binary));
  IREE_EXPECT_OK(iree_byte_sequence_read(
      binary, 0, iree_make_byte_span(bytes.data(), bytes.size())));
  iree_byte_sequence_release(binary);

  iree_vm_bytecode_module_plan_t loaded = {};
  IREE_ASSERT_OK(iree_vm_bytecode_module_plan_build(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &loaded));
  const auto& layout = loaded.layout;
  ASSERT_EQ(layout.strings.count, kCount);
  ASSERT_EQ(layout.callable_types.count, kCount);
  ASSERT_EQ(layout.rodata.count, kCount);
  for (uint16_t i = 0; i < kCount; ++i) {
    EXPECT_TRUE(iree_string_view_equal(
        iree_vm_bytecode_string_at(&layout.strings, i), strings[i]));
    EXPECT_EQ(layout.callable_types.rows[i].signature_ordinal_u16, i);
    EXPECT_EQ(layout.rodata.descriptors[i].byte_length_u64,
              blocks[i].contents.data_length);
    EXPECT_EQ(layout.rodata.descriptors[i].minimum_alignment_u32,
              blocks[i].alignment);
  }
}

}  // namespace
