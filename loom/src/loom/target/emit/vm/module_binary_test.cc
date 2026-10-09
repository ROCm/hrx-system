// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/module_binary.h"

#include <algorithm>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
    /*.destroy=*/counting_byte_sequence_destroy,
    /*.enumerate=*/counting_byte_sequence_enumerate,
    /*.try_get_contiguous_span=*/NULL,
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
      .bytecode_length_u32 = sizeof(function_bytes), .block_count_u32 = 1};
  loom_vm_program_plan_t plan = {.functions = &function_row,
                                 .function_count = 1,
                                 .function_bytecode = &function_bytecode.base,
                                 .maximum_block_count = 1};

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

}  // namespace
