// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_integer.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::vm::bytecode::testing {
namespace {

template <typename Record, typename Execute>
void ExpectComparisonPredicates(Execute execute, uint64_t lhs, uint64_t rhs,
                                const uint64_t (&expected)[10]) {
  for (uint8_t predicate = 0; predicate < IREE_ARRAYSIZE(expected);
       ++predicate) {
    SCOPED_TRACE(static_cast<int>(predicate));
    uint64_t values[] = {lhs, rhs, UINT64_MAX};
    Record record = {};
    record.dst_v8 = 2;
    record.lhs_v8 = 0;
    record.rhs_v8 = 1;
    record.predicate_u8 = predicate;
    execute(&record, values);
    EXPECT_EQ(values[2], expected[predicate]);
  }
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCompareI32Record) {
  const uint64_t unequal_expected[] = {0, 1, 1, 1, 0, 0, 0, 0, 1, 1};
  ExpectComparisonPredicates<iree_vm_isa_integer_compare_i32_record_t>(
      iree_vm_bytecode_execute_integer_compare_i32, UINT32_MAX, 1,
      unequal_expected);

  const uint64_t reversed_expected[] = {0, 1, 0, 0, 1, 1, 1, 1, 0, 0};
  ExpectComparisonPredicates<iree_vm_isa_integer_compare_i32_record_t>(
      iree_vm_bytecode_execute_integer_compare_i32, 1, UINT32_MAX,
      reversed_expected);

  const uint64_t equal_expected[] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 1};
  ExpectComparisonPredicates<iree_vm_isa_integer_compare_i32_record_t>(
      iree_vm_bytecode_execute_integer_compare_i32, UINT32_C(0x80000000),
      UINT32_C(0x80000000), equal_expected);

  uint64_t values[] = {UINT32_MAX, 1};
  iree_vm_isa_integer_compare_i32_record_t record = {};
  record.dst_v8 = 0;
  record.lhs_v8 = 0;
  record.rhs_v8 = 1;
  record.predicate_u8 = IREE_VM_ISA_INTEGER_COMPARE_SLT;
  iree_vm_bytecode_execute_integer_compare_i32(&record, values);
  EXPECT_EQ(values[0], 1u);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCompareI64Record) {
  const uint64_t unequal_expected[] = {0, 1, 1, 1, 0, 0, 0, 0, 1, 1};
  ExpectComparisonPredicates<iree_vm_isa_integer_compare_i64_record_t>(
      iree_vm_bytecode_execute_integer_compare_i64, UINT64_MAX, 1,
      unequal_expected);

  const uint64_t reversed_expected[] = {0, 1, 0, 0, 1, 1, 1, 1, 0, 0};
  ExpectComparisonPredicates<iree_vm_isa_integer_compare_i64_record_t>(
      iree_vm_bytecode_execute_integer_compare_i64, 1, UINT64_MAX,
      reversed_expected);

  const uint64_t equal_expected[] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 1};
  ExpectComparisonPredicates<iree_vm_isa_integer_compare_i64_record_t>(
      iree_vm_bytecode_execute_integer_compare_i64,
      UINT64_C(0x8000000000000000), UINT64_C(0x8000000000000000),
      equal_expected);

  uint64_t values[] = {UINT64_MAX, 1};
  iree_vm_isa_integer_compare_i64_record_t record = {};
  record.dst_v8 = 1;
  record.lhs_v8 = 0;
  record.rhs_v8 = 1;
  record.predicate_u8 = IREE_VM_ISA_INTEGER_COMPARE_UGT;
  iree_vm_bytecode_execute_integer_compare_i64(&record, values);
  EXPECT_EQ(values[1], 1u);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesLeaI32Record) {
  uint64_t values[] = {13, UINT32_C(0xFFFFFFFD), UINT64_MAX};
  iree_vm_isa_integer_lea_i32_record_t record = {};
  record.dst_v8 = 2;
  record.base_v8 = 0;
  record.index_v8 = 1;
  record.scale_u8 = 255;
  record.offset_i16 = 0;
  iree_vm_bytecode_execute_integer_lea_i32(&record, values);
  EXPECT_EQ(values[2], UINT32_C(13) + UINT32_C(0xFFFFFFFD) * UINT32_C(255));

  values[0] = 11;
  values[1] = UINT32_MAX;
  record.dst_v8 = 0;
  record.scale_u8 = 0;
  record.offset_i16 = INT16_MIN;
  iree_vm_bytecode_execute_integer_lea_i32(&record, values);
  EXPECT_EQ(values[0], UINT32_C(11) + static_cast<uint32_t>(
                                          static_cast<int32_t>(INT16_MIN)));

  values[0] = 7;
  values[1] = UINT32_MAX;
  record.dst_v8 = 1;
  record.scale_u8 = 255;
  record.offset_i16 = INT16_MAX;
  iree_vm_bytecode_execute_integer_lea_i32(&record, values);
  EXPECT_EQ(values[1], UINT32_C(7) + UINT32_MAX * UINT32_C(255) +
                           static_cast<uint32_t>(INT16_MAX));
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesLeaI64Record) {
  uint64_t values[] = {13, UINT64_C(0xFFFFFFFFFFFFFFFD), UINT64_MAX};
  iree_vm_isa_integer_lea_i64_record_t record = {};
  record.dst_v8 = 2;
  record.base_v8 = 0;
  record.index_v8 = 1;
  record.scale_u8 = 255;
  record.offset_i16 = 0;
  iree_vm_bytecode_execute_integer_lea_i64(&record, values);
  EXPECT_EQ(values[2],
            UINT64_C(13) + UINT64_C(0xFFFFFFFFFFFFFFFD) * UINT64_C(255));

  values[0] = 11;
  values[1] = UINT64_MAX;
  record.dst_v8 = 0;
  record.scale_u8 = 0;
  record.offset_i16 = INT16_MIN;
  iree_vm_bytecode_execute_integer_lea_i64(&record, values);
  EXPECT_EQ(values[0], UINT64_C(11) + static_cast<uint64_t>(
                                          static_cast<int64_t>(INT16_MIN)));

  values[0] = 7;
  values[1] = UINT64_MAX;
  record.dst_v8 = 1;
  record.scale_u8 = 255;
  record.offset_i16 = INT16_MAX;
  iree_vm_bytecode_execute_integer_lea_i64(&record, values);
  EXPECT_EQ(values[1], UINT64_C(7) + UINT64_MAX * UINT64_C(255) +
                           static_cast<uint64_t>(INT16_MAX));
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCeilDivPow2U32Record) {
  struct TestCase {
    uint32_t source;
    uint8_t log2;
    uint32_t expected;
  };
  const TestCase test_cases[] = {
      {0, 31, 0},          {1, 31, 1}, {UINT32_C(1) << 31, 31, 1},
      {8, 3, 1},           {9, 3, 2},  {UINT32_MAX, 0, UINT32_MAX},
      {UINT32_MAX, 31, 2},
  };
  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(static_cast<int>(test_case.log2));
    uint64_t values[] = {test_case.source};
    iree_vm_isa_integer_ceildiv_pow2_u32_record_t record = {};
    record.dst_v8 = 0;
    record.src_v8 = 0;
    record.log2_u8 = test_case.log2;
    iree_vm_bytecode_execute_integer_ceildiv_pow2_u32(&record, values);
    EXPECT_EQ(values[0], test_case.expected);
  }
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCeilDivPow2U64Record) {
  struct TestCase {
    uint64_t source;
    uint8_t log2;
    uint64_t expected;
  };
  const TestCase test_cases[] = {
      {0, 63, 0},          {1, 63, 1}, {UINT64_C(1) << 63, 63, 1},
      {8, 3, 1},           {9, 3, 2},  {UINT64_MAX, 0, UINT64_MAX},
      {UINT64_MAX, 63, 2},
  };
  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(static_cast<int>(test_case.log2));
    uint64_t values[] = {test_case.source};
    iree_vm_isa_integer_ceildiv_pow2_u64_record_t record = {};
    record.dst_v8 = 0;
    record.src_v8 = 0;
    record.log2_u8 = test_case.log2;
    iree_vm_bytecode_execute_integer_ceildiv_pow2_u64(&record, values);
    EXPECT_EQ(values[0], test_case.expected);
  }
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
