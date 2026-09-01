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
void ExpectBinaryResult(Execute execute, uint64_t lhs, uint64_t rhs,
                        uint64_t expected) {
  for (uint8_t destination = 0; destination < 3; ++destination) {
    SCOPED_TRACE(static_cast<int>(destination));
    uint64_t values[] = {lhs, rhs, UINT64_C(0xA55AA55AA55AA55A)};
    Record record = {};
    record.dst_v8 = destination;
    record.lhs_v8 = 0;
    record.rhs_v8 = 1;
    execute(&record, values);
    EXPECT_EQ(values[destination], expected);
  }
}

template <typename Record, typename Execute>
void ExpectUnaryResult(Execute execute, uint64_t source, uint64_t expected) {
  for (uint8_t destination = 0; destination < 2; ++destination) {
    SCOPED_TRACE(static_cast<int>(destination));
    uint64_t values[] = {source, UINT64_C(0xA55AA55AA55AA55A)};
    Record record = {};
    record.dst_v8 = destination;
    record.src_v8 = 0;
    execute(&record, values);
    EXPECT_EQ(values[destination], expected);
  }
}

template <typename Record, typename Execute>
void ExpectDivisionResult(
    Execute execute, uint64_t lhs, uint64_t rhs,
    iree_vm_bytecode_integer_division_failure_t expected_failure,
    uint64_t expected_result) {
  constexpr uint64_t kDestinationSentinel = UINT64_C(0xA55AA55AA55AA55A);
  uint64_t values[] = {lhs, rhs, kDestinationSentinel};
  Record record = {};
  record.dst_v8 = 2;
  record.lhs_v8 = 0;
  record.rhs_v8 = 1;
  const iree_vm_bytecode_integer_division_failure_t failure =
      execute(&record, values);
  EXPECT_EQ(failure, expected_failure);
  EXPECT_EQ(values[2], expected_result);
}

template <typename Record, typename Execute>
void ExpectDivisionAliasing(Execute execute, uint64_t lhs, uint64_t rhs,
                            uint64_t expected_result) {
  for (uint8_t destination = 0; destination < 2; ++destination) {
    SCOPED_TRACE(static_cast<int>(destination));
    uint64_t values[] = {lhs, rhs};
    Record record = {};
    record.dst_v8 = destination;
    record.lhs_v8 = 0;
    record.rhs_v8 = 1;
    EXPECT_EQ(execute(&record, values),
              IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE);
    EXPECT_EQ(values[destination], expected_result);
  }
}

template <typename Record>
Record MakeBitstreamRecord(uint8_t result_base, uint8_t source_base,
                           uint8_t field_width, uint8_t source_count,
                           uint8_t result_count, uint8_t source_width,
                           uint8_t result_width) {
  Record record = {};
  record.result_base_v8 = result_base;
  record.source_base_v8 = source_base;
  record.field_width_u8 = field_width;
  record.source_count_u8 = source_count;
  record.result_count_u8 = result_count;
  record.source_width_u8 = source_width;
  record.result_width_u8 = result_width;
  return record;
}

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

TEST(VMBytecodeInterpreterIntegerTest, ExecutesShiftLeftRecords) {
  constexpr auto execute_i32 = iree_vm_bytecode_execute_integer_shift_left_i32;
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i32_record_t>(execute_i32,
                                                                  1, 0, 1);
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i32_record_t>(
      execute_i32, 1, 31, UINT32_C(0x80000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i32_record_t>(execute_i32,
                                                                  3, 32, 3);
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i32_record_t>(
      execute_i32, UINT64_MAX, 255, UINT32_C(0x80000000));

  constexpr auto execute_i64 = iree_vm_bytecode_execute_integer_shift_left_i64;
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i64_record_t>(execute_i64,
                                                                  1, 0, 1);
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i64_record_t>(
      execute_i64, 1, 63, UINT64_C(0x8000000000000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i64_record_t>(execute_i64,
                                                                  3, 64, 3);
  ExpectBinaryResult<iree_vm_isa_integer_shift_left_i64_record_t>(
      execute_i64, 1, UINT64_MAX, UINT64_C(0x8000000000000000));
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesArithmeticShiftRightRecords) {
  constexpr auto execute_i32 = iree_vm_bytecode_execute_integer_shift_right_s32;
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s32_record_t>(
      execute_i32, UINT32_C(0x80000000), 0, UINT32_C(0x80000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s32_record_t>(
      execute_i32, UINT32_C(0x80000000), 31, UINT32_MAX);
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s32_record_t>(
      execute_i32, UINT32_C(0x80000000), 32, UINT32_C(0x80000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s32_record_t>(
      execute_i32, UINT64_MAX, 255, UINT32_MAX);

  constexpr auto execute_i64 = iree_vm_bytecode_execute_integer_shift_right_s64;
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 0,
      UINT64_C(0x8000000000000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 63, UINT64_MAX);
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 64,
      UINT64_C(0x8000000000000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_s64_record_t>(
      execute_i64, UINT64_MAX, UINT64_MAX, UINT64_MAX);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesLogicalShiftRightRecords) {
  constexpr auto execute_i32 = iree_vm_bytecode_execute_integer_shift_right_u32;
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u32_record_t>(
      execute_i32, UINT32_C(0x80000000), 0, UINT32_C(0x80000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u32_record_t>(
      execute_i32, UINT32_C(0x80000000), 31, 1);
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u32_record_t>(
      execute_i32, UINT32_C(0x80000000), 32, UINT32_C(0x80000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u32_record_t>(
      execute_i32, UINT64_MAX, 255, 1);

  constexpr auto execute_i64 = iree_vm_bytecode_execute_integer_shift_right_u64;
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 0,
      UINT64_C(0x8000000000000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 63, 1);
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 64,
      UINT64_C(0x8000000000000000));
  ExpectBinaryResult<iree_vm_isa_integer_shift_right_u64_record_t>(
      execute_i64, UINT64_MAX, UINT64_MAX, 1);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesRotateLeftRecords) {
  constexpr auto execute_i32 = iree_vm_bytecode_execute_integer_rotate_left_i32;
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i32_record_t>(execute_i32,
                                                                   1, 0, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i32_record_t>(
      execute_i32, UINT32_C(0x12345678), 4, UINT32_C(0x23456781));
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i32_record_t>(
      execute_i32, 1, 31, UINT32_C(0x80000000));
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i32_record_t>(execute_i32,
                                                                   1, 32, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i32_record_t>(
      execute_i32, 1, 255, UINT32_C(0x80000000));

  constexpr auto execute_i64 = iree_vm_bytecode_execute_integer_rotate_left_i64;
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i64_record_t>(execute_i64,
                                                                   1, 0, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i64_record_t>(
      execute_i64, UINT64_C(0x0123456789ABCDEF), 8,
      UINT64_C(0x23456789ABCDEF01));
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i64_record_t>(
      execute_i64, 1, 63, UINT64_C(0x8000000000000000));
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i64_record_t>(execute_i64,
                                                                   1, 64, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_left_i64_record_t>(
      execute_i64, 1, UINT64_MAX, UINT64_C(0x8000000000000000));
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesRotateRightRecords) {
  constexpr auto execute_i32 =
      iree_vm_bytecode_execute_integer_rotate_right_i32;
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i32_record_t>(execute_i32,
                                                                    1, 0, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i32_record_t>(
      execute_i32, UINT32_C(0x12345678), 4, UINT32_C(0x81234567));
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i32_record_t>(execute_i32,
                                                                    1, 31, 2);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i32_record_t>(execute_i32,
                                                                    1, 32, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i32_record_t>(execute_i32,
                                                                    1, 255, 2);

  constexpr auto execute_i64 =
      iree_vm_bytecode_execute_integer_rotate_right_i64;
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i64_record_t>(execute_i64,
                                                                    1, 0, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i64_record_t>(
      execute_i64, UINT64_C(0x0123456789ABCDEF), 8,
      UINT64_C(0xEF0123456789ABCD));
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i64_record_t>(execute_i64,
                                                                    1, 63, 2);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i64_record_t>(execute_i64,
                                                                    1, 64, 1);
  ExpectBinaryResult<iree_vm_isa_integer_rotate_right_i64_record_t>(
      execute_i64, 1, UINT64_MAX, 2);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCountLeadingZerosRecords) {
  constexpr auto execute_i32 =
      iree_vm_bytecode_execute_integer_count_leading_zeros_i32;
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i32_record_t>(
      execute_i32, 0, 32);
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i32_record_t>(
      execute_i32, 1, 31);
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i32_record_t>(
      execute_i32, UINT32_C(0x80000000), 0);
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i32_record_t>(
      execute_i32, UINT64_C(0xFFFFFFFF00000000), 32);

  constexpr auto execute_i64 =
      iree_vm_bytecode_execute_integer_count_leading_zeros_i64;
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i64_record_t>(
      execute_i64, 0, 64);
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i64_record_t>(
      execute_i64, 1, 63);
  ExpectUnaryResult<iree_vm_isa_integer_count_leading_zeros_i64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 0);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCountTrailingZerosRecords) {
  constexpr auto execute_i32 =
      iree_vm_bytecode_execute_integer_count_trailing_zeros_i32;
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i32_record_t>(
      execute_i32, 0, 32);
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i32_record_t>(
      execute_i32, 1, 0);
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i32_record_t>(
      execute_i32, UINT32_C(0x80000000), 31);
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i32_record_t>(
      execute_i32, UINT64_C(0xFFFFFFFF00000000), 32);

  constexpr auto execute_i64 =
      iree_vm_bytecode_execute_integer_count_trailing_zeros_i64;
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i64_record_t>(
      execute_i64, 0, 64);
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i64_record_t>(
      execute_i64, 1, 0);
  ExpectUnaryResult<iree_vm_isa_integer_count_trailing_zeros_i64_record_t>(
      execute_i64, UINT64_C(0x8000000000000000), 63);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesPopcountRecords) {
  constexpr auto execute_i32 = iree_vm_bytecode_execute_integer_popcount_i32;
  ExpectUnaryResult<iree_vm_isa_integer_popcount_i32_record_t>(execute_i32, 0,
                                                               0);
  ExpectUnaryResult<iree_vm_isa_integer_popcount_i32_record_t>(
      execute_i32, UINT32_C(0xAAAAAAAA), 16);
  ExpectUnaryResult<iree_vm_isa_integer_popcount_i32_record_t>(execute_i32,
                                                               UINT64_MAX, 32);

  constexpr auto execute_i64 = iree_vm_bytecode_execute_integer_popcount_i64;
  ExpectUnaryResult<iree_vm_isa_integer_popcount_i64_record_t>(execute_i64, 0,
                                                               0);
  ExpectUnaryResult<iree_vm_isa_integer_popcount_i64_record_t>(
      execute_i64, UINT64_C(0xAAAAAAAAAAAAAAAA), 32);
  ExpectUnaryResult<iree_vm_isa_integer_popcount_i64_record_t>(execute_i64,
                                                               UINT64_MAX, 64);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesDivS32Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_div_s32;
  constexpr auto kOk = IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
  ExpectDivisionResult<iree_vm_isa_integer_div_s32_record_t>(execute, 7, 3, kOk,
                                                             2);
  ExpectDivisionResult<iree_vm_isa_integer_div_s32_record_t>(
      execute, UINT32_C(0xFFFFFFF9), 3, kOk, UINT32_C(0xFFFFFFFE));
  ExpectDivisionResult<iree_vm_isa_integer_div_s32_record_t>(
      execute, 7, UINT32_C(0xFFFFFFFD), kOk, UINT32_C(0xFFFFFFFE));
  ExpectDivisionResult<iree_vm_isa_integer_div_s32_record_t>(
      execute, UINT32_C(0x80000000), 1, kOk, UINT32_C(0x80000000));
  ExpectDivisionResult<iree_vm_isa_integer_div_s32_record_t>(
      execute, UINT32_C(0x80000000), UINT32_MAX,
      IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionResult<iree_vm_isa_integer_div_s32_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_div_s32_record_t>(
      execute, UINT32_C(0xFFFFFFF9), UINT32_C(0xFFFFFFFD), 2);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesDivS64Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_div_s64;
  constexpr auto kOk = IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
  ExpectDivisionResult<iree_vm_isa_integer_div_s64_record_t>(execute, 7, 3, kOk,
                                                             2);
  ExpectDivisionResult<iree_vm_isa_integer_div_s64_record_t>(
      execute, UINT64_C(0xFFFFFFFFFFFFFFF9), 3, kOk,
      UINT64_C(0xFFFFFFFFFFFFFFFE));
  ExpectDivisionResult<iree_vm_isa_integer_div_s64_record_t>(
      execute, 7, UINT64_C(0xFFFFFFFFFFFFFFFD), kOk,
      UINT64_C(0xFFFFFFFFFFFFFFFE));
  ExpectDivisionResult<iree_vm_isa_integer_div_s64_record_t>(
      execute, UINT64_C(0x8000000000000000), 1, kOk,
      UINT64_C(0x8000000000000000));
  ExpectDivisionResult<iree_vm_isa_integer_div_s64_record_t>(
      execute, UINT64_C(0x8000000000000000), UINT64_MAX,
      IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionResult<iree_vm_isa_integer_div_s64_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_div_s64_record_t>(
      execute, UINT64_C(0xFFFFFFFFFFFFFFF9), UINT64_C(0xFFFFFFFFFFFFFFFD), 2);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesDivU32Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_div_u32;
  ExpectDivisionResult<iree_vm_isa_integer_div_u32_record_t>(
      execute, UINT64_C(0xFFFFFFFFFFFFFFFF), 2,
      IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE, UINT32_MAX / 2);
  ExpectDivisionResult<iree_vm_isa_integer_div_u32_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_div_u32_record_t>(
      execute, UINT32_MAX, 2, UINT32_MAX / 2);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesDivU64Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_div_u64;
  ExpectDivisionResult<iree_vm_isa_integer_div_u64_record_t>(
      execute, UINT64_MAX, 2, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE,
      UINT64_MAX / 2);
  ExpectDivisionResult<iree_vm_isa_integer_div_u64_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_div_u64_record_t>(
      execute, UINT64_MAX, 2, UINT64_MAX / 2);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesRemS32Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_rem_s32;
  constexpr auto kOk = IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
  ExpectDivisionResult<iree_vm_isa_integer_rem_s32_record_t>(execute, 7, 3, kOk,
                                                             1);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s32_record_t>(
      execute, UINT32_C(0xFFFFFFF9), 3, kOk, UINT32_MAX);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s32_record_t>(
      execute, 7, UINT32_C(0xFFFFFFFD), kOk, 1);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s32_record_t>(
      execute, UINT32_C(0x80000000), UINT32_MAX, kOk, 0);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s32_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_rem_s32_record_t>(
      execute, UINT32_C(0xFFFFFFF9), 3, UINT32_MAX);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesRemS64Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_rem_s64;
  constexpr auto kOk = IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
  ExpectDivisionResult<iree_vm_isa_integer_rem_s64_record_t>(execute, 7, 3, kOk,
                                                             1);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s64_record_t>(
      execute, UINT64_C(0xFFFFFFFFFFFFFFF9), 3, kOk, UINT64_MAX);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s64_record_t>(
      execute, 7, UINT64_C(0xFFFFFFFFFFFFFFFD), kOk, 1);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s64_record_t>(
      execute, UINT64_C(0x8000000000000000), UINT64_MAX, kOk, 0);
  ExpectDivisionResult<iree_vm_isa_integer_rem_s64_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_rem_s64_record_t>(
      execute, UINT64_C(0xFFFFFFFFFFFFFFF9), 3, UINT64_MAX);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesRemU32Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_rem_u32;
  ExpectDivisionResult<iree_vm_isa_integer_rem_u32_record_t>(
      execute, UINT64_C(0xFFFFFFFFFFFFFFFF), 3,
      IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE, UINT32_MAX % 3);
  ExpectDivisionResult<iree_vm_isa_integer_rem_u32_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_rem_u32_record_t>(
      execute, UINT32_MAX, 3, UINT32_MAX % 3);
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesRemU64Record) {
  constexpr auto execute = iree_vm_bytecode_execute_integer_rem_u64;
  ExpectDivisionResult<iree_vm_isa_integer_rem_u64_record_t>(
      execute, UINT64_MAX, 3, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE,
      UINT64_MAX % 3);
  ExpectDivisionResult<iree_vm_isa_integer_rem_u64_record_t>(
      execute, 1, 0, IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
      UINT64_C(0xA55AA55AA55AA55A));
  ExpectDivisionAliasing<iree_vm_isa_integer_rem_u64_record_t>(
      execute, UINT64_MAX, 3, UINT64_MAX % 3);
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

TEST(VMBytecodeInterpreterIntegerTest, PacksEveryCarrierWidth) {
  {
    uint64_t values[] = {1, 2, 3, 0, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_pack_record_t>(
            4, 0, 2, 4, 1, 8, 8);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[4], UINT64_C(0x39));
  }
  {
    uint64_t values[] = {0x1234, 0xAB12, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_pack_record_t>(
            2, 0, 8, 2, 1, 16, 16);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[2], UINT64_C(0x1234));
  }
  {
    uint64_t values[] = {0x89ABCDEF, 0x01234567, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_pack_record_t>(
            2, 0, 16, 2, 1, 32, 32);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[2], UINT64_C(0x4567CDEF));
  }
  {
    uint64_t values[] = {UINT64_C(0xFEDCBA9889ABCDEF),
                         UINT64_C(0x7654321001234567), UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_pack_record_t>(
            2, 0, 32, 2, 1, 64, 64);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[2], UINT64_C(0x0123456789ABCDEF));
  }
  {
    uint64_t values[] = {UINT64_MAX, 0};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_pack_record_t>(
            1, 0, 64, 1, 1, 64, 64);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[1], UINT64_MAX);
  }
}

TEST(VMBytecodeInterpreterIntegerTest, UnpacksEveryCarrierWidth) {
  {
    uint64_t values[] = {UINT64_C(0x39), UINT64_MAX, UINT64_MAX, UINT64_MAX,
                         UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_u_record_t>(
            1, 0, 2, 1, 4, 8, 8);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], 1u);
    EXPECT_EQ(values[2], 2u);
    EXPECT_EQ(values[3], 3u);
    EXPECT_EQ(values[4], 0u);
  }
  {
    uint64_t values[] = {UINT64_C(0xFF80), UINT64_MAX, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_u_record_t>(
            1, 0, 8, 1, 2, 16, 16);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], UINT64_C(0x80));
    EXPECT_EQ(values[2], UINT64_C(0xFF));
  }
  {
    uint64_t values[] = {UINT64_C(0x89ABCDEF), UINT64_MAX, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_u_record_t>(
            1, 0, 16, 1, 2, 32, 32);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], UINT64_C(0xCDEF));
    EXPECT_EQ(values[2], UINT64_C(0x89AB));
  }
  {
    uint64_t values[] = {UINT64_C(0x0123456789ABCDEF), UINT64_MAX, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_u_record_t>(
            1, 0, 32, 1, 2, 64, 64);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], UINT64_C(0x89ABCDEF));
    EXPECT_EQ(values[2], UINT64_C(0x01234567));
  }
}

TEST(VMBytecodeInterpreterIntegerTest, SignExtendsBitstreamFields) {
  uint64_t values[] = {UINT64_C(0xFF80), UINT64_MAX, UINT64_MAX};
  const auto record =
      MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_s_record_t>(
          1, 0, 8, 1, 2, 16, 16);
  iree_vm_bytecode_execute_integer_bitstream_unpack_s(&record, values);
  EXPECT_EQ(values[1], UINT64_C(0xFF80));
  EXPECT_EQ(values[2], UINT64_C(0xFFFF));

  uint64_t bits[65] = {UINT64_MAX};
  const auto bit_record =
      MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_s_record_t>(
          1, 0, 1, 1, 64, 64, 8);
  iree_vm_bytecode_execute_integer_bitstream_unpack_s(&bit_record, bits);
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(bits); ++i) {
    EXPECT_EQ(bits[i], UINT64_C(0xFF));
  }
}

TEST(VMBytecodeInterpreterIntegerTest, CapturesOverlappingBitstreamRanges) {
  constexpr uint8_t kRegisterCount = 16;
  for (uint8_t source_base = 0; source_base <= kRegisterCount - 8;
       ++source_base) {
    for (uint8_t result_base = 0; result_base <= kRegisterCount - 4;
         ++result_base) {
      SCOPED_TRACE(::testing::Message()
                   << "pack source_base=" << static_cast<int>(source_base)
                   << " result_base=" << static_cast<int>(result_base));
      uint64_t values[kRegisterCount];
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
        values[i] = UINT64_MAX;
      }
      for (uint8_t i = 0; i < 8; ++i) {
        values[source_base + i] = i + 1;
      }
      const auto record =
          MakeBitstreamRecord<iree_vm_isa_integer_bitstream_pack_record_t>(
              result_base, source_base, 4, 8, 4, 8, 8);
      iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
      EXPECT_EQ(values[result_base + 0], UINT64_C(0x21));
      EXPECT_EQ(values[result_base + 1], UINT64_C(0x43));
      EXPECT_EQ(values[result_base + 2], UINT64_C(0x65));
      EXPECT_EQ(values[result_base + 3], UINT64_C(0x87));
    }
  }

  for (uint8_t source_base = 0; source_base <= kRegisterCount - 4;
       ++source_base) {
    for (uint8_t result_base = 0; result_base <= kRegisterCount - 8;
         ++result_base) {
      SCOPED_TRACE(::testing::Message()
                   << "unpack source_base=" << static_cast<int>(source_base)
                   << " result_base=" << static_cast<int>(result_base));
      uint64_t values[kRegisterCount];
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
        values[i] = UINT64_MAX;
      }
      values[source_base + 0] = UINT64_C(0x21);
      values[source_base + 1] = UINT64_C(0x43);
      values[source_base + 2] = UINT64_C(0x65);
      values[source_base + 3] = UINT64_C(0x87);
      const auto record =
          MakeBitstreamRecord<iree_vm_isa_integer_bitstream_unpack_u_record_t>(
              result_base, source_base, 4, 4, 8, 8, 8);
      iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
      for (uint8_t i = 0; i < 8; ++i) {
        EXPECT_EQ(values[result_base + i], i + 1);
      }
    }
  }
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
