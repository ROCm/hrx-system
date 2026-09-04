// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_integer.h"

#include <cstdint>
#include <limits>

#include "iree/testing/gtest.h"

namespace iree::vm::bytecode::testing {
namespace {

template <typename Record>
void ExpectBinaryResult(void (*execute)(const Record*, uint64_t*), uint64_t lhs,
                        uint64_t rhs, uint64_t expected) {
  for (uint8_t destination = 0; destination < 3; ++destination) {
    SCOPED_TRACE(static_cast<int>(destination));
    uint64_t values[] = {lhs, rhs, UINT64_C(0xA55AA55AA55AA55A)};
    Record record = {};
    record.destination_v8 = destination;
    record.left_v8 = 0;
    record.right_v8 = 1;
    execute(&record, values);
    EXPECT_EQ(values[destination], expected);
  }
}

template <typename Record>
void ExpectUnaryResult(void (*execute)(const Record*, uint64_t*),
                       uint64_t source, uint64_t expected) {
  for (uint8_t destination = 0; destination < 2; ++destination) {
    SCOPED_TRACE(static_cast<int>(destination));
    uint64_t values[] = {source, UINT64_C(0xA55AA55AA55AA55A)};
    Record record = {};
    record.destination_v8 = destination;
    record.source_v8 = 0;
    execute(&record, values);
    EXPECT_EQ(values[destination], expected);
  }
}

template <typename Record>
void ExpectDivisionResult(
    iree_vm_bytecode_integer_division_failure_t (*execute)(const Record*,
                                                           uint64_t*),
    uint64_t lhs, uint64_t rhs,
    iree_vm_bytecode_integer_division_failure_t expected_failure,
    uint64_t expected_result) {
  constexpr uint64_t kDestinationSentinel = UINT64_C(0xA55AA55AA55AA55A);
  uint64_t values[] = {lhs, rhs, kDestinationSentinel};
  Record record = {};
  record.destination_v8 = 2;
  record.left_v8 = 0;
  record.right_v8 = 1;
  const iree_vm_bytecode_integer_division_failure_t failure =
      execute(&record, values);
  EXPECT_EQ(failure, expected_failure);
  EXPECT_EQ(values[2], expected_result);
}

template <typename Record>
void ExpectDivisionAliasing(iree_vm_bytecode_integer_division_failure_t (
                                *execute)(const Record*, uint64_t*),
                            uint64_t lhs, uint64_t rhs,
                            uint64_t expected_result) {
  for (uint8_t destination = 0; destination < 2; ++destination) {
    SCOPED_TRACE(static_cast<int>(destination));
    uint64_t values[] = {lhs, rhs};
    Record record = {};
    record.destination_v8 = destination;
    record.left_v8 = 0;
    record.right_v8 = 1;
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

template <typename Record>
void ExpectComparisonPredicates(void (*execute)(const Record*, uint64_t*),
                                uint64_t lhs, uint64_t rhs,
                                const uint64_t (&expected)[10]) {
  for (uint8_t predicate = 0; predicate < IREE_ARRAYSIZE(expected);
       ++predicate) {
    SCOPED_TRACE(static_cast<int>(predicate));
    uint64_t values[] = {lhs, rhs, UINT64_MAX};
    Record record = {};
    record.destination_v8 = 2;
    record.left_v8 = 0;
    record.right_v8 = 1;
    record.predicate_u8 = predicate;
    execute(&record, values);
    EXPECT_EQ(values[2], expected[predicate]);
  }
}

template <typename UInt>
struct IntegerTraits;

#define IREE_VM_BYTECODE_DEFINE_INTEGER_TRAITS(                               \
    unsigned_type, signed_type, width, rotate_source, rotate_count,           \
    rotate_left_expected, rotate_right_expected)                              \
  template <>                                                                 \
  struct IntegerTraits<unsigned_type> {                                       \
    using Signed = signed_type;                                               \
    using CompareRecord = iree_vm_bytecode_integer_compare_i##width##_t;      \
    using LeaRecord = iree_vm_bytecode_integer_lea_i##width##_t;              \
    using CeilDivRecord = iree_vm_bytecode_integer_ceildiv_pow2_u##width##_t; \
    static constexpr auto kAdd =                                              \
        iree_vm_bytecode_execute_integer_add_i##width;                        \
    static constexpr auto kSub =                                              \
        iree_vm_bytecode_execute_integer_sub_i##width;                        \
    static constexpr auto kMul =                                              \
        iree_vm_bytecode_execute_integer_mul_i##width;                        \
    static constexpr auto kMinS =                                             \
        iree_vm_bytecode_execute_integer_min_s##width;                        \
    static constexpr auto kMinU =                                             \
        iree_vm_bytecode_execute_integer_min_u##width;                        \
    static constexpr auto kMaxS =                                             \
        iree_vm_bytecode_execute_integer_max_s##width;                        \
    static constexpr auto kMaxU =                                             \
        iree_vm_bytecode_execute_integer_max_u##width;                        \
    static constexpr auto kAnd =                                              \
        iree_vm_bytecode_execute_integer_and_i##width;                        \
    static constexpr auto kOr = iree_vm_bytecode_execute_integer_or_i##width; \
    static constexpr auto kXor =                                              \
        iree_vm_bytecode_execute_integer_xor_i##width;                        \
    static constexpr auto kNeg =                                              \
        iree_vm_bytecode_execute_integer_neg_i##width;                        \
    static constexpr auto kAbs =                                              \
        iree_vm_bytecode_execute_integer_abs_s##width;                        \
    static constexpr auto kShiftLeft =                                        \
        iree_vm_bytecode_execute_integer_shift_left_i##width;                 \
    static constexpr auto kShiftRightS =                                      \
        iree_vm_bytecode_execute_integer_shift_right_s##width;                \
    static constexpr auto kShiftRightU =                                      \
        iree_vm_bytecode_execute_integer_shift_right_u##width;                \
    static constexpr auto kRotateLeft =                                       \
        iree_vm_bytecode_execute_integer_rotate_left_i##width;                \
    static constexpr auto kRotateRight =                                      \
        iree_vm_bytecode_execute_integer_rotate_right_i##width;               \
    static constexpr auto kCountLeadingZeros =                                \
        iree_vm_bytecode_execute_integer_count_leading_zeros_i##width;        \
    static constexpr auto kCountTrailingZeros =                               \
        iree_vm_bytecode_execute_integer_count_trailing_zeros_i##width;       \
    static constexpr auto kPopcount =                                         \
        iree_vm_bytecode_execute_integer_popcount_i##width;                   \
    static constexpr auto kDivS =                                             \
        iree_vm_bytecode_execute_integer_div_s##width;                        \
    static constexpr auto kDivU =                                             \
        iree_vm_bytecode_execute_integer_div_u##width;                        \
    static constexpr auto kRemS =                                             \
        iree_vm_bytecode_execute_integer_rem_s##width;                        \
    static constexpr auto kRemU =                                             \
        iree_vm_bytecode_execute_integer_rem_u##width;                        \
    static constexpr auto kCompare =                                          \
        iree_vm_bytecode_execute_integer_compare_i##width;                    \
    static constexpr auto kLea =                                              \
        iree_vm_bytecode_execute_integer_lea_i##width;                        \
    static constexpr auto kCeilDiv =                                          \
        iree_vm_bytecode_execute_integer_ceildiv_pow2_u##width;               \
    static constexpr uint64_t kRotateSource = rotate_source;                  \
    static constexpr uint32_t kRotateCount = rotate_count;                    \
    static constexpr uint64_t kRotateLeftExpected = rotate_left_expected;     \
    static constexpr uint64_t kRotateRightExpected = rotate_right_expected;   \
  }

IREE_VM_BYTECODE_DEFINE_INTEGER_TRAITS(uint32_t, int32_t, 32,
                                       UINT32_C(0x12345678), 4,
                                       UINT32_C(0x23456781),
                                       UINT32_C(0x81234567));
IREE_VM_BYTECODE_DEFINE_INTEGER_TRAITS(uint64_t, int64_t, 64,
                                       UINT64_C(0x0123456789ABCDEF), 8,
                                       UINT64_C(0x23456789ABCDEF01),
                                       UINT64_C(0xEF0123456789ABCD));

#undef IREE_VM_BYTECODE_DEFINE_INTEGER_TRAITS

template <typename UInt>
void CheckArithmeticRecords() {
  using Traits = IntegerTraits<UInt>;
  constexpr uint32_t kWidth = sizeof(UInt) * 8;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  constexpr uint64_t kSign = uint64_t{1} << (kWidth - 1);
  constexpr uint64_t kSignedMax = kSign - 1;
  constexpr uint64_t kNegativeTwo = kMax - 1;
  constexpr uint64_t kAlternating = kMax / 3;
  SCOPED_TRACE(kWidth);
  ExpectBinaryResult(Traits::kAdd, kMax, 2, 1);
  ExpectBinaryResult(Traits::kSub, 0, 1, kMax);
  ExpectBinaryResult(Traits::kMul, kSign + 1, 2, 2);
  ExpectBinaryResult(Traits::kMinS, kSign, kSignedMax, kSign);
  ExpectBinaryResult(Traits::kMinU, kSign, kSignedMax, kSignedMax);
  ExpectBinaryResult(Traits::kMaxS, kSign, kSignedMax, kSignedMax);
  ExpectBinaryResult(Traits::kMaxU, kSign, kSignedMax, kSign);
  ExpectBinaryResult(Traits::kAnd, kMax, kAlternating, kAlternating);
  ExpectBinaryResult(Traits::kOr, (~kMax) | (kMax ^ kAlternating), kAlternating,
                     kMax);
  ExpectBinaryResult(Traits::kXor, kMax ^ kAlternating, kAlternating, kMax);
  ExpectUnaryResult(Traits::kNeg, 1, kMax);
  ExpectUnaryResult(Traits::kAbs, kNegativeTwo, 2);
  ExpectUnaryResult(Traits::kAbs, kSign, kSign);
}

template <typename UInt>
void CheckShiftAndRotateRecords() {
  using Traits = IntegerTraits<UInt>;
  constexpr uint32_t kWidth = sizeof(UInt) * 8;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  constexpr uint64_t kSign = uint64_t{1} << (kWidth - 1);
  SCOPED_TRACE(kWidth);
  ExpectBinaryResult(Traits::kShiftLeft, 1, 0, 1);
  ExpectBinaryResult(Traits::kShiftLeft, 1, kWidth - 1, kSign);
  ExpectBinaryResult(Traits::kShiftLeft, 3, kWidth, 3);
  ExpectBinaryResult(Traits::kShiftLeft, 1, UINT64_MAX, kSign);
  ExpectBinaryResult(Traits::kShiftRightS, kSign, 0, kSign);
  ExpectBinaryResult(Traits::kShiftRightS, kSign, kWidth - 1, kMax);
  ExpectBinaryResult(Traits::kShiftRightS, kSign, kWidth, kSign);
  ExpectBinaryResult(Traits::kShiftRightS, kMax, UINT64_MAX, kMax);
  ExpectBinaryResult(Traits::kShiftRightU, kSign, 0, kSign);
  ExpectBinaryResult(Traits::kShiftRightU, kSign, kWidth - 1, 1);
  ExpectBinaryResult(Traits::kShiftRightU, kSign, kWidth, kSign);
  ExpectBinaryResult(Traits::kShiftRightU, kMax, UINT64_MAX, 1);
  ExpectBinaryResult(Traits::kRotateLeft, 1, 0, 1);
  ExpectBinaryResult(Traits::kRotateLeft, Traits::kRotateSource,
                     Traits::kRotateCount, Traits::kRotateLeftExpected);
  ExpectBinaryResult(Traits::kRotateLeft, 1, kWidth - 1, kSign);
  ExpectBinaryResult(Traits::kRotateLeft, 1, kWidth, 1);
  ExpectBinaryResult(Traits::kRotateLeft, 1, UINT64_MAX, kSign);
  ExpectBinaryResult(Traits::kRotateRight, 1, 0, 1);
  ExpectBinaryResult(Traits::kRotateRight, Traits::kRotateSource,
                     Traits::kRotateCount, Traits::kRotateRightExpected);
  ExpectBinaryResult(Traits::kRotateRight, 1, kWidth - 1, 2);
  ExpectBinaryResult(Traits::kRotateRight, 1, kWidth, 1);
  ExpectBinaryResult(Traits::kRotateRight, 1, UINT64_MAX, 2);
}

template <typename UInt>
void CheckCountRecords() {
  using Traits = IntegerTraits<UInt>;
  constexpr uint32_t kWidth = sizeof(UInt) * 8;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  constexpr uint64_t kSign = uint64_t{1} << (kWidth - 1);
  constexpr uint64_t kAlternating = kMax / 3;
  SCOPED_TRACE(kWidth);
  ExpectUnaryResult(Traits::kCountLeadingZeros, 0, kWidth);
  ExpectUnaryResult(Traits::kCountLeadingZeros, 1, kWidth - 1);
  ExpectUnaryResult(Traits::kCountLeadingZeros, kSign, 0);
  ExpectUnaryResult(Traits::kCountLeadingZeros, UINT64_MAX ^ kMax, kWidth);
  ExpectUnaryResult(Traits::kCountTrailingZeros, 0, kWidth);
  ExpectUnaryResult(Traits::kCountTrailingZeros, 1, 0);
  ExpectUnaryResult(Traits::kCountTrailingZeros, kSign, kWidth - 1);
  ExpectUnaryResult(Traits::kCountTrailingZeros, UINT64_MAX ^ kMax, kWidth);
  ExpectUnaryResult(Traits::kPopcount, 0, 0);
  ExpectUnaryResult(Traits::kPopcount, kMax ^ kAlternating, kWidth / 2);
  ExpectUnaryResult(Traits::kPopcount, UINT64_MAX, kWidth);
}

template <typename UInt>
void CheckDivisionRecords() {
  using Traits = IntegerTraits<UInt>;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  constexpr uint64_t kSign = uint64_t{1} << (sizeof(UInt) * 8 - 1);
  constexpr uint64_t kNegativeTwo = kMax - 1;
  constexpr uint64_t kNegativeThree = kMax - 2;
  constexpr uint64_t kNegativeSeven = kMax - 6;
  constexpr auto kOk = IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
  constexpr uint64_t kSentinel = UINT64_C(0xA55AA55AA55AA55A);
  SCOPED_TRACE(sizeof(UInt) * 8);
  ExpectDivisionResult(Traits::kDivS, 7, 3, kOk, 2);
  ExpectDivisionResult(Traits::kDivS, kNegativeSeven, 3, kOk, kNegativeTwo);
  ExpectDivisionResult(Traits::kDivS, 7, kNegativeThree, kOk, kNegativeTwo);
  ExpectDivisionResult(Traits::kDivS, kSign, 1, kOk, kSign);
  ExpectDivisionResult(
      Traits::kDivS, kSign, kMax,
      IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW, kSentinel);
  ExpectDivisionResult(Traits::kDivS, 1, 0,
                       IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
                       kSentinel);
  ExpectDivisionAliasing(Traits::kDivS, kNegativeSeven, kNegativeThree, 2);
  ExpectDivisionResult(Traits::kDivU, UINT64_MAX, 2, kOk, kMax / 2);
  ExpectDivisionResult(Traits::kDivU, 1, 0,
                       IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
                       kSentinel);
  ExpectDivisionAliasing(Traits::kDivU, kMax, 2, kMax / 2);
  ExpectDivisionResult(Traits::kRemS, 7, 3, kOk, 1);
  ExpectDivisionResult(Traits::kRemS, kNegativeSeven, 3, kOk, kMax);
  ExpectDivisionResult(Traits::kRemS, 7, kNegativeThree, kOk, 1);
  ExpectDivisionResult(Traits::kRemS, kSign, kMax, kOk, 0);
  ExpectDivisionResult(Traits::kRemS, 1, 0,
                       IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
                       kSentinel);
  ExpectDivisionAliasing(Traits::kRemS, kNegativeSeven, 3, kMax);
  ExpectDivisionResult(Traits::kRemU, UINT64_MAX, 3, kOk, kMax % 3);
  ExpectDivisionResult(Traits::kRemU, 1, 0,
                       IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO,
                       kSentinel);
  ExpectDivisionAliasing(Traits::kRemU, kMax, 3, kMax % 3);
}

template <typename UInt>
void CheckCompareRecords() {
  using Traits = IntegerTraits<UInt>;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  constexpr uint64_t kSign = uint64_t{1} << (sizeof(UInt) * 8 - 1);
  SCOPED_TRACE(sizeof(UInt) * 8);
  const uint64_t unequal_expected[] = {0, 1, 1, 1, 0, 0, 0, 0, 1, 1};
  ExpectComparisonPredicates(Traits::kCompare, kMax, 1, unequal_expected);
  const uint64_t reversed_expected[] = {0, 1, 0, 0, 1, 1, 1, 1, 0, 0};
  ExpectComparisonPredicates(Traits::kCompare, 1, kMax, reversed_expected);
  const uint64_t equal_expected[] = {1, 0, 0, 1, 0, 1, 0, 1, 0, 1};
  ExpectComparisonPredicates(Traits::kCompare, kSign, kSign, equal_expected);

  for (uint8_t destination = 0; destination < 2; ++destination) {
    uint64_t values[] = {kMax, 1};
    typename Traits::CompareRecord record = {};
    record.destination_v8 = destination;
    record.left_v8 = 0;
    record.right_v8 = 1;
    record.predicate_u8 = destination == 0
                              ? IREE_VM_BYTECODE_INTEGER_COMPARE_SLT
                              : IREE_VM_BYTECODE_INTEGER_COMPARE_UGT;
    Traits::kCompare(&record, values);
    EXPECT_EQ(values[destination], 1u);
  }
}

template <typename UInt>
void CheckLeaRecords() {
  using Traits = IntegerTraits<UInt>;
  using Signed = typename Traits::Signed;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  constexpr uint64_t kNegativeThree = kMax - 2;
  SCOPED_TRACE(sizeof(UInt) * 8);
  uint64_t lea_values[] = {13, kNegativeThree, UINT64_MAX};
  typename Traits::LeaRecord lea_record = {};
  lea_record.destination_v8 = 2;
  lea_record.base_v8 = 0;
  lea_record.index_v8 = 1;
  lea_record.scale_u8 = 255;
  Traits::kLea(&lea_record, lea_values);
  EXPECT_EQ(lea_values[2], static_cast<UInt>(13 + kNegativeThree * 255));
  lea_values[0] = 11;
  lea_values[1] = kMax;
  lea_record.destination_v8 = 0;
  lea_record.scale_u8 = 0;
  lea_record.offset_i16 = INT16_MIN;
  Traits::kLea(&lea_record, lea_values);
  EXPECT_EQ(lea_values[0],
            static_cast<UInt>(11) +
                static_cast<UInt>(static_cast<Signed>(INT16_MIN)));
  lea_values[0] = 7;
  lea_values[1] = kMax;
  lea_record.destination_v8 = 1;
  lea_record.scale_u8 = 255;
  lea_record.offset_i16 = INT16_MAX;
  Traits::kLea(&lea_record, lea_values);
  EXPECT_EQ(lea_values[1],
            static_cast<UInt>(7) + static_cast<UInt>(kMax) * 255 + INT16_MAX);
}

template <typename UInt>
void CheckCeilDivRecords() {
  using Traits = IntegerTraits<UInt>;
  constexpr uint32_t kWidth = sizeof(UInt) * 8;
  constexpr uint64_t kMax = std::numeric_limits<UInt>::max();
  SCOPED_TRACE(kWidth);
  struct CeilDivCase {
    UInt source;
    uint8_t log2;
    UInt expected;
  };
  const CeilDivCase ceildiv_cases[] = {
      {0, kWidth - 1, 0},
      {1, kWidth - 1, 1},
      {UInt{1} << (kWidth - 1), kWidth - 1, 1},
      {8, 3, 1},
      {9, 3, 2},
      {static_cast<UInt>(kMax), 0, static_cast<UInt>(kMax)},
      {static_cast<UInt>(kMax), kWidth - 1, 2},
  };
  for (const CeilDivCase& test_case : ceildiv_cases) {
    SCOPED_TRACE(static_cast<int>(test_case.log2));
    uint64_t values[] = {test_case.source};
    typename Traits::CeilDivRecord record = {};
    record.destination_v8 = 0;
    record.source_v8 = 0;
    record.log2_u8 = test_case.log2;
    Traits::kCeilDiv(&record, values);
    EXPECT_EQ(values[0], test_case.expected);
  }
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesArithmeticRecords) {
  CheckArithmeticRecords<uint32_t>();
  CheckArithmeticRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesShiftAndRotateRecords) {
  CheckShiftAndRotateRecords<uint32_t>();
  CheckShiftAndRotateRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCountRecords) {
  CheckCountRecords<uint32_t>();
  CheckCountRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesDivisionRecords) {
  CheckDivisionRecords<uint32_t>();
  CheckDivisionRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCompareRecords) {
  CheckCompareRecords<uint32_t>();
  CheckCompareRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesLeaRecords) {
  CheckLeaRecords<uint32_t>();
  CheckLeaRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, ExecutesCeilDivRecords) {
  CheckCeilDivRecords<uint32_t>();
  CheckCeilDivRecords<uint64_t>();
}

TEST(VMBytecodeInterpreterIntegerTest, PacksEveryCarrierWidth) {
  {
    uint64_t values[] = {1, 2, 3, 0, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_pack_t>(
            4, 0, 2, 4, 1, 8, 8);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[4], UINT64_C(0x39));
  }
  {
    uint64_t values[] = {0x1234, 0xAB12, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_pack_t>(
            2, 0, 8, 2, 1, 16, 16);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[2], UINT64_C(0x1234));
  }
  {
    uint64_t values[] = {0x89ABCDEF, 0x01234567, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_pack_t>(
            2, 0, 16, 2, 1, 32, 32);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[2], UINT64_C(0x4567CDEF));
  }
  {
    uint64_t values[] = {UINT64_C(0xFEDCBA9889ABCDEF),
                         UINT64_C(0x7654321001234567), UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_pack_t>(
            2, 0, 32, 2, 1, 64, 64);
    iree_vm_bytecode_execute_integer_bitstream_pack(&record, values);
    EXPECT_EQ(values[2], UINT64_C(0x0123456789ABCDEF));
  }
  {
    uint64_t values[] = {UINT64_MAX, 0};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_pack_t>(
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
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_u_t>(
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
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_u_t>(
            1, 0, 8, 1, 2, 16, 16);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], UINT64_C(0x80));
    EXPECT_EQ(values[2], UINT64_C(0xFF));
  }
  {
    uint64_t values[] = {UINT64_C(0x89ABCDEF), UINT64_MAX, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_u_t>(
            1, 0, 16, 1, 2, 32, 32);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], UINT64_C(0xCDEF));
    EXPECT_EQ(values[2], UINT64_C(0x89AB));
  }
  {
    uint64_t values[] = {UINT64_C(0x0123456789ABCDEF), UINT64_MAX, UINT64_MAX};
    const auto record =
        MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_u_t>(
            1, 0, 32, 1, 2, 64, 64);
    iree_vm_bytecode_execute_integer_bitstream_unpack_u(&record, values);
    EXPECT_EQ(values[1], UINT64_C(0x89ABCDEF));
    EXPECT_EQ(values[2], UINT64_C(0x01234567));
  }
}

TEST(VMBytecodeInterpreterIntegerTest, SignExtendsBitstreamFields) {
  uint64_t values[] = {UINT64_C(0xFF80), UINT64_MAX, UINT64_MAX};
  const auto record =
      MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_s_t>(
          1, 0, 8, 1, 2, 16, 16);
  iree_vm_bytecode_execute_integer_bitstream_unpack_s(&record, values);
  EXPECT_EQ(values[1], UINT64_C(0xFF80));
  EXPECT_EQ(values[2], UINT64_C(0xFFFF));

  uint64_t bits[65] = {UINT64_MAX};
  const auto bit_record =
      MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_s_t>(
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
          MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_pack_t>(
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
          MakeBitstreamRecord<iree_vm_bytecode_integer_bitstream_unpack_u_t>(
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
