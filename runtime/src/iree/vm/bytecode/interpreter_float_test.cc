// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_float.h"

#include <cstdint>

#include "iree/testing/gtest.h"

namespace iree::vm::bytecode::testing {
namespace {

template <typename Record, typename Execute>
uint64_t ExecuteBinary(Execute execute, uint64_t lhs, uint64_t rhs) {
  uint64_t values[] = {lhs, rhs};
  Record record = {};
  record.dst_v8 = 0;
  record.lhs_v8 = 0;
  record.rhs_v8 = 1;
  execute(&record, values);
  return values[0];
}

template <typename Record, typename Execute>
uint64_t ExecuteUnary(Execute execute, uint64_t source) {
  uint64_t values[] = {source};
  Record record = {};
  record.dst_v8 = 0;
  record.src_v8 = 0;
  execute(&record, values);
  return values[0];
}

template <typename Record, typename Execute>
uint64_t ExecuteMinMax(Execute execute, uint64_t lhs, uint64_t rhs,
                       uint8_t selector) {
  uint64_t values[] = {lhs, rhs};
  Record record = {};
  record.dst_v8 = 1;
  record.lhs_v8 = 0;
  record.rhs_v8 = 1;
  record.selector_u8 = selector;
  execute(&record, values);
  return values[1];
}

template <typename Record, typename Execute>
uint64_t ExecuteCompare(Execute execute, uint64_t lhs, uint64_t rhs,
                        uint8_t predicate) {
  uint64_t values[] = {lhs, rhs};
  Record record = {};
  record.dst_v8 = 0;
  record.lhs_v8 = 0;
  record.rhs_v8 = 1;
  record.predicate_u8 = predicate;
  execute(&record, values);
  return values[0];
}

template <typename Record, typename Execute>
uint64_t ExecuteClassify(Execute execute, uint64_t source, uint8_t selector) {
  uint64_t values[] = {source};
  Record record = {};
  record.dst_v8 = 0;
  record.src_v8 = 0;
  record.selector_u8 = selector;
  execute(&record, values);
  return values[0];
}

template <typename Record, typename Execute>
uint64_t ExecuteClamp(Execute execute, uint64_t value, uint64_t lower,
                      uint64_t upper, uint8_t mode) {
  uint64_t values[] = {value, lower, upper};
  Record record = {};
  record.dst_v8 = 1;
  record.value_v8 = 0;
  record.lower_v8 = 1;
  record.upper_v8 = 2;
  record.mode_u8 = mode;
  execute(&record, values);
  return values[1];
}

TEST(VMBytecodeInterpreterFloatTest, ExecutesSelectedWidthArithmetic) {
  constexpr uint32_t kF32One = UINT32_C(0x3F800000);
  constexpr uint32_t kF32Half = UINT32_C(0x3F000000);
  constexpr uint32_t kF32OneAndHalf = UINT32_C(0x3FC00000);
  constexpr uint32_t kF32Two = UINT32_C(0x40000000);
  constexpr uint32_t kF32NegativeFiveAndHalf = UINT32_C(0xC0B00000);
  constexpr uint32_t kF32NegativeOneAndHalf = UINT32_C(0xBFC00000);
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_add_f32_record_t>(
                iree_vm_bytecode_execute_float_add_f32, kF32One, kF32Half),
            kF32OneAndHalf);
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_sub_f32_record_t>(
                iree_vm_bytecode_execute_float_sub_f32, kF32Two, kF32Half),
            kF32OneAndHalf);
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_mul_f32_record_t>(
                iree_vm_bytecode_execute_float_mul_f32, UINT32_C(0x00800000),
                kF32Half),
            UINT32_C(0x00400000));
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_div_f32_record_t>(
                iree_vm_bytecode_execute_float_div_f32, UINT32_C(0xBF800000),
                UINT32_C(0x7F800000)),
            UINT32_C(0x80000000));
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_rem_f32_record_t>(
                iree_vm_bytecode_execute_float_rem_f32, kF32NegativeFiveAndHalf,
                kF32Two),
            kF32NegativeOneAndHalf);
  const uint64_t f32_invalid =
      ExecuteBinary<iree_vm_isa_float_div_f32_record_t>(
          iree_vm_bytecode_execute_float_div_f32, 0, 0);
  EXPECT_EQ(f32_invalid & UINT32_C(0x7FFFFFFF), UINT32_C(0x7FC00000));

  constexpr uint64_t kF64One = UINT64_C(0x3FF0000000000000);
  constexpr uint64_t kF64Half = UINT64_C(0x3FE0000000000000);
  constexpr uint64_t kF64OneAndHalf = UINT64_C(0x3FF8000000000000);
  constexpr uint64_t kF64Two = UINT64_C(0x4000000000000000);
  constexpr uint64_t kF64NegativeFiveAndHalf = UINT64_C(0xC016000000000000);
  constexpr uint64_t kF64NegativeOneAndHalf = UINT64_C(0xBFF8000000000000);
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_add_f64_record_t>(
                iree_vm_bytecode_execute_float_add_f64, kF64One, kF64Half),
            kF64OneAndHalf);
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_sub_f64_record_t>(
                iree_vm_bytecode_execute_float_sub_f64, kF64Two, kF64Half),
            kF64OneAndHalf);
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_mul_f64_record_t>(
                iree_vm_bytecode_execute_float_mul_f64,
                UINT64_C(0x0010000000000000), kF64Half),
            UINT64_C(0x0008000000000000));
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_div_f64_record_t>(
                iree_vm_bytecode_execute_float_div_f64,
                UINT64_C(0xBFF0000000000000), UINT64_C(0x7FF0000000000000)),
            UINT64_C(0x8000000000000000));
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_rem_f64_record_t>(
                iree_vm_bytecode_execute_float_rem_f64, kF64NegativeFiveAndHalf,
                kF64Two),
            kF64NegativeOneAndHalf);
  const uint64_t f64_invalid =
      ExecuteBinary<iree_vm_isa_float_div_f64_record_t>(
          iree_vm_bytecode_execute_float_div_f64, 0, 0);
  EXPECT_EQ(f64_invalid & UINT64_C(0x7FFFFFFFFFFFFFFF),
            UINT64_C(0x7FF8000000000000));
}

TEST(VMBytecodeInterpreterFloatTest, PreservesRawSignPayloadsAndAliasing) {
  constexpr uint64_t kF32SignalingNaNWithGarbage = UINT64_C(0xDEADBEEF7F800123);
  EXPECT_EQ(
      ExecuteUnary<iree_vm_isa_float_neg_f32_record_t>(
          iree_vm_bytecode_execute_float_neg_f32, kF32SignalingNaNWithGarbage),
      UINT32_C(0xFF800123));
  EXPECT_EQ(ExecuteUnary<iree_vm_isa_float_abs_f32_record_t>(
                iree_vm_bytecode_execute_float_abs_f32, UINT32_C(0xFFC00123)),
            UINT32_C(0x7FC00123));
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_copysign_f32_record_t>(
                iree_vm_bytecode_execute_float_copysign_f32,
                UINT32_C(0x7F800123), UINT32_C(0x80000000)),
            UINT32_C(0xFF800123));

  EXPECT_EQ(
      ExecuteUnary<iree_vm_isa_float_neg_f64_record_t>(
          iree_vm_bytecode_execute_float_neg_f64, UINT64_C(0x7FF0000000000123)),
      UINT64_C(0xFFF0000000000123));
  EXPECT_EQ(
      ExecuteUnary<iree_vm_isa_float_abs_f64_record_t>(
          iree_vm_bytecode_execute_float_abs_f64, UINT64_C(0xFFF8000000000123)),
      UINT64_C(0x7FF8000000000123));
  EXPECT_EQ(ExecuteBinary<iree_vm_isa_float_copysign_f64_record_t>(
                iree_vm_bytecode_execute_float_copysign_f64,
                UINT64_C(0x7FF0000000000123), UINT64_C(0x8000000000000000)),
            UINT64_C(0xFFF0000000000123));
}

TEST(VMBytecodeInterpreterFloatTest, ExecutesExactMinMaxRules) {
  constexpr uint32_t kF32PositiveZero = 0;
  constexpr uint32_t kF32NegativeZero = UINT32_C(0x80000000);
  constexpr uint32_t kF32NegativeOne = UINT32_C(0xBF800000);
  constexpr uint32_t kF32One = UINT32_C(0x3F800000);
  constexpr uint32_t kF32SignalingNaN = UINT32_C(0x7F800123);
  using F32Record = iree_vm_isa_float_minmax_f32_record_t;
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32One, kF32NegativeOne,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINIMUM),
            kF32NegativeOne);
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32NegativeOne, kF32One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM),
            kF32One);
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32PositiveZero, kF32NegativeZero,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINIMUM),
            kF32NegativeZero);
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32NegativeZero, kF32PositiveZero,
                                     IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM),
            kF32PositiveZero);
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32SignalingNaN, kF32One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINNUM),
            kF32One);
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32SignalingNaN, kF32One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MAXNUM),
            kF32One);
  EXPECT_EQ(ExecuteMinMax<F32Record>(iree_vm_bytecode_execute_float_minmax_f32,
                                     kF32SignalingNaN, kF32One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINIMUM),
            UINT32_C(0x7FC00000));

  constexpr uint64_t kF64NegativeZero = UINT64_C(0x8000000000000000);
  constexpr uint64_t kF64NegativeOne = UINT64_C(0xBFF0000000000000);
  constexpr uint64_t kF64One = UINT64_C(0x3FF0000000000000);
  constexpr uint64_t kF64SignalingNaN = UINT64_C(0x7FF0000000000123);
  using F64Record = iree_vm_isa_float_minmax_f64_record_t;
  EXPECT_EQ(ExecuteMinMax<F64Record>(iree_vm_bytecode_execute_float_minmax_f64,
                                     kF64One, kF64NegativeOne,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINNUM),
            kF64NegativeOne);
  EXPECT_EQ(ExecuteMinMax<F64Record>(iree_vm_bytecode_execute_float_minmax_f64,
                                     kF64NegativeOne, kF64One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MAXNUM),
            kF64One);
  EXPECT_EQ(ExecuteMinMax<F64Record>(iree_vm_bytecode_execute_float_minmax_f64,
                                     0, kF64NegativeZero,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINIMUM),
            kF64NegativeZero);
  EXPECT_EQ(ExecuteMinMax<F64Record>(iree_vm_bytecode_execute_float_minmax_f64,
                                     kF64SignalingNaN, kF64One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MINNUM),
            kF64One);
  EXPECT_EQ(ExecuteMinMax<F64Record>(iree_vm_bytecode_execute_float_minmax_f64,
                                     kF64SignalingNaN, kF64One,
                                     IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM),
            UINT64_C(0x7FF8000000000000));
}

TEST(VMBytecodeInterpreterFloatTest, ExecutesAllComparisonPredicates) {
  const uint64_t less_expected[] = {0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0};
  const uint64_t greater_expected[] = {0, 1, 1, 0, 0, 1, 1,
                                       0, 1, 1, 0, 0, 1, 0};
  const uint64_t equal_expected[] = {1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0};
  const uint64_t unordered_expected[] = {0, 0, 0, 0, 0, 0, 0,
                                         1, 1, 1, 1, 1, 1, 1};
  for (uint8_t predicate = 0; predicate <= IREE_VM_ISA_FLOAT_COMPARE_UNO;
       ++predicate) {
    SCOPED_TRACE(static_cast<int>(predicate));
    EXPECT_EQ(ExecuteCompare<iree_vm_isa_float_compare_f32_record_t>(
                  iree_vm_bytecode_execute_float_compare_f32,
                  UINT32_C(0x3F800000), UINT32_C(0x40000000), predicate),
              less_expected[predicate]);
    EXPECT_EQ(ExecuteCompare<iree_vm_isa_float_compare_f32_record_t>(
                  iree_vm_bytecode_execute_float_compare_f32,
                  UINT32_C(0x40000000), UINT32_C(0x3F800000), predicate),
              greater_expected[predicate]);
    EXPECT_EQ(ExecuteCompare<iree_vm_isa_float_compare_f32_record_t>(
                  iree_vm_bytecode_execute_float_compare_f32, 0,
                  UINT32_C(0x80000000), predicate),
              equal_expected[predicate]);
    EXPECT_EQ(ExecuteCompare<iree_vm_isa_float_compare_f32_record_t>(
                  iree_vm_bytecode_execute_float_compare_f32,
                  UINT32_C(0x7F800123), UINT32_C(0x3F800000), predicate),
              unordered_expected[predicate]);
    EXPECT_EQ(ExecuteCompare<iree_vm_isa_float_compare_f64_record_t>(
                  iree_vm_bytecode_execute_float_compare_f64,
                  UINT64_C(0x3FF0000000000000), UINT64_C(0x4000000000000000),
                  predicate),
              less_expected[predicate]);
    EXPECT_EQ(ExecuteCompare<iree_vm_isa_float_compare_f64_record_t>(
                  iree_vm_bytecode_execute_float_compare_f64,
                  UINT64_C(0x7FF0000000000123), UINT64_C(0x3FF0000000000000),
                  predicate),
              unordered_expected[predicate]);
  }
}

TEST(VMBytecodeInterpreterFloatTest, ClassifiesRawSpecialValues) {
  struct F32Case {
    // Raw f32 payload to classify.
    uint32_t bits;
    // Expected isnan, isinf, and isfinite results in selector order.
    uint64_t expected[3];
  };
  const F32Case f32_cases[] = {
      {0, {0, 0, 1}},
      {1, {0, 0, 1}},
      {UINT32_C(0x7F7FFFFF), {0, 0, 1}},
      {UINT32_C(0x7F800000), {0, 1, 0}},
      {UINT32_C(0x7FC00123), {1, 0, 0}},
      {UINT32_C(0x7F800123), {1, 0, 0}},
  };
  for (const F32Case& test_case : f32_cases) {
    for (uint8_t selector = 0; selector <= IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE;
         ++selector) {
      EXPECT_EQ(ExecuteClassify<iree_vm_isa_float_classify_f32_record_t>(
                    iree_vm_bytecode_execute_float_classify_f32, test_case.bits,
                    selector),
                test_case.expected[selector]);
    }
  }

  EXPECT_EQ(
      ExecuteClassify<iree_vm_isa_float_classify_f64_record_t>(
          iree_vm_bytecode_execute_float_classify_f64,
          UINT64_C(0x0000000000000001), IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE),
      1u);
  EXPECT_EQ(ExecuteClassify<iree_vm_isa_float_classify_f64_record_t>(
                iree_vm_bytecode_execute_float_classify_f64,
                UINT64_C(0xFFF0000000000000), IREE_VM_ISA_FLOAT_CLASSIFY_ISINF),
            1u);
  EXPECT_EQ(ExecuteClassify<iree_vm_isa_float_classify_f64_record_t>(
                iree_vm_bytecode_execute_float_classify_f64,
                UINT64_C(0x7FF0000000000123), IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN),
            1u);
}

TEST(VMBytecodeInterpreterFloatTest, ExecutesClampCompositions) {
  constexpr uint32_t kF32NegativeOne = UINT32_C(0xBF800000);
  constexpr uint32_t kF32PositiveZero = 0;
  constexpr uint32_t kF32One = UINT32_C(0x3F800000);
  constexpr uint32_t kF32PayloadNaN = UINT32_C(0x7FC00123);
  using F32Record = iree_vm_isa_float_clamp_f32_record_t;
  EXPECT_EQ(ExecuteClamp<F32Record>(iree_vm_bytecode_execute_float_clamp_f32,
                                    kF32PayloadNaN, kF32PositiveZero, kF32One,
                                    IREE_VM_ISA_FLOAT_CLAMP_ORDERED),
            kF32PayloadNaN);
  EXPECT_EQ(ExecuteClamp<F32Record>(iree_vm_bytecode_execute_float_clamp_f32,
                                    kF32NegativeOne, kF32PositiveZero, kF32One,
                                    IREE_VM_ISA_FLOAT_CLAMP_ORDERED),
            kF32PositiveZero);
  EXPECT_EQ(ExecuteClamp<F32Record>(iree_vm_bytecode_execute_float_clamp_f32,
                                    kF32PayloadNaN, kF32PositiveZero, kF32One,
                                    IREE_VM_ISA_FLOAT_CLAMP_NUMBER),
            kF32PositiveZero);
  EXPECT_EQ(ExecuteClamp<F32Record>(iree_vm_bytecode_execute_float_clamp_f32,
                                    kF32PayloadNaN, kF32PositiveZero, kF32One,
                                    IREE_VM_ISA_FLOAT_CLAMP_IEEE),
            UINT32_C(0x7FC00000));

  using F64Record = iree_vm_isa_float_clamp_f64_record_t;
  EXPECT_EQ(ExecuteClamp<F64Record>(iree_vm_bytecode_execute_float_clamp_f64,
                                    UINT64_C(0xBFF0000000000000), 0,
                                    UINT64_C(0x3FF0000000000000),
                                    IREE_VM_ISA_FLOAT_CLAMP_ORDERED),
            0u);
  EXPECT_EQ(ExecuteClamp<F64Record>(iree_vm_bytecode_execute_float_clamp_f64,
                                    UINT64_C(0x7FF8000000000123), 0,
                                    UINT64_C(0x3FF0000000000000),
                                    IREE_VM_ISA_FLOAT_CLAMP_IEEE),
            UINT64_C(0x7FF8000000000000));
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
