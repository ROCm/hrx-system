// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/base/internal/fpu_state.h"
#include "iree/testing/gtest.h"
#include "iree/vm/bytecode/interpreter_float.h"

namespace iree::vm::bytecode::testing {
namespace {

uint64_t ExecuteUnaryF32(uint8_t selector, uint32_t source) {
  uint64_t values[] = {UINT64_C(0xDEADBEEF00000000) | source};
  iree_vm_bytecode_float_math_unary_f32_t record = {
      .destination_v8 = 0,
      .source_v8 = 0,
      .selector_u8 = selector,
  };
  iree_vm_bytecode_execute_float_math_unary_f32(&record, values);
  return values[0];
}

uint64_t ExecuteUnaryF64(uint8_t selector, uint64_t source) {
  uint64_t values[] = {source};
  iree_vm_bytecode_float_math_unary_f64_t record = {
      .destination_v8 = 0,
      .source_v8 = 0,
      .selector_u8 = selector,
  };
  iree_vm_bytecode_execute_float_math_unary_f64(&record, values);
  return values[0];
}

template <typename Record, typename Execute>
uint64_t ExecuteFma(Execute execute, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t values[] = {a, b, c};
  Record record = {.destination_v8 = 0, .a_v8 = 0, .b_v8 = 1, .c_v8 = 2};
  execute(&record, values);
  return values[0];
}

class VMBytecodeInterpreterFloatMathTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fpu_state_ = iree_fpu_state_push(IREE_FPU_STATE_FLAG_MASK_EXCEPTIONS |
                                     IREE_FPU_STATE_FLAG_ROUND_TO_NEAREST);
  }

  void TearDown() override { iree_fpu_state_pop(fpu_state_); }

  // Caller floating-point control state restored after each test.
  iree_fpu_state_t fpu_state_ = {};
};

TEST_F(VMBytecodeInterpreterFloatMathTest, EvaluatesExactF32Selectors) {
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_CEIL,
                            UINT32_C(0x3FA00000)),
            UINT32_C(0x40000000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_FLOOR,
                            UINT32_C(0x3FE00000)),
            UINT32_C(0x3F800000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_ROUND_EVEN,
                            UINT32_C(0x40200000)),
            UINT32_C(0x40000000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_TRUNC,
                            UINT32_C(0xBFE00000)),
            UINT32_C(0xBF800000));

  const uint8_t rounding_selectors[] = {
      IREE_VM_BYTECODE_FLOAT_MATH_F32_CEIL,
      IREE_VM_BYTECODE_FLOAT_MATH_F32_FLOOR,
      IREE_VM_BYTECODE_FLOAT_MATH_F32_ROUND_EVEN,
      IREE_VM_BYTECODE_FLOAT_MATH_F32_TRUNC,
  };
  for (uint8_t selector : rounding_selectors) {
    EXPECT_EQ(ExecuteUnaryF32(selector, UINT32_C(0x00000000)),
              UINT32_C(0x00000000));
    EXPECT_EQ(ExecuteUnaryF32(selector, UINT32_C(0x80000000)),
              UINT32_C(0x80000000));
    EXPECT_EQ(ExecuteUnaryF32(selector, UINT32_C(0x7F800000)),
              UINT32_C(0x7F800000));
    EXPECT_EQ(ExecuteUnaryF32(selector, UINT32_C(0xFF800000)),
              UINT32_C(0xFF800000));
  }

  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_SIGN,
                            UINT32_C(0x7F800001)),
            UINT32_C(0x00000000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_SIGN,
                            UINT32_C(0x80000000)),
            UINT32_C(0x00000000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_SIGN,
                            UINT32_C(0xFF800000)),
            UINT32_C(0xBF800000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_SIGN,
                            UINT32_C(0x7F800000)),
            UINT32_C(0x3F800000));
}

TEST_F(VMBytecodeInterpreterFloatMathTest, EvaluatesExactF64Selectors) {
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_CEIL,
                            UINT64_C(0x3FF4000000000000)),
            UINT64_C(0x4000000000000000));
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_FLOOR,
                            UINT64_C(0x3FFC000000000000)),
            UINT64_C(0x3FF0000000000000));
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_ROUND_EVEN,
                            UINT64_C(0x4004000000000000)),
            UINT64_C(0x4000000000000000));
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_TRUNC,
                            UINT64_C(0xBFFC000000000000)),
            UINT64_C(0xBFF0000000000000));

  const uint8_t rounding_selectors[] = {
      IREE_VM_BYTECODE_FLOAT_MATH_F64_CEIL,
      IREE_VM_BYTECODE_FLOAT_MATH_F64_FLOOR,
      IREE_VM_BYTECODE_FLOAT_MATH_F64_ROUND_EVEN,
      IREE_VM_BYTECODE_FLOAT_MATH_F64_TRUNC,
  };
  for (uint8_t selector : rounding_selectors) {
    EXPECT_EQ(ExecuteUnaryF64(selector, UINT64_C(0x0000000000000000)),
              UINT64_C(0x0000000000000000));
    EXPECT_EQ(ExecuteUnaryF64(selector, UINT64_C(0x8000000000000000)),
              UINT64_C(0x8000000000000000));
    EXPECT_EQ(ExecuteUnaryF64(selector, UINT64_C(0x7FF0000000000000)),
              UINT64_C(0x7FF0000000000000));
    EXPECT_EQ(ExecuteUnaryF64(selector, UINT64_C(0xFFF0000000000000)),
              UINT64_C(0xFFF0000000000000));
  }

  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_SIGN,
                            UINT64_C(0x7FF0000000000001)),
            UINT64_C(0x0000000000000000));
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_SIGN,
                            UINT64_C(0x8000000000000000)),
            UINT64_C(0x0000000000000000));
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_SIGN,
                            UINT64_C(0xFFF0000000000000)),
            UINT64_C(0xBFF0000000000000));
  EXPECT_EQ(ExecuteUnaryF64(IREE_VM_BYTECODE_FLOAT_MATH_F64_SIGN,
                            UINT64_C(0x7FF0000000000000)),
            UINT64_C(0x3FF0000000000000));
}

TEST_F(VMBytecodeInterpreterFloatMathTest, SelectsFrozenF32Approximations) {
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_EXP2_APPROX,
                            UINT32_C(0x3F800B8B)),
            UINT32_C(0x40000801));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_LOG2_APPROX,
                            UINT32_C(0x70006067)),
            UINT32_C(0x42C2022B));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_RECIPROCAL_APPROX,
                            UINT32_C(0x40800000)),
            UINT32_C(0x3E800000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_RSQRT_APPROX,
                            UINT32_C(0x70000002)),
            UINT32_C(0x273504F1));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_SQRT_APPROX,
                            UINT32_C(0x6144BCA4)),
            UINT32_C(0x50606BB2));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_SIN_TURNS_APPROX,
                            UINT32_C(0x3E800000)),
            UINT32_C(0x3F800000));
  EXPECT_EQ(ExecuteUnaryF32(IREE_VM_BYTECODE_FLOAT_MATH_F32_COS_TURNS_APPROX,
                            UINT32_C(0x3E800000)),
            UINT32_C(0x00000000));
}

TEST_F(VMBytecodeInterpreterFloatMathTest, FmaRoundsOnce) {
  EXPECT_EQ(ExecuteFma<iree_vm_bytecode_float_fma_f32_t>(
                iree_vm_bytecode_execute_float_fma_f32, UINT32_C(0x3F800001),
                UINT32_C(0x3F7FFFFE), UINT32_C(0xBF800000)),
            UINT32_C(0xA8800000));
  EXPECT_EQ(
      ExecuteFma<iree_vm_bytecode_float_fma_f64_t>(
          iree_vm_bytecode_execute_float_fma_f64, UINT64_C(0x3FF0000000000001),
          UINT64_C(0x3FEFFFFFFFFFFFFE), UINT64_C(0xBFF0000000000000)),
      UINT64_C(0xB970000000000000));
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
