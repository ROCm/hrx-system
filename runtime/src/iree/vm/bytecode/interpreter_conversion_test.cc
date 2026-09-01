// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_conversion.h"

#include <cstdint>
#include <cstring>

#include "iree/base/internal/math.h"
#include "iree/testing/gtest.h"

namespace iree::vm::bytecode::testing {
namespace {

typedef struct ConversionTestRow {
  // Verified selector value.
  uint8_t selector;
  // Complete source value-cell bits.
  uint64_t source;
  // Expected canonical destination bits.
  uint64_t expected;
} ConversionTestRow;

#define IREE_VM_BYTECODE_INTEGER_TEST_ROW(selector, source, expected) \
  {(selector), (source), (expected)},
static const ConversionTestRow kIntegerTestRows[] = {
#define IREE_VM_BYTECODE_DEFINE_INTEGER_TEST_ROWS
#include "iree/vm/bytecode/conversion_test_vectors.inl"
#undef IREE_VM_BYTECODE_DEFINE_INTEGER_TEST_ROWS
};
#undef IREE_VM_BYTECODE_INTEGER_TEST_ROW

#define IREE_VM_BYTECODE_FLOAT_TRUNCATE_TEST_ROW(selector, source, expected) \
  {(selector), (source), (expected)},
static const ConversionTestRow kFloatTruncateTestRows[] = {
#define IREE_VM_BYTECODE_DEFINE_FLOAT_TRUNCATE_TEST_ROWS
#include "iree/vm/bytecode/conversion_test_vectors.inl"
#undef IREE_VM_BYTECODE_DEFINE_FLOAT_TRUNCATE_TEST_ROWS
};
#undef IREE_VM_BYTECODE_FLOAT_TRUNCATE_TEST_ROW

#define IREE_VM_BYTECODE_FLOAT_WIDTH_TEST_ROW(selector, source, expected) \
  {(selector), (source), (expected)},
static const ConversionTestRow kFloatWidthTestRows[] = {
#define IREE_VM_BYTECODE_DEFINE_FLOAT_WIDTH_TEST_ROWS
#include "iree/vm/bytecode/conversion_test_vectors.inl"
#undef IREE_VM_BYTECODE_DEFINE_FLOAT_WIDTH_TEST_ROWS
};
#undef IREE_VM_BYTECODE_FLOAT_WIDTH_TEST_ROW

#define IREE_VM_BYTECODE_INTEGER_TO_FLOAT_TEST_ROW(selector, source, expected) \
  {(selector), (source), (expected)},
static const ConversionTestRow kIntegerToFloatTestRows[] = {
#define IREE_VM_BYTECODE_DEFINE_INTEGER_TO_FLOAT_TEST_ROWS
#include "iree/vm/bytecode/conversion_test_vectors.inl"
#undef IREE_VM_BYTECODE_DEFINE_INTEGER_TO_FLOAT_TEST_ROWS
};
#undef IREE_VM_BYTECODE_INTEGER_TO_FLOAT_TEST_ROW

typedef struct FloatToIntegerTestRow {
  // Verified selector value.
  uint8_t selector;
  // Complete source value-cell bits.
  uint64_t source;
  // Expected execution failure classification.
  iree_vm_bytecode_conversion_failure_t expected_failure;
  // Expected canonical destination bits on success.
  uint64_t expected;
} FloatToIntegerTestRow;

#define IREE_VM_BYTECODE_FLOAT_TO_INTEGER_TEST_ROW(selector, source,           \
                                                   expected_failure, expected) \
  {(selector), (source),                                                       \
   IREE_VM_BYTECODE_CONVERSION_FAILURE_##expected_failure, (expected)},
static const FloatToIntegerTestRow kFloatToIntegerTestRows[] = {
#define IREE_VM_BYTECODE_DEFINE_FLOAT_TO_INTEGER_TEST_ROWS
#include "iree/vm/bytecode/conversion_test_vectors.inl"
#undef IREE_VM_BYTECODE_DEFINE_FLOAT_TO_INTEGER_TEST_ROWS
};
#undef IREE_VM_BYTECODE_FLOAT_TO_INTEGER_TEST_ROW

template <typename Record, typename Execute>
uint64_t ExecuteConversion(Execute execute, uint8_t selector, uint64_t source) {
  uint64_t values[] = {source, UINT64_C(0xA55AA55AA55AA55A)};
  Record record = {};
  record.dst_v8 = 1;
  record.src_v8 = 0;
  record.selector_u8 = selector;
  execute(&record, values);
  return values[1];
}

static bool IsF32NaN(uint32_t bits) {
  return (bits & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000);
}

static bool IsF8E4M3NaN(uint8_t bits) {
  return (bits & UINT8_C(0x7F)) == UINT8_C(0x7F);
}

static bool IsF8E5M2NaN(uint8_t bits) {
  return (bits & UINT8_C(0x7F)) > UINT8_C(0x7C);
}

static bool IsF16NaN(uint16_t bits) {
  return (bits & UINT16_C(0x7FFF)) > UINT16_C(0x7C00);
}

static bool IsBF16NaN(uint16_t bits) {
  return (bits & UINT16_C(0x7FFF)) > UINT16_C(0x7F80);
}

TEST(VMBytecodeInterpreterConversionTest, ExecutesEveryIntegerSelector) {
  bool selector_seen[IREE_VM_ISA_INTEGER_CONVERT_I64_TO_I32 + 1] = {};
  for (const ConversionTestRow& row : kIntegerTestRows) {
    SCOPED_TRACE(static_cast<int>(row.selector));
    ASSERT_LT(row.selector, IREE_ARRAYSIZE(selector_seen));
    selector_seen[row.selector] = true;
    EXPECT_EQ(ExecuteConversion<iree_vm_isa_conversion_integer_record_t>(
                  iree_vm_bytecode_execute_conversion_integer, row.selector,
                  row.source),
              row.expected);
  }
  for (bool was_seen : selector_seen) {
    EXPECT_TRUE(was_seen);
  }
}

TEST(VMBytecodeInterpreterConversionTest,
     ExtendsEveryNarrowPayloadStructurally) {
  for (uint32_t bits = 0; bits < 256; ++bits) {
    const uint32_t actual_e4 =
        ExecuteConversion<iree_vm_isa_conversion_float_extend_record_t>(
            iree_vm_bytecode_execute_conversion_float_extend,
            IREE_VM_ISA_FLOAT_EXTEND_F8E4M3_TO_F32, bits);
    const uint32_t expected_e4 = iree_math_f8e4m3fn_to_f32_bits(bits);
    EXPECT_TRUE(actual_e4 == expected_e4 ||
                (IsF32NaN(actual_e4) && IsF32NaN(expected_e4)));

    const uint32_t actual_e5 =
        ExecuteConversion<iree_vm_isa_conversion_float_extend_record_t>(
            iree_vm_bytecode_execute_conversion_float_extend,
            IREE_VM_ISA_FLOAT_EXTEND_F8E5M2_TO_F32, bits);
    const uint32_t expected_e5 = iree_math_f8e5m2_to_f32_bits(bits);
    EXPECT_TRUE(actual_e5 == expected_e5 ||
                (IsF32NaN(actual_e5) && IsF32NaN(expected_e5)));
  }
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    const uint32_t actual_f16 =
        ExecuteConversion<iree_vm_isa_conversion_float_extend_record_t>(
            iree_vm_bytecode_execute_conversion_float_extend,
            IREE_VM_ISA_FLOAT_EXTEND_F16_TO_F32, bits);
    const uint32_t expected_f16 = iree_math_f16_to_f32_bits(bits);
    EXPECT_TRUE(actual_f16 == expected_f16 ||
                (IsF32NaN(actual_f16) && IsF32NaN(expected_f16)));

    const uint32_t actual_bf16 =
        ExecuteConversion<iree_vm_isa_conversion_float_extend_record_t>(
            iree_vm_bytecode_execute_conversion_float_extend,
            IREE_VM_ISA_FLOAT_EXTEND_BF16_TO_F32, bits);
    const uint32_t expected_bf16 = iree_math_bf16_to_f32_bits(bits);
    EXPECT_TRUE(actual_bf16 == expected_bf16 ||
                (IsF32NaN(actual_bf16) && IsF32NaN(expected_bf16)));
  }
}

TEST(VMBytecodeInterpreterConversionTest, RoundsEveryFloatBoundaryWitness) {
  bool truncate_selector_seen[IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_BF16 + 1] = {};
  for (const ConversionTestRow& row : kFloatTruncateTestRows) {
    SCOPED_TRACE(static_cast<int>(row.selector));
    ASSERT_LT(row.selector, IREE_ARRAYSIZE(truncate_selector_seen));
    truncate_selector_seen[row.selector] = true;
    EXPECT_EQ(ExecuteConversion<iree_vm_isa_conversion_float_truncate_record_t>(
                  iree_vm_bytecode_execute_conversion_float_truncate,
                  row.selector, row.source),
              row.expected);
  }
  for (bool selector_seen : truncate_selector_seen) {
    EXPECT_TRUE(selector_seen);
  }

  bool width_selector_seen[IREE_VM_ISA_FLOAT_WIDTH_F64_TO_F32 + 1] = {};
  for (const ConversionTestRow& row : kFloatWidthTestRows) {
    SCOPED_TRACE(static_cast<int>(row.selector));
    ASSERT_LT(row.selector, IREE_ARRAYSIZE(width_selector_seen));
    width_selector_seen[row.selector] = true;
    EXPECT_EQ(ExecuteConversion<iree_vm_isa_conversion_float_width_record_t>(
                  iree_vm_bytecode_execute_conversion_float_width, row.selector,
                  row.source),
              row.expected);
  }
  for (bool selector_seen : width_selector_seen) {
    EXPECT_TRUE(selector_seen);
  }
}

TEST(VMBytecodeInterpreterConversionTest,
     MatchesEstablishedF32NarrowingAcrossPayloadSpace) {
  constexpr uint32_t kSampleCount = 1u << 18;
  constexpr uint32_t kPermutationMultiplier = UINT32_C(0x9E3779B1);
  for (uint32_t i = 0; i < kSampleCount; ++i) {
    const uint32_t source_bits = i * kPermutationMultiplier;
    float source = 0.0f;
    std::memcpy(&source, &source_bits, sizeof(source));
    const bool source_is_nan = IsF32NaN(source_bits);

    const uint8_t actual_e4 =
        ExecuteConversion<iree_vm_isa_conversion_float_truncate_record_t>(
            iree_vm_bytecode_execute_conversion_float_truncate,
            IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F8E4M3, source_bits);
    const uint8_t expected_e4 = iree_math_f32_to_f8e4m3fn(source);
    EXPECT_TRUE(source_is_nan ? IsF8E4M3NaN(actual_e4)
                              : actual_e4 == expected_e4)
        << source_bits;

    const uint8_t actual_e5 =
        ExecuteConversion<iree_vm_isa_conversion_float_truncate_record_t>(
            iree_vm_bytecode_execute_conversion_float_truncate,
            IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F8E5M2, source_bits);
    const uint8_t expected_e5 = iree_math_f32_to_f8e5m2(source);
    EXPECT_TRUE(source_is_nan ? IsF8E5M2NaN(actual_e5)
                              : actual_e5 == expected_e5)
        << source_bits;

    const uint16_t actual_f16 =
        ExecuteConversion<iree_vm_isa_conversion_float_truncate_record_t>(
            iree_vm_bytecode_execute_conversion_float_truncate,
            IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_F16, source_bits);
    const uint16_t expected_f16 = iree_math_f32_to_f16(source);
    EXPECT_TRUE(source_is_nan ? IsF16NaN(actual_f16)
                              : actual_f16 == expected_f16)
        << source_bits;

    const uint16_t actual_bf16 =
        ExecuteConversion<iree_vm_isa_conversion_float_truncate_record_t>(
            iree_vm_bytecode_execute_conversion_float_truncate,
            IREE_VM_ISA_FLOAT_TRUNCATE_F32_TO_BF16, source_bits);
    const uint16_t expected_bf16 = iree_math_f32_to_bf16(source);
    EXPECT_TRUE(source_is_nan ? IsBF16NaN(actual_bf16)
                              : actual_bf16 == expected_bf16)
        << source_bits;
  }
}

TEST(VMBytecodeInterpreterConversionTest, RoundsIntegersDirectly) {
  bool selector_seen[IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_BF16 + 1] = {};
  for (const ConversionTestRow& row : kIntegerToFloatTestRows) {
    SCOPED_TRACE(static_cast<int>(row.selector));
    ASSERT_LT(row.selector, IREE_ARRAYSIZE(selector_seen));
    selector_seen[row.selector] = true;
    EXPECT_EQ(
        ExecuteConversion<iree_vm_isa_conversion_integer_to_float_record_t>(
            iree_vm_bytecode_execute_conversion_integer_to_float, row.selector,
            row.source),
        row.expected);
  }
  for (bool was_seen : selector_seen) {
    EXPECT_TRUE(was_seen);
  }

  constexpr uint64_t kDoubleRoundingWitness = UINT64_C(2172649471);
  const uint64_t direct =
      ExecuteConversion<iree_vm_isa_conversion_integer_to_float_record_t>(
          iree_vm_bytecode_execute_conversion_integer_to_float,
          IREE_VM_ISA_INTEGER_TO_FLOAT_U64_TO_BF16, kDoubleRoundingWitness);
  const float staged_f32 = static_cast<float>(kDoubleRoundingWitness);
  EXPECT_EQ(direct, UINT16_C(0x4F01));
  EXPECT_EQ(iree_math_f32_to_bf16(staged_f32), UINT16_C(0x4F02));
}

TEST(VMBytecodeInterpreterConversionTest,
     ValidatesFloatToIntegerBeforePublishing) {
  constexpr uint64_t kSentinel = UINT64_C(0xA55AA55AA55AA55A);
  bool selector_seen[IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_U64 + 1] = {};
  for (const FloatToIntegerTestRow& row : kFloatToIntegerTestRows) {
    SCOPED_TRACE(static_cast<int>(row.selector));
    ASSERT_LT(row.selector, IREE_ARRAYSIZE(selector_seen));
    selector_seen[row.selector] = true;
    uint64_t values[] = {row.source, kSentinel};
    iree_vm_isa_conversion_float_to_integer_record_t record = {};
    record.dst_v8 = 1;
    record.src_v8 = 0;
    record.selector_u8 = row.selector;
    EXPECT_EQ(
        iree_vm_bytecode_execute_conversion_float_to_integer(&record, values),
        row.expected_failure);
    EXPECT_EQ(values[1],
              row.expected_failure == IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE
                  ? row.expected
                  : kSentinel);
  }
  for (bool was_seen : selector_seen) {
    EXPECT_TRUE(was_seen);
  }
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
