// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/variant.h"

#include <cstdint>
#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/wire/module.h"

namespace {

IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_NONE,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_INVALID,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_I8,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_I8,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_I16,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_I16,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_I32,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_I32,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_I64,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_I64,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_F8E4M3FN,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_F8E4M3FN,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_F8E5M2,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_F8E5M2,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_F16,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_F16,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_BF16,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_BF16,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_F32,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_F32,
                           "VM scalar type must match bytecode signature kind");
IREE_STATIC_ASSERT_ENUM_EQ(IREE_VM_SCALAR_TYPE_F64,
                           IREE_VM_BYTECODE_SIGNATURE_KIND_F64,
                           "VM scalar type must match bytecode signature kind");

struct TestObject {
  // Required offset-zero VM ref-count prefix.
  iree_vm_ref_object_t ref_object;
  // Number of final destruction callbacks observed by the test.
  int* destroy_count;
  // Payload used to verify typed extraction.
  int value;
};

struct TestTypes {
  // Type used for successful typed conversions.
  iree_vm_ref_type_t object;
  // Distinct type used for mismatch tests.
  iree_vm_ref_type_t other;
};

extern const iree_vm_ref_type_table_t kTestTable;

void DestroyTestObject(void* object) {
  TestObject* test_object = static_cast<TestObject*>(object);
  ++*test_object->destroy_count;
  delete test_object;
}

const iree_vm_ref_type_descriptor_t kTestObjectType = {
    DestroyTestObject,
    &kTestTable,
    IREE_SV("object"),
};
const iree_vm_ref_type_descriptor_t kTestOtherType = {
    nullptr,
    &kTestTable,
    IREE_SV("other"),
};
const TestTypes kTestTypes = {
    &kTestObjectType,
    &kTestOtherType,
};
const iree_vm_ref_type_table_t kTestTable = {
    sizeof(kTestTable),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("variant_test"),
    {&kTestTypes, 2},
};

IREE_VM_DEFINE_TYPE_ADAPTERS(test_object, TestTypes, object, TestObject)

TestObject* CreateTestObject(int* destroy_count) {
  TestObject* object = new TestObject;
  iree_vm_ref_object_initialize(&object->ref_object);
  object->destroy_count = destroy_count;
  object->value = 42;
  return object;
}

TEST(VMVariantTest, CanonicalSentinelsAreDistinct) {
  iree_vm_variant_t null_ref = iree_vm_variant_null();
  EXPECT_EQ(null_ref.payload, 0u);
  EXPECT_EQ(null_ref.metadata, 0u);
  EXPECT_TRUE(iree_vm_variant_is_null(null_ref));
  EXPECT_TRUE(iree_vm_variant_is_ref(null_ref));
  EXPECT_FALSE(iree_vm_variant_is_function_ref(null_ref));

  iree_vm_variant_t null_function =
      iree_vm_variant_from_function_ref(iree_vm_function_ref_null());
  EXPECT_EQ(null_function.payload, 0u);
  EXPECT_EQ(null_function.metadata, IREE_VM_VARIANT_TAG_FUNCTION_REF);
  EXPECT_TRUE(iree_vm_variant_is_null(null_function));
  EXPECT_TRUE(iree_vm_variant_is_function_ref(null_function));
  EXPECT_FALSE(iree_vm_variant_is_ref(null_function));

  iree_vm_variant_t empty = iree_vm_variant_empty();
  EXPECT_EQ(empty.payload, 0u);
  EXPECT_EQ(empty.metadata, IREE_VM_VARIANT_TAG_SCALAR);
  EXPECT_TRUE(iree_vm_variant_is_empty(empty));
  EXPECT_FALSE(iree_vm_variant_is_null(empty));
  EXPECT_FALSE(iree_vm_variant_is_scalar(empty));
}

TEST(VMVariantTest, FunctionCarrierSentinelsAreCanonical) {
  iree_vm_function_ref_t function_ref = iree_vm_function_ref_null();
  EXPECT_TRUE(iree_vm_function_ref_is_null(function_ref));
  iree_vm_function_ref_t extracted_ref = {
      UINT64_C(0x1111),
      UINT64_C(0x2222),
  };
  IREE_EXPECT_OK(iree_vm_function_ref_from_variant(
      iree_vm_variant_from_function_ref(function_ref), &extracted_ref));
  EXPECT_TRUE(iree_vm_function_ref_is_null(extracted_ref));

  iree_vm_function_t function = iree_vm_function_null();
  EXPECT_TRUE(iree_vm_function_is_null(function));
  function.process_bits = UINT64_C(0x1234);
  EXPECT_FALSE(iree_vm_function_is_null(function));
}

TEST(VMVariantTest, FixedIntegerScalarsRoundTripExactBits) {
  int8_t i8 = 0;
  IREE_EXPECT_OK(iree_vm_i8_from_variant(iree_vm_variant_from_i8(-1), &i8));
  EXPECT_EQ(i8, -1);

  int16_t i16 = 0;
  IREE_EXPECT_OK(iree_vm_i16_from_variant(iree_vm_variant_from_i16(-2), &i16));
  EXPECT_EQ(i16, -2);

  int32_t i32 = 0;
  IREE_EXPECT_OK(iree_vm_i32_from_variant(iree_vm_variant_from_i32(-3), &i32));
  EXPECT_EQ(i32, -3);

  int64_t i64 = 0;
  IREE_EXPECT_OK(
      iree_vm_i64_from_variant(iree_vm_variant_from_i64(INT64_MIN), &i64));
  EXPECT_EQ(i64, INT64_MIN);
}

TEST(VMVariantTest, FloatScalarsPreserveObjectRepresentation) {
  const uint32_t f32_bits = 0x7FC12345u;
  float f32 = 0.0f;
  std::memcpy(&f32, &f32_bits, sizeof(f32));
  float f32_result = 0.0f;
  IREE_EXPECT_OK(
      iree_vm_f32_from_variant(iree_vm_variant_from_f32(f32), &f32_result));
  uint32_t f32_result_bits = 0;
  std::memcpy(&f32_result_bits, &f32_result, sizeof(f32_result_bits));
  EXPECT_EQ(f32_result_bits, f32_bits);

  const uint64_t f64_bits = UINT64_C(0x7FF8123456789ABC);
  double f64 = 0.0;
  std::memcpy(&f64, &f64_bits, sizeof(f64));
  double f64_result = 0.0;
  IREE_EXPECT_OK(
      iree_vm_f64_from_variant(iree_vm_variant_from_f64(f64), &f64_result));
  uint64_t f64_result_bits = 0;
  std::memcpy(&f64_result_bits, &f64_result, sizeof(f64_result_bits));
  EXPECT_EQ(f64_result_bits, f64_bits);

  uint8_t f8_bits = 0;
  IREE_EXPECT_OK(iree_vm_f8e4m3fn_bits_from_variant(
      iree_vm_variant_from_f8e4m3fn_bits(0xFF), &f8_bits));
  EXPECT_EQ(f8_bits, 0xFF);
  IREE_EXPECT_OK(iree_vm_f8e5m2_bits_from_variant(
      iree_vm_variant_from_f8e5m2_bits(0xFE), &f8_bits));
  EXPECT_EQ(f8_bits, 0xFE);

  uint16_t f16_bits = 0;
  IREE_EXPECT_OK(iree_vm_f16_bits_from_variant(
      iree_vm_variant_from_f16_bits(0x7E01), &f16_bits));
  EXPECT_EQ(f16_bits, 0x7E01);
  IREE_EXPECT_OK(iree_vm_bf16_bits_from_variant(
      iree_vm_variant_from_bf16_bits(0x7FC1), &f16_bits));
  EXPECT_EQ(f16_bits, 0x7FC1);
}

TEST(VMVariantTest, DynamicScalarConstructionCanonicalizesNarrowBits) {
  struct Case {
    // Scalar type under test.
    iree_vm_scalar_type_t type;
    // Canonical payload after narrowing UINT64_MAX.
    uint64_t expected_bits;
  };
  const Case cases[] = {
      {IREE_VM_SCALAR_TYPE_I8, 0xFF},
      {IREE_VM_SCALAR_TYPE_F8E4M3FN, 0xFF},
      {IREE_VM_SCALAR_TYPE_F8E5M2, 0xFF},
      {IREE_VM_SCALAR_TYPE_I16, 0xFFFF},
      {IREE_VM_SCALAR_TYPE_F16, 0xFFFF},
      {IREE_VM_SCALAR_TYPE_BF16, 0xFFFF},
      {IREE_VM_SCALAR_TYPE_I32, 0xFFFFFFFF},
      {IREE_VM_SCALAR_TYPE_F32, 0xFFFFFFFF},
      {IREE_VM_SCALAR_TYPE_I64, UINT64_MAX},
      {IREE_VM_SCALAR_TYPE_F64, UINT64_MAX},
  };
  for (const Case& test_case : cases) {
    iree_vm_variant_t variant = iree_vm_variant_empty();
    IREE_EXPECT_OK(
        iree_vm_variant_from_scalar_bits(test_case.type, UINT64_MAX, &variant));
    EXPECT_EQ(iree_vm_variant_scalar_type(variant), test_case.type);
    EXPECT_EQ(variant.payload, test_case.expected_bits);
    uint64_t bits = 0;
    IREE_EXPECT_OK(
        iree_vm_scalar_bits_from_variant(variant, test_case.type, &bits));
    EXPECT_EQ(bits, test_case.expected_bits);

    variant.payload = UINT64_MAX;
    IREE_EXPECT_OK(
        iree_vm_scalar_bits_from_variant(variant, test_case.type, &bits));
    EXPECT_EQ(bits, test_case.expected_bits);
  }
}

TEST(VMVariantTest, ScalarFailuresLeaveOutputsUntouched) {
  iree_vm_variant_t output = {0x1111, 0x2222};
  const iree_vm_variant_t original_output = output;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_variant_from_scalar_bits(IREE_VM_SCALAR_TYPE_NONE, 0, &output));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_variant_from_scalar_bits(
                            (iree_vm_scalar_type_t)UINT8_MAX, 0, &output));
  EXPECT_EQ(output.payload, original_output.payload);
  EXPECT_EQ(output.metadata, original_output.metadata);

  int32_t value = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_i32_from_variant(iree_vm_variant_from_i64(1), &value));
  EXPECT_EQ(value, 123);
}

TEST(VMVariantTest, FunctionRefRoundTripsWithoutOwnership) {
  iree_vm_function_ref_t source = {
      UINT64_C(0x12345678),
      UINT64_C(0xFEDCBA9876543200),
  };
  iree_vm_variant_t variant = iree_vm_variant_from_function_ref(source);
  EXPECT_TRUE(iree_vm_variant_is_function_ref(variant));
  iree_vm_function_ref_t result = iree_vm_function_ref_null();
  IREE_EXPECT_OK(iree_vm_function_ref_from_variant(variant, &result));
  EXPECT_EQ(result.program_bits, source.program_bits);
  EXPECT_EQ(result.target_bits, source.target_bits);

  result = source;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_function_ref_from_variant(iree_vm_variant_from_i32(1), &result));
  EXPECT_EQ(result.program_bits, source.program_bits);
  EXPECT_EQ(result.target_bits, source.target_bits);
}

TEST(VMVariantRefTest, ConstructionOwnershipTransitionsAreExact) {
  int destroy_count = 0;

  TestObject* borrowed_object = CreateTestObject(&destroy_count);
  iree_vm_variant_t borrowed =
      test_object_variant_from_ptr_borrowed(&kTestTypes, borrowed_object);
  EXPECT_TRUE(test_object_variant_isa(&kTestTypes, borrowed));
  iree_vm_variant_reset(&borrowed);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_object_release(borrowed_object, &kTestObjectType);
  EXPECT_EQ(destroy_count, 1);

  TestObject* retained_object = CreateTestObject(&destroy_count);
  iree_vm_variant_t retained =
      test_object_variant_from_ptr_retained(&kTestTypes, retained_object);
  iree_vm_ref_object_release(retained_object, &kTestObjectType);
  EXPECT_EQ(destroy_count, 1);
  iree_vm_variant_reset(&retained);
  EXPECT_EQ(destroy_count, 2);

  TestObject* moved_object = CreateTestObject(&destroy_count);
  iree_vm_variant_t moved =
      test_object_variant_from_ptr_move(&kTestTypes, &moved_object);
  EXPECT_EQ(moved_object, nullptr);
  iree_vm_variant_reset(&moved);
  EXPECT_EQ(destroy_count, 3);
}

TEST(VMVariantRefTest, RefToVariantTransitionsAreExact) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_ref_t owner = test_object_ref_from_ptr_move(&kTestTypes, &object);

  iree_vm_variant_t borrowed = iree_vm_variant_from_ref_borrowed(owner);
  iree_vm_variant_t retained = iree_vm_variant_from_ref_retained(owner);
  iree_vm_variant_t moved = iree_vm_variant_from_ref_move(&owner);
  EXPECT_TRUE(iree_vm_ref_is_null(owner));

  iree_vm_variant_reset(&borrowed);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_variant_reset(&retained);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_variant_reset(&moved);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMVariantRefTest, MovingBorrowPromotesBeforeEscape) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_variant_t borrowed =
      test_object_variant_from_ptr_borrowed(&kTestTypes, object);
  iree_vm_variant_t promoted = iree_vm_variant_move(&borrowed);
  EXPECT_TRUE(iree_vm_variant_is_empty(borrowed));
  iree_vm_ref_object_release(object, &kTestObjectType);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_variant_reset(&promoted);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMVariantRefTest, RetainingRefsCreatesIndependentOwners) {
  int destroy_count = 0;

  TestObject* owned_object = CreateTestObject(&destroy_count);
  iree_vm_variant_t owned =
      test_object_variant_from_ptr_move(&kTestTypes, &owned_object);
  iree_vm_variant_t owned_copy = iree_vm_variant_retain(owned);
  iree_vm_variant_reset(&owned);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_variant_reset(&owned_copy);
  EXPECT_EQ(destroy_count, 1);

  TestObject* borrowed_object = CreateTestObject(&destroy_count);
  iree_vm_variant_t borrowed =
      test_object_variant_from_ptr_borrowed(&kTestTypes, borrowed_object);
  iree_vm_variant_t retained = iree_vm_variant_retain(borrowed);
  iree_vm_ref_object_release(borrowed_object, &kTestObjectType);
  EXPECT_EQ(destroy_count, 1);
  iree_vm_variant_reset(&borrowed);
  EXPECT_EQ(destroy_count, 1);
  iree_vm_variant_reset(&retained);
  EXPECT_EQ(destroy_count, 2);
}

TEST(VMVariantRefTest, MovingOwnedRefTransfersExistingOwner) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_variant_t owner =
      test_object_variant_from_ptr_move(&kTestTypes, &object);
  iree_vm_variant_t moved = iree_vm_variant_move(&owner);
  EXPECT_TRUE(iree_vm_variant_is_empty(owner));
  iree_vm_variant_reset(&moved);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMVariantRefTest, CheckedExtractionTransitionsAreExact) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_variant_t owner =
      test_object_variant_from_ptr_move(&kTestTypes, &object);

  void* raw_output = reinterpret_cast<void*>(uintptr_t{0x1234});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_ptr_from_variant_move(&owner, &kTestOtherType, &raw_output));
  EXPECT_EQ(raw_output, reinterpret_cast<void*>(uintptr_t{0x1234}));
  EXPECT_FALSE(iree_vm_variant_is_empty(owner));

  TestObject* output = nullptr;
  IREE_EXPECT_OK(
      test_object_ptr_from_variant_borrowed(&kTestTypes, owner, &output));
  EXPECT_EQ(output->value, 42);

  output = nullptr;
  IREE_EXPECT_OK(
      test_object_ptr_from_variant_retained(&kTestTypes, owner, &output));
  iree_vm_ref_object_release(output, &kTestObjectType);
  EXPECT_EQ(destroy_count, 0);

  output = nullptr;
  IREE_EXPECT_OK(
      test_object_ptr_from_variant_move(&kTestTypes, &owner, &output));
  EXPECT_TRUE(iree_vm_variant_is_empty(owner));
  iree_vm_ref_object_release(output, &kTestObjectType);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMVariantRefTest, TypedExtractionRejectsNullOutputWithoutMutation) {
  int destroy_count = 0;
  TestObject* ref_object = CreateTestObject(&destroy_count);
  iree_vm_ref_t ref = test_object_ref_from_ptr_move(&kTestTypes, &ref_object);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      test_object_ptr_from_ref_retained(&kTestTypes, ref, nullptr));
  EXPECT_EQ(destroy_count, 0);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      test_object_ptr_from_ref_move(&kTestTypes, &ref, nullptr));
  EXPECT_FALSE(iree_vm_ref_is_null(ref));
  EXPECT_EQ(destroy_count, 0);

  iree_vm_ref_reset(&ref);
  EXPECT_EQ(destroy_count, 1);

  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_variant_t owner =
      test_object_variant_from_ptr_move(&kTestTypes, &object);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      test_object_ptr_from_variant_retained(&kTestTypes, owner, nullptr));
  EXPECT_EQ(destroy_count, 1);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      test_object_ptr_from_variant_move(&kTestTypes, &owner, nullptr));
  EXPECT_FALSE(iree_vm_variant_is_empty(owner));
  EXPECT_EQ(destroy_count, 1);

  iree_vm_variant_reset(&owner);
  EXPECT_EQ(destroy_count, 2);
}

TEST(VMVariantRefTest, RefExtractionTransitionsAreExact) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_variant_t owner =
      test_object_variant_from_ptr_move(&kTestTypes, &object);

  iree_vm_ref_t borrowed = iree_vm_ref_null();
  IREE_EXPECT_OK(iree_vm_ref_from_variant_borrowed(owner, &borrowed));
  iree_vm_ref_t retained = iree_vm_ref_null();
  IREE_EXPECT_OK(iree_vm_ref_from_variant_retained(owner, &retained));
  iree_vm_ref_t moved = iree_vm_ref_null();
  IREE_EXPECT_OK(iree_vm_ref_from_variant_move(&owner, &moved));
  EXPECT_TRUE(iree_vm_variant_is_empty(owner));

  iree_vm_ref_reset(&borrowed);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&retained);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&moved);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMVariantRefTest, SpanResetHandlesMixedValues) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_variant_t values[] = {
      iree_vm_variant_from_i32(42),
      test_object_variant_from_ptr_move(&kTestTypes, &object),
      iree_vm_variant_from_function_ref(iree_vm_function_ref_null()),
      iree_vm_variant_null(),
  };
  iree_vm_variant_span_t span = iree_vm_variant_span_from_array(values);
  EXPECT_EQ(span.count, 4u);
  iree_vm_variant_span_reset(span);
  EXPECT_EQ(destroy_count, 1);
  for (iree_vm_variant_t value : values) {
    EXPECT_TRUE(iree_vm_variant_is_empty(value));
  }
}

TEST(VMVariantRefTest, CanonicalNullSupportsEveryRefConversion) {
  iree_vm_variant_t null_variant = iree_vm_variant_null();
  TestObject* output = reinterpret_cast<TestObject*>(uintptr_t{0x1234});
  IREE_EXPECT_OK(test_object_ptr_from_variant_borrowed(&kTestTypes,
                                                       null_variant, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_FALSE(test_object_variant_isa(&kTestTypes, null_variant));

  iree_vm_ref_t output_ref = {
      reinterpret_cast<void*>(uintptr_t{0x1234}),
      uintptr_t{0x5678},
  };
  IREE_EXPECT_OK(iree_vm_ref_from_variant_move(&null_variant, &output_ref));
  EXPECT_TRUE(iree_vm_ref_is_null(output_ref));
  EXPECT_TRUE(iree_vm_variant_is_empty(null_variant));
}

}  // namespace
