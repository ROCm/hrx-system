// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/ref.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct TestObject {
  // Required offset-zero VM ref-count prefix.
  iree_vm_ref_object_t ref_object;
  // Number of final destruction callbacks observed by the test.
  int* destroy_count;
  // Payload used to verify typed extraction.
  int value;
};

struct TestTypesV0 {
  // First append-order test type.
  iree_vm_ref_type_t object;
};

struct TestTypesV1 {
  // First append-order test type shared with version zero.
  iree_vm_ref_type_t object;
  // Type appended by version one.
  iree_vm_ref_type_t future;
};

extern const iree_vm_ref_type_table_t kTestTableV0;
extern const iree_vm_ref_type_table_t kTestTableV1;

void DestroyTestObject(void* object) {
  TestObject* test_object = static_cast<TestObject*>(object);
  ++*test_object->destroy_count;
  delete test_object;
}

const iree_vm_ref_type_descriptor_t kTestObjectTypeV0 = {
    DestroyTestObject,
    &kTestTableV0,
    IREE_SV("object"),
};
const TestTypesV0 kTestTypesV0 = {&kTestObjectTypeV0};
const iree_vm_ref_type_table_t kTestTableV0 = {
    sizeof(kTestTableV0),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("test"),
    {&kTestTypesV0, 1},
};

const iree_vm_ref_type_descriptor_t kTestObjectTypeV1 = {
    DestroyTestObject,
    &kTestTableV1,
    IREE_SV("object"),
};
const iree_vm_ref_type_descriptor_t kTestFutureTypeV1 = {
    nullptr,
    &kTestTableV1,
    IREE_SV("future"),
};
const TestTypesV1 kTestTypesV1 = {
    &kTestObjectTypeV1,
    &kTestFutureTypeV1,
};
const iree_vm_ref_type_table_t kTestTableV1 = {
    sizeof(kTestTableV1),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SV("test"),
    {&kTestTypesV1, 2},
};

TestObject* CreateTestObject(int* destroy_count) {
  TestObject* object = new TestObject;
  iree_vm_ref_object_initialize(&object->ref_object);
  object->destroy_count = destroy_count;
  object->value = 42;
  return object;
}

iree_status_t ResolveTestTypesV0(const iree_vm_ref_type_table_t* table,
                                 TestTypesV0* out_types) {
  if (!table || !out_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table and out_types are required");
  }
  if (!iree_string_view_equal(table->namespace_name, IREE_SV("test")) ||
      table->types.count < 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test ref-type prefix is unavailable");
  }
  TestTypesV0 types = {iree_vm_ref_type_storage_at(table->types, 0)};
  if (!iree_string_view_equal(types.object->type_name, IREE_SV("object"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test ref-type ordinal mismatch");
  }
  *out_types = types;
  return iree_ok_status();
}

iree_status_t ResolveTestTypesV1(const iree_vm_ref_type_table_t* table,
                                 TestTypesV1* out_types) {
  if (!table || !out_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table and out_types are required");
  }
  if (!iree_string_view_equal(table->namespace_name, IREE_SV("test")) ||
      table->types.count < 2) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test ref-type prefix is unavailable");
  }
  TestTypesV1 types = {
      iree_vm_ref_type_storage_at(table->types, 0),
      iree_vm_ref_type_storage_at(table->types, 1),
  };
  if (!iree_string_view_equal(types.object->type_name, IREE_SV("object")) ||
      !iree_string_view_equal(types.future->type_name, IREE_SV("future"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test ref-type ordinal mismatch");
  }
  *out_types = types;
  return iree_ok_status();
}

TEST(VMRefTypeTest, StructuredIdentityAndDenseStorage) {
  iree_vm_ref_type_key_t null_key = iree_vm_ref_type_key(nullptr);
  EXPECT_EQ(null_key.namespace_name.size, 0u);
  EXPECT_EQ(null_key.type_name.size, 0u);

  iree_vm_ref_type_key_t key = iree_vm_ref_type_key(&kTestObjectTypeV1);
  EXPECT_TRUE(iree_string_view_equal(key.namespace_name, IREE_SV("test")));
  EXPECT_TRUE(iree_string_view_equal(key.type_name, IREE_SV("object")));
  EXPECT_EQ(iree_vm_ref_type_storage_at(kTestTableV1.types, 0),
            &kTestObjectTypeV1);
  EXPECT_EQ(iree_vm_ref_type_storage_at(kTestTableV1.types, 1),
            &kTestFutureTypeV1);
}

TEST(VMRefTypeTest, ConsumerPrefixAcceptsNewerProvider) {
  TestTypesV0 resolved = {nullptr};
  IREE_EXPECT_OK(ResolveTestTypesV0(&kTestTableV1, &resolved));
  EXPECT_EQ(resolved.object, &kTestObjectTypeV1);
}

TEST(VMRefTypeTest, NewerConsumerRejectsOlderProviderAtomically) {
  TestTypesV1 resolved = {reinterpret_cast<iree_vm_ref_type_t>(uintptr_t{1}),
                          reinterpret_cast<iree_vm_ref_type_t>(uintptr_t{2})};
  const TestTypesV1 original = resolved;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        ResolveTestTypesV1(&kTestTableV0, &resolved));
  EXPECT_EQ(resolved.object, original.object);
  EXPECT_EQ(resolved.future, original.future);
}

TEST(VMRefObjectTest, RawRetainAndReleaseDestroyExactlyOnce) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_ref_object_retain(object);
  iree_vm_ref_object_release(object, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_object_release(object, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, BorrowedResetDoesNotRelease) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_ref_t borrowed =
      iree_vm_ref_from_ptr_borrowed(object, &kTestObjectTypeV1);
  EXPECT_TRUE(iree_vm_ref_isa(borrowed, &kTestObjectTypeV1));
  iree_vm_ref_reset(&borrowed);
  EXPECT_TRUE(iree_vm_ref_is_null(borrowed));
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_object_release(object, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, RetainedRefAddsIndependentOwner) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_ref_t retained =
      iree_vm_ref_from_ptr_retained(object, &kTestObjectTypeV1);
  iree_vm_ref_object_release(object, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&retained);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, PointerMoveClearsSourceAndTransfersOwner) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  void* pointer = object;
  iree_vm_ref_t owner = iree_vm_ref_from_ptr_move(&pointer, &kTestObjectTypeV1);
  EXPECT_EQ(pointer, nullptr);
  EXPECT_EQ(owner.object, object);
  iree_vm_ref_reset(&owner);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, RetainAndOwnedMovePreserveExactLifetime) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  void* pointer = object;
  iree_vm_ref_t owner = iree_vm_ref_from_ptr_move(&pointer, &kTestObjectTypeV1);
  iree_vm_ref_t clone = iree_vm_ref_retain(owner);
  iree_vm_ref_t moved = iree_vm_ref_move(&owner);
  EXPECT_TRUE(iree_vm_ref_is_null(owner));
  iree_vm_ref_reset(&clone);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&moved);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, RetainingBorrowCreatesIndependentOwner) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_ref_t borrowed =
      iree_vm_ref_from_ptr_borrowed(object, &kTestObjectTypeV1);
  iree_vm_ref_t retained = iree_vm_ref_retain(borrowed);

  iree_vm_ref_object_release(object, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&borrowed);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&retained);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, MovingBorrowPromotesBeforeEscape) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  iree_vm_ref_t borrowed =
      iree_vm_ref_from_ptr_borrowed(object, &kTestObjectTypeV1);
  iree_vm_ref_t promoted = iree_vm_ref_move(&borrowed);
  EXPECT_TRUE(iree_vm_ref_is_null(borrowed));
  iree_vm_ref_object_release(object, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 0);
  iree_vm_ref_reset(&promoted);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, CheckedPointerConversionsPreserveFailureAtomicity) {
  int destroy_count = 0;
  TestObject* object = CreateTestObject(&destroy_count);
  void* pointer = object;
  iree_vm_ref_t owner = iree_vm_ref_from_ptr_move(&pointer, &kTestObjectTypeV1);

  void* output = reinterpret_cast<void*>(uintptr_t{0x1234});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_ptr_from_ref_move(&owner, &kTestFutureTypeV1, &output));
  EXPECT_EQ(output, reinterpret_cast<void*>(uintptr_t{0x1234}));
  EXPECT_EQ(owner.object, object);

  IREE_EXPECT_OK(
      iree_vm_ptr_from_ref_borrowed(owner, &kTestObjectTypeV1, &output));
  EXPECT_EQ(output, object);

  IREE_EXPECT_OK(
      iree_vm_ptr_from_ref_retained(owner, &kTestObjectTypeV1, &output));
  EXPECT_EQ(output, object);
  iree_vm_ref_object_release(output, &kTestObjectTypeV1);

  output = nullptr;
  IREE_EXPECT_OK(
      iree_vm_ptr_from_ref_move(&owner, &kTestObjectTypeV1, &output));
  EXPECT_TRUE(iree_vm_ref_is_null(owner));
  EXPECT_EQ(output, object);
  iree_vm_ref_object_release(output, &kTestObjectTypeV1);
  EXPECT_EQ(destroy_count, 1);
}

TEST(VMRefTest, CanonicalNullConvertsWithoutATypeIdentity) {
  iree_vm_ref_t null_ref = iree_vm_ref_null();
  void* output = reinterpret_cast<void*>(uintptr_t{0x1234});
  IREE_EXPECT_OK(
      iree_vm_ptr_from_ref_borrowed(null_ref, &kTestObjectTypeV1, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_FALSE(iree_vm_ref_isa(null_ref, &kTestObjectTypeV1));

  output = reinterpret_cast<void*>(uintptr_t{0x1234});
  IREE_EXPECT_OK(
      iree_vm_ptr_from_ref_retained(null_ref, &kTestObjectTypeV1, &output));
  EXPECT_EQ(output, nullptr);

  output = reinterpret_cast<void*>(uintptr_t{0x1234});
  IREE_EXPECT_OK(
      iree_vm_ptr_from_ref_move(&null_ref, &kTestObjectTypeV1, &output));
  EXPECT_EQ(output, nullptr);
  EXPECT_TRUE(iree_vm_ref_is_null(null_ref));
}

}  // namespace
