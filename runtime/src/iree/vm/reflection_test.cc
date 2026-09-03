// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/reflection.h"

#include <cstddef>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module_test_provider.h"

namespace {

class VMReflectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    const iree_vm_ref_type_table_t* table =
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm"));
    ASSERT_NE(table, nullptr);
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(table, &types_));
    IREE_ASSERT_OK(iree_vm_module_test_provider_initialize(
        types_.buffer, &destroy_count_, &provider_));
    module_ = &provider_.base;
  }

  void TearDown() override {
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
  }

  static std::vector<std::max_align_t> AllocateStorage(
      iree_host_size_t byte_length) {
    return std::vector<std::max_align_t>(
        (byte_length + sizeof(std::max_align_t) - 1) /
        sizeof(std::max_align_t));
  }

  static iree_byte_span_t MakeStorageSpan(
      std::vector<std::max_align_t>& storage, iree_host_size_t byte_length) {
    return iree_make_byte_span(reinterpret_cast<uint8_t*>(storage.data()),
                               byte_length);
  }

  iree_vm_environment_t* environment_ = nullptr;
  iree_vm_ref_types_t types_ = {};
  iree_vm_module_test_provider_t provider_ = {};
  iree_vm_module_t* module_ = nullptr;
  int destroy_count_ = 0;
};

TEST_F(VMReflectionTest, DescribesExportWithCallerOwnedStorage) {
  iree_vm_export_t export_value = {};
  IREE_ASSERT_OK(
      iree_vm_module_lookup_export(module_, IREE_SV("add_one"), &export_value));

  iree_host_size_t required_size = 0;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      export_value, iree_byte_span_empty(), &required_size, nullptr));
  ASSERT_GT(required_size, 0u);
  auto storage = AllocateStorage(required_size);
  iree_vm_export_description_t description = {};
  IREE_ASSERT_OK(iree_vm_export_query_description(
      export_value, MakeStorageSpan(storage, required_size), &required_size,
      &description));

  EXPECT_TRUE(iree_string_view_equal(description.name, IREE_SV("add_one")));
  EXPECT_TRUE(iree_string_view_equal(description.documentation,
                                     IREE_SV("Adds one to a value.")));
  EXPECT_TRUE(iree_string_view_equal(description.authored_type,
                                     IREE_SV("(i32) -> i32")));
  ASSERT_EQ(description.arguments.count, 1u);
  EXPECT_EQ(description.arguments.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
  EXPECT_EQ(description.arguments.data[0].type.value.scalar,
            IREE_VM_SCALAR_TYPE_I32);
  EXPECT_TRUE(iree_string_view_equal(description.arguments.data[0].name,
                                     IREE_SV("value")));
  ASSERT_EQ(description.results.count, 1u);
  EXPECT_TRUE(
      iree_string_view_equal(description.results.data[0].name, IREE_SV("sum")));
}

TEST_F(VMReflectionTest, InsufficientStorageLeavesDescriptionUntouched) {
  iree_vm_export_t export_value = {};
  IREE_ASSERT_OK(
      iree_vm_module_lookup_export(module_, IREE_SV("add_one"), &export_value));
  iree_host_size_t required_size = 0;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      export_value, iree_byte_span_empty(), &required_size, nullptr));
  ASSERT_GT(required_size, 0u);

  auto storage = AllocateStorage(required_size - 1);
  uint8_t* storage_bytes = reinterpret_cast<uint8_t*>(storage.data());
  std::memset(storage_bytes, 0xA5, required_size - 1);
  const std::vector<uint8_t> original_storage(
      storage_bytes, storage_bytes + required_size - 1);
  iree_vm_export_description_t description = {};
  description.name = IREE_SV("untouched");
  iree_host_size_t repeated_required_size = 0;
  IREE_ASSERT_OK(iree_vm_export_query_description(
      export_value, MakeStorageSpan(storage, required_size - 1),
      &repeated_required_size, &description));
  EXPECT_EQ(repeated_required_size, required_size);
  EXPECT_TRUE(iree_string_view_equal(description.name, IREE_SV("untouched")));
  EXPECT_EQ(std::memcmp(storage_bytes, original_storage.data(),
                        original_storage.size()),
            0);
}

TEST_F(VMReflectionTest, OwnsProviderGeneratedPresentationInCallerStorage) {
  iree_vm_import_t import_value = {};
  IREE_ASSERT_OK(iree_vm_module_import_by_ordinal(module_, 0, &import_value));
  iree_host_size_t required_size = 0;
  IREE_ASSERT_OK(iree_vm_import_query_description(
      import_value, iree_byte_span_empty(), &required_size, nullptr));
  auto storage = AllocateStorage(required_size);
  iree_vm_import_description_t description = {};
  IREE_ASSERT_OK(iree_vm_import_query_description(
      import_value, MakeStorageSpan(storage, required_size), &required_size,
      &description));
  EXPECT_TRUE(
      iree_string_view_equal(description.documentation, IREE_SV("generated")));
  ASSERT_EQ(description.arguments.count, 1u);
  EXPECT_EQ(description.arguments.data[0].type.kind,
            IREE_VM_SIGNATURE_TYPE_KIND_REF);
  EXPECT_EQ(description.arguments.data[0].type.value.ref, types_.buffer);
  const uintptr_t storage_begin = reinterpret_cast<uintptr_t>(storage.data());
  const uintptr_t storage_end = storage_begin + required_size;
  const uintptr_t documentation =
      reinterpret_cast<uintptr_t>(description.documentation.data);
  EXPECT_GE(documentation, storage_begin);
  EXPECT_LT(documentation, storage_end);
}

TEST_F(VMReflectionTest, ResolvesNestedCallableTypes) {
  const iree_vm_callable_type_t callable_type = {module_, 2};
  iree_host_size_t required_size = 0;
  IREE_ASSERT_OK(iree_vm_callable_type_query_description(
      callable_type, iree_byte_span_empty(), &required_size, nullptr));
  auto storage = AllocateStorage(required_size);
  iree_vm_callable_type_description_t description = {};
  IREE_ASSERT_OK(iree_vm_callable_type_query_description(
      callable_type, MakeStorageSpan(storage, required_size), &required_size,
      &description));
  ASSERT_EQ(description.arguments.count, 1u);
  EXPECT_EQ(description.arguments.data[0].kind,
            IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION);
  EXPECT_EQ(description.arguments.data[0].value.callable.module, module_);
  EXPECT_EQ(description.arguments.data[0].value.callable.ordinal, 0u);
  ASSERT_EQ(description.results.count, 1u);
  EXPECT_EQ(description.results.data[0].kind,
            IREE_VM_SIGNATURE_TYPE_KIND_SCALAR);
}

TEST_F(VMReflectionTest, PreservesAliasLocalPresentationAndMetadata) {
  iree_vm_export_t add_one = {};
  iree_vm_export_t increment = {};
  IREE_ASSERT_OK(
      iree_vm_module_lookup_export(module_, IREE_SV("add_one"), &add_one));
  IREE_ASSERT_OK(
      iree_vm_module_lookup_export(module_, IREE_SV("increment"), &increment));

  iree_vm_module_export_declaration_t add_one_declaration = {};
  iree_vm_module_export_declaration_t increment_declaration = {};
  IREE_ASSERT_OK(iree_vm_module_query_export(module_, add_one.ordinal,
                                             &add_one_declaration));
  IREE_ASSERT_OK(iree_vm_module_query_export(module_, increment.ordinal,
                                             &increment_declaration));
  EXPECT_EQ(add_one_declaration.function_ordinal,
            increment_declaration.function_ordinal);

  bool found = false;
  iree_vm_metadata_value_t add_one_metadata = {};
  iree_vm_metadata_value_t increment_metadata = {};
  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(add_one, IREE_SV("alias"),
                                                    &found, &add_one_metadata));
  ASSERT_TRUE(found);
  IREE_ASSERT_OK(iree_vm_export_try_lookup_metadata(
      increment, IREE_SV("alias"), &found, &increment_metadata));
  ASSERT_TRUE(found);
  iree_string_view_t add_one_text = iree_string_view_empty();
  iree_string_view_t increment_text = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_vm_string_view_from_metadata_value(add_one_metadata, &add_one_text));
  IREE_ASSERT_OK(iree_vm_string_view_from_metadata_value(increment_metadata,
                                                         &increment_text));
  EXPECT_FALSE(iree_string_view_equal(add_one_text, increment_text));
}

TEST_F(VMReflectionTest, EnumeratesAndLooksUpTypedMetadata) {
  EXPECT_EQ(iree_vm_module_metadata_count(module_), 2u);
  iree_vm_metadata_entry_t entry = {};
  IREE_ASSERT_OK(iree_vm_module_metadata_by_ordinal(module_, 1, &entry));
  EXPECT_TRUE(iree_string_view_equal(entry.key, IREE_SV("revision")));
  uint64_t revision = 0;
  IREE_ASSERT_OK(iree_vm_u64_from_metadata_value(entry.value, &revision));
  EXPECT_EQ(revision, 7u);

  bool found = true;
  iree_vm_metadata_value_t value = {};
  value.type = 99;
  IREE_ASSERT_OK(iree_vm_module_try_lookup_metadata(module_, IREE_SV("missing"),
                                                    &found, &value));
  EXPECT_FALSE(found);
  EXPECT_EQ(value.type, 99u);

  iree_vm_import_t import_value = {};
  IREE_ASSERT_OK(iree_vm_module_import_by_ordinal(module_, 0, &import_value));
  EXPECT_EQ(iree_vm_import_metadata_count(import_value), 1u);
  IREE_ASSERT_OK(iree_vm_import_metadata_by_ordinal(import_value, 0, &entry));
  bool optional = false;
  IREE_ASSERT_OK(iree_vm_bool_from_metadata_value(entry.value, &optional));
  EXPECT_TRUE(optional);
}

TEST(VMReflectionStandaloneTest, ExtractsExactMetadataRepresentations) {
  const uint8_t integer_bytes[] = {0xF9, 0xFF, 0xFF, 0xFF,
                                   0xFF, 0xFF, 0xFF, 0xFF};
  const uint8_t floating_bytes[] = {0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0xF8, 0x3F};
  const uint8_t opaque_bytes[] = {1, 2, 3};
  int64_t integer_value = 0;
  IREE_ASSERT_OK(iree_vm_i64_from_metadata_value(
      {IREE_VM_METADATA_VALUE_TYPE_I64,
       iree_make_const_byte_span(integer_bytes, sizeof(integer_bytes))},
      &integer_value));
  EXPECT_EQ(integer_value, -7);
  double floating_value = 0.0;
  IREE_ASSERT_OK(iree_vm_f64_from_metadata_value(
      {IREE_VM_METADATA_VALUE_TYPE_F64,
       iree_make_const_byte_span(floating_bytes, sizeof(floating_bytes))},
      &floating_value));
  EXPECT_EQ(floating_value, 1.5);
  iree_const_byte_span_t opaque_value = iree_const_byte_span_empty();
  IREE_ASSERT_OK(iree_vm_const_byte_span_from_metadata_value(
      {IREE_VM_METADATA_VALUE_TYPE_BYTES,
       iree_make_const_byte_span(opaque_bytes, sizeof(opaque_bytes))},
      &opaque_value));
  EXPECT_EQ(opaque_value.data, opaque_bytes);
  EXPECT_EQ(opaque_value.data_length, sizeof(opaque_bytes));

  uint64_t untouched = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_u64_from_metadata_value(
          {IREE_VM_METADATA_VALUE_TYPE_I64,
           iree_make_const_byte_span(integer_bytes, sizeof(integer_bytes))},
          &untouched));
  EXPECT_EQ(untouched, 123u);
}

}  // namespace
