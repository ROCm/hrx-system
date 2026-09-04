// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/module.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/reflection.h"

namespace {

using iree::vm::testing::TableModule;

static const iree_vm_module_callable_type_declaration_t kLeafCallableTypes[] = {
    {{{nullptr, 0, 0, 0, 0}, {nullptr, 0, 0, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
};
static const iree_vm_module_export_declaration_t kUnsortedExports[] = {
    {IREE_SVL("z"), 0, 0, 0},
    {IREE_SVL("a"), 0, 0, 0},
};
static const iree_vm_module_signature_type_t kRecursiveArgument = {
    IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION,
    0,
};
static const iree_vm_module_callable_type_declaration_t
    kRecursiveCallableTypes[] = {
        {{{&kRecursiveArgument, 1, 0, 0, 1}, {nullptr, 0, 0, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
};
static const iree_vm_module_callable_type_declaration_t
    kWrongDepthCallableTypes[] = {
        {{{nullptr, 0, 0, 0, 0}, {nullptr, 0, 0, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         1,
         0},
};
static const iree_vm_module_callable_type_declaration_t
    kDuplicateCallableTypes[] = {
        {{{nullptr, 0, 0, 0, 0}, {nullptr, 0, 0, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{nullptr, 0, 0, 0, 0}, {nullptr, 0, 0, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
};
static const iree_vm_module_callable_type_declaration_t
    kUnsortedCallableTypes[] = {
        {{{nullptr, 0, 0, 0, 0}, {nullptr, 0, 0, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
         0,
         0},
        {{{nullptr, 0, 0, 0, 0}, {nullptr, 0, 0, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
};
static const iree_vm_module_import_group_t kTwoImportGroup[] = {
    {IREE_SVL("target.module"), 0, 2},
};
static const iree_vm_module_import_declaration_t kDuplicateTargetImports[] = {
    {IREE_SVL("target.module"), IREE_SVL("same_target"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("target.module"), IREE_SVL("same_target"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
};
static const iree_vm_module_import_declaration_t kUnsortedImports[] = {
    {IREE_SVL("target.module"), IREE_SVL("z"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("target.module"), IREE_SVL("a"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
};
static const iree_vm_metadata_entry_t kUnsortedMetadata[] = {
    {IREE_SVL("z"), {IREE_VM_METADATA_VALUE_TYPE_BYTES, {nullptr, 0}}},
    {IREE_SVL("a"), {IREE_VM_METADATA_VALUE_TYPE_BYTES, {nullptr, 0}}},
};
static const iree_vm_metadata_entry_t kInvalidMetadata[] = {
    {IREE_SVL("value"), {IREE_VM_METADATA_VALUE_TYPE_INVALID, {nullptr, 0}}},
};
static const iree_vm_metadata_entry_t kOutOfRangeMetadata[] = {
    {IREE_SVL("value"), {UINT16_MAX + 1u, {nullptr, 0}}},
};
static const iree_vm_metadata_entry_t kUnknownMetadata[] = {
    {IREE_SVL("value"), {99, {nullptr, 0}}},
};

static void ExpectRejected(TableModule& module) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, module.Publish());
  EXPECT_EQ(module.storage()->vtable, nullptr);
  EXPECT_EQ(module.storage()->descriptor, nullptr);
  EXPECT_EQ(module.destruction_count(), 0);
}

template <std::size_t N>
static void ExpectCallableTypesRejected(
    const iree_vm_module_callable_type_declaration_t (&values)[N]) {
  TableModule module;
  module.definition().WithCallableTypes(values);
  ExpectRejected(module);
}

template <std::size_t N>
static void ExpectMetadataRejected(
    const iree_vm_metadata_entry_t (&values)[N]) {
  TableModule module;
  module.definition().WithMetadata(values);
  ExpectRejected(module);
}

TEST(VMModuleValidationTest, PublishesOwnerOnlyAfterValidation) {
  TableModule module;
  IREE_ASSERT_OK(module.Publish());
  EXPECT_EQ(module.storage()->vtable, &module.vtable());
  EXPECT_EQ(module.storage()->descriptor, &module.descriptor());
  EXPECT_EQ(module.destruction_count(), 0);
  module.ReleaseOwner();
  EXPECT_EQ(module.destruction_count(), 1);
}

TEST(VMModuleValidationTest, RejectsIncompleteOrIncompatibleVtables) {
  TableModule module;
  module.vtable().structure_size = IREE_VM_MODULE_VTABLE_V0_REQUIRED_SIZE - 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE, module.Publish());
  EXPECT_EQ(module.storage()->vtable, nullptr);
  EXPECT_EQ(module.storage()->descriptor, nullptr);
  EXPECT_EQ(module.destruction_count(), 0);

  module.vtable().structure_size = sizeof(module.vtable());
  module.vtable().abi_version = IREE_VM_MODULE_ABI_VERSION_0 + 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE, module.Publish());
  EXPECT_EQ(module.storage()->vtable, nullptr);
  EXPECT_EQ(module.storage()->descriptor, nullptr);

  module.vtable().abi_version = IREE_VM_MODULE_ABI_VERSION_0;
  module.vtable().query_export = nullptr;
  ExpectRejected(module);
}

TEST(VMModuleValidationTest, RejectsMalformedSemanticTables) {
  {
    iree_vm_module_callable_type_declaration_t wrong_bank_counts[] = {
        kLeafCallableTypes[0]};
    wrong_bank_counts[0].signature.arguments.value_count = 1;
    ExpectCallableTypesRejected(wrong_bank_counts);
  }
  {
    TableModule module;
    module.definition().WithCallableTypes(kLeafCallableTypes);
    module.descriptor().counts.callable_fields.value_count = 1;
    ExpectRejected(module);
  }
  {
    TableModule module;
    module.definition()
        .WithCallableTypes(kLeafCallableTypes)
        .WithExports(kUnsortedExports);
    module.descriptor().counts.function_count = 1;
    ExpectRejected(module);
  }
  ExpectCallableTypesRejected(kRecursiveCallableTypes);
  ExpectCallableTypesRejected(kWrongDepthCallableTypes);
  ExpectCallableTypesRejected(kDuplicateCallableTypes);
  ExpectCallableTypesRejected(kUnsortedCallableTypes);
  {
    TableModule module;
    module.definition().WithCallableTypes(kLeafCallableTypes);
    module.descriptor().counts.import_count = 1;
    ExpectRejected(module);
  }
  {
    TableModule module;
    module.definition()
        .WithCallableTypes(kLeafCallableTypes)
        .WithImportGroups(kTwoImportGroup)
        .WithImports(kUnsortedImports);
    ExpectRejected(module);
  }
  ExpectMetadataRejected(kUnsortedMetadata);
  ExpectMetadataRejected(kInvalidMetadata);
  ExpectMetadataRejected(kOutOfRangeMetadata);
}

TEST(VMModuleValidationTest, AcceptsDuplicateTargetsAndUnknownMetadataTypes) {
  TableModule module;
  module.definition()
      .WithCallableTypes(kLeafCallableTypes)
      .WithImportGroups(kTwoImportGroup)
      .WithImports(kDuplicateTargetImports)
      .WithMetadata(kUnknownMetadata);
  IREE_ASSERT_OK(module.Publish());
  module.ReleaseOwner();
  EXPECT_EQ(module.destruction_count(), 1);
}

TEST(VMModuleValidationTest, ReflectionOnlyTypesCannotEnterLinkableModules) {
  iree_vm_ref_type_table_t table = {
      sizeof(table),
      IREE_VM_REF_TYPE_TABLE_FLAG_REFLECTION_ONLY,
      IREE_SV("external"),
      {},
  };
  const iree_vm_ref_type_descriptor_t descriptor = {
      nullptr,
      &table,
      IREE_SV("object"),
  };
  const iree_vm_ref_type_t types[] = {&descriptor};
  table.types = {types, IREE_ARRAYSIZE(types)};

  TableModule linkable_module;
  linkable_module.descriptor().flags = IREE_VM_MODULE_FLAG_LINKABLE;
  linkable_module.descriptor().ref_types = {types, IREE_ARRAYSIZE(types)};
  ExpectRejected(linkable_module);

  TableModule inspection_module;
  inspection_module.descriptor().ref_types = {types, IREE_ARRAYSIZE(types)};
  IREE_ASSERT_OK(inspection_module.Publish());
  inspection_module.ReleaseOwner();
  EXPECT_EQ(inspection_module.destruction_count(), 1);
}

#if UINTPTR_MAX > UINT32_MAX
TEST(VMModuleValidationTest, RejectsUnrepresentableProcessStorage) {
  TableModule module;
  module.descriptor().process_storage_size =
      static_cast<iree_host_size_t>(UINT32_MAX) + 1;
  ExpectRejected(module);
}
#endif  // UINTPTR_MAX > UINT32_MAX

}  // namespace
