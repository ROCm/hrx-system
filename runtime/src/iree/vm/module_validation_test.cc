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
#include "iree/vm/reflection.h"

namespace {

static const iree_vm_module_callable_type_declaration_t kLeafCallableTypes[] = {
    {{{nullptr, 0}, {nullptr, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 0, 0},
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
        {{{&kRecursiveArgument, 1}, {nullptr, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
};
static const iree_vm_module_callable_type_declaration_t
    kWrongDepthCallableTypes[] = {
        {{{nullptr, 0}, {nullptr, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 1, 0},
};
static const iree_vm_module_callable_type_declaration_t
    kDuplicateCallableTypes[] = {
        {{{nullptr, 0}, {nullptr, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 0, 0},
        {{{nullptr, 0}, {nullptr, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 0, 0},
};
static const iree_vm_module_callable_type_declaration_t
    kUnsortedCallableTypes[] = {
        {{{nullptr, 0}, {nullptr, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
         0,
         0},
        {{{nullptr, 0}, {nullptr, 0}}, IREE_VM_CALLABLE_TYPE_FLAG_NONE, 0, 0},
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

struct ValidationModule {
  // Generic base published by the operation under test.
  iree_vm_module_t base = {};
  // Mutable vtable used to inject ABI defects.
  iree_vm_module_vtable_t vtable = {};
  // Fixed descriptor configured by each test.
  iree_vm_module_descriptor_t descriptor = {};
  // Immutable import groups borrowed during publication.
  const iree_vm_module_import_group_t* import_groups = nullptr;
  // Immutable imports borrowed during publication.
  const iree_vm_module_import_declaration_t* imports = nullptr;
  // Immutable exports borrowed during publication.
  const iree_vm_module_export_declaration_t* exports = nullptr;
  // Immutable callable types borrowed during publication.
  const iree_vm_module_callable_type_declaration_t* callable_types = nullptr;
  // Immutable metadata entries borrowed during publication.
  const iree_vm_metadata_entry_t* metadata = nullptr;
  // Whether final release reached the provider callback.
  bool destroy_called = false;

  ValidationModule() {
    vtable = {
        sizeof(vtable),
        IREE_VM_MODULE_ABI_VERSION_0,
        Destroy,
        FunctionStart,
        iree_vm_module_function_resume_unreachable,
        nullptr,
        nullptr,
        nullptr,
        QueryImportGroup,
        QueryImport,
        QueryExport,
        QueryCallableType,
        iree_vm_module_query_presentation_none,
        MetadataByOrdinal,
    };
    descriptor.name = IREE_SV("validation.module");
  }

  template <std::size_t N>
  void SetImportGroups(const iree_vm_module_import_group_t (&values)[N]) {
    import_groups = values;
    descriptor.counts.import_group_count = N;
  }

  template <std::size_t N>
  void SetImports(const iree_vm_module_import_declaration_t (&values)[N]) {
    imports = values;
    descriptor.counts.import_count = N;
  }

  template <std::size_t N>
  void SetExports(const iree_vm_module_export_declaration_t (&values)[N]) {
    exports = values;
    descriptor.counts.export_count = N;
  }

  template <std::size_t N>
  void SetCallableTypes(
      const iree_vm_module_callable_type_declaration_t (&values)[N]) {
    callable_types = values;
    descriptor.counts.callable_type_count = N;
  }

  template <std::size_t N>
  void SetMetadata(const iree_vm_metadata_entry_t (&values)[N]) {
    metadata = values;
    descriptor.counts.metadata_count = N;
  }

  iree_status_t Publish() {
    return iree_vm_module_initialize(&vtable, &descriptor, &base);
  }

  static ValidationModule* Cast(iree_vm_module_t* module) {
    return iree_containerof(module, ValidationModule, base);
  }

  static const ValidationModule* Cast(const iree_vm_module_t* module) {
    return iree_containerof(module, ValidationModule, base);
  }

  static void Destroy(iree_vm_module_t* module) {
    Cast(module)->destroy_called = true;
  }

  static iree_status_t FunctionStart(
      iree_vm_module_t* module,
      const iree_vm_module_function_start_params_t* params,
      iree_vm_execution_outcome_t* out_outcome) {
    (void)module;
    (void)params;
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
    return iree_ok_status();
  }

  static void QueryImportGroup(const iree_vm_module_t* module,
                               iree_host_size_t ordinal,
                               iree_vm_module_import_group_t* out_group) {
    *out_group = Cast(module)->import_groups[ordinal];
  }

  static void QueryImport(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_import_declaration_t* out_import) {
    *out_import = Cast(module)->imports[ordinal];
  }

  static void QueryExport(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_export_declaration_t* out_export) {
    *out_export = Cast(module)->exports[ordinal];
  }

  static void QueryCallableType(
      const iree_vm_module_t* module, iree_host_size_t ordinal,
      iree_vm_module_callable_type_declaration_t* out_callable_type) {
    *out_callable_type = Cast(module)->callable_types[ordinal];
  }

  static void MetadataByOrdinal(const iree_vm_module_t* module,
                                const iree_vm_module_metadata_query_t* query,
                                iree_vm_metadata_entry_t* out_entry) {
    *out_entry = Cast(module)->metadata[query->ordinal];
  }
};

static_assert(offsetof(ValidationModule, base) == 0,
              "validation module base must remain at offset zero");

static void ExpectRejected(ValidationModule& module) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, module.Publish());
  EXPECT_EQ(module.base.vtable, nullptr);
  EXPECT_EQ(module.base.descriptor, nullptr);
  EXPECT_FALSE(module.destroy_called);
}

template <std::size_t N>
static void ExpectCallableTypesRejected(
    const iree_vm_module_callable_type_declaration_t (&values)[N]) {
  ValidationModule module;
  module.SetCallableTypes(values);
  ExpectRejected(module);
}

template <std::size_t N>
static void ExpectMetadataRejected(
    const iree_vm_metadata_entry_t (&values)[N]) {
  ValidationModule module;
  module.SetMetadata(values);
  ExpectRejected(module);
}

TEST(VMModuleValidationTest, PublishesOwnerOnlyAfterValidation) {
  ValidationModule module;
  IREE_ASSERT_OK(module.Publish());
  EXPECT_EQ(module.base.vtable, &module.vtable);
  EXPECT_EQ(module.base.descriptor, &module.descriptor);
  EXPECT_FALSE(module.destroy_called);
  iree_vm_module_release(&module.base);
  EXPECT_TRUE(module.destroy_called);
}

TEST(VMModuleValidationTest, RejectsIncompleteOrIncompatibleVtables) {
  ValidationModule module;
  module.vtable.structure_size = IREE_VM_MODULE_VTABLE_V0_REQUIRED_SIZE - 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE, module.Publish());
  EXPECT_EQ(module.base.vtable, nullptr);
  EXPECT_EQ(module.base.descriptor, nullptr);
  EXPECT_FALSE(module.destroy_called);

  module.vtable.structure_size = sizeof(module.vtable);
  module.vtable.abi_version = IREE_VM_MODULE_ABI_VERSION_0 + 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE, module.Publish());
  EXPECT_EQ(module.base.vtable, nullptr);
  EXPECT_EQ(module.base.descriptor, nullptr);

  module.vtable.abi_version = IREE_VM_MODULE_ABI_VERSION_0;
  module.vtable.query_export = nullptr;
  ExpectRejected(module);
}

TEST(VMModuleValidationTest, RejectsMalformedSemanticTables) {
  {
    ValidationModule module;
    module.SetCallableTypes(kLeafCallableTypes);
    module.SetExports(kUnsortedExports);
    module.descriptor.counts.function_count = 1;
    ExpectRejected(module);
  }
  ExpectCallableTypesRejected(kRecursiveCallableTypes);
  ExpectCallableTypesRejected(kWrongDepthCallableTypes);
  ExpectCallableTypesRejected(kDuplicateCallableTypes);
  ExpectCallableTypesRejected(kUnsortedCallableTypes);
  {
    ValidationModule module;
    module.SetCallableTypes(kLeafCallableTypes);
    module.descriptor.counts.import_count = 1;
    ExpectRejected(module);
  }
  {
    ValidationModule module;
    module.SetCallableTypes(kLeafCallableTypes);
    module.SetImportGroups(kTwoImportGroup);
    module.SetImports(kUnsortedImports);
    ExpectRejected(module);
  }
  ExpectMetadataRejected(kUnsortedMetadata);
  ExpectMetadataRejected(kInvalidMetadata);
  ExpectMetadataRejected(kOutOfRangeMetadata);
}

TEST(VMModuleValidationTest, AcceptsDuplicateTargetsAndUnknownMetadataTypes) {
  ValidationModule module;
  module.SetCallableTypes(kLeafCallableTypes);
  module.SetImportGroups(kTwoImportGroup);
  module.SetImports(kDuplicateTargetImports);
  module.SetMetadata(kUnknownMetadata);
  IREE_ASSERT_OK(module.Publish());
  iree_vm_module_release(&module.base);
  EXPECT_TRUE(module.destroy_called);
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

  ValidationModule linkable_module;
  linkable_module.descriptor.flags = IREE_VM_MODULE_FLAG_LINKABLE;
  linkable_module.descriptor.ref_types = {types, IREE_ARRAYSIZE(types)};
  ExpectRejected(linkable_module);

  ValidationModule inspection_module;
  inspection_module.descriptor.ref_types = {types, IREE_ARRAYSIZE(types)};
  IREE_ASSERT_OK(inspection_module.Publish());
  iree_vm_module_release(&inspection_module.base);
  EXPECT_TRUE(inspection_module.destroy_called);
}

#if UINTPTR_MAX > UINT32_MAX
TEST(VMModuleValidationTest, RejectsUnrepresentableProcessStorage) {
  ValidationModule module;
  module.descriptor.process_storage_size =
      static_cast<iree_host_size_t>(UINT32_MAX) + 1;
  ExpectRejected(module);
}
#endif  // UINTPTR_MAX > UINT32_MAX

}  // namespace
