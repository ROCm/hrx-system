// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/module.h"

namespace {

enum class DefinitionMode {
  kMinimal,
  kUnsortedExports,
  kRecursiveCallable,
  kImportsWithoutGroups,
  kSortedOverloadedImports,
  kUnsortedOverloadedImports,
  kUnsortedMetadata,
  kOversizedProcessStorage,
};

struct ValidationModule {
  // Generic module base published by the operation under test.
  iree_vm_module_t base = {};
  // Mutable test vtable used to inject ABI defects.
  iree_vm_module_vtable_t vtable = {};
  // Fixed descriptor configured for one validation case.
  iree_vm_module_descriptor_t descriptor = {};
  // Semantic defect exposed by the query callbacks.
  DefinitionMode mode = DefinitionMode::kMinimal;
  // Whether final release reached the provider's destroy callback.
  bool destroy_called = false;

  explicit ValidationModule(DefinitionMode definition_mode)
      : mode(definition_mode) {
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
    switch (mode) {
      case DefinitionMode::kMinimal:
        break;
      case DefinitionMode::kUnsortedExports:
        descriptor.counts.function_count = 1;
        descriptor.counts.callable_type_count = 1;
        descriptor.counts.export_count = 2;
        break;
      case DefinitionMode::kRecursiveCallable:
        descriptor.counts.callable_type_count = 1;
        break;
      case DefinitionMode::kImportsWithoutGroups:
        descriptor.counts.callable_type_count = 1;
        descriptor.counts.import_count = 1;
        break;
      case DefinitionMode::kSortedOverloadedImports:
      case DefinitionMode::kUnsortedOverloadedImports:
        descriptor.counts.callable_type_count = 2;
        descriptor.counts.import_group_count = 1;
        descriptor.counts.import_count = 2;
        break;
      case DefinitionMode::kUnsortedMetadata:
        descriptor.counts.metadata_count = 2;
        break;
      case DefinitionMode::kOversizedProcessStorage:
#if UINTPTR_MAX > UINT32_MAX
        descriptor.process_storage_size =
            static_cast<iree_host_size_t>(UINT32_MAX) + 1;
#endif  // UINTPTR_MAX > UINT32_MAX
        break;
    }
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
    (void)ordinal;
    const ValidationModule* validation_module = Cast(module);
    if (validation_module->mode == DefinitionMode::kSortedOverloadedImports ||
        validation_module->mode == DefinitionMode::kUnsortedOverloadedImports) {
      *out_group = {IREE_SV("target.module"), 0, 2};
    } else {
      *out_group = {};
    }
  }

  static void QueryImport(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_import_declaration_t* out_import) {
    const ValidationModule* validation_module = Cast(module);
    if (validation_module->mode == DefinitionMode::kSortedOverloadedImports ||
        validation_module->mode == DefinitionMode::kUnsortedOverloadedImports) {
      const bool is_sorted =
          validation_module->mode == DefinitionMode::kSortedOverloadedImports;
      *out_import = {
          IREE_SV("target.module"),
          IREE_SV("overloaded"),
          is_sorted ? ordinal : 1 - ordinal,
          IREE_VM_MODULE_IMPORT_FLAG_NONE,
          0,
      };
    } else {
      *out_import = {};
    }
  }

  static void QueryExport(const iree_vm_module_t* module,
                          iree_host_size_t ordinal,
                          iree_vm_module_export_declaration_t* out_export) {
    const ValidationModule* validation_module = Cast(module);
    if (validation_module->mode == DefinitionMode::kUnsortedExports) {
      *out_export = {
          ordinal == 0 ? IREE_SV("z") : IREE_SV("a"),
          0,
          0,
          0,
      };
    } else {
      *out_export = {};
    }
  }

  static void QueryCallableType(
      const iree_vm_module_t* module, iree_host_size_t ordinal,
      iree_vm_module_callable_type_declaration_t* out_callable_type) {
    (void)ordinal;
    static const iree_vm_module_signature_type_t recursive_argument = {
        IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION,
        0,
    };
    const ValidationModule* validation_module = Cast(module);
    if (validation_module->mode == DefinitionMode::kRecursiveCallable) {
      *out_callable_type = {
          {{&recursive_argument, 1}, {nullptr, 0}},
          IREE_VM_CALLABLE_TYPE_FLAG_NONE,
      };
    } else {
      *out_callable_type = {};
    }
  }

  static void MetadataByOrdinal(const iree_vm_module_t* module,
                                const iree_vm_module_metadata_query_t* query,
                                iree_vm_metadata_entry_t* out_entry) {
    (void)module;
    *out_entry = {
        query->ordinal == 0 ? IREE_SV("z") : IREE_SV("a"),
        {IREE_VM_METADATA_VALUE_TYPE_BYTES, {nullptr, 0}},
    };
  }
};

static_assert(offsetof(ValidationModule, base) == 0,
              "validation module base must remain at offset zero");

TEST(VMModuleValidationTest, PublishesFirstOwnerOnlyAfterValidation) {
  ValidationModule module(DefinitionMode::kMinimal);
  IREE_ASSERT_OK(module.Publish());
  EXPECT_EQ(module.base.vtable, &module.vtable);
  EXPECT_EQ(module.base.descriptor, &module.descriptor);
  EXPECT_FALSE(module.destroy_called);

  iree_vm_module_release(&module.base);
  EXPECT_TRUE(module.destroy_called);
}

TEST(VMModuleValidationTest, RejectsIncompleteOrIncompatibleVtables) {
  ValidationModule module(DefinitionMode::kMinimal);
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, module.Publish());
  EXPECT_EQ(module.base.vtable, nullptr);
  EXPECT_EQ(module.base.descriptor, nullptr);
  EXPECT_FALSE(module.destroy_called);
}

TEST(VMModuleValidationTest, RejectsMalformedSemanticGraphs) {
  for (DefinitionMode mode :
       {DefinitionMode::kUnsortedExports, DefinitionMode::kRecursiveCallable,
        DefinitionMode::kImportsWithoutGroups,
        DefinitionMode::kUnsortedOverloadedImports,
        DefinitionMode::kUnsortedMetadata}) {
    ValidationModule module(mode);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, module.Publish());
    EXPECT_EQ(module.base.vtable, nullptr);
    EXPECT_EQ(module.base.descriptor, nullptr);
    EXPECT_FALSE(module.destroy_called);
  }
}

TEST(VMModuleValidationTest, AcceptsImportsSortedByNameAndCallableType) {
  ValidationModule module(DefinitionMode::kSortedOverloadedImports);
  IREE_ASSERT_OK(module.Publish());
  iree_vm_module_release(&module.base);
  EXPECT_TRUE(module.destroy_called);
}

#if UINTPTR_MAX > UINT32_MAX
TEST(VMModuleValidationTest, RejectsUnrepresentableInlineProcessStorage) {
  ValidationModule module(DefinitionMode::kOversizedProcessStorage);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, module.Publish());
  EXPECT_EQ(module.base.vtable, nullptr);
  EXPECT_EQ(module.base.descriptor, nullptr);
  EXPECT_FALSE(module.destroy_called);
}
#endif  // UINTPTR_MAX > UINT32_MAX

TEST(VMModuleValidationTest, CheckedQueriesLeaveOutputsUntouched) {
  ValidationModule module(DefinitionMode::kMinimal);
  IREE_ASSERT_OK(module.Publish());

  iree_vm_export_t export_value = {
      reinterpret_cast<const iree_vm_module_t*>(uintptr_t{1}),
      123,
  };
  const iree_vm_export_t original_export = export_value;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_vm_module_export_by_ordinal(&module.base, 0, &export_value));
  EXPECT_EQ(export_value.module, original_export.module);
  EXPECT_EQ(export_value.ordinal, original_export.ordinal);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_vm_module_lookup_export(
                            &module.base, IREE_SV("missing"), &export_value));
  EXPECT_EQ(export_value.module, original_export.module);
  EXPECT_EQ(export_value.ordinal, original_export.ordinal);

  iree_vm_module_release(&module.base);
}

}  // namespace
