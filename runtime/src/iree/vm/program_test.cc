// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/program.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/module_test_util.h"
#include "iree/vm/program_storage.h"
#include "iree/vm/test_allocator.h"
#include "iree/vm/variant.h"

namespace {

using iree::vm::testing::CountingAllocator;
using iree::vm::testing::TableModule;
using iree::vm::testing::TableModuleDefinition;

std::string_view ToStringView(iree_string_view_t value) {
  return std::string_view(value.data ? value.data : "", value.size);
}

void DestroyTestRef(void* object) { (void)object; }

extern const iree_vm_ref_type_table_t kTestRefTypeTable;
static const iree_vm_ref_type_descriptor_t kAuxiliaryRefTypeStorage = {
    DestroyTestRef,
    &kTestRefTypeTable,
    IREE_SVL("auxiliary"),
};
static const iree_vm_ref_type_descriptor_t kSharedRefTypeStorage = {
    DestroyTestRef,
    &kTestRefTypeTable,
    IREE_SVL("shared"),
};
static const struct {
  // Auxiliary type at ordinal zero.
  iree_vm_ref_type_t auxiliary;
  // Shared type at ordinal one.
  iree_vm_ref_type_t shared;
} kTestRefTypes = {&kAuxiliaryRefTypeStorage, &kSharedRefTypeStorage};
const iree_vm_ref_type_table_t kTestRefTypeTable = {
    sizeof(kTestRefTypeTable),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SVL("program.test"),
    {&kTestRefTypes, 2},
};
static const iree_vm_ref_type_t kAuxiliaryRefType = &kAuxiliaryRefTypeStorage;
static const iree_vm_ref_type_t kSharedRefType = &kSharedRefTypeStorage;

TEST(VMProgramTest, PacksCompleteTargetDomainWithoutTagOverlap) {
  const uint32_t mapping =
      IREE_VM_PROGRAM_CALLABLE_TOKEN_MASK | IREE_VM_PROGRAM_CALLABLE_MAY_YIELD;
  const uint64_t target_bits =
      iree_vm_program_pack_target_bits(UINT16_MAX, UINT16_MAX, mapping);
  EXPECT_EQ(target_bits & 0x3u, 0u);
  EXPECT_EQ(iree_vm_program_target_module_ordinal(target_bits), UINT16_MAX);
  EXPECT_EQ(iree_vm_program_target_function_ordinal(target_bits), UINT16_MAX);
  EXPECT_EQ(iree_vm_program_target_callable_token(target_bits),
            IREE_VM_PROGRAM_CALLABLE_TOKEN_MASK);
  EXPECT_TRUE(iree_vm_program_target_may_yield(target_bits));
}

//===----------------------------------------------------------------------===//
// Successful cyclic composition
//===----------------------------------------------------------------------===//

static const iree_vm_module_signature_type_t kCallbackArguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t kCallbackResults[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
};
static const iree_vm_module_signature_type_t kAppWorkArguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
};
static const iree_vm_module_signature_type_t kAppWorkResults[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
};
static const iree_vm_module_signature_type_t kLibraryWorkArguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
};
static const iree_vm_module_signature_type_t kLibraryWorkResults[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 1},
};
static const iree_vm_module_signature_type_t kInitializeArguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION, 0},
};

static const iree_vm_module_callable_type_declaration_t kAppCallableTypes[] = {
    {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments), 1, 0, 0},
      {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults), 1, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
    {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments), 1, 0, 0},
      {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults), 1, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
     0,
     0},
    {{{kAppWorkArguments, IREE_ARRAYSIZE(kAppWorkArguments), 0, 1, 1},
      {kAppWorkResults, IREE_ARRAYSIZE(kAppWorkResults), 0, 1, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
     1,
     0},
    {{{kInitializeArguments, IREE_ARRAYSIZE(kInitializeArguments), 1, 1, 1},
      {nullptr, 0, 0, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
     1,
     0},
};

static const iree_vm_module_callable_type_declaration_t kAlphaCallableTypes[] =
    {
        {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments), 1, 0, 0},
          {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults), 1, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
        {{{kLibraryWorkArguments, IREE_ARRAYSIZE(kLibraryWorkArguments), 0, 1,
           1},
          {kLibraryWorkResults, IREE_ARRAYSIZE(kLibraryWorkResults), 0, 1, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
         1,
         0},
};

static const iree_vm_module_callable_type_declaration_t kBetaCallableTypes[] = {
    {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments), 1, 0, 0},
      {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults), 1, 0, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE,
     0,
     0},
};

static const iree_vm_module_import_group_t kAppImportGroups[] = {
    {IREE_SVL("lib.alpha"), 0, 2},
    {IREE_SVL("lib.beta"), 2, 2},
    {IREE_SVL("missing.feature"), 4, 1},
};
static const iree_vm_module_import_declaration_t kAppImports[] = {
    {IREE_SVL("lib.alpha"), IREE_SVL("ref_work"), 2,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("lib.alpha"), IREE_SVL("work"), 2,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("lib.beta"), IREE_SVL("absent"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
    {IREE_SVL("lib.beta"), IREE_SVL("support"), 1,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("missing.feature"), IREE_SVL("optional"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
};
static const iree_vm_module_export_declaration_t kAppExports[] = {
    {IREE_SVL("initialize"), 3, 0, 0},
    {IREE_SVL("run"), 2, 1, 0},
    {IREE_SVL("run_alias"), 2, 1, 0},
};

static const iree_vm_module_import_group_t kAlphaImportGroups[] = {
    {IREE_SVL("root.app"), 0, 1},
};
static const iree_vm_module_import_declaration_t kAlphaImports[] = {
    {IREE_SVL("root.app"), IREE_SVL("run"), 1, IREE_VM_MODULE_IMPORT_FLAG_NONE,
     0},
};
static const iree_vm_module_export_declaration_t kAlphaExports[] = {
    // Library `initialize` names have no implicit lifecycle role.
    {IREE_SVL("initialize"), 1, 0, 0},
    {IREE_SVL("ref_work"), 1, 0, 0},
    {IREE_SVL("work"), 1, 0, 0},
};

static const iree_vm_module_export_declaration_t kBetaExports[] = {
    {IREE_SVL("support"), 0, 0, 0},
};

static const iree_vm_ref_type_t kAppRefTypes[] = {
    kSharedRefType,
};
static const iree_vm_ref_type_t kAlphaRefTypes[] = {
    kAuxiliaryRefType,
    kSharedRefType,
};

static const TableModuleDefinition kAppDefinition =
    TableModuleDefinition(IREE_SVL("root.app"), 2, 17)
        .WithRefTypes(kAppRefTypes)
        .WithCallableTypes(kAppCallableTypes)
        .WithImports(kAppImportGroups, kAppImports)
        .WithExports(kAppExports);
static const TableModuleDefinition kAlphaDefinition =
    TableModuleDefinition(IREE_SVL("lib.alpha"), 1, 1)
        .WithRefTypes(kAlphaRefTypes)
        .WithCallableTypes(kAlphaCallableTypes)
        .WithImports(kAlphaImportGroups, kAlphaImports)
        .WithExports(kAlphaExports);
static const TableModuleDefinition kBetaDefinition =
    TableModuleDefinition(IREE_SVL("lib.beta"), 1)
        .WithCallableTypes(kBetaCallableTypes)
        .WithExports(kBetaExports);

TEST(VMProgramTest, LinksOneSlabAndPrecomputesComposition) {
  TableModule app;
  TableModule alpha;
  TableModule beta;
  IREE_ASSERT_OK(app.Publish(kAppDefinition));
  IREE_ASSERT_OK(alpha.Publish(kAlphaDefinition));
  IREE_ASSERT_OK(beta.Publish(kBetaDefinition));

  iree_vm_module_t* libraries[] = {beta.module(), alpha.module()};
  CountingAllocator allocator_a;
  CountingAllocator allocator_b;
  iree_vm_program_t* program_a = nullptr;
  iree_vm_program_t* program_b = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {app.module(), iree_vm_module_span_from_array(libraries)},
      allocator_a.allocator(), &program_a));
  IREE_ASSERT_OK(iree_vm_program_create(
      {app.module(), iree_vm_module_span_from_array(libraries)},
      allocator_b.allocator(), &program_b));
  ASSERT_NE(program_a, nullptr);
  ASSERT_NE(program_b, nullptr);
  EXPECT_EQ(allocator_a.allocation_count(), 1u);
  EXPECT_EQ(allocator_b.allocation_count(), 1u);

  ASSERT_EQ(program_a->linked_module_count, 3u);
  EXPECT_EQ(ToStringView(program_a->linked_modules[0].module->descriptor->name),
            "lib.alpha");
  EXPECT_EQ(ToStringView(program_a->linked_modules[1].module->descriptor->name),
            "lib.beta");
  EXPECT_EQ(ToStringView(program_a->linked_modules[2].module->descriptor->name),
            "root.app");
  EXPECT_EQ(program_a->executable_module_ordinal, 2u);
  EXPECT_EQ(program_a->callable_mapping_count, 7u);
  EXPECT_EQ(program_a->callable_abi_count, 3u);

  iree_host_size_t app_offset = 0;
  ASSERT_TRUE(iree_host_size_checked_align(1, iree_max_align_t, &app_offset));
  EXPECT_EQ(program_a->linked_modules[0].process_storage_offset, 0u);
  EXPECT_EQ(program_a->linked_modules[1].process_storage_offset, UINT32_MAX);
  EXPECT_EQ(program_a->linked_modules[2].process_storage_offset, app_offset);
  EXPECT_EQ(program_a->process_storage_size, app_offset + 17);

  const uint32_t alpha_callback_mapping = program_a->callable_mappings[0];
  const uint32_t alpha_work_mapping = program_a->callable_mappings[1];
  const uint32_t beta_callback_mapping = program_a->callable_mappings[2];
  const uint32_t app_callback_mapping = program_a->callable_mappings[3];
  const uint32_t app_permissive_callback_mapping =
      program_a->callable_mappings[4];
  const uint32_t app_work_mapping = program_a->callable_mappings[5];
  const uint32_t app_initialize_mapping = program_a->callable_mappings[6];
  EXPECT_EQ(iree_vm_program_callable_token(app_callback_mapping),
            iree_vm_program_callable_token(alpha_callback_mapping));
  EXPECT_EQ(iree_vm_program_callable_token(app_callback_mapping),
            iree_vm_program_callable_token(beta_callback_mapping));
  EXPECT_EQ(iree_vm_program_callable_token(app_work_mapping),
            iree_vm_program_callable_token(alpha_work_mapping));
  EXPECT_NE(iree_vm_program_callable_token(app_initialize_mapping),
            iree_vm_program_callable_token(app_work_mapping));
  EXPECT_EQ(iree_vm_program_callable_token(app_permissive_callback_mapping),
            iree_vm_program_callable_token(beta_callback_mapping));
  EXPECT_TRUE(
      iree_vm_program_callable_may_yield(app_permissive_callback_mapping));
  EXPECT_FALSE(iree_vm_program_callable_may_yield(beta_callback_mapping));

  const uint32_t callback_token =
      iree_vm_program_callable_token(app_callback_mapping);
  const iree_vm_program_callable_abi_t* callback_abi =
      iree_vm_program_resolve_callable_abi(program_a, callback_token);
  ASSERT_NE(callback_abi, nullptr);
  EXPECT_EQ(callback_abi->argument_counts.value_count, 1u);
  EXPECT_EQ(callback_abi->result_counts.value_count, 1u);
  ASSERT_NE(callback_abi->value_arguments, nullptr);
  EXPECT_EQ(callback_abi->value_arguments[0].variant_offset, 0u);
  EXPECT_EQ(callback_abi->value_arguments[0].payload_mask, UINT32_MAX);
  EXPECT_EQ(callback_abi->value_arguments[0].variant_metadata,
            (IREE_VM_SCALAR_TYPE_I32 << 2) | IREE_VM_VARIANT_TAG_SCALAR);
  EXPECT_EQ(callback_abi->root_layout.storage_size, 2 * sizeof(uint64_t));
  EXPECT_EQ(iree_vm_program_resolve_callable_abi(program_a, 0), nullptr);
  EXPECT_EQ(iree_vm_program_resolve_callable_abi(
                program_a, program_a->callable_abi_count + 1),
            nullptr);

  const uint64_t* app_imports = program_a->linked_modules[2].import_target_bits;
  ASSERT_NE(app_imports, nullptr);
  EXPECT_NE(app_imports[0], 0u);
  EXPECT_EQ(app_imports[0], app_imports[1]);
  EXPECT_EQ(iree_vm_program_target_module_ordinal(app_imports[0]), 0u);
  EXPECT_EQ(iree_vm_program_target_function_ordinal(app_imports[0]), 0u);
  EXPECT_TRUE(iree_vm_program_target_may_yield(app_imports[0]));
  EXPECT_EQ(app_imports[2], 0u);
  EXPECT_EQ(iree_vm_program_target_module_ordinal(app_imports[3]), 1u);
  EXPECT_FALSE(iree_vm_program_target_may_yield(app_imports[3]));
  EXPECT_EQ(app_imports[4], 0u);
  const uint64_t* alpha_imports =
      program_a->linked_modules[0].import_target_bits;
  ASSERT_NE(alpha_imports, nullptr);
  EXPECT_EQ(iree_vm_program_target_module_ordinal(alpha_imports[0]), 2u);
  EXPECT_EQ(iree_vm_program_target_function_ordinal(alpha_imports[0]), 1u);
  EXPECT_NE(program_a->linked_modules[2].import_target_bits,
            program_b->linked_modules[2].import_target_bits);

  EXPECT_NE(program_a->initializer.target_bits, 0u);
  EXPECT_EQ(
      iree_vm_program_target_module_ordinal(program_a->initializer.target_bits),
      2u);
  EXPECT_EQ(iree_vm_program_target_function_ordinal(
                program_a->initializer.target_bits),
            0u);
  EXPECT_TRUE(
      iree_vm_program_target_may_yield(program_a->initializer.target_bits));
  ASSERT_NE(program_a->initializer.callable_abi, nullptr);
  EXPECT_EQ(iree_vm_program_callable_abi_argument_count(
                program_a->initializer.callable_abi),
            3u);
  EXPECT_EQ(program_a->initializer.callable_abi->argument_counts.value_count,
            1u);
  EXPECT_EQ(program_a->initializer.callable_abi->argument_counts.ref_count, 1u);
  EXPECT_EQ(program_a->initializer.callable_abi->argument_counts.function_count,
            1u);
  ASSERT_NE(program_a->initializer.callable_abi->value_arguments, nullptr);
  EXPECT_EQ(
      program_a->initializer.callable_abi->value_arguments[0].variant_offset,
      0u);
  ASSERT_NE(program_a->initializer.callable_abi->ref_arguments, nullptr);
  EXPECT_EQ(program_a->initializer.callable_abi->ref_arguments[0].type,
            kSharedRefType);
  EXPECT_EQ(
      program_a->initializer.callable_abi->ref_arguments[0].variant_offset,
      sizeof(iree_vm_variant_t));
  ASSERT_NE(program_a->initializer.callable_abi->function_arguments, nullptr);
  EXPECT_EQ(iree_vm_program_callable_token(
                program_a->initializer.callable_abi->function_arguments[0]
                    .callable_mapping),
            callback_token);
  EXPECT_EQ(
      program_a->initializer.callable_abi->function_arguments[0].variant_offset,
      2 * sizeof(iree_vm_variant_t));

  iree_vm_export_t run_export = {};
  iree_vm_export_t alias_export = {};
  IREE_ASSERT_OK(
      iree_vm_module_export_by_ordinal(app.module(), 1, &run_export));
  IREE_ASSERT_OK(
      iree_vm_module_export_by_ordinal(app.module(), 2, &alias_export));
  iree_vm_function_ref_t run_function = iree_vm_function_ref_null();
  iree_vm_function_ref_t alias_function = iree_vm_function_ref_null();
  IREE_ASSERT_OK(
      iree_vm_function_ref_from_export(program_a, run_export, &run_function));
  IREE_ASSERT_OK(iree_vm_function_ref_from_export(program_a, alias_export,
                                                  &alias_function));
  EXPECT_EQ(run_function.program_bits, (uint64_t)(uintptr_t)program_a);
  EXPECT_EQ(run_function.target_bits, alias_function.target_bits);
  EXPECT_TRUE(iree_vm_program_function_ref_matches(
      program_a, run_function, &program_a->linked_modules[2], 2));
  EXPECT_TRUE(iree_vm_program_function_ref_matches(
      program_a, iree_vm_function_ref_null(), &program_a->linked_modules[2],
      2));
  EXPECT_FALSE(iree_vm_program_function_ref_matches(
      program_b, run_function, &program_b->linked_modules[2], 2));
  EXPECT_FALSE(iree_vm_program_function_ref_matches(
      program_a, run_function, &program_a->linked_modules[2], 0));

  iree_host_size_t module_ordinal = IREE_HOST_SIZE_MAX;
  EXPECT_EQ(iree_vm_program_lookup_linked_module(program_a, IREE_SV("lib.beta"),
                                                 &module_ordinal),
            &program_a->linked_modules[1]);
  EXPECT_EQ(module_ordinal, 1u);
  EXPECT_EQ(iree_vm_program_lookup_linked_module(program_a, IREE_SV("absent"),
                                                 &module_ordinal),
            nullptr);

  TableModule foreign_app;
  IREE_ASSERT_OK(foreign_app.Publish(kAppDefinition));
  iree_vm_export_t foreign_export = {};
  IREE_ASSERT_OK(iree_vm_module_export_by_ordinal(foreign_app.module(), 1,
                                                  &foreign_export));
  iree_vm_function_ref_t untouched = {123, 456};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_function_ref_from_export(program_a, foreign_export, &untouched));
  EXPECT_EQ(untouched.program_bits, 123u);
  EXPECT_EQ(untouched.target_bits, 456u);

  app.ReleaseOwner();
  alpha.ReleaseOwner();
  beta.ReleaseOwner();
  EXPECT_EQ(app.destruction_count(), 0);
  EXPECT_EQ(alpha.destruction_count(), 0);
  EXPECT_EQ(beta.destruction_count(), 0);
  iree_vm_program_release(program_a);
  EXPECT_EQ(allocator_a.free_count(), 1u);
  EXPECT_EQ(app.destruction_count(), 0);
  iree_vm_program_release(program_b);
  EXPECT_EQ(allocator_b.free_count(), 1u);
  EXPECT_EQ(app.destruction_count(), 1);
  EXPECT_EQ(alpha.destruction_count(), 1);
  EXPECT_EQ(beta.destruction_count(), 1);
}

void ExpectReverseOrderedComposition(iree_host_size_t module_count) {
  ASSERT_GE(module_count, 1u);
  ASSERT_LE(module_count, 17u);
  std::array<std::array<char, 16>, 17> names = {};
  std::array<TableModuleDefinition, 17> definitions = {};
  std::array<TableModule, 17> modules;
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    std::snprintf(names[i].data(), names[i].size(), "module.%02zu",
                  static_cast<size_t>(i));
    definitions[i].descriptor.name = iree_make_cstring_view(names[i].data());
    definitions[i].descriptor.flags = IREE_VM_MODULE_FLAG_LINKABLE;
    IREE_ASSERT_OK(modules[i].Publish(definitions[i]));
  }
  std::array<iree_vm_module_t*, 16> libraries = {};
  for (iree_host_size_t i = 0; i + 1 < module_count; ++i) {
    libraries[i] = modules[module_count - i - 2].module();
  }
  CountingAllocator allocator;
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {modules[module_count - 1].module(),
       iree_vm_module_span_from_ptr(libraries.data(), module_count - 1)},
      allocator.allocator(), &program));
  ASSERT_NE(program, nullptr);
  EXPECT_EQ(allocator.allocation_count(), 1u);
  ASSERT_EQ(program->linked_module_count, module_count);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    EXPECT_EQ(ToStringView(program->linked_modules[i].module->descriptor->name),
              names[i].data());
    EXPECT_EQ(program->linked_modules[i].process_storage_offset, UINT32_MAX);
  }
  EXPECT_EQ(program->executable_module_ordinal, module_count - 1);
  EXPECT_EQ(program->process_storage_size, 0u);
  iree_vm_program_release(program);
  EXPECT_EQ(allocator.free_count(), 1u);
}

TEST(VMProgramTest, SortsInsertionAndHeapBoundariesWithoutTransientStorage) {
  ExpectReverseOrderedComposition(16);
  ExpectReverseOrderedComposition(17);
}

TEST(VMProgramTest, LeavesInputsUnownedWhenSlabAllocationFails) {
  TableModule app;
  TableModule alpha;
  TableModule beta;
  IREE_ASSERT_OK(app.Publish(kAppDefinition));
  IREE_ASSERT_OK(alpha.Publish(kAlphaDefinition));
  IREE_ASSERT_OK(beta.Publish(kBetaDefinition));
  iree_vm_module_t* libraries[] = {beta.module(), alpha.module()};
  CountingAllocator allocator(/*fail_at_allocation=*/1);
  iree_vm_program_t* program = reinterpret_cast<iree_vm_program_t*>(1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_vm_program_create(
          {app.module(), iree_vm_module_span_from_array(libraries)},
          allocator.allocator(), &program));
  EXPECT_EQ(program, nullptr);
  EXPECT_EQ(allocator.allocation_count(), 1u);
  EXPECT_EQ(allocator.free_count(), 0u);
  EXPECT_EQ(app.destruction_count(), 0);
  EXPECT_EQ(alpha.destruction_count(), 0);
  EXPECT_EQ(beta.destruction_count(), 0);
}

//===----------------------------------------------------------------------===//
// Rejected compositions
//===----------------------------------------------------------------------===//

static const iree_vm_module_import_group_t kMissingModuleGroups[] = {
    {IREE_SVL("absent"), 0, 1},
};
static const iree_vm_module_import_declaration_t kMissingModuleImports[] = {
    {IREE_SVL("absent"), IREE_SVL("work"), 0, IREE_VM_MODULE_IMPORT_FLAG_NONE,
     0},
};
static const TableModuleDefinition kMissingModuleDefinition =
    TableModuleDefinition(IREE_SVL("missing.source"))
        .WithCallableTypes(kBetaCallableTypes)
        .WithImports(kMissingModuleGroups, kMissingModuleImports);

static const iree_vm_module_import_group_t kMissingExportGroups[] = {
    {IREE_SVL("target.empty"), 0, 1},
};
static const iree_vm_module_import_declaration_t kMissingExportImports[] = {
    {IREE_SVL("target.empty"), IREE_SVL("work"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
};
static const TableModuleDefinition kMissingExportSourceDefinition =
    TableModuleDefinition(IREE_SVL("source.empty"))
        .WithCallableTypes(kBetaCallableTypes)
        .WithImports(kMissingExportGroups, kMissingExportImports);
static const TableModuleDefinition kEmptyTargetDefinition(
    IREE_SVL("target.empty"));

static const iree_vm_module_signature_type_t kI64Results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0},
};
static const iree_vm_module_callable_type_declaration_t
    kIncompatibleCallableTypes[] = {
        {{{nullptr, 0, 0, 0, 0},
          {kI64Results, IREE_ARRAYSIZE(kI64Results), 1, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE,
         0,
         0},
};
static const iree_vm_module_import_group_t kIncompatibleImportGroups[] = {
    {IREE_SVL("target"), 0, 1},
};
static const iree_vm_module_import_declaration_t kIncompatibleImports[] = {
    {IREE_SVL("target"), IREE_SVL("work"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
};
static const TableModuleDefinition kIncompatibleSourceDefinition =
    TableModuleDefinition(IREE_SVL("source"))
        .WithCallableTypes(kBetaCallableTypes)
        .WithImports(kIncompatibleImportGroups, kIncompatibleImports);
static const iree_vm_module_export_declaration_t kIncompatibleExports[] = {
    {IREE_SVL("work"), 0, 0, 0},
};
static const TableModuleDefinition kIncompatibleTargetDefinition =
    TableModuleDefinition(IREE_SVL("target"), 1)
        .WithCallableTypes(kIncompatibleCallableTypes)
        .WithExports(kIncompatibleExports);

static const iree_vm_module_callable_type_declaration_t
    kYieldingCallableTypes[] = {
        {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments), 1, 0, 0},
          {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults), 1, 0, 0}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD,
         0,
         0},
};
static const iree_vm_module_import_group_t kYieldImportGroups[] = {
    {IREE_SVL("target.yield"), 0, 1},
};
static const iree_vm_module_import_declaration_t kYieldImports[] = {
    {IREE_SVL("target.yield"), IREE_SVL("work"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
};
static const TableModuleDefinition kYieldSourceDefinition =
    TableModuleDefinition(IREE_SVL("source.yield"))
        .WithCallableTypes(kBetaCallableTypes)
        .WithImports(kYieldImportGroups, kYieldImports);
static const iree_vm_module_export_declaration_t kYieldExports[] = {
    {IREE_SVL("work"), 0, 0, 0},
};
static const TableModuleDefinition kYieldTargetDefinition =
    TableModuleDefinition(IREE_SVL("target.yield"), 1)
        .WithCallableTypes(kYieldingCallableTypes)
        .WithExports(kYieldExports);

void ExpectLinkFailure(
    iree_status_code_t expected_code,
    const TableModuleDefinition* source_definition,
    const TableModuleDefinition* target_definition = nullptr) {
  TableModule source;
  TableModule target;
  IREE_ASSERT_OK(source.Publish(*source_definition));
  iree_vm_module_span_t libraries = iree_vm_module_span_empty();
  iree_vm_module_t* target_module = nullptr;
  if (target_definition) {
    IREE_ASSERT_OK(target.Publish(*target_definition));
    target_module = target.module();
    libraries = iree_vm_module_span_from_ptr(&target_module, 1);
  }
  iree_vm_program_t* program = nullptr;
  IREE_EXPECT_STATUS_IS(
      expected_code, iree_vm_program_create({source.module(), libraries},
                                            iree_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);
}

TEST(VMProgramTest, RejectsMissingAndIncompatibleImports) {
  ExpectLinkFailure(IREE_STATUS_NOT_FOUND, &kMissingModuleDefinition);
  ExpectLinkFailure(IREE_STATUS_NOT_FOUND, &kMissingExportSourceDefinition,
                    &kEmptyTargetDefinition);
  ExpectLinkFailure(IREE_STATUS_INVALID_ARGUMENT,
                    &kIncompatibleSourceDefinition,
                    &kIncompatibleTargetDefinition);
  ExpectLinkFailure(IREE_STATUS_INVALID_ARGUMENT, &kYieldSourceDefinition,
                    &kYieldTargetDefinition);
}

static const TableModuleDefinition kDuplicateDefinition(IREE_SVL("duplicate"));
static const TableModuleDefinition kNonLinkableDefinition(
    IREE_SVL("inspection.only"), 0, 0, IREE_VM_MODULE_FLAG_NONE);
static const TableModuleDefinition kLargeStateDefinition(
    IREE_SVL("storage.large"), 0, UINT32_MAX);
static const TableModuleDefinition kTailStateDefinition(
    IREE_SVL("storage.tail"), 0, 1);
static const iree_vm_module_export_declaration_t kInvalidInitializerExports[] =
    {
        {IREE_SVL("initialize"), 0, 0, 0},
};
static const TableModuleDefinition kInvalidInitializerDefinition =
    TableModuleDefinition(IREE_SVL("invalid.initializer"), 1)
        .WithCallableTypes(kBetaCallableTypes)
        .WithExports(kInvalidInitializerExports);

TEST(VMProgramTest, RejectsInvalidCompositionAndInitializer) {
  {
    TableModule executable;
    TableModule duplicate;
    IREE_ASSERT_OK(executable.Publish(kDuplicateDefinition));
    IREE_ASSERT_OK(duplicate.Publish(kDuplicateDefinition));
    iree_vm_module_t* libraries[] = {duplicate.module()};
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ALREADY_EXISTS,
        iree_vm_program_create(
            {executable.module(), iree_vm_module_span_from_array(libraries)},
            iree_allocator_system(), &program));
    EXPECT_EQ(program, nullptr);
  }
  {
    TableModule executable;
    IREE_ASSERT_OK(executable.Publish(kInvalidInitializerDefinition));
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_program_create(
            {executable.module(), iree_vm_module_span_empty()},
            iree_allocator_system(), &program));
    EXPECT_EQ(program, nullptr);
  }
  {
    TableModule module;
    IREE_ASSERT_OK(module.Publish(kNonLinkableDefinition));
    CountingAllocator allocator;
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_program_create({module.module(), iree_vm_module_span_empty()},
                               allocator.allocator(), &program));
    EXPECT_EQ(program, nullptr);
    EXPECT_EQ(allocator.allocation_count(), 0u);
  }
  {
    TableModule large;
    TableModule tail;
    IREE_ASSERT_OK(large.Publish(kLargeStateDefinition));
    IREE_ASSERT_OK(tail.Publish(kTailStateDefinition));
    iree_vm_module_t* libraries[] = {tail.module()};
    CountingAllocator allocator;
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_vm_program_create(
            {large.module(), iree_vm_module_span_from_array(libraries)},
            allocator.allocator(), &program));
    EXPECT_EQ(program, nullptr);
    EXPECT_EQ(allocator.allocation_count(), 1u);
    EXPECT_EQ(allocator.free_count(), 1u);
  }
}

TEST(VMProgramTest, RejectsInvalidPublicInputsBeforeProviderAccess) {
  iree_vm_program_t* program = reinterpret_cast<iree_vm_program_t*>(1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_program_create({nullptr, iree_vm_module_span_empty()},
                             iree_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);

  TableModule executable;
  IREE_ASSERT_OK(executable.Publish(kDuplicateDefinition));
  iree_vm_module_span_t malformed_span = {nullptr, 1};
  program = reinterpret_cast<iree_vm_program_t*>(1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_program_create({executable.module(), malformed_span},
                             iree_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);
}

}  // namespace
