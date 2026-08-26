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
#include "iree/vm/program_storage.h"
#include "iree/vm/program_test_provider.h"

namespace {

std::string_view ToStringView(iree_string_view_t value) {
  return std::string_view(value.data ? value.data : "", value.size);
}

struct CountingAllocator {
  // Allocator performing the actual memory operations.
  iree_allocator_t delegate;
  // Number of allocation-like commands forwarded.
  iree_host_size_t allocation_count;
  // Number of free commands forwarded.
  iree_host_size_t free_count;
  // One-based allocation command to fail, or zero to never fail.
  iree_host_size_t fail_at_allocation;
};

iree_status_t CountingAllocatorControl(void* self,
                                       iree_allocator_command_t command,
                                       const void* params, void** inout_ptr) {
  auto* allocator = static_cast<CountingAllocator*>(self);
  switch (command) {
    case IREE_ALLOCATOR_COMMAND_MALLOC:
    case IREE_ALLOCATOR_COMMAND_CALLOC:
    case IREE_ALLOCATOR_COMMAND_REALLOC:
      ++allocator->allocation_count;
      if (allocator->allocation_count == allocator->fail_at_allocation) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
      break;
    case IREE_ALLOCATOR_COMMAND_FREE:
      ++allocator->free_count;
      break;
    default:
      break;
  }
  return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                 inout_ptr);
}

iree_allocator_t MakeCountingAllocator(CountingAllocator* allocator) {
  return iree_allocator_t{allocator, CountingAllocatorControl};
}

struct TestModule {
  // Published native C module owner.
  iree_vm_module_t* module = nullptr;
  // Number of final provider destruction callbacks.
  int destruction_count = 0;

  TestModule() = default;
  TestModule(const TestModule&) = delete;
  TestModule& operator=(const TestModule&) = delete;

  ~TestModule() { ReleaseOwner(); }

  void Create(const iree_vm_program_test_module_definition_t* definition) {
    IREE_ASSERT_OK(iree_vm_program_test_module_create(
        definition, &destruction_count, iree_allocator_system(), &module));
  }

  void ReleaseOwner() {
    iree_vm_module_release(module);
    module = nullptr;
  }
};

TEST(VMProgramTest, PacksTheCompleteTargetDomainWithoutTagOverlap) {
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
    {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments)},
      {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults)}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE},
    {{{kAppWorkArguments, IREE_ARRAYSIZE(kAppWorkArguments)},
      {kAppWorkResults, IREE_ARRAYSIZE(kAppWorkResults)}},
     IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD},
    {{{kInitializeArguments, IREE_ARRAYSIZE(kInitializeArguments)},
      {nullptr, 0}},
     IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD},
    {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments)},
      {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults)}},
     IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD},
};

static const iree_vm_module_callable_type_declaration_t kAlphaCallableTypes[] =
    {
        {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments)},
          {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE},
        {{{kLibraryWorkArguments, IREE_ARRAYSIZE(kLibraryWorkArguments)},
          {kLibraryWorkResults, IREE_ARRAYSIZE(kLibraryWorkResults)}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD},
};

static const iree_vm_module_callable_type_declaration_t kBetaCallableTypes[] = {
    {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments)},
      {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults)}},
     IREE_VM_CALLABLE_TYPE_FLAG_NONE},
};

static const iree_vm_module_import_group_t kAppImportGroups[] = {
    {IREE_SVL("lib.alpha"), 0, 2},
    {IREE_SVL("lib.beta"), 2, 1},
    {IREE_SVL("missing.feature"), 3, 1},
};
static const iree_vm_module_import_declaration_t kAppImports[] = {
    {IREE_SVL("lib.alpha"), IREE_SVL("ref_work"), 1,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("lib.alpha"), IREE_SVL("work"), 1,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("lib.beta"), IREE_SVL("support"), 3,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
    {IREE_SVL("missing.feature"), IREE_SVL("optional"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
};
static const iree_vm_module_export_declaration_t kAppExports[] = {
    {IREE_SVL("initialize"), 2, 0, 0},
    {IREE_SVL("run"), 1, 1, 0},
    {IREE_SVL("run_alias"), 1, 1, 0},
};

static const iree_vm_module_import_group_t kAlphaImportGroups[] = {
    {IREE_SVL("root.app"), 0, 1},
};
static const iree_vm_module_import_declaration_t kAlphaImports[] = {
    {IREE_SVL("root.app"), IREE_SVL("run"), 1, IREE_VM_MODULE_IMPORT_FLAG_NONE,
     0},
};
static const iree_vm_module_export_declaration_t kAlphaExports[] = {
    // A library initializer-shaped name is an ordinary export and is ignored.
    {IREE_SVL("initialize"), 1, 0, 0},
    {IREE_SVL("ref_work"), 1, 0, 0},
    {IREE_SVL("work"), 1, 0, 0},
};

static const iree_vm_module_export_declaration_t kBetaExports[] = {
    {IREE_SVL("support"), 0, 0, 0},
};

static const iree_vm_ref_type_t kAppRefTypes[] = {
    iree_vm_program_test_shared_ref_type(),
};
static const iree_vm_ref_type_t kAlphaRefTypes[] = {
    iree_vm_program_test_auxiliary_ref_type(),
    iree_vm_program_test_shared_ref_type(),
};

static const iree_vm_program_test_module_definition_t kAppDefinition = {
    {IREE_SVL("root.app"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {kAppRefTypes, IREE_ARRAYSIZE(kAppRefTypes)},
     {2, IREE_ARRAYSIZE(kAppCallableTypes), IREE_ARRAYSIZE(kAppImportGroups),
      IREE_ARRAYSIZE(kAppImports), IREE_ARRAYSIZE(kAppExports), 0},
     17},
    kAppImportGroups,
    kAppImports,
    kAppExports,
    kAppCallableTypes,
};

static const iree_vm_program_test_module_definition_t kAlphaDefinition = {
    {IREE_SVL("lib.alpha"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {kAlphaRefTypes, IREE_ARRAYSIZE(kAlphaRefTypes)},
     {1, IREE_ARRAYSIZE(kAlphaCallableTypes),
      IREE_ARRAYSIZE(kAlphaImportGroups), IREE_ARRAYSIZE(kAlphaImports),
      IREE_ARRAYSIZE(kAlphaExports), 0},
     1},
    kAlphaImportGroups,
    kAlphaImports,
    kAlphaExports,
    kAlphaCallableTypes,
};

static const iree_vm_program_test_module_definition_t kBetaDefinition = {
    {IREE_SVL("lib.beta"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {1, IREE_ARRAYSIZE(kBetaCallableTypes), 0, 0, IREE_ARRAYSIZE(kBetaExports),
      0},
     0},
    nullptr,
    nullptr,
    kBetaExports,
    kBetaCallableTypes,
};

TEST(VMProgramTest, LinksOneImmutableSlabAndPrecomputesTargets) {
  TestModule app;
  TestModule alpha;
  TestModule beta;
  app.Create(&kAppDefinition);
  alpha.Create(&kAlphaDefinition);
  beta.Create(&kBetaDefinition);

  iree_vm_module_t* libraries[] = {beta.module, alpha.module};
  CountingAllocator allocator = {iree_allocator_system(), 0, 0, 0};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {app.module, iree_vm_module_span_from_array(libraries)},
      MakeCountingAllocator(&allocator), &program));
  ASSERT_NE(program, nullptr);
  EXPECT_EQ(allocator.allocation_count, 2u);
  EXPECT_EQ(allocator.free_count, 1u);

  ASSERT_EQ(program->linked_module_count, 3u);
  EXPECT_EQ(ToStringView(program->linked_modules[0].module->descriptor->name),
            "lib.alpha");
  EXPECT_EQ(ToStringView(program->linked_modules[1].module->descriptor->name),
            "lib.beta");
  EXPECT_EQ(ToStringView(program->linked_modules[2].module->descriptor->name),
            "root.app");
  EXPECT_EQ(program->executable_module_ordinal, 2u);

  iree_host_size_t app_offset = 0;
  ASSERT_TRUE(iree_host_size_checked_align(1, iree_max_align_t, &app_offset));
  EXPECT_EQ(program->linked_modules[0].process_storage_offset, 0u);
  EXPECT_EQ(program->linked_modules[1].process_storage_offset, UINT32_MAX);
  EXPECT_EQ(program->linked_modules[2].process_storage_offset, app_offset);
  EXPECT_EQ(program->process_storage_size, app_offset + 17);

  const uint32_t alpha_callback_mapping = program->callable_mappings[0];
  const uint32_t alpha_work_mapping = program->callable_mappings[1];
  const uint32_t beta_callback_mapping = program->callable_mappings[2];
  const uint32_t app_callback_mapping = program->callable_mappings[3];
  const uint32_t app_work_mapping = program->callable_mappings[4];
  const uint32_t app_initialize_mapping = program->callable_mappings[5];
  const uint32_t app_permissive_callback_mapping =
      program->callable_mappings[6];
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

  const uint64_t* app_imports = program->linked_modules[2].import_target_bits;
  ASSERT_NE(app_imports, nullptr);
  EXPECT_NE(app_imports[0], 0u);
  EXPECT_EQ(app_imports[0], app_imports[1]);
  EXPECT_EQ(iree_vm_program_target_module_ordinal(app_imports[0]), 0u);
  EXPECT_EQ(iree_vm_program_target_function_ordinal(app_imports[0]), 0u);
  EXPECT_TRUE(iree_vm_program_target_may_yield(app_imports[0]));
  EXPECT_EQ(iree_vm_program_target_module_ordinal(app_imports[2]), 1u);
  EXPECT_FALSE(iree_vm_program_target_may_yield(app_imports[2]));
  EXPECT_EQ(app_imports[3], 0u);

  const uint64_t* alpha_imports = program->linked_modules[0].import_target_bits;
  ASSERT_NE(alpha_imports, nullptr);
  EXPECT_EQ(iree_vm_program_target_module_ordinal(alpha_imports[0]), 2u);
  EXPECT_EQ(iree_vm_program_target_function_ordinal(alpha_imports[0]), 1u);

  EXPECT_NE(program->initializer.target_bits, 0u);
  EXPECT_EQ(
      iree_vm_program_target_module_ordinal(program->initializer.target_bits),
      2u);
  EXPECT_EQ(
      iree_vm_program_target_function_ordinal(program->initializer.target_bits),
      0u);
  EXPECT_TRUE(
      iree_vm_program_target_may_yield(program->initializer.target_bits));
  EXPECT_EQ(program->initializer.arguments.count, 3u);
  EXPECT_EQ(program->initializer.argument_counts.value_count, 1u);
  EXPECT_EQ(program->initializer.argument_counts.ref_count, 1u);
  EXPECT_EQ(program->initializer.argument_counts.function_count, 1u);

  iree_vm_export_t run_export = {};
  iree_vm_export_t alias_export = {};
  IREE_ASSERT_OK(iree_vm_module_export_by_ordinal(app.module, 1, &run_export));
  IREE_ASSERT_OK(
      iree_vm_module_export_by_ordinal(app.module, 2, &alias_export));
  iree_vm_function_ref_t run_function = iree_vm_function_ref_null();
  iree_vm_function_ref_t alias_function = iree_vm_function_ref_null();
  IREE_ASSERT_OK(
      iree_vm_function_ref_from_export(program, run_export, &run_function));
  IREE_ASSERT_OK(
      iree_vm_function_ref_from_export(program, alias_export, &alias_function));
  EXPECT_EQ(run_function.program_bits, (uint64_t)(uintptr_t)program);
  EXPECT_EQ(run_function.target_bits, alias_function.target_bits);

  TestModule foreign_app;
  foreign_app.Create(&kAppDefinition);
  iree_vm_export_t foreign_export = {};
  IREE_ASSERT_OK(
      iree_vm_module_export_by_ordinal(foreign_app.module, 1, &foreign_export));
  iree_vm_function_ref_t untouched = {123, 456};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_function_ref_from_export(program, foreign_export, &untouched));
  EXPECT_EQ(untouched.program_bits, 123u);
  EXPECT_EQ(untouched.target_bits, 456u);

  app.ReleaseOwner();
  alpha.ReleaseOwner();
  beta.ReleaseOwner();
  EXPECT_EQ(app.destruction_count, 0);
  EXPECT_EQ(alpha.destruction_count, 0);
  EXPECT_EQ(beta.destruction_count, 0);
  iree_vm_program_release(program);
  EXPECT_EQ(allocator.free_count, 2u);
  EXPECT_EQ(app.destruction_count, 1);
  EXPECT_EQ(alpha.destruction_count, 1);
  EXPECT_EQ(beta.destruction_count, 1);
}

void ExpectReverseOrderedComposition(iree_host_size_t module_count) {
  ASSERT_GE(module_count, 1u);
  ASSERT_LE(module_count, 17u);

  std::array<std::array<char, 16>, 17> names = {};
  std::array<iree_vm_program_test_module_definition_t, 17> definitions = {};
  std::array<TestModule, 17> modules;
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    std::snprintf(names[i].data(), names[i].size(), "module.%02zu",
                  static_cast<size_t>(i));
    definitions[i].descriptor.name = iree_make_cstring_view(names[i].data());
    definitions[i].descriptor.flags = IREE_VM_MODULE_FLAG_LINKABLE;
    modules[i].Create(&definitions[i]);
  }

  std::array<iree_vm_module_t*, 16> libraries = {};
  for (iree_host_size_t i = 0; i + 1 < module_count; ++i) {
    libraries[i] = modules[module_count - i - 2].module;
  }
  CountingAllocator allocator = {iree_allocator_system(), 0, 0, 0};
  iree_vm_program_t* program = nullptr;
  IREE_ASSERT_OK(iree_vm_program_create(
      {modules[module_count - 1].module,
       iree_vm_module_span_from_ptr(libraries.data(), module_count - 1)},
      MakeCountingAllocator(&allocator), &program));
  ASSERT_NE(program, nullptr);
  EXPECT_EQ(allocator.allocation_count, 1u);
  ASSERT_EQ(program->linked_module_count, module_count);
  for (iree_host_size_t i = 0; i < module_count; ++i) {
    EXPECT_EQ(ToStringView(program->linked_modules[i].module->descriptor->name),
              names[i].data());
    EXPECT_EQ(program->linked_modules[i].process_storage_offset, UINT32_MAX);
  }
  EXPECT_EQ(program->executable_module_ordinal, module_count - 1);
  EXPECT_EQ(program->process_storage_size, 0u);
  iree_vm_program_release(program);
  EXPECT_EQ(allocator.free_count, 1u);
}

TEST(VMProgramTest, SortsInsertionAndHeapBoundariesWithoutTransientStorage) {
  ExpectReverseOrderedComposition(16);
  ExpectReverseOrderedComposition(17);
}

TEST(VMProgramTest, ReleasesSlabWhenCallableIndexAllocationFails) {
  TestModule app;
  TestModule alpha;
  TestModule beta;
  app.Create(&kAppDefinition);
  alpha.Create(&kAlphaDefinition);
  beta.Create(&kBetaDefinition);

  iree_vm_module_t* libraries[] = {beta.module, alpha.module};
  CountingAllocator allocator = {iree_allocator_system(), 0, 0, 2};
  iree_vm_program_t* program = reinterpret_cast<iree_vm_program_t*>(1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_vm_program_create(
          {app.module, iree_vm_module_span_from_array(libraries)},
          MakeCountingAllocator(&allocator), &program));
  EXPECT_EQ(program, nullptr);
  EXPECT_EQ(allocator.allocation_count, 2u);
  EXPECT_EQ(allocator.free_count, 1u);
  EXPECT_EQ(app.destruction_count, 0);
  EXPECT_EQ(alpha.destruction_count, 0);
  EXPECT_EQ(beta.destruction_count, 0);
}

//===----------------------------------------------------------------------===//
// Rejected compositions
//===----------------------------------------------------------------------===//

static const iree_vm_module_import_group_t kMissingImportGroups[] = {
    {IREE_SVL("absent"), 0, 1},
};
static const iree_vm_module_import_declaration_t kMissingImports[] = {
    {IREE_SVL("absent"), IREE_SVL("work"), 0, IREE_VM_MODULE_IMPORT_FLAG_NONE,
     0},
};
static const iree_vm_program_test_module_definition_t kMissingDefinition = {
    {IREE_SVL("missing.source"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {0, IREE_ARRAYSIZE(kBetaCallableTypes),
      IREE_ARRAYSIZE(kMissingImportGroups), IREE_ARRAYSIZE(kMissingImports), 0,
      0},
     0},
    kMissingImportGroups,
    kMissingImports,
    nullptr,
    kBetaCallableTypes,
};

static const iree_vm_module_signature_type_t kEmptyResults[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0},
};
static const iree_vm_module_callable_type_declaration_t
    kIncompatibleCallableTypes[] = {
        {{{nullptr, 0}, {kEmptyResults, IREE_ARRAYSIZE(kEmptyResults)}},
         IREE_VM_CALLABLE_TYPE_FLAG_NONE},
};
static const iree_vm_module_import_group_t kIncompatibleImportGroups[] = {
    {IREE_SVL("target"), 0, 1},
};
static const iree_vm_module_import_declaration_t kIncompatibleImports[] = {
    {IREE_SVL("target"), IREE_SVL("work"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL, 0},
};
static const iree_vm_program_test_module_definition_t
    kIncompatibleSourceDefinition = {
        {IREE_SVL("source"),
         IREE_VM_MODULE_FLAG_LINKABLE,
         {nullptr, 0},
         {0, IREE_ARRAYSIZE(kBetaCallableTypes),
          IREE_ARRAYSIZE(kIncompatibleImportGroups),
          IREE_ARRAYSIZE(kIncompatibleImports), 0, 0},
         0},
        kIncompatibleImportGroups,
        kIncompatibleImports,
        nullptr,
        kBetaCallableTypes,
};
static const iree_vm_module_export_declaration_t kIncompatibleExports[] = {
    {IREE_SVL("work"), 0, 0, 0},
};
static const iree_vm_program_test_module_definition_t
    kIncompatibleTargetDefinition = {
        {IREE_SVL("target"),
         IREE_VM_MODULE_FLAG_LINKABLE,
         {nullptr, 0},
         {1, IREE_ARRAYSIZE(kIncompatibleCallableTypes), 0, 0,
          IREE_ARRAYSIZE(kIncompatibleExports), 0},
         0},
        nullptr,
        nullptr,
        kIncompatibleExports,
        kIncompatibleCallableTypes,
};

static const iree_vm_module_callable_type_declaration_t
    kYieldingCallableTypes[] = {
        {{{kCallbackArguments, IREE_ARRAYSIZE(kCallbackArguments)},
          {kCallbackResults, IREE_ARRAYSIZE(kCallbackResults)}},
         IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD},
};
static const iree_vm_module_import_group_t kYieldImportGroups[] = {
    {IREE_SVL("target.yield"), 0, 1},
};
static const iree_vm_module_import_declaration_t kYieldImports[] = {
    {IREE_SVL("target.yield"), IREE_SVL("work"), 0,
     IREE_VM_MODULE_IMPORT_FLAG_NONE, 0},
};
static const iree_vm_program_test_module_definition_t kYieldSourceDefinition = {
    {IREE_SVL("source.yield"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {0, IREE_ARRAYSIZE(kBetaCallableTypes), IREE_ARRAYSIZE(kYieldImportGroups),
      IREE_ARRAYSIZE(kYieldImports), 0, 0},
     0},
    kYieldImportGroups,
    kYieldImports,
    nullptr,
    kBetaCallableTypes,
};
static const iree_vm_module_export_declaration_t kYieldExports[] = {
    {IREE_SVL("work"), 0, 0, 0},
};
static const iree_vm_program_test_module_definition_t kYieldTargetDefinition = {
    {IREE_SVL("target.yield"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {1, IREE_ARRAYSIZE(kYieldingCallableTypes), 0, 0,
      IREE_ARRAYSIZE(kYieldExports), 0},
     0},
    nullptr,
    nullptr,
    kYieldExports,
    kYieldingCallableTypes,
};

TEST(VMProgramTest, RejectsMissingAndIncompatibleImports) {
  {
    TestModule source;
    source.Create(&kMissingDefinition);
    CountingAllocator allocator = {iree_allocator_system(), 0, 0, 0};
    iree_vm_program_t* program = reinterpret_cast<iree_vm_program_t*>(1);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_NOT_FOUND,
        iree_vm_program_create({source.module, iree_vm_module_span_empty()},
                               MakeCountingAllocator(&allocator), &program));
    EXPECT_EQ(program, nullptr);
    EXPECT_EQ(allocator.allocation_count, 1u);
    EXPECT_EQ(allocator.free_count, 1u);
    EXPECT_EQ(source.destruction_count, 0);
    source.ReleaseOwner();
    EXPECT_EQ(source.destruction_count, 1);
  }
  {
    TestModule source;
    TestModule target;
    source.Create(&kIncompatibleSourceDefinition);
    target.Create(&kIncompatibleTargetDefinition);
    iree_vm_module_t* libraries[] = {target.module};
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_program_create(
            {source.module, iree_vm_module_span_from_array(libraries)},
            iree_allocator_system(), &program));
    EXPECT_EQ(program, nullptr);
  }
  {
    TestModule source;
    TestModule target;
    source.Create(&kYieldSourceDefinition);
    target.Create(&kYieldTargetDefinition);
    iree_vm_module_t* libraries[] = {target.module};
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_program_create(
            {source.module, iree_vm_module_span_from_array(libraries)},
            iree_allocator_system(), &program));
    EXPECT_EQ(program, nullptr);
  }
}

static const iree_vm_program_test_module_definition_t kDuplicateDefinition = {
    {IREE_SVL("duplicate"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {0, 0, 0, 0, 0, 0},
     0},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

static const iree_vm_program_test_module_definition_t kNonLinkableDefinition = {
    {IREE_SVL("inspection.only"),
     IREE_VM_MODULE_FLAG_NONE,
     {nullptr, 0},
     {0, 0, 0, 0, 0, 0},
     0},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

static const iree_vm_program_test_module_definition_t kLargeStateDefinition = {
    {IREE_SVL("storage.large"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {0, 0, 0, 0, 0, 0},
     UINT32_MAX},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

static const iree_vm_program_test_module_definition_t kTailStateDefinition = {
    {IREE_SVL("storage.tail"),
     IREE_VM_MODULE_FLAG_LINKABLE,
     {nullptr, 0},
     {0, 0, 0, 0, 0, 0},
     1},
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

static const iree_vm_module_export_declaration_t kInvalidInitializerExports[] =
    {
        {IREE_SVL("initialize"), 0, 0, 0},
};
static const iree_vm_program_test_module_definition_t
    kInvalidInitializerDefinition = {
        {IREE_SVL("invalid.initializer"),
         IREE_VM_MODULE_FLAG_LINKABLE,
         {nullptr, 0},
         {1, IREE_ARRAYSIZE(kBetaCallableTypes), 0, 0,
          IREE_ARRAYSIZE(kInvalidInitializerExports), 0},
         0},
        nullptr,
        nullptr,
        kInvalidInitializerExports,
        kBetaCallableTypes,
};

TEST(VMProgramTest, RejectsDuplicateModulesAndInitializerResults) {
  {
    TestModule executable;
    TestModule duplicate;
    executable.Create(&kDuplicateDefinition);
    duplicate.Create(&kDuplicateDefinition);
    iree_vm_module_t* libraries[] = {duplicate.module};
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ALREADY_EXISTS,
        iree_vm_program_create(
            {executable.module, iree_vm_module_span_from_array(libraries)},
            iree_allocator_system(), &program));
    EXPECT_EQ(program, nullptr);
  }
  {
    TestModule executable;
    executable.Create(&kInvalidInitializerDefinition);
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_program_create({executable.module, iree_vm_module_span_empty()},
                               iree_allocator_system(), &program));
    EXPECT_EQ(program, nullptr);
  }
}

TEST(VMProgramTest, RejectsNonLinkableModulesAndOversizeProcessStorage) {
  {
    TestModule module;
    module.Create(&kNonLinkableDefinition);
    CountingAllocator allocator = {iree_allocator_system(), 0, 0, 0};
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_vm_program_create({module.module, iree_vm_module_span_empty()},
                               MakeCountingAllocator(&allocator), &program));
    EXPECT_EQ(program, nullptr);
    EXPECT_EQ(allocator.allocation_count, 0u);
  }
  {
    TestModule large;
    TestModule tail;
    large.Create(&kLargeStateDefinition);
    tail.Create(&kTailStateDefinition);
    iree_vm_module_t* libraries[] = {tail.module};
    CountingAllocator allocator = {iree_allocator_system(), 0, 0, 0};
    iree_vm_program_t* program = nullptr;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        iree_vm_program_create(
            {large.module, iree_vm_module_span_from_array(libraries)},
            MakeCountingAllocator(&allocator), &program));
    EXPECT_EQ(program, nullptr);
    EXPECT_EQ(allocator.allocation_count, 1u);
    EXPECT_EQ(allocator.free_count, 1u);
  }
}

}  // namespace
