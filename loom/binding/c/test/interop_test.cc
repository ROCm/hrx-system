// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/interop.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/source.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

ContextPtr CreateTargetContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .target_environment = target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

SourcePtr CreateSource(const char* identifier, const char* contents) {
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(contents, std::strlen(contents)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

ModulePtr Deserialize(loomc_context_t* context, loomc_workspace_t* workspace,
                      const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_text_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

TEST(InteropTest, ProjectsVerifiedModuleAndOwnedSources) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("module.loom", R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());
  source.reset();

  loomc_module_interop_view_t view = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_get_interop_view(
      module.get(), loomc_allocator_system(), &view, &result));
  ResultPtr result_ptr(result);
  ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(view.module, nullptr);
  ASSERT_NE(view.source_table, nullptr);
  ASSERT_EQ(view.source_table->module, view.module);
  ASSERT_EQ(view.source_table->count, 1u);
  EXPECT_TRUE(iree_string_view_equal(view.source_table->entries[0].filename,
                                     IREE_SV("module.loom")));
  EXPECT_FALSE(iree_string_view_is_empty(view.source_table->entries[0].source));
}

TEST(InteropTest, RejectsStructurallyInvalidModuleBeforeProjection) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("invalid.loom", R"(
func.def @missing_result(%value: i32) -> (i32, i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());

  loomc_module_interop_view_t view = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_get_interop_view(
      module.get(), loomc_allocator_system(), &view, &result));
  ResultPtr result_ptr(result);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(view.module, nullptr);
  EXPECT_EQ(view.source_table, nullptr);

  bool found_structural_error = false;
  for (loomc_host_size_t i = 0;
       i < loomc_result_diagnostic_count(result_ptr.get()); ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result_ptr.get(), i);
    found_structural_error |=
        diagnostic != nullptr &&
        loomc_string_view_equal(diagnostic->code,
                                loomc_make_cstring_view("STRUCTURE/008"));
  }
  EXPECT_TRUE(found_structural_error);
}

TEST(InteropTest, MutableProjectionReturnsToPublicVerification) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("module.loom", R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());

  loomc_module_mutable_interop_view_t mutable_view =
      loomc_module_get_mutable_interop_view(module.get());
  ASSERT_NE(mutable_view.module, nullptr);
  ASSERT_NE(mutable_view.source_table, nullptr);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(mutable_view.module,
                                           IREE_SV("adapted"), &name_id));
  mutable_view.module->name_id = name_id;

  loomc_module_interop_view_t verified_view = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_get_interop_view(
      module.get(), loomc_allocator_system(), &verified_view, &result));
  ResultPtr result_ptr(result);
  ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(verified_view.module, mutable_view.module);
  EXPECT_EQ(verified_view.source_table, mutable_view.source_table);
  EXPECT_TRUE(iree_string_view_equal(
      loom_string_table_get(&verified_view.module->strings,
                            verified_view.module->name_id),
      IREE_SV("adapted")));
}

TEST(InteropTest, CreatesOwnedNativeTargetEnvironment) {
  const loom_target_provider_set_t provider_set = {};

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  const loom_target_environment_t* native_environment =
      loomc_target_environment_get_interop_view(target_environment_ptr.get());
  ASSERT_NE(native_environment, nullptr);
  EXPECT_EQ(native_environment->provider_set, &provider_set);
}

TEST(InteropTest, BuildsNativeSourceLowPassProgram) {
  const loom_target_provider_set_t provider_set = {};

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  ContextPtr context = CreateTargetContext(target_environment_ptr.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_native_source_low_pipeline(
      context.get(), loom_target_pipeline_build_to_source_low_artifacts,
      loomc_string_view_empty(), /*options=*/nullptr, loomc_allocator_system(),
      &pass_program, &result));
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_NE(pass_program_ptr.get(), nullptr);

  result_ptr.reset();
  pass_program_ptr.reset();
  context.reset();
  target_environment_ptr.reset();
}

TEST(InteropTest, CompilesPublicModuleWithNativeReport) {
  const loom_target_provider_set_t provider_set = {};

  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_from_provider_set(
      &provider_set, loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  ContextPtr context = CreateTargetContext(target_environment_ptr.get());
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateSource("module.loom", R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)");
  ModulePtr module = Deserialize(context.get(), workspace.get(), source.get());

  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context.get(), /*options=*/nullptr,
                                        loomc_allocator_system(), &compiler));
  CompilerPtr compiler_ptr(compiler);
  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_OK(
      loomc_pass_program_create_empty(context.get(), /*options=*/nullptr,
                                      loomc_allocator_system(), &pass_program));
  PassProgramPtr pass_program_ptr(pass_program);

  loom_target_compile_report_t report = {};
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_compile_module_with_native_report(
      compiler_ptr.get(), workspace.get(), pass_program_ptr.get(), module.get(),
      /*options=*/nullptr, &report, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_NE(report.allocator.ctl, nullptr);
  EXPECT_EQ(report.status_code, IREE_STATUS_OK);
  loom_target_compile_report_deinitialize(&report);

  result_ptr.reset();
  pass_program_ptr.reset();
  compiler_ptr.reset();
  module.reset();
  source.reset();
  workspace.reset();
  context.reset();
  target_environment_ptr.reset();
}

}  // namespace
