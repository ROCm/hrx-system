// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  loomc_status_t status =
      loomc_context_create(nullptr, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(status);
  return WorkspacePtr(workspace);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

ModulePtr DeserializeModule(loomc_context_t* context,
                            loomc_workspace_t* workspace,
                            const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_module_deserialize_from_source(
      context, workspace, source, nullptr, loomc_allocator_system(), &module,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

ModulePtr CreatePipelineModule(loomc_context_t* context,
                               loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("pipelines.loom", R"(
pass.pipeline<module> @cleanup pipeline {
  for func {
    canonicalize
  }
  call @finish
}

pass.pipeline<module> @finish pipeline {
  for func {
    dce
  }
}
)");
  return DeserializeModule(context, workspace, source.get());
}

void ExpectInvalidPassProgramResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result));
  ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "PASS_PROGRAM/INVALID");
}

void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
  EXPECT_EQ(loomc_result_diagnostic_count(result), 0u);
}

TEST(PassProgramTest, RetainRelease) {
  ContextPtr context = CreateContext();
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), nullptr, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);

  loomc_pass_program_retain(pass_program_ptr.get());
  loomc_pass_program_release(pass_program_ptr.get());
}

TEST(PassProgramTest, CreatesFromPipelineText) {
  ContextPtr context = CreateContext();

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context.get(), loomc_make_cstring_view("canonicalize,dce"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
}

TEST(PassProgramTest, CreatesFromModuleSymbol) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreatePipelineModule(context.get(), workspace.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      module.get(), loomc_make_cstring_view("@cleanup"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
}

TEST(PassProgramTest, ReportsInvalidPipelineTextInResult) {
  ContextPtr context = CreateContext();

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
      context.get(), loomc_make_cstring_view("definitely-not-a-pass"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(pass_program, nullptr);
  ResultPtr result_ptr(result);
  ASSERT_NE(result_ptr.get(), nullptr);
  ExpectInvalidPassProgramResult(result_ptr.get());
}

TEST(PassProgramTest, ReportsMissingModuleSymbolInResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreatePipelineModule(context.get(), workspace.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      module.get(), loomc_make_cstring_view("@missing"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(pass_program, nullptr);
  ResultPtr result_ptr(result);
  ExpectInvalidPassProgramResult(result_ptr.get());
}

TEST(PassProgramTest, ReportsNonPipelineModuleSymbolInResult) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("not_pipeline.loom", R"(
func.def @not_pipeline() {
  func.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());

  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_module_symbol(
      module.get(), loomc_make_cstring_view("@not_pipeline"), nullptr,
      loomc_allocator_system(), &pass_program, &result);
  LOOMC_EXPECT_OK(status);
  EXPECT_EQ(pass_program, nullptr);
  ResultPtr result_ptr(result);
  ExpectInvalidPassProgramResult(result_ptr.get());
}

TEST(PassProgramTest, RejectsUnknownOptionStructure) {
  ContextPtr context = CreateContext();

  loomc_pass_program_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_status_t status = loomc_pass_program_create_empty(
      context.get(), &options, loomc_allocator_system(), &pass_program);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(pass_program, nullptr);
}

}  // namespace
