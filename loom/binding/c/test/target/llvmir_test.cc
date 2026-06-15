// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/llvmir.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

TargetEnvironmentPtr CreateLlvmirTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_llvmir(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateLlvmirContext(loomc_target_environment_t* target_environment) {
  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  loomc_status_t status = loomc_context_create(
      &context_options, loomc_allocator_system(), &context);
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

ResultPtr EmitModule(loomc_target_environment_t* target_environment,
                     loomc_workspace_t* workspace, loomc_module_t* module,
                     const char* artifact_format, const char* identifier) {
  loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/nullptr,
      /*.artifact_format=*/loomc_make_cstring_view(artifact_format),
      /*.identifier=*/loomc_make_cstring_view(identifier),
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_emit_module(target_environment, workspace, module, &emit_options,
                        loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  return ResultPtr(result);
}

ModulePtr CreateLowAddModule(loomc_context_t* context,
                             loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("low_add.loom", R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}

low.func.def target(@target) abi(object_function) @low_add(%lhs: reg<llvmir.i32>, %rhs: reg<llvmir.i32>) -> (reg<llvmir.i32>) asm<llvmir.generic.core> {
  %sum = add.i32 %lhs, %rhs
  return %sum
}
)");
  return DeserializeModule(context, workspace, source.get());
}

ModulePtr CreateNoFunctionsModule(loomc_context_t* context,
                                  loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("no_functions.loom", R"(
llvmir.target<object> @target {
  triple = "loom-direct64-unknown-none",
  data_layout = "e-p:64:64-i64:64-n8:16:32:64-S128"
}
)");
  return DeserializeModule(context, workspace, source.get());
}

TEST(TargetLlvmirTest, EmitsTextAndBitcodeArtifacts) {
  TargetEnvironmentPtr target_environment = CreateLlvmirTargetEnvironment();
  ContextPtr context = CreateLlvmirContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateLowAddModule(context.get(), workspace.get());

  ResultPtr text_result =
      EmitModule(target_environment.get(), workspace.get(), module.get(),
                 LOOMC_ARTIFACT_FORMAT_LLVMIR_TEXT, "low_add.ll");
  ASSERT_TRUE(loomc_result_succeeded(text_result.get()));
  ASSERT_EQ(loomc_result_artifact_count(text_result.get()), 1u);
  const loomc_artifact_t* text_artifact =
      loomc_result_artifact_at(text_result.get(), 0);
  ASSERT_NE(text_artifact, nullptr);
  EXPECT_EQ(text_artifact->kind, LOOMC_ARTIFACT_KIND_TEXT);
  EXPECT_EQ(ToString(text_artifact->format), LOOMC_ARTIFACT_FORMAT_LLVMIR_TEXT);
  EXPECT_EQ(ToString(text_artifact->identifier), "low_add.ll");
  EXPECT_NE(ToString(text_artifact->contents)
                .find("define dso_local i32 @low_add(i32 %lhs, i32 %rhs)"),
            std::string::npos);

  ResultPtr bitcode_result =
      EmitModule(target_environment.get(), workspace.get(), module.get(),
                 LOOMC_ARTIFACT_FORMAT_LLVMIR_BITCODE, "low_add.bc");
  ASSERT_TRUE(loomc_result_succeeded(bitcode_result.get()));
  ASSERT_EQ(loomc_result_artifact_count(bitcode_result.get()), 1u);
  const loomc_artifact_t* bitcode_artifact =
      loomc_result_artifact_at(bitcode_result.get(), 0);
  ASSERT_NE(bitcode_artifact, nullptr);
  EXPECT_EQ(bitcode_artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(bitcode_artifact->format),
            LOOMC_ARTIFACT_FORMAT_LLVMIR_BITCODE);
  EXPECT_EQ(ToString(bitcode_artifact->identifier), "low_add.bc");
  EXPECT_GT(bitcode_artifact->contents.data_length, 4u);
}

TEST(TargetLlvmirTest, MarksEmissionDiagnosticsFailed) {
  TargetEnvironmentPtr target_environment = CreateLlvmirTargetEnvironment();
  ContextPtr context = CreateLlvmirContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = CreateNoFunctionsModule(context.get(), workspace.get());

  ResultPtr result =
      EmitModule(target_environment.get(), workspace.get(), module.get(),
                 LOOMC_ARTIFACT_FORMAT_LLVMIR_TEXT, "no_functions.ll");
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "TARGET/011");
}

}  // namespace
