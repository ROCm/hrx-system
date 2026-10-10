// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/vm.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
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
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

constexpr char kSource[] = R"(
func.def public @identity(%value: i32) -> (i32) {
  func.return %value : i32
}
)";

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

::testing::AssertionResult Succeeded(const loomc_result_t* result) {
  if (result != nullptr && loomc_result_succeeded(result)) {
    return ::testing::AssertionSuccess();
  }
  auto failure = ::testing::AssertionFailure();
  if (result == nullptr) {
    return failure << "operation did not return a result";
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    failure << ToString(diagnostic->message);
  }
  return failure;
}

TEST(TargetVmTest, RejectsInvalidNamedProfiles) {
  loomc_target_environment_t* raw_target_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                     &raw_target_environment));
  TargetEnvironmentPtr target_environment(raw_target_environment);

  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_select(target_environment.get(),
                                  loomc_make_cstring_view("vm"),
                                  loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_NOT_FOUND,
      loomc_target_profile_select(target_environment.get(),
                                  loomc_make_cstring_view("vm:unknown"),
                                  loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_select(target_environment.get(),
                                  loomc_make_cstring_view("wasm:simd128"),
                                  loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(TargetVmTest, CompilesAndEmitsBytecodeModule) {
  loomc_target_environment_t* raw_target_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_vm(loomc_allocator_system(),
                                                     &raw_target_environment));
  TargetEnvironmentPtr target_environment(raw_target_environment);
  loomc_target_profile_t* raw_target_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(), loomc_make_cstring_view("vm:core"),
      loomc_allocator_system(), &raw_target_profile));
  TargetProfilePtr target_profile(raw_target_profile);

  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .target_environment = target_environment.get(),
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &raw_context));
  ContextPtr context(raw_context);

  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  WorkspacePtr workspace(raw_workspace);

  loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("identity.loom"),
      .contents = loomc_make_byte_span(kSource, sizeof(kSource) - 1),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_source_create(&source_options, loomc_allocator_system(),
                                      &raw_source));
  SourcePtr source(raw_source);

  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), workspace.get(), source.get(), nullptr,
      loomc_allocator_system(), &raw_module, &raw_result));
  ModulePtr module(raw_module);
  ResultPtr result(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  loomc_pass_program_t* raw_pass_program = nullptr;
  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
      context.get(), nullptr, loomc_allocator_system(), &raw_pass_program,
      &raw_result));
  PassProgramPtr pass_program(raw_pass_program);
  result.reset(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  CompilerPtr compiler(raw_compiler);
  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("identity"),
      .target_profile = target_profile.get(),
  };
  const loomc_target_specialization_options_t target_compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_compile_options),
      .next = nullptr,
      .specializations = &specialization,
      .specialization_count = 1,
  };
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_compile_options,
  };
  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));

  raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_emit_module(target_environment.get(), workspace.get(),
                                    module.get(), nullptr,
                                    loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ASSERT_TRUE(Succeeded(result.get()));
  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);

  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_VM);
  EXPECT_EQ(ToString(artifact->identifier), "module.vm");
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_ASSERT_OK(loomc_byte_sequence_clone(
      artifact->contents, loomc_allocator_system(), &contents));
  ASSERT_GE(contents.data_length, 8u);
  EXPECT_EQ(memcmp(contents.data, "IREEVM\0\0", 8), 0);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
}

}  // namespace
