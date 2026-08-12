// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config.h"

#include <cstdint>
#include <cstring>
#include <utility>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ProgramPtr = HandlePtr<loomc_launch_config_program_t,
                             loomc_launch_config_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

struct LaunchArtifact {
  SourcePtr source;
  loomc_artifact_t artifact;
};

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

SourcePtr CreateTextSource(const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("launch_config.loom"),
      /*.contents=*/loomc_make_byte_span(contents, strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

ModulePtr ParseModule(loomc_context_t* context, loomc_workspace_t* workspace,
                      const char* text) {
  SourcePtr source = CreateTextSource(text);
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source.get(), nullptr, loomc_allocator_system(),
      &module, &result));
  ResultPtr result_ptr(result);
  EXPECT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

LaunchArtifact CompileLaunchArtifact(const char* text) {
  ContextPtr context = CreateContext();
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module = ParseModule(context.get(), workspace.get(), text);
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
      /*.identifier=*/loomc_make_cstring_view("launch_config.loombc"),
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(loomc_module_serialize_to_source(
      module.get(), &options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);
  const loomc_artifact_t artifact = {
      .kind = LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      .format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      .identifier = loomc_make_cstring_view("launch_config.loombc"),
      .contents = loomc_source_contents(source_ptr.get()),
  };
  return LaunchArtifact{std::move(source_ptr), artifact};
}

ProgramPtr LoadProgram(const LaunchArtifact& artifact) {
  loomc_launch_config_program_t* program = nullptr;
  LOOMC_EXPECT_OK(
      loomc_launch_config_program_load(&artifact.artifact, nullptr, nullptr,
                                       loomc_allocator_system(), &program));
  return ProgramPtr(program);
}

loomc_launch_config_t EmptyConfig() {
  return loomc_launch_config_t{
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(loomc_launch_config_t),
  };
}

const char kLaunchProgram[] = R"(
func.def public pure @prefill(%token_count: index) -> (index, index, index, index, index, index, index, index, index, index, index) where [range(%token_count, 1, 512)] {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c1024 = index.constant 1024 : index
  %group_count = index.add %token_count, %c1 : index
  func.return %group_count, %c1, %c1, %c64, %c1, %c1, %c1, %c1, %c1, %c32, %c1024 : index, index, index, index, index, index, index, index, index, index, index
}

func.def public pure @decode(%row_count: i32, %scale: bf16) -> (index, index, index, index, index, index, index, index, index, index, index) where [range(%row_count, 1, 64)] {
  %row_count_index = index.cast %row_count : i32 to index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %c256 = index.constant 256 : index
  func.return %row_count_index, %c1, %c1, %c32, %c1, %c1, %c1, %c1, %c1, %c32, %c256 : index, index, index, index, index, index, index, index, index, index, index
}
)";

TEST(LaunchConfigProgramTest, LoadsAndLooksUpMultipleFunctions) {
  LaunchArtifact artifact = CompileLaunchArtifact(kLaunchProgram);
  ProgramPtr program = LoadProgram(artifact);
  ASSERT_NE(program.get(), nullptr);

  loomc_launch_config_function_t prefill =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
      program.get(), loomc_make_cstring_view("prefill"), &prefill));
  EXPECT_TRUE(loomc_launch_config_function_is_valid(prefill));

  loomc_launch_config_function_t decode =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
      program.get(), loomc_make_cstring_view("decode"), &decode));
  EXPECT_TRUE(loomc_launch_config_function_is_valid(decode));
  EXPECT_NE(prefill.value, decode.value);

  loomc_launch_config_function_t missing =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_NOT_FOUND,
      loomc_launch_config_program_lookup_function(
          program.get(), loomc_make_cstring_view("missing"), &missing));
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_program_lookup_function(
          program.get(), loomc_make_cstring_view("@prefill"), &missing));
}

TEST(LaunchConfigProgramTest, InvokesCompleteLaunchContract) {
  LaunchArtifact artifact = CompileLaunchArtifact(kLaunchProgram);
  ProgramPtr program = LoadProgram(artifact);
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      program.get(), loomc_make_cstring_view("prefill"), &function));

  const uint64_t arguments[] = {127};
  loomc_launch_config_t config = EmptyConfig();
  LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(program.get(), function,
                                                     arguments, 1, &config));
  EXPECT_EQ(config.workgroup_count.x, 128u);
  EXPECT_EQ(config.workgroup_count.y, 1u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(config.workgroup_size.x, 64u);
  EXPECT_EQ(config.workgroup_size.y, 1u);
  EXPECT_EQ(config.workgroup_size.z, 1u);
  EXPECT_EQ(config.workgroup_cluster_size.x, 1u);
  EXPECT_EQ(config.workgroup_cluster_size.y, 1u);
  EXPECT_EQ(config.workgroup_cluster_size.z, 1u);
  EXPECT_EQ(config.subgroup_size, 32u);
  EXPECT_EQ(config.workgroup_storage_bytes, 1024u);
}

TEST(LaunchConfigProgramTest, SeedsDeclaredWidthScalarsAndChecksPredicates) {
  LaunchArtifact artifact = CompileLaunchArtifact(kLaunchProgram);
  ProgramPtr program = LoadProgram(artifact);
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      program.get(), loomc_make_cstring_view("decode"), &function));

  const uint64_t arguments[] = {
      UINT64_C(0xDEADBEEF00000020),
      UINT64_C(0xDEADBEEF00003F80),
  };
  loomc_launch_config_t config = EmptyConfig();
  LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(program.get(), function,
                                                     arguments, 2, &config));
  EXPECT_EQ(config.workgroup_count.x, 32u);
  EXPECT_EQ(config.workgroup_size.x, 32u);
  EXPECT_EQ(config.workgroup_storage_bytes, 256u);

  const uint64_t invalid_arguments[] = {0, UINT64_C(0x3F80)};
  config.workgroup_count.x = 777;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_program_invoke(program.get(), function,
                                         invalid_arguments, 2, &config));
  EXPECT_EQ(config.workgroup_count.x, 777u);
}

TEST(LaunchConfigProgramTest, SupportsRepeatedInvocations) {
  LaunchArtifact artifact = CompileLaunchArtifact(kLaunchProgram);
  ProgramPtr program = LoadProgram(artifact);
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      program.get(), loomc_make_cstring_view("prefill"), &function));

  for (uint64_t token_count = 1; token_count <= 512; ++token_count) {
    loomc_launch_config_t config = EmptyConfig();
    LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(
        program.get(), function, &token_count, 1, &config));
    EXPECT_EQ(config.workgroup_count.x, token_count + 1);
  }
}

void CountRelease(void* user_data, loomc_byte_span_t contents) {
  int* release_count = static_cast<int*>(user_data);
  EXPECT_NE(contents.data, nullptr);
  EXPECT_NE(contents.data_length, 0u);
  ++*release_count;
}

TEST(LaunchConfigProgramTest, ReleasesTransferredArtifactAfterLoad) {
  LaunchArtifact artifact = CompileLaunchArtifact(kLaunchProgram);
  int release_count = 0;
  loomc_launch_config_program_t* program = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_program_load(
      &artifact.artifact, CountRelease, &release_count,
      loomc_allocator_system(), &program));
  ProgramPtr program_ptr(program);
  EXPECT_EQ(release_count, 1);
}

TEST(LaunchConfigProgramTest, RejectsMalformedProgramContract) {
  LaunchArtifact artifact = CompileLaunchArtifact(R"(
func.def public pure @incomplete() -> (index, index, index) {
  %c1 = index.constant 1 : index
  func.return %c1, %c1, %c1 : index, index, index
}
)");
  int release_count = 0;
  loomc_launch_config_program_t* program = nullptr;
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_launch_config_program_load(
                             &artifact.artifact, CountRelease, &release_count,
                             loomc_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);
  EXPECT_EQ(release_count, 1);
}

TEST(LaunchConfigProgramTest, ClearsOutputsOnInvalidArguments) {
  loomc_launch_config_program_t* program =
      reinterpret_cast<loomc_launch_config_program_t*>(UINTPTR_MAX);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_program_load(
          /*artifact=*/nullptr, /*release=*/nullptr,
          /*release_user_data=*/nullptr, loomc_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);

  loomc_launch_config_function_t function = {.value = 0};
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_program_lookup_function(
          /*program=*/nullptr, loomc_make_cstring_view("missing"), &function));
  EXPECT_FALSE(loomc_launch_config_function_is_valid(function));
}

}  // namespace
