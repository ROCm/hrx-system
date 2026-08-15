// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loom/target/arch/cmd/package.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/context.h"
#include "loomc/launch_config.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/program_plan.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program_plan.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LaunchConfigProgramPtr = HandlePtr<loomc_launch_config_program_t,
                                         loomc_launch_config_program_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProgramPlanPtr =
    HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data == nullptr ? std::string()
                               : std::string(value.data, value.size);
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
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        ToString(artifact->format) == format) {
      return artifact;
    }
  }
  return nullptr;
}

TargetEnvironmentPtr CreateTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment));
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateContext(loomc_target_environment_t* target_environment) {
  const loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  const loomc_context_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(&options, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context, /*options=*/nullptr,
                                        loomc_allocator_system(), &compiler));
  return CompilerPtr(compiler);
}

PassProgramPtr CreateEmptyPassProgram(loomc_context_t* context) {
  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_empty(
      context, /*options=*/nullptr, loomc_allocator_system(), &pass_program));
  return PassProgramPtr(pass_program);
}

PassProgramPtr CreatePreparedLowPassProgram(loomc_context_t* context) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("command-dependency-test"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &options, loomc_allocator_system(), &pass_program, &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return PassProgramPtr(pass_program);
}

SourcePtr CreateSource(const char* contents) {
  const loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("command_program.loom"),
      /*.contents=*/loomc_make_byte_span(contents, std::strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

ModulePtr ParseModule(loomc_context_t* context, loomc_workspace_t* workspace,
                      const char* contents) {
  SourcePtr source = CreateSource(contents);
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source.get(), /*options=*/nullptr,
      loomc_allocator_system(), &module, &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

uint32_t LoadLittleEndianU32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

const char kCommandProgram[] = R"(
amdgpu.target<gfx1151> @gfx1151

kernel.def target(@gfx1151) @prefill_kernel(%token_count: index) {
  %one = index.constant 1 : index
  %group_count = index.add %token_count, %one : index
  kernel.launch.config workgroups(%group_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  kernel.return
}

kernel.def target(@gfx1151) @decode_kernel(%token_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%token_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  kernel.return
}

command.program.def public @prefill(%token_count: index) launch(%source: buffer, %target: buffer) where [range(%token_count, 1, 512)] {
  kernel.launch @prefill_kernel[%token_count](%source, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public @decode() launch(%source: buffer, %target: buffer) {
  %one = index.constant 1 : index
  kernel.launch @decode_kernel[%one](%source, %target) : [index](buffer, buffer)
  command.return
}
)";

TEST(CommandProgramTest, CompilesIndependentMultiRootProducts) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      ParseModule(context.get(), workspace.get(), kCommandProgram);

  const loomc_string_view_t root_names[] = {
      loomc_make_cstring_view("prefill"),
      loomc_make_cstring_view("decode"),
  };
  loomc_program_plan_t* plan = nullptr;
  loomc_result_t* prepare_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_prepare(
      workspace.get(), module.get(), root_names, IREE_ARRAYSIZE(root_names),
      /*options=*/nullptr, loomc_allocator_system(), &plan, &prepare_result));
  ProgramPlanPtr plan_ptr(plan);
  ResultPtr prepare_result_ptr(prepare_result);
  ExpectSucceededResult(prepare_result_ptr.get());

  ASSERT_EQ(loomc_program_plan_root_count(plan_ptr.get()), 2u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan_ptr.get()), 3u);
  const loomc_program_plan_root_t prefill_root =
      loomc_program_plan_root_at(plan_ptr.get(), 0);
  const loomc_program_plan_root_t decode_root =
      loomc_program_plan_root_at(plan_ptr.get(), 1);

  loomc_program_plan_root_info_t prefill_info = {};
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(plan_ptr.get(), prefill_root,
                                               &prefill_info));
  EXPECT_EQ(ToString(prefill_info.name), "prefill");
  ASSERT_EQ(prefill_info.required_unit_count, 3u);
  EXPECT_EQ(prefill_info.required_units[0].value, 0u);
  EXPECT_EQ(prefill_info.required_units[1].value, 1u);
  EXPECT_EQ(prefill_info.required_units[2].value, 2u);

  loomc_program_plan_root_info_t decode_info = {};
  LOOMC_ASSERT_OK(
      loomc_program_plan_root_info(plan_ptr.get(), decode_root, &decode_info));
  EXPECT_EQ(ToString(decode_info.name), "decode");
  ASSERT_EQ(decode_info.required_unit_count, 2u);
  EXPECT_EQ(decode_info.required_units[0].value, 0u);
  EXPECT_EQ(decode_info.required_units[1].value, 2u);

  loomc_cmd_program_plan_root_info_t prefill_cmd_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_plan_root_info(plan_ptr.get(), prefill_root,
                                                   &prefill_cmd_info));
  EXPECT_EQ(prefill_cmd_info.package_unit.value, 0u);
  EXPECT_EQ(prefill_cmd_info.launch_config_unit.value, 1u);
  ASSERT_EQ(prefill_cmd_info.executable_requirement_count, 1u);
  EXPECT_EQ(prefill_cmd_info.executable_requirements[0].unit.value, 2u);
  EXPECT_TRUE(loomc_string_view_is_empty(
      prefill_cmd_info.executable_requirements[0].import_name));

  loomc_cmd_program_plan_root_info_t decode_cmd_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_plan_root_info(plan_ptr.get(), decode_root,
                                                   &decode_cmd_info));
  EXPECT_EQ(decode_cmd_info.package_unit.value, 0u);
  EXPECT_FALSE(
      loomc_program_plan_unit_is_valid(decode_cmd_info.launch_config_unit));
  ASSERT_EQ(decode_cmd_info.executable_requirement_count, 1u);
  EXPECT_EQ(decode_cmd_info.executable_requirements[0].unit.value, 2u);

  // The plan owns its hermetic unit modules; the linked source and preparation
  // diagnostics are no longer needed before any unit is compiled.
  module.reset();
  prepare_result_ptr.reset();

  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr empty_pass_program = CreateEmptyPassProgram(context.get());
  PassProgramPtr dependency_pass_program =
      CreatePreparedLowPassProgram(context.get());
  const loomc_compile_report_options_t report_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      /*.structure_size=*/sizeof(report_options),
      /*.next=*/nullptr,
      /*.mode=*/LOOMC_COMPILE_REPORT_MODE_SUMMARY,
  };
  const loomc_program_plan_unit_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&report_options,
  };

  loomc_result_t* package_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan_ptr.get(), compiler.get(), workspace.get(),
      prefill_cmd_info.package_unit, empty_pass_program.get(), &compile_options,
      loomc_allocator_system(), &package_result));
  ResultPtr package_result_ptr(package_result);
  ExpectSucceededResult(package_result_ptr.get());

  loomc_result_t* launch_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan_ptr.get(), compiler.get(), workspace.get(),
      prefill_cmd_info.launch_config_unit, empty_pass_program.get(),
      &compile_options, loomc_allocator_system(), &launch_result));
  ResultPtr launch_result_ptr(launch_result);
  ExpectSucceededResult(launch_result_ptr.get());

  loomc_result_t* dependency_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan_ptr.get(), compiler.get(), workspace.get(),
      prefill_cmd_info.executable_requirements[0].unit,
      dependency_pass_program.get(), &compile_options, loomc_allocator_system(),
      &dependency_result));
  ResultPtr dependency_result_ptr(dependency_result);
  ExpectSucceededResult(dependency_result_ptr.get());

  const loomc_artifact_t* package_artifact =
      FindArtifact(package_result_ptr.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
  ASSERT_NE(package_artifact, nullptr);
  EXPECT_NE(FindArtifact(package_result_ptr.get(), LOOMC_ARTIFACT_KIND_REPORT,
                         LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON),
            nullptr);
  const loomc_artifact_t* launch_artifact = FindArtifact(
      launch_result_ptr.get(), LOOMC_ARTIFACT_KIND_COMMAND_LAUNCH_CONFIG,
      LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  ASSERT_NE(launch_artifact, nullptr);
  EXPECT_NE(FindArtifact(launch_result_ptr.get(), LOOMC_ARTIFACT_KIND_REPORT,
                         LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON),
            nullptr);
  const loomc_artifact_t* dependency_artifact =
      FindArtifact(dependency_result_ptr.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  ASSERT_NE(dependency_artifact, nullptr);
  ASSERT_GE(dependency_artifact->contents.data_length, 4u);
  static constexpr uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
  EXPECT_EQ(std::memcmp(dependency_artifact->contents.data, kElfMagic,
                        sizeof(kElfMagic)),
            0);
  EXPECT_NE(
      FindArtifact(dependency_result_ptr.get(), LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON),
      nullptr);

  loom_cmd_program_package_t package = {};
  IREE_ASSERT_OK(loom_cmd_program_package_parse(
      iree_make_const_byte_span(package_artifact->contents.data,
                                package_artifact->contents.data_length),
      &package));
  ASSERT_EQ(package.export_count, 2u);
  loom_cmd_program_package_export_t prefill_export = {};
  ASSERT_TRUE(loom_cmd_program_package_lookup_export(
      &package, IREE_SV("prefill"), &prefill_export));
  loom_cmd_program_package_export_t decode_export = {};
  ASSERT_TRUE(loom_cmd_program_package_lookup_export(
      &package, IREE_SV("decode"), &decode_export));
  ASSERT_EQ(prefill_export.entry_count, 1u);
  ASSERT_EQ(decode_export.entry_count, 1u);
  EXPECT_EQ(
      loom_cmd_program_package_export_entry_at(&package, &prefill_export, 0)
          .executable_index,
      0u);
  EXPECT_EQ(
      loom_cmd_program_package_export_entry_at(&package, &decode_export, 0)
          .executable_index,
      0u);

  const loom_cmd_program_t& prefill_program = prefill_export.program;
  ASSERT_NE(prefill_program.requirements.launch_counts.binding_index,
            UINT32_MAX);
  ASSERT_GT(prefill_program.requirements.launch_counts.required_byte_length,
            0u);
  const loom_cmd_program_t& decode_program = decode_export.program;
  EXPECT_EQ(decode_program.requirements.launch_counts.binding_index,
            UINT32_MAX);
  EXPECT_EQ(decode_program.requirements.launch_counts.required_byte_length, 0u);

  // Compiled products own all bytes needed by their runtime loaders.
  plan_ptr.reset();

  loomc_launch_config_program_t* launch_program = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_program_load(
      launch_artifact, /*release=*/nullptr, /*release_user_data=*/nullptr,
      loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);
  loomc_launch_config_function_t prefill_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("prefill"),
      &prefill_function));

  std::vector<uint8_t> launch_data(
      prefill_program.requirements.launch_counts.required_byte_length);
  loomc_cmd_launch_config_t launch_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(launch_config),
      /*.next=*/nullptr,
      /*.data=*/
      loomc_make_mutable_byte_span(launch_data.data(), launch_data.size()),
  };
  const uint64_t argument_bits[] = {17};
  LOOMC_ASSERT_OK(loomc_launch_config_program_invoke_cmd(
      launch_program_ptr.get(), prefill_function, argument_bits,
      IREE_ARRAYSIZE(argument_bits), &launch_config));
  ASSERT_GE(launch_data.size(), 3 * sizeof(uint32_t));
  EXPECT_EQ(LoadLittleEndianU32(&launch_data[0]), 18u);
  EXPECT_EQ(LoadLittleEndianU32(&launch_data[4]), 1u);
  EXPECT_EQ(LoadLittleEndianU32(&launch_data[8]), 1u);
}

}  // namespace
