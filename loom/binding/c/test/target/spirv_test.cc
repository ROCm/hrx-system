// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
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
#include "loomc/target.h"
#include "loomc/target/spirv/emit.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

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
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
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

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateSpirvContext(loomc_target_environment_t* target_environment) {
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

TargetProfilePtr CreateEmptyProfile(
    loomc_target_environment_t* target_environment) {
  loomc_target_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("spirv-test-profile"),
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_empty(
      target_environment, &options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TargetSelectionPtr CreateSelectionFromProfile(loomc_target_profile_t* profile) {
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile, loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
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
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

SourcePtr SerializeModuleText(
    loomc_module_t* module,
    loomc_module_text_presentation_t text_presentation =
        LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT,
    loomc_string_view_t low_asm_descriptor_set_key =
        loomc_string_view_empty()) {
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("roundtrip.loom"),
      /*.text_presentation=*/text_presentation,
      /*.low_asm_descriptor_set_key=*/low_asm_descriptor_set_key,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

std::string SourceContentsToString(const loomc_source_t* source) {
  loomc_byte_span_t contents = loomc_source_contents(source);
  return std::string(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
}

ModulePtr CreateBarrierSpirvLowModule(loomc_context_t* context,
                                      loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("barrier_spirv_low.loom", R"(
spirv.target<vulkan1_3> @target

low.func.def target(@target) abi(shader_entry_point) @spirv_barriers() asm<spirv.logical.core> {
  OpControlBarrier.subgroup.workgroup.acq_rel
  OpControlBarrier.workgroup.workgroup.acq_rel
  return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

void ExpectSpirvArtifact(const loomc_result_t* result,
                         const char* expected_identifier) {
  ASSERT_EQ(loomc_result_artifact_count(result), 1u);

  const loomc_artifact_t* artifact = loomc_result_artifact_at(result, 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_SPIRV);
  EXPECT_EQ(ToString(artifact->identifier), expected_identifier);
  ASSERT_GE(artifact->contents.data_length, sizeof(uint32_t));
  uint32_t magic = 0;
  memcpy(&magic, artifact->contents.data, sizeof(magic));
  EXPECT_EQ(magic, 0x07230203u);
}

TEST(TargetSpirvTest, CreatesTargetPipelinePassProgram) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
      /*.identifier=*/loomc_make_cstring_view("spirv-prepared-low"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      context.get(), &options, loomc_allocator_system(), &pass_program,
      &result);
  LOOMC_EXPECT_OK(status);
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result_ptr(result);
  EXPECT_NE(pass_program_ptr.get(), nullptr);
  ExpectSucceededResult(result_ptr.get());
}

TEST(TargetSpirvTest, EmitsSpirvBinaryArtifact) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  TargetProfilePtr profile = CreateEmptyProfile(target_environment.get());
  TargetSelectionPtr selection = CreateSelectionFromProfile(profile.get());
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* module_workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &module_workspace_handle));
  WorkspacePtr module_workspace(module_workspace_handle);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), module_workspace.get());
  SourcePtr serialized = SerializeModuleText(module.get());
  std::string serialized_text = SourceContentsToString(serialized.get());
  EXPECT_NE(serialized_text.find("asm<spirv.logical.core>"), std::string::npos)
      << serialized_text;
  EXPECT_NE(serialized_text.find("OpControlBarrier.subgroup.workgroup.acq_rel"),
            std::string::npos)
      << serialized_text;
  ModulePtr round_trip_module = DeserializeModule(
      context.get(), module_workspace.get(), serialized.get());

  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_spirv_emit_options_t spirv_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(spirv_options),
      /*.next=*/&target_options,
  };
  const loomc_option_entry_t emit_entries[] = {
      {
          /*.key=*/loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          /*.value=*/loomc_make_cstring_view("ignored.spv"),
      },
      {
          /*.key=*/loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          /*.value=*/loomc_make_cstring_view("spirv_barriers.spv"),
      },
  };
  loomc_option_dict_t option_dict = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      /*.structure_size=*/sizeof(option_dict),
      /*.next=*/&spirv_options,
      /*.entries=*/emit_entries,
      /*.entry_count=*/2,
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&option_dict,
      /*.artifact_format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      /*.identifier=*/loomc_make_cstring_view("typed.spv"),
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };

  for (int i = 0; i < 2; ++i) {
    loomc_workspace_t* workspace = nullptr;
    loomc_status_t workspace_status =
        loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
    LOOMC_EXPECT_OK(workspace_status);
    WorkspacePtr workspace_ptr(workspace);

    loomc_result_t* result = nullptr;
    loomc_status_t status = loomc_emit_module(
        target_environment.get(), workspace_ptr.get(), round_trip_module.get(),
        &options, loomc_allocator_system(), &result);
    LOOMC_EXPECT_OK(status);
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
    ExpectSpirvArtifact(result_ptr.get(), "spirv_barriers.spv");
  }
}

TEST(TargetSpirvTest, SerializesGenericTargetLowTextWhenRequested) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &workspace_handle));
  WorkspacePtr workspace(workspace_handle);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), workspace.get());

  SourcePtr serialized =
      SerializeModuleText(module.get(), LOOMC_MODULE_TEXT_PRESENTATION_GENERIC);
  std::string serialized_text = SourceContentsToString(serialized.get());
  EXPECT_NE(serialized_text.find("low.op<spirv.op_control_barrier"),
            std::string::npos)
      << serialized_text;
  EXPECT_EQ(serialized_text.find("OpControlBarrier.subgroup.workgroup.acq_rel"),
            std::string::npos)
      << serialized_text;
  ModulePtr round_trip_module =
      DeserializeModule(context.get(), workspace.get(), serialized.get());
  SourcePtr round_trip_text = SerializeModuleText(
      round_trip_module.get(), LOOMC_MODULE_TEXT_PRESENTATION_GENERIC);
  EXPECT_NE(SourceContentsToString(round_trip_text.get())
                .find("low.op<spirv.op_control_barrier"),
            std::string::npos);
}

TEST(TargetSpirvTest, RejectsUnknownLowAsmDescriptorSet) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &workspace_handle));
  WorkspacePtr workspace(workspace_handle);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), workspace.get());

  loomc_string_view_t unknown_descriptor_set_key =
      loomc_make_cstring_view("spirv.definitely_not_real");
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("roundtrip.loom"),
      /*.text_presentation=*/LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM,
      /*.low_asm_descriptor_set_key=*/unknown_descriptor_set_key,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module.get(), &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
  EXPECT_EQ(source, nullptr);
}

TEST(TargetSpirvTest, EmitsSpirvWithDefaultOptions) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* workspace = nullptr;
  loomc_status_t workspace_status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(workspace_status);
  WorkspacePtr workspace_ptr(workspace);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), workspace_ptr.get());

  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_emit_module(
      target_environment.get(), workspace_ptr.get(), module.get(), nullptr,
      loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ExpectSpirvArtifact(result_ptr.get(), "module.spv");
}

TEST(TargetSpirvTest, RejectsUnknownEmitDictOptionThroughResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  const loomc_option_entry_t emit_entries[] = {
      {
          /*.key=*/loomc_make_cstring_view("emit.definitely_not_real"),
          /*.value=*/loomc_make_cstring_view("1"),
      },
  };
  loomc_option_dict_t option_dict = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      /*.structure_size=*/sizeof(option_dict),
      /*.next=*/nullptr,
      /*.entries=*/emit_entries,
      /*.entry_count=*/1,
  };
  loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&option_dict,
      /*.artifact_format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      /*.identifier=*/loomc_make_cstring_view("spirv_barriers.spv"),
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };

  loomc_workspace_t* workspace = nullptr;
  loomc_status_t workspace_status =
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace);
  LOOMC_EXPECT_OK(workspace_status);
  WorkspacePtr workspace_ptr(workspace);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), workspace_ptr.get());

  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_emit_module(
      target_environment.get(), workspace_ptr.get(), module.get(), &options,
      loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "EMIT/OPTION");
  EXPECT_NE(ToString(diagnostic->message).find("emit.definitely_not_real"),
            std::string::npos);
  EXPECT_EQ(loomc_result_artifact_count(result_ptr.get()), 0u);
}

}  // namespace
