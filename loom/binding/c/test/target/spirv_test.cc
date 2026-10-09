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
#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/target/spirv/emit.h"
#include "loomc/target/spirv/profile.h"
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
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
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

TargetEnvironmentPtr CreateSpirvTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateSpirvContext(loomc_target_environment_t* target_environment) {
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
  loomc_status_t status = loomc_context_create(
      &context_options, loomc_allocator_system(), &context);
  LOOMC_EXPECT_OK(status);
  return ContextPtr(context);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view(identifier),
      .contents = loomc_make_byte_span(contents, strlen(contents)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
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
    loomc_module_t* module, loomc_module_text_presentation_t text_presentation =
                                LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT) {
  loomc_module_serialize_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("roundtrip.loom"),
      .text_presentation = text_presentation,
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

low.func.def target<spirv.logical.core>(@target) abi(shader_entry_point) @spirv_barriers() asm {
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
  const std::string contents = ToString(artifact->contents);
  ASSERT_GE(contents.size(), sizeof(uint32_t));
  uint32_t magic = 0;
  memcpy(&magic, contents.data(), sizeof(magic));
  EXPECT_EQ(magic, 0x07230203u);
}

TEST(TargetSpirvTest, CreatesTargetPipelinePassProgram) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());

  loomc_target_pipeline_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("spirv-prepared-low"),
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
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

TEST(TargetSpirvTest, ConfigIdentitySurvivesCloneAndContinuedCompilation) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  WorkspacePtr workspace(raw_workspace);
  SourcePtr source = CreateTextSource("configured.loom", R"(
spirv.target<vulkan1_3> @target
config.def @tile_size = 1 : index
kernel.def target(@target) @configured() {
  %one = index.constant 1 : index
  %size = config.get @tile_size : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%size, %one, %one) : index
} launch() {
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  SourcePtr config_source =
      CreateTextSource("config.loom", "config.def @tile_size = 32 : index\n");
  ModulePtr config =
      DeserializeModule(context.get(), workspace.get(), config_source.get());
  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  HandlePtr<loomc_compiler_t, loomc_compiler_release> compiler(raw_compiler);
  loomc_target_pipeline_options_t pipeline_options = {
      .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG};
  loomc_pass_program_t* raw_pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
      context.get(), &pipeline_options, loomc_allocator_system(),
      &raw_pass_program, &raw_result));
  PassProgramPtr pass_program(raw_pass_program);
  ResultPtr result(raw_result);
  ExpectSucceededResult(result.get());
  result.reset();
  loomc_compile_options_t compile_options = {.config_module = config.get()};
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &raw_result));
  result.reset(raw_result);
  ExpectSucceededResult(result.get());
  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  result.reset();
  config.reset();
  config_source.reset();
  loomc_module_t* raw_clone = nullptr;
  LOOMC_ASSERT_OK(loomc_module_clone(module.get(), workspace.get(),
                                     loomc_allocator_system(), &raw_clone));
  ModulePtr clone(raw_clone);
  module.reset();
  source.reset();
  loomc_workspace_trim(workspace.get());

  for (int invocation = 0; invocation < 2; ++invocation) {
    for (auto mode : {LOOMC_COMPILE_REPORT_MODE_SUMMARY,
                      LOOMC_COMPILE_REPORT_MODE_DETAILS}) {
      loomc_compile_report_options_t report_options = {
          .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
          .structure_size = sizeof(report_options),
          .mode = mode};
      loomc_emit_options_t emit_options = {
          .next = &report_options,
          .artifact_format =
              loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
          .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY};
      LOOMC_ASSERT_OK(loomc_emit_module(
          target_environment.get(), workspace.get(), clone.get(), &emit_options,
          loomc_allocator_system(), &raw_result));
      result.reset(raw_result);
      ExpectSucceededResult(result.get());
      ASSERT_EQ(loomc_result_artifact_count(result.get()), 2u);
      const loomc_artifact_t* report =
          loomc_result_artifact_at(result.get(), 1);
      ASSERT_EQ(ToString(report->format),
                LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
      const std::string json = ToString(report->contents);
      EXPECT_NE(json.find("\"key\":\"tile_size\",\"value\":\"32\""),
                std::string::npos)
          << json;
      result.reset();
      loomc_workspace_trim(workspace.get());
    }
    if (invocation == 0) {
      LOOMC_ASSERT_OK(loomc_compile_module(
          compiler.get(), workspace.get(), pass_program.get(), clone.get(),
          nullptr, loomc_allocator_system(), &raw_result));
      result.reset(raw_result);
      ExpectSucceededResult(result.get());
      result.reset();
    }
  }
}

TEST(TargetSpirvTest, EmitsSpirvBinaryArtifact) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* module_workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &module_workspace_handle));
  WorkspacePtr module_workspace(module_workspace_handle);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), module_workspace.get());
  SourcePtr serialized = SerializeModuleText(module.get());
  std::string serialized_text = SourceContentsToString(serialized.get());
  EXPECT_NE(serialized_text.find("asm {"), std::string::npos)
      << serialized_text;
  EXPECT_NE(serialized_text.find("OpControlBarrier.subgroup.workgroup.acq_rel"),
            std::string::npos)
      << serialized_text;
  ModulePtr round_trip_module = DeserializeModule(
      context.get(), module_workspace.get(), serialized.get());

  loomc_spirv_emit_options_t spirv_options = {
      .type = LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
      .structure_size = sizeof(spirv_options),
      .next = nullptr,
  };
  const loomc_option_entry_t emit_entries[] = {
      {
          .key = loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          .value = loomc_make_cstring_view("ignored.spv"),
      },
      {
          .key = loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
          .value = loomc_make_cstring_view("spirv_barriers.spv"),
      },
  };
  loomc_option_dict_t option_dict = {
      .type = LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      .structure_size = sizeof(option_dict),
      .next = &spirv_options,
      .entries = emit_entries,
      .entry_count = 2,
  };
  loomc_emit_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &option_dict,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .identifier = loomc_make_cstring_view("typed.spv"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
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

TEST(TargetSpirvTest, EmitsArtifactManifestAndTargetReport) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &workspace_handle));
  WorkspacePtr workspace(workspace_handle);
  ModulePtr module =
      CreateBarrierSpirvLowModule(context.get(), workspace.get());

  loomc_compile_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .next = nullptr,
      .mode = LOOMC_COMPILE_REPORT_MODE_DETAILS,
      .format = LOOMC_COMPILE_REPORT_FORMAT_JSON,
  };
  loomc_artifact_manifest_options_t manifest_options = {
      .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      .structure_size = sizeof(manifest_options),
      .next = &report_options,
      .mode = LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY,
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &manifest_options,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .identifier = loomc_make_cstring_view("spirv_barriers.spv"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_result_t* result_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_emit_module(target_environment.get(), workspace.get(),
                                    module.get(), &emit_options,
                                    loomc_allocator_system(), &result_handle));
  ResultPtr result(result_handle);
  ExpectSucceededResult(result.get());

  const loomc_artifact_t* manifest =
      FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON);
  ASSERT_NE(manifest, nullptr);
  const std::string manifest_text = ToString(manifest->contents);
  EXPECT_NE(manifest_text.find("\"format\":\"spirv-binary\""),
            std::string::npos)
      << manifest_text;
  EXPECT_NE(manifest_text.find("\"name\":\"spirv_barriers.spv\""),
            std::string::npos)
      << manifest_text;
  EXPECT_NE(manifest_text.find("\"targets\":[{\"name\":\"target\""),
            std::string::npos)
      << manifest_text;
  EXPECT_NE(manifest_text.find("\"functions\":[{\"name\":\"spirv_barriers\""),
            std::string::npos)
      << manifest_text;

  const loomc_artifact_t* report =
      FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  ASSERT_NE(report, nullptr);
  const std::string report_text = ToString(report->contents);
  EXPECT_NE(report_text.find("\"artifact_kind\":\"target-artifact\""),
            std::string::npos)
      << report_text;
  EXPECT_NE(report_text.find("\"backend\":\"spirv\""), std::string::npos)
      << report_text;
  EXPECT_NE(report_text.find("\"target_family\":\"spirv\""), std::string::npos)
      << report_text;
  EXPECT_NE(report_text.find("\"target_bundle\":\"target\""), std::string::npos)
      << report_text;
  EXPECT_NE(report_text.find("\"target_export\":\"spirv_barriers\""),
            std::string::npos)
      << report_text;
  EXPECT_NE(report_text.find("\"target_config\":\"target\""), std::string::npos)
      << report_text;
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

TEST(TargetSpirvTest, SerializesTargetlessLowFromRepresentationContract) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* workspace_handle = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &workspace_handle));
  WorkspacePtr workspace(workspace_handle);
  SourcePtr source = CreateTextSource("targetless_low.loom", R"(
low.func.def target<spirv.logical.core> abi(shader_entry_point) @targetless() asm {
  OpControlBarrier.subgroup.workgroup.acq_rel
  return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());

  SourcePtr serialized =
      SerializeModuleText(module.get(), LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM);
  const std::string serialized_text = SourceContentsToString(serialized.get());
  EXPECT_NE(
      serialized_text.find(
          "low.func.def target<spirv.logical.core> abi(shader_entry_point) "
          "@targetless() asm {"),
      std::string::npos)
      << serialized_text;
  EXPECT_NE(serialized_text.find("OpControlBarrier.subgroup.workgroup.acq_rel"),
            std::string::npos)
      << serialized_text;
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

TEST(TargetSpirvTest, DirectEmissionRunsTargetLowVerificationProviders) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  WorkspacePtr workspace(raw_workspace);
  SourcePtr source = CreateTextSource("untyped.loom", R"(
spirv.target<vulkan1_3> @target
low.func.def target<spirv.logical.core>(@target) @untyped(%value: reg<spirv.id>) asm {
  return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_emit_module(target_environment.get(), workspace.get(),
                                    module.get(), nullptr,
                                    loomc_allocator_system(), &raw_result));
  ResultPtr result(raw_result);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
            "SPIRV/005");
}

TEST(TargetSpirvTest, CompileArtifactVerifiesPassProgramOutputBeforeEmission) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  loomc_workspace_t* raw_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(nullptr, loomc_allocator_system(),
                                         &raw_workspace));
  WorkspacePtr workspace(raw_workspace);
  SourcePtr source = CreateTextSource("selected_buffer.loom", R"(
func.def @choose_buffer(%first: buffer, %second: buffer, %choose_first: i1) -> (buffer) {
  %selected = scf.select %choose_first, %first, %second : buffer
  func.return %selected : buffer
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());

  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(
      context.get(), nullptr, loomc_allocator_system(), &raw_compiler));
  CompilerPtr compiler(raw_compiler);
  const loomc_target_pipeline_options_t pipeline_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      .structure_size = sizeof(pipeline_options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("spirv-source-low"),
      .kind = LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW,
      .control_flow_lowering =
          LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
      .source_to_low_max_errors = 20,
  };
  loomc_pass_program_t* raw_pass_program = nullptr;
  loomc_result_t* raw_pipeline_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
      context.get(), &pipeline_options, loomc_allocator_system(),
      &raw_pass_program, &raw_pipeline_result));
  PassProgramPtr pass_program(raw_pass_program);
  ResultPtr pipeline_result(raw_pipeline_result);
  ExpectSucceededResult(pipeline_result.get());

  loomc_target_profile_t* raw_target_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(),
      loomc_make_cstring_view("spirv:vulkan1.3+bda+extended-types"),
      loomc_allocator_system(), &raw_target_profile));
  TargetProfilePtr target_profile(raw_target_profile);
  const loomc_string_view_t roots[] = {
      loomc_make_cstring_view("choose_buffer"),
  };
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = nullptr,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .identifier = loomc_make_cstring_view("selected_buffer.spv"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = nullptr,
      .roots = roots,
      .root_count = IREE_ARRAYSIZE(roots),
      .excluded_roots = nullptr,
      .excluded_root_count = 0,
      .target_profile = target_profile.get(),
      .config = nullptr,
      .emit_options = &emit_options,
  };
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &raw_result));
  ResultPtr result(raw_result);
  EXPECT_FALSE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(loomc_result_artifact_count(result.get()), 0u);
  ASSERT_EQ(loomc_result_diagnostic_count(result.get()), 1u);
  EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
            "SPIRV/016");
}

TEST(TargetSpirvTest, RejectsUnknownEmitDictOptionThroughResult) {
  TargetEnvironmentPtr target_environment = CreateSpirvTargetEnvironment();
  ContextPtr context = CreateSpirvContext(target_environment.get());
  const loomc_option_entry_t emit_entries[] = {
      {
          .key = loomc_make_cstring_view("emit.definitely_not_real"),
          .value = loomc_make_cstring_view("1"),
      },
  };
  loomc_option_dict_t option_dict = {
      .type = LOOMC_STRUCTURE_TYPE_OPTION_DICT,
      .structure_size = sizeof(option_dict),
      .next = nullptr,
      .entries = emit_entries,
      .entry_count = 1,
  };
  loomc_emit_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &option_dict,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .identifier = loomc_make_cstring_view("spirv_barriers.spv"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
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
