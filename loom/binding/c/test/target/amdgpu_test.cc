// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/artifact.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target.h"
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
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(loomc_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
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

TargetEnvironmentPtr CreateAmdgpuTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment);
  LOOMC_EXPECT_OK(status);
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateAmdgpuContext(loomc_target_environment_t* target_environment) {
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

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  loomc_status_t status = loomc_compiler_create(
      context, nullptr, loomc_allocator_system(), &compiler);
  LOOMC_EXPECT_OK(status);
  return CompilerPtr(compiler);
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

PassProgramPtr CreatePreparedLowPassProgram(
    loomc_context_t* context, loomc_target_selection_t* target_selection) {
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/target_selection,
  };
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/&target_options,
      /*.identifier=*/loomc_make_cstring_view("amdgpu-prepared-low-test"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_pass_program_create_from_target_pipeline(
      context, &pipeline_options, loomc_allocator_system(), &pass_program,
      &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return PassProgramPtr(pass_program);
}

ModulePtr CreatePreparedArithmeticModule(loomc_context_t* context,
                                         loomc_workspace_t* workspace) {
  SourcePtr source = CreateTextSource("amdgpu_prepared_arithmetic.loom", R"(
amdgpu.target<gfx11-generic> @gfx_target

low.kernel.def target(@gfx_target) workgroup_size(64, 1, 1) @loom_kernel() {
  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : reg<amdgpu.vgpr>
  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

TargetProfilePtr CreateTargetProfile(
    loomc_target_environment_t* target_environment, const char* processor) {
  loomc_amdgpu_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view(processor),
      /*.identity=*/
      {
          /*.processor=*/loomc_make_cstring_view(processor),
      },
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      target_environment, &profile_options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TargetSelectionPtr CreateTargetSelection(
    loomc_target_environment_t* target_environment, const char* processor) {
  TargetProfilePtr profile = CreateTargetProfile(target_environment, processor);
  loomc_target_selection_t* selection = nullptr;
  loomc_status_t status = loomc_target_selection_create_from_profile(
      profile.get(), loomc_allocator_system(), &selection);
  LOOMC_EXPECT_OK(status);
  return TargetSelectionPtr(selection);
}

TEST(AmdgpuTargetTest, TargetProfilePreservesAsicRevision) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  struct TestCase {
    loomc_amdgpu_asic_revision_t requested_revision;
    uint32_t expected_revision;
  };
  const TestCase test_cases[] = {
      {{/*.specified=*/false, /*.value=*/0}, 1},
      {{/*.specified=*/true, /*.value=*/0}, 0},
      {{/*.specified=*/true, /*.value=*/1}, 1},
  };
  for (const TestCase& test_case : test_cases) {
    loomc_amdgpu_profile_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.identifier=*/loomc_make_cstring_view("gfx1250-test"),
        /*.identity=*/
        {
            /*.processor=*/loomc_make_cstring_view("gfx1250"),
            /*.amdhsa_features=*/{},
            /*.asic_revision=*/test_case.requested_revision,
        },
    };
    loomc_target_profile_t* profile = nullptr;
    LOOMC_EXPECT_OK(
        loomc_target_profile_create_amdgpu(target_environment.get(), &options,
                                           loomc_allocator_system(), &profile));
    TargetProfilePtr profile_ptr(profile);
    ASSERT_NE(profile_ptr.get(), nullptr);
    loomc_amdgpu_target_identity_t identity = {};
    LOOMC_EXPECT_OK(loomc_amdgpu_target_profile_query_identity(
        profile_ptr.get(), &identity));
    EXPECT_EQ(ToString(identity.processor), "gfx1250");
    EXPECT_TRUE(identity.asic_revision.specified);
    EXPECT_EQ(identity.asic_revision.value, test_case.expected_revision);
  }
}

TEST(AmdgpuTargetTest, RejectsUnknownAsicRevision) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  loomc_amdgpu_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("gfx1250-unknown"),
      /*.identity=*/
      {
          /*.processor=*/loomc_make_cstring_view("gfx1250"),
          /*.amdhsa_features=*/{},
          /*.asic_revision=*/
          {
              /*.specified=*/true,
              /*.value=*/2,
          },
      },
  };
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_amdgpu(target_environment.get(), &options,
                                         loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(AmdgpuTargetTest, TargetProfilePreservesTargetIdFeatureStates) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  loomc_amdgpu_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("gfx942-features"),
      /*.identity=*/
      {
          /*.processor=*/loomc_make_cstring_view("gfx942"),
          /*.amdhsa_features=*/
          {
              /*.sramecc=*/LOOMC_AMDGPU_TARGET_FEATURE_ON,
              /*.xnack=*/LOOMC_AMDGPU_TARGET_FEATURE_OFF,
          },
      },
  };
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_OK(loomc_target_profile_create_amdgpu(
      target_environment.get(), &options, loomc_allocator_system(), &profile));
  TargetProfilePtr profile_ptr(profile);

  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(
      loomc_amdgpu_target_profile_query_identity(profile_ptr.get(), &identity));
  EXPECT_EQ(ToString(identity.processor), "gfx942");
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOMC_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOMC_AMDGPU_TARGET_FEATURE_OFF);
  EXPECT_FALSE(identity.asic_revision.specified);
}

TEST(AmdgpuTargetTest,
     TargetProfileDistinguishesUnconstrainedAndUnsupportedFeatures) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  struct TestCase {
    const char* processor;
    loomc_amdgpu_target_feature_state_t expected_state;
  };
  const TestCase test_cases[] = {
      {"gfx942", LOOMC_AMDGPU_TARGET_FEATURE_ANY},
      {"gfx1151", LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED},
  };
  for (const TestCase& test_case : test_cases) {
    TargetProfilePtr profile =
        CreateTargetProfile(target_environment.get(), test_case.processor);
    loomc_amdgpu_target_identity_t identity = {};
    LOOMC_EXPECT_OK(
        loomc_amdgpu_target_profile_query_identity(profile.get(), &identity));
    EXPECT_EQ(identity.amdhsa_features.sramecc, test_case.expected_state);
    EXPECT_EQ(identity.amdhsa_features.xnack, test_case.expected_state);
    EXPECT_FALSE(identity.asic_revision.specified);
  }
}

TEST(AmdgpuTargetTest, TargetProfileRejectsUnsupportedFeatureSelection) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  loomc_amdgpu_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("gfx1151-sramecc"),
      /*.identity=*/
      {
          /*.processor=*/loomc_make_cstring_view("gfx1151"),
          /*.amdhsa_features=*/
          {
              /*.sramecc=*/LOOMC_AMDGPU_TARGET_FEATURE_ON,
          },
      },
  };
  loomc_target_profile_t* profile = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_target_profile_create_amdgpu(target_environment.get(), &options,
                                         loomc_allocator_system(), &profile));
  EXPECT_EQ(profile, nullptr);
}

TEST(AmdgpuTargetTest, HsaAdapterPreservesQualifiedIdentity) {
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_from_hsa_isa_name(
      loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
      /*asic_revision=*/0, &identity));
  EXPECT_EQ(ToString(identity.processor), "gfx942");
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOMC_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOMC_AMDGPU_TARGET_FEATURE_OFF);
  EXPECT_FALSE(identity.asic_revision.specified);

  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_from_hsa_isa_name(
      loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx1250"),
      /*asic_revision=*/0, &identity));
  EXPECT_EQ(ToString(identity.processor), "gfx1250");
  EXPECT_TRUE(identity.asic_revision.specified);
  EXPECT_EQ(identity.asic_revision.value, 0u);

  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_from_hsa_isa_name(
      loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx1250"),
      /*asic_revision=*/1, &identity));
  EXPECT_TRUE(identity.asic_revision.specified);
  EXPECT_EQ(identity.asic_revision.value, 1u);
}

TEST(AmdgpuTargetTest, HsaAdapterRejectsUnknownRevision) {
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_amdgpu_target_identity_from_hsa_isa_name(
          loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx1250"),
          /*asic_revision=*/2, &identity));
}

TEST(AmdgpuTargetTest, HsaAdapterRejectsNonAmdhsaFeature) {
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_amdgpu_target_identity_from_hsa_isa_name(
          loomc_make_cstring_view(
              "amdgcn-amd-amdhsa--gfx1250:not-an-amdhsa-feature+"),
          /*asic_revision=*/0, &identity));
}

ResultPtr EmitModule(loomc_target_environment_t* target_environment,
                     loomc_workspace_t* workspace, loomc_module_t* module,
                     loomc_target_selection_t* selection,
                     loomc_amdgpu_runtime_global_flags_t runtime_globals) {
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection,
  };
  loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/&target_options,
      /*.runtime_globals=*/runtime_globals,
  };
  loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/&amdgpu_options,
      /*.artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      /*.identifier=*/loomc_make_cstring_view("loom_kernel.hsaco"),
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_emit_module(target_environment, workspace, module, &emit_options,
                        loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  return ResultPtr(result);
}

TEST(AmdgpuTargetTest, CompileConfiguredHalKernelEmitsModuleTextArtifact) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  TargetSelectionPtr selection =
      CreateTargetSelection(target_environment.get(), "gfx11-generic");
  PassProgramPtr pass_program =
      CreatePreparedLowPassProgram(context.get(), selection.get());
  SourcePtr source = CreateTextSource("configured_store.loom", R"(
config.decl @test.workgroups_x : %value: index where [range(%value, 1, 16)]
config.decl @test.workgroup_size_x : %value: index where [range(%value, 1, 256)]

kernel.def @configured_store() {
  %workgroups_x = config.get @test.workgroups_x : index
  %workgroup_size_x = config.get @test.workgroup_size_x : index
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%workgroups_x, %one, %one) workgroup_size(%workgroup_size_x, %one, %one) : index
} launch(%output: buffer) {
  %zero_offset = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %value = scalar.constant 7 : i32
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32, #dense>
  view.store %value, %view[%zero_index] : i32, view<1xi32, #dense>
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_config_binding_t bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("test.workgroups_x"),
          /*.value=*/loomc_make_cstring_view("2"),
      },
      {
          /*.key=*/loomc_make_cstring_view("test.workgroup_size_x"),
          /*.value=*/loomc_make_cstring_view("64"),
      },
  };
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/selection.get(),
  };
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("configured_store"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
      /*.config=*/
      {
          /*.bindings=*/bindings,
          /*.binding_count=*/IREE_ARRAYSIZE(bindings),
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN |
              LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };

  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());

  const loomc_artifact_t* text_artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                   LOOMC_ARTIFACT_FORMAT_LOOM_TEXT);
  ASSERT_NE(text_artifact, nullptr);
  EXPECT_EQ(ToString(text_artifact->identifier), "configured_store.loom");
  EXPECT_NE(text_artifact->contents.data_length, 0u);
}

TEST(AmdgpuTargetTest, EmitRuntimeGlobalsFromTargetOptions) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreatePreparedArithmeticModule(context.get(), workspace.get());
  TargetSelectionPtr selection =
      CreateTargetSelection(target_environment.get(), "gfx11-generic");

  ResultPtr result = EmitModule(target_environment.get(), workspace.get(),
                                module.get(), selection.get(),
                                LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG |
                                    LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG);
  ExpectSucceededResult(result.get());

  ASSERT_EQ(loomc_result_artifact_count(result.get()), 1u);
  const loomc_artifact_t* artifact = loomc_result_artifact_at(result.get(), 0);
  ASSERT_NE(artifact, nullptr);
  EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_EQ(ToString(artifact->format), LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);

  const std::string hsaco = ToString(artifact->contents);
  EXPECT_NE(hsaco.find("iree_asan_config"), std::string::npos);
  EXPECT_NE(hsaco.find("iree_feedback_config"), std::string::npos);
}

}  // namespace
