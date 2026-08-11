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
#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/launch_config.h"
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
using LaunchConfigProgramPtr = HandlePtr<loomc_launch_config_program_t,
                                         loomc_launch_config_program_release>;
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

SourcePtr SerializeModule(const loomc_module_t* module,
                          loomc_source_format_t format) {
  loomc_module_serialize_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/format,
      /*.identifier=*/format == LOOMC_SOURCE_FORMAT_BYTECODE
          ? loomc_make_cstring_view("compiled.loombc")
          : loomc_make_cstring_view("compiled.loom"),
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status = loomc_module_serialize_to_source(
      module, &options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

std::string SerializeModuleToText(const loomc_module_t* module) {
  SourcePtr source = SerializeModule(module, LOOMC_SOURCE_FORMAT_TEXT);
  return source ? ToString(loomc_source_contents(source.get())) : std::string();
}

PassProgramPtr CreatePreparedLowPassProgram(loomc_context_t* context) {
  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/nullptr,
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

low.kernel.def target<amdgpu.gfx11.generic.core>(@gfx_target) workgroup_size(64, 1, 1) @loom_kernel() {
  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : reg<amdgpu.vgpr>
  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

TargetProfilePtr CreateTargetProfile(
    loomc_target_environment_t* target_environment, const char* target) {
  loomc_amdgpu_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view(target),
      /*.identity=*/
      {
          /*.target=*/loomc_make_cstring_view(target),
      },
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      target_environment, &profile_options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TEST(AmdgpuTargetTest, TargetProfilePreservesCanonicalTarget) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  const char* target_names[] = {
      "gfx1250-a0",
      "gfx1250",
      "gfx12-5-generic",
  };
  for (const char* target_name : target_names) {
    loomc_amdgpu_profile_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.identifier=*/{},
        /*.identity=*/
        {
            /*.target=*/loomc_make_cstring_view(target_name),
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
    EXPECT_EQ(ToString(identity.target), target_name);
  }
}

TEST(AmdgpuTargetTest, RejectsUnknownTarget) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  loomc_amdgpu_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/{},
      /*.identity=*/
      {
          /*.target=*/loomc_make_cstring_view("gfx1250-a1"),
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
          /*.target=*/loomc_make_cstring_view("gfx942"),
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
  EXPECT_EQ(ToString(identity.target), "gfx942");
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOMC_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOMC_AMDGPU_TARGET_FEATURE_OFF);
}

TEST(AmdgpuTargetTest,
     TargetProfileDistinguishesUnconstrainedAndUnsupportedFeatures) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  struct TestCase {
    const char* target;
    loomc_amdgpu_target_feature_state_t expected_state;
  };
  const TestCase test_cases[] = {
      {"gfx942", LOOMC_AMDGPU_TARGET_FEATURE_ANY},
      {"gfx1151", LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED},
  };
  for (const TestCase& test_case : test_cases) {
    TargetProfilePtr profile =
        CreateTargetProfile(target_environment.get(), test_case.target);
    loomc_amdgpu_target_identity_t identity = {};
    LOOMC_EXPECT_OK(
        loomc_amdgpu_target_profile_query_identity(profile.get(), &identity));
    EXPECT_EQ(identity.amdhsa_features.sramecc, test_case.expected_state);
    EXPECT_EQ(identity.amdhsa_features.xnack, test_case.expected_state);
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
          /*.target=*/loomc_make_cstring_view("gfx1151"),
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

TEST(AmdgpuTargetTest, HsaAdapterResolvesCanonicalIdentity) {
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_from_hsa_isa_name(
      loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
      /*asic_revision=*/0, &identity));
  EXPECT_EQ(ToString(identity.target), "gfx942");
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOMC_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOMC_AMDGPU_TARGET_FEATURE_OFF);

  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_from_hsa_isa_name(
      loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx1250"),
      /*asic_revision=*/0, &identity));
  EXPECT_EQ(ToString(identity.target), "gfx1250-a0");

  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_from_hsa_isa_name(
      loomc_make_cstring_view("amdgcn-amd-amdhsa--gfx1250"),
      /*asic_revision=*/1, &identity));
  EXPECT_EQ(ToString(identity.target), "gfx1250");
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
                     loomc_amdgpu_runtime_global_flags_t runtime_globals,
                     loomc_artifact_manifest_mode_t artifact_manifest_mode =
                         LOOMC_ARTIFACT_MANIFEST_MODE_NONE) {
  loomc_artifact_manifest_options_t artifact_manifest_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      /*.structure_size=*/sizeof(artifact_manifest_options),
      /*.next=*/nullptr,
      /*.mode=*/artifact_manifest_mode,
      /*.identifier=*/loomc_string_view_empty(),
  };
  loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/artifact_manifest_mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE
          ? &artifact_manifest_options
          : nullptr,
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

TEST(AmdgpuTargetTest,
     LaunchConfigArtifactCombinesIdentityAndSubgroupRequirements) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("target_specialized_launch.loom", R"(
amdgpu.target<gfx11-generic> @gfx11_wave64 {subgroup_size = 64}
amdgpu.target<gfx1151> @gfx1151

func.template<test.experts_per_wave> target(@gfx1151) requires [#target.subgroup.size<64>] priority(20) @two_experts_per_wave(%expert_count: index) -> (index) {
  %c2 = index.constant 2 : index
  func.return %c2 : index
}

func.template<test.experts_per_wave> priority(1) @four_experts_per_wave(%expert_count: index) -> (index) {
  %c4 = index.constant 4 : index
  func.return %c4 : index
}

kernel.def target(@gfx11_wave64) @target_specialized_launch(%expert_count: index) {
  %c1 = index.constant 1 : index
  %wave_size = target.subgroup.size : index
  %experts_per_wave = func.apply<test.experts_per_wave>(%expert_count) : (index) -> (index)
  %workgroup_count = index.div %expert_count, %experts_per_wave : index
  kernel.launch.config workgroups(%workgroup_count, %c1, %c1) workgroup_size(%wave_size, %c1, %c1) : index
} launch(%output: buffer) {
  %scratch_bytes = index.constant 256 : offset
  %base = index.constant 0 : offset
  %origin = index.constant 0 : index
  %zero = scalar.constant 0 : i32
  %scratch = buffer.alloca<workgroup> align(64) %scratch_bytes : buffer
  %scratch_view = buffer.view %scratch[%base] : buffer -> view<1xi32>
  view.store %zero, %scratch_view[%origin] : i32, view<1xi32>
  kernel.return
}
)");

  struct TargetCase {
    const char* target;
    uint32_t expected_workgroup_count;
  };
  const TargetCase target_cases[] = {
      {"gfx1100", 32},
      {"gfx1151", 64},
  };
  for (const TargetCase& target_case : target_cases) {
    ModulePtr module =
        DeserializeModule(context.get(), workspace.get(), source.get());
    TargetProfilePtr profile =
        CreateTargetProfile(target_environment.get(), target_case.target);
    const loomc_target_specialization_t specialization = {
        /*.function_symbol=*/
        loomc_make_cstring_view("target_specialized_launch"),
        /*.target_profile=*/profile.get(),
    };
    const loomc_target_specialization_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.specializations=*/&specialization,
        /*.specialization_count=*/1,
    };
    const loomc_compile_options_t options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(options),
        /*.next=*/&target_options,
        /*.module_name=*/
        loomc_make_cstring_view("target_specialized_launch"),
        /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
        /*.config=*/{},
    };

    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_compile_module(
        compiler.get(), workspace.get(), pass_program.get(), module.get(),
        &options, loomc_allocator_system(), &result));
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
    const loomc_artifact_t* artifact =
        FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                     LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    ASSERT_NE(artifact, nullptr);

    loomc_launch_config_program_t* launch_program = nullptr;
    LOOMC_EXPECT_OK(loomc_launch_config_program_load(
        artifact, /*release=*/nullptr, /*release_user_data=*/nullptr,
        loomc_allocator_system(), &launch_program));
    LaunchConfigProgramPtr launch_program_ptr(launch_program);
    result_ptr.reset();

    loomc_launch_config_function_t launch_function =
        loomc_launch_config_function_invalid();
    LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
        launch_program_ptr.get(),
        loomc_make_cstring_view("target_specialized_launch"),
        &launch_function));
    const uint64_t workload_argument_bits[] = {128};
    loomc_launch_config_t launch_config = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        /*.structure_size=*/sizeof(launch_config),
    };
    LOOMC_EXPECT_OK(loomc_launch_config_program_invoke(
        launch_program_ptr.get(), launch_function, workload_argument_bits,
        IREE_ARRAYSIZE(workload_argument_bits), &launch_config));
    EXPECT_EQ(launch_config.workgroup_count.x,
              target_case.expected_workgroup_count)
        << target_case.target;
    EXPECT_EQ(launch_config.workgroup_count.y, 1u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_count.z, 1u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_size.x, 64u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_size.y, 1u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_size.z, 1u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_cluster_size.x, 1u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_cluster_size.y, 1u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_cluster_size.z, 1u) << target_case.target;
    EXPECT_EQ(launch_config.subgroup_size, 64u) << target_case.target;
    EXPECT_EQ(launch_config.workgroup_storage_bytes, 256u)
        << target_case.target;
  }
}

TEST(AmdgpuTargetTest, LaunchConfigArtifactContainsAllKernelExports) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("multi_launch_config.loom", R"(
amdgpu.target<gfx1151> @gfx1151

kernel.def target(@gfx1151) @prefill(%token_count: index) {
  %one = index.constant 1 : index
  %workgroup_count = index.add %token_count, %one : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}

kernel.def target(@gfx1151) @decode(%row_count: i32, %scale: bf16) {
  %one = index.constant 1 : index
  %row_count_index = index.cast %row_count : i32 to index
  %scale_f32 = scalar.extf %scale : bf16 to f32
  %scale_i32 = scalar.fptoui %scale_f32 : f32 to i32
  %scale_index = index.cast %scale_i32 : i32 to index
  %workgroup_count = index.mul %row_count_index, %scale_index : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("multi_launch_config"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
      /*.config=*/{},
  };
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  const loomc_artifact_t* artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                   LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  ASSERT_NE(artifact, nullptr);

  loomc_launch_config_program_t* launch_program = nullptr;
  LOOMC_EXPECT_OK(loomc_launch_config_program_load(
      artifact, /*release=*/nullptr, /*release_user_data=*/nullptr,
      loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);

  loomc_launch_config_function_t prefill_function =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("prefill"),
      &prefill_function));
  const uint64_t prefill_arguments[] = {128};
  loomc_launch_config_t prefill_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(prefill_config),
  };
  LOOMC_EXPECT_OK(loomc_launch_config_program_invoke(
      launch_program_ptr.get(), prefill_function, prefill_arguments,
      IREE_ARRAYSIZE(prefill_arguments), &prefill_config));
  EXPECT_EQ(prefill_config.workgroup_count.x, 129u);

  loomc_launch_config_function_t decode_function =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("decode"),
      &decode_function));
  const uint64_t decode_arguments[] = {
      32,
      UINT64_C(0xDEADBEEF00004000),
  };
  loomc_launch_config_t decode_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(decode_config),
  };
  LOOMC_EXPECT_OK(loomc_launch_config_program_invoke(
      launch_program_ptr.get(), decode_function, decode_arguments,
      IREE_ARRAYSIZE(decode_arguments), &decode_config));
  EXPECT_EQ(decode_config.workgroup_count.x, 64u);
}

TEST(AmdgpuTargetTest,
     SpecializationPreservesGenericHalTargetAndEmitsExactArtifact) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("configured_store.loom", R"(
amdgpu.target<gfx11-generic> @gfx11_generic

config.decl @test.workgroups_x : %value: index where [range(%value, 1, 16)]
config.decl @test.workgroup_size_x : %value: index where [range(%value, 1, 256)]

kernel.def target(@gfx11_generic) @configured_store() {
  %workgroups_x = config.get @test.workgroups_x : index
  %workgroup_size_x = config.get @test.workgroup_size_x : index
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%workgroups_x, %one, %one) workgroup_size(%workgroup_size_x, %one, %one) : index
} launch(%output: buffer) {
  %zero_offset = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %value = scalar.constant 7 : i32
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32>
  view.store %value, %view[%zero_index] : i32, view<1xi32>
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
  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("configured_store"),
      /*.target_profile=*/profile.get(),
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
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
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
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
  const std::string module_text = ToString(text_artifact->contents);
  EXPECT_NE(module_text.find("amdgpu.target<gfx11-generic> @gfx11_generic"),
            std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("low.kernel.def target<amdgpu.rdna3_5.core>"
                             "(@__loom_sealed_target_0)"),
            std::string::npos)
      << module_text;
  EXPECT_EQ(
      module_text.find("low.kernel.def target<amdgpu.gfx11.generic.core>"),
      std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("amdgpu.target<gfx1151> @__loom_sealed_target_0"),
            std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("@configured_store("), std::string::npos)
      << module_text;

  // Compiled function-version facts are module-owned and must not borrow the
  // invocation profile used to produce them.
  profile.reset();
  ResultPtr emit_result =
      EmitModule(target_environment.get(), workspace.get(), module.get(),
                 /*runtime_globals=*/0, LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY);
  ExpectSucceededResult(emit_result.get());

  const loomc_artifact_t* hsaco_artifact =
      FindArtifact(emit_result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  ASSERT_NE(hsaco_artifact, nullptr);
  const std::string hsaco = ToString(hsaco_artifact->contents);
  EXPECT_NE(hsaco.find("amdgcn-amd-amdhsa--gfx1151"), std::string::npos);

  const loomc_artifact_t* manifest_artifact =
      FindArtifact(emit_result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON);
  ASSERT_NE(manifest_artifact, nullptr);
  const std::string manifest = ToString(manifest_artifact->contents);
  EXPECT_NE(manifest.find("\"family\":\"amdgpu\""), std::string::npos)
      << manifest;
  EXPECT_NE(manifest.find("\"selector\":\"gfx1151\""), std::string::npos)
      << manifest;
  EXPECT_NE(manifest.find("\"processor\":\"gfx1151\""), std::string::npos)
      << manifest;
  EXPECT_NE(
      manifest.find("\"code_object_target\":\"amdgcn-amd-amdhsa--gfx1151\""),
      std::string::npos)
      << manifest;
}

TEST(AmdgpuTargetTest, CompileSpecializesRetainedHelpersForEachTargetContext) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  TargetProfilePtr wave32_profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  TargetProfilePtr wave64_profile =
      CreateTargetProfile(target_environment.get(), "gfx942");
  SourcePtr source = CreateTextSource("retained_helpers.loom", R"(
func.def @read_subgroup_size() -> (index) {
  %size = target.subgroup.size : index
  func.return %size : index
}

func.def @forward_subgroup_size() -> (index) {
  %size = func.call @read_subgroup_size() : () -> (index)
  func.return %size : index
}

kernel.def @wave32_root() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%output: buffer) {
  %size = func.call @forward_subgroup_size() : () -> (index)
  %size_i32 = index.cast %size : index to i32
  %zero_offset = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32>
  view.store %size_i32, %view[%zero_index] : i32, view<1xi32>
  kernel.return
}

kernel.def @wave64_root() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%output: buffer) {
  %size = func.call @forward_subgroup_size() : () -> (index)
  %size_i32 = index.cast %size : index to i32
  %zero_offset = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %global = buffer.assume.memory_space<global> %output : buffer
  %view = buffer.view %global[%zero_offset] : buffer -> view<1xi32>
  view.store %size_i32, %view[%zero_index] : i32, view<1xi32>
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_target_specialization_t specializations[] = {
      {
          /*.function_symbol=*/loomc_make_cstring_view("wave32_root"),
          /*.target_profile=*/wave32_profile.get(),
      },
      {
          /*.function_symbol=*/loomc_make_cstring_view("wave64_root"),
          /*.target_profile=*/wave64_profile.get(),
      },
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations,
      /*.specialization_count=*/IREE_ARRAYSIZE(specializations),
  };
  const loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("retained_helpers"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };

  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());

  const loomc_artifact_t* text_artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                   LOOMC_ARTIFACT_FORMAT_LOOM_TEXT);
  ASSERT_NE(text_artifact, nullptr);
  const std::string module_text = ToString(text_artifact->contents);

  const size_t wave32_leaf = module_text.find(
      "low.func.def target<amdgpu.rdna3_5.core>"
      "(@__loom_sealed_target_0) @read_subgroup_size()");
  const size_t wave64_leaf = module_text.find(
      "low.func.def target<amdgpu.cdna3.core>"
      "(@__loom_sealed_target_1) "
      "@read_subgroup_size_spec0()");
  const size_t wave32_forward = module_text.find(
      "low.func.def target<amdgpu.rdna3_5.core>"
      "(@__loom_sealed_target_0) @forward_subgroup_size()");
  const size_t wave64_forward = module_text.find(
      "low.func.def target<amdgpu.cdna3.core>"
      "(@__loom_sealed_target_1) "
      "@forward_subgroup_size_spec0()");
  const size_t wave32_root = module_text.find(
      "low.kernel.def target<amdgpu.rdna3_5.core>"
      "(@__loom_sealed_target_0)",
      wave64_forward);
  const size_t wave64_root = module_text.find(
      "low.kernel.def target<amdgpu.cdna3.core>"
      "(@__loom_sealed_target_1)",
      wave32_root);
  ASSERT_NE(wave32_leaf, std::string::npos) << module_text;
  ASSERT_NE(wave64_leaf, std::string::npos) << module_text;
  ASSERT_NE(wave32_forward, std::string::npos) << module_text;
  ASSERT_NE(wave64_forward, std::string::npos) << module_text;
  ASSERT_NE(wave32_root, std::string::npos) << module_text;
  ASSERT_NE(wave64_root, std::string::npos) << module_text;
  EXPECT_LT(module_text.find("@wave32_root()", wave32_root), wave64_root)
      << module_text;
  EXPECT_NE(module_text.find("@wave64_root()", wave64_root), std::string::npos)
      << module_text;

  const size_t wave32_constant = module_text.find("s_mov_b32 32", wave32_leaf);
  const size_t wave64_constant = module_text.find("s_mov_b32 64", wave64_leaf);
  EXPECT_LT(wave32_constant, wave64_leaf) << module_text;
  EXPECT_LT(wave64_constant, wave32_forward) << module_text;

  const size_t wave32_leaf_call = module_text.find(
      "low.func.call pure @read_subgroup_size()", wave32_forward);
  const size_t wave64_leaf_call = module_text.find(
      "low.func.call pure @read_subgroup_size_spec0()", wave64_forward);
  EXPECT_LT(wave32_leaf_call, wave64_forward) << module_text;
  EXPECT_LT(wave64_leaf_call, wave32_root) << module_text;

  const size_t wave32_forward_call = module_text.find(
      "low.func.call pure @forward_subgroup_size()", wave32_root);
  const size_t wave64_forward_call = module_text.find(
      "low.func.call pure @forward_subgroup_size_spec0()", wave64_root);
  EXPECT_LT(wave32_forward_call, wave64_root) << module_text;
  EXPECT_NE(wave64_forward_call, std::string::npos) << module_text;
  EXPECT_EQ(module_text.find("target.subgroup.size"), std::string::npos)
      << module_text;
  EXPECT_EQ(module_text.find("amdgpu.target<gfx1151>"), 0u) << module_text;
  EXPECT_NE(module_text.find("amdgpu.target<gfx942>"), std::string::npos)
      << module_text;
  EXPECT_EQ(SerializeModuleToText(module.get()), module_text);

  WorkspacePtr clone_workspace = CreateWorkspace();
  loomc_module_t* raw_clone = nullptr;
  LOOMC_ASSERT_OK(loomc_module_clone(module.get(), clone_workspace.get(),
                                     loomc_allocator_system(), &raw_clone));
  ModulePtr clone(raw_clone);
  EXPECT_EQ(SerializeModuleToText(clone.get()), module_text);

  SourcePtr bytecode =
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  WorkspacePtr round_trip_workspace = CreateWorkspace();
  ModulePtr round_trip = DeserializeModule(
      context.get(), round_trip_workspace.get(), bytecode.get());
  const std::string round_trip_text = SerializeModuleToText(round_trip.get());
  EXPECT_NE(round_trip_text.find("amdgpu.target<gfx1151>"), std::string::npos)
      << round_trip_text;
  EXPECT_NE(round_trip_text.find("amdgpu.target<gfx942>"), std::string::npos)
      << round_trip_text;
  EXPECT_NE(round_trip_text.find(
                "target<amdgpu.rdna3_5.core>(@__loom_sealed_target_0)"),
            std::string::npos)
      << round_trip_text;
  EXPECT_NE(round_trip_text.find(
                "target<amdgpu.cdna3.core>(@__loom_sealed_target_1)"),
            std::string::npos)
      << round_trip_text;
  EXPECT_EQ(round_trip_text.find("target.subgroup.size"), std::string::npos)
      << round_trip_text;
}

TEST(AmdgpuTargetTest,
     CompileRejectsIncompatibleExactTargetWithoutBindingAnyFunction) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("incompatible_targets.loom", R"(
amdgpu.target<gfx1170> @gfx1170

func.def public target(@gfx1170) @incompatible() {
  func.return
}

func.def public @otherwise_compatible() {
  func.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_target_specialization_t specializations[] = {
      {
          /*.function_symbol=*/loomc_make_cstring_view("incompatible"),
          /*.target_profile=*/profile.get(),
      },
      {
          /*.function_symbol=*/loomc_make_cstring_view("otherwise_compatible"),
          /*.target_profile=*/profile.get(),
      },
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations,
      /*.specialization_count=*/IREE_ARRAYSIZE(specializations),
  };
  loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
  };

  loomc_result_t* result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result_ptr(result);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_EQ(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "TARGET/052");
  const std::string diagnostic_message = ToString(diagnostic->message);
  EXPECT_NE(diagnostic_message.find("gfx1170"), std::string::npos)
      << diagnostic_message;
  EXPECT_NE(diagnostic_message.find("gfx1151"), std::string::npos)
      << diagnostic_message;
  EXPECT_EQ(diagnostic_message.find("amdgpu-rdna3-5"), std::string::npos)
      << diagnostic_message;

  const std::string module_text = SerializeModuleToText(module.get());
  EXPECT_NE(module_text.find("func.def public target(@gfx1170) @incompatible"),
            std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("func.def public @otherwise_compatible"),
            std::string::npos)
      << module_text;
}

TEST(AmdgpuTargetTest, EmitRuntimeGlobalsFromAmdgpuOptions) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  ModulePtr module =
      CreatePreparedArithmeticModule(context.get(), workspace.get());

  ResultPtr result =
      EmitModule(target_environment.get(), workspace.get(), module.get(),
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
