// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loom/binding/c/test/target/amdgpu/testdata/launch_config_callable_closure_testdata.h"
#include "loomc/artifact.h"
#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/launch_config.h"
#include "loomc/link.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/product.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/status.h"
#include "loomc/target.h"
#include "loomc/target/cmd.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ProductPtr = HandlePtr<loomc_product_t, loomc_product_release>;
using LaunchConfigProgramPtr = HandlePtr<loomc_launch_config_program_t,
                                         loomc_launch_config_program_release>;
using LinkIndexBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using RequestPtr = HandlePtr<loomc_request_t, loomc_request_release>;
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

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result = ToString(contents);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
}

struct KernelRequestCapture {
  // Requests transferred by one command-product construction.
  std::vector<RequestPtr> requests;
};

loomc_status_t CaptureKernelRequest(void* user_data, loomc_request_t* request) {
  KernelRequestCapture* capture = static_cast<KernelRequestCapture*>(user_data);
  capture->requests.push_back(RequestPtr(request));
  return loomc_ok_status();
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

SourcePtr CreateSource(loomc_source_format_t format, const char* identifier,
                       const void* contents, size_t contents_length) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/format,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, contents_length),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  loomc_status_t status =
      loomc_source_create(&options, loomc_allocator_system(), &source);
  LOOMC_EXPECT_OK(status);
  return SourcePtr(source);
}

SourcePtr CreateTextSource(const char* identifier, const char* contents) {
  return CreateSource(LOOMC_SOURCE_FORMAT_TEXT, identifier, contents,
                      strlen(contents));
}

const iree_file_toc_t* FindLaunchConfigCallableClosureSource(
    const char* filename) {
  const iree_file_toc_t* files =
      loomc_launch_config_callable_closure_testdata_create();
  for (iree_host_size_t i = 0;
       i < loomc_launch_config_callable_closure_testdata_size(); ++i) {
    if (strcmp(files[i].name, filename) == 0) return &files[i];
  }
  return nullptr;
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

LinkIndexPtr CreateLinkIndex(loomc_context_t* context, loomc_source_t* source) {
  loomc_link_index_builder_t* raw_builder = nullptr;
  LOOMC_EXPECT_OK(loomc_link_index_builder_create(
      context, nullptr, loomc_allocator_system(), &raw_builder));
  LinkIndexBuilderPtr builder(raw_builder);
  const loomc_link_index_source_options_t source_options = {
      /*.provider_name=*/loomc_make_cstring_view("sealed-replay"),
      /*.role=*/LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  LOOMC_EXPECT_OK(loomc_link_index_builder_add_source(
      builder.get(), source, &source_options, nullptr));

  loomc_link_index_t* raw_index = nullptr;
  loomc_result_t* raw_index_result = nullptr;
  LOOMC_EXPECT_OK(loomc_link_index_builder_finish(builder.get(), &raw_index,
                                                  &raw_index_result));
  LinkIndexPtr index(raw_index);
  ResultPtr index_result(raw_index_result);
  ExpectSucceededResult(index_result.get());
  return index;
}

LinkerPtr CreateLinker(loomc_context_t* context) {
  loomc_linker_t* raw_linker = nullptr;
  LOOMC_EXPECT_OK(loomc_linker_create(context, nullptr,
                                      loomc_allocator_system(), &raw_linker));
  return LinkerPtr(raw_linker);
}

ModulePtr LinkModule(loomc_linker_t* linker, loomc_workspace_t* workspace,
                     loomc_link_index_t* index,
                     const loomc_string_view_t* root_symbols = nullptr,
                     loomc_host_size_t root_symbol_count = 0) {
  const loomc_link_options_t link_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      /*.structure_size=*/sizeof(link_options),
      /*.next=*/nullptr,
      /*.link_index=*/index,
      /*.module_name=*/loomc_make_cstring_view("sealed_replay"),
      /*.mode=*/root_symbol_count == 0 ? LOOMC_LINK_MODE_MERGE
                                       : LOOMC_LINK_MODE_LINK,
      /*.root_symbols=*/root_symbols,
      /*.root_symbol_count=*/root_symbol_count,
  };
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_link_result = nullptr;
  LOOMC_EXPECT_OK(loomc_link_module(linker, workspace, &link_options,
                                    &raw_module, &raw_link_result));
  ResultPtr link_result(raw_link_result);
  ExpectSucceededResult(link_result.get());
  return ModulePtr(raw_module);
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

low.kernel.def target<amdgpu.gfx11.generic.core>(@gfx_target) workgroup_size(64, 1, 1) @loom_kernel() asm {
  %zero = v_mov_b32 0
  %one = v_mov_b32 1
  %sum = v_add_u32 %zero, %one
  return
}
)");
  return DeserializeModule(context, workspace, source.get());
}

TargetProfilePtr CreateTargetProfile(
    loomc_target_environment_t* target_environment, const char* target,
    loomc_amdgpu_amdhsa_feature_states_t amdhsa_features = {}) {
  loomc_amdgpu_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view(target),
      /*.identity=*/
      {
          /*.target=*/loomc_make_cstring_view(target),
          /*.amdhsa_features=*/amdhsa_features,
      },
  };
  loomc_target_profile_t* profile = nullptr;
  loomc_status_t status = loomc_target_profile_create_amdgpu(
      target_environment, &profile_options, loomc_allocator_system(), &profile);
  LOOMC_EXPECT_OK(status);
  return TargetProfilePtr(profile);
}

TEST(AmdgpuTargetTest, ParsesArtifactKeyIdentity) {
  loomc_amdgpu_target_identity_t identity = {};
  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_parse_artifact_key(
      loomc_make_cstring_view("gfx942:sramecc+:xnack-"), &identity));
  EXPECT_EQ(ToString(identity.target), "gfx942");
  EXPECT_EQ(identity.amdhsa_features.sramecc, LOOMC_AMDGPU_TARGET_FEATURE_ON);
  EXPECT_EQ(identity.amdhsa_features.xnack, LOOMC_AMDGPU_TARGET_FEATURE_OFF);

  LOOMC_EXPECT_OK(loomc_amdgpu_target_identity_parse_artifact_key(
      loomc_make_cstring_view("gfx1250-a0"), &identity));
  EXPECT_EQ(ToString(identity.target), "gfx1250-a0");
  EXPECT_EQ(identity.amdhsa_features.sramecc,
            LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED);
  EXPECT_EQ(identity.amdhsa_features.xnack,
            LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED);

  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_amdgpu_target_identity_parse_artifact_key(
          loomc_make_cstring_view("gfx1250:xnack+"), &identity));
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
                         LOOMC_ARTIFACT_MANIFEST_MODE_NONE,
                     loomc_compile_report_mode_t compile_report_mode =
                         LOOMC_COMPILE_REPORT_MODE_NONE) {
  loomc_compile_report_options_t compile_report_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      /*.structure_size=*/sizeof(compile_report_options),
      /*.next=*/nullptr,
      /*.mode=*/compile_report_mode,
      /*.identifier=*/loomc_string_view_empty(),
  };
  loomc_artifact_manifest_options_t artifact_manifest_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      /*.structure_size=*/sizeof(artifact_manifest_options),
      /*.next=*/compile_report_mode != LOOMC_COMPILE_REPORT_MODE_NONE
          ? &compile_report_options
          : nullptr,
      /*.mode=*/artifact_manifest_mode,
      /*.identifier=*/loomc_string_view_empty(),
  };
  loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/artifact_manifest_mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE
          ? static_cast<const void*>(&artifact_manifest_options)
      : compile_report_mode != LOOMC_COMPILE_REPORT_MODE_NONE
          ? static_cast<const void*>(&compile_report_options)
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

void ExpectReplayEmission(const loomc_result_t* result, const char* selector,
                          const char* code_object_target,
                          const char* feature_list = nullptr) {
  ExpectSucceededResult(result);

  const loomc_artifact_t* hsaco =
      FindArtifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
  ASSERT_NE(hsaco, nullptr);
  const std::string hsaco_contents = ToString(hsaco->contents);
  static constexpr uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
  ASSERT_GE(hsaco_contents.size(), sizeof(kElfMagic));
  EXPECT_EQ(std::memcmp(hsaco_contents.data(), kElfMagic, sizeof(kElfMagic)),
            0);
  EXPECT_NE(hsaco_contents.find(code_object_target), std::string::npos);

  const loomc_artifact_t* manifest =
      FindArtifact(result, LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON);
  ASSERT_NE(manifest, nullptr);
  const std::string manifest_text = ToString(manifest->contents);
  EXPECT_NE(
      manifest_text.find(std::string("\"selector\":\"") + selector + "\""),
      std::string::npos)
      << manifest_text;
  EXPECT_NE(
      manifest_text.find(std::string("\"processor\":\"") + selector + "\""),
      std::string::npos)
      << manifest_text;
  EXPECT_NE(manifest_text.find(std::string("\"code_object_target\":\"") +
                               code_object_target + "\""),
            std::string::npos)
      << manifest_text;
  if (feature_list != nullptr) {
    EXPECT_NE(manifest_text.find(std::string("\"features\":") + feature_list),
              std::string::npos)
        << manifest_text;
  }

  const loomc_artifact_t* report =
      FindArtifact(result, LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  ASSERT_NE(report, nullptr);
  const std::string report_text = ToString(report->contents);
  EXPECT_NE(
      report_text.find(std::string("\"namespace\":\"amdgpu\",\"key\":"
                                   "\"processor\",\"value_kind\":\"string\","
                                   "\"value_string\":\"") +
                       selector + "\""),
      std::string::npos)
      << report_text;
}

void ExpectSelectivelyLinkedTarget(const loomc_module_t* module,
                                   const char* target_definition,
                                   const char* representation_contract,
                                   const char* root_symbol,
                                   const char* specialized_constant,
                                   const char* excluded_root_symbol) {
  const std::string module_text = SerializeModuleToText(module);
  EXPECT_NE(module_text.find(target_definition), std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find(representation_contract), std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find(root_symbol), std::string::npos) << module_text;
  EXPECT_NE(module_text.find(specialized_constant), std::string::npos)
      << module_text;
  EXPECT_EQ(module_text.find(excluded_root_symbol), std::string::npos)
      << module_text;
  EXPECT_EQ(module_text.find("target.subgroup.size"), std::string::npos)
      << module_text;
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

template.decl @test.experts_per_wave(%expert_count: index) -> (index)

template.def<@test.experts_per_wave> target(@gfx1151) requires [#target.subgroup.size<64>] priority(20) @two_experts_per_wave(%expert_count: index) -> (index) {
  %c2 = index.constant 2 : index
  template.return %c2 : index
}

template.def<@test.experts_per_wave> priority(1) @four_experts_per_wave(%expert_count: index) -> (index) {
  %c4 = index.constant 4 : index
  template.return %c4 : index
}

kernel.def target(@gfx11_wave64) @target_specialized_launch(%expert_count: index) {
  %c1 = index.constant 1 : index
  %wave_size = target.subgroup.size : index
  %experts_per_wave = template.apply<@test.experts_per_wave>(%expert_count) pure : (index) -> (index)
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
        /*.config_flags=*/0,
        /*.config_module=*/nullptr,
    };

    loomc_result_t* result = nullptr;
    LOOMC_EXPECT_OK(loomc_compile_module(
        compiler.get(), workspace.get(), pass_program.get(), module.get(),
        &options, loomc_allocator_system(), &result));
    ResultPtr result_ptr(result);
    ExpectSucceededResult(result_ptr.get());
    const loomc_artifact_t* artifact =
        FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                     LOOMC_ARTIFACT_FORMAT_VM_BYTECODE);
    ASSERT_NE(artifact, nullptr);

    loomc_launch_config_program_t* launch_program = nullptr;
    LOOMC_EXPECT_OK(loomc_launch_config_program_load(
        artifact, loomc_allocator_system(), &launch_program));
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

TEST(AmdgpuTargetTest, LaunchConfigArtifactInvokesPublicKernelExports) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("multi_launch_config.loom", R"(
amdgpu.target<gfx1151> @gfx1151

kernel.def target(@gfx1151) @initialize(%token_count: index) where [range(%token_count, 1, 512)] {
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
      /*.config_flags=*/0,
      /*.config_module=*/nullptr,
  };
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  const loomc_artifact_t* artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                   LOOMC_ARTIFACT_FORMAT_VM_BYTECODE);
  ASSERT_NE(artifact, nullptr);

  loomc_launch_config_program_t* launch_program = nullptr;
  LOOMC_EXPECT_OK(loomc_launch_config_program_load(
      artifact, loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);

  loomc_launch_config_function_t initialize_function =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("initialize"),
      &initialize_function));
  const uint64_t initialize_arguments[] = {128};
  loomc_launch_config_t initialize_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(initialize_config),
  };
  LOOMC_EXPECT_OK(loomc_launch_config_program_invoke(
      launch_program_ptr.get(), initialize_function, initialize_arguments,
      IREE_ARRAYSIZE(initialize_arguments), &initialize_config));
  EXPECT_EQ(initialize_config.workgroup_count.x, 129u);

  const uint64_t rejected_initialize_arguments[] = {0};
  loomc_launch_config_t rejected_initialize_config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(rejected_initialize_config),
      /*.next=*/nullptr,
      /*.workgroup_count=*/{777, 778, 779},
  };
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_FAILED_PRECONDITION,
                         loomc_launch_config_program_invoke(
                             launch_program_ptr.get(), initialize_function,
                             rejected_initialize_arguments,
                             IREE_ARRAYSIZE(rejected_initialize_arguments),
                             &rejected_initialize_config));
  EXPECT_EQ(rejected_initialize_config.workgroup_count.x, 777u);
  EXPECT_EQ(rejected_initialize_config.workgroup_count.y, 778u);
  EXPECT_EQ(rejected_initialize_config.workgroup_count.z, 779u);

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

TEST(AmdgpuTargetTest, LaunchConfigArtifactPreservesCallableClosure) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  const iree_file_toc_t* source_file = FindLaunchConfigCallableClosureSource(
      "launch_config_callable_closure.loom");
  ASSERT_NE(source_file, nullptr);
  SourcePtr source = CreateSource(LOOMC_SOURCE_FORMAT_TEXT, source_file->name,
                                  source_file->data, source_file->size);
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_compile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("callable_closure"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };
  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  const loomc_artifact_t* artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                   LOOMC_ARTIFACT_FORMAT_VM_BYTECODE);
  ASSERT_NE(artifact, nullptr);

  loomc_launch_config_program_t* launch_program = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_program_load(
      artifact, loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);

  struct TestCase {
    const char* function_name;
    uint64_t argument;
    uint32_t expected_workgroup_count;
  };
  const TestCase test_cases[] = {
      {"call_chain", 5, 18},
      {"call_shared_and_recursive", 7, 21},
  };
  for (const TestCase& test_case : test_cases) {
    loomc_launch_config_function_t launch_function =
        loomc_launch_config_function_invalid();
    LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
        launch_program_ptr.get(),
        loomc_make_cstring_view(test_case.function_name), &launch_function));
    const uint64_t arguments[] = {test_case.argument};
    loomc_launch_config_t launch_config = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        /*.structure_size=*/sizeof(launch_config),
    };
    LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(
        launch_program_ptr.get(), launch_function, arguments,
        IREE_ARRAYSIZE(arguments), &launch_config));
    EXPECT_EQ(launch_config.workgroup_count.x,
              test_case.expected_workgroup_count)
        << test_case.function_name;
    EXPECT_EQ(launch_config.workgroup_size.x, 64u) << test_case.function_name;
  }

  loomc_launch_config_function_t unrelated_function =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_NOT_FOUND,
      loomc_launch_config_program_lookup_function(
          launch_program_ptr.get(), loomc_make_cstring_view("unrelated_extent"),
          &unrelated_function));
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
  SourcePtr config_source = CreateTextSource("configured_store_config.loom", R"(
config.def @test.workgroups_x = 2 : index
config.def @test.workgroup_size_x = 64 : index
)");
  ModulePtr config_module =
      DeserializeModule(context.get(), workspace.get(), config_source.get());
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
      /*.config_flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      /*.config_module=*/config_module.get(),
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
                             "(@__loom_target_context_0_0)"),
            std::string::npos)
      << module_text;
  EXPECT_EQ(
      module_text.find("low.kernel.def target<amdgpu.gfx11.generic.core>"),
      std::string::npos)
      << module_text;
  EXPECT_NE(
      module_text.find("amdgpu.target<gfx1151> @__loom_target_context_0_0"),
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

TEST(AmdgpuTargetTest,
     CommandProductTargetBindingPreservesLaunchConfigurationProjection) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  SourcePtr source = CreateTextSource("specialized_command_request.loom", R"(
target.decl @device

kernel.def target(@device) @record_subgroup_size() {
  %one = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%size, %one, %one) : index
} launch() {
  kernel.return
}

command.program.def public target(@device) @dispatch() launch() {
  kernel.launch @record_subgroup_size() : ()
  command.return
}
)");
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());

  const loomc_target_binding_t target_binding = {
      /*.target_symbol=*/loomc_make_cstring_view("device"),
      /*.target_profile=*/profile.get(),
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/nullptr,
      /*.specialization_count=*/0,
      /*.target_bindings=*/&target_binding,
      /*.target_binding_count=*/1,
  };
  KernelRequestCapture request_capture;
  const loomc_cmd_program_product_options_t product_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      /*.structure_size=*/sizeof(product_options),
      /*.next=*/&target_options,
      /*.link_index=*/index.get(),
      /*.root_symbol_ordinals=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      /*.config=*/{},
      /*.request_sink=*/
      {
          /*.publish=*/CaptureKernelRequest,
          /*.user_data=*/&request_capture,
      },
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* product_result = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_product_build(
      workspace.get(), &product_options, loomc_allocator_system(), &product,
      &product_result));
  ProductPtr product_ptr(product);
  ResultPtr product_result_ptr(product_result);
  ExpectSucceededResult(product_result_ptr.get());
  ASSERT_EQ(loomc_cmd_program_product_program_count(product_ptr.get()), 1u);
  ASSERT_EQ(request_capture.requests.size(), 1u);

  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  const loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/nullptr,
      /*.module_name=*/loomc_make_cstring_view("specialized_command_request"),
      /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
  };
  loomc_product_t* kernel_product = nullptr;
  loomc_result_t* compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_request(
      compiler.get(), workspace.get(), pass_program.get(),
      request_capture.requests[0].get(), &compile_options,
      loomc_allocator_system(), &kernel_product, &compile_result));
  ProductPtr kernel_product_ptr(kernel_product);
  ResultPtr compile_result_ptr(compile_result);
  ExpectSucceededResult(compile_result_ptr.get());
  const loomc_artifact_t* text_artifact =
      loomc_product_artifact_at(kernel_product_ptr.get(), 0);
  ASSERT_NE(text_artifact, nullptr);
  const std::string module_text = ToString(text_artifact->contents);
  EXPECT_NE(module_text.find("target<amdgpu.rdna3_5.core>"), std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("workgroup_size(32, 1, 1)"), std::string::npos)
      << module_text;
}

TEST(AmdgpuTargetTest,
     CommandKernelRequestCompilesForIndependentTargetProfiles) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  TargetProfilePtr wave32_profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  TargetProfilePtr wave64_profile =
      CreateTargetProfile(target_environment.get(), "gfx942");
  SourcePtr source = CreateTextSource("command_request.loom", R"(
kernel.def @record_subgroup_size() {
  %one = index.constant 1 : index
  %size = target.subgroup.size : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%size, %one, %one) : index
} launch() {
  kernel.return
}

command.program.def public @dispatch() launch() {
  kernel.launch @record_subgroup_size() : ()
  command.return
}
)");
  LinkIndexPtr index = CreateLinkIndex(context.get(), source.get());

  KernelRequestCapture request_capture;
  const loomc_cmd_program_product_options_t product_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PRODUCT_OPTIONS,
      /*.structure_size=*/sizeof(product_options),
      /*.next=*/nullptr,
      /*.link_index=*/index.get(),
      /*.root_symbol_ordinals=*/nullptr,
      /*.root_symbol_count=*/0,
      /*.flags=*/LOOMC_CMD_PROGRAM_PRODUCT_FLAG_INCLUDE_INPUT_EXPORTS,
      /*.config=*/{},
      /*.request_sink=*/
      {
          /*.publish=*/CaptureKernelRequest,
          /*.user_data=*/&request_capture,
      },
  };
  loomc_product_t* product = nullptr;
  loomc_result_t* product_result = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_product_build(
      workspace.get(), &product_options, loomc_allocator_system(), &product,
      &product_result));
  ProductPtr product_ptr(product);
  ResultPtr product_result_ptr(product_result);
  ExpectSucceededResult(product_result_ptr.get());
  ASSERT_EQ(loomc_cmd_program_product_program_count(product_ptr.get()), 1u);
  ASSERT_EQ(request_capture.requests.size(), 1u);
  loomc_request_t* request = request_capture.requests[0].get();
  ASSERT_EQ(loomc_request_root_count(request), 1u);
  ASSERT_EQ(loomc_request_binding_count(request), 1u);
  loomc_request_binding_t request_binding = {};
  ASSERT_TRUE(loomc_request_binding_at(request, 0, &request_binding));
  EXPECT_EQ(request_binding.requirement_ordinal, 0u);
  EXPECT_EQ(request_binding.root_ordinal, 0u);
  loomc_source_t* request_source = loomc_request_source(request);
  ASSERT_NE(request_source, nullptr);
  EXPECT_EQ(loomc_source_format(request_source), LOOMC_SOURCE_FORMAT_BYTECODE);

  product_result_ptr.reset();
  product_ptr.reset();
  index.reset();
  source.reset();
  pass_program.reset();
  compiler.reset();
  workspace.reset();
  context.reset();

  ContextPtr request_context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr request_workspace = CreateWorkspace();
  CompilerPtr request_compiler = CreateCompiler(request_context.get());
  PassProgramPtr request_pass_program =
      CreatePreparedLowPassProgram(request_context.get());

  struct TargetCase {
    // Prepared target profile used for this independent compilation.
    loomc_target_profile_t* profile;
    // Target-specific Low descriptor set expected after compilation.
    const char* descriptor_set;
    // Target subgroup size expected in the specialized kernel geometry.
    const char* workgroup_size;
  };
  const TargetCase target_cases[] = {
      {
          wave32_profile.get(),
          "target<amdgpu.rdna3_5.core>",
          "workgroup_size(32, 1, 1)",
      },
      {
          wave64_profile.get(),
          "target<amdgpu.cdna3.core>",
          "workgroup_size(64, 1, 1)",
      },
  };
  for (const TargetCase& target_case : target_cases) {
    ModulePtr request_module = DeserializeModule(
        request_context.get(), request_workspace.get(), request_source);
    const loomc_target_specialization_t specialization = {
        /*.function_symbol=*/
        loomc_make_cstring_view("record_subgroup_size"),
        /*.target_profile=*/target_case.profile,
    };
    const loomc_target_specialization_options_t target_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        /*.structure_size=*/sizeof(target_options),
        /*.next=*/nullptr,
        /*.specializations=*/&specialization,
        /*.specialization_count=*/1,
    };
    const loomc_compile_options_t compile_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        /*.structure_size=*/sizeof(compile_options),
        /*.next=*/&target_options,
        /*.module_name=*/loomc_make_cstring_view("command_request"),
        /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
    };
    loomc_result_t* compile_result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_module(
        request_compiler.get(), request_workspace.get(),
        request_pass_program.get(), request_module.get(), &compile_options,
        loomc_allocator_system(), &compile_result));
    ResultPtr compile_result_ptr(compile_result);
    ExpectSucceededResult(compile_result_ptr.get());
    const loomc_artifact_t* text_artifact =
        FindArtifact(compile_result_ptr.get(), LOOMC_ARTIFACT_KIND_MODULE,
                     LOOMC_ARTIFACT_FORMAT_LOOM_TEXT);
    ASSERT_NE(text_artifact, nullptr);
    const std::string module_text = ToString(text_artifact->contents);
    EXPECT_NE(module_text.find(target_case.descriptor_set), std::string::npos)
        << module_text;
    EXPECT_NE(module_text.find(target_case.workgroup_size), std::string::npos)
        << module_text;
    EXPECT_EQ(module_text.find("target.subgroup.size"), std::string::npos)
        << module_text;
  }
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
      "(@__loom_target_context_0_0) @read_subgroup_size()");
  const size_t wave64_leaf = module_text.find(
      "low.func.def target<amdgpu.cdna3.core>"
      "(@__loom_target_context_0_1) "
      "@read_subgroup_size_spec0()");
  const size_t wave32_forward = module_text.find(
      "low.func.def target<amdgpu.rdna3_5.core>"
      "(@__loom_target_context_0_0) @forward_subgroup_size()");
  const size_t wave64_forward = module_text.find(
      "low.func.def target<amdgpu.cdna3.core>"
      "(@__loom_target_context_0_1) "
      "@forward_subgroup_size_spec0()");
  const size_t wave32_root = module_text.find(
      "low.kernel.def target<amdgpu.rdna3_5.core>"
      "(@__loom_target_context_0_0)",
      wave64_forward);
  const size_t wave64_root = module_text.find(
      "low.kernel.def target<amdgpu.cdna3.core>"
      "(@__loom_target_context_0_1)",
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
  EXPECT_NE(module_text.find("amdgpu.target<gfx1151>"), std::string::npos)
      << module_text;
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
                "target<amdgpu.rdna3_5.core>(@__loom_target_context_0_0)"),
            std::string::npos)
      << round_trip_text;
  EXPECT_NE(round_trip_text.find(
                "target<amdgpu.cdna3.core>(@__loom_target_context_0_1)"),
            std::string::npos)
      << round_trip_text;
  EXPECT_EQ(round_trip_text.find("target.subgroup.size"), std::string::npos)
      << round_trip_text;
}

TEST(AmdgpuTargetTest, CompiledTargetsSurviveFreshContextLinkingAndEmission) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  TargetProfilePtr wave32_profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  TargetProfilePtr wave64_profile =
      CreateTargetProfile(target_environment.get(), "gfx942",
                          {
                              /*.sramecc=*/LOOMC_AMDGPU_TARGET_FEATURE_ON,
                              /*.xnack=*/LOOMC_AMDGPU_TARGET_FEATURE_OFF,
                          });
  SourcePtr source = CreateTextSource("sealed_replay.loom", R"(
amdgpu.target<gfx11-generic> @wave32_requirement {subgroup_size = 32}

kernel.def target(@wave32_requirement) @wave32_root() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%output: buffer) {
  %size = target.subgroup.size : index
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
  %size = target.subgroup.size : index
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
      /*.module_name=*/loomc_make_cstring_view("sealed_replay"),
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
  EXPECT_NE(
      module_text.find("amdgpu.target<gfx1151> @__loom_target_context_0_0 "
                       "{subgroup_size = 32}"),
      std::string::npos)
      << module_text;
  EXPECT_NE(module_text.find("amdgpu.target<gfx942> @__loom_target_context_0_1 "
                             "{features = [sramecc, -xnack]}"),
            std::string::npos)
      << module_text;
  const size_t wave32_root = module_text.find(
      "low.kernel.def target<amdgpu.rdna3_5.core>"
      "(@__loom_target_context_0_0)");
  const size_t wave64_root = module_text.find(
      "low.kernel.def target<amdgpu.cdna3.core>"
      "(@__loom_target_context_0_1)");
  ASSERT_NE(wave32_root, std::string::npos) << module_text;
  ASSERT_NE(wave64_root, std::string::npos) << module_text;
  EXPECT_LT(module_text.find("@wave32_root", wave32_root), wave64_root)
      << module_text;
  EXPECT_NE(module_text.find("@wave64_root", wave64_root), std::string::npos)
      << module_text;
  EXPECT_LT(module_text.find("v_mov_b32 32", wave32_root), wave64_root)
      << module_text;
  EXPECT_NE(module_text.find("v_mov_b32 64", wave64_root), std::string::npos)
      << module_text;
  EXPECT_EQ(module_text.find("target.subgroup.size"), std::string::npos)
      << module_text;

  SourcePtr bytecode =
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE);
  const loomc_byte_span_t bytecode_contents =
      loomc_source_contents(bytecode.get());
  ASSERT_NE(bytecode_contents.data, nullptr);
  ASSERT_NE(bytecode_contents.data_length, 0u);
  const std::vector<uint8_t> sealed_bytecode(
      bytecode_contents.data,
      bytecode_contents.data + bytecode_contents.data_length);

  // Leave only ordinary serialized bytes alive before constructing the replay
  // context. No profile or compiler-owned function version may be reachable by
  // linking, reporting, or emission below.
  bytecode.reset();
  result_ptr.reset();
  source.reset();
  wave64_profile.reset();
  wave32_profile.reset();
  pass_program.reset();
  compiler.reset();
  module.reset();
  workspace.reset();
  context.reset();
  target_environment.reset();

  TargetEnvironmentPtr replay_target_environment =
      CreateAmdgpuTargetEnvironment();
  ContextPtr replay_context =
      CreateAmdgpuContext(replay_target_environment.get());
  SourcePtr replay_text_source =
      CreateSource(LOOMC_SOURCE_FORMAT_TEXT, "sealed_replay.loom",
                   module_text.data(), module_text.size());
  SourcePtr replay_bytecode_source =
      CreateSource(LOOMC_SOURCE_FORMAT_BYTECODE, "sealed_replay.loombc",
                   sealed_bytecode.data(), sealed_bytecode.size());
  LinkIndexPtr text_link_index =
      CreateLinkIndex(replay_context.get(), replay_text_source.get());
  LinkIndexPtr bytecode_link_index =
      CreateLinkIndex(replay_context.get(), replay_bytecode_source.get());
  LinkerPtr replay_linker = CreateLinker(replay_context.get());

  WorkspacePtr text_archive_workspace = CreateWorkspace();
  ModulePtr text_archive = LinkModule(
      replay_linker.get(), text_archive_workspace.get(), text_link_index.get());
  WorkspacePtr bytecode_archive_workspace = CreateWorkspace();
  ModulePtr bytecode_archive =
      LinkModule(replay_linker.get(), bytecode_archive_workspace.get(),
                 bytecode_link_index.get());
  const std::string text_archive_text =
      SerializeModuleToText(text_archive.get());
  const std::string bytecode_archive_text =
      SerializeModuleToText(bytecode_archive.get());
  EXPECT_EQ(text_archive_text, bytecode_archive_text);
  EXPECT_NE(text_archive_text.find(
                "amdgpu.target<gfx1151> @__loom_target_context_0_0 "
                "{subgroup_size = 32}"),
            std::string::npos)
      << text_archive_text;
  EXPECT_NE(
      text_archive_text.find("amdgpu.target<gfx942> @__loom_target_context_0_1 "
                             "{features = [sramecc, -xnack]}"),
      std::string::npos)
      << text_archive_text;
  EXPECT_NE(text_archive_text.find(
                "target<amdgpu.rdna3_5.core>(@__loom_target_context_0_0)"),
            std::string::npos)
      << text_archive_text;
  EXPECT_NE(text_archive_text.find(
                "target<amdgpu.cdna3.core>(@__loom_target_context_0_1)"),
            std::string::npos)
      << text_archive_text;
  EXPECT_EQ(text_archive_text.find("target.subgroup.size"), std::string::npos)
      << text_archive_text;

  const loomc_string_view_t wave32_root_symbol =
      loomc_make_cstring_view("@wave32_root");
  WorkspacePtr text_wave32_workspace = CreateWorkspace();
  ModulePtr text_wave32 =
      LinkModule(replay_linker.get(), text_wave32_workspace.get(),
                 text_link_index.get(), &wave32_root_symbol, 1);
  WorkspacePtr bytecode_wave32_workspace = CreateWorkspace();
  ModulePtr bytecode_wave32 =
      LinkModule(replay_linker.get(), bytecode_wave32_workspace.get(),
                 bytecode_link_index.get(), &wave32_root_symbol, 1);
  static constexpr const char kWave32Target[] =
      "amdgpu.target<gfx1151> @__loom_target_context_0_0 "
      "{subgroup_size = 32}";
  static constexpr const char kWave32Contract[] =
      "target<amdgpu.rdna3_5.core>(@__loom_target_context_0_0)";
  ExpectSelectivelyLinkedTarget(text_wave32.get(), kWave32Target,
                                kWave32Contract, "@wave32_root", "v_mov_b32 32",
                                "@wave64_root");
  ExpectSelectivelyLinkedTarget(bytecode_wave32.get(), kWave32Target,
                                kWave32Contract, "@wave32_root", "v_mov_b32 32",
                                "@wave64_root");

  const loomc_string_view_t wave64_root_symbol =
      loomc_make_cstring_view("@wave64_root");
  WorkspacePtr text_wave64_workspace = CreateWorkspace();
  ModulePtr text_wave64 =
      LinkModule(replay_linker.get(), text_wave64_workspace.get(),
                 text_link_index.get(), &wave64_root_symbol, 1);
  WorkspacePtr bytecode_wave64_workspace = CreateWorkspace();
  ModulePtr bytecode_wave64 =
      LinkModule(replay_linker.get(), bytecode_wave64_workspace.get(),
                 bytecode_link_index.get(), &wave64_root_symbol, 1);
  static constexpr const char kWave64Target[] =
      "amdgpu.target<gfx942> @__loom_target_context_0_1 "
      "{features = [sramecc, -xnack]}";
  static constexpr const char kWave64Contract[] =
      "target<amdgpu.cdna3.core>(@__loom_target_context_0_1)";
  ExpectSelectivelyLinkedTarget(text_wave64.get(), kWave64Target,
                                kWave64Contract, "@wave64_root", "v_mov_b32 64",
                                "@wave32_root");
  ExpectSelectivelyLinkedTarget(bytecode_wave64.get(), kWave64Target,
                                kWave64Contract, "@wave64_root", "v_mov_b32 64",
                                "@wave32_root");

  ResultPtr wave32_emit = EmitModule(
      replay_target_environment.get(), text_wave32_workspace.get(),
      text_wave32.get(), LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
      LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY, LOOMC_COMPILE_REPORT_MODE_DETAILS);
  ExpectReplayEmission(wave32_emit.get(), "gfx1151",
                       "amdgcn-amd-amdhsa--gfx1151");

  ResultPtr wave64_emit = EmitModule(
      replay_target_environment.get(), bytecode_wave64_workspace.get(),
      bytecode_wave64.get(), LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
      LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY, LOOMC_COMPILE_REPORT_MODE_DETAILS);
  ExpectReplayEmission(wave64_emit.get(), "gfx942",
                       "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-",
                       "[\"sramecc+\",\"xnack-\"]");
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
