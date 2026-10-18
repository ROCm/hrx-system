// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/reporting/format.h"
#include "loomc/artifact.h"
#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/compile_report.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/interop.h"
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
#include "loomc/workspace.h"
#include "target_test_util.h"
#include "test/util.h"

namespace loomc::testing::amdgpu {
namespace {

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

TEST(AmdgpuTargetTest, NativeReportSurvivesFailedExecutionCleanup) {
  auto environment = CreateAmdgpuTargetEnvironment();
  auto context = CreateAmdgpuContext(environment.get());
  auto workspace = CreateWorkspace();
  auto source = CreateTextSource("report.loom", R"(
amdgpu.target<gfx11-generic> @target

kernel.def target(@target) @report_kernel() {
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c64, %c1, %c1) : index
} launch(%lhs_buffer: buffer, %rhs_buffer: buffer, %output: buffer) {
  %tid = kernel.workitem.id<x> : index
  %base = index.constant 0 : offset
  %lhs_view = buffer.view %lhs_buffer[%base] : buffer -> view<64xf32>
  %rhs_view = buffer.view %rhs_buffer[%base] : buffer -> view<64xf32>
  %output_view = buffer.view %output[%base] : buffer -> view<64xf32>
  %lhs = vector.load %lhs_view[%tid] : view<64xf32> -> vector<1xf32>
  %rhs = vector.load %rhs_view[%tid] : view<64xf32> -> vector<1xf32>
  %sum = vector.addf %lhs, %rhs : vector<1xf32>
  vector.store %sum, %output_view[%tid] : vector<1xf32>, view<64xf32>
  kernel.return
}
)");
  auto module = DeserializeModule(context.get(), workspace.get(), source.get());
  auto program = CreatePreparedLowPassProgram(context.get());
  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      loom_target_compile_report_requested_detail_flags(
          LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS);
  const loomc_string_view_t filter = loomc_make_cstring_view("source-to-low");
  const loomc_pass_trace_options_t trace = {
      .type = LOOMC_STRUCTURE_TYPE_PASS_TRACE_OPTIONS,
      .structure_size = sizeof(trace),
      .after_filters = &filter,
      .after_filter_count = 1,
      .sink =
          {
              .write =
                  [](void*, loomc_string_view_t) {
                    return loomc_status_from_code(LOOMC_STATUS_UNAVAILABLE);
                  },
          },
  };
  const loomc_native_compile_report_options_t report_options = {
      .type = LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(report_options),
      .next = &trace,
      .report = &report,
  };
  const loomc_pass_run_options_t options = {.next = &report_options};
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_UNAVAILABLE,
      loomc_pass_program_run(program.get(), workspace.get(), module.get(),
                             &options, loomc_allocator_system(), &raw_result));
  ResultPtr result(raw_result);
  EXPECT_EQ(result.get(), nullptr);
  EXPECT_GT(report.source_low_rows.count, 0u);
  // Formatting consumes every retained row after invocation arenas and failed
  // compilation facts have been reclaimed, while the public module stays live.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  const loom_target_compile_report_format_options_t format = {
      .mode = LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS,
  };
  IREE_EXPECT_OK(
      loom_target_compile_report_format_json(&report, &format, &stream));
  const auto text = iree_string_builder_view(&builder);
  EXPECT_NE(std::string(text.data, text.size).find("report_kernel"),
            std::string::npos);
  iree_string_builder_deinitialize(&builder);
  loom_target_compile_report_deinitialize(&report);
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
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
        .structure_size = sizeof(options),
        .next = nullptr,
        .identifier = {},
        .identity =
            {
                .target = loomc_make_cstring_view(target_name),
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
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = {},
      .identity =
          {
              .target = loomc_make_cstring_view("gfx1250-a1"),
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
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("gfx942-features"),
      .identity =
          {
              .target = loomc_make_cstring_view("gfx942"),
              .amdhsa_features =
                  {
                      .sramecc = LOOMC_AMDGPU_TARGET_FEATURE_ON,
                      .xnack = LOOMC_AMDGPU_TARGET_FEATURE_OFF,
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
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("gfx1151-sramecc"),
      .identity =
          {
              .target = loomc_make_cstring_view("gfx1151"),
              .amdhsa_features =
                  {
                      .sramecc = LOOMC_AMDGPU_TARGET_FEATURE_ON,
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
                     loomc_artifact_manifest_mode_t artifact_manifest_mode =
                         LOOMC_ARTIFACT_MANIFEST_MODE_NONE,
                     loomc_compile_report_mode_t compile_report_mode =
                         LOOMC_COMPILE_REPORT_MODE_NONE) {
  loomc_compile_report_options_t compile_report_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
      .structure_size = sizeof(compile_report_options),
      .next = nullptr,
      .mode = compile_report_mode,
      .format = LOOMC_COMPILE_REPORT_FORMAT_JSON,
      .identifier = loomc_string_view_empty(),
  };
  loomc_artifact_manifest_options_t artifact_manifest_options = {
      .type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS,
      .structure_size = sizeof(artifact_manifest_options),
      .next = compile_report_mode != LOOMC_COMPILE_REPORT_MODE_NONE
                  ? &compile_report_options
                  : nullptr,
      .mode = artifact_manifest_mode,
      .identifier = loomc_string_view_empty(),
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = artifact_manifest_mode != LOOMC_ARTIFACT_MANIFEST_MODE_NONE
                  ? static_cast<const void*>(&artifact_manifest_options)
              : compile_report_mode != LOOMC_COMPILE_REPORT_MODE_NONE
                  ? static_cast<const void*>(&compile_report_options)
                  : nullptr,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .identifier = loomc_make_cstring_view("loom_kernel.hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_emit_module(target_environment, workspace, module, &emit_options,
                        loomc_allocator_system(), &result);
  LOOMC_EXPECT_OK(status);
  return ResultPtr(result);
}

void ExpectReplayEmission(const loomc_result_t* result, const char* selector,
                          const char* artifact_target_key,
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
  EXPECT_NE(report_text.find("\"artifact_kind\":\"hal-executable\""),
            std::string::npos);
  EXPECT_NE(report_text.find("\"backend\":\"amdgpu-hal\""), std::string::npos);
  EXPECT_NE(report_text.find("\"target_family\":\"amdgpu\""),
            std::string::npos);
  EXPECT_NE(report_text.find(std::string("\"target_key\":\"") +
                             artifact_target_key + "\""),
            std::string::npos);
  EXPECT_NE(report_text.find("\"residency_constraints\":{"), std::string::npos);
  EXPECT_NE(
      report_text.find("\"name\":\"amdgpu.lds\",\"kind\":\"pooled_resource\""),
      std::string::npos);
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
        .function_symbol = loomc_make_cstring_view("target_specialized_launch"),
        .target_profile = profile.get(),
    };
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = nullptr,
        .specializations = &specialization,
        .specialization_count = 1,
    };
    const loomc_compile_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(options),
        .next = &target_options,
        .module_name = loomc_make_cstring_view("target_specialized_launch"),
        .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
        .config_flags = 0,
        .config_module = nullptr,
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
        .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
        .structure_size = sizeof(launch_config),
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

TEST(AmdgpuTargetTest, LaunchConfigArtifactProjectsScalarWorkloads) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("projected_launch.loom", R"(
amdgpu.target<gfx1151> @gfx1151

kernel.def target(@gfx1151) @projected_launch(%before: index, %counts: buffer, %after: index) {
  %bounded_before = index.assume %before [range(%before, 0, 255)] : index
  %bounded_after = index.assume %after [range(%after, 0, 255)] : index
  %base = index.constant 0 : offset
  %loaded = buffer.load.i8.u %counts[%base]
  %zero = scalar.constant 0 : i32
  %folded = scalar.andi %loaded, %zero : i32
  %folded_index = index.cast %folded : i32 to index
  %two = index.constant 2 : index
  %scaled_before = index.mul %bounded_before, %two : index
  %sum = index.add %scaled_before, %bounded_after : index
  %workgroup_count = index.add %sum, %folded_index : index
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  const loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .module_name = loomc_make_cstring_view("projected_launch"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
      .config_flags = 0,
      .config_module = nullptr,
  };

  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  const loomc_artifact_t* artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                   LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  ASSERT_NE(artifact, nullptr);

  loomc_launch_config_program_t* launch_program = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_program_load(
      artifact, loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);
  result_ptr.reset();

  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("projected_launch"),
      &launch_function));
  const uint64_t scalar_workload_argument_bits[] = {5, 7};
  loomc_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(
      launch_program_ptr.get(), launch_function, scalar_workload_argument_bits,
      IREE_ARRAYSIZE(scalar_workload_argument_bits), &launch_config));
  EXPECT_EQ(launch_config.workgroup_count.x, 17u);
}

TEST(AmdgpuTargetTest, LaunchConfigArtifactRejectsLiveNonScalarWorkload) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("live_reference_launch.loom", R"(
amdgpu.target<gfx1151> @gfx1151

kernel.def target(@gfx1151) @live_reference_launch(%counts: buffer) {
  %base = index.constant 0 : offset
  %loaded = buffer.load.i8.u %counts[%base]
  %workgroup_count = index.cast %loaded : i32 to index
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");

  const loomc_compile_options_t baseline_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(baseline_options),
      .next = nullptr,
      .module_name = loomc_make_cstring_view("live_reference_launch"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
      .config_flags = 0,
      .config_module = nullptr,
  };
  ModulePtr baseline_module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_result_t* baseline_result = nullptr;
  LOOMC_ASSERT_OK(
      loomc_compile_module(compiler.get(), workspace.get(), pass_program.get(),
                           baseline_module.get(), &baseline_options,
                           loomc_allocator_system(), &baseline_result));
  ResultPtr baseline_result_ptr(baseline_result);
  ExpectSucceededResult(baseline_result_ptr.get());

  const loomc_compile_options_t launch_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(launch_options),
      .next = nullptr,
      .module_name = loomc_make_cstring_view("live_reference_launch"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
      .config_flags = 0,
      .config_module = nullptr,
  };
  ModulePtr launch_module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  loomc_result_t* launch_result = nullptr;
  iree::Status status = ConsumeStatusForTest(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), launch_module.get(),
      &launch_options, loomc_allocator_system(), &launch_result));
  EXPECT_THAT(status, iree::testing::status::StatusIs(
                          iree::StatusCode::kFailedPrecondition));
  EXPECT_THAT(
      status.ToString(),
      ::testing::HasSubstr(
          "kernel '@live_reference_launch' launch-config artifact still "
          "depends on non-scalar workload argument 0 (%counts) after "
          "specialization"));
  EXPECT_EQ(launch_result, nullptr);
}

TEST(AmdgpuTargetTest,
     CompileArtifactReturnsExecutableLaunchConfigAndTargetListing) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  SourcePtr source = CreateTextSource("complete_launch.loom", R"(
amdgpu.target<gfx1151> @gfx1151

kernel.def target(@gfx1151) @complete_launch(%group_count: index) {
  %one = index.constant 1 : index
  %wave_size = index.constant 64 : index
  kernel.launch.config workgroups(%group_count, %one, %one) workgroup_size(%wave_size, %one, %one) : index
} launch() {
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  TargetProfilePtr profile =
      CreateTargetProfile(target_environment.get(), "gfx1151");
  const loomc_string_view_t root = loomc_make_cstring_view("complete_launch");
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = nullptr,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .identifier = loomc_make_cstring_view("complete_launch.hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY |
                        LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING,
  };
  const loomc_compile_artifact_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .roots = &root,
      .root_count = 1,
      .excluded_roots = nullptr,
      .excluded_root_count = 0,
      .target_profile = profile.get(),
      .config = nullptr,
      .emit_options = &emit_options,
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
  };

  loomc_result_t* result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &options, loomc_allocator_system(), &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  ASSERT_NE(FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                         LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
            nullptr);
  const loomc_artifact_t* target_listing = FindArtifact(
      result_ptr.get(), LOOMC_ARTIFACT_KIND_TEXT, "amdgpu-assembly");
  ASSERT_NE(target_listing, nullptr);
  EXPECT_EQ(ToString(target_listing->identifier),
            "complete_launch.hsaco.listing");
  EXPECT_NE(ToString(target_listing->contents).find("complete_launch"),
            std::string::npos);
  const loomc_artifact_t* launch_artifact =
      FindArtifact(result_ptr.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
                   LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
  ASSERT_NE(launch_artifact, nullptr);

  loomc_launch_config_program_t* launch_program = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_program_load(
      launch_artifact, loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);
  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("complete_launch"),
      &launch_function));
  const uint64_t workload_argument_bits[] = {17};
  loomc_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_program_invoke(
      launch_program_ptr.get(), launch_function, workload_argument_bits,
      IREE_ARRAYSIZE(workload_argument_bits), &launch_config));
  EXPECT_EQ(launch_config.workgroup_count.x, 17u);
  EXPECT_EQ(launch_config.workgroup_size.x, 64u);
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
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .module_name = loomc_make_cstring_view("multi_launch_config"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG,
      .config_flags = 0,
      .config_module = nullptr,
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
      artifact, loomc_allocator_system(), &launch_program));
  LaunchConfigProgramPtr launch_program_ptr(launch_program);

  loomc_launch_config_function_t prefill_function =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_program_lookup_function(
      launch_program_ptr.get(), loomc_make_cstring_view("prefill"),
      &prefill_function));
  const uint64_t prefill_arguments[] = {128};
  loomc_launch_config_t prefill_config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(prefill_config),
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
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(decode_config),
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
  SourcePtr config_source = CreateTextSource("configured_store_config.loom", R"(
config.def @test.workgroups_x = 2 : index
config.def @test.workgroup_size_x = 64 : index
)");
  ModulePtr config_module =
      DeserializeModule(context.get(), workspace.get(), config_source.get());
  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view("configured_store"),
      .target_profile = profile.get(),
  };
  loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .specializations = &specialization,
      .specialization_count = 1,
  };
  loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("configured_store"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
      .config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      .config_module = config_module.get(),
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
                 LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY);
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
     CompileSpecializesAndInlinesHelpersForEachTargetContext) {
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
          .function_symbol = loomc_make_cstring_view("wave32_root"),
          .target_profile = wave32_profile.get(),
      },
      {
          .function_symbol = loomc_make_cstring_view("wave64_root"),
          .target_profile = wave64_profile.get(),
      },
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .specializations = specializations,
      .specialization_count = IREE_ARRAYSIZE(specializations),
  };
  const loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("retained_helpers"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
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

  const size_t wave32_root = module_text.find(
      "low.kernel.def target<amdgpu.rdna3_5.core>"
      "(@__loom_target_context_0_0)");
  const size_t wave64_root = module_text.find(
      "low.kernel.def target<amdgpu.cdna3.core>"
      "(@__loom_target_context_0_1)",
      wave32_root);
  ASSERT_NE(wave32_root, std::string::npos) << module_text;
  ASSERT_NE(wave64_root, std::string::npos) << module_text;
  EXPECT_LT(module_text.find("@wave32_root()", wave32_root), wave64_root)
      << module_text;
  EXPECT_NE(module_text.find("@wave64_root()", wave64_root), std::string::npos)
      << module_text;

  const size_t wave32_constant = module_text.find("v_mov_b32 32", wave32_root);
  const size_t wave64_constant = module_text.find("v_mov_b32 64", wave64_root);
  EXPECT_LT(wave32_constant, wave64_root) << module_text;
  EXPECT_NE(wave64_constant, std::string::npos) << module_text;
  EXPECT_EQ(module_text.find("low.func.def"), std::string::npos) << module_text;
  EXPECT_EQ(module_text.find("low.func.call"), std::string::npos)
      << module_text;
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
                              .sramecc = LOOMC_AMDGPU_TARGET_FEATURE_ON,
                              .xnack = LOOMC_AMDGPU_TARGET_FEATURE_OFF,
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
          .function_symbol = loomc_make_cstring_view("wave32_root"),
          .target_profile = wave32_profile.get(),
      },
      {
          .function_symbol = loomc_make_cstring_view("wave64_root"),
          .target_profile = wave64_profile.get(),
      },
  };
  const loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .specializations = specializations,
      .specialization_count = IREE_ARRAYSIZE(specializations),
  };
  const loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("sealed_replay"),
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT,
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

  ResultPtr wave32_emit =
      EmitModule(replay_target_environment.get(), text_wave32_workspace.get(),
                 text_wave32.get(), LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY,
                 LOOMC_COMPILE_REPORT_MODE_DETAILS);
  ExpectReplayEmission(wave32_emit.get(), "gfx1151", "gfx1151",
                       "amdgcn-amd-amdhsa--gfx1151");

  ResultPtr wave64_emit = EmitModule(
      replay_target_environment.get(), bytecode_wave64_workspace.get(),
      bytecode_wave64.get(), LOOMC_ARTIFACT_MANIFEST_MODE_SUMMARY,
      LOOMC_COMPILE_REPORT_MODE_DETAILS);
  ExpectReplayEmission(wave64_emit.get(), "gfx942", "gfx942:sramecc+:xnack-",
                       "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-",
                       "[\"sramecc+\",\"xnack-\"]");

  ResultPtr summary_emit =
      EmitModule(replay_target_environment.get(), text_wave32_workspace.get(),
                 text_wave32.get(), LOOMC_ARTIFACT_MANIFEST_MODE_NONE,
                 LOOMC_COMPILE_REPORT_MODE_SUMMARY);
  ExpectSucceededResult(summary_emit.get());
  text_wave32.reset();
  text_wave32_workspace.reset();
  const auto* summary_report =
      FindArtifact(summary_emit.get(), LOOMC_ARTIFACT_KIND_REPORT,
                   LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
  ASSERT_NE(summary_report, nullptr);
  const std::string summary_text = ToString(summary_report->contents);
  EXPECT_NE(summary_text.find("\"mode\":\"summary\""), std::string::npos);
  EXPECT_NE(summary_text.find("\"target_family\":\"amdgpu\""),
            std::string::npos);
  EXPECT_NE(summary_text.find("\"target_key\":\"gfx1151\""), std::string::npos);
  EXPECT_NE(
      summary_text.find("\"residency_constraints\":{\"count\":4,\"rows\":["),
      std::string::npos);
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
          .function_symbol = loomc_make_cstring_view("incompatible"),
          .target_profile = profile.get(),
      },
      {
          .function_symbol = loomc_make_cstring_view("otherwise_compatible"),
          .target_profile = profile.get(),
      },
  };
  loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .specializations = specializations,
      .specialization_count = IREE_ARRAYSIZE(specializations),
  };
  loomc_compile_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(options),
      .next = &target_options,
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

TEST(AmdgpuTargetTest, PipelineSchedulesSurviveSeparateCompileAndRepeatedEmit) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreatePreparedLowPassProgram(context.get());
  SourcePtr source = CreateTextSource("loop_pipeline.loom", R"(
amdgpu.target<gfx1151> @gpu
config.def @first_depth = 1 : index
config.def @second_depth = 4 : index

kernel.def target(@gpu) @loop_pipeline() {
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%count: index, %input: buffer, %output: buffer) {
  %end = index.assume %count [range(%count, 0, 64)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %first_depth = config.get @first_depth : index
  %second_depth = config.get @second_depth : index
  %base = index.constant 0 : offset
  %initial = scalar.constant 0.0 : f32
  %data = buffer.view %input[%base] : buffer -> view<64xf32>
  %destination = buffer.view %output[%base] : buffer -> view<1xf32>
  %first = scf.for %i = [%c0 to %end step %c1](%sum = %initial : f32) -> (f32) pipeline(%first_depth) {
    %value = view.load %data[%i] : view<64xf32> -> f32
    %next = scalar.addf %sum, %value : f32
    scf.yield %next : f32
  }
  %second = scf.for %j = [%c0 to %end step %c1](%sum = %first : f32) -> (f32) pipeline(%second_depth) {
    %value = view.load %data[%j] : view<64xf32> -> f32
    %next = scalar.addf %sum, %value : f32
    scf.yield %next : f32
  }
  %third = scf.for %k = [%c0 to %end step %c1](%sum = %second : f32) -> (f32) {
    %value = view.load %data[%k] : view<64xf32> -> f32
    %next = scalar.addf %sum, %value : f32
    scf.yield %next : f32
  }
  view.store %third, %destination[0] : f32, view<1xf32>
  kernel.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  SourcePtr config_source = CreateTextSource("loop_config.loom", R"(
config.def @first_depth = 3 : index
config.def @second_depth = 1 : index
config.def @unused = 9 : index
)");
  ModulePtr config =
      DeserializeModule(context.get(), workspace.get(), config_source.get());
  loomc_compile_options_t compile_options = {.config_module = config.get()};
  loomc_result_t* raw_compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      &compile_options, loomc_allocator_system(), &raw_compile_result));
  ResultPtr compile_result(raw_compile_result);
  ExpectSucceededResult(compile_result.get());
  ASSERT_TRUE(loomc_result_succeeded(compile_result.get()));
  compile_result.reset();

  // Continue through both compilation and prepared execution. Serialization
  // must preserve live schedules and configuration for subsequent emission.
  raw_compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_module(
      compiler.get(), workspace.get(), pass_program.get(), module.get(),
      nullptr, loomc_allocator_system(), &raw_compile_result));
  compile_result.reset(raw_compile_result);
  ExpectSucceededResult(compile_result.get());
  ASSERT_TRUE(loomc_result_succeeded(compile_result.get()));
  compile_result.reset();
  SourcePtr serialized[] = {
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_TEXT),
      SerializeModule(module.get(), LOOMC_SOURCE_FORMAT_BYTECODE),
  };
  raw_compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_run(
      pass_program.get(), workspace.get(), module.get(), nullptr,
      loomc_allocator_system(), &raw_compile_result));
  compile_result.reset(raw_compile_result);
  ExpectSucceededResult(compile_result.get());
  ASSERT_TRUE(loomc_result_succeeded(compile_result.get()));
  compile_result.reset();
  raw_compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_verify(
      module.get(), nullptr, loomc_allocator_system(), &raw_compile_result));
  compile_result.reset(raw_compile_result);
  ExpectSucceededResult(compile_result.get());
  ASSERT_TRUE(loomc_result_succeeded(compile_result.get()));
  compile_result.reset();
  WorkspacePtr clone_workspace = CreateWorkspace();
  loomc_module_t* raw_clone = nullptr;
  LOOMC_ASSERT_OK(loomc_module_clone(module.get(), clone_workspace.get(),
                                     loomc_allocator_system(), &raw_clone));
  module.reset(raw_clone);
  workspace = std::move(clone_workspace);
  config.reset();
  config_source.reset();
  pass_program.reset();
  compiler.reset();
  source.reset();
  loomc_workspace_trim(workspace.get());

  std::string first_details;
  for (loomc_compile_report_mode_t mode :
       {LOOMC_COMPILE_REPORT_MODE_SUMMARY, LOOMC_COMPILE_REPORT_MODE_DETAILS,
        LOOMC_COMPILE_REPORT_MODE_DETAILS}) {
    ResultPtr emitted =
        EmitModule(target_environment.get(), workspace.get(), module.get(),
                   LOOMC_ARTIFACT_MANIFEST_MODE_NONE, mode);
    ExpectSucceededResult(emitted.get());
    ASSERT_NE(FindArtifact(emitted.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                           LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
              nullptr);
    const loomc_artifact_t* artifact =
        FindArtifact(emitted.get(), LOOMC_ARTIFACT_KIND_REPORT,
                     LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON);
    ASSERT_NE(artifact, nullptr);
    const std::string report = ToString(artifact->contents);
    EXPECT_NE(report.find("\"key\":\"first_depth\",\"value\":\"3\""),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("\"key\":\"second_depth\",\"value\":\"1\""),
              std::string::npos)
        << report;
    EXPECT_EQ(report.find("unused"), std::string::npos);
    const size_t begin = report.find("\"loop_pipelines\":");
    ASSERT_NE(begin, std::string::npos) << report;
    const size_t end = report.find("\"memory\":", begin);
    ASSERT_NE(end, std::string::npos);
    const std::string schedules = report.substr(begin, end - begin);
    EXPECT_NE(schedules.find("\"count\":2"), std::string::npos);
    EXPECT_NE(schedules.find("\"depth\":3"), std::string::npos);
    EXPECT_NE(schedules.find("\"depth\":1"), std::string::npos);
    EXPECT_NE(schedules.find("\"outcome\":\"serial\""), std::string::npos);
    if (mode == LOOMC_COMPILE_REPORT_MODE_SUMMARY) {
      EXPECT_EQ(schedules.find("\"stages\":"), std::string::npos);
    } else {
      EXPECT_NE(schedules.find("\"stage\":\"producer\""), std::string::npos);
      EXPECT_NE(schedules.find("\"stage\":\"consumer\""), std::string::npos);
      EXPECT_NE(schedules.find("\"iteration_lookahead\":2"), std::string::npos);
      if (first_details.empty()) {
        first_details = schedules;
      } else {
        EXPECT_EQ(schedules, first_details);
      }
    }
    emitted.reset();
    loomc_workspace_trim(workspace.get());
  }

  module.reset();
  workspace.reset();
  context.reset();
  target_environment.reset();
  for (const auto& snapshot : serialized) {
    auto replay_environment = CreateAmdgpuTargetEnvironment();
    auto replay_context = CreateAmdgpuContext(replay_environment.get());
    auto replay_workspace = CreateWorkspace();
    auto replay_module = DeserializeModule(
        replay_context.get(), replay_workspace.get(), snapshot.get());
    auto replay_program = CreatePreparedLowPassProgram(replay_context.get());
    loomc_result_t* result = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_run(
        replay_program.get(), replay_workspace.get(), replay_module.get(),
        nullptr, loomc_allocator_system(), &result));
    ResultPtr result_owner(result);
    ExpectSucceededResult(result);
    ASSERT_TRUE(loomc_result_succeeded(result));
    auto emitted = EmitModule(replay_environment.get(), replay_workspace.get(),
                              replay_module.get());
    ExpectSucceededResult(emitted.get());
    ASSERT_NE(FindArtifact(emitted.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                           LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
              nullptr);
  }
}

TEST(AmdgpuTargetTest, EmitRuntimeGlobalsFromPreparedModule) {
  TargetEnvironmentPtr target_environment = CreateAmdgpuTargetEnvironment();
  ContextPtr context = CreateAmdgpuContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource("amdgpu_runtime_globals.loom", R"(
global.rodata.decl @iree_asan_config
global.rodata.decl @iree_feedback_config

amdgpu.target<gfx11-generic> @gfx_target

low.kernel.def target<amdgpu.gfx11.generic.core>(@gfx_target) workgroup_size(64, 1, 1) @loom_kernel() {
  low.return
}
)");
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());

  ResultPtr result =
      EmitModule(target_environment.get(), workspace.get(), module.get());
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
}  // namespace loomc::testing::amdgpu
