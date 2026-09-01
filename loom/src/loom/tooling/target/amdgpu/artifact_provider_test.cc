// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amdgpu/artifact_provider.h"

#include <string>

#include "iree/base/byte_sequence.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/sanitizer/options.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/runtime_globals.h"
#include "loom/testing/byte_sequence.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::ModulePtr;

static const loom_amdgpu_target_info_t* LookupTarget(const char* name) {
  const loom_amdgpu_target_info_t* target = nullptr;
  IREE_CHECK_OK(loom_amdgpu_target_info_lookup_target(
      iree_make_cstring_view(name), &target));
  return target;
}

iree_status_t InitializeAmdgpuContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_ops_register_dialect(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

class AmdgpuArtifactProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeAmdgpuContext(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_text_parse_options_t ParseOptions() const {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    return options;
  }

  iree_status_t ParseModule(iree_string_view_t source, ModulePtr* out_module) {
    loom_text_parse_options_t options = ParseOptions();
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_text_parse(source, IREE_SV("amdgpu_provider_test.loom"), &context_,
                        &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_status_t ParsePreparedArithmeticModule(iree_string_view_t target,
                                              iree_string_view_t target_attrs,
                                              ModulePtr* out_module) {
    const loom_amdgpu_target_info_t* target_info = nullptr;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_target_info_lookup_target(target, &target_info));
    std::string source = "amdgpu.target<";
    source.append(target.data, target.size);
    source.append("> @gfx_target");
    if (!iree_string_view_is_empty(target_attrs)) {
      source.append(" {");
      source.append(target_attrs.data, target_attrs.size);
      source.append("}");
    }
    source.append("\nlow.kernel.def target<");
    source.append(target_info->descriptor_set_key.data,
                  target_info->descriptor_set_key.size);
    source.append(
        ">(@gfx_target) workgroup_size(64, 1, 1) "
        "@loom_kernel() {\n"
        "  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : "
        "reg<amdgpu.vgpr>\n"
        "  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : "
        "reg<amdgpu.vgpr>\n"
        "  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : "
        "(reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>\n"
        "  low.return\n"
        "}\n");
    return ParseModule(iree_make_string_view(source.data(), source.size()),
                       out_module);
  }

  iree_status_t ParsePreparedArithmeticModule(iree_string_view_t target,
                                              ModulePtr* out_module) {
    return ParsePreparedArithmeticModule(target, iree_string_view_empty(),
                                         out_module);
  }

  void ExpectEmitsModuleTarget(iree_string_view_t target_name) {
    ModulePtr module;
    IREE_ASSERT_OK(ParsePreparedArithmeticModule(target_name, &module));
    ASSERT_NE(module.get(), nullptr);

    loom_artifact_target_t target = {};
    loom_compile_options_t options = {};
    loom_compile_options_initialize(&options);
    loom_artifact_t artifact = {};
    bool emitted = false;
    IREE_ASSERT_OK(loom_amdgpu_artifact_provider.emit_artifact(
        &loom_amdgpu_artifact_provider, module.get(), &target, &options,
        iree_allocator_system(), &emitted, &artifact));
    EXPECT_TRUE(emitted);
    EXPECT_NE(iree_string_view_find(artifact.target_key, target_name, 0),
              IREE_STRING_VIEW_NPOS);
    EXPECT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
    EXPECT_EQ(artifact.target_artifact_data, artifact.executable_data);
    ASSERT_NE(artifact.target_artifact_data, nullptr);
    EXPECT_GT(iree_byte_sequence_length(artifact.target_artifact_data), 0u);
    ASSERT_NE(artifact.target_bundle, nullptr);
    EXPECT_EQ(artifact.target_bundle->snapshot->codegen_format,
              LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE);
    EXPECT_EQ(artifact.target_bundle->snapshot->artifact_format,
              LOOM_TARGET_ARTIFACT_FORMAT_ELF);
    EXPECT_EQ(artifact.target_bundle->export_plan->abi_kind,
              LOOM_TARGET_ABI_HAL_KERNEL);

    loom_amdgpu_artifact_provider.deinitialize_artifact(
        &loom_amdgpu_artifact_provider, &artifact, iree_allocator_system());
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(AmdgpuArtifactProviderTest, SelectTargetKeyBuildsOfflineTarget) {
  loom_artifact_target_t target = {};
  IREE_ASSERT_OK(loom_artifact_provider_select_target(
      &loom_amdgpu_artifact_provider, IREE_SV("gfx1100"),
      iree_allocator_system(), &target));

  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(target.target_profile);
  ASSERT_NE(profile, nullptr);
  EXPECT_NE(loom_artifact_target_bundle(&target), nullptr);
  EXPECT_TRUE(iree_string_view_equal(target.target_key, IREE_SV("gfx1100")));
  ASSERT_NE(profile->identity.target, nullptr);
  EXPECT_TRUE(iree_string_view_equal(profile->identity.target->name,
                                     IREE_SV("gfx1100")));

  loom_amdgpu_artifact_provider.deinitialize_target(
      &loom_amdgpu_artifact_provider, &target, iree_allocator_system());
}

TEST_F(AmdgpuArtifactProviderTest, SelectTargetKeyPreservesAllFeatureStates) {
  struct TestCase {
    iree_string_view_t target_key;
    loom_amdgpu_amdhsa_feature_states_t expected_features;
  };
  const TestCase test_cases[] = {
      {
          IREE_SV("gfx942"),
          {
              LOOM_AMDGPU_TARGET_FEATURE_ANY,
              LOOM_AMDGPU_TARGET_FEATURE_ANY,
          },
      },
      {
          IREE_SV("gfx942:sramecc+:xnack-"),
          {
              LOOM_AMDGPU_TARGET_FEATURE_ON,
              LOOM_AMDGPU_TARGET_FEATURE_OFF,
          },
      },
      {
          IREE_SV("gfx1151"),
          {
              LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED,
              LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED,
          },
      },
  };
  for (const TestCase& test_case : test_cases) {
    loom_artifact_target_t target = {};
    IREE_ASSERT_OK(loom_artifact_provider_select_target(
        &loom_amdgpu_artifact_provider, test_case.target_key,
        iree_allocator_system(), &target));

    const loom_amdgpu_target_profile_t* profile =
        loom_amdgpu_target_profile_cast(target.target_profile);
    ASSERT_NE(profile, nullptr);
    EXPECT_EQ(profile->identity.amdhsa_features.sramecc,
              test_case.expected_features.sramecc);
    EXPECT_EQ(profile->identity.amdhsa_features.xnack,
              test_case.expected_features.xnack);
    EXPECT_TRUE(
        iree_string_view_equal(target.target_key, test_case.target_key));

    loom_amdgpu_artifact_provider.deinitialize_target(
        &loom_amdgpu_artifact_provider, &target, iree_allocator_system());
  }
}

TEST_F(AmdgpuArtifactProviderTest,
       SelectTargetKeyPreservesEveryCanonicalTarget) {
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  ASSERT_GT(target_count, 0u);
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target_info =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target_info, nullptr);
    const loom_amdgpu_processor_info_t* processor =
        loom_amdgpu_target_info_target_processor(target_info);
    ASSERT_NE(processor, nullptr);
    ASSERT_TRUE(
        loom_amdgpu_processor_properties_support_hsaco(&processor->properties));

    loom_amdgpu_target_identity_t identity = {};
    loom_amdgpu_target_identity_initialize(target_info, &identity);
    char target_key_storage[128] = {};
    iree_string_view_t target_key = iree_string_view_empty();
    IREE_ASSERT_OK(
        loom_amdgpu_artifact_key_format(&identity, sizeof(target_key_storage),
                                        target_key_storage, &target_key));
    loom_artifact_target_t target = {};
    IREE_ASSERT_OK(loom_artifact_provider_select_target(
        &loom_amdgpu_artifact_provider, target_key, iree_allocator_system(),
        &target));
    const loom_amdgpu_target_profile_t* target_profile =
        loom_amdgpu_target_profile_cast(target.target_profile);
    ASSERT_NE(target_profile, nullptr);
    EXPECT_NE(loom_artifact_target_bundle(&target), nullptr);
    EXPECT_TRUE(iree_string_view_equal(target.target_key, target_key));
    EXPECT_EQ(target_profile->identity.target, target_info);
    EXPECT_TRUE(loom_amdgpu_target_identity_equal(&target_profile->identity,
                                                  &identity));
    loom_amdgpu_artifact_provider.deinitialize_target(
        &loom_amdgpu_artifact_provider, &target, iree_allocator_system());
  }
}

TEST_F(AmdgpuArtifactProviderTest, RecordsDetailedReportRows) {
  ModulePtr module;
  IREE_ASSERT_OK(ParsePreparedArithmeticModule(IREE_SV("gfx1100"), &module));
  ASSERT_NE(module.get(), nullptr);

  const loom_amdgpu_target_info_t* target_info = LookupTarget("gfx1100");

  loom_target_compile_report_t report = {};
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;

  const loom_amdgpu_target_identity_t identity = {
      /*.target=*/target_info,
  };
  loom_amdgpu_target_profile_t target_profile = {};
  IREE_ASSERT_OK(
      loom_amdgpu_target_profile_initialize(&identity, &target_profile));

  loom_artifact_target_t target = {
      /*.target_profile=*/&target_profile.base,
      /*.target_key=*/target_info->name,
  };
  loom_compile_options_t options = {};
  loom_compile_options_initialize(&options);
  options.report = &report;
  options.artifact_flags = LOOM_COMPILE_ARTIFACT_FLAG_TARGET_LISTING;
  loom_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_artifact_provider.emit_artifact(
      &loom_amdgpu_artifact_provider, module.get(), &target, &options,
      iree_allocator_system(), &emitted, &artifact));
  EXPECT_TRUE(emitted);
  EXPECT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(artifact.target_artifact_data, artifact.executable_data);
  EXPECT_NE(artifact.target_artifact_data, nullptr);
  if (artifact.target_artifact_data != nullptr) {
    EXPECT_GT(iree_byte_sequence_length(artifact.target_artifact_data), 0u);
  }
  EXPECT_TRUE(iree_string_view_equal(artifact.target_listing_format,
                                     IREE_SV("amdgpu-assembly")));
  EXPECT_NE(artifact.target_listing_data, nullptr);
  if (artifact.target_listing_data != nullptr) {
    EXPECT_GT(iree_byte_sequence_length(artifact.target_listing_data), 0u);
  }

  EXPECT_EQ(report.source_low_rows.count, 0u);
  EXPECT_GT(report.pressure_rows.count, 0u);
  EXPECT_NE(report.pressure_rows.head, nullptr);

  loom_amdgpu_artifact_provider.deinitialize_artifact(
      &loom_amdgpu_artifact_provider, &artifact, iree_allocator_system());
  loom_target_compile_report_deinitialize(&report);
}

TEST_F(AmdgpuArtifactProviderTest,
       EmitsRuntimeGlobalsFromPipelineRequirements) {
  ModulePtr module;
  IREE_ASSERT_OK(ParsePreparedArithmeticModule(IREE_SV("gfx1100"), &module));
  ASSERT_NE(module.get(), nullptr);

  const loom_amdgpu_target_info_t* target_info = LookupTarget("gfx1100");
  const loom_amdgpu_target_identity_t identity = {
      /*.target=*/target_info,
  };
  loom_amdgpu_target_profile_t target_profile = {};
  IREE_ASSERT_OK(
      loom_amdgpu_target_profile_initialize(&identity, &target_profile));

  const loom_artifact_target_t target = {
      /*.target_profile=*/&target_profile.base,
      /*.target_key=*/target_info->name,
  };
  const loom_target_pipeline_options_t target_pipeline_options = {
      /*.source_to_low_max_errors=*/{},
      /*.source_to_low_legality_diagnostic_flags=*/{},
      /*.control_flow_lowering=*/{},
      /*.sanitizer=*/
      {
          /*.checks=*/LOOM_SANITIZER_CHECK_ACCESS | LOOM_SANITIZER_CHECK_RACE,
      },
  };
  loom_compile_options_t options = {};
  loom_compile_options_initialize(&options);
  options.target_pipeline_options = target_pipeline_options;
  loom_artifact_t artifact = {};
  bool emitted = false;
  IREE_ASSERT_OK(loom_amdgpu_artifact_provider.emit_artifact(
      &loom_amdgpu_artifact_provider, module.get(), &target, &options,
      iree_allocator_system(), &emitted, &artifact));

  EXPECT_TRUE(emitted);
  ASSERT_NE(artifact.target_artifact_data, nullptr);
  testing::ByteSequenceClone hsaco_contents(iree_allocator_system());
  IREE_ASSERT_OK(hsaco_contents.Clone(artifact.target_artifact_data));
  const iree_const_byte_span_t hsaco_bytes = hsaco_contents.contents();
  const iree_string_view_t hsaco = iree_make_string_view(
      (const char*)hsaco_bytes.data, hsaco_bytes.data_length);
  EXPECT_NE(iree_string_view_find(
                hsaco, LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_NAME, 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                hsaco, LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_NAME, 0),
            IREE_STRING_VIEW_NPOS);
  EXPECT_NE(iree_string_view_find(
                hsaco, LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME, 0),
            IREE_STRING_VIEW_NPOS);

  loom_amdgpu_artifact_provider.deinitialize_artifact(
      &loom_amdgpu_artifact_provider, &artifact, iree_allocator_system());
}

TEST_F(AmdgpuArtifactProviderTest, EmitsEveryAuthoredModuleTarget) {
  const iree_host_size_t target_count = loom_amdgpu_target_info_target_count();
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const loom_amdgpu_target_info_t* target_info =
        loom_amdgpu_target_info_target_at(i);
    ASSERT_NE(target_info, nullptr) << "target index " << i;
    ExpectEmitsModuleTarget(target_info->name);
  }
}

}  // namespace
}  // namespace loom
