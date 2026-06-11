// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test/target/iree_hal_execution.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "loomc/target/spirv.h"
#include "loomc/target/spirv/iree_hal.h"

namespace {

constexpr char kSourceText[] = R"(
spirv.target<vulkan1_3> @target {abi = hal_kernel}

kernel.def target(@target) @double_i32_at_byte_offset() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%input: buffer, %output: buffer, %byte_offset: offset) {
  %byte_offset_aligned = index.assume %byte_offset [mul(%byte_offset, 4)] : offset
  %input_aligned = buffer.assume.alignment %input {minimum_alignment = 4} : buffer
  %output_aligned = buffer.assume.alignment %output {minimum_alignment = 4} : buffer
  %input_view = buffer.view %input_aligned[%byte_offset_aligned] : buffer -> view<1xi32, #dense>
  %loaded = view.load %input_view[0] : view<1xi32, #dense> -> i32
  %doubled = scalar.addi %loaded, %loaded : i32
  %output_view = buffer.view %output_aligned[%byte_offset_aligned] : buffer -> view<1xi32, #dense>
  view.store %doubled, %output_view[0] : i32, view<1xi32, #dense>
  kernel.return
}
)";

loomc_status_t CreateSpirvTargetEnvironment(
    loomc_allocator_t host_allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_spirv(host_allocator,
                                               out_target_environment);
}

loomc_status_t QueryProfileFeature(loomc_target_profile_t* target_profile,
                                   loomc_spirv_feature_t feature,
                                   bool* out_present) {
  loomc_target_fact_state_t state = LOOMC_TARGET_FACT_STATE_UNKNOWN;
  loomc_status_t status =
      loomc_spirv_target_profile_query_feature(target_profile, feature, &state);
  *out_present = state == LOOMC_TARGET_FACT_STATE_TRUE;
  return status;
}

loomc_status_t ValidateSpirvVulkanProfile(
    loomc_target_profile_t* target_profile, const char** out_skip_reason) {
  *out_skip_reason = nullptr;

  bool feature_present = false;
  loomc_status_t status = QueryProfileFeature(
      target_profile, LOOMC_SPIRV_FEATURE_VULKAN_SHADER, &feature_present);
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  if (!feature_present) {
    *out_skip_reason =
        "live Vulkan HAL device cannot run Vulkan SPIR-V shaders";
    return loomc_ok_status();
  }

  status = QueryProfileFeature(target_profile,
                               LOOMC_SPIRV_FEATURE_PHYSICAL_STORAGE_BUFFER,
                               &feature_present);
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  if (!feature_present) {
    *out_skip_reason =
        "live Vulkan HAL device does not expose buffer device address";
  }
  return loomc_ok_status();
}

loomc_status_t EmitSpirvModule(loomc_target_environment_t* target_environment,
                               loomc_workspace_t* workspace,
                               loomc_module_t* module,
                               loomc_target_selection_t* target_selection,
                               loomc_string_view_t artifact_format,
                               loomc_string_view_t artifact_identifier,
                               loomc_result_t** out_result) {
  loomc_target_selection_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_selection=*/target_selection,
  };
  loomc_spirv_emit_options_t spirv_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(spirv_options),
      /*.next=*/&target_options,
  };
  loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/&spirv_options,
      /*.artifact_format=*/artifact_format,
      /*.identifier=*/artifact_identifier,
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  return loomc_emit_module(target_environment, workspace, module, &emit_options,
                           loomc_allocator_system(), out_result);
}

TEST(LoomcSpirvIreeHalExecutionTest,
     CompilesEmitsAndExecutesOnLiveVulkanHalDevice) {
  const loomc_iree_hal_profile_provider_t* profile_providers[] = {
      loomc_spirv_iree_hal_profile_provider(),
  };

  loomc::testing::target::IreeHalKernelExecutionTarget target = {};
  target.label = "SPIR-V Vulkan";
  target.device_uri = IREE_SV("vulkan");
  target.executable_cache_identifier = IREE_SV("loomc-spirv-execution-test");
  target.target_profile_identifier = loomc_make_cstring_view("live-vulkan");
  target.source_identifier =
      loomc_make_cstring_view("double_i32_at_byte_offset.loom");
  target.source_text = loomc_make_cstring_view(kSourceText);
  target.module_name = loomc_make_cstring_view("live_vulkan_execution_test");
  target.kernel_function_symbol =
      loomc_make_cstring_view("@double_i32_at_byte_offset");
  target.target_pipeline_identifier =
      loomc_make_cstring_view("live-vulkan-prepared-low");
  target.target_pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
  target.control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
  target.source_to_low_max_errors = 20;
  target.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV);
  target.artifact_identifier =
      loomc_make_cstring_view("double_i32_at_byte_offset.spv");
  target.executable_format = IREE_SV("vulkan-spirv-bda");
  target.profile_providers = profile_providers;
  target.profile_provider_count = 1;
  target.create_target_environment = CreateSpirvTargetEnvironment;
  target.validate_target_profile = ValidateSpirvVulkanProfile;
  target.emit_module = EmitSpirvModule;

  loomc::testing::target::RunIreeHalKernelExecutionTest(target);
}

}  // namespace
