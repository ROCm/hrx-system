// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "test/target/iree_hal_execution.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"

namespace {

constexpr char kSourceText[] = R"(
kernel.def @double_i32_at_byte_offset() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%input: buffer, %output: buffer, %byte_offset: offset) {
  %byte_offset_aligned = index.assume %byte_offset [mul(%byte_offset, 4)] : offset
  %input_aligned = buffer.assume.alignment %input {minimum_alignment = 4} : buffer
  %output_aligned = buffer.assume.alignment %output {minimum_alignment = 4} : buffer
  %input_view = buffer.view %input_aligned[%byte_offset_aligned] : buffer -> view<1xi32>
  %loaded = view.load %input_view[0] : view<1xi32> -> i32
  %doubled = scalar.addi %loaded, %loaded : i32
  %output_view = buffer.view %output_aligned[%byte_offset_aligned] : buffer -> view<1xi32>
  view.store %doubled, %output_view[0] : i32, view<1xi32>
  kernel.return
}
)";

loomc_status_t CreateAmdgpuTargetEnvironment(
    loomc_allocator_t host_allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_amdgpu(host_allocator,
                                                out_target_environment);
}

loomc_status_t ValidateAmdgpuProfile(loomc_target_profile_t* target_profile,
                                     const char** out_skip_reason) {
  *out_skip_reason = nullptr;
  loomc_amdgpu_target_identity_t identity = {};
  return loomc_amdgpu_target_profile_query_identity(target_profile, &identity);
}

TEST(LoomcAmdgpuIreeHalExecutionTest,
     SpecializesEmitsAndExecutesOnLiveAmdgpuHalDevice) {
  const loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/nullptr,
      /*.runtime_globals=*/LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
  };
  const loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/&amdgpu_options,
      /*.artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      /*.identifier=*/
      loomc_make_cstring_view("double_i32_at_byte_offset.hsaco"),
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_iree_hal_profile_provider_t* profile_providers[] = {
      loomc_amdgpu_iree_hal_profile_provider(),
  };

  loomc::testing::target::IreeHalKernelExecutionTarget target = {};
  target.label = "AMDGPU";
  target.device_uri = IREE_SV("amdgpu");
  target.target_profile_identifier = loomc_make_cstring_view("live-amdgpu");
  target.source_identifier =
      loomc_make_cstring_view("double_i32_at_byte_offset.loom");
  target.source_text = loomc_make_cstring_view(kSourceText);
  target.module_name = loomc_make_cstring_view("live_amdgpu_execution_test");
  target.kernel_export_name =
      loomc_make_cstring_view("double_i32_at_byte_offset");
  target.target_pipeline_identifier =
      loomc_make_cstring_view("live-amdgpu-prepared-low");
  target.target_pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
  target.control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
  target.source_to_low_max_errors = 20;
  target.emit_options = &emit_options;
  target.executable_target_selection = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  target.profile_providers = profile_providers;
  target.profile_provider_count = 1;
  target.create_target_environment = CreateAmdgpuTargetEnvironment;
  target.validate_target_profile = ValidateAmdgpuProfile;

  loomc::testing::target::RunIreeHalKernelExecutionTest(target);
}

}  // namespace
