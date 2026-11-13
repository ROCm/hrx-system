// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/loomc.h"
#include "loomc/target/spirv.h"
#include "loomc/target/spirv/iree_hal.h"
#include "test/target/iree_hal_execution_test_util.h"
#include "test/util.h"

namespace {

constexpr char kSourceText[] = R"(
spirv.target<vulkan1_3> @target {abi = hal_kernel}

kernel.def target(@target) @double_i32_at_byte_offset() {
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
} launch(%input: buffer, %output: buffer, %byte_offset: offset) {
  %byte_offset_aligned = index.assume %byte_offset [multiple_of(%byte_offset, 4)] : offset
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

loomc_status_t ValidateSelectableSubgroups(loomc_target_profile_t* profile,
                                           const char** out_skip_reason) {
  LOOMC_RETURN_IF_ERROR(ValidateSpirvVulkanProfile(profile, out_skip_reason));
  if (*out_skip_reason != nullptr) {
    return loomc_ok_status();
  }
  loomc_spirv_limit_value_t minimum = {}, maximum = {};
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_limit(
      profile, LOOMC_SPIRV_LIMIT_MIN_SUBGROUP_SIZE, &minimum));
  LOOMC_RETURN_IF_ERROR(loomc_spirv_target_profile_query_limit(
      profile, LOOMC_SPIRV_LIMIT_MAX_SUBGROUP_SIZE, &maximum));
  if (minimum.state != LOOMC_TARGET_FACT_STATE_TRUE ||
      maximum.state != LOOMC_TARGET_FACT_STATE_TRUE || minimum.value > 32 ||
      maximum.value < 64) {
    *out_skip_reason = "device does not support both 32- and 64-lane subgroups";
  }
  return loomc_ok_status();
}

void RunSubgroupExecution(
    const loomc::testing::target::IreeHalKernelExecution& execution,
    uint32_t width) {
  EXPECT_EQ(execution.launch_config.workgroup_size.x, 2 * width);
  std::array<uint32_t, 128> input, output;
  for (uint32_t i = 0; i < input.size(); ++i) {
    input[i] = 7 + 3 * i;
  }
  output.fill(UINT32_MAX);
  using BufferPtr =
      loomc::testing::HandlePtr<iree_hal_buffer_t, iree_hal_buffer_release>;
  iree_hal_buffer_t* input_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      execution.pool, iree_hal_make_pool_reservation_request(sizeof(input), 1),
      iree_infinite_timeout(), &input_buffer));
  BufferPtr owned_input(input_buffer);
  iree_hal_buffer_t* output_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      execution.pool, iree_hal_make_pool_reservation_request(sizeof(output), 1),
      iree_infinite_timeout(), &output_buffer));
  BufferPtr owned_output(output_buffer);
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      execution.device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  loomc::testing::HandlePtr<iree_hal_semaphore_t, iree_hal_semaphore_release>
      owned_semaphore(semaphore);
  uint64_t upload_value = 1, dispatch_value = 2, download_value = 3;
  const iree_hal_semaphore_list_t uploaded = {1, &semaphore, &upload_value};
  const iree_hal_semaphore_list_t dispatched = {1, &semaphore, &dispatch_value};
  const iree_hal_semaphore_list_t downloaded = {1, &semaphore, &download_value};
  iree_hal_transfer_operation_t uploads[2] = {};
  uploads[0].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  uploads[0].upload.source = input.data();
  uploads[0].upload.target_buffer = input_buffer;
  uploads[0].upload.length = sizeof(input);
  uploads[1].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  uploads[1].upload.source = output.data();
  uploads[1].upload.target_buffer = output_buffer;
  uploads[1].upload.length = sizeof(output);
  uint64_t completion_value = 0;
  iree_status_t status = iree_hal_queue_transfer(
      execution.transfer_queue, iree_hal_semaphore_list_empty(), uploaded,
      IREE_ARRAYSIZE(uploads), uploads, nullptr);
  if (iree_status_is_ok(status)) {
    completion_value = upload_value;
    status = loomc::testing::target::DispatchIreeHalKernel(
        execution, iree_const_byte_span_empty(), input_buffer, output_buffer,
        uploaded, dispatched);
  }
  if (iree_status_is_ok(status)) {
    completion_value = dispatch_value;
    status = iree_hal_queue_download(execution.transfer_queue, dispatched,
                                     downloaded, output_buffer, 0,
                                     output.data(), sizeof(output), nullptr);
  }
  if (iree_status_is_ok(status)) {
    completion_value = download_value;
  }
  if (completion_value != 0) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(semaphore, completion_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
  }
  IREE_ASSERT_OK(status);
  for (uint32_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], i < 2 * width ? input[i] + i % width : UINT32_MAX)
        << "element " << i;
  }
}

using LoomcSpirvIreeHalExecutionTest =
    loomc::testing::target::IreeHalExecutionTest;

TEST_F(LoomcSpirvIreeHalExecutionTest,
       SelectsAndExecutesRequiredSubgroupWidths) {
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_spirv_iree_hal_target_provider(),
  };
  for (uint32_t width : {32u, 64u}) {
    SCOPED_TRACE(width);
    const std::string source =
        "spirv.target<vulkan1_3> @target {abi = hal_kernel, subgroup_size = " +
        std::to_string(width) + R"(}

template.decl @record_lanes(%input: buffer, %output: buffer)
template.def<@record_lanes> @record(%input: buffer, %output: buffer) {
  %thread = kernel.workitem.id<x> : index
  %lane = kernel.subgroup.lane.id : index
  %lane_word = index.cast %lane : index to i32
  %base = index.constant 0 : offset
  %input_view = buffer.view %input[%base] : buffer -> view<128xi32>
  %output_view = buffer.view %output[%base] : buffer -> view<128xi32>
  %value = view.load %input_view[%thread] : view<128xi32> -> i32
  %result = scalar.addi %value, %lane_word : i32
  view.store %result, %output_view[%thread] : i32, view<128xi32>
  template.return
}

kernel.def target(@target) @record_subgroups() {
  %unit = index.constant 1 : index
  %subgroups = index.constant 2 : index
  %width = target.subgroup.size : index
  %threads = index.mul %width, %subgroups : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
} launch(%input: buffer, %output: buffer) {
  template.apply<@record_lanes>(%input, %output) : (buffer, buffer)
  kernel.return
}
)";
    const loomc::testing::target::IreeHalKernelExecutionTarget target = {
        .label = "SPIR-V subgroup selection",
        .device_uri = IREE_SV("vulkan"),
        .target_profile_identifier = loomc_make_cstring_view("live-subgroups"),
        .source_identifier = loomc_make_cstring_view("subgroups.loom"),
        .source_text = loomc_make_string_view(source.data(), source.size()),
        .kernel_export_name = loomc_make_cstring_view("record_subgroups"),
        .target_pipeline_identifier = loomc_make_cstring_view("subgroup-low"),
        .target_pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        .control_flow_lowering =
            LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
        .source_to_low_max_errors = 20,
        .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
        .artifact_identifier = loomc_make_cstring_view("subgroups.spv"),
        .target_providers = providers,
        .target_provider_count = 1,
        .create_target_environment = CreateSpirvTargetEnvironment,
        .validate_target_profile = ValidateSelectableSubgroups,
    };
    RunKernel(target, [width](const auto& execution) {
      RunSubgroupExecution(execution, width);
    });
  }
}

TEST_F(LoomcSpirvIreeHalExecutionTest,
       CompilesEmitsAndExecutesOnLiveVulkanHalDevice) {
  const loomc_iree_hal_target_provider_t* target_providers[] = {
      loomc_spirv_iree_hal_target_provider(),
  };

  loomc::testing::target::IreeHalKernelExecutionTarget target = {
      .label = "SPIR-V Vulkan",
      .device_uri = IREE_SV("vulkan"),
      .target_profile_identifier = loomc_make_cstring_view("live-vulkan"),
      .source_identifier =
          loomc_make_cstring_view("double_i32_at_byte_offset.loom"),
      .source_text = loomc_make_cstring_view(kSourceText),
      .kernel_export_name =
          loomc_make_cstring_view("double_i32_at_byte_offset"),
      .target_pipeline_identifier =
          loomc_make_cstring_view("live-vulkan-prepared-low"),
      .target_pipeline_kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      .source_to_low_max_errors = 20,
      .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      .artifact_identifier =
          loomc_make_cstring_view("double_i32_at_byte_offset.spv"),
      .target_providers = target_providers,
      .target_provider_count = 1,
      .create_target_environment = CreateSpirvTargetEnvironment,
      .validate_target_profile = ValidateSpirvVulkanProfile,
  };

  RunKernel(target, loomc::testing::target::RunIreeHalByteOffsetExecution);
}

}  // namespace
