// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "build_tools/vulkan/testing/compute.h"

namespace iree::vulkan::testing {

Compute::~Compute() {
  if (pipeline_) {
    device_->Get<PFN_vkDestroyPipeline>("vkDestroyPipeline")(
        device_->handle(), pipeline_, nullptr);
  }
  if (layout_) {
    device_->Get<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(
        device_->handle(), layout_, nullptr);
  }
}

::testing::AssertionResult Compute::Initialize(
    const Device& device, size_t code_size, const uint32_t* code,
    const char* entry_point, uint32_t push_constant_range_count,
    const VkPushConstantRange* push_constant_ranges,
    const VkSpecializationInfo* specialization) {
  device_ = &device;
  VkPipelineLayoutCreateInfo layout_info = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout_info.pushConstantRangeCount = push_constant_range_count;
  layout_info.pPushConstantRanges = push_constant_ranges;
  auto result = CheckResult(
      device.Get<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(
          device.handle(), &layout_info, nullptr, &layout_),
      "vkCreatePipelineLayout");
  if (!result) {
    return result;
  }
  VkShaderModuleCreateInfo shader_info = {
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  shader_info.codeSize = code_size;
  shader_info.pCode = code;
  VkShaderModule shader = VK_NULL_HANDLE;
  result =
      CheckResult(device.Get<PFN_vkCreateShaderModule>("vkCreateShaderModule")(
                      device.handle(), &shader_info, nullptr, &shader),
                  "vkCreateShaderModule");
  if (!result) {
    return result;
  }
  VkComputePipelineCreateInfo pipeline_info = {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeline_info.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = shader;
  pipeline_info.stage.pName = entry_point;
  pipeline_info.stage.pSpecializationInfo = specialization;
  pipeline_info.layout = layout_;
  result = CheckResult(
      device.Get<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
          device.handle(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
          &pipeline_),
      "vkCreateComputePipelines");
  device.Get<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(
      device.handle(), shader, nullptr);
  return result;
}

}  // namespace iree::vulkan::testing
