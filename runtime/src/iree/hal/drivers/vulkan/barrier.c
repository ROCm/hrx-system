// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/barrier.h"

static VkPipelineStageFlags2
iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
    iree_hal_execution_stage_t stage_mask, VkAccessFlags2 access_mask) {
  VkPipelineStageFlags2 pipeline_stage_mask = 0;
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_HOST_BIT;
  }
  if (pipeline_stage_mask) return pipeline_stage_mask;

  if (access_mask) return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE)) {
    return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE)) {
    return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  }
  return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
}

void iree_hal_vulkan_barrier_record(const iree_hal_vulkan_device_syms_t* syms,
                                    VkCommandBuffer command_buffer,
                                    const iree_hal_vulkan_barrier_t* barrier) {
  VkMemoryBarrier2 memory_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask =
          iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
              barrier->source_stage_mask, barrier->source_access_mask),
      .srcAccessMask = barrier->source_access_mask,
      .dstStageMask =
          iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
              barrier->target_stage_mask, barrier->target_access_mask),
      .dstAccessMask = barrier->target_access_mask,
  };
  VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memory_barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency_info);
}
