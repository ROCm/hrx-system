// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/barrier.h"

iree_hal_vulkan_barrier_t iree_hal_vulkan_barrier_resolve(
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_barrier_flags_t flags, bool has_memory_visibility) {
  const bool acquire_system_scope =
      iree_any_bit_set(flags, IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE) ||
      iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST);
  const bool release_system_scope =
      iree_any_bit_set(flags, IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST);
  iree_hal_vulkan_barrier_t barrier = {
      .source_stage_mask = source_stage_mask,
      .source_access_mask =
          has_memory_visibility ? VK_ACCESS_2_MEMORY_WRITE_BIT : 0,
      .target_stage_mask = target_stage_mask,
      .target_access_mask =
          has_memory_visibility
              ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
              : 0,
  };
  // A global cache action still covers local commands when the caller adds no
  // separate stage dependency. Preserve that local domain before adding HOST
  // stages for the remote side of the visibility handoff.
  if (release_system_scope && !barrier.source_stage_mask) {
    barrier.source_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS;
  }
  if (acquire_system_scope && !barrier.target_stage_mask) {
    barrier.target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS;
  }
  if (acquire_system_scope) {
    barrier.source_stage_mask |= IREE_HAL_EXECUTION_STAGE_HOST;
    barrier.source_access_mask |= VK_ACCESS_2_HOST_WRITE_BIT;
  }
  if (release_system_scope) {
    barrier.target_stage_mask |= IREE_HAL_EXECUTION_STAGE_HOST;
    barrier.target_access_mask |=
        VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
  }
  return barrier;
}

static iree_hal_vulkan_barrier_t iree_hal_vulkan_barrier_list_resolve(
    const iree_hal_barrier_list_t* barriers) {
  if (!barriers) {
    return iree_hal_vulkan_barrier_resolve(
        IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
        IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS,
        IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
            IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
        /*has_memory_visibility=*/true);
  }
  iree_hal_vulkan_barrier_t result = {0};
  for (iree_host_size_t i = 0; i < barriers->count; ++i) {
    const iree_hal_barrier_t* barrier = &barriers->values[i];
    iree_hal_barrier_flags_t flags = iree_hal_barrier_resolve_flags(barrier);
    for (iree_host_size_t j = 0; j < barrier->buffer_barrier_count; ++j) {
      const iree_hal_memory_transition_recipe_t* recipe =
          barrier->buffer_barriers[j].recipe;
      if (!recipe) {
        continue;
      }
      for (uint32_t k = 0; k < recipe->operation_count; ++k) {
        if (recipe->operations[k].operation ==
            IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM) {
          flags |= IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
        } else {
          flags |= IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE;
        }
      }
    }
    iree_hal_vulkan_barrier_t value = iree_hal_vulkan_barrier_resolve(
        barrier->source_stage_mask, barrier->target_stage_mask, flags,
        flags || barrier->memory_barrier_count ||
            barrier->buffer_barrier_count);
    for (iree_host_size_t j = 0; j < barrier->buffer_barrier_count; ++j) {
      value.source_access_mask |= iree_hal_vulkan_access_scope_mask(
          barrier->buffer_barriers[j].source_scope);
      value.target_access_mask |= iree_hal_vulkan_access_scope_mask(
          barrier->buffer_barriers[j].target_scope);
    }
    result.source_stage_mask |= value.source_stage_mask;
    result.target_stage_mask |= value.target_stage_mask;
    result.source_access_mask |= value.source_access_mask;
    result.target_access_mask |= value.target_access_mask;
  }
  return result;
}

iree_hal_vulkan_queue_barriers_t iree_hal_vulkan_queue_barriers_resolve(
    const iree_hal_queue_barriers_t* barriers) {
  return (iree_hal_vulkan_queue_barriers_t){
      .before = iree_hal_vulkan_barrier_list_resolve(barriers ? barriers->before
                                                              : NULL),
      .after = iree_hal_vulkan_barrier_list_resolve(barriers ? barriers->after
                                                             : NULL),
  };
}

static VkPipelineStageFlags2
iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
    VkQueueFlags queue_flags, iree_hal_execution_stage_t stage_mask,
    VkAccessFlags2 access_mask) {
  VkPipelineStageFlags2 pipeline_stage_mask = 0;
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_ATOMIC)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    if (iree_any_bit_set(queue_flags, VK_QUEUE_COMPUTE_BIT)) {
      // Unaligned fills and updates execute compute patches in the logical
      // transfer domain. Their accesses participate in the same dependency.
      pipeline_stage_mask |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_HOST_BIT;
  }
  if (!pipeline_stage_mask && access_mask) {
    return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE)) {
    pipeline_stage_mask |= VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  }
  return pipeline_stage_mask;
}

VkAccessFlags2 iree_hal_vulkan_barrier_source_access_mask(
    iree_hal_execution_stage_t stage_mask) {
  VkAccessFlags2 access_mask = 0;
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)) {
    access_mask |= VK_ACCESS_2_MEMORY_WRITE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH |
                                       IREE_HAL_EXECUTION_STAGE_ATOMIC)) {
    access_mask |= VK_ACCESS_2_SHADER_WRITE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    access_mask |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    access_mask |= VK_ACCESS_2_HOST_WRITE_BIT;
  }
  return access_mask;
}

VkAccessFlags2 iree_hal_vulkan_barrier_target_access_mask(
    iree_hal_execution_stage_t stage_mask) {
  VkAccessFlags2 access_mask = 0;
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)) {
    access_mask |= VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH |
                                       IREE_HAL_EXECUTION_STAGE_ATOMIC)) {
    access_mask |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    access_mask |=
        VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  }
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    access_mask |= VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
  }
  return access_mask;
}

VkAccessFlags2 iree_hal_vulkan_access_scope_mask(
    iree_hal_access_scope_t access_scope) {
  VkAccessFlags2 access_mask = 0;
  if (iree_any_bit_set(access_scope,
                       IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ)) {
    access_mask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_CONSTANT_READ)) {
    access_mask |= VK_ACCESS_2_UNIFORM_READ_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                                         IREE_HAL_ACCESS_SCOPE_ATOMIC_READ)) {
    access_mask |= VK_ACCESS_2_SHADER_READ_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                                         IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE)) {
    access_mask |= VK_ACCESS_2_SHADER_WRITE_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_TRANSFER_READ)) {
    access_mask |= VK_ACCESS_2_TRANSFER_READ_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE)) {
    access_mask |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_HOST_READ)) {
    access_mask |= VK_ACCESS_2_HOST_READ_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_HOST_WRITE)) {
    access_mask |= VK_ACCESS_2_HOST_WRITE_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_MEMORY_READ)) {
    access_mask |= VK_ACCESS_2_MEMORY_READ_BIT;
  }
  if (iree_any_bit_set(access_scope, IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE)) {
    access_mask |= VK_ACCESS_2_MEMORY_WRITE_BIT;
  }
  return access_mask;
}

static VkAccessFlags2 iree_hal_vulkan_transfer_access_mask(
    VkQueueFlags queue_flags, VkAccessFlags2 access_mask, bool is_target) {
  if (iree_any_bit_set(queue_flags, VK_QUEUE_COMPUTE_BIT)) {
    if (iree_any_bit_set(access_mask, VK_ACCESS_2_TRANSFER_READ_BIT)) {
      access_mask |= VK_ACCESS_2_SHADER_READ_BIT;
    }
    if (iree_any_bit_set(access_mask, VK_ACCESS_2_TRANSFER_WRITE_BIT)) {
      access_mask |= VK_ACCESS_2_SHADER_WRITE_BIT;
      if (is_target) {
        // Partial-word patches preserve neighboring bytes with atomic RMWs.
        access_mask |= VK_ACCESS_2_SHADER_READ_BIT;
      }
    }
  }
  return access_mask;
}

void iree_hal_vulkan_barrier_record(const iree_hal_vulkan_device_syms_t* syms,
                                    VkQueueFlags queue_flags,
                                    VkCommandBuffer command_buffer,
                                    const iree_hal_vulkan_barrier_t* barrier) {
  if (iree_hal_vulkan_barrier_is_empty(barrier)) {
    return;
  }
  VkMemoryBarrier2 memory_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask =
          iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
              queue_flags, barrier->source_stage_mask,
              barrier->source_access_mask),
      .srcAccessMask = iree_hal_vulkan_transfer_access_mask(
          queue_flags, barrier->source_access_mask, /*is_target=*/false),
      .dstStageMask =
          iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
              queue_flags, barrier->target_stage_mask,
              barrier->target_access_mask),
      .dstAccessMask = iree_hal_vulkan_transfer_access_mask(
          queue_flags, barrier->target_access_mask, /*is_target=*/true),
  };
  VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memory_barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency_info);
}

void iree_hal_vulkan_buffer_barrier_record(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, const iree_hal_vulkan_barrier_t* barrier,
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize length) {
  if (iree_hal_vulkan_barrier_is_empty(barrier)) {
    return;
  }
  VkBufferMemoryBarrier2 buffer_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask =
          iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
              queue_flags, barrier->source_stage_mask,
              barrier->source_access_mask),
      .srcAccessMask = iree_hal_vulkan_transfer_access_mask(
          queue_flags, barrier->source_access_mask, /*is_target=*/false),
      .dstStageMask =
          iree_hal_vulkan_pipeline_stage_mask_from_hal_execution_stage(
              queue_flags, barrier->target_stage_mask,
              barrier->target_access_mask),
      .dstAccessMask = iree_hal_vulkan_transfer_access_mask(
          queue_flags, barrier->target_access_mask, /*is_target=*/true),
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = offset,
      .size = length,
  };
  VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &buffer_barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency_info);
}
