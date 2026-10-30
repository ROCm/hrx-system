// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/timestamp.h"

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/vulkan/allocator.h"
#include "iree/hal/drivers/vulkan/barrier.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/device/library.h"

void iree_hal_vulkan_timestamp_pipeline_deinitialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline) {
  if (pipeline->handle) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(syms), logical_device,
                           pipeline->handle, NULL);
  }
  if (pipeline->layout) {
    iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(syms), logical_device,
                                 pipeline->layout, NULL);
  }
  *pipeline = (iree_hal_vulkan_timestamp_pipeline_t){0};
}

static iree_status_t iree_hal_vulkan_timestamp_pipeline_prepare(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline) {
  if (pipeline->handle) {
    return iree_ok_status();
  }
  const VkPushConstantRange push_constant_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(uint64_t[2]),
  };
  const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_constant_range,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status =
      iree_vkCreatePipelineLayout(IREE_VULKAN_DEVICE(syms), logical_device,
                                  &layout_info, NULL, &pipeline->layout);
  const iree_const_byte_span_t code =
      iree_hal_vulkan_device_library_lookup(IREE_SV("timestamp_resolve.spv"));
  const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.data_length,
      .pCode = (const uint32_t*)code.data,
  };
  VkShaderModule module = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    status = iree_vkCreateShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                                       &module_info, NULL, &module);
  }
  if (iree_status_is_ok(status)) {
    const VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
        .layout = pipeline->layout,
    };
    status = iree_vkCreateComputePipelines(
        IREE_VULKAN_DEVICE(syms), logical_device, VK_NULL_HANDLE, 1,
        &pipeline_info, NULL, &pipeline->handle);
  }
  IREE_LEAK_CHECK_DISABLE_POP();
  if (module) {
    iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(syms), logical_device, module,
                               NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_timestamp_pipeline_deinitialize(syms, logical_device,
                                                    pipeline);
  }
  return status;
}

void iree_hal_vulkan_timestamp_state_deinitialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_timestamp_state_t* state) {
  if (state->queries) {
    iree_vkDestroyQueryPool(IREE_VULKAN_DEVICE(syms), logical_device,
                            state->queries, NULL);
  }
  iree_hal_buffer_release(state->resolve_buffer);
  *state = (iree_hal_vulkan_timestamp_state_t){0};
}

iree_status_t iree_hal_vulkan_timestamp_state_prepare(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint32_t query_count, uint32_t resolve_count,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline,
    iree_hal_vulkan_timestamp_state_t* state) {
  if (query_count > state->query_capacity) {
    const VkQueryPoolCreateInfo query_info = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = query_count,
    };
    VkQueryPool queries = VK_NULL_HANDLE;
    IREE_RETURN_IF_ERROR(iree_vkCreateQueryPool(
        IREE_VULKAN_DEVICE(syms), logical_device, &query_info, NULL, &queries));
    if (state->queries) {
      iree_vkDestroyQueryPool(IREE_VULKAN_DEVICE(syms), logical_device,
                              state->queries, NULL);
    }
    state->queries = queries;
    state->query_capacity = query_count;
  }
  if (resolve_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_timestamp_pipeline_prepare(
      syms, logical_device, pipeline));
  state->pipeline = pipeline;
  if (resolve_count <= state->resolve_capacity) {
    return iree_ok_status();
  }

  const iree_hal_vulkan_buffer_spec_t spec = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_STORAGE_READ,
      .queue_family_affinity = queue_family_affinity,
  };
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_allocate_direct_buffer(
      device_allocator, &spec,
      (iree_device_size_t)resolve_count * sizeof(uint64_t), &buffer));
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceAddress address = 0;
  iree_status_t status =
      iree_hal_vulkan_buffer_handle(buffer, &memory, &handle);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_buffer_device_address(buffer, &address);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_release(state->resolve_buffer);
    state->resolve_buffer = buffer;
    state->resolve_handle = handle;
    state->resolve_offset =
        iree_hal_vulkan_buffer_native(buffer)
            ->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
            .vulkan.offset;
    state->resolve_address = address;
    state->resolve_capacity = resolve_count;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static void iree_hal_vulkan_timestamp_record_sample(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask, VkBuffer target_buffer,
    VkDeviceSize target_offset) {
  // Vulkan accepts one stage. Compute and atomic work share a stage; composed
  // transfer scopes include compute polyfills and therefore join all commands.
  VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  if (!(source_stage_mask & ~IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE)) {
    stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
  } else if (!(source_stage_mask & ~(IREE_HAL_EXECUTION_STAGE_DISPATCH |
                                     IREE_HAL_EXECUTION_STAGE_ATOMIC |
                                     IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE))) {
    stage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  }
  iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(syms), command_buffer, stage,
                            queries, query_index);
  // WAIT supplies the query execution dependency entirely on the device.
  iree_vkCmdCopyQueryPoolResults(
      IREE_VULKAN_DEVICE(syms), command_buffer, queries, query_index, 1,
      target_buffer, target_offset, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

static void iree_hal_vulkan_timestamp_record_dependency(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkPipelineStageFlags2 source_stage,
    iree_hal_execution_stage_t target_stage_mask) {
  if (!target_stage_mask) {
    return;
  }
  const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = source_stage,
      .dstStageMask =
          iree_hal_vulkan_barrier_stage_mask(queue_flags, target_stage_mask, 0),
  };
  const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency);
}

void iree_hal_vulkan_timestamp_record_direct(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, VkBuffer target_buffer,
    VkDeviceSize target_offset) {
  iree_hal_vulkan_timestamp_record_sample(syms, command_buffer, queries,
                                          query_index, source_stage_mask,
                                          target_buffer, target_offset);
  iree_hal_vulkan_timestamp_record_dependency(syms, queue_flags, command_buffer,
                                              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                              target_stage_mask);
}

void iree_hal_vulkan_timestamp_record_indirect(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer,
    const iree_hal_vulkan_timestamp_state_t* state, uint32_t query_index,
    uint32_t resolve_index, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    VkDeviceAddress target_slot_address) {
  const VkDeviceSize offset = (VkDeviceSize)resolve_index * sizeof(uint64_t);
  iree_hal_vulkan_timestamp_record_sample(
      syms, command_buffer, state->queries, query_index, source_stage_mask,
      state->resolve_handle, state->resolve_offset + offset);
  const VkBufferMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = state->resolve_handle,
      .offset = state->resolve_offset + offset,
      .size = sizeof(uint64_t),
  };
  const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency);
  const uint64_t constants[] = {state->resolve_address + offset,
                                target_slot_address};
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         state->pipeline->handle);
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(syms), command_buffer,
                          state->pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(constants), constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(syms), command_buffer, 1, 1, 1);
  iree_hal_vulkan_timestamp_record_dependency(
      syms, queue_flags, command_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      target_stage_mask);
}
