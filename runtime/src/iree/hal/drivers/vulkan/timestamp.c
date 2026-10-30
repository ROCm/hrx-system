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
      .size = sizeof(uint64_t[3]),
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
    iree_allocator_t host_allocator, iree_hal_vulkan_timestamp_state_t* state) {
  if (state->queries) {
    iree_vkDestroyQueryPool(IREE_VULKAN_DEVICE(syms), logical_device,
                            state->queries, NULL);
  }
  iree_hal_buffer_release(state->resolve_buffer);
  iree_allocator_free(host_allocator, state->pending);
  *state = (iree_hal_vulkan_timestamp_state_t){0};
}

iree_status_t iree_hal_vulkan_timestamp_state_prepare(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    uint32_t query_count, uint32_t resolve_count,
    iree_hal_vulkan_timestamp_pipeline_t* pipeline,
    iree_allocator_t host_allocator, iree_hal_vulkan_timestamp_state_t* state) {
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
  if (query_count > state->pending_capacity) {
    iree_host_size_t byte_length = 0;
    if (!iree_host_size_checked_mul(query_count, sizeof(*state->pending),
                                    &byte_length)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "timestamp recording workspace size overflows");
    }
    iree_hal_vulkan_timestamp_pending_t* pending = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        host_allocator, byte_length, (void**)&pending));
    iree_allocator_free(host_allocator, state->pending);
    state->pending = pending;
    state->pending_capacity = query_count;
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

static VkPipelineStageFlags2 iree_hal_vulkan_timestamp_stage_mask(
    VkQueueFlags queue_flags, iree_hal_execution_stage_t stage_mask) {
  VkPipelineStageFlags2 stages =
      iree_hal_vulkan_barrier_stage_mask(queue_flags, stage_mask, 0);
  if (iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)) {
    stages |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  }
  return stages;
}

static void iree_hal_vulkan_timestamp_record_dependency(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    VkPipelineStageFlags2 source_stage, VkPipelineStageFlags2 target_stage) {
  const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = source_stage,
      .dstStageMask = target_stage,
  };
  const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency);
}

void iree_hal_vulkan_timestamp_record_capture(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask) {
  // Separate the requested execution join from capture. Top-stage capture can
  // use the command processor's clock packet; an end-of-pipe query followed by
  // immediate resolution can instead introduce an asynchronous query wait.
  // Zero masks emit no join, preserving concurrent sibling commands.
  if (source_stage_mask & ~IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE) {
    iree_hal_vulkan_timestamp_record_dependency(
        syms, command_buffer,
        iree_hal_vulkan_timestamp_stage_mask(queue_flags, source_stage_mask),
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
  }
  iree_vkCmdWriteTimestamp2(IREE_VULKAN_DEVICE(syms), command_buffer,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries,
                            query_index);
  if (target_stage_mask) {
    iree_hal_vulkan_timestamp_record_dependency(
        syms, command_buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        iree_hal_vulkan_timestamp_stage_mask(queue_flags, target_stage_mask));
  }
}

static void iree_hal_vulkan_timestamp_record_copy(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    VkQueryPool queries, uint32_t first, uint32_t count, VkBuffer target,
    VkDeviceSize offset) {
  // WAIT supplies the query availability dependency on the device. It is paid
  // only at an actual result dependency or normal command-buffer completion.
  iree_vkCmdCopyQueryPoolResults(
      IREE_VULKAN_DEVICE(syms), command_buffer, queries, first, count, target,
      offset, sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

void iree_hal_vulkan_timestamp_record_direct(
    const iree_hal_vulkan_device_syms_t* syms, VkQueueFlags queue_flags,
    VkCommandBuffer command_buffer, VkQueryPool queries, uint32_t query_index,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask, VkBuffer target_buffer,
    VkDeviceSize target_offset) {
  iree_hal_vulkan_timestamp_record_capture(
      syms, queue_flags, command_buffer, queries, query_index,
      source_stage_mask, target_stage_mask);
  iree_hal_vulkan_timestamp_record_copy(syms, command_buffer, queries,
                                        query_index, 1, target_buffer,
                                        target_offset);
}

static void iree_hal_vulkan_timestamp_record_scatter(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    const iree_hal_vulkan_timestamp_state_t* state, VkDeviceSize offset,
    uint32_t count, VkDeviceAddress target_table) {
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
      .size = (VkDeviceSize)count * sizeof(uint64_t),
  };
  const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), command_buffer,
                             &dependency);
  const uint64_t constants[] = {state->resolve_address + offset,
                                target_table + offset, count};
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         state->pipeline->handle);
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(syms), command_buffer,
                          state->pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(constants), constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(syms), command_buffer, 1, 1, 1);
}

static bool iree_hal_vulkan_timestamp_matches_range(
    const iree_hal_vulkan_timestamp_pending_t* pending,
    iree_hal_buffer_ref_t range) {
  if (!pending->buffer) {
    return false;
  }
  if (!range.buffer) {
    return true;
  }
  if (!range.length) {
    return false;
  }
  // Distinct imports may share storage without sharing logical byte
  // coordinates. Only a shared allocation proves range disjointness.
  if (iree_hal_buffer_allocated_buffer(pending->buffer) !=
      iree_hal_buffer_allocated_buffer(range.buffer)) {
    return true;
  }
  return iree_hal_buffer_test_overlap(
             pending->buffer, pending->offset, 8, range.buffer, range.offset,
             range.length) != IREE_HAL_BUFFER_OVERLAP_DISJOINT;
}

void iree_hal_vulkan_timestamp_record_delivery(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    iree_hal_vulkan_timestamp_recording_t* recording,
    iree_hal_buffer_ref_t range) {
  if (recording->first_pending == recording->count) {
    return;
  }
  const iree_hal_vulkan_timestamp_state_t* state = recording->state;
  uint32_t first = recording->first_pending;
  while (first < recording->count) {
    iree_hal_vulkan_timestamp_pending_t* pending = &state->pending[first];
    if (!iree_hal_vulkan_timestamp_matches_range(pending, range)) {
      ++first;
      continue;
    }
    uint32_t end = first + 1;
    while (end < recording->count) {
      const iree_hal_vulkan_timestamp_pending_t* next = &state->pending[end];
      if (!iree_hal_vulkan_timestamp_matches_range(next, range) ||
          next->target_handle != pending->target_handle) {
        break;
      }
      const uint32_t displacement = end - first;
      if (next->target_offset !=
          pending->target_offset + (VkDeviceSize)displacement * 8) {
        break;
      }
      ++end;
    }
    const uint32_t count = end - first;
    if (pending->target_handle) {
      iree_hal_vulkan_timestamp_record_copy(
          syms, command_buffer, state->queries, first, count,
          pending->target_handle, pending->target_offset);
    } else {
      iree_hal_vulkan_timestamp_record_copy(
          syms, command_buffer, state->queries, first, count,
          state->resolve_handle,
          state->resolve_offset + pending->target_offset);
      iree_hal_vulkan_timestamp_record_scatter(syms, command_buffer, state,
                                               pending->target_offset, count,
                                               recording->target_table);
    }
    for (uint32_t i = first; i < end; ++i) {
      state->pending[i].buffer = NULL;
    }
    first = end;
  }
  while (recording->first_pending < recording->count &&
         !state->pending[recording->first_pending].buffer) {
    ++recording->first_pending;
  }
}
