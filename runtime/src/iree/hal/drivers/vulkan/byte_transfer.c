// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/byte_transfer.h"

#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/vulkan/device/library.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

#define IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE 64u

typedef enum iree_hal_vulkan_byte_copy_flag_bits_e {
  // No special copy behavior.
  IREE_HAL_VULKAN_BYTE_COPY_FLAG_NONE = 0u,

  // Loads source words atomically because widened source and target dwords
  // overlap even though the requested byte ranges do not.
  IREE_HAL_VULKAN_BYTE_COPY_FLAG_ATOMIC_SOURCE = 1u << 0,
} iree_hal_vulkan_byte_copy_flag_bits_t;
typedef uint32_t iree_hal_vulkan_byte_copy_flags_t;

typedef struct iree_hal_vulkan_byte_copy_push_constants_t {
  // Dword-aligned source buffer device address.
  uint64_t source_address;

  // Dword-aligned target buffer device address.
  uint64_t target_address;

  // Byte offset of the first source byte within source_address.
  uint32_t source_byte_offset;

  // Byte offset of the first target byte within target_address.
  uint32_t target_byte_offset;

  // Number of bytes copied by this segment.
  uint32_t byte_length;

  // Number of target dwords touched by this segment.
  uint32_t target_word_count;

  // First target dword processed by the current dispatch.
  uint32_t first_target_word;

  // Bitfield of iree_hal_vulkan_byte_copy_flag_bits_t values.
  iree_hal_vulkan_byte_copy_flags_t flags;
} iree_hal_vulkan_byte_copy_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_byte_copy_push_constants_t) == 40,
              "push constant layout must match the byte copy shader");

typedef struct iree_hal_vulkan_byte_fill_push_constants_t {
  // Dword-aligned target buffer device address.
  uint64_t target_address;

  // Repeated fill pattern stored in the low bytes.
  uint32_t fill_pattern;

  // Byte width of the fill pattern.
  uint32_t fill_pattern_width;

  // Byte offset of the first target byte within target_address.
  uint32_t target_byte_offset;

  // Number of bytes filled by this segment.
  uint32_t byte_length;

  // Number of target dwords touched by this segment.
  uint32_t target_word_count;

  // First target dword processed by the current dispatch.
  uint32_t first_target_word;

  // Pattern byte corresponding to the first byte of this segment.
  uint32_t pattern_byte_offset;
} iree_hal_vulkan_byte_fill_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_byte_fill_push_constants_t) == 40,
              "push constant layout must match the byte fill shader");

static iree_status_t iree_hal_vulkan_byte_transfer_create_pipeline(
    iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    iree_string_view_t spirv_file_name, VkPipeline* out_pipeline) {
  *out_pipeline = VK_NULL_HANDLE;
  const iree_const_byte_span_t spirv_module =
      iree_hal_vulkan_device_library_lookup(spirv_file_name);

  const VkShaderModuleCreateInfo shader_module_create_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv_module.data_length,
      .pCode = (const uint32_t*)spirv_module.data,
  };
  VkShaderModule shader_module = VK_NULL_HANDLE;
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status = iree_vkCreateShaderModule(
      IREE_VULKAN_DEVICE(&pipelines->syms), pipelines->logical_device,
      &shader_module_create_info, /*pAllocator=*/NULL, &shader_module);
  IREE_LEAK_CHECK_DISABLE_POP();
  if (iree_status_is_ok(status)) {
    const VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
    };
    const VkComputePipelineCreateInfo pipeline_create_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_create_info,
        .layout = pipelines->pipeline_layout,
    };
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_vkCreateComputePipelines(
        IREE_VULKAN_DEVICE(&pipelines->syms), pipelines->logical_device,
        /*pipelineCache=*/VK_NULL_HANDLE, /*createInfoCount=*/1,
        &pipeline_create_info, /*pAllocator=*/NULL, out_pipeline);
    IREE_LEAK_CHECK_DISABLE_POP();
  }
  if (shader_module) {
    iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(&pipelines->syms),
                               pipelines->logical_device, shader_module,
                               /*pAllocator=*/NULL);
  }
  return status;
}

iree_status_t iree_hal_vulkan_byte_transfer_pipelines_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_byte_transfer_pipelines_t* out_pipelines) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(out_pipelines);
  memset(out_pipelines, 0, sizeof(*out_pipelines));
  out_pipelines->syms = *syms;
  out_pipelines->logical_device = logical_device;
  out_pipelines->max_workgroup_count_x =
      physical_device->properties2.properties.limits
          .maxComputeWorkGroupCount[0];
  if (out_pipelines->max_workgroup_count_x == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan reported a zero X-dimension compute workgroup limit");
  }

  const VkPushConstantRange push_constant_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(iree_hal_vulkan_byte_copy_push_constants_t),
  };
  const VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_constant_range,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status = iree_vkCreatePipelineLayout(
      IREE_VULKAN_DEVICE(&out_pipelines->syms), out_pipelines->logical_device,
      &pipeline_layout_create_info, /*pAllocator=*/NULL,
      &out_pipelines->pipeline_layout);
  IREE_LEAK_CHECK_DISABLE_POP();
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_byte_transfer_create_pipeline(
        out_pipelines, IREE_SV("copy_bytes.spv"),
        &out_pipelines->copy_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_byte_transfer_create_pipeline(
        out_pipelines, IREE_SV("fill_bytes.spv"),
        &out_pipelines->fill_pipeline);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_byte_transfer_pipelines_deinitialize(out_pipelines);
  }
  return status;
}

void iree_hal_vulkan_byte_transfer_pipelines_deinitialize(
    iree_hal_vulkan_byte_transfer_pipelines_t* pipelines) {
  if (!pipelines || !pipelines->logical_device) return;
  if (pipelines->fill_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&pipelines->syms),
                           pipelines->logical_device, pipelines->fill_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (pipelines->copy_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&pipelines->syms),
                           pipelines->logical_device, pipelines->copy_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (pipelines->pipeline_layout) {
    iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(&pipelines->syms),
                                 pipelines->logical_device,
                                 pipelines->pipeline_layout,
                                 /*pAllocator=*/NULL);
  }
  memset(pipelines, 0, sizeof(*pipelines));
}

// Selects a segment length that fits in shader uint32 arithmetic. The maximum
// source/target byte offset plus segment length must fit in uint32 because the
// shader derives both byte indices with uint32 arithmetic. Non-final segments
// end at a target dword boundary so no two dispatches patch the same target
// word.
static uint32_t iree_hal_vulkan_byte_transfer_segment_length(
    VkDeviceAddress target_address, uint32_t maximum_byte_offset,
    VkDeviceSize remaining_length) {
  const uint32_t target_byte_offset = (uint32_t)(target_address & 3u);
  const uint32_t maximum_length = UINT32_MAX - maximum_byte_offset;
  if (remaining_length <= maximum_length) {
    return (uint32_t)remaining_length;
  }
  const uint32_t required_remainder =
      (sizeof(uint32_t) - target_byte_offset) & 3u;
  return maximum_length - ((maximum_length - required_remainder) &
                           (uint32_t)(sizeof(uint32_t) - 1));
}

static uint32_t iree_hal_vulkan_byte_transfer_target_word_count(
    uint32_t target_byte_offset, uint32_t byte_length) {
  return (uint32_t)(((uint64_t)target_byte_offset + byte_length + 3u) /
                    sizeof(uint32_t));
}

static iree_status_t iree_hal_vulkan_byte_transfer_validate_range(
    VkDeviceAddress address, VkDeviceSize length,
    iree_string_view_t range_name) {
  if (length > UINT64_MAX - address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan byte transfer %.*s range overflows",
                            (int)range_name.size, range_name.data);
  }
  if (address + length > UINT64_MAX - (sizeof(uint32_t) - 1)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan byte transfer %.*s dword-aligned range overflows",
        (int)range_name.size, range_name.data);
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_byte_transfer_aligned_ranges_overlap(
    VkDeviceAddress source_address, VkDeviceAddress target_address,
    VkDeviceSize length) {
  const VkDeviceAddress source_start = source_address & ~(VkDeviceAddress)3;
  const VkDeviceAddress source_end =
      (source_address + length + 3u) & ~(VkDeviceAddress)3;
  const VkDeviceAddress target_start = target_address & ~(VkDeviceAddress)3;
  const VkDeviceAddress target_end =
      (target_address + length + 3u) & ~(VkDeviceAddress)3;
  return source_start < target_end && target_start < source_end;
}

iree_status_t iree_hal_vulkan_byte_transfer_record_copy(
    const iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress source_address,
    VkDeviceAddress target_address, VkDeviceSize length) {
  IREE_ASSERT_ARGUMENT(pipelines);
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (length == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_validate_range(
      source_address, length, IREE_SV("source")));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_validate_range(
      target_address, length, IREE_SV("target")));
  const VkDeviceAddress source_end = source_address + length;
  const VkDeviceAddress target_end = target_address + length;
  if (source_address < target_end && target_address < source_end) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan byte copy source and target ranges overlap");
  }

  iree_hal_vulkan_byte_copy_flags_t flags = IREE_HAL_VULKAN_BYTE_COPY_FLAG_NONE;
  if (iree_hal_vulkan_byte_transfer_aligned_ranges_overlap(
          source_address, target_address, length)) {
    flags |= IREE_HAL_VULKAN_BYTE_COPY_FLAG_ATOMIC_SOURCE;
  }

  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         pipelines->copy_pipeline);
  VkDeviceSize remaining_length = length;
  while (remaining_length != 0) {
    const uint32_t source_byte_offset = (uint32_t)(source_address & 3u);
    const uint32_t target_byte_offset = (uint32_t)(target_address & 3u);
    const uint32_t segment_length =
        iree_hal_vulkan_byte_transfer_segment_length(
            target_address, iree_max(source_byte_offset, target_byte_offset),
            remaining_length);
    iree_hal_vulkan_byte_copy_push_constants_t constants = {
        .source_address = source_address & ~(VkDeviceAddress)3,
        .target_address = target_address & ~(VkDeviceAddress)3,
        .source_byte_offset = source_byte_offset,
        .target_byte_offset = target_byte_offset,
        .byte_length = segment_length,
        .target_word_count = iree_hal_vulkan_byte_transfer_target_word_count(
            target_byte_offset, segment_length),
        .flags = flags,
    };
    while (constants.first_target_word < constants.target_word_count) {
      const uint32_t remaining_word_count =
          constants.target_word_count - constants.first_target_word;
      const uint64_t required_workgroup_count =
          ((uint64_t)remaining_word_count +
           IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE - 1) /
          IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE;
      const uint32_t workgroup_count = (uint32_t)iree_min(
          required_workgroup_count, (uint64_t)pipelines->max_workgroup_count_x);
      iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&pipelines->syms),
                              command_buffer, pipelines->pipeline_layout,
                              VK_SHADER_STAGE_COMPUTE_BIT,
                              /*offset=*/0, sizeof(constants), &constants);
      iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
                         workgroup_count, /*groupCountY=*/1,
                         /*groupCountZ=*/1);
      const uint64_t dispatched_word_count =
          (uint64_t)workgroup_count *
          IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE;
      constants.first_target_word += (uint32_t)iree_min(
          dispatched_word_count, (uint64_t)remaining_word_count);
    }

    source_address += segment_length;
    target_address += segment_length;
    remaining_length -= segment_length;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_byte_transfer_expand_pattern(
    const uint8_t* pattern, iree_host_size_t pattern_length,
    uint32_t* out_pattern) {
  *out_pattern = 0;
  switch (pattern_length) {
    case 1:
    case 2:
    case 4:
      memcpy(out_pattern, pattern, pattern_length);
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan byte fill pattern length must be 1, 2, or 4 bytes "
          "(got %" PRIhsz ")",
          pattern_length);
  }
}

iree_status_t iree_hal_vulkan_byte_transfer_record_fill(
    const iree_hal_vulkan_byte_transfer_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress target_address,
    VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(pipelines);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  if (length == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_validate_range(
      target_address, length, IREE_SV("target")));
  uint32_t fill_pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_byte_transfer_expand_pattern(
      pattern, pattern_length, &fill_pattern));

  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         pipelines->fill_pipeline);
  VkDeviceSize remaining_length = length;
  VkDeviceSize fill_offset = 0;
  while (remaining_length != 0) {
    const uint32_t target_byte_offset = (uint32_t)(target_address & 3u);
    const uint32_t segment_length =
        iree_hal_vulkan_byte_transfer_segment_length(
            target_address, target_byte_offset, remaining_length);
    iree_hal_vulkan_byte_fill_push_constants_t constants = {
        .target_address = target_address & ~(VkDeviceAddress)3,
        .fill_pattern = fill_pattern,
        .fill_pattern_width = (uint32_t)pattern_length,
        .target_byte_offset = target_byte_offset,
        .byte_length = segment_length,
        .target_word_count = iree_hal_vulkan_byte_transfer_target_word_count(
            target_byte_offset, segment_length),
        .pattern_byte_offset = (uint32_t)(fill_offset % pattern_length),
    };
    while (constants.first_target_word < constants.target_word_count) {
      const uint32_t remaining_word_count =
          constants.target_word_count - constants.first_target_word;
      const uint64_t required_workgroup_count =
          ((uint64_t)remaining_word_count +
           IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE - 1) /
          IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE;
      const uint32_t workgroup_count = (uint32_t)iree_min(
          required_workgroup_count, (uint64_t)pipelines->max_workgroup_count_x);
      iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&pipelines->syms),
                              command_buffer, pipelines->pipeline_layout,
                              VK_SHADER_STAGE_COMPUTE_BIT,
                              /*offset=*/0, sizeof(constants), &constants);
      iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
                         workgroup_count, /*groupCountY=*/1,
                         /*groupCountZ=*/1);
      const uint64_t dispatched_word_count =
          (uint64_t)workgroup_count *
          IREE_HAL_VULKAN_BYTE_TRANSFER_WORKGROUP_SIZE;
      constants.first_target_word += (uint32_t)iree_min(
          dispatched_word_count, (uint64_t)remaining_word_count);
    }

    target_address += segment_length;
    fill_offset += segment_length;
    remaining_length -= segment_length;
  }
  return iree_ok_status();
}
