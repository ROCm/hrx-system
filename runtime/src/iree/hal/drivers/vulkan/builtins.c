// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/builtins.h"

#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/vulkan/device/library.h"
#include "iree/hal/drivers/vulkan/physical_device.h"

typedef struct iree_hal_vulkan_fill_unaligned_push_constants_t {
  // Dword-aligned device address containing the edge bytes.
  uint64_t target_address;

  // Repeated fill pattern stored in the low bytes.
  uint32_t fill_pattern;

  // Byte width of the fill pattern.
  uint32_t fill_pattern_width;

  // Byte offset within target_address where the edge begins.
  uint32_t target_word_byte_offset;

  // Byte offset into fill_pattern corresponding to the first edge byte.
  uint32_t pattern_byte_offset;

  // Number of bytes to patch in target_word_index.
  uint32_t fill_length_bytes;
} iree_hal_vulkan_fill_unaligned_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_fill_unaligned_push_constants_t) == 32,
              "push constant layout must match the built-in shader");

typedef struct iree_hal_vulkan_update_unaligned_push_constants_t {
  // Dword-aligned device address containing the edge bytes.
  uint64_t target_address;

  // Source bytes to patch into the target word, stored in the low bytes.
  uint32_t update_word;

  // Byte offset within target_address where the edge begins.
  uint32_t target_word_byte_offset;

  // Number of bytes to patch in target_word_index.
  uint32_t update_length_bytes;
} iree_hal_vulkan_update_unaligned_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_update_unaligned_push_constants_t) == 24,
              "push constant layout must match the built-in shader");

static iree_status_t iree_hal_vulkan_fill_unaligned_expand_pattern(
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
          "Vulkan built-in fill pattern length must be 1, 2, or 4 bytes "
          "(got %" PRIhsz ")",
          pattern_length);
  }
}

static iree_status_t iree_hal_vulkan_builtins_create_compute_pipeline(
    iree_hal_vulkan_builtins_t* builtins, iree_string_view_t spirv_file_name,
    VkPipeline* out_pipeline) {
  *out_pipeline = VK_NULL_HANDLE;
  const iree_const_byte_span_t spirv_module =
      iree_hal_vulkan_device_library_lookup(spirv_file_name);
  VkShaderModule shader_module = VK_NULL_HANDLE;
  VkShaderModuleCreateInfo shader_module_create_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv_module.data_length,
      .pCode = (const uint32_t*)spirv_module.data,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status = iree_vkCreateShaderModule(
      IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
      &shader_module_create_info, /*pAllocator=*/NULL, &shader_module);
  IREE_LEAK_CHECK_DISABLE_POP();
  if (iree_status_is_ok(status)) {
    VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
    };
    VkComputePipelineCreateInfo pipeline_create_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_create_info,
        .layout = builtins->edge_patch_pipeline_layout,
    };
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_vkCreateComputePipelines(
        IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
        /*pipelineCache=*/VK_NULL_HANDLE, /*createInfoCount=*/1,
        &pipeline_create_info, /*pAllocator=*/NULL, out_pipeline);
    IREE_LEAK_CHECK_DISABLE_POP();
  }
  if (shader_module) {
    iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(&builtins->syms),
                               builtins->logical_device, shader_module,
                               /*pAllocator=*/NULL);
  }
  return status;
}

iree_status_t iree_hal_vulkan_builtins_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_builtins_t* out_builtins) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(out_builtins);
  memset(out_builtins, 0, sizeof(*out_builtins));
  out_builtins->syms = *syms;
  out_builtins->logical_device = logical_device;

  VkPushConstantRange push_constant_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = (uint32_t)sizeof(iree_hal_vulkan_fill_unaligned_push_constants_t),
  };
  VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_constant_range,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status = iree_vkCreatePipelineLayout(
      IREE_VULKAN_DEVICE(&out_builtins->syms), out_builtins->logical_device,
      &pipeline_layout_create_info, /*pAllocator=*/NULL,
      &out_builtins->edge_patch_pipeline_layout);
  IREE_LEAK_CHECK_DISABLE_POP();

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_create_compute_pipeline(
        out_builtins, IREE_SV("fill_unaligned.spv"),
        &out_builtins->fill_edge_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_create_compute_pipeline(
        out_builtins, IREE_SV("update_unaligned.spv"),
        &out_builtins->update_edge_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_byte_transfer_pipelines_initialize(
        syms, logical_device, physical_device,
        &out_builtins->byte_transfer_pipelines);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_atomic_pipelines_initialize(
        syms, logical_device, enabled_features,
        &out_builtins->atomic_pipelines);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_builtins_deinitialize(out_builtins);
  }
  return status;
}

void iree_hal_vulkan_builtins_deinitialize(
    iree_hal_vulkan_builtins_t* builtins) {
  if (!builtins || !builtins->logical_device) return;
  iree_hal_vulkan_atomic_pipelines_deinitialize(&builtins->atomic_pipelines);
  iree_hal_vulkan_byte_transfer_pipelines_deinitialize(
      &builtins->byte_transfer_pipelines);
  if (builtins->update_edge_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                           builtins->logical_device,
                           builtins->update_edge_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (builtins->fill_edge_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                           builtins->logical_device,
                           builtins->fill_edge_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (builtins->edge_patch_pipeline_layout) {
    iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(&builtins->syms),
                                 builtins->logical_device,
                                 builtins->edge_patch_pipeline_layout,
                                 /*pAllocator=*/NULL);
  }
  memset(builtins, 0, sizeof(*builtins));
}

typedef struct iree_hal_vulkan_edge_patches_t {
  // Number of bytes patched at the beginning of the range.
  uint32_t prefix_length;

  // Byte offset of the suffix from the beginning of the range.
  VkDeviceSize suffix_offset;

  // Number of bytes patched at the end of the range.
  uint32_t suffix_length;
} iree_hal_vulkan_edge_patches_t;

static iree_status_t iree_hal_vulkan_builtins_calculate_edge_patches(
    VkDeviceAddress target_address, VkDeviceSize length,
    iree_hal_vulkan_edge_patches_t* out_patches) {
  *out_patches = (iree_hal_vulkan_edge_patches_t){0};
  if (length == 0) return iree_ok_status();
  if (length > UINT64_MAX - target_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in edge patch range overflows");
  }

  const uint32_t target_word_byte_offset =
      (uint32_t)(target_address % sizeof(uint32_t));
  if (target_word_byte_offset != 0) {
    out_patches->prefix_length =
        (uint32_t)iree_min(length, sizeof(uint32_t) - target_word_byte_offset);
  } else if (length < sizeof(uint32_t)) {
    out_patches->prefix_length = (uint32_t)length;
  }

  const VkDeviceAddress target_end = target_address + length;
  const uint32_t suffix_length = (uint32_t)(target_end % sizeof(uint32_t));
  const VkDeviceAddress suffix_address = target_end - suffix_length;
  if (suffix_length != 0 &&
      suffix_address >= target_address + out_patches->prefix_length) {
    out_patches->suffix_offset = suffix_address - target_address;
    out_patches->suffix_length = suffix_length;
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_builtins_record_fill_edge(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDeviceAddress fill_address, VkDeviceAddress edge_address,
    uint32_t edge_length, uint32_t fill_pattern,
    iree_host_size_t pattern_length) {
  const iree_hal_vulkan_fill_unaligned_push_constants_t constants = {
      .target_address = edge_address & ~(VkDeviceAddress)3,
      .fill_pattern = fill_pattern,
      .fill_pattern_width = (uint32_t)pattern_length,
      .target_word_byte_offset = (uint32_t)(edge_address % sizeof(uint32_t)),
      .pattern_byte_offset =
          (uint32_t)((edge_address - fill_address) % pattern_length),
      .fill_length_bytes = edge_length,
  };
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                          builtins->edge_patch_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
                          sizeof(constants), &constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1,
                     /*groupCountZ=*/1);
}

iree_status_t iree_hal_vulkan_builtins_record_fill_edges(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDeviceAddress target_address, VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  if (length == 0) return iree_ok_status();

  iree_hal_vulkan_edge_patches_t patches;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_builtins_calculate_edge_patches(
      target_address, length, &patches));
  if (patches.prefix_length == 0 && patches.suffix_length == 0) {
    return iree_ok_status();
  }

  uint32_t fill_pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_fill_unaligned_expand_pattern(
      pattern, pattern_length, &fill_pattern));
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         builtins->fill_edge_pipeline);
  if (patches.prefix_length != 0) {
    iree_hal_vulkan_builtins_record_fill_edge(
        builtins, command_buffer, target_address, target_address,
        patches.prefix_length, fill_pattern, pattern_length);
  }
  if (patches.suffix_length != 0) {
    iree_hal_vulkan_builtins_record_fill_edge(
        builtins, command_buffer, target_address,
        target_address + patches.suffix_offset, patches.suffix_length,
        fill_pattern, pattern_length);
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_builtins_record_update_edge(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDeviceAddress edge_address, uint32_t edge_length,
    const uint8_t* source_data) {
  uint32_t update_word = 0;
  memcpy(&update_word, source_data, edge_length);
  const iree_hal_vulkan_update_unaligned_push_constants_t constants = {
      .target_address = edge_address & ~(VkDeviceAddress)3,
      .update_word = update_word,
      .target_word_byte_offset = (uint32_t)(edge_address % sizeof(uint32_t)),
      .update_length_bytes = edge_length,
  };
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                          builtins->edge_patch_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
                          sizeof(constants), &constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1,
                     /*groupCountZ=*/1);
}

iree_status_t iree_hal_vulkan_builtins_record_update_edges(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDeviceAddress target_address, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (length == 0) return iree_ok_status();
  if (!source_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan built-in update source is NULL");
  }
  if (length > (VkDeviceSize)source_data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update source length %" PRIhsz
                            " is smaller than update length %" PRIu64,
                            source_data_length, (uint64_t)length);
  }

  iree_hal_vulkan_edge_patches_t patches;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_builtins_calculate_edge_patches(
      target_address, length, &patches));
  if (patches.prefix_length == 0 && patches.suffix_length == 0) {
    return iree_ok_status();
  }

  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         builtins->update_edge_pipeline);
  if (patches.prefix_length != 0) {
    iree_hal_vulkan_builtins_record_update_edge(
        builtins, command_buffer, target_address, patches.prefix_length,
        source_data);
  }
  if (patches.suffix_length != 0) {
    iree_hal_vulkan_builtins_record_update_edge(
        builtins, command_buffer, target_address + patches.suffix_offset,
        patches.suffix_length,
        source_data + (iree_host_size_t)patches.suffix_offset);
  }
  return iree_ok_status();
}
