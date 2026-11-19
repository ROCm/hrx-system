// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/builtins.h"

#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/hal/driver/vulkan/device/library.h"
#include "iree/hal/driver/vulkan/physical_device.h"

typedef struct iree_hal_vulkan_fill_unaligned_push_constants_t {
  // Four bytes of repeated fill pattern.
  uint32_t pattern;
  // First row byte relative to the descriptor base.
  uint32_t target_offset;
  // Pattern phase at the first useful byte of each row.
  uint32_t pattern_offset;
  // Useful bytes in each row.
  uint32_t row_length;
  // Number of rows in each slice.
  uint32_t row_count;
  // Number of slices.
  uint32_t slice_count;
  // Byte distance between rows, or zero for one row.
  uint32_t row_stride;
  // Byte distance between slices, or zero for one slice.
  uint32_t slice_stride;
} iree_hal_vulkan_fill_unaligned_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_fill_unaligned_push_constants_t) == 32,
              "push constant layout must match the built-in shader");

typedef struct iree_hal_vulkan_update_unaligned_push_constants_t {
  // Source bytes to patch into the target word, stored in the low bytes.
  uint32_t update_word;

  // Storage-buffer uint32 element containing the first edge byte.
  uint32_t target_word_index;

  // Byte offset within target_word_index where the edge begins.
  uint32_t target_word_byte_offset;

  // Number of bytes to patch in target_word_index.
  uint32_t update_length_bytes;
} iree_hal_vulkan_update_unaligned_push_constants_t;

static_assert(sizeof(iree_hal_vulkan_update_unaligned_push_constants_t) == 16,
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
      if (pattern_length == 1) {
        *out_pattern *= 0x01010101u;
      } else if (pattern_length == 2) {
        *out_pattern |= *out_pattern << 16;
      }
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
    VkPipelineLayout pipeline_layout, VkPipeline* out_pipeline) {
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
        .layout = pipeline_layout,
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

  VkDeviceSize min_storage_buffer_offset_alignment =
      physical_device->properties2.properties.limits
          .minStorageBufferOffsetAlignment;
  if (min_storage_buffer_offset_alignment == 0) {
    min_storage_buffer_offset_alignment = 1;
  }
  if ((min_storage_buffer_offset_alignment &
       (min_storage_buffer_offset_alignment - 1)) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan reported non-power-of-two "
                            "minStorageBufferOffsetAlignment %" PRIu64,
                            (uint64_t)min_storage_buffer_offset_alignment);
  }
  out_builtins->min_storage_buffer_offset_alignment =
      min_storage_buffer_offset_alignment;
  out_builtins->max_storage_buffer_range =
      physical_device->properties2.properties.limits.maxStorageBufferRange;

  VkDescriptorSetLayoutBinding fill_layout_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  };
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &fill_layout_binding,
  };
  IREE_LEAK_CHECK_DISABLE_PUSH();
  iree_status_t status = iree_vkCreateDescriptorSetLayout(
      IREE_VULKAN_DEVICE(&out_builtins->syms), out_builtins->logical_device,
      &descriptor_set_layout_create_info, /*pAllocator=*/NULL,
      &out_builtins->storage_buffer_descriptor_set_layout);
  IREE_LEAK_CHECK_DISABLE_POP();

  if (iree_status_is_ok(status)) {
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size =
            (uint32_t)sizeof(iree_hal_vulkan_fill_unaligned_push_constants_t),
    };
    VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &out_builtins->storage_buffer_descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_vkCreatePipelineLayout(
        IREE_VULKAN_DEVICE(&out_builtins->syms), out_builtins->logical_device,
        &pipeline_layout_create_info, /*pAllocator=*/NULL,
        &out_builtins->storage_buffer_pipeline_layout);
    IREE_LEAK_CHECK_DISABLE_POP();
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_create_compute_pipeline(
        out_builtins, IREE_SV("fill_unaligned.spv"),
        out_builtins->storage_buffer_pipeline_layout,
        &out_builtins->fill_pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_builtins_create_compute_pipeline(
        out_builtins, IREE_SV("update_unaligned.spv"),
        out_builtins->storage_buffer_pipeline_layout,
        &out_builtins->update_pipeline);
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
  if (!builtins || !builtins->logical_device) {
    return;
  }
  iree_hal_vulkan_atomic_pipelines_deinitialize(&builtins->atomic_pipelines);
  if (builtins->update_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                           builtins->logical_device, builtins->update_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (builtins->fill_pipeline) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                           builtins->logical_device, builtins->fill_pipeline,
                           /*pAllocator=*/NULL);
  }
  if (builtins->storage_buffer_pipeline_layout) {
    iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(&builtins->syms),
                                 builtins->logical_device,
                                 builtins->storage_buffer_pipeline_layout,
                                 /*pAllocator=*/NULL);
  }
  if (builtins->storage_buffer_descriptor_set_layout) {
    iree_vkDestroyDescriptorSetLayout(
        IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
        builtins->storage_buffer_descriptor_set_layout,
        /*pAllocator=*/NULL);
  }
  memset(builtins, 0, sizeof(*builtins));
}

void iree_hal_vulkan_builtins_record_copy_region(
    const iree_hal_vulkan_device_syms_t* syms, VkCommandBuffer command_buffer,
    VkBuffer source, VkDeviceSize source_offset, VkBuffer target,
    VkDeviceSize target_offset, const iree_hal_transfer_region_t* region) {
  if (iree_hal_transfer_range_is_linear(region->row_length, region->row_count,
                                        region->slice_count,
                                        &region->source_strides) &&
      iree_hal_transfer_range_is_linear(region->row_length, region->row_count,
                                        region->slice_count,
                                        &region->target_strides)) {
    const VkBufferCopy copy = {.srcOffset = source_offset,
                               .dstOffset = target_offset,
                               .size = iree_hal_transfer_region_length(region)};
    iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(syms), command_buffer, source,
                         target, 1, &copy);
    return;
  }
  VkBufferCopy copies[64];
  uint32_t count = 0;
  for (uint64_t slice = 0; slice < region->slice_count; ++slice) {
    for (uint64_t row = 0; row < region->row_count; ++row) {
      copies[count++] = (VkBufferCopy){
          .srcOffset =
              source_offset +
              (region->row_count > 1 ? row * region->source_strides.row_stride
                                     : 0) +
              (region->slice_count > 1
                   ? slice * region->source_strides.slice_stride
                   : 0),
          .dstOffset =
              target_offset +
              (region->row_count > 1 ? row * region->target_strides.row_stride
                                     : 0) +
              (region->slice_count > 1
                   ? slice * region->target_strides.slice_stride
                   : 0),
          .size = region->row_length,
      };
      if (count == IREE_ARRAYSIZE(copies)) {
        iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(syms), command_buffer, source,
                             target, count, copies);
        count = 0;
      }
    }
  }
  if (count) {
    iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(syms), command_buffer, source,
                         target, count, copies);
  }
}

iree_hal_fill_region_t iree_hal_vulkan_builtins_fill_tile(
    const iree_hal_vulkan_builtins_t* builtins,
    const iree_hal_fill_region_t* region) {
  // Leave room for descriptor alignment and the final partial dword.
  const uint64_t capacity =
      (builtins->max_storage_buffer_range -
       builtins->min_storage_buffer_offset_alignment - 4) &
      ~UINT64_C(3);
  iree_hal_fill_region_t tile = *region;
  tile.row_length = iree_min(region->row_length, capacity);
  tile.row_count = 1;
  tile.slice_count = 1;
  if (tile.row_length == region->row_length && region->row_count > 1) {
    tile.row_count = iree_min(
        iree_min(region->row_count, UINT64_C(65535)),
        1 + (capacity - tile.row_length) / region->target_strides.row_stride);
  }
  if (tile.row_count == region->row_count && region->slice_count > 1) {
    const uint64_t slice_length =
        tile.row_length +
        (tile.row_count - 1) * region->target_strides.row_stride;
    tile.slice_count = iree_min(
        iree_min(region->slice_count, UINT64_C(65535)),
        1 + (capacity - slice_length) / region->target_strides.slice_stride);
  }
  return tile;
}

iree_status_t iree_hal_vulkan_builtins_record_fill_region(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, const VkDescriptorSet* descriptor_sets,
    VkBuffer target_buffer, VkDeviceSize target_offset, const uint8_t* pattern,
    iree_host_size_t pattern_length, const iree_hal_fill_region_t* region) {
  uint32_t expanded_pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_fill_unaligned_expand_pattern(
      pattern, pattern_length, &expanded_pattern));
  const iree_hal_fill_region_t tile =
      iree_hal_vulkan_builtins_fill_tile(builtins, region);
  iree_status_t status = iree_ok_status();
  for (uint64_t slice = 0;
       iree_status_is_ok(status) && slice < region->slice_count;
       slice += tile.slice_count) {
    for (uint64_t row = 0; iree_status_is_ok(status) && row < region->row_count;
         row += tile.row_count) {
      for (uint64_t column = 0;
           iree_status_is_ok(status) && column < region->row_length;
           column += tile.row_length) {
        const VkDeviceSize offset =
            target_offset + column +
            (region->row_count > 1 ? row * region->target_strides.row_stride
                                   : 0) +
            (region->slice_count > 1
                 ? slice * region->target_strides.slice_stride
                 : 0);
        const VkDeviceSize descriptor_offset =
            offset & ~(builtins->min_storage_buffer_offset_alignment - 1);
        const iree_hal_vulkan_fill_unaligned_push_constants_t constants = {
            .pattern = expanded_pattern,
            .target_offset = (uint32_t)(offset - descriptor_offset),
            .row_length = (uint32_t)iree_min(tile.row_length,
                                             region->row_length - column),
            .row_count =
                (uint32_t)iree_min(tile.row_count, region->row_count - row),
            .slice_count = (uint32_t)iree_min(tile.slice_count,
                                              region->slice_count - slice),
            .row_stride = tile.row_count > 1
                              ? (uint32_t)region->target_strides.row_stride
                              : 0,
            .slice_stride = tile.slice_count > 1
                                ? (uint32_t)region->target_strides.slice_stride
                                : 0,
        };
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        if (descriptor_sets) {
          descriptor_set = *descriptor_sets++;
        } else {
          const VkDescriptorSetAllocateInfo allocate_info = {
              .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
              .descriptorPool = descriptor_pool,
              .descriptorSetCount = 1,
              .pSetLayouts = &builtins->storage_buffer_descriptor_set_layout,
          };
          status = iree_vkAllocateDescriptorSets(
              IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
              &allocate_info, &descriptor_set);
          if (!iree_status_is_ok(status)) {
            break;
          }
        }
        const VkDescriptorBufferInfo buffer_info = {
            .buffer = target_buffer,
            .offset = descriptor_offset,
            .range = iree_device_align(
                constants.target_offset +
                    (VkDeviceSize)(constants.row_count - 1) *
                        constants.row_stride +
                    (VkDeviceSize)(constants.slice_count - 1) *
                        constants.slice_stride +
                    constants.row_length,
                4),
        };
        const VkWriteDescriptorSet write_info = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_info,
        };
        iree_vkUpdateDescriptorSets(IREE_VULKAN_DEVICE(&builtins->syms),
                                    builtins->logical_device, 1, &write_info, 0,
                                    NULL);
        iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms),
                               command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               builtins->fill_pipeline);
        iree_vkCmdBindDescriptorSets(IREE_VULKAN_DEVICE(&builtins->syms),
                                     command_buffer,
                                     VK_PIPELINE_BIND_POINT_COMPUTE,
                                     builtins->storage_buffer_pipeline_layout,
                                     0, 1, &descriptor_set, 0, NULL);
        iree_vkCmdPushConstants(
            IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
            builtins->storage_buffer_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
        const uint32_t word_count = (constants.row_length + 6) / 4;
        iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                           iree_min((word_count + 63) / 64, 256u),
                           constants.row_count, constants.slice_count);
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_builtins_record_fill_edge(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorSet descriptor_set, VkBuffer target_buffer,
    VkDeviceSize fill_offset, VkDeviceSize edge_offset, uint32_t edge_length,
    const uint8_t* pattern, iree_host_size_t pattern_length) {
  if (edge_length == 0) {
    return iree_ok_status();
  }
  if (!descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan built-in fill requires a descriptor set");
  }

  const VkDeviceSize target_word_offset = edge_offset & ~(VkDeviceSize)3;
  const VkDeviceSize descriptor_offset =
      target_word_offset & ~(builtins->min_storage_buffer_offset_alignment - 1);
  const VkDeviceSize target_word_index =
      (target_word_offset - descriptor_offset) / sizeof(uint32_t);
  if (target_word_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in fill descriptor alignment "
                            "produced word index %" PRIu64,
                            (uint64_t)target_word_index);
  }

  uint32_t fill_pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_fill_unaligned_expand_pattern(
      pattern, pattern_length, &fill_pattern));

  VkDescriptorBufferInfo buffer_info = {
      .buffer = target_buffer,
      .offset = descriptor_offset,
      .range = target_word_offset - descriptor_offset + sizeof(uint32_t),
  };
  VkWriteDescriptorSet write_info = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info,
  };
  iree_vkUpdateDescriptorSets(IREE_VULKAN_DEVICE(&builtins->syms),
                              builtins->logical_device,
                              /*descriptorWriteCount=*/1, &write_info,
                              /*descriptorCopyCount=*/0,
                              /*pDescriptorCopies=*/NULL);

  const iree_hal_vulkan_fill_unaligned_push_constants_t constants = {
      .pattern = fill_pattern,
      .target_offset = (uint32_t)(edge_offset - descriptor_offset),
      .pattern_offset = (uint32_t)((edge_offset - fill_offset) & 3),
      .row_length = edge_length,
      .row_count = 1,
      .slice_count = 1,
  };
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         builtins->fill_pipeline);
  iree_vkCmdBindDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE, builtins->storage_buffer_pipeline_layout,
      /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptor_set,
      /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/NULL);
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                          builtins->storage_buffer_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
                          sizeof(constants), &constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1,
                     /*groupCountZ=*/1);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned_impl(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) {
    return iree_ok_status();
  }
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in fill range overflows");
  }

  const VkDeviceSize target_end = target_offset + length;
  const VkDeviceSize target_start_word_byte_offset =
      target_offset % sizeof(uint32_t);
  uint32_t descriptor_set_ordinal = 0;
  VkDeviceSize start_edge_length = 0;
  if (target_start_word_byte_offset != 0) {
    start_edge_length = sizeof(uint32_t) - target_start_word_byte_offset;
    if (start_edge_length > length) {
      start_edge_length = length;
    }
  } else if (length < sizeof(uint32_t)) {
    start_edge_length = length;
  }

  iree_status_t status = iree_ok_status();
  if (start_edge_length != 0) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in fill received too few descriptor sets");
    }
    status = iree_hal_vulkan_builtins_record_fill_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, target_offset, target_offset,
        (uint32_t)start_edge_length, pattern, pattern_length);
  }

  const VkDeviceSize end_edge_length = target_end % sizeof(uint32_t);
  const VkDeviceSize end_edge_offset = target_end - end_edge_length;
  if (iree_status_is_ok(status) && end_edge_length != 0 &&
      end_edge_offset >= target_offset + start_edge_length) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in fill received too few descriptor sets");
    }
    status = iree_hal_vulkan_builtins_record_fill_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, target_offset, end_edge_offset,
        (uint32_t)end_edge_length, pattern, pattern_length);
  }
  return status;
}

iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, VkBuffer target_buffer,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) {
    return iree_ok_status();
  }
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in fill range overflows");
  }
  if (!descriptor_pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan built-in fill requires a descriptor pool");
  }
  const uint32_t descriptor_set_count =
      iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
          target_offset, length);
  if (descriptor_set_count == 0) {
    return iree_ok_status();
  }
  VkDescriptorSet descriptor_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layouts[2] = {
      builtins->storage_buffer_descriptor_set_layout,
      builtins->storage_buffer_descriptor_set_layout,
  };
  VkDescriptorSetAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = descriptor_set_count,
      .pSetLayouts = descriptor_set_layouts,
  };
  IREE_RETURN_IF_ERROR(iree_vkAllocateDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
      &allocate_info, descriptor_sets));
  return iree_hal_vulkan_builtins_record_fill_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, pattern, pattern_length);
}

uint32_t iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
    VkDeviceSize target_offset, VkDeviceSize length) {
  if (length == 0) {
    return 0;
  }
  const VkDeviceSize target_end = target_offset + length;
  const VkDeviceSize target_start_word_byte_offset =
      target_offset % sizeof(uint32_t);
  VkDeviceSize start_edge_length = 0;
  if (target_start_word_byte_offset != 0) {
    start_edge_length = sizeof(uint32_t) - target_start_word_byte_offset;
    if (start_edge_length > length) {
      start_edge_length = length;
    }
  } else if (length < sizeof(uint32_t)) {
    start_edge_length = length;
  }
  uint32_t descriptor_set_count = start_edge_length != 0 ? 1u : 0u;
  const VkDeviceSize end_edge_length = target_end % sizeof(uint32_t);
  const VkDeviceSize end_edge_offset = target_end - end_edge_length;
  if (end_edge_length != 0 &&
      end_edge_offset >= target_offset + start_edge_length) {
    descriptor_set_count = descriptor_set_count + 1;
  }
  return descriptor_set_count;
}

iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned_descriptor_sets(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* pattern, iree_host_size_t pattern_length) {
  IREE_ASSERT_ARGUMENT(descriptor_sets);
  return iree_hal_vulkan_builtins_record_fill_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, pattern, pattern_length);
}

static iree_status_t iree_hal_vulkan_builtins_record_update_edge(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorSet descriptor_set, VkBuffer target_buffer,
    VkDeviceSize edge_offset, uint32_t edge_length,
    const uint8_t* source_data) {
  if (edge_length == 0) {
    return iree_ok_status();
  }
  if (!descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan built-in update requires a descriptor set");
  }

  const VkDeviceSize target_word_offset = edge_offset & ~(VkDeviceSize)3;
  const VkDeviceSize descriptor_offset =
      target_word_offset & ~(builtins->min_storage_buffer_offset_alignment - 1);
  const VkDeviceSize target_word_index =
      (target_word_offset - descriptor_offset) / sizeof(uint32_t);
  if (target_word_index > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update descriptor alignment "
                            "produced word index %" PRIu64,
                            (uint64_t)target_word_index);
  }

  VkDescriptorBufferInfo buffer_info = {
      .buffer = target_buffer,
      .offset = descriptor_offset,
      .range = VK_WHOLE_SIZE,
  };
  VkWriteDescriptorSet write_info = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info,
  };
  iree_vkUpdateDescriptorSets(IREE_VULKAN_DEVICE(&builtins->syms),
                              builtins->logical_device,
                              /*descriptorWriteCount=*/1, &write_info,
                              /*descriptorCopyCount=*/0,
                              /*pDescriptorCopies=*/NULL);

  uint32_t update_word = 0;
  memcpy(&update_word, source_data, edge_length);
  const iree_hal_vulkan_update_unaligned_push_constants_t constants = {
      .update_word = update_word,
      .target_word_index = (uint32_t)target_word_index,
      .target_word_byte_offset = (uint32_t)(edge_offset % sizeof(uint32_t)),
      .update_length_bytes = edge_length,
  };
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE,
                         builtins->update_pipeline);
  iree_vkCmdBindDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
      VK_PIPELINE_BIND_POINT_COMPUTE, builtins->storage_buffer_pipeline_layout,
      /*firstSet=*/0, /*descriptorSetCount=*/1, &descriptor_set,
      /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/NULL);
  iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                          builtins->storage_buffer_pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, /*offset=*/0,
                          sizeof(constants), &constants);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&builtins->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1,
                     /*groupCountZ=*/1);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_builtins_record_update_unaligned_impl(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) {
    return iree_ok_status();
  }
  if (!source_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan built-in update source is NULL");
  }
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update range overflows");
  }
  if (length > (VkDeviceSize)source_data_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update source length %" PRIhsz
                            " is smaller than update length %" PRIu64,
                            source_data_length, (uint64_t)length);
  }

  const VkDeviceSize target_end = target_offset + length;
  const VkDeviceSize target_start_word_byte_offset =
      target_offset % sizeof(uint32_t);
  uint32_t descriptor_set_ordinal = 0;
  VkDeviceSize start_edge_length = 0;
  if (target_start_word_byte_offset != 0) {
    start_edge_length = sizeof(uint32_t) - target_start_word_byte_offset;
    if (start_edge_length > length) {
      start_edge_length = length;
    }
  } else if (length < sizeof(uint32_t)) {
    start_edge_length = length;
  }

  iree_status_t status = iree_ok_status();
  if (start_edge_length != 0) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in update received too few descriptor sets");
    }
    status = iree_hal_vulkan_builtins_record_update_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, target_offset, (uint32_t)start_edge_length, source_data);
  }

  const VkDeviceSize end_edge_length = target_end % sizeof(uint32_t);
  const VkDeviceSize end_edge_offset = target_end - end_edge_length;
  if (iree_status_is_ok(status) && end_edge_length != 0 &&
      end_edge_offset >= target_offset + start_edge_length) {
    if (descriptor_set_ordinal >= descriptor_set_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan built-in update received too few descriptor sets");
    }
    const VkDeviceSize source_offset = end_edge_offset - target_offset;
    status = iree_hal_vulkan_builtins_record_update_edge(
        builtins, command_buffer, descriptor_sets[descriptor_set_ordinal++],
        target_buffer, end_edge_offset, (uint32_t)end_edge_length,
        source_data + (iree_host_size_t)source_offset);
  }
  return status;
}

uint32_t iree_hal_vulkan_builtins_update_unaligned_descriptor_set_count(
    VkDeviceSize target_offset, VkDeviceSize length) {
  return iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
      target_offset, length);
}

iree_status_t iree_hal_vulkan_builtins_record_update_unaligned(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, VkBuffer target_buffer,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* source_data,
    iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  if (length == 0) {
    return iree_ok_status();
  }
  if (length > UINT64_MAX - target_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan built-in update range overflows");
  }
  if (!descriptor_pool) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan built-in update requires a descriptor pool");
  }
  const uint32_t descriptor_set_count =
      iree_hal_vulkan_builtins_update_unaligned_descriptor_set_count(
          target_offset, length);
  if (descriptor_set_count == 0) {
    return iree_ok_status();
  }
  VkDescriptorSet descriptor_sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layouts[2] = {
      builtins->storage_buffer_descriptor_set_layout,
      builtins->storage_buffer_descriptor_set_layout,
  };
  VkDescriptorSetAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = descriptor_set_count,
      .pSetLayouts = descriptor_set_layouts,
  };
  IREE_RETURN_IF_ERROR(iree_vkAllocateDescriptorSets(
      IREE_VULKAN_DEVICE(&builtins->syms), builtins->logical_device,
      &allocate_info, descriptor_sets));
  return iree_hal_vulkan_builtins_record_update_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, source_data, source_data_length);
}

iree_status_t iree_hal_vulkan_builtins_record_update_unaligned_descriptor_sets(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length) {
  IREE_ASSERT_ARGUMENT(descriptor_sets);
  return iree_hal_vulkan_builtins_record_update_unaligned_impl(
      builtins, command_buffer, descriptor_sets, descriptor_set_count,
      target_buffer, target_offset, length, source_data, source_data_length);
}
