// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/atomic.h"

#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/vulkan/barrier.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/device/library.h"

typedef struct iree_hal_vulkan_atomic_push_constants_32_t {
  // Final target BDA or BDA publication slot address.
  uint64_t target_address;

  // Value compared, stored, or used as the RMW operand.
  uint32_t value;

  // Mask applied by wait operations.
  uint32_t mask;

  // Built-in shader operation selector.
  uint32_t operation;

  // Built-in shader recording flags.
  uint32_t flags;
} iree_hal_vulkan_atomic_push_constants_32_t;

static_assert(sizeof(iree_hal_vulkan_atomic_push_constants_32_t) == 24,
              "push constant layout must match the built-in shader");

typedef struct iree_hal_vulkan_atomic_push_constants_64_t {
  // Final target BDA or BDA publication slot address.
  uint64_t target_address;

  // Value compared, stored, or used as the RMW operand.
  uint64_t value;

  // Mask applied by wait operations.
  uint64_t mask;

  // Built-in shader operation selector.
  uint32_t operation;

  // Built-in shader recording flags.
  uint32_t flags;
} iree_hal_vulkan_atomic_push_constants_64_t;

static_assert(sizeof(iree_hal_vulkan_atomic_push_constants_64_t) == 32,
              "push constant layout must match the built-in shader");

static iree_status_t iree_hal_vulkan_atomic_create_pipeline(
    iree_hal_vulkan_atomic_pipelines_t* pipelines,
    iree_string_view_t spirv_file_name, VkPipeline* out_pipeline) {
  *out_pipeline = VK_NULL_HANDLE;
  const iree_const_byte_span_t spirv_module =
      iree_hal_vulkan_device_library_lookup(spirv_file_name);

  VkShaderModule shader_module = VK_NULL_HANDLE;
  const VkShaderModuleCreateInfo shader_module_create_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv_module.data_length,
      .pCode = (const uint32_t*)spirv_module.data,
  };
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

iree_status_t iree_hal_vulkan_atomic_pipelines_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_features_t enabled_features,
    iree_hal_vulkan_atomic_pipelines_t* out_pipelines) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_pipelines);
  memset(out_pipelines, 0, sizeof(*out_pipelines));
  out_pipelines->syms = *syms;
  out_pipelines->logical_device = logical_device;

  const iree_hal_atomic_capabilities_t capabilities =
      iree_hal_vulkan_atomic_capabilities(enabled_features);
  if (capabilities.operations.device_scope_32 == 0) {
    return iree_ok_status();
  }

  const VkPushConstantRange push_constant_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(iree_hal_vulkan_atomic_push_constants_64_t),
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
    status = iree_hal_vulkan_atomic_create_pipeline(
        out_pipelines, IREE_SV("atomic_32.spv"), &out_pipelines->pipeline_32);
  }
  if (iree_status_is_ok(status) &&
      capabilities.operations.device_scope_64 != 0) {
    status = iree_hal_vulkan_atomic_create_pipeline(
        out_pipelines, IREE_SV("atomic_64.spv"), &out_pipelines->pipeline_64);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_atomic_pipelines_deinitialize(out_pipelines);
  }
  return status;
}

void iree_hal_vulkan_atomic_pipelines_deinitialize(
    iree_hal_vulkan_atomic_pipelines_t* pipelines) {
  if (!pipelines || !pipelines->logical_device) return;
  if (pipelines->pipeline_64) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&pipelines->syms),
                           pipelines->logical_device, pipelines->pipeline_64,
                           /*pAllocator=*/NULL);
  }
  if (pipelines->pipeline_32) {
    iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(&pipelines->syms),
                           pipelines->logical_device, pipelines->pipeline_32,
                           /*pAllocator=*/NULL);
  }
  if (pipelines->pipeline_layout) {
    iree_vkDestroyPipelineLayout(
        IREE_VULKAN_DEVICE(&pipelines->syms), pipelines->logical_device,
        pipelines->pipeline_layout, /*pAllocator=*/NULL);
  }
  memset(pipelines, 0, sizeof(*pipelines));
}

iree_hal_vulkan_atomic_params_t iree_hal_vulkan_atomic_params_from_wait(
    iree_hal_atomic_wait_params_t params) {
  iree_hal_vulkan_atomic_operation_t operation =
      IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_EQUAL;
  switch (params.condition) {
    case IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_EQUAL;
      break;
    case IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_NOT_EQUAL;
      break;
    case IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_UNSIGNED_GREATER_EQUAL;
      break;
  }
  return (iree_hal_vulkan_atomic_params_t){
      .value = params.value,
      .mask = params.mask,
      .flags = params.flags,
      .width = params.width,
      .operation = operation,
  };
}

iree_hal_vulkan_atomic_params_t iree_hal_vulkan_atomic_params_from_store(
    iree_hal_atomic_store_params_t params) {
  return (iree_hal_vulkan_atomic_params_t){
      .value = params.value,
      .flags = params.flags,
      .width = params.width,
      .operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_STORE,
  };
}

iree_hal_vulkan_atomic_params_t iree_hal_vulkan_atomic_params_from_rmw(
    iree_hal_atomic_rmw_params_t params) {
  iree_hal_vulkan_atomic_operation_t operation =
      IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_ADD;
  switch (params.operation) {
    case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_ADD;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_SUBTRACT;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_AND;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_OR;
      break;
    case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
      operation = IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_XOR;
      break;
  }
  return (iree_hal_vulkan_atomic_params_t){
      .value = params.operand,
      .flags = params.flags,
      .width = params.width,
      .operation = operation,
  };
}

iree_status_t iree_hal_vulkan_atomic_validate(
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    iree_hal_vulkan_atomic_params_t params) {
  if (iree_any_bit_set(params.flags, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan built-in atomics do not support system scope");
  }
  switch (params.width) {
    case IREE_HAL_ATOMIC_WIDTH_32:
      if (pipelines->pipeline_32) return iree_ok_status();
      break;
    case IREE_HAL_ATOMIC_WIDTH_64:
      if (pipelines->pipeline_64) return iree_ok_status();
      break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan built-in atomics do not support %u-bit "
                          "operations",
                          params.width);
}

VkAccessFlags2 iree_hal_vulkan_atomic_access_mask(
    iree_hal_vulkan_atomic_params_t params) {
  switch (params.operation) {
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_EQUAL:
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_NOT_EQUAL:
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_WAIT_UNSIGNED_GREATER_EQUAL:
      return VK_ACCESS_2_SHADER_READ_BIT;
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_STORE:
      return VK_ACCESS_2_SHADER_WRITE_BIT;
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_ADD:
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_SUBTRACT:
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_AND:
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_OR:
    case IREE_HAL_VULKAN_ATOMIC_OPERATION_RMW_XOR:
      return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    default:
      IREE_ASSERT_UNREACHABLE("atomic parameters must be validated");
      return 0;
  }
}

iree_status_t iree_hal_vulkan_atomic_resolve_target_address(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_hal_atomic_width_t width, VkDeviceAddress* out_target_address) {
  *out_target_address = 0;
  VkDeviceAddress buffer_address = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_device_address(target_buffer, &buffer_address));
  if (buffer_address == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan atomic target has no device address");
  }
  if (target_offset > UINT64_MAX - buffer_address) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan atomic target device address overflows");
  }
  const VkDeviceAddress target_address = buffer_address + target_offset;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_atomic_validate_target_address(target_address, width));
  *out_target_address = target_address;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_atomic_validate_target_address(
    VkDeviceAddress target_address, iree_hal_atomic_width_t width) {
  const iree_device_size_t alignment = iree_hal_atomic_width_byte_count(width);
  if (alignment == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid Vulkan atomic width %u", width);
  }
  if ((target_address & (alignment - 1)) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan atomic target device address 0x%" PRIx64
                            " does not satisfy %" PRIdsz "-byte alignment",
                            (uint64_t)target_address, alignment);
  }
  return iree_ok_status();
}

void iree_hal_vulkan_atomic_record(
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    VkCommandBuffer command_buffer, VkDeviceAddress target_address,
    iree_hal_vulkan_atomic_record_flags_t record_flags,
    iree_hal_vulkan_atomic_params_t params) {
  const uint32_t shader_flags =
      iree_any_bit_set(record_flags,
                       IREE_HAL_VULKAN_ATOMIC_RECORD_FLAG_INDIRECT_TARGET)
          ? 1u
          : 0u;
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (params.width == IREE_HAL_ATOMIC_WIDTH_32) {
    const iree_hal_vulkan_atomic_push_constants_32_t push_constants = {
        .target_address = target_address,
        .value = (uint32_t)params.value,
        .mask = (uint32_t)params.mask,
        .operation = params.operation,
        .flags = shader_flags,
    };
    iree_vkCmdPushConstants(
        IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
        pipelines->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        /*offset=*/0, sizeof(push_constants), &push_constants);
    pipeline = pipelines->pipeline_32;
  } else {
    const iree_hal_vulkan_atomic_push_constants_64_t push_constants = {
        .target_address = target_address,
        .value = params.value,
        .mask = params.mask,
        .operation = params.operation,
        .flags = shader_flags,
    };
    iree_vkCmdPushConstants(
        IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
        pipelines->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        /*offset=*/0, sizeof(push_constants), &push_constants);
    pipeline = pipelines->pipeline_64;
  }
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  iree_vkCmdDispatch(IREE_VULKAN_DEVICE(&pipelines->syms), command_buffer,
                     /*groupCountX=*/1, /*groupCountY=*/1, /*groupCountZ=*/1);
}

void iree_hal_vulkan_atomic_record_command(
    const iree_hal_vulkan_device_syms_t* syms,
    const iree_hal_vulkan_atomic_pipelines_t* pipelines,
    VkCommandBuffer command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    VkDeviceAddress target_address,
    iree_hal_vulkan_atomic_record_flags_t record_flags,
    iree_hal_vulkan_barrier_flags_t barrier_flags,
    iree_hal_vulkan_atomic_params_t params) {
  const VkAccessFlags2 atomic_access_mask =
      iree_hal_vulkan_atomic_access_mask(params);
  const bool has_release =
      iree_any_bit_set(params.flags, IREE_HAL_ATOMIC_FLAG_RELEASE) &&
      iree_any_bit_set(atomic_access_mask, VK_ACCESS_2_SHADER_WRITE_BIT);
  const iree_hal_vulkan_barrier_t pre_barrier = {
      .flags = barrier_flags,
      .source_stage_mask = source_stage_mask,
      .source_access_mask = has_release
                                ? iree_hal_vulkan_barrier_source_access_mask(
                                      source_stage_mask, barrier_flags)
                                : 0,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_ATOMIC,
      .target_access_mask = has_release ? atomic_access_mask : 0,
  };
  iree_hal_vulkan_barrier_record(syms, command_buffer, &pre_barrier);

  iree_hal_vulkan_atomic_record(pipelines, command_buffer, target_address,
                                record_flags, params);

  const bool has_acquire =
      iree_any_bit_set(params.flags, IREE_HAL_ATOMIC_FLAG_ACQUIRE) &&
      iree_any_bit_set(atomic_access_mask, VK_ACCESS_2_SHADER_READ_BIT);
  const iree_hal_vulkan_barrier_t post_barrier = {
      .flags = barrier_flags,
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_ATOMIC,
      .source_access_mask = has_acquire ? atomic_access_mask : 0,
      .target_stage_mask = target_stage_mask,
      .target_access_mask = has_acquire
                                ? iree_hal_vulkan_barrier_target_access_mask(
                                      target_stage_mask, barrier_flags)
                                : 0,
  };
  iree_hal_vulkan_barrier_record(syms, command_buffer, &post_barrier);
}
