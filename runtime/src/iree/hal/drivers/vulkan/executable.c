// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/executable.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"
#include "iree/hal/drivers/vulkan/spirv.h"

//===----------------------------------------------------------------------===//
// Vulkan Object Creation
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_vulkan_create_specialization_info(
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, VkSpecializationInfo* out_info,
    VkSpecializationMapEntry** out_map_entries) {
  memset(out_info, 0, sizeof(*out_info));
  *out_map_entries = NULL;
  if (load_params->constant_count == 0) {
    return iree_ok_status();
  }
  if (!load_params->constants) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable declares %" PRIhsz
                            " specialization constants but no value storage",
                            load_params->constant_count);
  }
  if (load_params->constant_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable declares %" PRIhsz
                            " specialization constants, exceeding Vulkan "
                            "limit %u",
                            load_params->constant_count, UINT32_MAX);
  }
  if (load_params->constant_count > IREE_HOST_SIZE_MAX / sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable specialization constant data size "
                            "overflows");
  }
  const iree_host_size_t max_constant_offset_count =
      (iree_host_size_t)UINT32_MAX / sizeof(uint32_t) + 1;
  if (load_params->constant_count > max_constant_offset_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable declares %" PRIhsz
                            " specialization constants, exceeding Vulkan "
                            "constant offset limit %" PRIhsz,
                            load_params->constant_count,
                            max_constant_offset_count);
  }

  VkSpecializationMapEntry* map_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, load_params->constant_count, sizeof(map_entries[0]),
      (void**)&map_entries));
  for (iree_host_size_t i = 0; i < load_params->constant_count; ++i) {
    map_entries[i] = (VkSpecializationMapEntry){
        .constantID = (uint32_t)i,
        .offset = (uint32_t)(i * sizeof(uint32_t)),
        .size = sizeof(uint32_t),
    };
  }

  *out_info = (VkSpecializationInfo){
      .mapEntryCount = (uint32_t)load_params->constant_count,
      .pMapEntries = map_entries,
      .dataSize = load_params->constant_count * sizeof(uint32_t),
      .pData = load_params->constants,
  };
  *out_map_entries = map_entries;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_initialize_pipeline_bda_metadata(
    const iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata,
    iree_allocator_t host_allocator, iree_hal_vulkan_pipeline_t* out_pipeline) {
  out_pipeline->constant_byte_length = metadata->constant_byte_length;
  out_pipeline->binding_count = metadata->binding_count;
  out_pipeline->bda.root_push_constant_offset =
      metadata->root_push_constant_offset;
  out_pipeline->bda.root_push_constant_length =
      metadata->root_push_constant_length;
  out_pipeline->bda.constant_push_constant_offset =
      metadata->constant_push_constant_offset;
  out_pipeline->bda.binding_count_known = metadata->binding_count_known;

  if (metadata->binding_requirement_count == 0) {
    return iree_ok_status();
  }
  out_pipeline->bda.binding_requirement_count =
      metadata->binding_requirement_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, metadata->binding_requirement_count,
      sizeof(out_pipeline->bda.binding_requirements[0]),
      (void**)&out_pipeline->bda.binding_requirements));
  for (iree_host_size_t i = 0; i < metadata->binding_requirement_count; ++i) {
    out_pipeline->bda.binding_requirements[i] =
        (iree_hal_vulkan_bda_binding_requirement_t){
            .minimum_alignment =
                metadata->binding_requirements[i].minimum_alignment,
            .minimum_length = metadata->binding_requirements[i].minimum_length,
        };
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_pipeline_validate_bda_dispatch(
    const iree_hal_vulkan_pipeline_t* pipeline,
    iree_const_byte_span_t constants, iree_host_size_t binding_count,
    iree_string_view_t operation) {
  if (constants.data_length != pipeline->constant_byte_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s provides %" PRIhsz " constant bytes but BDA pipeline expects %u",
        (int)operation.size, operation.data, constants.data_length,
        pipeline->constant_byte_length);
  }
  if (pipeline->bda.binding_count_known &&
      binding_count != pipeline->binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s provides %" PRIhsz
                            " bindings but BDA pipeline expects %u",
                            (int)operation.size, operation.data, binding_count,
                            pipeline->binding_count);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_executable_t
//===----------------------------------------------------------------------===//

static iree_atomic_int64_t iree_hal_vulkan_executable_next_profile_id =
    IREE_ATOMIC_VAR_INIT(1);

typedef struct iree_hal_vulkan_executable_t {
  // Common executable state.
  iree_hal_executable_t base;

  // Host allocator used for executable lifetime.
  iree_allocator_t host_allocator;

  // Borrowed logical-device dispatch table.
  const iree_hal_vulkan_device_syms_t* syms;

  // Borrowed logical-device handle.
  VkDevice logical_device;

  // Process-local nonzero executable identifier used by profiling sessions.
  uint64_t profile_id;

  // Vulkan pipeline layout handle owned by the executable.
  VkPipelineLayout pipeline_layout;

  // Number of prepared compute pipelines in |pipelines|.
  iree_host_size_t pipeline_count;

  // Prepared compute pipelines and export metadata.
  iree_hal_vulkan_pipeline_t* pipelines;

  // Per-export offsets into |parameters|, with one terminal sentinel.
  iree_host_size_t* parameter_offsets;

  // Compact logical parameter descriptors retained outside hot pipelines.
  iree_hal_vulkan_spirv_bda_parameter_t* parameters;

  // Storage backing all exported name string views.
  char* name_storage;
} iree_hal_vulkan_executable_t;

static const iree_hal_executable_vtable_t iree_hal_vulkan_executable_vtable;

static iree_hal_vulkan_executable_t* iree_hal_vulkan_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_executable_vtable);
  return (iree_hal_vulkan_executable_t*)base_value;
}

bool iree_hal_vulkan_executable_isa(iree_hal_executable_t* executable) {
  return iree_hal_resource_is((const iree_hal_resource_t*)executable,
                              &iree_hal_vulkan_executable_vtable);
}

uint64_t iree_hal_vulkan_executable_profile_id(
    iree_hal_executable_t* executable) {
  iree_hal_vulkan_executable_t* vulkan_executable =
      iree_hal_vulkan_executable_cast(executable);
  return vulkan_executable->profile_id;
}

static iree_status_t iree_hal_vulkan_allocate_executable(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_queue_family_t* queue_family,
    iree_host_size_t pipeline_count, iree_host_size_t name_storage_size,
    iree_host_size_t parameter_count, iree_allocator_t host_allocator,
    iree_hal_vulkan_executable_t** out_executable) {
  iree_hal_vulkan_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable));
  memset(executable, 0, sizeof(*executable));
  iree_hal_executable_initialize(
      queue_family, &iree_hal_vulkan_executable_vtable, &executable->base);
  executable->host_allocator = host_allocator;
  executable->syms = syms;
  executable->logical_device = logical_device;
  executable->profile_id = (uint64_t)iree_atomic_fetch_add(
      &iree_hal_vulkan_executable_next_profile_id, 1,
      iree_memory_order_relaxed);
  executable->pipeline_count = pipeline_count;

  iree_status_t status = iree_ok_status();
  if (pipeline_count > 0) {
    status = iree_allocator_malloc_array(host_allocator, pipeline_count,
                                         sizeof(executable->pipelines[0]),
                                         (void**)&executable->pipelines);
    if (iree_status_is_ok(status)) {
      memset(executable->pipelines, 0,
             pipeline_count * sizeof(executable->pipelines[0]));
    }
  }
  if (iree_status_is_ok(status) && name_storage_size > 0) {
    status = iree_allocator_malloc(host_allocator, name_storage_size,
                                   (void**)&executable->name_storage);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc_array(host_allocator, pipeline_count + 1,
                                    sizeof(executable->parameter_offsets[0]),
                                    (void**)&executable->parameter_offsets);
  }
  if (iree_status_is_ok(status) && parameter_count > 0) {
    status = iree_allocator_malloc_array(host_allocator, parameter_count,
                                         sizeof(executable->parameters[0]),
                                         (void**)&executable->parameters);
  }

  if (iree_status_is_ok(status)) {
    *out_executable = executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }
  return status;
}

static bool iree_hal_vulkan_requires_subgroup_size_control(
    uint32_t spirv_version, uint32_t subgroup_size,
    uint32_t default_subgroup_size) {
  // Without ALLOW_VARYING_SUBGROUP_SIZE, pre-1.6 modules execute at the device
  // default width. SPIR-V 1.6 permits varying widths unless explicitly fixed.
  return subgroup_size != 0 && (spirv_version >= 0x00010600u ||
                                subgroup_size != default_subgroup_size);
}

static iree_status_t iree_hal_vulkan_validate_subgroup_size(
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features,
    const iree_hal_vulkan_spirv_compute_entry_point_t* entry_point,
    uint32_t spirv_version, uint32_t subgroup_size) {
  if (!iree_hal_vulkan_requires_subgroup_size_control(
          spirv_version, subgroup_size,
          physical_device->subgroup_properties.subgroupSize)) {
    return iree_ok_status();
  }
  const VkPhysicalDeviceSubgroupSizeControlProperties* properties =
      &physical_device->subgroup_size_control_properties;
  if (!iree_all_bits_set(
          enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SUBGROUP_SIZE_CONTROL) ||
      !iree_all_bits_set(properties->requiredSubgroupSizeStages,
                         VK_SHADER_STAGE_COMPUTE_BIT)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V entry point '%.*s' requires subgroup size %u, but compute "
        "subgroup-size control is not enabled (device default %u)",
        (int)entry_point->name.size, entry_point->name.data, subgroup_size,
        physical_device->subgroup_properties.subgroupSize);
  }
  if (subgroup_size < properties->minSubgroupSize ||
      subgroup_size > properties->maxSubgroupSize) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V entry point '%.*s' requires subgroup size %u outside device "
        "range [%u, %u]",
        (int)entry_point->name.size, entry_point->name.data, subgroup_size,
        properties->minSubgroupSize, properties->maxSubgroupSize);
  }
  if (entry_point->workgroup_size[0] == 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "SPIR-V entry point '%.*s' requires a static LocalSize to validate "
        "subgroup size %u",
        (int)entry_point->name.size, entry_point->name.data, subgroup_size);
  }
  const uint64_t workgroup_xy =
      (uint64_t)entry_point->workgroup_size[0] * entry_point->workgroup_size[1];
  const uint64_t maximum_invocations =
      (uint64_t)subgroup_size * properties->maxComputeWorkgroupSubgroups;
  if (workgroup_xy > maximum_invocations / entry_point->workgroup_size[2]) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V entry point '%.*s' workgroup [%u, %u, %u] exceeds %u "
        "subgroups of size %u",
        (int)entry_point->name.size, entry_point->name.data,
        entry_point->workgroup_size[0], entry_point->workgroup_size[1],
        entry_point->workgroup_size[2],
        properties->maxComputeWorkgroupSubgroups, subgroup_size);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_create_bda_executable(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features, VkPipelineCache pipeline_cache,
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (load_params->executable_data.data_length % sizeof(uint32_t) != 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "Vulkan SPIR-V executable byte length must be a "
                             "multiple of 4"));
  }

  const iree_host_size_t spirv_word_count =
      load_params->executable_data.data_length / sizeof(uint32_t);
  const uint32_t* spirv_words =
      (const uint32_t*)load_params->executable_data.data;
  uint32_t* aligned_spirv_words = NULL;
  if (load_params->executable_data.data_length != 0 &&
      !iree_host_ptr_has_alignment(spirv_words, sizeof(uint32_t))) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc(host_allocator,
                                  load_params->executable_data.data_length,
                                  (void**)&aligned_spirv_words));
    memcpy(aligned_spirv_words, load_params->executable_data.data,
           load_params->executable_data.data_length);
    spirv_words = aligned_spirv_words;
  }

  iree_hal_vulkan_spirv_module_analysis_t analysis = {0};
  iree_status_t status = iree_hal_vulkan_spirv_analyze_module(
      spirv_words, spirv_word_count, &analysis);
  iree_host_size_t entry_point_count = 0;
  iree_host_size_t name_storage_size = 0;
  if (iree_status_is_ok(status)) {
    entry_point_count = analysis.compute_entry_point_count;
    name_storage_size = analysis.compute_entry_point_name_storage_size;
    if (entry_point_count == 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Vulkan SPIR-V executable has no compute entry "
                                "points");
    }
  }

  iree_hal_vulkan_spirv_compute_entry_point_t* entry_points = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, entry_point_count,
                                         sizeof(entry_points[0]),
                                         (void**)&entry_points);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_spirv_parse_compute_entry_points(
        spirv_words, spirv_word_count, entry_point_count, entry_points);
  }

  iree_hal_vulkan_spirv_bda_dispatch_metadata_t* bda_metadata = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, entry_point_count,
                                         sizeof(bda_metadata[0]),
                                         (void**)&bda_metadata);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_spirv_parse_bda_dispatch_metadata(
        spirv_words, spirv_word_count, entry_point_count, entry_points,
        host_allocator, bda_metadata);
  }
  uint32_t push_constant_range_end = 0;
  iree_host_size_t parameter_count = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < entry_point_count; ++i) {
    const iree_hal_vulkan_spirv_bda_dispatch_metadata_t* metadata =
        &bda_metadata[i];
    if (!metadata->is_present) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Vulkan SPIR-V entry point '%.*s' must declare "
                                "iree.vulkan.bda.v1 metadata",
                                (int)entry_points[i].name.size,
                                entry_points[i].name.data);
    } else {
      // The physical layout covers every entry; logical dispatch arguments
      // stay exact in each pipeline's own metadata.
      push_constant_range_end = iree_max(
          push_constant_range_end, metadata->root_push_constant_offset +
                                       metadata->root_push_constant_length);
      if (metadata->constant_byte_length != 0) {
        push_constant_range_end = iree_max(
            push_constant_range_end, metadata->constant_push_constant_offset +
                                         metadata->constant_byte_length);
      }
      if (!iree_host_size_checked_add(
              parameter_count, metadata->parameter_count, &parameter_count)) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "Vulkan SPIR-V logical parameter table size overflows");
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_vulkan_validate_subgroup_size(
            physical_device, enabled_features, &entry_points[i],
            analysis.version, metadata->subgroup_size);
      }
    }
  }

  iree_hal_vulkan_spirv_bda_verification_flags_t verification_flags =
      IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_REQUIRE_PUSH_CONSTANT_ROOT;
  if (iree_all_bits_set(load_params->flags,
                        IREE_HAL_EXECUTABLE_LOAD_FLAG_DISABLE_VERIFICATION)) {
    verification_flags = IREE_HAL_VULKAN_SPIRV_BDA_VERIFICATION_FLAG_NONE;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_spirv_verify_bda_module_analysis(
        &analysis, spirv_words, spirv_word_count, verification_flags);
  }

  iree_hal_vulkan_executable_t* executable = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_allocate_executable(
        syms, logical_device, queue_family, entry_point_count,
        name_storage_size, parameter_count, host_allocator, &executable);
  }

  VkShaderModule shader_module = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = load_params->executable_data.data_length,
        .pCode = spirv_words,
    };
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_vkCreateShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                                       &create_info,
                                       /*pAllocator=*/NULL, &shader_module);
    IREE_LEAK_CHECK_DISABLE_POP();
  }

  if (iree_status_is_ok(status)) {
    VkPushConstantRange root_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_range_end,
    };
    VkPipelineLayoutCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &root_range,
    };
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_vkCreatePipelineLayout(
        IREE_VULKAN_DEVICE(syms), logical_device, &create_info,
        /*pAllocator=*/NULL, &executable->pipeline_layout);
    IREE_LEAK_CHECK_DISABLE_POP();
  }

  VkSpecializationInfo specialization_info;
  VkSpecializationMapEntry* specialization_map_entries = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_create_specialization_info(
        load_params, host_allocator, &specialization_info,
        &specialization_map_entries);
  }

  char* name_storage = executable ? executable->name_storage : NULL;
  iree_host_size_t name_storage_offset = 0;
  iree_host_size_t parameter_offset = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < entry_point_count; ++i) {
    iree_hal_vulkan_pipeline_t* pipeline = &executable->pipelines[i];
    const iree_string_view_t entry_point = entry_points[i].name;
    memcpy(name_storage + name_storage_offset, entry_point.data,
           entry_point.size);
    name_storage[name_storage_offset + entry_point.size] = 0;
    pipeline->name = iree_make_string_view(name_storage + name_storage_offset,
                                           entry_point.size);
    name_storage_offset += entry_point.size + /*NUL=*/1;

    pipeline->handle = VK_NULL_HANDLE;
    pipeline->layout = executable->pipeline_layout;
    pipeline->subgroup_size = bda_metadata[i].subgroup_size;
    status = iree_hal_vulkan_initialize_pipeline_bda_metadata(
        &bda_metadata[i], host_allocator, pipeline);
    if (!iree_status_is_ok(status)) {
      break;
    }
    memcpy(pipeline->workgroup_size, entry_points[i].workgroup_size,
           sizeof(pipeline->workgroup_size));
    executable->parameter_offsets[i] = parameter_offset;
    if (bda_metadata[i].parameter_count != 0) {
      memcpy(
          &executable->parameters[parameter_offset], bda_metadata[i].parameters,
          bda_metadata[i].parameter_count * sizeof(executable->parameters[0]));
      parameter_offset += bda_metadata[i].parameter_count;
    }

    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_info = {
        .sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = pipeline->subgroup_size,
    };
    const bool needs_subgroup_control =
        iree_hal_vulkan_requires_subgroup_size_control(
            analysis.version, pipeline->subgroup_size,
            physical_device->subgroup_properties.subgroupSize);
    const bool requires_full_subgroups =
        pipeline->subgroup_size != 0 && pipeline->workgroup_size[0] != 0 &&
        pipeline->workgroup_size[0] % pipeline->subgroup_size == 0;
    VkPipelineShaderStageCreateInfo stage_create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = needs_subgroup_control ? &subgroup_info : NULL,
        .flags =
            requires_full_subgroups
                ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT
                : 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = pipeline->name.data,
        .pSpecializationInfo = &specialization_info,
    };
    VkComputePipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_create_info,
        .layout = pipeline->layout,
    };
    if (!iree_all_bits_set(load_params->flags,
                           IREE_HAL_EXECUTABLE_LOAD_FLAG_ALLOW_OPTIMIZATION)) {
      create_info.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
    }
    IREE_LEAK_CHECK_DISABLE_PUSH();
    status = iree_vkCreateComputePipelines(
        IREE_VULKAN_DEVICE(syms), logical_device, pipeline_cache,
        /*createInfoCount=*/1, &create_info, /*pAllocator=*/NULL,
        &pipeline->handle);
    IREE_LEAK_CHECK_DISABLE_POP();
    if (!iree_status_is_ok(status)) {
      status =
          iree_status_annotate_f(status, "SPIR-V BDA entry point '%.*s'",
                                 (int)pipeline->name.size, pipeline->name.data);
    }
  }
  if (executable) {
    executable->parameter_offsets[entry_point_count] = parameter_offset;
  }

  iree_allocator_free(host_allocator, specialization_map_entries);
  if (shader_module) {
    iree_vkDestroyShaderModule(IREE_VULKAN_DEVICE(syms), logical_device,
                               shader_module, /*pAllocator=*/NULL);
  }
  if (bda_metadata) {
    for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
      iree_hal_vulkan_spirv_bda_dispatch_metadata_deinitialize(&bda_metadata[i],
                                                               host_allocator);
    }
    iree_allocator_free(host_allocator, bda_metadata);
  }
  iree_allocator_free(host_allocator, entry_points);
  iree_allocator_free(host_allocator, aligned_spirv_words);

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else if (executable) {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_executable_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_features_t enabled_features, VkPipelineCache pipeline_cache,
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  return iree_hal_vulkan_create_bda_executable(
      syms, logical_device, physical_device, enabled_features, pipeline_cache,
      queue_family, load_params, host_allocator, out_executable);
}

static void iree_hal_vulkan_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    if (executable->pipelines[i].handle) {
      iree_vkDestroyPipeline(IREE_VULKAN_DEVICE(executable->syms),
                             executable->logical_device,
                             executable->pipelines[i].handle,
                             /*pAllocator=*/NULL);
    }
    iree_allocator_free(host_allocator,
                        executable->pipelines[i].bda.binding_requirements);
  }
  if (executable->pipeline_layout) {
    iree_vkDestroyPipelineLayout(IREE_VULKAN_DEVICE(executable->syms),
                                 executable->logical_device,
                                 executable->pipeline_layout,
                                 /*pAllocator=*/NULL);
  }
  iree_allocator_free(host_allocator, executable->name_storage);
  iree_allocator_free(host_allocator, executable->parameters);
  iree_allocator_free(host_allocator, executable->parameter_offsets);
  iree_allocator_free(host_allocator, executable->pipelines);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_vulkan_executable_lookup_pipeline(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    const iree_hal_vulkan_pipeline_t** out_pipeline) {
  IREE_ASSERT_ARGUMENT(out_pipeline);
  *out_pipeline = NULL;
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->pipeline_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "function id %" PRIu64
                            " out of range; executable has %" PRIhsz " exports",
                            function.value, executable->pipeline_count);
  }
  const uint32_t export_ordinal = iree_hal_executable_function_index(function);
  *out_pipeline = &executable->pipelines[export_ordinal];
  return iree_ok_status();
}

static iree_host_size_t iree_hal_vulkan_executable_export_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  return executable->pipeline_count;
}

static iree_status_t iree_hal_vulkan_executable_export_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal,
    iree_hal_executable_function_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      base_executable, export_ordinal, &pipeline));
  out_info->name = pipeline->name;
  out_info->constant_byte_length = pipeline->constant_byte_length;
  out_info->binding_count = pipeline->binding_count;
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  const uint32_t ordinal = iree_hal_executable_function_index(export_ordinal);
  out_info->parameter_count =
      (uint16_t)(executable->parameter_offsets[ordinal + 1] -
                 executable->parameter_offsets[ordinal]);
  memcpy(out_info->workgroup_size, pipeline->workgroup_size,
         sizeof(out_info->workgroup_size));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_executable_export_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t export_ordinal, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  IREE_ASSERT_ARGUMENT(out_parameters || capacity == 0);
  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      base_executable, export_ordinal, &pipeline));
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  const uint32_t ordinal = iree_hal_executable_function_index(export_ordinal);
  const iree_host_size_t parameter_begin =
      executable->parameter_offsets[ordinal];
  const iree_host_size_t parameter_end =
      executable->parameter_offsets[ordinal + 1];
  const iree_host_size_t copy_count =
      iree_min(capacity, parameter_end - parameter_begin);
  for (iree_host_size_t i = 0; i < copy_count; ++i) {
    const iree_hal_vulkan_spirv_bda_parameter_t* source =
        &executable->parameters[parameter_begin + i];
    iree_hal_executable_function_parameter_t* target = &out_parameters[i];
    *target = (iree_hal_executable_function_parameter_t){
        .type = source->type,
        .size = source->size,
        .offset = source->offset,
    };
    if (source->type == IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT) {
      target->flags =
          IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET;
      target->native_abi_offset =
          (uint16_t)(pipeline->bda.constant_push_constant_offset +
                     source->offset);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_executable_lookup_export_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_export_ordinal) {
  iree_hal_vulkan_executable_t* executable =
      iree_hal_vulkan_executable_cast(base_executable);
  for (iree_host_size_t i = 0; i < executable->pipeline_count; ++i) {
    if (iree_string_view_equal(executable->pipelines[i].name, name)) {
      *out_export_ordinal =
          iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "export '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_vulkan_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  (void)base_executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  (void)base_executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid Vulkan executable global");
}

static iree_status_t iree_hal_vulkan_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_buffer_t** out_buffer) {
  (void)base_executable;
  (void)global;
  (void)out_buffer;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid Vulkan executable global");
}

static const iree_hal_executable_vtable_t iree_hal_vulkan_executable_vtable = {
    .destroy = iree_hal_vulkan_executable_destroy,
    .function_count = iree_hal_vulkan_executable_export_count,
    .function_info = iree_hal_vulkan_executable_export_info,
    .function_parameters = iree_hal_vulkan_executable_export_parameters,
    .lookup_function_by_name = iree_hal_vulkan_executable_lookup_export_by_name,
    .try_lookup_global_by_name =
        iree_hal_vulkan_executable_try_lookup_global_by_name,
    .global_info = iree_hal_vulkan_executable_global_info,
    .global_buffer = iree_hal_vulkan_executable_global_buffer,
};
