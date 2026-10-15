// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

iree_status_t iree_hal_amd_gpu_pm4_dispatch_prepare(
    const iree_hal_amd_gpu_pm4_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch) {
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  if (function->launch.requirements || function->kernarg->requirements) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch requires native launch services 0x%x and argument services "
        "0x%x that this queue has not provisioned",
        function->launch.requirements, function->kernarg->requirements);
  }
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->metadata;
  if (metadata->has_workgroup_cluster_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 workgroup clusters require native queue "
                            "admission and cluster launch state");
  }
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(flags);
  const bool native_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  if (native_arguments && indirect_arguments) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native direct and indirect arguments conflict");
  }
  if (indirect_arguments) {
    if (binding_count != 1 || constants.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "indirect arguments require one binding and "
                              "no constants");
    }
  } else if (native_arguments) {
    if (binding_count ||
        constants.data_length < function->kernarg->native_byte_length ||
        constants.data_length > function->kernarg->byte_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native argument extent does not match the "
                              "function ABI");
    }
  } else if (binding_count != function->kernarg->binding_count ||
             constants.data_length != function->kernarg->constant_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dense arguments do not match function reflection");
  }
  iree_hal_amd_gpu_pm4_dispatch_t dispatch = {0};
  const bool default_size =
      !(config.workgroup_size[0] | config.workgroup_size[1] |
        config.workgroup_size[2]);
  const bool exact =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT);
  if (exact && indirect_parameters) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact work-item counts conflict with indirect parameters");
  }
  if (indirect_parameters) {
    const uint16_t copy_count =
        indirect_arguments ? 0 : function->kernarg->indirect_copy_count;
    const uint32_t launch_words = function->launch.user_data_count
                                      ? 24 + function->launch.user_data_count
                                      : 22;
    if (launch_words +
            iree_hal_amd_gpu_pm4_dispatch_indirect_word_count(copy_count) >
        IREE_HAL_AMD_GPU_PM4_DISPATCH_MAX_WORDS) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "indirect argument copies exceed the native launch window");
    }
  }
  uint64_t flat_size = 1;
  bool partial = false;
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = default_size
                              ? (metadata->has_required_workgroup_size
                                     ? metadata->required_workgroup_size[i]
                                     : 1)
                              : config.workgroup_size[i];
    if (!size || size > 1024 ||
        (metadata->has_required_workgroup_size &&
         size != metadata->required_workgroup_size[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "workgroup dimension %d violates function or "
                              "PM4 geometry",
                              i);
    }
    flat_size *= size;
    dispatch.workgroup_size[i] = size;
    if (exact) {
      if (!config.workitem_count[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "exact work-item dimensions must be nonzero");
      }
      const uint32_t remainder = config.workitem_count[i] % size;
      dispatch.workgroup_count[i] =
          config.workitem_count[i] / size + (remainder != 0);
      dispatch.partial_size[i] = remainder ? remainder : size;
      partial |= remainder != 0;
    } else {
      if (config.workitem_count[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "work-item dimensions require the exact flag");
      }
      dispatch.workgroup_count[i] =
          indirect_parameters ? 0 : config.workgroup_count[i];
    }
    if (metadata->max_workgroup_count[i] &&
        dispatch.workgroup_count[i] > metadata->max_workgroup_count[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "workgroup count exceeds function metadata");
    }
  }
  if (flat_size > 1024 || (metadata->max_flat_workgroup_size &&
                           flat_size > metadata->max_flat_workgroup_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup exceeds function or PM4 invocation "
                            "capacity");
  }
  if (partial && metadata->uniform_workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function requires complete workgroups");
  }
  if (!partial) {
    memset(dispatch.partial_size, 0, sizeof(dispatch.partial_size));
  }
  const uint64_t group_bytes = (uint64_t)metadata->group_segment_fixed_size +
                               config.dynamic_workgroup_local_memory;
  if (group_bytes > maximum_group_segment_byte_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch LDS exceeds the native queue capacity");
  }
  dispatch.group_segment_byte_length = (uint32_t)group_bytes;
  *out_dispatch = dispatch;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_pm4_dispatch_validate_direct_bindings(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_hal_buffer_ref_t parameters, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    if (!parameters.buffer || (parameters.offset & 3)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "indirect parameters require a DWORD-aligned buffer");
    }
    if (parameters.length != IREE_HAL_WHOLE_BUFFER && parameters.length < 12) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "indirect parameters require three DWORD counts");
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(parameters.buffer,
                                                        parameters.offset, 12));
  }
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(flags);
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (!bindings.values[i].buffer) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct dispatch binding %" PRIhsz " is null", i);
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
        bindings.values[i].buffer, bindings.values[i].offset,
        bindings.values[i].length));
  }
  if (indirect_arguments) {
    const iree_hal_buffer_ref_t reference = bindings.values[0];
    const iree_device_size_t length =
        reference.length == IREE_HAL_WHOLE_BUFFER
            ? iree_hal_buffer_byte_length(reference.buffer) - reference.offset
            : reference.length;
    if (length < layout->byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "indirect arguments do not cover the native ABI");
    }
  }
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags, void* target) {
  iree_hal_amd_gpu_kernarg_inputs_t inputs = {
      .dynamic_lds_size = dispatch->group_segment_byte_length -
                          function->metadata->group_segment_fixed_size,
      .grid_dimensions = 1,
  };
  for (int i = 0; i < 3; ++i) {
    const uint32_t partial = dispatch->partial_size[i];
    const bool remainder = partial && partial != dispatch->workgroup_size[i];
    inputs.block_count[i] = dispatch->workgroup_count[i] - remainder;
    inputs.group_size[i] = (uint16_t)dispatch->workgroup_size[i];
    inputs.remainder[i] = remainder ? (uint16_t)partial : 0;
    if (dispatch->workgroup_count[i] != 1 || dispatch->workgroup_size[i] != 1) {
      inputs.grid_dimensions = (uint16_t)(i + 1);
    }
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    // HAL supplies a three-axis tuple without a separate rank. Device copies
    // populate its counts; unused dimensions have size and count one.
    inputs.grid_dimensions = 3;
  }
  if (iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    iree_hal_amd_gpu_kernarg_layout_emplace_native(function->kernarg, constants,
                                                   &inputs, target);
  } else {
    iree_hal_amd_gpu_kernarg_layout_emplace(function->kernarg, constants.data,
                                            binding_addresses, &inputs, target);
  }
}

uint32_t* iree_hal_amd_gpu_pm4_dispatch_emit_indirect(
    uint32_t* words, const iree_hal_amd_gpu_kernarg_layout_t* layout,
    uint16_t copy_count, uint64_t parameters_address, uint64_t argument_address,
    uint32_t cache_control, uint32_t initiator) {
  const iree_hal_amd_gpu_kernarg_copy_t* copies =
      iree_hal_amd_gpu_kernarg_layout_indirect_copies(layout);
  for (uint16_t i = 0; i < copy_count; ++i) {
    words = iree_hal_amd_gpu_pm4_emit_dma_copy(
        words, parameters_address + copies[i].source_offset,
        argument_address + copies[i].target_offset, copies[i].length);
  }
  if (copy_count) {
    words = iree_hal_amd_gpu_pm4_emit_dma_flush(words);
    words = iree_hal_amd_gpu_pm4_emit_acquire(words, cache_control);
  }
  return iree_hal_amd_gpu_pm4_emit_dispatch_indirect(words, parameters_address,
                                                     initiator);
}
