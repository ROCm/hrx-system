// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

// The relocation-free program in dispatch_info.loom uses two pointer kernargs,
// no scratch or LDS, one workitem, and exactly three output DWORDs. These ISA
// variants keep runtime-only builds independent of a compiler toolchain.
// dispatch_program_test recompiles that source for all supported families and
// compares its instruction bytes and complete resource/argument ABI here.
static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx11[] = {
    0xf4080100u, 0xf8000000u, 0x7e000280u, 0xbf89fc07u, 0xdc520000u,
    0x01040000u, 0xdc520004u, 0x02040000u, 0xdc520008u, 0x03040000u,
    0xdc520006u, 0x04060000u, 0xdc4e0004u, 0x05060000u, 0xbf8907f7u,
    0xd6110006u, 0x02410104u, 0x34080890u, 0xbf8903f7u, 0x360a0affu,
    0x0000ffffu, 0xbf870193u, 0x360c0cffu, 0x0000ffffu, 0x360808ffu,
    0x0000ffffu, 0xbf870193u, 0xd72c0001u, 0x02020b01u, 0xd72c0002u,
    0x02020d02u, 0xbf870003u, 0xd72c0003u, 0x02020903u, 0xdc6a000cu,
    0x00060100u, 0xdc6a0010u, 0x00060200u, 0xdc6a0014u, 0x00060300u,
    0xbc7c0000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx12[] = {
    0xf4004100u, 0xf8000000u, 0x7e000280u, 0xbfc70000u, 0xee050004u,
    0x00000001u, 0x00000000u, 0xee050004u, 0x00000002u, 0x00000400u,
    0xee050004u, 0x00000003u, 0x00000800u, 0xee050006u, 0x00000004u,
    0x00000600u, 0xee04c006u, 0x00000005u, 0x00000400u, 0xbfc00001u,
    0xd6110006u, 0x02410104u, 0x34080890u, 0xbfc00000u, 0x360a0affu,
    0x0000ffffu, 0xbf870193u, 0x360c0cffu, 0x0000ffffu, 0x360808ffu,
    0x0000ffffu, 0xbf870193u, 0xd72c0001u, 0x02020b01u, 0xd72c0002u,
    0x02020d02u, 0xbf870003u, 0xd72c0003u, 0x02020903u, 0xee068006u,
    0x00800000u, 0x00000c00u, 0xee068006u, 0x01000000u, 0x00001000u,
    0xee068006u, 0x01800000u, 0x00001400u, 0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4004100u, 0xf8000000u, 0x7e000280u, 0xbfc70000u,
    0xee050004u, 0x00000001u, 0x00000000u, 0xee050004u, 0x00000002u,
    0x00000400u, 0xee050004u, 0x00000003u, 0x00000800u, 0xee050006u,
    0x00000004u, 0x00000600u, 0xee04c006u, 0x00000005u, 0x00000400u,
    0xbfc00001u, 0xd6110006u, 0x02410104u, 0x34080890u, 0xbfc00000u,
    0x360a0affu, 0x0000ffffu, 0xbf870193u, 0x360c0cffu, 0x0000ffffu,
    0x360808ffu, 0x0000ffffu, 0xbf870193u, 0xd72c0001u, 0x02020b01u,
    0xd72c0002u, 0x02020d02u, 0xbf870003u, 0xd72c0003u, 0x02020903u,
    0xee068006u, 0x00800000u, 0x00000c00u, 0xee068006u, 0x01000000u,
    0x00001000u, 0xee068006u, 0x01800000u, 0x00001400u, 0xbfc10000u,
    0xbfb00000u,
};

iree_status_t iree_hal_amd_gpu_pm4_dispatch_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    iree_hal_amd_gpu_pm4_dispatch_program_t* out_program) {
  iree_hal_amd_gpu_pm4_dispatch_program_t program = {
      .kernel = {.compute_pgm_rsrc2 = 4, .code_properties = 0x408},
  };
  if (gfxip.major == 11 &&
      (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7)) {
    program.code = iree_make_const_byte_span(
        iree_hal_amd_gpu_pm4_dispatch_program_gfx11,
        sizeof(iree_hal_amd_gpu_pm4_dispatch_program_gfx11));
    program.kernel.compute_pgm_rsrc1 = 0xe0af0000u;
  } else if (gfxip.major == 12 && gfxip.minor == 0) {
    program.code = iree_make_const_byte_span(
        iree_hal_amd_gpu_pm4_dispatch_program_gfx12,
        sizeof(iree_hal_amd_gpu_pm4_dispatch_program_gfx12));
    program.kernel.compute_pgm_rsrc1 = 0xe00f0000u;
  } else if (gfxip.major == 12 && gfxip.minor == 5) {
    program.code = iree_make_const_byte_span(
        iree_hal_amd_gpu_pm4_dispatch_program_gfx125,
        sizeof(iree_hal_amd_gpu_pm4_dispatch_program_gfx125));
    program.kernel.compute_pgm_rsrc1 = 0xc00f0000u;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "dispatch packet preparation is unavailable for "
                            "gfx%u.%u.%u",
                            gfxip.major, gfxip.minor, gfxip.stepping);
  }
  *out_program = program;
  return iree_ok_status();
}

iree_hal_amd_gpu_pm4_dispatch_storage_t iree_hal_amd_gpu_pm4_dispatch_storage(
    const iree_hal_amd_gpu_pm4_function_t* function,
    iree_hal_dispatch_flags_t flags) {
  const bool indirect = iree_hal_dispatch_uses_indirect_arguments(flags);
  iree_hal_amd_gpu_pm4_dispatch_storage_t storage = {
      .length = indirect ? 0 : function->kernarg->byte_length,
      .alignment = indirect ? 1 : function->kernarg->alignment,
  };
  if (function->packet) {
    storage.alignment = iree_max(storage.alignment, 64u);
    storage.packet_offset = (storage.length + 63u) & ~63u;
    storage.length = storage.packet_offset + 64u;
    if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
      storage.length += 16u;
    }
  }
  return storage;
}

iree_status_t iree_hal_amd_gpu_pm4_dispatch_prepare(
    const iree_hal_amd_gpu_pm4_function_t* function,
    uint32_t maximum_group_segment_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch) {
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  const uint16_t launch_requirements =
      function->launch.requirements &
      ~IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DISPATCH_POINTER;
  if (launch_requirements || function->kernarg->requirements) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch requires native launch services 0x%x and argument services "
        "0x%x that this queue has not provisioned",
        launch_requirements, function->kernarg->requirements);
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
            (function->packet ? IREE_HAL_AMD_GPU_PM4_DISPATCH_PREPARE_WORDS
                              : 0u) +
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
      if (function->packet && dispatch.workgroup_count[i] > UINT32_MAX / size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "dispatch packet grid dimension %d exceeds "
                                "32-bit workitem counts",
                                i);
      }
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
    const iree_hal_buffer_ref_t reference = bindings.values[i];
    if (!reference.buffer) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct dispatch binding %" PRIhsz " is null", i);
    }
    // The remaining range is valid whenever its starting offset is valid.
    // Native arguments consume only the address, so no length is materialized.
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
        reference.buffer, reference.offset,
        reference.length == IREE_HAL_WHOLE_BUFFER ? 0 : reference.length));
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
  if (!iree_hal_dispatch_uses_indirect_arguments(flags)) {
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
      if (dispatch->workgroup_count[i] != 1 ||
          dispatch->workgroup_size[i] != 1) {
        inputs.grid_dimensions = (uint16_t)(i + 1);
      }
    }
    if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
      // HAL supplies a three-axis tuple without a separate rank. Device copies
      // populate its counts; unused dimensions have size and count one.
      inputs.grid_dimensions = 3;
    }
    if (iree_any_bit_set(flags,
                         IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
      iree_hal_amd_gpu_kernarg_layout_emplace_native(
          function->kernarg, constants, &inputs, target);
    } else {
      iree_hal_amd_gpu_kernarg_layout_emplace(function->kernarg, constants.data,
                                              binding_addresses, &inputs,
                                              target);
    }
  }
  if (function->packet) {
    const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
        iree_hal_amd_gpu_pm4_dispatch_storage(function, flags);
    const uint32_t kernarg_length =
        iree_hal_dispatch_uses_indirect_arguments(flags)
            ? 0
            : function->kernarg->byte_length;
    memset((uint8_t*)target + kernarg_length, 0,
           storage.packet_offset - kernarg_length);
    if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
      memset((uint8_t*)target + storage.packet_offset + 64, 0, 16);
    }
    iree_hal_amd_gpu_pm4_dispatch_emplace_packet(
        function, dispatch, flags,
        (iree_hal_amd_gpu_pm4_dispatch_packet_t*)((uint8_t*)target +
                                                  storage.packet_offset));
  }
}

void iree_hal_amd_gpu_pm4_dispatch_emplace_packet(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_packet_t* target) {
  *target = (iree_hal_amd_gpu_pm4_dispatch_packet_t){
      .header = 2,
      .setup = 1,
      .private_segment_size = function->metadata->private_segment_fixed_size,
      .group_segment_size = dispatch->group_segment_byte_length,
      .kernel_object = function->packet->kernel_object,
  };
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = dispatch->workgroup_size[i];
    const uint32_t partial = dispatch->partial_size[i];
    target->workgroup_size[i] = (uint16_t)size;
    target->grid_size[i] =
        dispatch->workgroup_count[i] * size - (partial ? size - partial : 0);
    if (size != 1 || dispatch->workgroup_count[i] != 1) {
      target->setup = (uint16_t)(i + 1);
    }
  }
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    target->setup = 3;
  }
}

uint32_t* iree_hal_amd_gpu_pm4_dispatch_emit_prepare(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* prepare,
    uint64_t argument_address, uint32_t cache_control) {
  const uint32_t one[3] = {1, 1, 1};
  const uint32_t zero[3] = {0};
  const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
      .kernarg_pointer = argument_address,
  };
  words =
      iree_hal_amd_gpu_pm4_launch_emit(words, prepare, one, zero, 0, &inputs);
  words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
      words, one, prepare->dispatch_initiator);
  words = iree_hal_amd_gpu_pm4_emit_compute_flush(words);
  return iree_hal_amd_gpu_pm4_emit_acquire(words, cache_control);
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
