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
    0xf4080100u, 0xf8000000u, 0x7e000280u, 0xbf89fc07u, 0xdc520006u,
    0x01060000u, 0xdc4e0004u, 0x02060000u, 0xdc520000u, 0x03040000u,
    0xdc520004u, 0x04040000u, 0xdc520008u, 0x05040000u, 0xbf8913f7u,
    0xd6110006u, 0x02410101u, 0x34020290u, 0xbf890ff7u, 0x360404ffu,
    0x0000ffffu, 0xbf870193u, 0x360c0cffu, 0x0000ffffu, 0x360202ffu,
    0x0000ffffu, 0xbf890bf7u, 0xbf8701a3u, 0xd72c0002u, 0x02020503u,
    0xbf8907f7u, 0xd72c0003u, 0x02020d04u, 0xbf8903f7u, 0xbf870003u,
    0xd72c0001u, 0x02020305u, 0xdc6a000cu, 0x00060200u, 0xdc6a0010u,
    0x00060300u, 0xdc6a0014u, 0x00060100u, 0xbc7c0000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx12[] = {
    0xf4004100u, 0xf8000000u, 0x7e000280u, 0xbfc70000u, 0xee050006u,
    0x00000001u, 0x00000600u, 0xee04c006u, 0x00000002u, 0x00000400u,
    0xee050004u, 0x00000003u, 0x00000000u, 0xee050004u, 0x00000004u,
    0x00000400u, 0xee050004u, 0x00000005u, 0x00000800u, 0xbfc00004u,
    0xd6110006u, 0x02410101u, 0x34020290u, 0xbfc00003u, 0x360404ffu,
    0x0000ffffu, 0xbf870193u, 0x360c0cffu, 0x0000ffffu, 0x360202ffu,
    0x0000ffffu, 0xbfc00002u, 0xbf8701a3u, 0xd72c0002u, 0x02020503u,
    0xbfc00001u, 0xd72c0003u, 0x02020d04u, 0xbfc00000u, 0xbf870003u,
    0xd72c0001u, 0x02020305u, 0xee068006u, 0x01000000u, 0x00000c00u,
    0xee068006u, 0x01800000u, 0x00001000u, 0xee068006u, 0x00800000u,
    0x00001400u, 0xbfc10000u, 0xbfb00000u,
};

static const uint32_t iree_hal_amd_gpu_pm4_dispatch_program_gfx125[] = {
    0xee174000u, 0x00040000u, 0x00000000u, 0x7e000000u, 0xb9800641u,
    0x00000001u, 0xf4004100u, 0xf8000000u, 0x7e000280u, 0xbfc70000u,
    0xee050006u, 0x00000001u, 0x00000600u, 0xee04c006u, 0x00000002u,
    0x00000400u, 0xee050004u, 0x00000003u, 0x00000000u, 0xee050004u,
    0x00000004u, 0x00000400u, 0xee050004u, 0x00000005u, 0x00000800u,
    0xbfc00004u, 0xd6110006u, 0x02410101u, 0x34020290u, 0xbfc00003u,
    0x360404ffu, 0x0000ffffu, 0xbf870193u, 0x360c0cffu, 0x0000ffffu,
    0x360202ffu, 0x0000ffffu, 0xbfc00002u, 0xbf8701a3u, 0xd72c0002u,
    0x02020503u, 0xbfc00001u, 0xd72c0003u, 0x02020d04u, 0xbfc00000u,
    0xbf870003u, 0xd72c0001u, 0x02020305u, 0xee068006u, 0x01000000u,
    0x00000c00u, 0xee068006u, 0x01800000u, 0x00001000u, 0xee068006u,
    0x00800000u, 0x00001400u, 0xbfc10000u, 0xbfb00000u,
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
      .length = indirect ? 0 : function->base.kernarg->byte_length,
      .alignment = indirect ? 1 : function->base.kernarg->alignment,
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
    uint32_t maximum_scratch_wave_byte_length,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_pm4_dispatch_t* out_dispatch) {
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->base.metadata;
  uint16_t launch_requirements =
      function->launch.requirements &
      ~(IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DISPATCH_POINTER |
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_TSAN);
  uint32_t scratch_wave_byte_length = 0;
  if (maximum_scratch_wave_byte_length &&
      (launch_requirements & IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH)) {
    const uint64_t required =
        iree_align_uint64((uint64_t)metadata->private_segment_fixed_size
                              << function->launch.wavefront_size_log2,
                          256);
    if (!required || required > maximum_scratch_wave_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kernel private storage requires %" PRIu64
                              " bytes per wave; native capacity is %u",
                              required, maximum_scratch_wave_byte_length);
    }
    scratch_wave_byte_length = (uint32_t)required;
    launch_requirements &= ~IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH;
  }
  if (launch_requirements || function->base.kernarg->requirements) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel '%.*s' requires native launch services 0x%x and argument "
        "services 0x%x that this queue has not provisioned "
        "(fixed private segment: %u bytes per workitem)",
        (int)metadata->reflection_name.size, metadata->reflection_name.data,
        launch_requirements, function->base.kernarg->requirements,
        metadata->private_segment_fixed_size);
  }
  if (metadata->has_workgroup_cluster_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 workgroup clusters require native queue "
                            "admission and cluster launch state");
  }
  if (indirect_parameters) {
    const uint16_t copy_count =
        iree_hal_dispatch_uses_indirect_arguments(flags)
            ? 0
            : function->base.kernarg->indirect_copy_count;
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
  iree_hal_amd_gpu_pm4_dispatch_t dispatch = {.scratch_wave_byte_length =
                                                  scratch_wave_byte_length};
  IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_dispatch_prepare(
      &function->base, maximum_group_segment_byte_length,
      function->packet != NULL, config, constants, binding_count, flags,
      &dispatch.geometry));
  *out_dispatch = dispatch;
  return iree_ok_status();
}

void iree_hal_amd_gpu_pm4_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_pm4_function_t* function,
    const iree_hal_amd_gpu_pm4_dispatch_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags, void* target) {
  iree_hal_amd_gpu_dispatch_emplace_arguments(&function->base,
                                              &dispatch->geometry, constants,
                                              binding_addresses, flags, target);
  if (function->packet) {
    const iree_hal_amd_gpu_pm4_dispatch_storage_t storage =
        iree_hal_amd_gpu_pm4_dispatch_storage(function, flags);
    const uint32_t kernarg_length =
        iree_hal_dispatch_uses_indirect_arguments(flags)
            ? 0
            : function->base.kernarg->byte_length;
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
      .private_segment_size =
          function->base.metadata->private_segment_fixed_size,
      .group_segment_size = dispatch->geometry.group_segment_byte_length,
      .kernel_object = function->packet->kernel_object,
  };
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = dispatch->geometry.workgroup_size[i];
    const uint32_t partial = dispatch->geometry.partial_size[i];
    target->workgroup_size[i] = (uint16_t)size;
    target->grid_size[i] = dispatch->geometry.workgroup_count[i] * size -
                           (partial ? size - partial : 0);
    if (size != 1 || dispatch->geometry.workgroup_count[i] != 1) {
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
