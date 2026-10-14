// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/launch.h"

#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

// RDNA uses architected flat scratch. The old private-buffer/flat-init SGPR
// fields are invalid there; only these five descriptor inputs are available.
enum {
  IREE_HAL_AMD_GPU_PM4_INPUT_DISPATCH_POINTER = 1u << 1,
  IREE_HAL_AMD_GPU_PM4_INPUT_QUEUE_POINTER = 1u << 2,
  IREE_HAL_AMD_GPU_PM4_INPUT_KERNARG_POINTER = 1u << 3,
  IREE_HAL_AMD_GPU_PM4_INPUT_DISPATCH_ID = 1u << 4,
  IREE_HAL_AMD_GPU_PM4_INPUT_PRIVATE_SEGMENT_SIZE = 1u << 6,
  IREE_HAL_AMD_GPU_PM4_PROPERTY_WAVE32 = 1u << 10,
  IREE_HAL_AMD_GPU_PM4_PROPERTY_DYNAMIC_STACK = 1u << 11,
  IREE_HAL_AMD_GPU_PM4_LDS_MASK = 0x00ff8000u,
  IREE_HAL_AMD_GPU_PM4_USER_DATA_CAPACITY = 16,
};

static uint8_t iree_hal_amd_gpu_pm4_launch_assign_input(uint16_t properties,
                                                        uint16_t mask,
                                                        uint8_t width,
                                                        uint8_t* next) {
  if (!(properties & mask)) {
    return UINT8_MAX;
  }
  uint8_t index = *next;
  *next += width;
  return index;
}

iree_status_t iree_hal_amd_gpu_pm4_launch_initialize(
    iree_hal_amd_gpu_gfxip_version_t gfxip,
    const iree_hal_amd_gpu_image_kernel_t* kernel, uint64_t image_address,
    iree_hal_amd_gpu_pm4_launch_t* out_launch) {
  const bool gfx11 = gfxip.major == 11 &&
                     (gfxip.minor == 0 || gfxip.minor == 5 || gfxip.minor == 7);
  const bool gfx12 = gfxip.major == 12 && gfxip.minor == 0;
  const bool gfx125 = gfxip.major == 12 && gfxip.minor == 5;
  if (!gfx11 && !gfx12 && !gfx125) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "PM4 launch encoding is unavailable for gfx%u.%u.%u", gfxip.major,
        gfxip.minor, gfxip.stepping);
  }
  const uint16_t properties = kernel->code_properties;
  if (properties & ~UINT16_C(0x0c5e)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel requests invalid initial SGPR/property bits for RDNA: 0x%04x",
        (unsigned)(properties & ~UINT16_C(0x0c5e)));
  }
  if (kernel->kernarg_preload) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "RDNA kernel requests a kernarg preload ABI");
  }
  const uint8_t wavefront_size =
      properties & IREE_HAL_AMD_GPU_PM4_PROPERTY_WAVE32 ? 32 : 64;
  if (gfx125 && wavefront_size != 32) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "gfx12.5 requires a wave32 kernel");
  }
  const uint64_t address_limit = UINT64_C(1) << 48;
  if (image_address >= address_limit ||
      kernel->entry_offset >= address_limit - image_address ||
      ((image_address + kernel->entry_offset) & 255u)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "PM4 kernel entry requires a 256-byte-aligned "
                            "address below 2^48");
  }

  const uint64_t entry = image_address + kernel->entry_offset;
  iree_hal_amd_gpu_pm4_launch_t launch = {
      .program = {(uint32_t)(entry >> 8), (uint32_t)(entry >> 40)},
      .resources =
          {
              .resource1 = kernel->compute_pgm_rsrc1,
              .resource2 =
                  kernel->compute_pgm_rsrc2 & ~IREE_HAL_AMD_GPU_PM4_LDS_MASK,
              .resource3 = kernel->compute_pgm_rsrc3,
          },
      .resource3_register = gfx125 ? 0x2e23u : 0x2e28u,
      .dispatch_initiator = 0x5u | (wavefront_size == 32 ? 0x8000u : 0),
      .user_data_count =
          (uint8_t)((kernel->compute_pgm_rsrc2 >> 1) & (gfx125 ? 63u : 31u)),
      .wavefront_size_log2 = wavefront_size == 32 ? 5 : 6,
  };
  uint8_t input_count = 0;
  launch.user_data.dispatch_pointer = iree_hal_amd_gpu_pm4_launch_assign_input(
      properties, IREE_HAL_AMD_GPU_PM4_INPUT_DISPATCH_POINTER, 2, &input_count);
  launch.user_data.queue_pointer = iree_hal_amd_gpu_pm4_launch_assign_input(
      properties, IREE_HAL_AMD_GPU_PM4_INPUT_QUEUE_POINTER, 2, &input_count);
  launch.user_data.kernarg_pointer = iree_hal_amd_gpu_pm4_launch_assign_input(
      properties, IREE_HAL_AMD_GPU_PM4_INPUT_KERNARG_POINTER, 2, &input_count);
  launch.user_data.dispatch_id = iree_hal_amd_gpu_pm4_launch_assign_input(
      properties, IREE_HAL_AMD_GPU_PM4_INPUT_DISPATCH_ID, 2, &input_count);
  launch.user_data.private_segment_size =
      iree_hal_amd_gpu_pm4_launch_assign_input(
          properties, IREE_HAL_AMD_GPU_PM4_INPUT_PRIVATE_SEGMENT_SIZE, 1,
          &input_count);
  if (launch.user_data_count < input_count ||
      launch.user_data_count > IREE_HAL_AMD_GPU_PM4_USER_DATA_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel has %u initial user inputs and %u hardware user SGPRs; "
        "PM4 requires inputs <= hardware count <= 16",
        input_count, launch.user_data_count);
  }
  if (kernel->compute_pgm_rsrc2 & 1u) {
    launch.requirements |= IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH;
  }
  if (properties & IREE_HAL_AMD_GPU_PM4_PROPERTY_DYNAMIC_STACK) {
    launch.requirements |=
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_SCRATCH |
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DYNAMIC_STACK;
  }
  if (properties & IREE_HAL_AMD_GPU_PM4_INPUT_DISPATCH_POINTER) {
    launch.requirements |=
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_DISPATCH_POINTER;
  }
  if (properties & IREE_HAL_AMD_GPU_PM4_INPUT_QUEUE_POINTER) {
    launch.requirements |=
        IREE_HAL_AMD_GPU_PM4_LAUNCH_REQUIREMENT_QUEUE_POINTER;
  }
  *out_launch = launch;
  return iree_ok_status();
}

static void iree_hal_amd_gpu_pm4_launch_store_pointer(uint32_t* words,
                                                      uint64_t value) {
  words[0] = (uint32_t)value;
  words[1] = (uint32_t)(value >> 32);
}

uint32_t* iree_hal_amd_gpu_pm4_launch_emit(
    uint32_t* words, const iree_hal_amd_gpu_pm4_launch_t* launch,
    const uint32_t workgroup_size[3], const uint32_t partial_size[3],
    uint32_t group_segment_byte_length,
    const iree_hal_amd_gpu_pm4_launch_inputs_t* inputs) {
  words =
      iree_hal_amd_gpu_pm4_emit_registers(words, 0x2e0c, 2, launch->program);
  const uint32_t resources[] = {
      launch->resources.resource1,
      launch->resources.resource2 |
          (((group_segment_byte_length + 511u) / 512u) << 15),
  };
  words = iree_hal_amd_gpu_pm4_emit_registers(words, 0x2e12, 2, resources);
  words = iree_hal_amd_gpu_pm4_emit_registers(words, launch->resource3_register,
                                              1, &launch->resources.resource3);
  const uint32_t workitem_count =
      workgroup_size[0] * workgroup_size[1] * workgroup_size[2];
  const uint32_t wave_count =
      (workitem_count + (1u << launch->wavefront_size_log2) - 1u) >>
      launch->wavefront_size_log2;
  const uint32_t resource_limits = wave_count % 4 == 0 ? 1u << 22 : 0;
  words =
      iree_hal_amd_gpu_pm4_emit_registers(words, 0x2e15, 1, &resource_limits);
  // Stop before PIPELINESTAT_ENABLE/PERFCOUNT_ENABLE. Those are native context
  // registers, not padding between geometry and the program address.
  const uint32_t geometry[] = {
      0,
      0,
      0,
      workgroup_size[0] | (partial_size[0] << 16),
      workgroup_size[1] | (partial_size[1] << 16),
      workgroup_size[2] | (partial_size[2] << 16),
  };
  words = iree_hal_amd_gpu_pm4_emit_registers(words, 0x2e04, 6, geometry);
  if (launch->user_data_count) {
    uint32_t user_data[IREE_HAL_AMD_GPU_PM4_USER_DATA_CAPACITY] = {0};
    if (launch->user_data.dispatch_pointer != UINT8_MAX) {
      iree_hal_amd_gpu_pm4_launch_store_pointer(
          user_data + launch->user_data.dispatch_pointer,
          inputs->dispatch_pointer);
    }
    if (launch->user_data.queue_pointer != UINT8_MAX) {
      iree_hal_amd_gpu_pm4_launch_store_pointer(
          user_data + launch->user_data.queue_pointer, inputs->queue_pointer);
    }
    if (launch->user_data.kernarg_pointer != UINT8_MAX) {
      iree_hal_amd_gpu_pm4_launch_store_pointer(
          user_data + launch->user_data.kernarg_pointer,
          inputs->kernarg_pointer);
    }
    if (launch->user_data.dispatch_id != UINT8_MAX) {
      iree_hal_amd_gpu_pm4_launch_store_pointer(
          user_data + launch->user_data.dispatch_id, inputs->dispatch_id);
    }
    if (launch->user_data.private_segment_size != UINT8_MAX) {
      user_data[launch->user_data.private_segment_size] =
          inputs->private_segment_size;
    }
    words = iree_hal_amd_gpu_pm4_emit_registers(
        words, 0x2e40, launch->user_data_count, user_data);
  }
  return words;
}
