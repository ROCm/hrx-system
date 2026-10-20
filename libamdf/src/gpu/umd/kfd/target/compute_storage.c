// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/compute_storage.h"

#include <linux/kfd_ioctl.h>

bool amdf_gpu_kfd_compute_storage_plan(
    const amdf_gpu_kfd_topology_t* topology,
    amdf_queue_command_type_t command_type, size_t page_size,
    amdf_gpu_kfd_compute_storage_plan_t* out_plan) {
  const amdf_gpu_endpoint_properties_t* properties = &topology->properties;
  const uint32_t xcc_count = properties->topology.xcc_count;
  if (page_size != 4096 || xcc_count == 0 ||
      properties->compute.compute_unit_count == 0 ||
      properties->compute.compute_unit_count % xcc_count != 0) {
    return false;
  }
  const uint32_t compute_units_per_xcc =
      properties->compute.compute_unit_count / xcc_count;
  const bool cdna =
      properties->gfx_ip.major == 9 &&
      ((properties->gfx_ip.minor == 0 && properties->gfx_ip.stepping == 10) ||
       properties->gfx_ip.minor == 4 || properties->gfx_ip.minor == 5);
  const bool expanded_save_area =
      properties->gfx_ip.major == 12 && properties->gfx_ip.minor == 5;
  if (!cdna && properties->gfx_ip.major != 11 &&
      !(properties->gfx_ip.major == 12 &&
        (properties->gfx_ip.minor == 0 || expanded_save_area))) {
    return false;
  }

  // KFD's debugger area follows the save protocol's wave bound, independently
  // of active occupancy or the packet language used to submit work. Its CDNA
  // bound uses the node's total shader engines even for a per-XCC save area.
  uint64_t saved_wave_count =
      (uint64_t)compute_units_per_xcc * (expanded_save_area ? 64 : 32);
  if (cdna) {
    if (properties->topology.shader_engine_count_per_xcc == 0) {
      return false;
    }
    saved_wave_count = (uint64_t)compute_units_per_xcc * 40;
    const uint64_t stack_wave_limit =
        (uint64_t)properties->topology.shader_engine_count_per_xcc * xcc_count *
        512;
    if (saved_wave_count > stack_wave_limit) {
      saved_wave_count = stack_wave_limit;
    }
  }
  uint64_t control_stack_byte_length = topology->control_stack_byte_length;
  uint64_t context_byte_length = topology->context_save_restore_byte_length;
  if (context_byte_length == 0 && control_stack_byte_length == 0) {
    // Older KFD versions omit both size properties. Match the thunk's save
    // layout on those kernels; newer kernels' explicit sizes remain
    // authoritative, including ASIC-specific register-file differences.
    if (expanded_save_area) {
      return false;
    }
    if (!cdna) {
      if (properties->compute.maximum_wave_count_per_compute_unit == 0) {
        return false;
      }
      saved_wave_count =
          (uint64_t)compute_units_per_xcc *
          properties->compute.maximum_wave_count_per_compute_unit;
    }
    control_stack_byte_length =
        (sizeof(struct kfd_context_save_area_header) +
         saved_wave_count * (cdna ? 8 : 12) + 8 + page_size - 1) &
        ~(uint64_t)(page_size - 1);
    // CDNA2/3 save 64 KiB LDS/CU; CDNA4 uses the topology's enlarged LDS.
    const uint64_t lds_byte_length =
        !cdna || properties->gfx_ip.minor == 5
            ? properties->compute.local_data_share_byte_length
            : UINT64_C(0x10000);
    if (lds_byte_length == 0 || lds_byte_length > UINT32_MAX) {
      return false;
    }
    const uint64_t compute_unit_byte_length =
        (cdna ? UINT64_C(0x80000) : UINT64_C(0x60000)) + 0x4000 +
        lds_byte_length + 0x1000;
    if (compute_unit_byte_length > UINT32_MAX / compute_units_per_xcc) {
      return false;
    }
    context_byte_length =
        control_stack_byte_length +
        ((compute_unit_byte_length * compute_units_per_xcc + page_size - 1) &
         ~(uint64_t)(page_size - 1));
  }
  const uint64_t debug_byte_length =
      ((saved_wave_count * 32 + 63) & ~UINT64_C(63)) * xcc_count;
  if (context_byte_length == 0 || control_stack_byte_length == 0 ||
      control_stack_byte_length > context_byte_length ||
      control_stack_byte_length % page_size != 0 ||
      context_byte_length % page_size != 0 ||
      context_byte_length > UINT32_MAX / xcc_count ||
      debug_byte_length > UINT32_MAX) {
    return false;
  }
  const uint64_t all_contexts_byte_length = context_byte_length * xcc_count;
  amdf_gpu_kfd_compute_storage_plan_t plan = {
      .context_storage =
          {
              .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                              AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                              KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                              KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
              .byte_length = (size_t)((all_contexts_byte_length +
                                       debug_byte_length + page_size - 1) &
                                      ~(uint64_t)(page_size - 1)),
              .alignment = page_size,
              .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
          },
      .context_save_restore_byte_length = (uint32_t)context_byte_length,
      .control_stack_byte_length = (uint32_t)control_stack_byte_length,
      .context_count = xcc_count,
      .debug_byte_offset = (uint32_t)all_contexts_byte_length,
      .debug_byte_length = (uint32_t)debug_byte_length,
  };
  // GFX9.4 AQL firmware supplies its EOP storage. Other compute queues use a
  // caller-provided device-local EOP ring, including GFX9.5 AQL.
  if (!(command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL && cdna &&
        properties->gfx_ip.minor == 4)) {
    plan.end_of_pipe_storage = (amdf_gpu_kfd_buffer_create_info_t){
        .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                        AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                        KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
        .byte_length = 4096,
        .alignment = page_size,
        .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE,
    };
  }
  *out_plan = plan;
  return true;
}
