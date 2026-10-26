// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/pm4_queue.h"

#include <linux/kfd_ioctl.h>

enum {
  AMDF_GPU_KFD_PM4_PAGE_SIZE = 4096,
  AMDF_GPU_KFD_PM4_RING_BYTE_LENGTH = 4096,
  AMDF_GPU_KFD_PM4_DOORBELL_MAPPING_BYTE_LENGTH = 8192,
};

static bool amdf_gpu_kfd_pm4_queue_is_supported(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size) {
  if (topology == NULL || page_size != AMDF_GPU_KFD_PM4_PAGE_SIZE ||
      cache_line_size < sizeof(uint64_t) ||
      (cache_line_size & (cache_line_size - 1)) != 0 ||
      cache_line_size > page_size / 3 ||
      topology->properties.compute.wavefront_size != 32 ||
      topology->properties.compute.compute_unit_count == 0 ||
      topology->properties.compute.maximum_wave_count_per_compute_unit == 0 ||
      (topology->properties.gfx_ip.major == 11 && !topology->gc_ip.exact) ||
      (topology->properties.topology.xcc_count != 1 &&
       !(topology->properties.gfx_ip.major == 12 &&
         topology->properties.gfx_ip.minor == 5)) ||
      topology->compute_queue_count == 0) {
    return false;
  }
  return true;
}

static amdf_gpu_queue_family_properties_t
amdf_gpu_kfd_pm4_queue_family_properties(
    const amdf_gpu_kfd_topology_t* topology) {
  amdf_queue_format_features_t format_features =
      AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR;
  if (topology->properties.topology.xcc_count == 1 &&
      topology->properties.topology.shader_engine_count_per_xcc != 0 &&
      topology->properties.compute
              .maximum_scratch_wave_count_per_compute_unit != 0) {
    format_features |= AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH;
  }
  if (topology->properties.gfx_ip.major < 12) {
    // Raw PM4 replaces the compiler descriptor's runtime-owned trap bit.
    format_features |= AMDF_GPU_PM4_FORMAT_FEATURE_COMPUTE_TRAP_PRESENT;
  }
  if (topology->gc_ip.major == 11 && topology->gc_ip.minor == 0 &&
      topology->gc_ip.revision <= 3) {
    // KFD's kfd_dbg_has_cwsr_workaround selects software traps for these
    // native GC identities. Starting waves with PRIV=1 is part of that
    // protocol. Compiler gfx suffixes differ, and unaffected GFX11 parts
    // must not inherit this setting.
    format_features |= AMDF_GPU_PM4_FORMAT_FEATURE_COMPUTE_PRIV;
  }
  return (amdf_gpu_queue_family_properties_t){
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .format_features = format_features,
      .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
      .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
               AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .atomic_capabilities = amdf_gpu_pm4_atomic_capabilities(),
      .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
      .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_LOW |
                               AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL |
                               AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH,
      .minimum_ring_byte_length = AMDF_GPU_KFD_PM4_RING_BYTE_LENGTH,
      .maximum_ring_byte_length = UINT64_C(1) << 31,
      .ring_byte_length_alignment = AMDF_GPU_KFD_PM4_RING_BYTE_LENGTH,
  };
}

bool amdf_gpu_kfd_pm4_queue_plan(const amdf_gpu_kfd_topology_t* topology,
                                 size_t page_size, uint32_t cache_line_size,
                                 amdf_gpu_kfd_user_queue_plan_t* out_plan) {
  if (!amdf_gpu_kfd_pm4_queue_is_supported(topology, page_size,
                                           cache_line_size)) {
    return false;
  }
  amdf_gpu_kfd_compute_storage_plan_t compute;
  if (!amdf_gpu_kfd_compute_storage_plan(
          topology, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, page_size, &compute)) {
    return false;
  }
  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  const amdf_gpu_kfd_user_queue_plan_t plan = {
      .family = amdf_gpu_kfd_pm4_queue_family_properties(topology),
      .native_queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE,
      .ring =
          {
              .storage =
                  {
                      .native_flags = host_storage_flags,
                      .byte_length = AMDF_GPU_KFD_PM4_RING_BYTE_LENGTH,
                      .alignment = AMDF_GPU_KFD_PM4_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .primary_byte_length = AMDF_GPU_KFD_PM4_RING_BYTE_LENGTH,
          },
      .control =
          {
              .storage =
                  {
                      .native_flags =
                          host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
                      .byte_length = AMDF_GPU_KFD_PM4_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_PM4_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .read_index_byte_offset = 0,
              .write_index_byte_offset = cache_line_size,
              .error_payload_byte_offset = 2u * cache_line_size,
              .error_payload_byte_length = sizeof(uint64_t),
              .index_bit_count = 64,
              .read_index_mask = AMDF_GPU_KFD_PM4_RING_BYTE_LENGTH / 4 - 1,
          },
      .compute = compute,
      .retirement =
          {
              .flush_trigger_storage =
                  {
                      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                      AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                                      KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
                      .byte_length = AMDF_GPU_KFD_PM4_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_PM4_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE,
                  },
          },
      .doorbell =
          {
              .mapping_byte_length =
                  AMDF_GPU_KFD_PM4_DOORBELL_MAPPING_BYTE_LENGTH,
              .bit_count = 64,
          },
  };
  *out_plan = plan;
  return true;
}
