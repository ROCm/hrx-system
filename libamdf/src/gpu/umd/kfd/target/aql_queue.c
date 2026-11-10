// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/aql_queue.h"

#include <linux/kfd_ioctl.h>

#include "libamdf/src/gpu/umd/aql_descriptor.h"

bool amdf_gpu_kfd_aql_queue_plan(const amdf_gpu_kfd_topology_t* topology,
                                 size_t page_size, uint32_t cache_line_size,
                                 amdf_gpu_kfd_user_queue_plan_t* out_plan) {
  const amdf_gpu_endpoint_properties_t* properties = &topology->properties;
  if (page_size != 4096 || cache_line_size != 64 ||
      topology->compute_queue_count == 0) {
    return false;
  }
  // CLR's native BARRIER_VALUE path selects CDNA firmware independently of
  // ordinary AQL dispatch and standard AND/OR barriers.
  const bool barrier_value = properties->gfx_ip.major == 9 &&
                             properties->gfx_ip.minor >= 4 &&
                             properties->gfx_ip.stepping <= 2;
  // GFX125x is the compiler name for native GC12.1. Its private aperture is
  // wider than earlier RDNA; exact native discovery selects the base pair.
  const bool extended_apertures = topology->gc_ip.exact &&
                                  topology->gc_ip.major == 12 &&
                                  topology->gc_ip.minor == 1;
  if (properties->gfx_ip.major == 12 && properties->gfx_ip.minor == 5 &&
      !extended_apertures) {
    return false;
  }

  amdf_gpu_aql_descriptor_profile_t aql_profile;
  if (!amdf_gpu_aql_descriptor_profile_initialize(
          properties, extended_apertures ? 0x20000000 : 0x10000,
          extended_apertures ? 0x10000000 : 0x20000, &aql_profile)) {
    return false;
  }

  amdf_gpu_kfd_compute_storage_plan_t compute;
  if (!amdf_gpu_kfd_compute_storage_plan(
          topology, AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, page_size, &compute)) {
    return false;
  }
  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  const amdf_gpu_kfd_buffer_create_info_t host_page = {
      .native_flags = host_storage_flags,
      .byte_length = page_size,
      .alignment = page_size,
      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
  };
  amdf_gpu_kfd_user_queue_plan_t plan = {
      .family =
          {
              .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
              .format_version = AMDF_GPU_AQL_QUEUE_FORMAT_VERSION_1,
              .format_features =
                  AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS |
                  AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE |
                  topology->aql_scalar_register_features |
                  (barrier_value ? AMDF_GPU_AQL_FORMAT_FEATURE_BARRIER_VALUE
                                 : 0),
              .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
              .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
                       AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
              .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                  AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
              .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
              // The format-1 vendor carrier executes MEC PM4 commands. Their
              // encoding capabilities do not qualify target-memory atomics.
              .atomic_capabilities = amdf_gpu_pm4_atomic_capabilities(),
              .user_queue_capabilities =
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                  AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
              .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE |
                                AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI,
              .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_LOW |
                                       AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL |
                                       AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH,
              .minimum_ring_byte_length = 4096,
              .maximum_ring_byte_length = UINT64_C(1) << 31,
              .ring_byte_length_alignment = 4096,
          },
      .native_queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL,
      .ring = {.storage = host_page, .primary_byte_length = 4096},
      .control =
          {
              .storage = host_page,
              .read_index_byte_offset =
                  offsetof(amdf_gpu_aql_descriptor_t, read_dispatch_id),
              .write_index_byte_offset =
                  offsetof(amdf_gpu_aql_descriptor_t, write_dispatch_id),
              .error_payload_byte_offset = 320,
              .error_payload_byte_length = sizeof(uint64_t),
              .index_bit_count = 64,
              .read_index_mask = UINT64_MAX,
          },
      .compute = compute,
      .aql = aql_profile,
      .retirement = {.flush_trigger_storage = host_page},
      .doorbell = {.mapping_byte_length = 8192, .bit_count = 64},
  };
  plan.control.storage.native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
  plan.retirement.flush_trigger_storage.host_access =
      AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE;
  *out_plan = plan;
  return true;
}
