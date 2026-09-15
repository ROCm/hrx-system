// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/gfx1151/sdma_queue.h"

#include <linux/kfd_ioctl.h>

enum {
  AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE = 4096,
  AMDF_GPU_KFD_GFX1151_SDMA_RING_BYTE_LENGTH = 4096,
  AMDF_GPU_KFD_GFX1151_SDMA_DOORBELL_MAPPING_BYTE_LENGTH = 8192,
};

bool amdf_gpu_kfd_gfx1151_sdma_queue_plan(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size, amdf_gpu_kfd_user_queue_plan_t* out_plan) {
  if (topology == NULL || page_size != AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE ||
      cache_line_size < sizeof(uint64_t) ||
      (cache_line_size & (cache_line_size - 1)) != 0 ||
      cache_line_size > page_size / 2 ||
      topology->properties.gfx_ip.major != 11 ||
      topology->properties.gfx_ip.minor != 5 ||
      topology->properties.gfx_ip.stepping != 1 ||
      topology->properties.topology.xcc_count != 1 ||
      topology->sdma.engine_count == 0 ||
      topology->sdma.queue_count_per_engine == 0 || !topology->sdma.ip.exact ||
      topology->sdma.ip.major != 6 || topology->sdma.ip.minor != 1 ||
      topology->sdma.ip.revision != 1 ||
      topology->virtual_address.alignment != page_size) {
    return false;
  }
  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  const amdf_gpu_kfd_user_queue_plan_t plan = {
      .family =
          {
              .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
              .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
              .format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_GCR,
              .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
              .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
              .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                  AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
              .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
              .user_queue_capabilities =
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
              .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
              .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
              .minimum_ring_byte_length =
                  AMDF_GPU_KFD_GFX1151_SDMA_RING_BYTE_LENGTH,
              .maximum_ring_byte_length =
                  AMDF_GPU_KFD_GFX1151_SDMA_RING_BYTE_LENGTH,
              .ring_byte_length_alignment =
                  AMDF_GPU_KFD_GFX1151_SDMA_RING_BYTE_LENGTH,
          },
      .native_queue_type = KFD_IOC_QUEUE_TYPE_SDMA,
      .ring =
          {
              .storage =
                  {
                      .native_flags = host_storage_flags,
                      .byte_length = AMDF_GPU_KFD_GFX1151_SDMA_RING_BYTE_LENGTH,
                      .alignment = AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .primary_byte_length = AMDF_GPU_KFD_GFX1151_SDMA_RING_BYTE_LENGTH,
          },
      .control =
          {
              .storage =
                  {
                      .native_flags =
                          host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
                      .byte_length = AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .read_index_byte_offset = 0,
              .write_index_byte_offset = cache_line_size,
              .index_bit_count = 64,
          },
      .retirement =
          {
              .flush_trigger_storage =
                  {
                      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                      KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                      KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
                      .byte_length = AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_GFX1151_SDMA_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE,
                  },
          },
      .doorbell =
          {
              .mapping_byte_length =
                  AMDF_GPU_KFD_GFX1151_SDMA_DOORBELL_MAPPING_BYTE_LENGTH,
              .bit_count = 64,
          },
  };
  *out_plan = plan;
  return true;
}
