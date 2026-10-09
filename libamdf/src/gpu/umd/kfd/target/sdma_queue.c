// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/sdma_queue.h"

#include <linux/kfd_ioctl.h>

enum {
  AMDF_GPU_KFD_SDMA_PAGE_SIZE = 4096,
  AMDF_GPU_KFD_SDMA_RING_BYTE_LENGTH = 4096,
  AMDF_GPU_KFD_SDMA_DOORBELL_MAPPING_BYTE_LENGTH = 8192,
};

// Linux's discovery selects these exact SDMA implementations independently
// of compute IP. ROCr's fence/scope builders and PAL/Mesa's rectangular-copy
// builders define their user packet fields. Classic Z widens before the
// separate SDMA7 pitch/coordinate layout change.
static bool amdf_gpu_kfd_sdma_format_features(
    const amdf_gpu_kfd_ip_version_t* ip,
    amdf_queue_format_features_t* out_features) {
  if (!ip->exact) {
    return false;
  }
  if (ip->major == 4 && ip->minor == 4 &&
      (ip->revision == 2 || ip->revision == 4 || ip->revision == 5)) {
    *out_features = AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT;
    return true;
  }
  if (ip->major == 6 && ((ip->minor == 0 && ip->revision <= 3) ||
                         (ip->minor == 1 && ip->revision <= 4) ||
                         (ip->minor == 4 && ip->revision == 0))) {
    *out_features = AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_MEMORY_TYPE |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_EXTENDED_Z;
    return true;
  }
  if (ip->major == 7 && ip->minor == 0 && ip->revision <= 1) {
    *out_features = AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE;
    return true;
  }
  if (ip->major == 7 && ip->minor == 1 && ip->revision == 0) {
    *out_features = AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT |
                    AMDF_GPU_SDMA_FORMAT_FEATURE_COPY_LINEAR_RECT_WIDE;
    return true;
  }
  return false;
}

bool amdf_gpu_kfd_sdma_queue_plan(const amdf_gpu_kfd_topology_t* topology,
                                  size_t page_size, uint32_t cache_line_size,
                                  amdf_gpu_kfd_user_queue_plan_t* out_plan) {
  amdf_queue_format_features_t format_features = 0;
  if (topology == NULL ||
      !amdf_gpu_kfd_sdma_format_features(&topology->sdma.ip,
                                         &format_features) ||
      page_size != AMDF_GPU_KFD_SDMA_PAGE_SIZE ||
      cache_line_size < sizeof(uint64_t) ||
      (cache_line_size & (cache_line_size - 1)) != 0 ||
      cache_line_size > page_size / 2 || topology->sdma.engine_count == 0 ||
      topology->sdma.queue_count_per_engine == 0) {
    return false;
  }
  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  const bool user_gcr =
      (format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  const amdf_gpu_kfd_user_queue_plan_t plan = {
      .family =
          {
              .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
              .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
              .format_features = format_features,
              .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
              .roles = AMDF_QUEUE_ROLE_TRANSFER |
                       (user_gcr ? AMDF_QUEUE_ROLE_CACHE_CONTROL : 0),
              .cache_operations =
                  user_gcr ? AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                 AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM
                           : 0,
              .cache_transition_kinds =
                  user_gcr ? AMDF_CACHE_TRANSITION_KINDS_GLOBAL : 0,
              .user_queue_capabilities =
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                  AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
              .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
              .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
              .minimum_ring_byte_length = AMDF_GPU_KFD_SDMA_RING_BYTE_LENGTH,
              .maximum_ring_byte_length = UINT64_C(1) << 31,
              .ring_byte_length_alignment = AMDF_GPU_KFD_SDMA_RING_BYTE_LENGTH,
          },
      .native_queue_type = KFD_IOC_QUEUE_TYPE_SDMA,
      .ring =
          {
              .storage =
                  {
                      .native_flags = host_storage_flags,
                      .byte_length = AMDF_GPU_KFD_SDMA_RING_BYTE_LENGTH,
                      .alignment = AMDF_GPU_KFD_SDMA_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .primary_byte_length = AMDF_GPU_KFD_SDMA_RING_BYTE_LENGTH,
          },
      .control =
          {
              .storage =
                  {
                      .native_flags =
                          host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
                      .byte_length = AMDF_GPU_KFD_SDMA_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_SDMA_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .read_index_byte_offset = 0,
              .write_index_byte_offset = cache_line_size,
              .index_bit_count = 64,
              .read_index_mask = UINT64_MAX,
          },
      .retirement =
          {
              .flush_trigger_storage =
                  {
                      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                      AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
                                      KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
                      .byte_length = AMDF_GPU_KFD_SDMA_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_SDMA_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE,
                  },
          },
      .doorbell =
          {
              .mapping_byte_length =
                  AMDF_GPU_KFD_SDMA_DOORBELL_MAPPING_BYTE_LENGTH,
              .bit_count = 64,
          },
  };
  *out_plan = plan;
  return true;
}
