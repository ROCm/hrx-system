// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/gfx1151/pm4_queue.h"

#include <linux/kfd_ioctl.h>

enum {
  AMDF_GPU_KFD_GFX1151_PAGE_SIZE = 4096,
  AMDF_GPU_KFD_GFX1151_RING_BYTE_LENGTH = 4096,
  AMDF_GPU_KFD_GFX1151_END_OF_PIPE_BYTE_LENGTH = 4096,
  AMDF_GPU_KFD_GFX1151_DOORBELL_MAPPING_BYTE_LENGTH = 8192,
  AMDF_GPU_KFD_GFX1151_WAVE_COUNT_PER_COMPUTE_UNIT = 32,
  AMDF_GPU_KFD_GFX1151_DEBUG_BYTE_LENGTH_PER_WAVE = 32,
};

static bool amdf_gpu_kfd_gfx1151_pm4_queue_is_supported(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size) {
  if (topology == NULL || page_size != AMDF_GPU_KFD_GFX1151_PAGE_SIZE ||
      cache_line_size < sizeof(uint64_t) ||
      (cache_line_size & (cache_line_size - 1)) != 0 ||
      cache_line_size > page_size / 3 ||
      topology->properties.gfx_ip.major != 11 ||
      topology->properties.gfx_ip.minor != 5 ||
      topology->properties.gfx_ip.stepping != 1 ||
      topology->properties.compute.wavefront_size != 32 ||
      topology->properties.compute.compute_unit_count == 0 ||
      topology->properties.compute.maximum_wave_count_per_compute_unit !=
          AMDF_GPU_KFD_GFX1151_WAVE_COUNT_PER_COMPUTE_UNIT ||
      topology->properties.topology.xcc_count != 1 ||
      topology->compute_queue_count == 0 ||
      topology->context_save_restore_byte_length == 0 ||
      topology->context_save_restore_byte_length % page_size != 0 ||
      topology->control_stack_byte_length == 0 ||
      topology->control_stack_byte_length % page_size != 0 ||
      topology->control_stack_byte_length >
          topology->context_save_restore_byte_length ||
      topology->virtual_address.alignment != page_size) {
    return false;
  }
  const uint64_t debug_byte_length =
      (uint64_t)topology->properties.compute.compute_unit_count *
      AMDF_GPU_KFD_GFX1151_WAVE_COUNT_PER_COMPUTE_UNIT *
      AMDF_GPU_KFD_GFX1151_DEBUG_BYTE_LENGTH_PER_WAVE;
  return debug_byte_length <= UINT32_MAX &&
         topology->context_save_restore_byte_length <=
             SIZE_MAX - (size_t)debug_byte_length;
}

static amdf_gpu_queue_family_properties_t
amdf_gpu_kfd_gfx1151_pm4_queue_family_properties(void) {
  const amdf_atomic_operations_t atomic_operations =
      AMDF_ATOMIC_OPERATION_WAIT | AMDF_ATOMIC_OPERATION_STORE |
      AMDF_ATOMIC_OPERATION_ADD | AMDF_ATOMIC_OPERATION_SUBTRACT |
      AMDF_ATOMIC_OPERATION_AND | AMDF_ATOMIC_OPERATION_OR |
      AMDF_ATOMIC_OPERATION_XOR;
  const amdf_atomic_wait_conditions_t atomic_wait_conditions =
      AMDF_ATOMIC_WAIT_CONDITION_EQUAL | AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL |
      AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
  return (amdf_gpu_queue_family_properties_t){
      .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
      .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
      .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
      .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER,
      .roles = AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER |
               AMDF_QUEUE_ROLE_ATOMIC | AMDF_QUEUE_ROLE_CACHE_CONTROL,
      .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                          AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
      .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
      .atomic_capabilities =
          {
              .operations_32 = atomic_operations,
              .operations_64 = atomic_operations,
              .wait_conditions_32 = atomic_wait_conditions,
              .wait_conditions_64 = atomic_wait_conditions,
              .operations_without_dispatch_32 = atomic_operations,
              .operations_without_dispatch_64 = atomic_operations,
          },
      .user_queue_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
      .producer_modes = AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE,
      .priority_capabilities = AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL,
      .minimum_ring_byte_length = AMDF_GPU_KFD_GFX1151_RING_BYTE_LENGTH,
      .maximum_ring_byte_length = AMDF_GPU_KFD_GFX1151_RING_BYTE_LENGTH,
      .ring_byte_length_alignment = AMDF_GPU_KFD_GFX1151_RING_BYTE_LENGTH,
  };
}

bool amdf_gpu_kfd_gfx1151_pm4_queue_plan(
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    uint32_t cache_line_size, amdf_gpu_kfd_user_queue_plan_t* out_plan) {
  if (!amdf_gpu_kfd_gfx1151_pm4_queue_is_supported(topology, page_size,
                                                   cache_line_size)) {
    return false;
  }
  const uint32_t host_storage_flags =
      KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  const uint32_t debug_byte_length =
      topology->properties.compute.compute_unit_count *
      AMDF_GPU_KFD_GFX1151_WAVE_COUNT_PER_COMPUTE_UNIT *
      AMDF_GPU_KFD_GFX1151_DEBUG_BYTE_LENGTH_PER_WAVE;
  const size_t context_byte_length =
      topology->context_save_restore_byte_length + debug_byte_length;
  const amdf_gpu_kfd_user_queue_plan_t plan = {
      .family = amdf_gpu_kfd_gfx1151_pm4_queue_family_properties(),
      .native_queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE,
      .ring =
          {
              .storage =
                  {
                      .native_flags = host_storage_flags,
                      .byte_length = AMDF_GPU_KFD_GFX1151_RING_BYTE_LENGTH,
                      .alignment = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .primary_byte_length = AMDF_GPU_KFD_GFX1151_RING_BYTE_LENGTH,
          },
      .control =
          {
              .storage =
                  {
                      .native_flags =
                          host_storage_flags | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
                      .byte_length = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .read_index_byte_offset = 0,
              .write_index_byte_offset = cache_line_size,
              .error_payload_byte_offset = 2u * cache_line_size,
              .error_payload_byte_length = sizeof(uint64_t),
              .index_bit_count = 64,
          },
      .compute =
          {
              .end_of_pipe_storage =
                  {
                      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                                      KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
                      .byte_length =
                          AMDF_GPU_KFD_GFX1151_END_OF_PIPE_BYTE_LENGTH,
                      .alignment = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE,
                  },
              .context_storage =
                  {
                      .native_flags = host_storage_flags,
                      .byte_length =
                          (context_byte_length +
                           AMDF_GPU_KFD_GFX1151_PAGE_SIZE - 1) &
                          ~(size_t)(AMDF_GPU_KFD_GFX1151_PAGE_SIZE - 1),
                      .alignment = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
                  },
              .context_save_restore_byte_length =
                  topology->context_save_restore_byte_length,
              .control_stack_byte_length = topology->control_stack_byte_length,
              .debug_byte_offset = topology->context_save_restore_byte_length,
              .debug_byte_length = debug_byte_length,
          },
      .retirement =
          {
              .flush_trigger_storage =
                  {
                      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                      KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                      KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
                      .byte_length = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .alignment = AMDF_GPU_KFD_GFX1151_PAGE_SIZE,
                      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE,
                  },
          },
      .doorbell =
          {
              .mapping_byte_length =
                  AMDF_GPU_KFD_GFX1151_DOORBELL_MAPPING_BYTE_LENGTH,
              .bit_count = 64,
          },
  };
  *out_plan = plan;
  return true;
}
