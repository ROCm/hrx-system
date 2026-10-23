// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_TARGET_QUEUE_PLAN_H_
#define AMDF_SRC_GPU_UMD_KFD_TARGET_QUEUE_PLAN_H_

#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/gpu/endpoint_profile.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/target/compute_storage.h"

// Complete target-qualified construction plan for one KFD user queue.
typedef struct amdf_gpu_kfd_user_queue_plan_t {
  // Public queue-family contract implemented by this native plan.
  amdf_gpu_queue_family_properties_t family;
  // KFD_IOC_QUEUE_TYPE_* value used to construct the native queue.
  uint32_t native_queue_type;
  // Primary and optional coupled metadata ring storage.
  struct {
    // Achieved locality of the primary command storage.
    amdf_queue_placement_t placement;
    // Required host release executor before command commit.
    amdf_cache_transition_executor_t host_publish_executor;
    // Host ordering between packet stores and header/index publication.
    amdf_host_cache_fence_t host_publish_fence;
    // Complete allocation containing both ring regions.
    amdf_gpu_kfd_buffer_create_info_t storage;
    // Byte offset of the primary ring within `storage`.
    size_t primary_byte_offset;
    // Writable primary ring capacity in bytes.
    size_t primary_byte_length;
    // Byte offset of the metadata ring within `storage` when present.
    size_t metadata_byte_offset;
    // Writable metadata ring capacity in bytes, or zero when absent.
    size_t metadata_byte_length;
  } ring;
  // Device-visible publication indices and optional exception payload.
  struct {
    // Host-mapped native allocation containing every control field.
    amdf_gpu_kfd_buffer_create_info_t storage;
    // Consumer-owned read-index byte offset within `storage`.
    size_t read_index_byte_offset;
    // Producer-owned write-index byte offset within `storage`.
    size_t write_index_byte_offset;
    // KFD exception-payload byte offset within `storage` when present.
    size_t error_payload_byte_offset;
    // KFD exception-payload length in bytes, or zero when absent.
    size_t error_payload_byte_length;
    // Storage width shared by the read and write indices.
    uint32_t index_bit_count;
    // Mask of the native read counter, UINT64_MAX for a monotonic counter.
    uint64_t read_index_mask;
  } control;
  // Side storage required only by KFD compute queues.
  amdf_gpu_kfd_compute_storage_plan_t compute;
  // AMD AQL descriptor and fixed-scratch encoding, absent when xcc_count is
  // zero.
  struct {
    // Number of equal scratch partitions addressed by the command processors.
    uint32_t xcc_count;
    // Highest physical CU identifier accepted by the firmware descriptor.
    uint32_t maximum_compute_unit_id;
    // Highest resident wave identifier within one CU.
    uint32_t maximum_wave_id;
    // Native flat-address apertures programmed by KFD for this process.
    struct {
      // High 32 bits of the LDS aperture base.
      uint32_t group_base_hi;
      // High 32 bits of the private aperture base.
      uint32_t private_base_hi;
    } apertures;
    // Fixed scratch slot provisioning and target register encodings.
    struct {
      // Physical slots per XCC, including shader-engine rounding.
      uint32_t slot_count_per_xcc;
      // COMPUTE_TMPRING_SIZE.WAVES, in the target's scheduling unit.
      uint32_t temporary_ring_wave_count;
      // Log2 byte granularity of COMPUTE_TMPRING_SIZE.WAVESIZE.
      uint32_t wave_size_shift;
      // Largest supported scratch allocation in bytes per 64-lane wave.
      uint32_t maximum_wave_byte_length;
      // Address-free buffer descriptor template; WORD2 capacity is initially 0.
      uint32_t resource_descriptor[4];
    } scratch;
    // Native firmware signal block byte offset in control storage.
    size_t inactive_signal_byte_offset;
  } aql;
  // Storage used to establish safe native queue retirement.
  struct {
    // Mapped allocation that is never reachable by the queue. Invalidating
    // this mapping after native queue removal advances the VM TLB sequence and
    // forces the target's synchronous heavyweight flush while every
    // queue-reachable mapping remains valid. Zero when native removal already
    // provides an equivalent retirement guarantee.
    amdf_gpu_kfd_buffer_create_info_t flush_trigger_storage;
  } retirement;
  // Direct notification aperture selected by KFD queue construction.
  struct {
    // Complete native doorbell aperture mapping length in bytes.
    size_t mapping_byte_length;
    // Atomic write width of the selected doorbell.
    uint32_t bit_count;
  } doorbell;
} amdf_gpu_kfd_user_queue_plan_t;

#endif  // AMDF_SRC_GPU_UMD_KFD_TARGET_QUEUE_PLAN_H_
