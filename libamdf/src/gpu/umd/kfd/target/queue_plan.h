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

// Complete target-qualified construction plan for one KFD user queue.
typedef struct amdf_gpu_kfd_user_queue_plan_t {
  // Public queue-family contract implemented by this native plan.
  amdf_gpu_queue_family_properties_t family;
  // KFD_IOC_QUEUE_TYPE_* value used to construct the native queue.
  uint32_t native_queue_type;
  // Primary and optional coupled metadata ring storage.
  struct {
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
  } control;
  // Side storage required only by KFD compute queues.
  struct {
    // Device-local end-of-pipe ring allocation, or zero when absent.
    amdf_gpu_kfd_buffer_create_info_t end_of_pipe_storage;
    // Host-mapped context-save, control-stack, and debug allocation.
    amdf_gpu_kfd_buffer_create_info_t context_storage;
    // Context-save/restore bytes reported to KFD.
    uint32_t context_save_restore_byte_length;
    // Control-stack bytes reported to KFD.
    uint32_t control_stack_byte_length;
    // Debug-state byte offset within `context_storage`.
    uint32_t debug_byte_offset;
    // Debug-state byte length required by the active compute units.
    uint32_t debug_byte_length;
  } compute;
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
