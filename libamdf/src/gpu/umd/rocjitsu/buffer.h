// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_BUFFER_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_BUFFER_H_

#include "libamdf/src/gpu/umd/rocjitsu/source.h"

typedef struct amdf_rocjitsu_buffer_t {
  // Allocator owning this resource's metadata.
  amdf_allocator_t host_allocator;
  // Explicit VM/process borrowed through final native release.
  amdf_rocjitsu_source_t* source;
  // Entire owned address reservation, including alignment padding.
  void* reservation;
  // Byte extent of the complete address reservation.
  size_t reservation_length;
  // Aligned host mapping inside the owned reservation.
  void* pointer;
  // Complete page-rounded reservation/allocation extent.
  uint64_t byte_length;
  // Alignment established by the original address reservation.
  uint64_t alignment;
  // Native allocation handle, meaningful when allocated is true.
  uint64_t handle;
  // Whether the native allocation still needs release.
  bool allocated;
  // Whether the reserved range contains a host view retaining physical pages.
  bool host_mapped;
  // Number of attached GPUs in the retained command's inline ID prefix.
  uint32_t mapped_count;
  // Reusable KFD mapping header followed immediately by inline GPU IDs.
  uint8_t command[];
} amdf_rocjitsu_buffer_t;

// One-shot preparation in an already-live owner's NULL state slot. Partial
// progress remains in the slot on failure for explicit release or abandonment.
// The nonempty array identifies GPUs in source and is borrowed for the call;
// its first ID owns allocation.
amdf_status_t amdf_rocjitsu_buffer_prepare(
    amdf_rocjitsu_source_t* source, uint32_t gpu_count, const uint32_t* gpu_ids,
    amdf_memory_class_t memory_class, uint64_t byte_length,
    uint64_t minimum_alignment, amdf_memory_access_t access,
    amdf_rocjitsu_buffer_t** state);

// Releases GPU attachments, allocation, host address, and metadata in order.
// Failure preserves remaining progress and its native-reachable backing.
amdf_status_t amdf_rocjitsu_buffer_destroy(amdf_rocjitsu_buffer_t* buffer);

// Consumes metadata after terminal cleanup failure, preserving reachable bytes.
void amdf_rocjitsu_buffer_abandon(amdf_rocjitsu_buffer_t* buffer);

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_BUFFER_H_
