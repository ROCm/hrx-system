// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_BUFFER_H_
#define AMDF_SRC_GPU_UMD_KFD_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include "libamdf/src/gpu/umd/kfd/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_gpu_kfd_buffer_t amdf_gpu_kfd_buffer_t;

// Host-view ownership selected for one native KFD allocation.
typedef uint32_t amdf_gpu_kfd_buffer_host_access_t;
enum amdf_gpu_kfd_buffer_host_access_e {
  // The native allocation has no host view.
  AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE = 0,
  // Map the native allocation into its owned CPU/GPU VA reservation.
  AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED = 1,
  // Register and borrow caller-owned host pages.
  AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED = 2,
};

// Construction parameters for ordinary GTT, VRAM or registered host storage.
// Doorbell and MMIO allocations have separate native ownership contracts.
typedef struct amdf_gpu_kfd_buffer_create_info_t {
  // Number of additional live consumers in this instance's native context.
  uint32_t peer_count;
  // Borrowed devices with the same exact access permissions as the owner.
  // Duplicate native GPU IDs are mapped only once. NULL when peer_count is
  // zero.
  amdf_gpu_umd_device_t* const* peer_devices;
  // Native KFD allocation flags.
  uint32_t native_flags;
  // Page-covered native backing length in bytes.
  size_t byte_length;
  // Power-of-two GPU VA alignment in bytes.
  size_t alignment;
  // Host-view ownership established by construction.
  amdf_gpu_kfd_buffer_host_access_t host_access;
  // Borrowed first caller byte for BORROWED host access; otherwise NULL.
  void* host_pointer;
  // Offset of the first caller byte within its registered host page.
  size_t host_byte_offset;
} amdf_gpu_kfd_buffer_create_info_t;

// Immutable addresses published with one complete KFD buffer.
typedef struct amdf_gpu_kfd_buffer_result_t {
  // GPU virtual address of the first requested byte.
  uint64_t device_address;
  // Host address of the first requested byte, or NULL when unavailable.
  void* host_pointer;
} amdf_gpu_kfd_buffer_result_t;

// Prepares a KFD buffer in an already-live owner's initially NULL native slot.
// Partial allocation, mapping and reservation state remains in `buffer_state`
// on error for explicit destruction; preparation performs no rollback. The
// result is written only on success. This transition is one-shot.
amdf_status_t amdf_gpu_kfd_buffer_prepare(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_kfd_buffer_create_info_t* create_info,
    amdf_gpu_kfd_buffer_t** buffer_state,
    amdf_gpu_kfd_buffer_result_t* out_result);

// Creates one KFD allocation with stable mappings in the complete GPU group.
// Failure rolls back locally and leaves both outputs unchanged. A terminal
// native rollback failure leaks unreleased native resources and VA reservation;
// buffer metadata is freed without transferring cleanup to another owner.
amdf_status_t amdf_gpu_kfd_buffer_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_kfd_buffer_create_info_t* create_info,
    amdf_gpu_kfd_buffer_t** out_buffer,
    amdf_gpu_kfd_buffer_result_t* out_result);

// Returns the native KFD allocation identity.
uint64_t amdf_gpu_kfd_buffer_handle(const amdf_gpu_kfd_buffer_t* buffer);

// Returns the page-covered native backing length in bytes.
size_t amdf_gpu_kfd_buffer_byte_length(const amdf_gpu_kfd_buffer_t* buffer);

// Releases the GPU mapping, allocation, CPU VA reservation, and object.
// Failure retains the object and unreleased resources with the caller.
amdf_status_t amdf_gpu_kfd_buffer_destroy(amdf_gpu_kfd_buffer_t* buffer);

// Rolls back an unpublished buffer and consumes its metadata on every result.
// Terminal native failure is reported and leaves unreleased backing and its VA
// reservation intact as a leak, not as a deferred cleanup object.
amdf_status_t amdf_gpu_kfd_buffer_discard(amdf_gpu_kfd_buffer_t* buffer);

// Consumes unpublished metadata only. Performs no native release or unmap;
// unreleased backing and its VA reservation remain intact after abandonment.
void amdf_gpu_kfd_buffer_abandon(amdf_gpu_kfd_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KFD_BUFFER_H_
