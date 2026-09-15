// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_VM_H_
#define AMDF_SRC_GPU_UMD_KFD_VM_H_

#include "libamdf/src/gpu/umd/kfd/topology.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native dependencies of the cold DRM-to-KFD VM handover.
typedef struct amdf_gpu_kfd_vm_native_api_t {
  // Opaque state passed to each operation.
  void* user_data;
  // Executes one native ioctl, preserving its failure status.
  amdf_status_t (*ioctl)(void* user_data, int descriptor, unsigned long request,
                         void* arguments);
  // Maps a complete read/write shared GEM view; failure preserves the output.
  amdf_status_t (*map)(void* user_data, int descriptor, uint64_t byte_offset,
                       size_t byte_length, void** out_mapping);
  // Unmaps a complete view, retaining ownership on failure.
  amdf_status_t (*unmap)(void* user_data, void* mapping, size_t byte_length);
} amdf_gpu_kfd_vm_native_api_t;

// Instance-owned bootstrap resources, initially zero. Successful acquisition
// leaves these live until release; failed acquisition retains partial progress
// for the instance's next preparation or teardown attempt.
typedef struct amdf_gpu_kfd_vm_bootstrap_t {
  // Native operations borrowed until all bootstrap resources are released.
  const amdf_gpu_kfd_vm_native_api_t* native_api;
  // Instance render file borrowed throughout acquisition and release.
  int render_descriptor;
  // Owned DRM context; the kernel allocates nonzero context identifiers.
  uint32_t context_identifier;
  // Owned GEM command buffer; zero denotes no native handle.
  uint32_t buffer_handle;
  // Shared CPU view that also reserves the temporary GPU virtual address.
  struct {
    // First mapped host byte, valid while byte_length is nonzero.
    void* pointer;
    // Complete mapped length in bytes, or zero when no view is owned.
    size_t byte_length;
  } mapping;
} amdf_gpu_kfd_vm_bootstrap_t;

// Returns the production native operations.
const amdf_gpu_kfd_vm_native_api_t* amdf_gpu_kfd_vm_default_native_api(void);

// Completes a real DRM command before acquiring the fresh render VM for KFD.
// The command's VM dependencies include outstanding SDMA page-table clears.
// bootstrap is an initially zero, already-owned record; any return may retain
// resources requiring release. Its render file and KFD file must remain live.
// Success means ACQUIRE_VM completed, not that bootstrap resources are
// released.
amdf_status_t amdf_gpu_kfd_vm_acquire(
    int kfd_descriptor, int render_descriptor,
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    const amdf_gpu_kfd_vm_native_api_t* native_api,
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap);

// Releases bootstrap resources after acquisition, or aborts failed acquisition.
// Closing the BO before successful acquisition would enqueue new SDMA mapping
// work after the completion barrier. Failure retains unreleased resources.
amdf_status_t amdf_gpu_kfd_vm_bootstrap_release(
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_KFD_VM_H_
