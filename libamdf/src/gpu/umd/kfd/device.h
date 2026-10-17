// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_KFD_DEVICE_H_
#define AMDF_SRC_GPU_UMD_KFD_DEVICE_H_

#include <stddef.h>

#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/kfd/memory_lock.h"
#include "libamdf/src/gpu/umd/kfd/topology.h"

typedef struct amdf_gpu_kfd_user_queue_native_api_t
    amdf_gpu_kfd_user_queue_native_api_t;
typedef struct amdf_gpu_kfd_hdp_t amdf_gpu_kfd_hdp_t;

// KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE spells bit 31 as a signed shift in the
// native UAPI. Use its unsigned wire value when constructing allocation flags.
#define AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE (UINT32_C(1) << 31)

// Execution state borrowing its instance's KFD context and per-GPU VM binding.
struct amdf_gpu_umd_device_t {
  // Host allocator copied for device and child metadata.
  amdf_allocator_t host_allocator;
  // Native connection owner for shared event backing and terminal sources.
  amdf_gpu_umd_instance_t* instance;
  // Borrowed instance KFD file selecting the primary or secondary context.
  int descriptor;
  // Borrowed instance render file whose VM is acquired by the KFD context.
  int render_descriptor;
  // Borrowed connection-wide lock for cold VM mutations, including peers.
  // Queue publication and progress operations never acquire this lock.
  amdf_gpu_kfd_memory_lock_t* memory_lock;
  // Native lifetime inherited from the provider instance.
  amdf_native_lifetime_t native_lifetime;
  // Native identity, geometry and address limits established at creation.
  amdf_gpu_kfd_topology_t topology;
  // Native CPU page length in bytes.
  size_t page_size;
  // Qualified host cache-line length in bytes.
  uint32_t cache_line_size;
  // Optional device-lifetime HDP register mapping for host-write publication.
  amdf_gpu_kfd_hdp_t* hdp;
  // Native KFD queue operations borrowed through device destruction.
  const amdf_gpu_kfd_user_queue_native_api_t* user_queue_native_api;
};

// KFD releases its process mutex before issuing compute-PASID invalidations.
// The VM's last-flushed sequence can become visible before that invalidation
// finishes. Serialize foreground mutation calls through their complete native
// return, not merely page-table mutation. This does not synchronize device work
// or replace the kernel's eviction/restore ordering.
static inline void amdf_gpu_kfd_device_lock_memory(
    amdf_gpu_umd_device_t* device, amdf_gpu_kfd_memory_priority_t priority) {
  amdf_gpu_kfd_memory_lock_acquire(device->memory_lock, priority);
}

static inline void amdf_gpu_kfd_device_unlock_memory(
    amdf_gpu_umd_device_t* device) {
  amdf_gpu_kfd_memory_lock_release(device->memory_lock);
}

#endif  // AMDF_SRC_GPU_UMD_KFD_DEVICE_H_
