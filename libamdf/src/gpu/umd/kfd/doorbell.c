// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/doorbell.h"

#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/platform/linux/file.h"

// One queue-owned GPU view, independent of the host's KFD mmap view.
struct amdf_gpu_kfd_doorbell_t {
  // Device borrowed through terminal release.
  amdf_gpu_umd_device_t* device;
  // Reserved CPU interval keeping the same GPU VA unavailable for reuse.
  void* reservation;
  // Complete native process-slice extent in bytes.
  size_t byte_length;
  // Native allocation identity, or zero before allocation succeeds.
  uint64_t handle;
  // Completed GPU map prefix, including a pending final synchronization.
  uint32_t mapped_count;
};

static amdf_status_t amdf_gpu_kfd_doorbell_release_native(
    amdf_gpu_kfd_doorbell_t* doorbell) {
  if (doorbell->mapped_count != 0) {
    uint32_t gpu_id = doorbell->device->topology.gpu_id;
    struct kfd_ioctl_unmap_memory_from_gpu_args unmap = {
        .handle = doorbell->handle,
        .device_ids_array_ptr = (uintptr_t)&gpu_id,
        .n_devices = doorbell->mapped_count,
    };
    amdf_gpu_kfd_device_lock_memory(doorbell->device,
                                    AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE);
    int result;
    do {
      // KFD preserves its completed prefix across interrupted final waits.
      result = ioctl(doorbell->device->descriptor,
                     AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);
    } while (result == -1 && errno == EINTR);
    const int error = errno;
    amdf_gpu_kfd_device_unlock_memory(doorbell->device);
    if (result != 0) {
      return amdf_linux_error(error);
    }
    if (unmap.n_success != doorbell->mapped_count) {
      return amdf_linux_error(EPROTO);
    }
  }
  if (doorbell->handle != 0) {
    struct kfd_ioctl_free_memory_of_gpu_args release = {
        .handle = doorbell->handle,
    };
    // Special allocation release has different consumption semantics from
    // ordinary GTT/VRAM buffers. Any error is terminal for this owner.
    amdf_gpu_kfd_device_lock_memory(doorbell->device,
                                    AMDF_GPU_KFD_MEMORY_PRIORITY_RELEASE);
    const int result = ioctl(doorbell->device->descriptor,
                             AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release);
    const int error = errno;
    amdf_gpu_kfd_device_unlock_memory(doorbell->device);
    if (result != 0) {
      return amdf_linux_error(error);
    }
  }
  if (doorbell->reservation != NULL &&
      munmap(doorbell->reservation, doorbell->byte_length) != 0) {
    return amdf_linux_error(errno);
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_doorbell_destroy(amdf_gpu_kfd_doorbell_t* doorbell) {
  const amdf_status_t status = amdf_gpu_kfd_doorbell_release_native(doorbell);
  amdf_free(doorbell->device->host_allocator, doorbell);
  return status;
}

void amdf_gpu_kfd_doorbell_abandon(amdf_gpu_kfd_doorbell_t* doorbell) {
  amdf_free(doorbell->device->host_allocator, doorbell);
}

amdf_status_t amdf_gpu_kfd_doorbell_create(
    amdf_gpu_umd_device_t* device, size_t byte_length,
    amdf_gpu_kfd_doorbell_t** out_doorbell, uint64_t* out_device_address) {
  amdf_gpu_kfd_doorbell_t* doorbell = NULL;
  amdf_status_t status =
      amdf_calloc(device->host_allocator, sizeof(*doorbell),
                  amdf_alignof(amdf_gpu_kfd_doorbell_t), (void**)&doorbell);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  doorbell->device = device;
  doorbell->byte_length = byte_length;
  void* reservation =
      mmap(NULL, byte_length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation == MAP_FAILED) {
    status = amdf_linux_error(errno);
  } else {
    doorbell->reservation = reservation;
  }

  const uint64_t address = (uintptr_t)doorbell->reservation;
  if (amdf_status_is_ok(status) &&
      (address < device->topology.virtual_address.begin ||
       address >= device->topology.virtual_address.end ||
       byte_length > device->topology.virtual_address.end - address)) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  struct kfd_ioctl_alloc_memory_of_gpu_args allocate = {
      .va_addr = address,
      .size = byte_length,
      .gpu_id = device->topology.gpu_id,
      .flags = KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL |
               AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
               KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
  };
  if (amdf_status_is_ok(status)) {
    amdf_gpu_kfd_device_lock_memory(device,
                                    AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
    if (ioctl(device->descriptor, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocate) !=
        0) {
      status = amdf_linux_error(errno);
    } else {
      doorbell->handle = allocate.handle;
      if (doorbell->handle == 0) {
        status = amdf_linux_error(EPROTO);
      }
    }
    amdf_gpu_kfd_device_unlock_memory(device);
  }
  if (amdf_status_is_ok(status)) {
    uint32_t gpu_id = device->topology.gpu_id;
    struct kfd_ioctl_map_memory_to_gpu_args map = {
        .handle = doorbell->handle,
        .device_ids_array_ptr = (uintptr_t)&gpu_id,
        .n_devices = 1,
    };
    amdf_gpu_kfd_device_lock_memory(device,
                                    AMDF_GPU_KFD_MEMORY_PRIORITY_ACQUIRE);
    int result;
    do {
      result = ioctl(device->descriptor, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map);
    } while (result == -1 && errno == EINTR);
    const int error = errno;
    amdf_gpu_kfd_device_unlock_memory(device);
    doorbell->mapped_count = map.n_success;
    if (result != 0) {
      status = amdf_linux_error(error);
    } else if (map.n_success != 1) {
      status = amdf_linux_error(EPROTO);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_doorbell = doorbell;
    *out_device_address = address;
  } else {
    const amdf_status_t release_status =
        amdf_gpu_kfd_doorbell_destroy(doorbell);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  return status;
}
