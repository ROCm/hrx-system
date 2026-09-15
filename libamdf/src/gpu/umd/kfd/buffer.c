// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/buffer.h"

#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

// One native allocation and its stable CPU/GPU address reservation.
struct amdf_gpu_kfd_buffer_t {
  // Device borrowed while buffer metadata is live.
  amdf_gpu_umd_device_t* device;
  // Native KFD allocation identity, or zero after release.
  uint64_t handle;
  // Page-covered native backing length in bytes.
  size_t byte_length;
  // Mapped GPU-ID prefix, retained through final unmap synchronization.
  uint32_t mapped_gpu_count;
  // Native unmap progress retained when an error interrupts full teardown.
  uint32_t unmap_success_count;
  // Owned CPU VA interval reserving the GPU VA and any CPU backing mapping.
  struct {
    // First reserved host address, or NULL after release.
    void* base;
    // Complete reserved interval length, including alignment padding.
    size_t byte_length;
  } reservation;
  // Immutable unique GPU IDs in owner-first native map order.
  uint32_t gpu_ids[];
};

// Map/unmap retain their completed prefix in the ioctl payload; an interrupted
// wait must resume with that progress. Free of an ordinary buffer is
// interrupted before consuming its handle. Other native failures remain
// terminal results of this call, not reasons to retry.
static int amdf_gpu_kfd_buffer_ioctl(int descriptor, unsigned long request,
                                     void* arguments) {
  int result;
  do {
    result = ioctl(descriptor, request, arguments);
  } while (result == -1 && errno == EINTR);
  return result;
}

static amdf_status_t amdf_gpu_kfd_buffer_release_native(
    amdf_gpu_kfd_buffer_t* buffer) {
  if (buffer->mapped_gpu_count != 0) {
    struct kfd_ioctl_unmap_memory_from_gpu_args unmap = {
        .handle = buffer->handle,
        .device_ids_array_ptr = (uintptr_t)buffer->gpu_ids,
        .n_devices = buffer->mapped_gpu_count,
        .n_success = buffer->unmap_success_count,
    };
    const int result = amdf_gpu_kfd_buffer_ioctl(
        buffer->device->descriptor, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &unmap);
    buffer->unmap_success_count = unmap.n_success;
    if (result != 0) return amdf_linux_error(errno);
    if (unmap.n_success != buffer->mapped_gpu_count) {
      return amdf_linux_error(EPROTO);
    }
    buffer->mapped_gpu_count = 0;
  }
  if (buffer->handle != 0) {
    struct kfd_ioctl_free_memory_of_gpu_args release = {
        .handle = buffer->handle,
    };
    if (amdf_gpu_kfd_buffer_ioctl(buffer->device->descriptor,
                                  AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                                  &release) != 0) {
      return amdf_linux_error(errno);
    }
    buffer->handle = 0;
  }
  if (buffer->reservation.base != NULL) {
    if (munmap(buffer->reservation.base, buffer->reservation.byte_length) !=
        0) {
      return amdf_linux_error(errno);
    }
    buffer->reservation.base = NULL;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_buffer_destroy(amdf_gpu_kfd_buffer_t* buffer) {
  const amdf_status_t status = amdf_gpu_kfd_buffer_release_native(buffer);
  if (amdf_status_is_ok(status)) {
    amdf_free(buffer->device->host_allocator, buffer);
  }
  return status;
}

amdf_status_t amdf_gpu_kfd_buffer_discard(amdf_gpu_kfd_buffer_t* buffer) {
  const amdf_status_t status = amdf_gpu_kfd_buffer_release_native(buffer);
  // A failed native release leaves its backing and VA reservation intact. Only
  // this unpublished metadata is freed; no parent inherits a retry obligation.
  amdf_free(buffer->device->host_allocator, buffer);
  return status;
}

void amdf_gpu_kfd_buffer_abandon(amdf_gpu_kfd_buffer_t* buffer) {
  amdf_free(buffer->device->host_allocator, buffer);
}

amdf_status_t amdf_gpu_kfd_buffer_prepare(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_kfd_buffer_create_info_t* create_info,
    amdf_gpu_kfd_buffer_t** buffer_state,
    amdf_gpu_kfd_buffer_result_t* out_result) {
  if (device == NULL || create_info == NULL || buffer_state == NULL ||
      out_result == NULL || create_info->byte_length == 0 ||
      create_info->alignment < device->page_size ||
      (create_info->alignment & (create_info->alignment - 1)) != 0 ||
      (create_info->byte_length & (device->page_size - 1)) != 0 ||
      create_info->host_byte_offset >= device->page_size ||
      create_info->byte_length >
          SIZE_MAX - (create_info->alignment - device->page_size)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((create_info->host_access == AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED) !=
          (create_info->host_pointer != NULL) ||
      (create_info->host_access != AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED &&
       create_info->host_byte_offset != 0) ||
      create_info->host_access > AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  amdf_gpu_kfd_buffer_t* buffer = NULL;
  amdf_status_t status = amdf_calloc_with_trailing(
      device->host_allocator, sizeof(*buffer),
      ((size_t)create_info->peer_count + 1) * sizeof(*buffer->gpu_ids),
      amdf_alignof(amdf_gpu_kfd_buffer_t), (void**)&buffer);
  if (!amdf_status_is_ok(status)) return status;
  buffer->device = device;
  *buffer_state = buffer;
  uint32_t gpu_count = 1;
  buffer->gpu_ids[0] = device->topology.gpu_id;
  uint64_t minimum_address = device->topology.virtual_address.begin;
  uint64_t end_address = device->topology.virtual_address.end;
  for (uint32_t i = 0; i < create_info->peer_count; ++i) {
    const amdf_gpu_kfd_topology_t* topology =
        &create_info->peer_devices[i]->topology;
    uint32_t j = 0;
    while (j < gpu_count && buffer->gpu_ids[j] != topology->gpu_id) ++j;
    if (j == gpu_count) buffer->gpu_ids[gpu_count++] = topology->gpu_id;
    if (topology->virtual_address.begin > minimum_address) {
      minimum_address = topology->virtual_address.begin;
    }
    if (topology->virtual_address.end < end_address) {
      end_address = topology->virtual_address.end;
    }
  }
  buffer->byte_length = create_info->byte_length;
  buffer->reservation.byte_length =
      create_info->byte_length + create_info->alignment - device->page_size;
  void* reservation = mmap(NULL, buffer->reservation.byte_length, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation == MAP_FAILED) {
    status = amdf_linux_error(errno);
  } else {
    buffer->reservation.base = reservation;
  }

  uintptr_t address = 0;
  if (amdf_status_is_ok(status)) {
    address = ((uintptr_t)reservation + create_info->alignment - 1) &
              ~(create_info->alignment - 1);
    if (address < minimum_address || address >= end_address ||
        create_info->byte_length > end_address - address) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }

  struct kfd_ioctl_alloc_memory_of_gpu_args allocate = {
      .va_addr = address,
      .size = create_info->byte_length,
      .gpu_id = device->topology.gpu_id,
      .flags = create_info->native_flags,
  };
  if (create_info->host_access == AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED) {
    allocate.mmap_offset =
        (uintptr_t)create_info->host_pointer - create_info->host_byte_offset;
  }
  if (amdf_status_is_ok(status) &&
      ioctl(device->descriptor, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &allocate) !=
          0) {
    status = amdf_linux_error(errno);
  }
  if (amdf_status_is_ok(status)) {
    buffer->handle = allocate.handle;
    if (buffer->handle == 0) status = amdf_linux_error(EPROTO);
  }

  void* host_pointer = create_info->host_pointer;
  if (amdf_status_is_ok(status) &&
      create_info->host_access == AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED) {
    // MAP_FIXED only replaces pages in this object's own PROT_NONE reservation.
    void* mapping = mmap((void*)address, create_info->byte_length,
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                         device->render_descriptor, allocate.mmap_offset);
    if (mapping == MAP_FAILED) {
      status = amdf_linux_error(errno);
    } else {
      host_pointer = mapping;
    }
  }

  if (amdf_status_is_ok(status)) {
    struct kfd_ioctl_map_memory_to_gpu_args map = {
        .handle = buffer->handle,
        .device_ids_array_ptr = (uintptr_t)buffer->gpu_ids,
        .n_devices = gpu_count,
    };
    const int result = amdf_gpu_kfd_buffer_ioctl(
        device->descriptor, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &map);
    // Mapping can succeed before the final residency/page-table wait fails.
    buffer->mapped_gpu_count = map.n_success;
    if (result != 0) {
      status = amdf_linux_error(errno);
    } else if (map.n_success != gpu_count) {
      status = amdf_linux_error(EPROTO);
    }
  }

  if (amdf_status_is_ok(status)) {
    const amdf_gpu_kfd_buffer_result_t result = {
        .device_address = address + create_info->host_byte_offset,
        .host_pointer = host_pointer,
    };
    *out_result = result;
  }
  return status;
}

amdf_status_t amdf_gpu_kfd_buffer_create(
    amdf_gpu_umd_device_t* device,
    const amdf_gpu_kfd_buffer_create_info_t* create_info,
    amdf_gpu_kfd_buffer_t** out_buffer,
    amdf_gpu_kfd_buffer_result_t* out_result) {
  if (out_buffer == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_gpu_kfd_buffer_t* buffer = NULL;
  amdf_status_t status =
      amdf_gpu_kfd_buffer_prepare(device, create_info, &buffer, out_result);
  if (amdf_status_is_ok(status)) {
    *out_buffer = buffer;
  } else if (buffer != NULL) {
    const amdf_status_t release_status = amdf_gpu_kfd_buffer_discard(buffer);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  return status;
}

uint64_t amdf_gpu_kfd_buffer_handle(const amdf_gpu_kfd_buffer_t* buffer) {
  return buffer->handle;
}

size_t amdf_gpu_kfd_buffer_byte_length(const amdf_gpu_kfd_buffer_t* buffer) {
  return buffer->byte_length;
}
