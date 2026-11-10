// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/rocjitsu/buffer.h"

#include <linux/kfd_ioctl.h>
#include <string.h>
#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

_Static_assert(sizeof(struct kfd_ioctl_map_memory_to_gpu_args) ==
                   sizeof(struct kfd_ioctl_unmap_memory_from_gpu_args),
               "KFD map and unmap share the inline device-list wire layout");

amdf_status_t amdf_rocjitsu_buffer_prepare(
    amdf_rocjitsu_source_t* source, uint32_t gpu_count, const uint32_t* gpu_ids,
    amdf_memory_class_t memory_class, uint64_t byte_length,
    uint64_t minimum_alignment, amdf_memory_access_t access,
    amdf_rocjitsu_buffer_t** state) {
  amdf_status_t status = amdf_rocjitsu_source_status(source);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const size_t command_length =
      sizeof(struct kfd_ioctl_map_memory_to_gpu_args) +
      (size_t)gpu_count * sizeof(uint32_t);
  amdf_rocjitsu_buffer_t* buffer = NULL;
  status = amdf_calloc(source->host_allocator, sizeof(*buffer) + command_length,
                       amdf_alignof(amdf_rocjitsu_buffer_t), (void**)&buffer);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  buffer->host_allocator = source->host_allocator;
  buffer->source = source;
  buffer->byte_length = (byte_length + 4095) & ~UINT64_C(4095);
  buffer->alignment = minimum_alignment > 4096 ? minimum_alignment : 4096;
  *state = buffer;

  const size_t reserve_length = buffer->byte_length + buffer->alignment - 4096;
  void* reservation =
      mmap(NULL, reserve_length, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (reservation == MAP_FAILED) {
    return amdf_linux_error(errno);
  }
  buffer->reservation = reservation;
  buffer->reservation_length = reserve_length;
  const uintptr_t aligned = ((uintptr_t)reservation + buffer->alignment - 1) &
                            ~(uintptr_t)(buffer->alignment - 1);
  buffer->pointer = (void*)aligned;
  if (aligned >= (UINT64_C(1) << 48) ||
      buffer->byte_length > (UINT64_C(1) << 48) - aligned) {
    return amdf_linux_error(EOVERFLOW);
  }
  struct kfd_ioctl_alloc_memory_of_gpu_args allocation = {
      .va_addr = aligned,
      .size = buffer->byte_length,
      .gpu_id = gpu_ids[0],
      .flags =
          (memory_class == AMDF_MEMORY_CLASS_LOCAL
               ? KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
               : KFD_IOC_ALLOC_MEM_FLAGS_GTT) |
          KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
          ((access & AMDF_MEMORY_ACCESS_WRITE)
               ? KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
               : 0) |
          ((access & AMDF_MEMORY_ACCESS_EXECUTE)
               ? KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
               : 0),
  };
  status = amdf_rocjitsu_source_command(source, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                                        &allocation, sizeof(allocation));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  buffer->handle = allocation.handle;
  buffer->allocated = true;
  struct kfd_ioctl_map_memory_to_gpu_args* mapping = (void*)buffer->command;
  mapping->handle = buffer->handle;
  mapping->n_devices = gpu_count;
  memcpy(mapping + 1, gpu_ids, (size_t)gpu_count * sizeof(*gpu_ids));
  status = amdf_rocjitsu_source_command(source, AMDKFD_IOC_MAP_MEMORY_TO_GPU,
                                        mapping, command_length);
  buffer->mapped_count = mapping->n_success;
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (mapping->n_success != mapping->n_devices) {
    return amdf_linux_error(EPROTO);
  }
  rj_vm_map_t host_mapping = {
      .addr = aligned,
      .length = buffer->byte_length,
      .offset = (int64_t)allocation.mmap_offset,
      .prot = PROT_READ | PROT_WRITE,
      .flags = MAP_SHARED | MAP_FIXED,
  };
  status = amdf_rocjitsu_source_map(source, &host_mapping);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (host_mapping.mapped_addr != aligned) {
    return amdf_linux_error(EPROTO);
  }
  buffer->host_mapped = true;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_rocjitsu_buffer_destroy(amdf_rocjitsu_buffer_t* buffer) {
  if (buffer == NULL) {
    return AMDF_STATUS_OK;
  }
  if (buffer->mapped_count != 0) {
    struct kfd_ioctl_unmap_memory_from_gpu_args* detach =
        (void*)buffer->command;
    uint32_t* gpu_ids = (void*)(detach + 1);
    detach->handle = buffer->handle;
    detach->n_devices = buffer->mapped_count;
    detach->n_success = 0;
    const amdf_status_t status = amdf_rocjitsu_source_command(
        buffer->source, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, detach,
        sizeof(*detach) + (size_t)buffer->mapped_count * sizeof(*gpu_ids));
    buffer->mapped_count -= detach->n_success;
    memmove(gpu_ids, gpu_ids + detach->n_success,
            (size_t)buffer->mapped_count * sizeof(*gpu_ids));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (buffer->mapped_count != 0) {
      return amdf_linux_error(EPROTO);
    }
  }
  if (buffer->allocated) {
    struct kfd_ioctl_free_memory_of_gpu_args release = {.handle =
                                                            buffer->handle};
    const amdf_status_t status = amdf_rocjitsu_source_command(
        buffer->source, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &release,
        sizeof(release));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    buffer->allocated = false;
  }
  if (buffer->reservation != NULL &&
      munmap(buffer->reservation, buffer->reservation_length) != 0) {
    return amdf_linux_error(errno);
  }
  amdf_free(buffer->host_allocator, buffer);
  return AMDF_STATUS_OK;
}

void amdf_rocjitsu_buffer_abandon(amdf_rocjitsu_buffer_t* buffer) {
  if (buffer != NULL) {
    amdf_free(buffer->host_allocator, buffer);
  }
}
