// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/media/drm/buffer.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

static void amdf_media_drm_buffer_unlink(amdf_media_drm_buffer_t* buffer) {
  amdf_media_drm_buffer_t** cursor = &buffer->device->buffers;
  while (*cursor != buffer) {
    cursor = &(*cursor)->next;
  }
  *cursor = buffer->next;
}

static amdf_status_t amdf_media_drm_buffer_place(
    amdf_media_drm_buffer_t* buffer) {
  amdf_media_drm_device_t* device = buffer->device;
  uint64_t address = device->address_begin;
  amdf_media_drm_buffer_t** cursor = &device->buffers;
  while (*cursor != NULL) {
    // Partial owners without a reserved range precede every placed owner.
    if ((*cursor)->address != 0) {
      if (buffer->byte_length <= (*cursor)->address - address) {
        break;
      }
      address = (*cursor)->address + (*cursor)->byte_length;
    }
    cursor = &(*cursor)->next;
  }
  if (buffer->byte_length > device->address_end - address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  buffer->address = address;
  buffer->next = *cursor;
  *cursor = buffer;
  struct drm_amdgpu_gem_va mapping = {
      .handle = buffer->handle,
      .operation = AMDGPU_VA_OP_MAP,
      .flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE |
               AMDGPU_VM_MTYPE_UC,
      .va_address = buffer->address,
      .map_size = buffer->byte_length,
  };
  if (ioctl(device->descriptor, DRM_IOCTL_AMDGPU_GEM_VA, &mapping) != 0) {
    return amdf_linux_error(errno);
  }
  buffer->mapped = true;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_drm_buffer_prepare(amdf_media_drm_device_t* device,
                                            int descriptor,
                                            uint64_t byte_length,
                                            bool host_visible,
                                            amdf_media_drm_buffer_t** binding) {
  amdf_media_drm_buffer_t* buffer = NULL;
  amdf_status_t status =
      amdf_calloc(device->base.base.host_allocator, sizeof(*buffer),
                  amdf_alignof(amdf_media_drm_buffer_t), (void**)&buffer);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  buffer->device = device;
  buffer->reference_count = 1;
  buffer->byte_length = byte_length;
  pthread_mutex_lock(&device->memory_mutex);
  status = device->memory_status;
  if (amdf_status_is_ok(status) && descriptor < 0) {
    union drm_amdgpu_gem_create create = {
        .in = {.bo_size = byte_length,
               .alignment = device->page_size,
               .domains = AMDGPU_GEM_DOMAIN_GTT,
               .domain_flags = AMDGPU_GEM_CREATE_CPU_GTT_USWC |
                               AMDGPU_GEM_CREATE_EXPLICIT_SYNC},
    };
    if (ioctl(device->descriptor, DRM_IOCTL_AMDGPU_GEM_CREATE, &create) != 0) {
      status = amdf_linux_error(errno);
    } else {
      buffer->handle = create.out.handle;
    }
  } else if (amdf_status_is_ok(status)) {
    struct drm_prime_handle import = {.fd = descriptor};
    if (ioctl(device->descriptor, DRM_IOCTL_PRIME_FD_TO_HANDLE, &import) != 0) {
      status = amdf_linux_error(errno);
    } else {
      buffer->handle = import.handle;
      // PRIME returns an existing handle without another native reference.
      // Share its owner; closing the temporary alias would consume that owner.
      for (amdf_media_drm_buffer_t* existing = device->buffers;
           existing != NULL; existing = existing->next) {
        if (existing->handle == buffer->handle) {
          ++existing->reference_count;
          *binding = existing;
          pthread_mutex_unlock(&device->memory_mutex);
          amdf_free(device->base.base.host_allocator, buffer);
          return existing->preparation_status;
        }
      }
    }
  }
  if (amdf_status_is_ok(status) && descriptor >= 0) {
    struct drm_amdgpu_gem_create_in information = {0};
    struct drm_amdgpu_gem_op query = {
        .handle = buffer->handle,
        .op = AMDGPU_GEM_OP_GET_GEM_CREATE_INFO,
        .value = (uintptr_t)&information,
    };
    if (ioctl(device->descriptor, DRM_IOCTL_AMDGPU_GEM_OP, &query) != 0) {
      status = amdf_linux_error(errno);
    } else if (information.domains != AMDGPU_GEM_DOMAIN_GTT) {
      // This attachment profile promises system placement. A DMA-BUF handle
      // alone carries no placement guarantee and may name local VRAM.
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_media_drm_buffer_place(buffer);
  }
  if (buffer->address == 0) {
    buffer->next = device->buffers;
    device->buffers = buffer;
  }
  *binding = buffer;
  if (amdf_status_is_ok(status) && host_visible) {
    union drm_amdgpu_gem_mmap mapping = {.in = {.handle = buffer->handle}};
    if (ioctl(device->descriptor, DRM_IOCTL_AMDGPU_GEM_MMAP, &mapping) != 0) {
      status = amdf_linux_error(errno);
    } else {
      void* pointer =
          mmap(NULL, byte_length, PROT_READ | PROT_WRITE, MAP_SHARED,
               device->descriptor, (off_t)mapping.out.addr_ptr);
      if (pointer == MAP_FAILED) {
        status = amdf_linux_error(errno);
      } else {
        buffer->host_pointer = pointer;
      }
    }
  }
  buffer->preparation_status = status;
  pthread_mutex_unlock(&device->memory_mutex);
  return status;
}

amdf_status_t amdf_media_drm_buffer_release(amdf_media_drm_buffer_t** binding) {
  amdf_media_drm_buffer_t* buffer = *binding;
  if (buffer == NULL) {
    return AMDF_STATUS_OK;
  }
  amdf_media_drm_device_t* device = buffer->device;
  pthread_mutex_lock(&device->memory_mutex);
  if (!amdf_status_is_ok(device->memory_status)) {
    const amdf_status_t status = device->memory_status;
    pthread_mutex_unlock(&device->memory_mutex);
    return status;
  }
  if (buffer->reference_count != 1) {
    --buffer->reference_count;
    *binding = NULL;
    pthread_mutex_unlock(&device->memory_mutex);
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = AMDF_STATUS_OK;
  if (buffer->host_pointer != NULL) {
    if (munmap(buffer->host_pointer, buffer->byte_length) != 0) {
      status = amdf_linux_error(errno);
    } else {
      buffer->host_pointer = NULL;
    }
  }
  if (amdf_status_is_ok(status) && buffer->mapped) {
    struct drm_amdgpu_gem_va mapping = {
        .handle = buffer->handle,
        .operation = AMDGPU_VA_OP_UNMAP,
        .va_address = buffer->address,
        .map_size = buffer->byte_length,
    };
    if (ioctl(device->descriptor, DRM_IOCTL_AMDGPU_GEM_VA, &mapping) != 0) {
      status = amdf_linux_error(errno);
    } else {
      buffer->mapped = false;
    }
  }
  if (amdf_status_is_ok(status) && buffer->handle != 0) {
    struct drm_gem_close close = {.handle = buffer->handle};
    if (ioctl(device->descriptor, DRM_IOCTL_GEM_CLOSE, &close) != 0) {
      status = amdf_linux_error(errno);
    } else {
      buffer->handle = 0;
    }
  }
  if (amdf_status_is_ok(status)) {
    amdf_media_drm_buffer_unlink(buffer);
    *binding = NULL;
    amdf_free(device->base.base.host_allocator, buffer);
  }
  pthread_mutex_unlock(&device->memory_mutex);
  return status;
}

void amdf_media_drm_buffer_abandon(amdf_media_drm_buffer_t** binding) {
  amdf_media_drm_buffer_t* buffer = *binding;
  if (buffer == NULL) {
    return;
  }
  amdf_media_drm_device_t* device = buffer->device;
  pthread_mutex_lock(&device->memory_mutex);
  device->memory_status = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  if (--buffer->reference_count == 0) {
    amdf_media_drm_buffer_unlink(buffer);
    amdf_free(device->base.base.host_allocator, buffer);
  }
  *binding = NULL;
  pthread_mutex_unlock(&device->memory_mutex);
}

amdf_status_t amdf_media_drm_buffer_export(
    const amdf_media_drm_buffer_t* buffer, int* out_descriptor) {
  struct drm_prime_handle export = {
      .handle = buffer->handle, .flags = DRM_CLOEXEC | DRM_RDWR, .fd = -1};
  if (ioctl(buffer->device->descriptor, DRM_IOCTL_PRIME_HANDLE_TO_FD,
            &export) != 0) {
    return amdf_linux_error(errno);
  }
  *out_descriptor = export.fd;
  return AMDF_STATUS_OK;
}
