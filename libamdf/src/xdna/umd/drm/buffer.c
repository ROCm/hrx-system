// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/xdna/umd/drm/buffer.h"

#include <drm/amdxdna_accel.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/platform/linux/file.h"

amdf_status_t amdf_linux_xdna_buffer_create(
    int descriptor, uint32_t type, size_t byte_length,
    amdf_linux_xdna_buffer_t* out_buffer) {
  struct amdxdna_drm_create_bo create = {.type = type, .size = byte_length};
  if (ioctl(descriptor, DRM_IOCTL_AMDXDNA_CREATE_BO, &create) != 0) {
    return amdf_linux_error(errno);
  }
  if (create.handle == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *out_buffer = (amdf_linux_xdna_buffer_t){
      .handle = create.handle,
      .type = type,
      .byte_length = byte_length,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_xdna_buffer_import_dma_buf(
    int descriptor, int dma_buf_descriptor, size_t byte_length,
    amdf_linux_xdna_buffer_t* out_buffer) {
  struct amdxdna_drm_va_tbl virtual_address_table = {
      .dmabuf_fd = dma_buf_descriptor,
  };
  struct amdxdna_drm_create_bo create = {
      .vaddr = (uintptr_t)&virtual_address_table,
      .size = byte_length,
      .type = AMDXDNA_BO_SHARE,
  };
  if (ioctl(descriptor, DRM_IOCTL_AMDXDNA_CREATE_BO, &create) != 0) {
    return amdf_linux_error(errno);
  }
  if (create.handle == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *out_buffer = (amdf_linux_xdna_buffer_t){
      .handle = create.handle,
      .type = AMDXDNA_BO_SHARE,
      .byte_length = byte_length,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_xdna_buffer_register_host_pages(
    int descriptor, void* host_page_base, size_t byte_length,
    amdf_linux_xdna_buffer_t* out_buffer) {
  _Alignas(struct amdxdna_drm_va_tbl) unsigned char
      table_storage[sizeof(struct amdxdna_drm_va_tbl) +
                    sizeof(struct amdxdna_drm_va_entry)] = {0};
  struct amdxdna_drm_va_tbl* virtual_address_table =
      (struct amdxdna_drm_va_tbl*)table_storage;
  virtual_address_table->dmabuf_fd = -1;
  virtual_address_table->num_entries = 1;
  virtual_address_table->va_entries[0] = (struct amdxdna_drm_va_entry){
      .vaddr = (uintptr_t)host_page_base,
      .len = byte_length,
  };
  struct amdxdna_drm_create_bo create = {
      .vaddr = (uintptr_t)virtual_address_table,
      .type = AMDXDNA_BO_SHARE,
  };
  if (ioctl(descriptor, DRM_IOCTL_AMDXDNA_CREATE_BO, &create) != 0) {
    return amdf_linux_error(errno);
  }
  if (create.handle == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *out_buffer = (amdf_linux_xdna_buffer_t){
      .handle = create.handle,
      .type = AMDXDNA_BO_SHARE,
      .byte_length = byte_length,
      .host_pointer = host_page_base,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_xdna_buffer_export_dma_buf(
    int descriptor, const amdf_linux_xdna_buffer_t* buffer,
    int* out_dma_buf_descriptor) {
  struct drm_prime_handle export_args = {
      .handle = buffer->handle,
      .flags = DRM_CLOEXEC | DRM_RDWR,
      .fd = -1,
  };
  if (ioctl(descriptor, DRM_IOCTL_PRIME_HANDLE_TO_FD, &export_args) != 0) {
    return amdf_linux_error(errno);
  }
  if (export_args.fd < 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *out_dma_buf_descriptor = export_args.fd;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_xdna_buffer_attach(
    int descriptor, size_t alignment, size_t page_size,
    const amdf_linux_xdna_buffer_t* heap, amdf_linux_xdna_buffer_t* buffer) {
  struct amdxdna_drm_get_bo_info info = {.handle = buffer->handle};
  if (ioctl(descriptor, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &info) != 0) {
    return amdf_linux_error(errno);
  }
  if (buffer->host_pointer != NULL && buffer->mapping.base == NULL) {
    // Registered SHARE buffers have no mmap contract. PASID mode reports no
    // numeric address because the original caller VA is the SVA; forced-IOVA
    // mode reports the independently assigned device address.
    buffer->device_address = info.xdna_addr == AMDXDNA_INVALID_ADDR
                                 ? (uintptr_t)buffer->host_pointer
                                 : info.xdna_addr;
    return AMDF_STATUS_OK;
  } else if (buffer->type == AMDXDNA_BO_DEV) {
    // The kernel reports a subrange of the one live heap mapping. Validate
    // native output before turning it into a host pointer used for copying.
    if (info.xdna_addr < heap->device_address ||
        buffer->byte_length > heap->byte_length ||
        info.xdna_addr - heap->device_address >
            heap->byte_length - buffer->byte_length ||
        info.vaddr != (uintptr_t)heap->host_pointer + info.xdna_addr -
                          heap->device_address) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    buffer->host_pointer = (void*)(uintptr_t)info.vaddr;
  } else {
    // Retain a full native mapping independently of public host views. For
    // SHARE/CMD backing the GEM mmap path acquires the complete page array;
    // this mapping owns that reference until buffer teardown, without a
    // submission-time BO list or a second host registration.
    void* address = NULL;
    int flags = MAP_SHARED;
    if (alignment > page_size) {
      const size_t reserved_length = buffer->byte_length + alignment;
      void* reservation = mmap(NULL, reserved_length, PROT_NONE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (reservation == MAP_FAILED) return amdf_linux_error(errno);
      buffer->mapping.base = reservation;
      buffer->mapping.byte_length = reserved_length;
      address = (void*)(((uintptr_t)reservation + alignment - 1) &
                        ~(uintptr_t)(alignment - 1));
      // Replace only pages within our own live PROT_NONE reservation.
      flags |= MAP_FIXED;
    }
    void* mapping = mmap(address, buffer->byte_length, PROT_READ | PROT_WRITE,
                         flags, descriptor, (off_t)info.map_offset);
    if (mapping == MAP_FAILED) return amdf_linux_error(errno);
    if (buffer->mapping.base == NULL) {
      buffer->mapping.base = mapping;
      buffer->mapping.byte_length = buffer->byte_length;
    }
    // mmap establishes SVA; the pre-mmap query does not yet have its address.
    if (ioctl(descriptor, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &info) != 0) {
      return amdf_linux_error(errno);
    }
    if (info.vaddr != (uintptr_t)mapping ||
        info.xdna_addr == AMDXDNA_INVALID_ADDR) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    buffer->host_pointer = mapping;
  }
  buffer->device_address = info.xdna_addr;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_linux_xdna_buffer_deinitialize(
    int descriptor, amdf_linux_xdna_buffer_t* buffer) {
  if (buffer->mapping.base != NULL) {
    if (munmap(buffer->mapping.base, buffer->mapping.byte_length) != 0) {
      return amdf_linux_error(errno);
    }
    buffer->mapping.base = NULL;
    buffer->mapping.byte_length = 0;
    buffer->host_pointer = NULL;
  }
  if (buffer->handle != 0) {
    struct drm_gem_close close_buffer = {.handle = buffer->handle};
    if (ioctl(descriptor, DRM_IOCTL_GEM_CLOSE, &close_buffer) != 0) {
      return amdf_linux_error(errno);
    }
    buffer->handle = 0;
  }
  buffer->host_pointer = NULL;
  buffer->device_address = 0;
  return AMDF_STATUS_OK;
}
