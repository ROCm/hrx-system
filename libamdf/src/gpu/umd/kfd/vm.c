// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/vm.h"

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <emmintrin.h>
#include <linux/kfd_ioctl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/platform/linux/file.h"

static amdf_status_t amdf_gpu_kfd_vm_ioctl(void* user_data, int descriptor,
                                           unsigned long request,
                                           void* arguments) {
  (void)user_data;
  return ioctl(descriptor, request, arguments) == 0 ? AMDF_STATUS_OK
                                                    : amdf_linux_error(errno);
}

static amdf_status_t amdf_gpu_kfd_vm_map(void* user_data, int descriptor,
                                         uint64_t byte_offset,
                                         size_t byte_length,
                                         void** out_mapping) {
  (void)user_data;
  void* mapping = mmap(NULL, byte_length, PROT_READ | PROT_WRITE, MAP_SHARED,
                       descriptor, (off_t)byte_offset);
  if (mapping == MAP_FAILED) return amdf_linux_error(errno);
  *out_mapping = mapping;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_vm_unmap(void* user_data, void* mapping,
                                           size_t byte_length) {
  (void)user_data;
  return munmap(mapping, byte_length) == 0 ? AMDF_STATUS_OK
                                           : amdf_linux_error(errno);
}

static const amdf_gpu_kfd_vm_native_api_t amdf_gpu_kfd_vm_native_api = {
    .ioctl = amdf_gpu_kfd_vm_ioctl,
    .map = amdf_gpu_kfd_vm_map,
    .unmap = amdf_gpu_kfd_vm_unmap,
};

const amdf_gpu_kfd_vm_native_api_t* amdf_gpu_kfd_vm_default_native_api(void) {
  return &amdf_gpu_kfd_vm_native_api;
}

amdf_status_t amdf_gpu_kfd_vm_bootstrap_release(
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap) {
  const amdf_gpu_kfd_vm_native_api_t* native_api = bootstrap->native_api;
  if (bootstrap->buffer_handle != 0) {
    struct drm_gem_close release = {.handle = bootstrap->buffer_handle};
    const amdf_status_t status =
        native_api->ioctl(native_api->user_data, bootstrap->render_descriptor,
                          DRM_IOCTL_GEM_CLOSE, &release);
    if (!amdf_status_is_ok(status)) return status;
    bootstrap->buffer_handle = 0;
  }
  if (bootstrap->mapping.byte_length != 0) {
    const amdf_status_t status =
        native_api->unmap(native_api->user_data, bootstrap->mapping.pointer,
                          bootstrap->mapping.byte_length);
    if (!amdf_status_is_ok(status)) return status;
    bootstrap->mapping.pointer = NULL;
    bootstrap->mapping.byte_length = 0;
  }
  if (bootstrap->context_identifier != 0) {
    union drm_amdgpu_ctx context = {
        .in = {.op = AMDGPU_CTX_OP_FREE_CTX,
               .ctx_id = bootstrap->context_identifier},
    };
    const amdf_status_t status =
        native_api->ioctl(native_api->user_data, bootstrap->render_descriptor,
                          DRM_IOCTL_AMDGPU_CTX, &context);
    if (!amdf_status_is_ok(status)) return status;
    bootstrap->context_identifier = 0;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_kfd_vm_acquire(
    int kfd_descriptor, int render_descriptor,
    const amdf_gpu_kfd_topology_t* topology, size_t page_size,
    const amdf_gpu_kfd_vm_native_api_t* native_api,
    amdf_gpu_kfd_vm_bootstrap_t* bootstrap) {
  bootstrap->native_api = native_api;
  bootstrap->render_descriptor = render_descriptor;
  struct drm_amdgpu_info_hw_ip ip = {0};
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)&ip,
      .return_size = sizeof(ip),
      .query = AMDGPU_INFO_HW_IP_INFO,
      .query_hw_ip = {.type = AMDGPU_HW_IP_DMA},
  };
  amdf_status_t status = native_api->ioctl(
      native_api->user_data, render_descriptor, DRM_IOCTL_AMDGPU_INFO, &query);
  if (!amdf_status_is_ok(status)) return status;

  // SDMA 2 through 7 share the one-dword, all-zero NOP encoding. The native
  // ring-test IBs use it too. Other instruction formats require qualification;
  // no target-specific dispatch encoder is needed for this fixed command.
  if (ip.hw_ip_version_major < 2 || ip.hw_ip_version_major > 7 ||
      ip.available_rings == 0 || ip.ib_size_alignment == 0 ||
      ip.ib_size_alignment % sizeof(uint32_t) != 0 ||
      ip.ib_size_alignment > page_size || ip.ib_start_alignment == 0 ||
      page_size % ip.ib_start_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint32_t ring = (uint32_t)__builtin_ctz(ip.available_rings);
  union drm_amdgpu_ctx context = {
      .in = {.op = AMDGPU_CTX_OP_ALLOC_CTX,
             .priority = AMDGPU_CTX_PRIORITY_NORMAL},
  };
  status = native_api->ioctl(native_api->user_data, render_descriptor,
                             DRM_IOCTL_AMDGPU_CTX, &context);
  if (!amdf_status_is_ok(status)) return status;
  bootstrap->context_identifier = context.out.alloc.ctx_id;

  union drm_amdgpu_gem_create create = {
      .in = {.bo_size = page_size,
             .alignment = page_size,
             .domains = AMDGPU_GEM_DOMAIN_GTT,
             .domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED},
  };
  status = native_api->ioctl(native_api->user_data, render_descriptor,
                             DRM_IOCTL_AMDGPU_GEM_CREATE, &create);
  if (!amdf_status_is_ok(status)) return status;
  bootstrap->buffer_handle = create.out.handle;

  union drm_amdgpu_gem_mmap mapping = {
      .in = {.handle = bootstrap->buffer_handle},
  };
  status = native_api->ioctl(native_api->user_data, render_descriptor,
                             DRM_IOCTL_AMDGPU_GEM_MMAP, &mapping);
  if (!amdf_status_is_ok(status)) return status;
  status = native_api->map(native_api->user_data, render_descriptor,
                           mapping.out.addr_ptr, page_size,
                           &bootstrap->mapping.pointer);
  if (!amdf_status_is_ok(status)) return status;
  bootstrap->mapping.byte_length = page_size;

  const uint64_t address = (uintptr_t)bootstrap->mapping.pointer;
  if (address < topology->virtual_address.begin ||
      address >= topology->virtual_address.end ||
      page_size > topology->virtual_address.end - address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  memset(bootstrap->mapping.pointer, 0, page_size);
  // The GTT view is coherent; publish the complete IB before native submission.
  _mm_mfence();
  struct drm_amdgpu_gem_va map = {
      .handle = bootstrap->buffer_handle,
      .operation = AMDGPU_VA_OP_MAP,
      .flags = AMDGPU_VM_DELAY_UPDATE | AMDGPU_VM_PAGE_READABLE |
               AMDGPU_VM_PAGE_EXECUTABLE,
      .va_address = address,
      .map_size = page_size,
  };
  status = native_api->ioctl(native_api->user_data, render_descriptor,
                             DRM_IOCTL_AMDGPU_GEM_VA, &map);
  if (!amdf_status_is_ok(status)) return status;

  // Realize the mapping through CS, which propagates VM update errors and
  // includes page-table dependencies. GEM_VA completion alone can expose a
  // stub or stale fence on native update failures and is not this barrier.
  struct drm_amdgpu_cs_chunk_ib ib = {
      .va_start = address,
      .ib_bytes = ip.ib_size_alignment,
      .ip_type = AMDGPU_HW_IP_DMA,
      .ring = ring,
  };
  struct drm_amdgpu_bo_list_entry entry = {
      .bo_handle = bootstrap->buffer_handle,
  };
  struct drm_amdgpu_bo_list_in buffers = {
      .operation = AMDGPU_BO_LIST_OP_CREATE,
      .bo_number = 1,
      .bo_info_size = sizeof(entry),
      .bo_info_ptr = (uintptr_t)&entry,
  };
  struct drm_amdgpu_cs_chunk chunks[] = {
      {.chunk_id = AMDGPU_CHUNK_ID_IB,
       .length_dw = sizeof(ib) / sizeof(uint32_t),
       .chunk_data = (uintptr_t)&ib},
      {.chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES,
       .length_dw = sizeof(buffers) / sizeof(uint32_t),
       .chunk_data = (uintptr_t)&buffers},
  };
  const uint64_t chunk_pointers[] = {(uintptr_t)&chunks[0],
                                     (uintptr_t)&chunks[1]};
  union drm_amdgpu_cs submission = {
      .in = {.ctx_id = bootstrap->context_identifier,
             .num_chunks = 2,
             .chunks = (uintptr_t)chunk_pointers},
  };
  status = native_api->ioctl(native_api->user_data, render_descriptor,
                             DRM_IOCTL_AMDGPU_CS, &submission);
  if (!amdf_status_is_ok(status)) return status;
  union drm_amdgpu_wait_cs wait = {
      .in = {.handle = submission.out.handle,
             .timeout = UINT64_MAX,
             .ip_type = AMDGPU_HW_IP_DMA,
             .ctx_id = bootstrap->context_identifier,
             .ring = ring},
  };
  status = native_api->ioctl(native_api->user_data, render_descriptor,
                             DRM_IOCTL_AMDGPU_WAIT_CS, &wait);
  if (!amdf_status_is_ok(status)) return status;
  if (wait.out.status != 0) return amdf_linux_error(ETIME);

  // The kernel's implicit BO wait during SDMA-to-CPU conversion excludes its
  // own BOOKKEEP initialization fences. CS completion above covers them. Keep
  // the mapped BO alive across conversion: GEM_CLOSE can enqueue VM updates.
  struct kfd_ioctl_acquire_vm_args acquire = {
      .drm_fd = (uint32_t)render_descriptor,
      .gpu_id = topology->gpu_id,
  };
  return native_api->ioctl(native_api->user_data, kfd_descriptor,
                           AMDKFD_IOC_ACQUIRE_VM, &acquire);
}
