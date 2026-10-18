// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/kfd/memory.h"

#include <drm/drm.h>
#include <emmintrin.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/hdp.h"
#include "libamdf/src/gpu/umd/kfd/memory_profile.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"
#include "libamdf/src/platform/linux/dma_buf.h"
#include "libamdf/src/platform/linux/file.h"
#include "libamdf/src/platform/linux/host_cache.h"

// One backing handle and GPU attachment, including incomplete teardown state.
struct amdf_gpu_umd_memory_t {
  // Device borrowed while memory metadata is live.
  amdf_gpu_umd_device_t* device;
  // Native allocation and CPU/GPU mapping, including partial preparation.
  amdf_gpu_kfd_buffer_t* buffer;
  // Optional ordinary address, using the same exact-permission alias protocol.
  amdf_gpu_umd_memory_reservation_t* default_reservation;
  // Page-covered native backing length in bytes.
  size_t byte_length;
  // Canonical DMA-BUF identity for shareable owned GTT.
  amdf_physical_memory_id_t physical_backing_id;
  // Borrowed caller pages or persistent mapping into the owned reservation.
  void* host_pointer;
  // Construction properties determining native allocation and host policy.
  amdf_memory_flags_t flags;
  // Number of unique prepared render-VM source handles in the trailing array.
  uint32_t source_count;
  // Owned DMA-BUF for explicit host aliases, or -1 without that requirement.
  int host_mapping_source;
  // Eager source handles for the requested native consumer group.
  struct {
    // Borrowed exact render connection, shared by aliases of the same VM.
    int descriptor;
    // Owned GEM reference released before the original KFD residency owner.
    uint32_t handle;
  } sources[];
};

// Native allocation plan derived from one selected memory profile.
typedef struct amdf_gpu_kfd_memory_plan_t {
  // Properties established by the selected allocation class and native flags.
  amdf_memory_flags_t flags;
  // Native KFD allocation flags.
  uint32_t native_flags;
  // Page-covered native backing length.
  size_t byte_length;
  // Page-aligned GPU VA base alignment, including any stronger caller request.
  size_t alignment;
  // Offset of the requested range within registered caller pages.
  size_t host_byte_offset;
} amdf_gpu_kfd_memory_plan_t;

static void amdf_gpu_kfd_memory_plan(
    const amdf_gpu_umd_device_t* device,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_kfd_memory_plan_t* plan) {
  plan->flags = profile->guaranteed_flags | create_info->required_flags;
  plan->native_flags =
      (create_info->device_access & AMDF_MEMORY_ACCESS_WRITE) != 0
          ? AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE
          : 0;
  if (profile->memory_class == AMDF_MEMORY_CLASS_LOCAL) {
    plan->native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
    if ((plan->flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
      plan->native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
    }
  } else {
    // CPU snooping does not require uncached GPU payload access. Only control
    // and registered profiles request KFD's coherent GPU mapping policy.
    if (profile->ordinal == AMDF_GPU_KFD_MEMORY_PROFILE_SYSTEM_CONTROL ||
        (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
      plan->native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
    }
    if ((profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0) {
      const uintptr_t pointer = (uintptr_t)create_info->registered_host_pointer;
      plan->host_byte_offset = pointer & (device->page_size - 1);
      plan->native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
    } else {
      plan->native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_GTT;
    }
  }
  if ((create_info->device_access & AMDF_MEMORY_ACCESS_EXECUTE) != 0) {
    plan->native_flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
  }
  const size_t padding = plan->host_byte_offset + device->page_size - 1;
  plan->byte_length =
      (create_info->byte_length + padding) & ~(device->page_size - 1);
  plan->alignment = create_info->minimum_alignment > device->page_size
                        ? (size_t)create_info->minimum_alignment
                        : device->page_size;
}

amdf_status_t amdf_gpu_umd_memory_destroy(amdf_gpu_umd_memory_t* memory) {
  if (memory->default_reservation != NULL) {
    amdf_gpu_umd_memory_reservation_t* reservation =
        memory->default_reservation;
    memory->default_reservation = NULL;
    // Reservation destruction consumes metadata on both success and failure.
    // A failed clear retains source/residency ownership and physical backing.
    const amdf_status_t status =
        amdf_gpu_umd_memory_reservation_destroy(reservation);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  const amdf_status_t source_status =
      amdf_linux_file_close(&memory->host_mapping_source);
  if (!amdf_status_is_ok(source_status)) {
    return source_status;
  }
  while (memory->source_count != 0) {
    const uint32_t i = memory->source_count - 1;
    struct drm_gem_close release = {.handle = memory->sources[i].handle};
    if (ioctl(memory->sources[i].descriptor, DRM_IOCTL_GEM_CLOSE, &release) !=
        0) {
      return amdf_linux_error(errno);
    }
    --memory->source_count;
  }
  if (memory->buffer != NULL) {
    const amdf_status_t status = amdf_gpu_kfd_buffer_destroy(memory->buffer);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    memory->buffer = NULL;
  }
  amdf_free(memory->device->host_allocator, memory);
  return AMDF_STATUS_OK;
}

uint32_t amdf_gpu_kfd_memory_mapping_handle(amdf_gpu_umd_memory_t* memory,
                                            amdf_gpu_umd_device_t* device) {
  for (uint32_t i = 0; i < memory->source_count; ++i) {
    if (memory->sources[i].descriptor == device->render_descriptor) {
      return memory->sources[i].handle;
    }
  }
  amdf_assert(false && "MAPPING_SOURCE construction prepares every consumer");
  return 0;
}

uintptr_t amdf_gpu_umd_memory_host_mapping_source(
    amdf_gpu_umd_memory_t* memory) {
  return (uintptr_t)memory->host_mapping_source;
}

void amdf_gpu_umd_memory_abandon(amdf_gpu_umd_memory_t* memory) {
  if (memory->buffer != NULL) {
    amdf_gpu_kfd_buffer_abandon(memory->buffer);
  }
  amdf_free(memory->device->host_allocator, memory);
}

static amdf_status_t amdf_gpu_kfd_memory_open_dma_buf(
    amdf_gpu_umd_memory_t* memory, int* out_descriptor,
    amdf_linux_dma_buf_info_t* out_info) {
  struct kfd_ioctl_export_dmabuf_args export_args = {
      .handle = amdf_gpu_kfd_buffer_handle(memory->buffer),
      .flags = O_CLOEXEC,
  };
  if (ioctl(memory->device->descriptor, AMDKFD_IOC_EXPORT_DMABUF,
            &export_args) != 0) {
    return amdf_linux_error(errno);
  }
  amdf_assert(export_args.dmabuf_fd <= INT_MAX &&
              "successful DMA-BUF export must return a native int descriptor");
  int descriptor = (int)export_args.dmabuf_fd;
  amdf_linux_dma_buf_info_t info;
  amdf_status_t status = amdf_linux_dma_buf_query(descriptor, &info);
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
    return status;
  }
  *out_descriptor = descriptor;
  *out_info = info;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_memory_prepare_export(
    amdf_gpu_umd_memory_t* memory, amdf_linux_dma_buf_info_t* out_info) {
  int descriptor = -1;
  amdf_linux_dma_buf_info_t info;
  amdf_status_t status =
      amdf_gpu_kfd_memory_open_dma_buf(memory, &descriptor, &info);
  if (amdf_status_is_ok(status) && info.byte_length != memory->byte_length) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (amdf_status_is_ok(status) &&
      (memory->flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) != 0) {
    memory->host_mapping_source = descriptor;
  } else {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_info = info;
  }
  return status;
}

static amdf_memory_host_description_t amdf_gpu_umd_memory_describe_host(
    const void* data, amdf_external_memory_type_t external_memory_type,
    amdf_memory_flags_t flags) {
  (void)external_memory_type;
  const amdf_gpu_umd_device_t* device = data;
  const bool device_local = (flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) != 0;
  // GTT and caller-owned pages have ordinary CPU mappings independently of
  // the GPU PTE cache policy. Only local VRAM uses a WC aperture.
  const bool write_back = !device_local;
  const uint32_t line_size = write_back ? device->cache_line_size : 0;
  amdf_cache_transition_t flush = {0};
  if (write_back) {
    flush = (amdf_cache_transition_t){
        .kind = AMDF_CACHE_TRANSITION_KIND_RANGE,
        .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
        .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
        .host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
        .host_fence_before = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
        .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
        .range_granularity = line_size,
    };
  } else if (device->hdp != NULL) {
    flush = (amdf_cache_transition_t){
        .kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL,
        .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API,
        .host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH,
    };
  }
  const amdf_cache_transition_t invalidate = {
      .kind = write_back ? AMDF_CACHE_TRANSITION_KIND_RANGE
                         : AMDF_CACHE_TRANSITION_KIND_GLOBAL,
      .executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
      .host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      .host_instruction = write_back ? AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH
                                     : AMDF_HOST_CACHE_INSTRUCTION_NONE,
      .host_fence_before = write_back ? AMDF_HOST_CACHE_FENCE_X86_MFENCE
                                      : AMDF_HOST_CACHE_FENCE_NONE,
      .host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE,
      .range_granularity = line_size,
  };
  return (amdf_memory_host_description_t){
      .cacheability = write_back ? AMDF_HOST_CACHEABILITY_WRITE_BACK
                                 : AMDF_HOST_CACHEABILITY_WRITE_COMBINED,
      .cache_line_size = line_size,
      .flush = flush,
      .invalidate = invalidate,
  };
}

amdf_status_t amdf_gpu_umd_device_query_memory_profile(
    amdf_gpu_umd_device_t* device, uint32_t memory_profile_ordinal,
    amdf_memory_native_profile_t* out_profile) {
  const amdf_status_t status = amdf_gpu_kfd_query_memory_profile(
      &device->topology, device->page_size, device->native_lifetime,
      memory_profile_ordinal, out_profile);
  if (amdf_status_is_ok(status)) {
    out_profile->visibility.describe_host = amdf_gpu_umd_memory_describe_host;
    out_profile->visibility.data = device;
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_prepare_import(
    amdf_gpu_umd_device_t* device, const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_import_info_t* import_info,
    const amdf_external_memory_t* external_memory,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  // Native profile admission restricts this to fresh owned-page exports. A
  // repeated foreign import can share a non-refcounted GEM handle and fence.
  amdf_assert(import_info->origin ==
              AMDF_MEMORY_NATIVE_IMPORT_ORIGIN_CONSTRUCTION);
  amdf_gpu_umd_memory_t* memory = NULL;
  amdf_status_t status = amdf_calloc(
      device->host_allocator, sizeof(*memory) + sizeof(memory->sources[0]),
      amdf_alignof(amdf_gpu_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->device = device;
  memory->host_mapping_source = -1;
  memory->byte_length = (size_t)external_memory->byte_length;
  memory->physical_backing_id = external_memory->physical_backing_id;
  memory->flags = profile->guaranteed_flags | import_info->required_flags;
  *memory_state = memory;
  status = amdf_gpu_kfd_buffer_prepare_import(
      device, (int)external_memory->payload.file_descriptor,
      memory->byte_length, &memory->buffer);
  if (amdf_status_is_ok(status)) {
    struct drm_prime_handle import = {
        .fd = (int)external_memory->payload.file_descriptor,
    };
    if (ioctl(device->render_descriptor, DRM_IOCTL_PRIME_FD_TO_HANDLE,
              &import) != 0) {
      status = amdf_linux_error(errno);
    } else {
      memory->sources[0].descriptor = device->render_descriptor;
      memory->sources[0].handle = import.handle;
      memory->source_count = 1;
    }
  }
  uint64_t address = 0;
  const uint64_t alignment = import_info->minimum_alignment > device->page_size
                                 ? import_info->minimum_alignment
                                 : device->page_size;
  if (amdf_status_is_ok(status) &&
      (memory->flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS) != 0) {
    status = amdf_gpu_umd_memory_reservation_create(
        device->host_allocator, 1, &device, memory->byte_length, alignment,
        &memory->default_reservation);
    if (amdf_status_is_ok(status)) {
      const amdf_gpu_umd_mapping_update_t update = {
          .byte_length = memory->byte_length,
          .memory = memory,
          .access = import_info->device_access,
      };
      bool mutation_started = false;
      status = amdf_gpu_umd_memory_reservation_update(
          memory->default_reservation, 0, 1, &update, &mutation_started);
    }
    if (amdf_status_is_ok(status)) {
      uint32_t owner;
      uint64_t granularity;
      amdf_memory_access_t required_access;
      amdf_gpu_umd_memory_reservation_query_access(
          memory->default_reservation, 0, &address, &owner, &granularity,
          &required_access);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_result = (amdf_gpu_umd_memory_result_t){
        .flags = memory->flags,
        .byte_length = memory->byte_length,
        .alignment = alignment,
        .native_allocation_byte_length = memory->byte_length,
        .native_allocation_granularity = device->page_size,
        .physical_backing_id = memory->physical_backing_id,
        .device_address = address,
    };
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_export(
    amdf_gpu_umd_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value) {
  (void)export_info;
  int descriptor = -1;
  amdf_linux_dma_buf_info_t info;
  amdf_status_t status =
      amdf_gpu_kfd_memory_open_dma_buf(memory, &descriptor, &info);
  if (amdf_status_is_ok(status) &&
      (info.byte_length != memory->byte_length ||
       !amdf_physical_memory_id_is_equal(&info.physical_backing_id,
                                         &memory->physical_backing_id))) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (amdf_status_is_ok(status)) {
    *out_value = (amdf_external_memory_t){
        .payload.file_descriptor = descriptor,
        .release = amdf_linux_dma_buf_release,
    };
  } else {
    const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
    if (!amdf_status_is_ok(close_status)) {
      status = close_status;
    }
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_prepare(
    amdf_gpu_umd_device_t* device, uint32_t peer_count,
    amdf_gpu_umd_device_t* const* peer_devices,
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_native_create_info_t* create_info,
    amdf_gpu_umd_memory_t** memory_state,
    amdf_gpu_umd_memory_result_t* out_result) {
  amdf_gpu_kfd_memory_plan_t plan = {0};
  amdf_gpu_kfd_memory_plan(device, profile, create_info, &plan);
  amdf_gpu_umd_memory_t* memory = NULL;
  const size_t source_capacity =
      (plan.flags & AMDF_MEMORY_FLAG_MAPPING_SOURCE) != 0
          ? (size_t)peer_count + 1
          : 0;
  if (source_capacity >
      (SIZE_MAX - sizeof(*memory)) / sizeof(memory->sources[0])) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_status_t status = amdf_calloc(
      device->host_allocator,
      sizeof(*memory) + source_capacity * sizeof(memory->sources[0]),
      amdf_alignof(amdf_gpu_umd_memory_t), (void**)&memory);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memory->device = device;
  memory->host_mapping_source = -1;
  *memory_state = memory;
  memory->byte_length = plan.byte_length;
  const bool registered =
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER) != 0;
  const amdf_gpu_kfd_buffer_create_info_t buffer_create_info = {
      .peer_count = peer_count,
      .peer_devices = peer_devices,
      .native_flags = plan.native_flags,
      .byte_length = plan.byte_length,
      .alignment = plan.alignment,
      .host_access = registered
                         ? AMDF_GPU_KFD_BUFFER_HOST_ACCESS_BORROWED
                         : ((plan.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0
                                ? AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED
                                : AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE),
      .host_pointer = create_info->registered_host_pointer,
      .host_byte_offset = plan.host_byte_offset,
  };
  amdf_gpu_kfd_buffer_result_t buffer_result = {0};
  status = amdf_gpu_kfd_buffer_prepare(device, &buffer_create_info,
                                       &memory->buffer, &buffer_result);
  if (amdf_status_is_ok(status)) {
    memory->host_pointer = buffer_result.host_pointer;
    memory->flags = plan.flags;
  }
  if (amdf_status_is_ok(status) &&
      (profile->roles & AMDF_MEMORY_PROFILE_ROLE_EXPORT) != 0) {
    amdf_linux_dma_buf_info_t info;
    status = amdf_gpu_kfd_memory_prepare_export(memory, &info);
    if (amdf_status_is_ok(status)) {
      memory->physical_backing_id = info.physical_backing_id;
    }
  }
  for (size_t i = 0; amdf_status_is_ok(status) && i < source_capacity; ++i) {
    const int descriptor =
        (i == 0 ? device : peer_devices[i - 1])->render_descriptor;
    uint32_t j = 0;
    while (j < memory->source_count &&
           memory->sources[j].descriptor != descriptor) {
      ++j;
    }
    if (j != memory->source_count) {
      continue;
    }
    uint32_t handle = 0;
    status =
        amdf_gpu_kfd_buffer_export_gem(memory->buffer, descriptor, &handle);
    if (amdf_status_is_ok(status)) {
      memory->sources[j].descriptor = descriptor;
      memory->sources[j].handle = handle;
      ++memory->source_count;
    }
  }
  if (amdf_status_is_ok(status)) {
    // A subpage registration promises only the alignment shared by the caller
    // address and the GPU view, and never exposes unborrowed neighboring bytes.
    const uint64_t alignment =
        registered && plan.host_byte_offset != 0
            ? plan.host_byte_offset & -plan.host_byte_offset
            : plan.alignment;
    const uint64_t length =
        registered ? create_info->byte_length : plan.byte_length;
    const amdf_physical_memory_id_t physical_backing_id =
        (profile->roles & AMDF_MEMORY_PROFILE_ROLE_EXPORT) != 0
            ? memory->physical_backing_id
            : (amdf_physical_memory_id_t){
                  .words = {registered ? (uintptr_t)memory->host_pointer -
                                             plan.host_byte_offset
                                       : (uintptr_t)memory,
                            plan.byte_length},
              };
    *out_result = (amdf_gpu_umd_memory_result_t){
        .flags = plan.flags,
        .source_byte_offset = registered ? plan.host_byte_offset : 0,
        .byte_length = length,
        .alignment = alignment,
        .native_allocation_byte_length = plan.byte_length,
        .native_allocation_granularity = device->page_size,
        .physical_backing_id = physical_backing_id,
        .device_address = buffer_result.device_address,
    };
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_map(
    amdf_gpu_umd_memory_t* memory,
    const amdf_host_mapping_capabilities_t* capabilities,
    const amdf_memory_map_info_t* map_info,
    amdf_gpu_umd_host_mapping_t** out_mapping,
    amdf_gpu_umd_host_mapping_result_t* out_result) {
  *out_result = (amdf_gpu_umd_host_mapping_result_t){
      .flags = capabilities->supported_access,
      .pointer = (uint8_t*)memory->host_pointer + map_info->byte_offset,
      .byte_length = map_info->byte_length,
      .visibility =
          amdf_gpu_umd_memory_describe_host(memory->device, 0, memory->flags),
  };
  // The common host-view object owns the borrow; the native mapping persists
  // with memory and needs no separate allocation or per-view native resource.
  *out_mapping = (amdf_gpu_umd_host_mapping_t*)memory;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_umd_host_mapping_cache_control(
    amdf_gpu_umd_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t memory_byte_offset, uint64_t byte_length) {
  const amdf_gpu_umd_memory_t* memory = (amdf_gpu_umd_memory_t*)mapping;
  if ((memory->flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL) == 0) {
    // GPU snooping does not make this backing coherent with other consumers.
    // Explicit CPU operations cover the requested bytes even on coherent GTT.
    amdf_linux_host_cache_transfer(
        (uint8_t*)memory->host_pointer + memory_byte_offset, byte_length,
        memory->device->cache_line_size);
  } else if (byte_length != 0) {
    // WC VRAM views first drain host stores into the device aperture.
    _mm_mfence();
    if (operation == AMDF_HOST_CACHE_OPERATION_FLUSH) {
      return amdf_gpu_kfd_hdp_publish_host_writes(memory->device->hdp);
    }
  }
  return AMDF_STATUS_OK;
}

void amdf_gpu_umd_host_mapping_destroy(amdf_gpu_umd_host_mapping_t* mapping) {
  (void)mapping;
}
