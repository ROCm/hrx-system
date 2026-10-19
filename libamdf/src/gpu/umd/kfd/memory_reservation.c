// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/memory_reservation.h"

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <emmintrin.h>
#include <linux/kfd_ioctl.h>
#include <linux/sync_file.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/memory.h"
#include "libamdf/src/platform/linux/file.h"

typedef struct amdf_gpu_kfd_reservation_target_t {
  // Live native consumer, borrowed through final removal.
  amdf_gpu_umd_device_t* device;
  // First consumer sharing this exact acquired render VM.
  uint32_t owner_ordinal;
  // Owned publication context, or zero before preparation succeeds.
  uint32_t context;
  // Ordinary pinned storage for the immutable all-zero publication IB.
  amdf_gpu_kfd_buffer_t* transport;
  // Owned transport handle in the acquired render VM.
  uint32_t transport_handle;
  // Stable GPU address of the NOP IB.
  uint64_t transport_address;
  // Exact queried native NOP IB size in bytes.
  uint32_t instruction_byte_length;
  // First available native SDMA scheduling ring.
  uint32_t ring;
} amdf_gpu_kfd_reservation_target_t;

struct amdf_gpu_umd_memory_reservation_t {
  // Allocator for the owner and its immutable target array.
  amdf_allocator_t host_allocator;
  // Owned PROT_NONE host interval excluding all GPU aliases from VA reuse.
  void* host_reservation;
  // Complete interval length including strong-alignment slack.
  size_t host_reservation_byte_length;
  // Common usable GPU base inside the owned host interval.
  uint64_t address;
  // Complete usable reservation extent.
  uint64_t byte_length;
  // Number of caller-ordered consumers, including native aliases.
  uint32_t target_count;
  // Targets stored in the owner's tail; only canonical targets own resources.
  amdf_gpu_kfd_reservation_target_t targets[];
};

bool amdf_gpu_umd_devices_share_mapping_domain(const amdf_gpu_umd_device_t* a,
                                               const amdf_gpu_umd_device_t* b) {
  return a->render_descriptor == b->render_descriptor;
}

amdf_status_t amdf_gpu_umd_query_memory_reservation_capabilities(
    amdf_gpu_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  if (address_domain_ordinal != 0 || device->topology.sdma.engine_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t span = device->topology.virtual_address.end -
                        device->topology.virtual_address.begin;
  const uint64_t maximum_length =
      (span / 2) & ~(uint64_t)(device->page_size - 1);
  uint64_t maximum_alignment = 1;
  while (maximum_alignment <= maximum_length / 2) {
    maximum_alignment <<= 1;
  }
  *out_capabilities = (amdf_memory_reservation_capabilities_t){
      .supported_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                          AMDF_MEMORY_ACCESS_EXECUTE,
      .address_domain_ordinal = 0,
      .maximum_byte_length = maximum_length,
      .reservation_granularity = device->page_size,
      .maximum_alignment = maximum_alignment,
      .mapping_granularity = device->page_size,
      .minimum_address = device->topology.virtual_address.begin,
      .maximum_address = device->topology.virtual_address.end - 1,
  };
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_kfd_reservation_ioctl(
    const amdf_gpu_kfd_reservation_target_t* target, unsigned long request,
    void* arguments) {
  return ioctl(target->device->render_descriptor, request, arguments) == 0
             ? AMDF_STATUS_OK
             : amdf_linux_error(errno);
}

// Called with the connection's memory-mutation lock held. This operation owns
// no submission-facing BO list: only this update's new sources and its fixed
// native publication transport appear in the cold validation list.
static amdf_status_t amdf_gpu_kfd_reservation_publish(
    amdf_gpu_kfd_reservation_target_t* target, uint32_t source_count,
    const struct drm_amdgpu_bo_list_entry* sources) {
  const struct drm_amdgpu_cs_chunk_ib instruction = {
      .va_start = target->transport_address,
      .ib_bytes = target->instruction_byte_length,
      .ip_type = AMDGPU_HW_IP_DMA,
      .ring = target->ring,
  };
  const struct drm_amdgpu_bo_list_in list = {
      .operation = AMDGPU_BO_LIST_OP_CREATE,
      .bo_number = source_count,
      .bo_info_size = sizeof(*sources),
      .bo_info_ptr = (uintptr_t)sources,
  };
  const struct drm_amdgpu_cs_chunk chunks[] = {
      {.chunk_id = AMDGPU_CHUNK_ID_IB,
       .length_dw = sizeof(instruction) / 4,
       .chunk_data = (uintptr_t)&instruction},
      {.chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES,
       .length_dw = sizeof(list) / 4,
       .chunk_data = (uintptr_t)&list},
  };
  const uint64_t pointers[] = {(uintptr_t)&chunks[0], (uintptr_t)&chunks[1]};
  union drm_amdgpu_cs submit = {0};
  submit.in.ctx_id = target->context;
  submit.in.num_chunks = 2;
  submit.in.chunks = (uintptr_t)pointers;
  amdf_status_t status =
      amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_CS, &submit);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  union drm_amdgpu_fence_to_handle export = {0};
  export.in.fence.ctx_id = target->context;
  export.in.fence.ip_type = AMDGPU_HW_IP_DMA;
  export.in.fence.ring = target->ring;
  export.in.fence.seq_no = submit.out.handle;
  export.in.what = AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD;
  status = amdf_gpu_kfd_reservation_ioctl(
      target, DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE, &export);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  int descriptor = (int)export.out.handle;
  union drm_amdgpu_wait_cs wait = {0};
  do {
    // The ioctl's result overlays its input. Rebuild the request when a signal
    // interrupts the wait instead of depending on partially returned bytes.
    wait = (union drm_amdgpu_wait_cs){0};
    wait.in.handle = submit.out.handle;
    wait.in.timeout = UINT64_MAX;
    wait.in.ip_type = AMDGPU_HW_IP_DMA;
    wait.in.ctx_id = target->context;
    wait.in.ring = target->ring;
    status =
        amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_WAIT_CS, &wait);
  } while (status == amdf_linux_error(EINTR));
  if (amdf_status_is_ok(status) && wait.out.status != 0) {
    status = amdf_linux_error(EPROTO);
  }
  if (amdf_status_is_ok(status)) {
    struct sync_file_info info = {0};
    if (ioctl(descriptor, SYNC_IOC_FILE_INFO, &info) != 0) {
      status = amdf_linux_error(errno);
    } else if (info.status != 1) {
      status = amdf_linux_error(info.status < 0 ? -info.status : EPROTO);
    }
  }
  const amdf_status_t close_status = amdf_linux_file_close(&descriptor);
  if (!amdf_status_is_ok(close_status)) {
    status = close_status;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  // A completed DRM VM update does not invalidate already-running KFD user
  // queues. Reaffirm this existing attachment to complete native compute-PASID
  // publication without allocating or changing the attachment set.
  const uint32_t gpu_id = target->device->topology.gpu_id;
  struct kfd_ioctl_map_memory_to_gpu_args ready = {
      .handle = amdf_gpu_kfd_buffer_handle(target->transport),
      .device_ids_array_ptr = (uintptr_t)&gpu_id,
      .n_devices = 1,
  };
  int result;
  do {
    result =
        ioctl(target->device->descriptor, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &ready);
  } while (result == -1 && errno == EINTR);
  if (result != 0) {
    return amdf_linux_error(errno);
  }
  return ready.n_success == 1 ? AMDF_STATUS_OK : amdf_linux_error(EPROTO);
}

static amdf_status_t amdf_gpu_kfd_reservation_clear(
    amdf_gpu_kfd_reservation_target_t* target, uint64_t address,
    uint64_t byte_length) {
  struct drm_amdgpu_gem_va clear = {
      .operation = AMDGPU_VA_OP_CLEAR,
      .flags = AMDGPU_VM_DELAY_UPDATE,
      .va_address = address,
      .map_size = byte_length,
  };
  const struct drm_amdgpu_bo_list_entry transport = {
      .bo_handle = target->transport_handle,
  };
  amdf_gpu_kfd_device_lock_memory(target->device,
                                  AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
  amdf_status_t status =
      amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_GEM_VA, &clear);
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_reservation_publish(target, 1, &transport);
  }
  amdf_gpu_kfd_device_unlock_memory(target->device);
  return status;
}

static amdf_status_t amdf_gpu_kfd_reservation_release_target(
    amdf_gpu_kfd_reservation_target_t* target) {
  if (target->context != 0) {
    union drm_amdgpu_ctx context = {0};
    context.in.op = AMDGPU_CTX_OP_FREE_CTX;
    context.in.ctx_id = target->context;
    const amdf_status_t status =
        amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_CTX, &context);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    target->context = 0;
  }
  if (target->transport_handle != 0) {
    struct drm_gem_close release = {.handle = target->transport_handle};
    const amdf_status_t status =
        amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_GEM_CLOSE, &release);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    target->transport_handle = 0;
  }
  if (target->transport != NULL) {
    const amdf_status_t status = amdf_gpu_kfd_buffer_destroy(target->transport);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    target->transport = NULL;
  }
  return AMDF_STATUS_OK;
}

// After a failure, consume bookkeeping but preserve every native resource
// whose safe release was not established. No owner or retry list survives.
static amdf_status_t amdf_gpu_kfd_reservation_release(
    amdf_gpu_umd_memory_reservation_t* reservation, amdf_status_t status) {
  for (uint32_t i = 0;
       i < reservation->target_count && amdf_status_is_ok(status); ++i) {
    if (reservation->targets[i].owner_ordinal == i) {
      status =
          amdf_gpu_kfd_reservation_release_target(&reservation->targets[i]);
    }
  }
  if (amdf_status_is_ok(status) && reservation->host_reservation != NULL &&
      munmap(reservation->host_reservation,
             reservation->host_reservation_byte_length) != 0) {
    status = amdf_linux_error(errno);
  }
  for (uint32_t i = 0; i < reservation->target_count; ++i) {
    if (reservation->targets[i].transport != NULL) {
      amdf_gpu_kfd_buffer_abandon(reservation->targets[i].transport);
    }
  }
  amdf_free(reservation->host_allocator, reservation);
  return status;
}

static amdf_status_t amdf_gpu_kfd_reservation_prepare_target(
    amdf_gpu_kfd_reservation_target_t* target) {
  amdf_gpu_umd_device_t* device = target->device;
  struct drm_amdgpu_info_hw_ip ip = {0};
  struct drm_amdgpu_info query = {
      .return_pointer = (uintptr_t)&ip,
      .return_size = sizeof(ip),
      .query = AMDGPU_INFO_HW_IP_INFO,
      .query_hw_ip = {.type = AMDGPU_HW_IP_DMA},
  };
  amdf_status_t status =
      amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_INFO, &query);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (ip.available_rings == 0 || ip.ib_size_alignment == 0 ||
      ip.ib_size_alignment > device->page_size ||
      ip.ib_size_alignment % 4 != 0 ||
      ip.ib_start_alignment > device->page_size) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  target->ring = (uint32_t)__builtin_ctz(ip.available_rings);
  target->instruction_byte_length = ip.ib_size_alignment;
  const amdf_gpu_kfd_buffer_create_info_t create = {
      .native_flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                      KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                      AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE,
      .byte_length = device->page_size,
      .alignment = device->page_size,
      .host_access = AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED,
  };
  amdf_gpu_kfd_buffer_result_t result = {0};
  status =
      amdf_gpu_kfd_buffer_prepare(device, &create, &target->transport, &result);
  if (amdf_status_is_ok(status)) {
    target->transport_address = result.device_address;
    memset(result.host_pointer, 0, device->page_size);
    _mm_mfence();
    status = amdf_gpu_kfd_buffer_export_gem(target->transport,
                                            device->render_descriptor,
                                            &target->transport_handle);
  }
  if (amdf_status_is_ok(status)) {
    union drm_amdgpu_ctx context = {0};
    context.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
    context.in.priority = AMDGPU_CTX_PRIORITY_NORMAL;
    status =
        amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_CTX, &context);
    if (amdf_status_is_ok(status)) {
      target->context = context.out.alloc.ctx_id;
    }
  }
  return status;
}

amdf_status_t amdf_gpu_umd_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_umd_device_t* const* devices, uint64_t byte_length,
    uint64_t minimum_alignment,
    amdf_gpu_umd_memory_reservation_t** out_reservation) {
  amdf_gpu_umd_memory_reservation_t* reservation = NULL;
  amdf_status_t status = amdf_calloc_with_trailing(
      host_allocator, sizeof(*reservation),
      device_count * sizeof(reservation->targets[0]),
      amdf_alignof(amdf_gpu_umd_memory_reservation_t), (void**)&reservation);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  reservation->host_allocator = host_allocator;
  reservation->byte_length = byte_length;
  reservation->target_count = device_count;
  size_t alignment = minimum_alignment > devices[0]->page_size
                         ? (size_t)minimum_alignment
                         : devices[0]->page_size;
  // Keep large aliases eligible for native PTE fragments even when their final
  // physical extent is short. Alignment consumes only reserved address space;
  // backing length and independently replaceable mapping units stay unchanged.
  for (uint32_t i = 0; i < device_count; ++i) {
    const size_t fragment_size =
        devices[i]->topology.virtual_address.fragment_size;
    if (byte_length >= fragment_size && alignment < fragment_size) {
      alignment = fragment_size;
    }
  }
  reservation->host_reservation_byte_length =
      (size_t)byte_length + alignment - devices[0]->page_size;
  void* pointer =
      mmap(NULL, reservation->host_reservation_byte_length, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (pointer == MAP_FAILED) {
    status = amdf_linux_error(errno);
  } else {
    reservation->host_reservation = pointer;
    reservation->address =
        ((uintptr_t)pointer + alignment - 1) & ~(alignment - 1);
  }
  for (uint32_t i = 0; i < device_count; ++i) {
    amdf_gpu_kfd_reservation_target_t* target = &reservation->targets[i];
    target->device = devices[i];
    target->owner_ordinal = i;
    for (uint32_t j = 0; j < i; ++j) {
      if (amdf_gpu_umd_devices_share_mapping_domain(devices[i], devices[j])) {
        target->owner_ordinal = j;
        break;
      }
    }
    if (amdf_status_is_ok(status) &&
        (reservation->address < devices[i]->topology.virtual_address.begin ||
         reservation->address >= devices[i]->topology.virtual_address.end ||
         byte_length >
             devices[i]->topology.virtual_address.end - reservation->address)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  for (uint32_t i = 0; i < device_count && amdf_status_is_ok(status); ++i) {
    if (reservation->targets[i].owner_ordinal == i) {
      status =
          amdf_gpu_kfd_reservation_prepare_target(&reservation->targets[i]);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_reservation = reservation;
  } else {
    // No aliases have been installed: release preparation without publishing a
    // clear or waiting on any application work.
    const amdf_status_t release_status =
        amdf_gpu_kfd_reservation_release(reservation, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  return status;
}

void amdf_gpu_umd_memory_reservation_query_access(
    const amdf_gpu_umd_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access) {
  *out_address = reservation->address;
  *out_mapping_access_ordinal =
      reservation->targets[access_ordinal].owner_ordinal;
  *out_mapping_granularity =
      reservation->targets[access_ordinal].device->page_size;
  *out_required_access = 0;
}

amdf_status_t amdf_gpu_umd_memory_reservation_update(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started) {
  amdf_gpu_kfd_reservation_target_t* target =
      &reservation->targets[reservation->targets[access_ordinal].owner_ordinal];
  if (update_count == UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  struct drm_amdgpu_bo_list_entry* sources = NULL;
  amdf_status_t status = amdf_calloc_array(
      reservation->host_allocator, (size_t)update_count + 1, sizeof(*sources),
      amdf_alignof(struct drm_amdgpu_bo_list_entry), (void**)&sources);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  sources[0].bo_handle = target->transport_handle;
  uint32_t source_count = 1;
  for (uint32_t i = 0; i < update_count; ++i) {
    if (updates[i].memory != NULL) {
      sources[source_count++].bo_handle =
          amdf_gpu_kfd_memory_mapping_handle(updates[i].memory, target->device);
    }
  }
  amdf_gpu_kfd_device_lock_memory(target->device,
                                  AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
  *mutation_started = true;
  uint32_t source = 1;
  for (uint32_t i = 0; i < update_count && amdf_status_is_ok(status); ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    struct drm_amdgpu_gem_va update = {
        .handle = entry->memory != NULL ? sources[source++].bo_handle : 0,
        .operation =
            entry->memory != NULL ? AMDGPU_VA_OP_REPLACE : AMDGPU_VA_OP_CLEAR,
        .flags =
            AMDGPU_VM_DELAY_UPDATE |
            ((entry->access & AMDF_MEMORY_ACCESS_READ) ? AMDGPU_VM_PAGE_READABLE
                                                       : 0) |
            ((entry->access & AMDF_MEMORY_ACCESS_WRITE)
                 ? AMDGPU_VM_PAGE_WRITEABLE
                 : 0) |
            ((entry->access & AMDF_MEMORY_ACCESS_EXECUTE)
                 ? AMDGPU_VM_PAGE_EXECUTABLE
                 : 0),
        .va_address = reservation->address + entry->target_byte_offset,
        .offset_in_bo = entry->source_byte_offset,
        .map_size = entry->byte_length,
    };
    status = amdf_gpu_kfd_reservation_ioctl(target, DRM_IOCTL_AMDGPU_GEM_VA,
                                            &update);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_gpu_kfd_reservation_publish(target, source_count, sources);
  }
  amdf_gpu_kfd_device_unlock_memory(target->device);
  amdf_free(reservation->host_allocator, sources);
  return status;
}

amdf_status_t amdf_gpu_umd_memory_reservation_detach(
    amdf_gpu_umd_memory_reservation_t* reservation, uint32_t access_ordinal) {
  amdf_gpu_kfd_reservation_target_t* target =
      &reservation->targets[reservation->targets[access_ordinal].owner_ordinal];
  return amdf_gpu_kfd_reservation_clear(target, reservation->address,
                                        reservation->byte_length);
}

amdf_status_t amdf_gpu_umd_memory_reservation_destroy(
    amdf_gpu_umd_memory_reservation_t* reservation) {
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0;
       i < reservation->target_count && amdf_status_is_ok(status); ++i) {
    if (reservation->targets[i].owner_ordinal == i) {
      status = amdf_gpu_umd_memory_reservation_detach(reservation, i);
    }
  }
  return amdf_gpu_kfd_reservation_release(reservation, status);
}

amdf_status_t amdf_gpu_umd_memory_reservation_update_on_queue(
    amdf_gpu_umd_memory_reservation_t* reservation,
    amdf_gpu_umd_kernel_queue_t* queue, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    uint64_t submission, bool* mutation_started, bool* accepted) {
  (void)reservation;
  (void)queue;
  (void)access_ordinal;
  (void)update_count;
  (void)updates;
  (void)submission;
  (void)mutation_started;
  (void)accepted;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}
