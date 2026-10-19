// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_reservation.h"

#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/xdna/umd/mcdm/memory.h"
#include "libamdf/src/xdna/umd/mcdm/memory_profile.h"

// A standard allocation's size is bounded by UINT32_MAX. Larger public mapping
// units use several power-of-two native views, without changing public
// geometry.
#define AMDF_WINDOWS_XDNA_MAXIMUM_VIEW_SIZE (UINT64_C(1) << 31)

typedef struct amdf_windows_xdna_reservation_t {
  // Live native consumer borrowed through terminal destruction.
  amdf_xdna_umd_device_t* device;
  // First native VA, including any slack used for stronger alignment.
  uint64_t reservation_address;
  // Complete native extent owned by this reservation.
  uint64_t reservation_byte_length;
  // Aligned native base before the target's shim-DMA address displacement.
  uint64_t address;
  // Maximum size of each independent native registration, in bytes.
  uint64_t view_byte_length;
  // Fixed number of native view slots allocated during reservation creation.
  uint32_t view_count;
  // Permanent removal error: failed batches have no per-handle progress result
  // and cannot safely be retried or treated as permission to reuse backing.
  amdf_status_t removal_failure;
  // Fixed allocation handles, indexed by destination unit; zero means a hole.
  D3DKMT_HANDLE* views;
  // Pre-sized compact handle list for synchronous destruction and residency.
  // Transient batch storage only; it owns no registrations independently.
  D3DKMT_HANDLE* batch;
} amdf_windows_xdna_reservation_t;

static amdf_status_t amdf_windows_xdna_reservation_remove(
    amdf_windows_xdna_reservation_t* reservation, uint32_t count) {
  if (!amdf_status_is_ok(reservation->removal_failure)) {
    return reservation->removal_failure;
  }
  if (count == 0) {
    return AMDF_STATUS_OK;
  }
  D3DKMT_DESTROYALLOCATION2 destroy = {0};
  destroy.hDevice = reservation->device->device;
  destroy.phAllocationList = reservation->batch;
  destroy.AllocationCount = count;
  // The caller has completed these views' users, not unrelated commands on
  // the device. Native reclamation removes each view's actual IOMMU record.
  destroy.Flags.AssumeNotInUse = 1;
  destroy.Flags.SynchronousDestroy = 1;
  const amdf_status_t status = amdf_kmt_make_status(
      reservation->device->kmt->destroy_allocation(&destroy));
  if (!amdf_status_is_ok(status)) {
    reservation->removal_failure = status;
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_reservation_detach(
    amdf_memory_reservation_t* base, uint32_t ordinal) {
  amdf_windows_xdna_reservation_t* reservation = base->accesses[ordinal].native;
  uint32_t count = 0;
  for (uint32_t i = 0; i < reservation->view_count; ++i) {
    if (reservation->views[i] != 0) {
      reservation->batch[count++] = reservation->views[i];
    }
  }
  const amdf_status_t status =
      amdf_windows_xdna_reservation_remove(reservation, count);
  if (amdf_status_is_ok(status)) {
    memset(reservation->views, 0,
           reservation->view_count * sizeof(*reservation->views));
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_reservation_destroy(
    amdf_memory_reservation_t* base, uint32_t ordinal) {
  amdf_windows_xdna_reservation_t* reservation = base->accesses[ordinal].native;
  amdf_status_t status = amdf_windows_xdna_reservation_detach(base, ordinal);
  if (amdf_status_is_ok(status) && reservation->reservation_byte_length != 0) {
    D3DKMT_FREEGPUVIRTUALADDRESS release = {0};
    release.hAdapter = reservation->device->adapter;
    release.BaseAddress = reservation->reservation_address;
    release.Size = reservation->reservation_byte_length;
    status = amdf_kmt_make_status(
        reservation->device->kmt->free_gpu_virtual_address(&release));
  }
  // Uncertain native removal never releases the VA or authorizes backing
  // reuse. The public owner is still consumed; there is no deferred retry list.
  amdf_free(base->host_allocator, reservation);
  return status;
}

static amdf_status_t amdf_windows_xdna_reservation_prepare(
    amdf_memory_reservation_t* base,
    const amdf_memory_reservation_create_info_t* create,
    const amdf_memory_reservation_group_t* group) {
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < group->access_count && amdf_status_is_ok(status);
       ++i) {
    const uint32_t ordinal = group->access_ordinals[i];
    const amdf_memory_reservation_profile_t* profile =
        &group->profiles[ordinal];
    const amdf_memory_reservation_capabilities_t* limits =
        &profile->capabilities;
    const amdf_memory_reservation_access_t* request =
        &create->accesses[ordinal];
    const uint64_t unit = request->mapping_granularity != 0
                              ? request->mapping_granularity
                              : limits->mapping_granularity;
    const uint64_t view_size = unit < AMDF_WINDOWS_XDNA_MAXIMUM_VIEW_SIZE
                                   ? unit
                                   : AMDF_WINDOWS_XDNA_MAXIMUM_VIEW_SIZE;
    const uint64_t count = (create->byte_length - 1) / view_size + 1;
    if (count > UINT32_MAX ||
        count > (SIZE_MAX - sizeof(amdf_windows_xdna_reservation_t)) /
                    (2 * sizeof(D3DKMT_HANDLE))) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      break;
    }
    amdf_windows_xdna_reservation_t* reservation = NULL;
    status = amdf_calloc_with_trailing(
        base->host_allocator, sizeof(*reservation),
        count * 2 * sizeof(D3DKMT_HANDLE),
        amdf_alignof(amdf_windows_xdna_reservation_t), (void**)&reservation);
    if (!amdf_status_is_ok(status)) {
      break;
    }
    reservation->device = profile->native_device;
    reservation->view_count = (uint32_t)count;
    reservation->view_byte_length = view_size;
    reservation->views = (D3DKMT_HANDLE*)(reservation + 1);
    reservation->batch = reservation->views + count;
    amdf_memory_reservation_access_state_t* access = &base->accesses[ordinal];
    // Each consumer owns a distinct address reservation, even when two logical
    // devices share the process IOMMU domain. No address equality is promised.
    access->native = reservation;
    access->native_owner_ordinal = ordinal;
    access->native_access_ordinal = 0;
    const uint64_t alignment =
        create->minimum_alignment > limits->reservation_granularity
            ? create->minimum_alignment
            : limits->reservation_granularity;
    const uint64_t bias = reservation->device->profile->dma.byte_offset;
    D3DDDI_RESERVEGPUVIRTUALADDRESS reserve = {0};
    reserve.hAdapter = reservation->device->adapter;
    reserve.Size =
        create->byte_length + alignment - limits->reservation_granularity;
    reserve.MinimumAddress = limits->minimum_address - bias;
    const uint64_t native_maximum = limits->maximum_address - bias;
    reserve.MaximumAddress =
        native_maximum == UINT64_MAX ? 0 : native_maximum + 1;
    status = amdf_kmt_make_status(
        reservation->device->kmt->reserve_gpu_virtual_address(&reserve));
    if (amdf_status_is_ok(status)) {
      reservation->reservation_address = reserve.VirtualAddress;
      reservation->reservation_byte_length = reserve.Size;
      const uint64_t address =
          (reserve.VirtualAddress + bias + alignment - 1) & ~(alignment - 1);
      reservation->address = address - bias;
      access->required_access = limits->required_access;
      access->info = (amdf_memory_reservation_access_info_t){
          .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
          .structure_size = sizeof(access->info),
          .device_id = profile->device_id,
          .ordinal = ordinal,
          .mapping_access_ordinal = ordinal,
          .address_domain_ordinal = limits->address_domain_ordinal,
          .access = request->access,
          .address_kind = AMDF_MEMORY_ADDRESS_XDNA_DMA,
          .address = address,
          .byte_length = create->byte_length,
          .mapping_granularity = unit,
          .reset_epoch = profile->reset_epoch,
      };
    }
  }
  return status;
}

static amdf_status_t amdf_windows_xdna_reservation_map(
    amdf_windows_xdna_reservation_t* reservation, uint32_t slot,
    uint64_t byte_length, uint64_t* paging_fence) {
  D3DDDI_MAPGPUVIRTUALADDRESS map = {0};
  map.hPagingQueue = reservation->device->paging_queue;
  map.BaseAddress = reservation->address + slot * reservation->view_byte_length;
  map.hAllocation = reservation->views[slot];
  map.SizeInPages = byte_length / AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT;
  map.Protection.Write = 1;
  const NTSTATUS native_status =
      reservation->device->kmt->map_gpu_virtual_address(&map);
  if (!amdf_kmt_status_is_success_or_pending(native_status)) {
    return amdf_kmt_make_status(native_status);
  }
  if (map.VirtualAddress != map.BaseAddress) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (map.PagingFenceValue > *paging_fence) {
    *paging_fence = map.PagingFenceValue;
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_xdna_reservation_update(
    amdf_memory_reservation_t* base, uint32_t ordinal, uint32_t update_count,
    const amdf_memory_mapping_update_t* updates, bool* mutation_started) {
  amdf_windows_xdna_reservation_t* reservation = base->accesses[ordinal].native;
  const uint64_t unit = reservation->view_byte_length;
  uint32_t old_count = 0;
  for (uint32_t i = 0; i < update_count; ++i) {
    const uint64_t first = updates[i].target_byte_offset / unit;
    const uint64_t end = first + (updates[i].byte_length - 1) / unit + 1;
    for (uint64_t slot = first; slot < end; ++slot) {
      if (reservation->views[slot] != 0) {
        reservation->batch[old_count++] = reservation->views[slot];
      }
    }
  }
  *mutation_started = true;
  amdf_status_t status =
      amdf_windows_xdna_reservation_remove(reservation, old_count);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  // Successful destruction consumes every old handle before new registration
  // begins. A later failure leaves only actually acquired new views to detach.
  for (uint32_t i = 0; i < update_count; ++i) {
    memset(reservation->views + updates[i].target_byte_offset / unit, 0,
           ((updates[i].byte_length - 1) / unit + 1) *
               sizeof(*reservation->views));
  }
  uint32_t new_count = 0;
  uint64_t paging_fence = 0;
  for (uint32_t i = 0; i < update_count && amdf_status_is_ok(status); ++i) {
    const amdf_memory_mapping_update_t* update = &updates[i];
    if (update->memory == NULL) {
      continue;
    }
    const amdf_memory_t* memory = update->memory;
    const uint32_t owner =
        memory->accesses[update->memory_access_ordinal].native_owner_ordinal;
    uint8_t* pointer = amdf_windows_xdna_memory_mapping_source(
        memory->accesses[owner].native,
        memory->info.source_byte_offset + update->source_byte_offset);
    for (uint64_t offset = 0;
         offset < update->byte_length && amdf_status_is_ok(status);) {
      const uint64_t remaining = update->byte_length - offset;
      const uint64_t unit_length = remaining < unit ? remaining : unit;
      const uint32_t slot =
          (uint32_t)((update->target_byte_offset + offset) / unit);
      status = amdf_windows_xdna_memory_register_pages(
          reservation->device, pointer + offset, unit_length,
          &reservation->views[slot]);
      if (amdf_status_is_ok(status)) {
        reservation->batch[new_count++] = reservation->views[slot];
        status = amdf_windows_xdna_reservation_map(reservation, slot,
                                                   unit_length, &paging_fence);
      }
      offset += unit_length;
    }
  }
  amdf_xdna_umd_device_t* device = reservation->device;
  if (amdf_status_is_ok(status) && new_count != 0) {
    D3DDDI_MAKERESIDENT resident = {0};
    resident.hPagingQueue = device->paging_queue;
    resident.NumAllocations = new_count;
    resident.AllocationList = reservation->batch;
    const NTSTATUS native_status = device->kmt->make_resident(&resident);
    if (!amdf_kmt_status_is_success_or_pending(native_status)) {
      status = amdf_kmt_make_status(native_status);
    } else if (resident.NumAllocations != new_count) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    } else if (resident.PagingFenceValue > paging_fence) {
      paging_fence = resident.PagingFenceValue;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_kmt_device_status_wait_for_paging(
        &device->status, device->kmt, device->device,
        device->paging_sync_object, device->paging_fence, paging_fence);
  }
  return status;
}

static const amdf_memory_reservation_vtable_t
    amdf_windows_xdna_reservation_vtable = {
        .prepare = amdf_windows_xdna_reservation_prepare,
        .update = amdf_windows_xdna_reservation_update,
        .detach = amdf_windows_xdna_reservation_detach,
        .destroy = amdf_windows_xdna_reservation_destroy,
};

amdf_status_t amdf_xdna_umd_device_query_memory_reservation_profile(
    amdf_xdna_umd_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_profile_t* out_profile) {
  if (address_domain_ordinal != 0 ||
      !amdf_kmt_api_supports_memory(device->kmt) ||
      device->kmt->reserve_gpu_virtual_address == NULL ||
      device->kmt->free_gpu_virtual_address == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t maximum =
      UINT64_MAX >> (64 - device->profile->dma.address_bit_count);
  const uint64_t bias = device->profile->dma.byte_offset;
  const uint64_t granularity = AMDF_WINDOWS_XDNA_ALLOCATION_GRANULARITY;
  const uint64_t maximum_length = ((maximum - bias) / 2) & ~(granularity - 1);
  uint64_t maximum_alignment = granularity;
  while (maximum_alignment <= maximum_length / 2) {
    maximum_alignment <<= 1;
  }
  *out_profile = (amdf_memory_reservation_profile_t){
      .capabilities =
          {
              .supported_access =
                  AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .required_access =
                  AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
              .address_domain_ordinal = 0,
              .maximum_byte_length = maximum_length,
              .reservation_granularity = granularity,
              .maximum_alignment = maximum_alignment,
              .mapping_granularity = AMDF_WINDOWS_XDNA_ADDRESS_ALIGNMENT,
              .minimum_address = granularity + bias,
              .maximum_address = maximum,
          },
      .vtable = &amdf_windows_xdna_reservation_vtable,
      .native_device = device,
  };
  return AMDF_STATUS_OK;
}
