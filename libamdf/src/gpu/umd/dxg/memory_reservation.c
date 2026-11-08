// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/memory_reservation.h"

#include "libamdf/src/allocator.h"

typedef struct amdf_gpu_dxg_memory_reservation_attachment_t {
  // Live provider device forwarded to contextual native operations.
  void* native_context;
  // Native operation table selected by this device provider.
  const amdf_gpu_dxg_memory_native_api_t* native_api;
  // CPU mapping of paging progress, or NULL when native waiting is required.
  const volatile uint64_t* paging_progress_fence;
  // Query adapter owning GPU virtual-address reservations.
  uint32_t adapter_handle;
  // Logical KMT device owning physical allocations.
  uint32_t device_handle;
  // Paging queue used for mapping operations.
  uint32_t paging_queue_handle;
  // Paging synchronization object waited from the host.
  uint32_t paging_synchronization_object_handle;
} amdf_gpu_dxg_memory_reservation_attachment_t;

struct amdf_gpu_dxg_memory_reservation_t {
  // Allocator for this owner and its immutable device array.
  amdf_allocator_t host_allocator;
  // Base of the complete native reservation, including alignment slack.
  uint64_t reservation_address;
  // Native extent required by the address-release operation.
  uint64_t reservation_byte_length;
  // Aligned application-visible base in the native reservation.
  uint64_t address;
  // Application-visible extent shared by every admitted device.
  uint64_t byte_length;
  // Provider-supported replacement unit shared by this mapping domain.
  uint64_t mapping_granularity;
  // Access that every mapping must include on this physical adapter.
  amdf_memory_access_t required_access;
  // Device whose paging timeline owns the latest mutation.
  uint32_t publisher_ordinal;
  // Compact live device attachments borrowed through destruction.
  amdf_gpu_dxg_memory_reservation_attachment_t attachments[];
};

bool amdf_gpu_dxg_devices_share_mapping_domain(
    const amdf_gpu_dxg_memory_reservation_device_t* a,
    const amdf_gpu_dxg_memory_reservation_device_t* b) {
  return a->adapter_luid == b->adapter_luid &&
         a->physical_adapter_index == b->physical_adapter_index;
}

static amdf_memory_access_t amdf_gpu_dxg_required_mapping_access(
    const amdf_gpu_dxg_memory_reservation_device_t* device) {
  // MapGpuVirtualAddress cannot deny reads. Hardware lacking individual PTE
  // protections cannot promise read-only or non-executable mappings either.
  return AMDF_MEMORY_ACCESS_READ |
         (device->read_only_memory_supported ? 0 : AMDF_MEMORY_ACCESS_WRITE) |
         (device->no_execute_memory_supported ? 0 : AMDF_MEMORY_ACCESS_EXECUTE);
}

amdf_status_t amdf_gpu_dxg_query_memory_reservation_capabilities(
    const amdf_gpu_dxg_memory_reservation_device_t* device,
    uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  if (address_domain_ordinal != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint32_t bits = device->virtual_address_bit_count;
  const uint64_t maximum_address =
      bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
  const uint64_t maximum_length =
      (maximum_address / 2) & ~(AMDF_GPU_DXG_RESERVATION_GRANULARITY - 1);
  uint64_t maximum_alignment = 1;
  while (maximum_alignment <= maximum_length / 2) {
    maximum_alignment <<= 1;
  }
  *out_capabilities = (amdf_memory_reservation_capabilities_t){
      .supported_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                          AMDF_MEMORY_ACCESS_EXECUTE,
      .required_access = amdf_gpu_dxg_required_mapping_access(device),
      .supported_flags = AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS,
      .maximum_byte_length = maximum_length,
      .reservation_granularity = AMDF_GPU_DXG_RESERVATION_GRANULARITY,
      .maximum_alignment = maximum_alignment,
      .mapping_granularity = device->mapping_granularity,
      .minimum_address = AMDF_GPU_DXG_RESERVATION_GRANULARITY,
      .maximum_address = maximum_address,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_dxg_memory_reservation_create(
    amdf_allocator_t host_allocator, uint32_t device_count,
    amdf_gpu_dxg_memory_reservation_describe_device_fn_t describe_device,
    const void* describe_context, uint64_t byte_length,
    uint64_t minimum_alignment, uint64_t common_address,
    amdf_gpu_dxg_memory_reservation_t** out_reservation) {
  amdf_gpu_dxg_memory_reservation_t* reservation = NULL;
  amdf_status_t status = amdf_calloc_with_trailing(
      host_allocator, sizeof(*reservation),
      device_count * sizeof(reservation->attachments[0]),
      amdf_alignof(amdf_gpu_dxg_memory_reservation_t), (void**)&reservation);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  reservation->host_allocator = host_allocator;
  reservation->byte_length = byte_length;
  amdf_gpu_dxg_memory_reservation_device_t first_device = {0};
  for (uint32_t i = 0; i < device_count; ++i) {
    const amdf_gpu_dxg_memory_reservation_device_t device =
        describe_device(describe_context, i);
    if (i == 0) {
      first_device = device;
      reservation->required_access =
          amdf_gpu_dxg_required_mapping_access(&device);
      reservation->mapping_granularity = device.mapping_granularity;
    } else if (!amdf_gpu_dxg_devices_share_mapping_domain(&first_device,
                                                          &device)) {
      amdf_free(host_allocator, reservation);
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    reservation->attachments[i] =
        (amdf_gpu_dxg_memory_reservation_attachment_t){
            .native_context = device.native_context,
            .native_api = device.native_api,
            .paging_progress_fence = device.paging_progress_fence,
            .adapter_handle = device.adapter_handle,
            .device_handle = device.device_handle,
            .paging_queue_handle = device.paging_queue_handle,
            .paging_synchronization_object_handle =
                device.paging_synchronization_object_handle,
        };
  }

  const uint64_t alignment =
      minimum_alignment > AMDF_GPU_DXG_RESERVATION_GRANULARITY
          ? minimum_alignment
          : AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  const uint64_t reservation_byte_length =
      common_address != 0
          ? byte_length
          : byte_length + alignment - AMDF_GPU_DXG_RESERVATION_GRANULARITY;
  uint64_t reservation_address = 0;
  const amdf_gpu_dxg_memory_reservation_attachment_t* attachment =
      &reservation->attachments[0];
  status = attachment->native_api->reserve_address(
      attachment->native_context, attachment->adapter_handle, common_address,
      AMDF_GPU_DXG_RESERVATION_GRANULARITY, reservation_byte_length,
      &reservation_address);
  if (amdf_status_is_ok(status)) {
    reservation->reservation_address = reservation_address;
    reservation->reservation_byte_length = reservation_byte_length;
    reservation->address =
        (reservation_address + alignment - 1) & ~(alignment - 1);
    *out_reservation = reservation;
  } else {
    amdf_free(host_allocator, reservation);
  }
  return status;
}

void amdf_gpu_dxg_memory_reservation_query_access(
    const amdf_gpu_dxg_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint64_t* out_address,
    uint32_t* out_mapping_access_ordinal, uint64_t* out_mapping_granularity,
    amdf_memory_access_t* out_required_access) {
  *out_address = reservation->address;
  *out_mapping_access_ordinal = 0;
  *out_mapping_granularity = reservation->mapping_granularity;
  *out_required_access = reservation->required_access;
  (void)access_ordinal;
}

static amdf_status_t amdf_gpu_dxg_memory_reservation_map(
    const amdf_gpu_dxg_memory_reservation_attachment_t* attachment,
    uint64_t address, uint64_t byte_length, uint32_t allocation_handle,
    uint64_t source_byte_offset, amdf_memory_access_t access,
    uint64_t* last_paging_fence) {
  uint64_t mapped_address = 0;
  uint64_t paging_fence = 0;
  const amdf_status_t status = attachment->native_api->map_allocation(
      attachment->native_context, attachment->paging_queue_handle, address,
      allocation_handle, source_byte_offset, byte_length, access,
      &mapped_address, &paging_fence);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (mapped_address != address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (paging_fence > *last_paging_fence) {
    *last_paging_fence = paging_fence;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_dxg_memory_reservation_update(
    amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_gpu_umd_mapping_update_t* updates,
    bool* mutation_started) {
  const amdf_gpu_dxg_memory_reservation_attachment_t* attachment =
      &reservation->attachments[access_ordinal];
  reservation->publisher_ordinal = access_ordinal;
  uint64_t fence = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  *mutation_started = true;
  for (uint32_t i = 0; i < update_count && amdf_status_is_ok(status); ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    uint64_t completed = 0;
    while (completed < entry->byte_length && amdf_status_is_ok(status)) {
      amdf_gpu_dxg_memory_span_t source = {
          .byte_length = entry->byte_length,
      };
      if (entry->memory != NULL) {
        source = amdf_gpu_dxg_memory_span(
            (const amdf_gpu_dxg_memory_t*)entry->memory,
            entry->source_byte_offset + completed);
      }
      const uint64_t remaining = entry->byte_length - completed;
      const uint64_t length =
          remaining < source.byte_length ? remaining : source.byte_length;
      status = amdf_gpu_dxg_memory_reservation_map(
          attachment,
          reservation->address + entry->target_byte_offset + completed, length,
          source.allocation_handle, source.byte_offset, entry->access, &fence);
      completed += length;
    }
  }
  if (amdf_status_is_ok(status)) {
    status = attachment->native_api->wait_for_paging(
        attachment->native_context, attachment->device_handle,
        attachment->paging_synchronization_object_handle,
        attachment->paging_progress_fence, fence);
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_memory_reservation_detach(
    amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t access_ordinal) {
  (void)access_ordinal;
  const amdf_gpu_dxg_memory_reservation_attachment_t* attachment =
      &reservation->attachments[reservation->publisher_ordinal];
  uint64_t fence = 0;
  amdf_status_t status = amdf_gpu_dxg_memory_reservation_map(
      attachment, reservation->address, reservation->byte_length, 0, 0, 0,
      &fence);
  if (amdf_status_is_ok(status)) {
    status = attachment->native_api->wait_for_paging(
        attachment->native_context, attachment->device_handle,
        attachment->paging_synchronization_object_handle,
        attachment->paging_progress_fence, fence);
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_memory_reservation_destroy(
    amdf_gpu_dxg_memory_reservation_t* reservation) {
  const amdf_gpu_dxg_memory_reservation_attachment_t* attachment =
      &reservation->attachments[reservation->publisher_ordinal];
  amdf_status_t status = amdf_gpu_dxg_memory_reservation_detach(reservation, 0);
  if (amdf_status_is_ok(status)) {
    status = attachment->native_api->free_address(
        attachment->native_context, attachment->adapter_handle,
        reservation->reservation_address, reservation->reservation_byte_length);
  }
  amdf_free(reservation->host_allocator, reservation);
  return status;
}

uint64_t amdf_gpu_dxg_memory_reservation_mapping_operation_count(
    const amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t update_count,
    const amdf_gpu_umd_mapping_update_t* updates) {
  (void)reservation;
  uint64_t operation_count = 0;
  for (uint32_t i = 0; i < update_count; ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    uint64_t position = 0;
    do {
      const uint64_t span =
          entry->memory == NULL
              ? entry->byte_length
              : amdf_gpu_dxg_memory_span(
                    (const amdf_gpu_dxg_memory_t*)entry->memory,
                    entry->source_byte_offset + position)
                    .byte_length;
      const uint64_t remaining = entry->byte_length - position;
      position += span < remaining ? span : remaining;
      ++operation_count;
    } while (position < entry->byte_length);
  }
  return operation_count;
}

void amdf_gpu_dxg_memory_reservation_visit_mapping_operations(
    const amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t update_count,
    const amdf_gpu_umd_mapping_update_t* updates,
    amdf_gpu_dxg_mapping_operation_visitor_fn_t visitor,
    void* visitor_context) {
  uint32_t ordinal = 0;
  for (uint32_t i = 0; i < update_count; ++i) {
    const amdf_gpu_umd_mapping_update_t* entry = &updates[i];
    uint64_t position = 0;
    do {
      amdf_gpu_dxg_mapping_operation_t operation = {
          .target_address =
              reservation->address + entry->target_byte_offset + position,
          .byte_length = entry->byte_length - position,
          .access = entry->access,
      };
      if (entry->memory != NULL) {
        const amdf_gpu_dxg_memory_span_t source = amdf_gpu_dxg_memory_span(
            (const amdf_gpu_dxg_memory_t*)entry->memory,
            entry->source_byte_offset + position);
        if (operation.byte_length > source.byte_length) {
          operation.byte_length = source.byte_length;
        }
        operation.allocation_handle = source.allocation_handle;
        operation.source_byte_offset = source.byte_offset;
      }
      visitor(visitor_context, ordinal++, &operation);
      position += operation.byte_length;
    } while (position < entry->byte_length);
  }
}

void amdf_gpu_dxg_memory_reservation_select_publisher(
    amdf_gpu_dxg_memory_reservation_t* reservation, uint32_t access_ordinal) {
  reservation->publisher_ordinal = access_ordinal;
}

amdf_allocator_t amdf_gpu_dxg_memory_reservation_host_allocator(
    const amdf_gpu_dxg_memory_reservation_t* reservation) {
  return reservation->host_allocator;
}
