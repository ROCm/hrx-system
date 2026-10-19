// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/memory_reservation.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/device.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"
#include "libamdf/src/memory_resource.h"

static amdf_gpu_umd_memory_reservation_t* amdf_gpu_memory_reservation_native(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal) {
  return reservation
      ->accesses[reservation->accesses[access_ordinal].native_owner_ordinal]
      .native;
}

static amdf_status_t amdf_gpu_memory_reservation_apply(
    amdf_memory_reservation_t* base, uint32_t access_ordinal,
    uint32_t update_count, const amdf_memory_mapping_update_t* updates,
    bool* mutation_started, amdf_gpu_umd_kernel_queue_t* queue,
    uint64_t submission, bool* accepted) {
  amdf_gpu_umd_memory_reservation_t* native_reservation =
      amdf_gpu_memory_reservation_native(base, access_ordinal);
  const uint32_t native_ordinal =
      base->accesses[access_ordinal].native_access_ordinal;
  amdf_gpu_umd_mapping_update_t* native = NULL;
  amdf_status_t status = amdf_calloc_array(
      base->host_allocator, update_count, sizeof(*native),
      amdf_alignof(amdf_gpu_umd_mapping_update_t), (void**)&native);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (uint32_t i = 0; i < update_count; ++i) {
    native[i].target_byte_offset = updates[i].target_byte_offset;
    native[i].byte_length = updates[i].byte_length;
    native[i].access = updates[i].access;
    if (updates[i].memory != NULL) {
      const amdf_memory_t* memory = updates[i].memory;
      const uint32_t owner = memory->accesses[updates[i].memory_access_ordinal]
                                 .native_owner_ordinal;
      native[i].memory = memory->accesses[owner].native;
      native[i].source_byte_offset =
          memory->info.source_byte_offset + updates[i].source_byte_offset;
    }
  }
  status = queue != NULL
               ? amdf_gpu_umd_memory_reservation_update_on_queue(
                     native_reservation, queue, native_ordinal, update_count,
                     native, submission, mutation_started, accepted)
               : amdf_gpu_umd_memory_reservation_update(
                     native_reservation, native_ordinal, update_count, native,
                     mutation_started);
  amdf_free(base->host_allocator, native);
  return status;
}

static amdf_status_t amdf_gpu_memory_reservation_update(
    amdf_memory_reservation_t* base, uint32_t access_ordinal,
    uint32_t update_count, const amdf_memory_mapping_update_t* updates,
    bool* mutation_started) {
  return amdf_gpu_memory_reservation_apply(base, access_ordinal, update_count,
                                           updates, mutation_started, NULL, 0,
                                           NULL);
}

amdf_status_t amdf_gpu_memory_reservation_update_on_queue(
    amdf_memory_reservation_t* base, amdf_gpu_umd_kernel_queue_t* queue,
    uint32_t access_ordinal, uint32_t update_count,
    const amdf_memory_mapping_update_t* updates, uint64_t submission,
    bool* mutation_started, bool* accepted) {
  return amdf_gpu_memory_reservation_apply(base, access_ordinal, update_count,
                                           updates, mutation_started, queue,
                                           submission, accepted);
}

static amdf_status_t amdf_gpu_memory_reservation_detach(
    amdf_memory_reservation_t* base, uint32_t access_ordinal) {
  return amdf_gpu_umd_memory_reservation_detach(
      amdf_gpu_memory_reservation_native(base, access_ordinal),
      base->accesses[access_ordinal].native_access_ordinal);
}

static amdf_status_t amdf_gpu_memory_reservation_destroy(
    amdf_memory_reservation_t* base, uint32_t owner_ordinal) {
  return amdf_gpu_umd_memory_reservation_destroy(
      base->accesses[owner_ordinal].native);
}

static amdf_status_t amdf_gpu_memory_reservation_prepare(
    amdf_memory_reservation_t* reservation,
    const amdf_memory_reservation_create_info_t* create_info,
    const amdf_memory_reservation_group_t* group) {
  const amdf_allocator_t allocator = reservation->host_allocator;
  const uint32_t count = group->access_count;
  const uint32_t* ordinals = group->access_ordinals;
  for (uint32_t i = 0; i < count; ++i) {
    for (uint32_t j = 0; j < i; ++j) {
      const amdf_memory_reservation_access_t* a =
          &create_info->accesses[ordinals[i]];
      const amdf_memory_reservation_access_t* b =
          &create_info->accesses[ordinals[j]];
      if (a->access != b->access && amdf_gpu_umd_devices_share_mapping_domain(
                                        amdf_gpu_device_get_umd(a->device),
                                        amdf_gpu_device_get_umd(b->device))) {
        return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      }
    }
  }
  amdf_gpu_umd_device_t** devices = NULL;
  amdf_status_t status =
      amdf_calloc_array(allocator, count, sizeof(*devices),
                        amdf_alignof(amdf_gpu_umd_device_t*), (void**)&devices);
  amdf_gpu_umd_memory_reservation_t* native = NULL;
  if (amdf_status_is_ok(status)) {
    for (uint32_t i = 0; i < count; ++i) {
      devices[i] =
          amdf_gpu_device_get_umd(create_info->accesses[ordinals[i]].device);
    }
    status = amdf_gpu_umd_memory_reservation_create(
        allocator, count, devices, create_info->byte_length,
        create_info->minimum_alignment, &native);
  }
  amdf_free(allocator, devices);
  if (amdf_status_is_ok(status)) {
    reservation->accesses[ordinals[0]].native = native;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t ordinal = ordinals[i];
      amdf_memory_reservation_access_state_t* access =
          &reservation->accesses[ordinal];
      access->info = (amdf_memory_reservation_access_info_t){
          .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
          .structure_size = sizeof(access->info),
          .device_id = amdf_gpu_device_get_info(access->device)->id,
          .ordinal = ordinal,
          .address_domain_ordinal =
              create_info->accesses[ordinal].address_domain_ordinal,
          .access = create_info->accesses[ordinal].access,
          .byte_length = create_info->byte_length,
          .reset_epoch = amdf_gpu_device_query_reset_epoch(access->device),
      };
      amdf_gpu_umd_memory_reservation_query_access(
          native, i, &access->info.address,
          &access->info.mapping_access_ordinal,
          &access->info.mapping_granularity, &access->required_access);
      access->info.mapping_access_ordinal =
          ordinals[access->info.mapping_access_ordinal];
      if (create_info->accesses[ordinal].mapping_granularity != 0) {
        access->info.mapping_granularity =
            create_info->accesses[ordinal].mapping_granularity;
      }
    }
  }
  return status;
}

static const amdf_memory_reservation_vtable_t
    amdf_gpu_memory_reservation_vtable = {
        .prepare = amdf_gpu_memory_reservation_prepare,
        .update = amdf_gpu_memory_reservation_update,
        .detach = amdf_gpu_memory_reservation_detach,
        .destroy = amdf_gpu_memory_reservation_destroy,
};

amdf_status_t amdf_gpu_device_query_memory_reservation_profile(
    amdf_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_profile_t* out_profile) {
  amdf_memory_reservation_profile_t profile = {
      .vtable = &amdf_gpu_memory_reservation_vtable,
  };
  const amdf_status_t status =
      amdf_gpu_umd_query_memory_reservation_capabilities(
          amdf_gpu_device_get_umd(device), address_domain_ordinal,
          &profile.capabilities);
  if (amdf_status_is_ok(status)) {
    *out_profile = profile;
  }
  return status;
}
