// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/host_reservation.h"

#include "libamdf/src/memory_resource.h"
#include "libamdf/src/platform/host_reservation.h"

// Admission examines the explicitly supplied backing's prepared consumers, not
// installed mappings. Every SVA consumer must keep that backing natively
// pinned.
static amdf_status_t amdf_host_reservation_validate_source(
    const amdf_memory_reservation_t* reservation, uint32_t owner_ordinal,
    const amdf_memory_mapping_update_t* update) {
  const amdf_memory_t* memory = update->memory;
  if (memory == NULL) {
    return AMDF_STATUS_OK;
  }
  const amdf_memory_map_flags_t required =
      AMDF_MEMORY_MAP_FLAG_READ |
      ((update->access & AMDF_MEMORY_ACCESS_WRITE) != 0
           ? AMDF_MEMORY_MAP_FLAG_WRITE
           : 0);
  if ((memory->info.flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) == 0 ||
      (required & ~memory->host_mapping.supported_access) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  for (uint32_t i = 0; i < reservation->access_count; ++i) {
    const amdf_memory_reservation_access_state_t* target =
        &reservation->accesses[i];
    if (target->native_owner_ordinal != owner_ordinal ||
        target->device == NULL) {
      continue;
    }
    bool prepared = false;
    for (uint32_t j = 0; j < memory->info.access_count; ++j) {
      const amdf_memory_access_state_t* source = &memory->accesses[j];
      if (source->device == target->device &&
          source->info.address_domain_ordinal ==
              target->info.address_domain_ordinal &&
          (source->info.flags & AMDF_MEMORY_FLAG_MAPPING_SOURCE) != 0 &&
          (update->access & ~source->info.access) == 0) {
        prepared = true;
        break;
      }
    }
    if (!prepared) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_host_reservation_update(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_memory_mapping_update_t* updates,
    bool* mutation_started) {
  const uint32_t owner_ordinal =
      reservation->accesses[access_ordinal].native_owner_ordinal;
  const amdf_memory_reservation_access_state_t* owner =
      &reservation->accesses[owner_ordinal];
  for (uint32_t i = 0; i < update_count; ++i) {
    const amdf_status_t status = amdf_host_reservation_validate_source(
        reservation, owner_ordinal, &updates[i]);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < update_count && amdf_status_is_ok(status); ++i) {
    const amdf_memory_mapping_update_t* update = &updates[i];
    void* pointer = (uint8_t*)owner->native + update->target_byte_offset;
    *mutation_started = true;
    if (update->memory != NULL) {
      amdf_memory_t* source = update->memory;
      const uint32_t source_owner = source->backing_access_ordinal;
      status = amdf_platform_host_reservation_replace(
          pointer, update->byte_length, owner->info.mapping_granularity,
          source->accesses[source_owner].vtable->host_mapping_source(
              source, source_owner),
          source->info.source_byte_offset + update->source_byte_offset,
          update->access);
      if (amdf_status_is_ok(status) &&
          owner->host.prepare_translations != NULL) {
        owner->host.prepare_translations(pointer, update->byte_length,
                                         owner->host.page_size);
      }
    } else {
      status =
          amdf_platform_host_reservation_clear(pointer, update->byte_length);
    }
  }
  return status;
}

static amdf_status_t amdf_host_reservation_detach(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal) {
  const amdf_memory_reservation_access_state_t* owner =
      &reservation->accesses[reservation->accesses[access_ordinal]
                                 .native_owner_ordinal];
  return amdf_platform_host_reservation_clear(owner->native,
                                              owner->info.byte_length);
}

static amdf_status_t amdf_host_reservation_destroy(
    amdf_memory_reservation_t* reservation, uint32_t owner_ordinal) {
  const amdf_memory_reservation_access_state_t* owner =
      &reservation->accesses[owner_ordinal];
  return amdf_platform_host_reservation_destroy(owner->native,
                                                owner->info.byte_length);
}

static amdf_status_t amdf_host_reservation_prepare(
    amdf_memory_reservation_t* reservation,
    const amdf_memory_reservation_create_info_t* create_info,
    const amdf_memory_reservation_group_t* group) {
  const uint32_t owner_ordinal = group->access_ordinals[0];
  amdf_memory_reservation_access_state_t* owner =
      &reservation->accesses[owner_ordinal];
  const amdf_memory_access_t permissions =
      create_info->accesses[owner_ordinal].access;
  uint64_t alignment = create_info->minimum_alignment;
  uint64_t native_alignment = 0;
  amdf_memory_access_t required_access = 0;
  for (uint32_t i = 0; i < group->access_count; ++i) {
    const uint32_t ordinal = group->access_ordinals[i];
    const amdf_memory_reservation_profile_t* profile =
        &group->profiles[ordinal];
    const amdf_memory_reservation_access_t* request =
        &create_info->accesses[ordinal];
    if (request->access != permissions) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if (alignment < profile->capabilities.reservation_granularity) {
      alignment = profile->capabilities.reservation_granularity;
    }
    if (native_alignment < profile->capabilities.reservation_granularity) {
      native_alignment = profile->capabilities.reservation_granularity;
    }
    required_access |= profile->capabilities.required_access;
    if (profile->host.prepare_translations != NULL) {
      owner->host.prepare_translations = profile->host.prepare_translations;
      if (owner->host.page_size == 0 ||
          owner->host.page_size > profile->host.page_size) {
        owner->host.page_size = profile->host.page_size;
      }
    }
  }
  // Set the teardown extent before acquiring the address range. A later native
  // address-envelope failure leaves that range in the unpublished owner.
  owner->info.byte_length = create_info->byte_length;
  amdf_status_t status = amdf_platform_host_reservation_create(
      create_info->byte_length, alignment > native_alignment ? alignment : 0,
      &owner->native);
  for (uint32_t i = 0; i < group->access_count && amdf_status_is_ok(status);
       ++i) {
    const uint32_t ordinal = group->access_ordinals[i];
    const amdf_memory_reservation_profile_t* profile =
        &group->profiles[ordinal];
    const amdf_memory_reservation_capabilities_t* capabilities =
        &profile->capabilities;
    const uint64_t base = (uintptr_t)owner->native;
    const uint64_t bias = profile->host.address_byte_offset;
    if (base > capabilities->maximum_address ||
        bias > capabilities->maximum_address - base ||
        base + bias < capabilities->minimum_address ||
        create_info->byte_length - 1 >
            capabilities->maximum_address - base - bias) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    } else {
      amdf_memory_reservation_access_state_t* access =
          &reservation->accesses[ordinal];
      access->required_access = required_access;
      access->info = (amdf_memory_reservation_access_info_t){
          .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
          .structure_size = sizeof(access->info),
          .device_id = profile->device_id,
          .ordinal = ordinal,
          .mapping_access_ordinal = owner_ordinal,
          .address_domain_ordinal = capabilities->address_domain_ordinal,
          .access = permissions,
          .address_kind = profile->host.address_kind,
          .address = base + bias,
          .byte_length = create_info->byte_length,
          .mapping_granularity =
              create_info->accesses[ordinal].mapping_granularity != 0
                  ? create_info->accesses[ordinal].mapping_granularity
                  : capabilities->mapping_granularity,
          .reset_epoch = profile->reset_epoch,
      };
    }
  }
  return status;
}

const amdf_memory_reservation_vtable_t amdf_host_reservation_vtable = {
    .prepare = amdf_host_reservation_prepare,
    .update = amdf_host_reservation_update,
    .detach = amdf_host_reservation_detach,
    .destroy = amdf_host_reservation_destroy,
};
