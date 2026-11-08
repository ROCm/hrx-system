// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_reservation.h"

#include <stdlib.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/host_memory.h"
#include "libamdf/src/host_reservation.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/kernel_queue.h"
#include "libamdf/src/memory_resource.h"
#include "libamdf/src/platform/host_reservation.h"
#include "libamdf/src/structure.h"

amdf_status_t AMDF_CALL
amdf_instance_query_host_memory_reservation_capabilities(
    amdf_instance_t* instance,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  if (instance == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_output(
      out_capabilities, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      sizeof(*out_capabilities));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_memory_reservation_capabilities_t capabilities = {0};
  status = amdf_platform_host_reservation_query_capabilities(&capabilities);
  if (amdf_status_is_ok(status)) {
    capabilities.type = out_capabilities->type;
    capabilities.structure_size = out_capabilities->structure_size;
    capabilities.next = out_capabilities->next;
    *out_capabilities = capabilities;
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_device_query_memory_reservation_capabilities(
    amdf_device_t* device, uint32_t address_domain_ordinal,
    amdf_memory_reservation_capabilities_t* out_capabilities) {
  if (device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_output(
      out_capabilities, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      sizeof(*out_capabilities));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (device->vtable->query_memory_reservation_profile == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_memory_reservation_profile_t profile = {0};
  status = device->vtable->query_memory_reservation_profile(
      device, address_domain_ordinal, &profile);
  if (amdf_status_is_ok(status)) {
    amdf_memory_reservation_capabilities_t capabilities = profile.capabilities;
    capabilities.type = out_capabilities->type;
    capabilities.structure_size = out_capabilities->structure_size;
    capabilities.next = out_capabilities->next;
    *out_capabilities = capabilities;
  }
  return status;
}

static amdf_status_t amdf_memory_reservation_query_profile(
    amdf_instance_t* instance,
    const amdf_memory_reservation_create_info_t* create_info, uint32_t ordinal,
    amdf_memory_reservation_profile_t* out_profile) {
  const amdf_memory_reservation_access_t* access =
      &create_info->accesses[ordinal];
  if ((access->device != NULL &&
       access->device->provider_instance != instance) ||
      access->access == 0 ||
      (access->mapping_granularity & (access->mapping_granularity - 1)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  for (uint32_t j = 0; j < ordinal; ++j) {
    if (create_info->accesses[j].device == access->device) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  amdf_status_t status = AMDF_STATUS_OK;
  if (access->device == NULL) {
    if (access->address_domain_ordinal != AMDF_ADDRESS_DOMAIN_ORDINAL_NONE) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    out_profile->vtable = &amdf_host_reservation_vtable;
    out_profile->host.address_kind = AMDF_MEMORY_ADDRESS_HOST;
    status = amdf_platform_host_reservation_query_capabilities(
        &out_profile->capabilities);
  } else {
    if (access->device->vtable->query_memory_reservation_profile == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    status = access->device->vtable->query_memory_reservation_profile(
        access->device, access->address_domain_ordinal, out_profile);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_memory_reservation_capabilities_t* capabilities =
      &out_profile->capabilities;
  if ((create_info->flags & ~capabilities->supported_flags) != 0 ||
      (access->access & ~capabilities->supported_access) != 0 ||
      (access->access & capabilities->required_access) !=
          capabilities->required_access ||
      create_info->byte_length > capabilities->maximum_byte_length ||
      create_info->byte_length % capabilities->reservation_granularity != 0 ||
      create_info->byte_length % capabilities->mapping_granularity != 0 ||
      create_info->minimum_alignment > capabilities->maximum_alignment ||
      (access->mapping_granularity != 0 &&
       access->mapping_granularity % capabilities->mapping_granularity != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_reservation_acquire_common_address(
    amdf_memory_reservation_t* reservation,
    const amdf_memory_reservation_create_info_t* create_info,
    const amdf_memory_reservation_profile_t* profiles) {
  amdf_memory_reservation_capabilities_t host = {0};
  amdf_status_t status =
      amdf_platform_host_reservation_query_capabilities(&host);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint64_t minimum_address = host.minimum_address;
  uint64_t maximum_address = host.maximum_address;
  uint64_t alignment = create_info->minimum_alignment;
  if (alignment < host.reservation_granularity) {
    alignment = host.reservation_granularity;
  }
  uint64_t maximum_alignment = host.maximum_alignment;
  for (uint32_t i = 0; i < create_info->access_count; ++i) {
    const amdf_memory_reservation_capabilities_t* capabilities =
        &profiles[i].capabilities;
    if (minimum_address < capabilities->minimum_address) {
      minimum_address = capabilities->minimum_address;
    }
    if (maximum_address > capabilities->maximum_address) {
      maximum_address = capabilities->maximum_address;
    }
    if (alignment < capabilities->reservation_granularity) {
      alignment = capabilities->reservation_granularity;
    }
    if (maximum_alignment > capabilities->maximum_alignment) {
      maximum_alignment = capabilities->maximum_alignment;
    }
  }
  if (alignment > maximum_alignment || minimum_address > maximum_address ||
      create_info->byte_length > host.maximum_byte_length ||
      create_info->byte_length % host.reservation_granularity != 0 ||
      create_info->byte_length - 1 > maximum_address - minimum_address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  reservation->common_byte_length = create_info->byte_length;
  return amdf_platform_host_reservation_create(
      create_info->byte_length, alignment, minimum_address, maximum_address,
      &reservation->common_address);
}

amdf_status_t AMDF_CALL amdf_memory_reservation_create(
    amdf_instance_t* instance,
    const amdf_memory_reservation_create_info_t* create_info,
    amdf_memory_reservation_t** out_reservation) {
  if (instance == NULL || out_reservation == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if ((create_info->flags &
       ~AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS) != 0 ||
      create_info->byte_length == 0 || create_info->access_count == 0 ||
      create_info->accesses == NULL ||
      (create_info->minimum_alignment & (create_info->minimum_alignment - 1)) !=
          0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_allocator_t allocator = instance->host_allocator;
  const uint32_t count = create_info->access_count;
  amdf_memory_reservation_profile_t* profiles = NULL;
  status = amdf_calloc_array(
      allocator, count, sizeof(*profiles) + sizeof(uint32_t),
      amdf_alignof(amdf_memory_reservation_profile_t), (void**)&profiles);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  uint32_t* ordinals = (uint32_t*)(profiles + count);
  for (uint32_t i = 0; i < count && amdf_status_is_ok(status); ++i) {
    status = amdf_memory_reservation_query_profile(instance, create_info, i,
                                                   &profiles[i]);
  }
  amdf_memory_reservation_t* reservation = NULL;
  if (amdf_status_is_ok(status)) {
    const size_t access_count = count;
    if (access_count >
        (SIZE_MAX - sizeof(*reservation)) / sizeof(*reservation->accesses)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    } else {
      status = amdf_calloc_with_trailing(
          allocator, sizeof(*reservation),
          count * sizeof(*reservation->accesses),
          amdf_alignof(amdf_memory_reservation_t), (void**)&reservation);
    }
  }
  if (amdf_status_is_ok(status)) {
    reservation->host_allocator = allocator;
    reservation->access_count = count;
    reservation->accesses =
        (amdf_memory_reservation_access_state_t*)(reservation + 1);
    if ((create_info->flags &
         AMDF_MEMORY_RESERVATION_FLAG_REQUIRE_COMMON_ADDRESS) != 0) {
      status = amdf_memory_reservation_acquire_common_address(
          reservation, create_info, profiles);
    }
    for (uint32_t i = 0; i < count && amdf_status_is_ok(status); ++i) {
      if (reservation->accesses[i].vtable != NULL) {
        continue;
      }
      uint32_t group_count = 0;
      for (uint32_t j = i; j < count; ++j) {
        if (profiles[j].vtable == profiles[i].vtable) {
          amdf_memory_reservation_access_state_t* access =
              &reservation->accesses[j];
          access->device = create_info->accesses[j].device;
          access->vtable = profiles[j].vtable;
          access->native_owner_ordinal = i;
          access->native_access_ordinal = group_count;
          ordinals[group_count++] = j;
        }
      }
      const amdf_memory_reservation_group_t group = {
          .access_count = group_count,
          .access_ordinals = ordinals,
          .profiles = profiles,
      };
      status = profiles[i].vtable->prepare(reservation, create_info, &group);
    }
  }
  amdf_free(allocator, profiles);
  if (amdf_status_is_ok(status)) {
    *out_reservation = reservation;
  } else if (reservation != NULL) {
    const amdf_status_t release_status =
        amdf_memory_reservation_destroy(reservation);
    if (!amdf_status_is_ok(release_status)) {
      status = release_status;
    }
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_reservation_map(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping) {
  if (reservation == NULL || out_mapping == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_input(
      map_info, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO, sizeof(*map_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (access_ordinal >= reservation->access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_reservation_access_state_t* access =
      &reservation->accesses[access_ordinal];
  if (access->device != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (map_info->byte_length == 0 ||
      map_info->byte_offset > access->info.byte_length ||
      map_info->byte_length >
          access->info.byte_length - map_info->byte_offset ||
      map_info->flags == 0 ||
      (map_info->flags &
       ~(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_memory_map_flags_t supported =
      ((access->info.access & AMDF_MEMORY_ACCESS_READ) != 0
           ? AMDF_MEMORY_MAP_FLAG_READ
           : 0) |
      ((access->info.access & AMDF_MEMORY_ACCESS_WRITE) != 0
           ? AMDF_MEMORY_MAP_FLAG_WRITE
           : 0);
  if ((map_info->flags & ~supported) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return amdf_host_memory_create_view(reservation->host_allocator, NULL,
                                      (void*)(uintptr_t)access->info.address,
                                      map_info, out_mapping);
}

amdf_status_t AMDF_CALL amdf_memory_reservation_query_access_info(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    amdf_memory_reservation_access_info_t* out_info) {
  if (reservation == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (access_ordinal >= reservation->access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_memory_reservation_access_info_t info =
      reservation->accesses[access_ordinal].info;
  info.type = out_info->type;
  info.structure_size = out_info->structure_size;
  info.next = out_info->next;
  *out_info = info;
  return AMDF_STATUS_OK;
}

static int amdf_memory_mapping_compare(const void* a, const void* b) {
  const uint64_t lhs =
      ((const amdf_memory_mapping_update_t*)a)->target_byte_offset;
  const uint64_t rhs =
      ((const amdf_memory_mapping_update_t*)b)->target_byte_offset;
  return (lhs > rhs) - (lhs < rhs);
}

// This is the public range/ownership boundary. Native providers consume these
// established facts directly and add only their concrete native source checks.
static amdf_status_t amdf_memory_reservation_validate_source(
    const amdf_memory_reservation_access_state_t* target,
    const amdf_memory_t* memory, uint32_t access_ordinal,
    uint64_t source_byte_offset, uint64_t byte_length,
    amdf_memory_access_t access) {
  const uint64_t granularity = target->info.mapping_granularity;
  if (memory == NULL) {
    return access_ordinal == 0 && access == 0 && source_byte_offset == 0
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (source_byte_offset > memory->info.byte_length ||
      byte_length > memory->info.byte_length - source_byte_offset) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if ((memory->info.source_byte_offset + source_byte_offset) % granularity !=
      0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (target->device == NULL) {
    if (access_ordinal != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    const amdf_memory_map_flags_t required =
        ((access & AMDF_MEMORY_ACCESS_READ) != 0 ? AMDF_MEMORY_MAP_FLAG_READ
                                                 : 0) |
        ((access & AMDF_MEMORY_ACCESS_WRITE) != 0 ? AMDF_MEMORY_MAP_FLAG_WRITE
                                                  : 0);
    return (memory->info.flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE) != 0 &&
                   (required & ~memory->host_mapping.supported_access) == 0
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (access_ordinal >= memory->info.access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_memory_access_state_t* source = &memory->accesses[access_ordinal];
  if (source->device != target->device ||
      source->info.address_domain_ordinal !=
          target->info.address_domain_ordinal ||
      (source->info.flags & AMDF_MEMORY_FLAG_MAPPING_SOURCE) == 0 ||
      (access & ~source->info.access) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_reservation_validate_update(
    const amdf_memory_reservation_t* reservation,
    const amdf_memory_reservation_access_state_t* target,
    const amdf_memory_mapping_update_t* update) {
  const uint64_t granularity = target->info.mapping_granularity;
  if (update->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (update->byte_length == 0 ||
      update->target_byte_offset > target->info.byte_length ||
      update->byte_length >
          target->info.byte_length - update->target_byte_offset ||
      update->target_byte_offset % granularity != 0 ||
      (update->byte_length % granularity != 0 &&
       update->byte_length !=
           target->info.byte_length - update->target_byte_offset)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (update->memory != NULL &&
      (update->access == 0 || (update->access & ~target->info.access) != 0 ||
       (update->access & target->required_access) != target->required_access)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_status_t status = amdf_memory_reservation_validate_source(
      target, update->memory, update->memory_access_ordinal,
      update->source_byte_offset, update->byte_length, update->access);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  // Prior backing may have been installed through another logical consumer
  // sharing these page tables. Its own admitted projection names the native
  // source; the current publisher need not own an access to that allocation.
  const amdf_memory_reservation_access_state_t* previous_target = target;
  if (target->device && update->previous_memory &&
      update->previous_memory_access_ordinal <
          update->previous_memory->info.access_count) {
    const amdf_memory_access_state_t* source =
        &update->previous_memory
             ->accesses[update->previous_memory_access_ordinal];
    if (source->device != target->device) {
      previous_target = NULL;
      for (uint32_t i = 0; i < reservation->access_count; ++i) {
        const amdf_memory_reservation_access_state_t* candidate =
            &reservation->accesses[i];
        if (candidate->device == source->device &&
            candidate->info.address_domain_ordinal ==
                source->info.address_domain_ordinal &&
            candidate->info.mapping_access_ordinal ==
                target->info.mapping_access_ordinal) {
          previous_target = candidate;
          break;
        }
      }
      if (!previous_target) {
        return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      }
    }
  }
  return amdf_memory_reservation_validate_source(
      previous_target, update->previous_memory,
      update->previous_memory_access_ordinal,
      update->previous_source_byte_offset, update->byte_length, 0);
}

static amdf_status_t amdf_memory_reservation_apply(
    amdf_kernel_queue_t* queue, amdf_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint32_t update_count,
    const amdf_memory_mapping_update_t* updates, uint64_t* out_submission) {
  if (reservation == NULL || (update_count != 0 && updates == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (access_ordinal >= reservation->access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_memory_reservation_access_state_t* target =
      &reservation->accesses[access_ordinal];
  if (queue != NULL && queue->device != target->device) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t* failure =
      &reservation->accesses[target->info.mapping_access_ordinal].failure;
  if (!amdf_status_is_ok(*failure)) {
    return *failure;
  }
  if (update_count == 0) {
    return AMDF_STATUS_OK;
  }
  for (uint32_t i = 0; i < update_count; ++i) {
    const amdf_status_t status = amdf_memory_reservation_validate_update(
        reservation, target, &updates[i]);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  amdf_memory_mapping_update_t* ordered = NULL;
  amdf_status_t status = amdf_calloc_array(
      reservation->host_allocator, update_count, sizeof(*ordered),
      amdf_alignof(amdf_memory_mapping_update_t), (void**)&ordered);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memcpy(ordered, updates, update_count * sizeof(*ordered));
  qsort(ordered, update_count, sizeof(*ordered), amdf_memory_mapping_compare);
  // Native providers need source runs, not the caller's page bookkeeping.
  // Compact the already-owned sorted vector without crossing holes, source
  // aliases or permission boundaries. Every original range was validated above.
  uint32_t ordered_count = 1;
  for (uint32_t i = 1; i < update_count && amdf_status_is_ok(status); ++i) {
    amdf_memory_mapping_update_t* previous = &ordered[ordered_count - 1];
    const amdf_memory_mapping_update_t* current = &ordered[i];
    const uint64_t previous_end =
        previous->target_byte_offset + previous->byte_length;
    if (current->target_byte_offset < previous_end) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    } else if (current->target_byte_offset == previous_end &&
               current->memory == previous->memory &&
               current->memory_access_ordinal ==
                   previous->memory_access_ordinal &&
               current->access == previous->access &&
               current->previous_memory == previous->previous_memory &&
               current->previous_memory_access_ordinal ==
                   previous->previous_memory_access_ordinal &&
               (current->previous_memory == NULL ||
                current->previous_source_byte_offset ==
                    previous->previous_source_byte_offset +
                        previous->byte_length) &&
               (current->memory == NULL ||
                current->source_byte_offset ==
                    previous->source_byte_offset + previous->byte_length)) {
      previous->byte_length += current->byte_length;
    } else {
      ordered[ordered_count++] = *current;
    }
  }
  bool mutation_started = false;
  uint64_t accepted_submission = 0;
  if (amdf_status_is_ok(status)) {
    status =
        queue != NULL
            ? queue->vtable->update_memory(
                  queue, reservation, access_ordinal, ordered_count, ordered,
                  &mutation_started, &accepted_submission)
            : target->vtable->update(reservation, access_ordinal, ordered_count,
                                     ordered, &mutation_started);
  }
  if (!amdf_status_is_ok(status) && mutation_started) {
    *failure = status;
  }
  amdf_free(reservation->host_allocator, ordered);
  if (accepted_submission != 0) {
    *out_submission = accepted_submission;
    return AMDF_STATUS_OK;
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_reservation_update(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal,
    uint32_t update_count, const amdf_memory_mapping_update_t* updates) {
  return amdf_memory_reservation_apply(NULL, reservation, access_ordinal,
                                       update_count, updates, NULL);
}

amdf_status_t AMDF_CALL amdf_kernel_queue_update_memory(
    amdf_kernel_queue_t* queue, amdf_memory_reservation_t* reservation,
    uint32_t access_ordinal, uint32_t update_count,
    const amdf_memory_mapping_update_t* updates, uint64_t* out_submission) {
  if (queue == NULL || out_submission == NULL || update_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((queue->info.capabilities &
       AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return amdf_memory_reservation_apply(queue, reservation, access_ordinal,
                                       update_count, updates, out_submission);
}

amdf_status_t AMDF_CALL amdf_memory_reservation_detach(
    amdf_memory_reservation_t* reservation, uint32_t access_ordinal) {
  if (reservation == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (access_ordinal >= reservation->access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  // Full removal must reach the native provider even after a partial update
  // failed. Neither successful removal nor a later error resets the first one.
  const amdf_status_t status =
      reservation->accesses[access_ordinal].vtable->detach(reservation,
                                                           access_ordinal);
  const uint32_t mapping_access_ordinal =
      reservation->accesses[access_ordinal].info.mapping_access_ordinal;
  amdf_status_t* failure =
      &reservation->accesses[mapping_access_ordinal].failure;
  if (!amdf_status_is_ok(status) && amdf_status_is_ok(*failure)) {
    *failure = status;
  }
  return status;
}

amdf_status_t AMDF_CALL
amdf_memory_reservation_destroy(amdf_memory_reservation_t* reservation) {
  if (reservation == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; i < reservation->access_count; ++i) {
    amdf_memory_reservation_access_state_t* access = &reservation->accesses[i];
    if (access->native != NULL) {
      const amdf_status_t release_status =
          access->vtable->destroy(reservation, i);
      if (!amdf_status_is_ok(release_status)) {
        status = release_status;
      }
    }
  }
  if (amdf_status_is_ok(status) && reservation->common_address != NULL) {
    status = amdf_platform_host_reservation_destroy(
        reservation->common_address, reservation->common_byte_length);
  }
  amdf_free(reservation->host_allocator, reservation);
  return status;
}
