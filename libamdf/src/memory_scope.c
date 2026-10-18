// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_scope.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/host_memory.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/platform/host_memory.h"
#include "libamdf/src/structure.h"

static uint32_t amdf_memory_scope_profile_count(
    const amdf_memory_scope_t* scope) {
  return scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM  ? 5
         : scope->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL ? 2
                                                       : 1;
}

static amdf_memory_profile_roles_t amdf_memory_scope_profile_role(
    const amdf_memory_scope_t* scope, uint32_t ordinal) {
  if (ordinal == 0 ||
      (scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM && ordinal >= 3)) {
    return AMDF_MEMORY_PROFILE_ROLE_CREATE;
  }
  return scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM && ordinal == 1
             ? AMDF_MEMORY_PROFILE_ROLE_REGISTER
             : AMDF_MEMORY_PROFILE_ROLE_IMPORT;
}

amdf_instance_t* amdf_memory_scope_instance(const amdf_memory_scope_t* scope) {
  if (scope->kind == AMDF_MEMORY_SCOPE_KIND_PRIVATE) {
    return scope->owner.private_storage.device->provider_instance;
  }
  return scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM
             ? scope->owner.instance
             : amdf_endpoint_get_instance(scope->owner.endpoint);
}

amdf_status_t AMDF_CALL amdf_instance_enumerate_memory_scopes(
    amdf_instance_t* instance, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count) {
  if (instance == NULL || out_count == NULL ||
      (capacity != 0 && scopes == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (capacity != 0) {
    scopes[0] = &instance->system_memory_scope;
  }
  *out_count = 1;
  return capacity == 0 ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
                       : AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_device_enumerate_memory_scopes(
    amdf_device_t* device, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count) {
  if (device == NULL || out_count == NULL ||
      (capacity != 0 && scopes == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  bool available = false;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t ordinal = 0; amdf_status_is_ok(status) && !available;
       ++ordinal) {
    amdf_memory_native_profile_t profile;
    status = device->vtable->query_memory_profile(device, ordinal, &profile);
    if (amdf_status_is_ok(status)) {
      available = profile.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
                  (profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0;
    }
  }
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE) ||
      status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    status = AMDF_STATUS_OK;
  }
  if (amdf_status_is_ok(status)) {
    if (available && capacity != 0) {
      scopes[0] = amdf_endpoint_local_memory_scope(device->endpoint);
    }
    *out_count = available ? 1 : 0;
    if (available && capacity == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    }
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_memory_scope_query_info(
    amdf_memory_scope_t* scope, amdf_memory_scope_info_t* out_info) {
  if (scope == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO, sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = out_info->structure_size,
      .next = out_info->next,
      .kind = scope->kind,
      .memory_profile_count = amdf_memory_scope_profile_count(scope),
  };
  if (scope->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL) {
    info.physical_endpoint_id =
        amdf_endpoint_get_cached_info(scope->owner.endpoint)->id;
  }
  *out_info = info;
  return AMDF_STATUS_OK;
}

static const amdf_memory_access_requirements_t*
amdf_memory_access_query_requirements(const amdf_memory_access_query_t* query,
                                      uint32_t index) {
  return &query->accesses[index].requirements;
}

static amdf_endpoint_t* amdf_memory_access_query_endpoint(
    const amdf_memory_access_query_t* query, uint32_t index) {
  return query->accesses[index].device->endpoint;
}

static amdf_status_t amdf_memory_access_query_native_profile(
    const amdf_memory_access_query_t* query, uint32_t index,
    uint32_t profile_ordinal, amdf_memory_native_profile_t* out_profile) {
  amdf_device_t* device = query->accesses[index].device;
  return device->vtable->query_memory_profile(device, profile_ordinal,
                                              out_profile);
}

static amdf_status_t amdf_memory_access_query_validate(
    amdf_memory_scope_t* scope, const amdf_memory_access_query_t* query) {
  if (query->count != 0 && query->accesses == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_instance_t* instance = amdf_memory_scope_instance(scope);
  for (uint32_t i = 0; i < query->count; ++i) {
    amdf_device_t* device = query->accesses[i].device;
    if (device == NULL || device->provider_instance != instance) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    for (uint32_t j = 0; j < i; ++j) {
      if (device == query->accesses[j].device) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
    const amdf_memory_access_requirements_t* requirements =
        amdf_memory_access_query_requirements(query, i);
    if (requirements->reserved != 0 ||
        (requirements->access &
         ~(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
           AMDF_MEMORY_ACCESS_EXECUTE)) != 0 ||
        (requirements->flags & ~AMDF_MEMORY_ACCESS_FLAGS) != 0 ||
        (requirements->address_kinds &
         ~((UINT64_C(1) << (AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE + 1)) - 1)) !=
            0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  return AMDF_STATUS_OK;
}

static bool amdf_memory_native_profile_supports_access(
    const amdf_memory_native_profile_t* profile,
    const amdf_memory_access_requirements_t* requirements) {
  return (requirements->access & profile->guaranteed_device_access) ==
             profile->guaranteed_device_access &&
         (requirements->access & ~profile->supported_device_access) == 0 &&
         (requirements->flags & ~profile->supported_flags) == 0 &&
         (requirements->address_kinds & ~profile->address_kinds) == 0;
}

static const amdf_external_memory_support_t*
amdf_memory_native_profile_find_transport(
    const amdf_memory_native_profile_t* profile,
    amdf_external_memory_type_t type,
    amdf_external_memory_support_flags_t required_role) {
  for (uint32_t i = 0; i < profile->external_memory_support_count; ++i) {
    const amdf_external_memory_support_t* support =
        &profile->external_memory_support[i];
    if (support->type == type && (support->flags & required_role) != 0) {
      return support;
    }
  }
  return NULL;
}

// Exhausted native ordinals and absent contracts are a metadata miss. Native
// qualification failures retain their exact status and stop selection.
static amdf_status_t amdf_memory_access_find_profile(
    const amdf_memory_access_query_t* query, uint32_t access_ordinal,
    amdf_memory_class_t memory_class, amdf_memory_profile_roles_t role,
    const amdf_external_memory_support_t* transport,
    const amdf_memory_native_profile_t* backing,
    amdf_memory_native_profile_t* out_profile, bool* out_found) {
  bool found = false;
  amdf_memory_native_profile_t profile;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t ordinal = 0; amdf_status_is_ok(status) && !found; ++ordinal) {
    status = amdf_memory_access_query_native_profile(query, access_ordinal,
                                                     ordinal, &profile);
    if (!amdf_status_is_ok(status)) {
      continue;
    }
    if (backing != NULL &&
        (profile.construction.query_access !=
             backing->construction.query_access ||
         !backing->construction.query_access(backing, &profile, &profile))) {
      continue;
    }
    if (profile.memory_class != memory_class ||
        (profile.roles & role) != role ||
        !amdf_memory_native_profile_supports_access(
            &profile,
            amdf_memory_access_query_requirements(query, access_ordinal))) {
      continue;
    }
    if (transport != NULL) {
      const amdf_external_memory_support_t* imported =
          amdf_memory_native_profile_find_transport(
              &profile, transport->type,
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT);
      if (imported == NULL ||
          !amdf_external_memory_provenance_is_equal(&transport->provenance,
                                                    &imported->provenance)) {
        continue;
      }
    }
    found = true;
  }
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE) ||
      status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    status = AMDF_STATUS_OK;
  }
  if (amdf_status_is_ok(status)) {
    if (found) {
      *out_profile = profile;
    }
    *out_found = found;
  }
  return status;
}

static uint64_t amdf_memory_minimum(uint64_t a, uint64_t b) {
  return a < b ? a : b;
}

static uint64_t amdf_memory_maximum(uint64_t a, uint64_t b) {
  return a > b ? a : b;
}

// Native lengths can have different granularities. Their least common multiple
// is the exact composite requirement; overflow means no representable request.
static bool amdf_memory_merge_granularity(uint64_t a, uint64_t b,
                                          uint64_t* out_value) {
  uint64_t x = a;
  uint64_t y = b;
  while (y != 0) {
    const uint64_t remainder = x % y;
    x = y;
    y = remainder;
  }
  const uint64_t quotient = a / x;
  if (quotient > UINT64_MAX / b) {
    return false;
  }
  *out_value = quotient * b;
  return true;
}

static bool amdf_memory_merge_construction(
    const amdf_memory_construction_capabilities_t* consumer,
    amdf_memory_construction_capabilities_t* backing) {
  if (backing->registered_host_cacheability !=
      consumer->registered_host_cacheability) {
    return false;
  }
  backing->maximum_byte_length = amdf_memory_minimum(
      backing->maximum_byte_length, consumer->maximum_byte_length);
  backing->registered_host_pointer_alignment =
      amdf_memory_maximum(backing->registered_host_pointer_alignment,
                          consumer->registered_host_pointer_alignment);
  backing->minimum_alignment = amdf_memory_minimum(backing->minimum_alignment,
                                                   consumer->minimum_alignment);
  backing->maximum_alignment = amdf_memory_minimum(backing->maximum_alignment,
                                                   consumer->maximum_alignment);
  return amdf_memory_merge_granularity(backing->byte_length_granularity,
                                       consumer->byte_length_granularity,
                                       &backing->byte_length_granularity);
}

static amdf_memory_native_profile_t amdf_memory_host_profile(
    amdf_instance_t* instance, uint32_t ordinal) {
  const uint64_t granularity =
      amdf_platform_instance_host_allocation_granularity(instance->platform);
  const uint64_t maximum_byte_length = (SIZE_MAX / 2) & ~(granularity - 1);
  amdf_memory_native_profile_t profile = {
      .ordinal = ordinal,
      .memory_class = AMDF_MEMORY_CLASS_SYSTEM,
      .roles = AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      .guaranteed_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .supported_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .device_address = {.address_domain_ordinal =
                             AMDF_ADDRESS_DOMAIN_ORDINAL_NONE},
      .host_mapping =
          {
              .maximum_byte_length = maximum_byte_length,
              .byte_offset_granularity = 1,
              .byte_length_granularity = 1,
              .supported_access =
                  AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
          },
  };
  if (ordinal == 0) {
    profile.roles |= AMDF_MEMORY_PROFILE_ROLE_CREATE |
                     AMDF_MEMORY_PROFILE_ROLE_MAPPING_SOURCE;
    profile.supported_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
    profile.guaranteed_flags |= AMDF_MEMORY_FLAG_HOST_FILE_IO;
    profile.supported_flags |= AMDF_MEMORY_FLAG_HOST_FILE_IO;
    profile.host_mapping.features = AMDF_HOST_MAPPING_FEATURE_FILE_IO;
    profile.allocation = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = 1,
        .placement_kinds = amdf_platform_host_memory_placement_kinds(),
        .minimum_alignment = granularity,
        .maximum_alignment = granularity,
        .native_byte_length_granularity = granularity,
    };
  } else {
    profile.roles |= AMDF_MEMORY_PROFILE_ROLE_REGISTER;
    profile.registration = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = 1,
        .registered_host_pointer_alignment = 1,
        .registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
        .minimum_alignment = 1,
        .maximum_alignment = granularity,
        .native_byte_length_granularity = 1,
    };
  }
  return profile;
}

static void amdf_memory_scope_set_backing_profile(
    uint32_t ordinal, const amdf_memory_native_profile_t* native,
    amdf_memory_profile_t* profile) {
  *profile = (amdf_memory_profile_t){
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(*profile),
      .ordinal = ordinal,
      .memory_class = native->memory_class,
      .roles = native->roles,
      .guaranteed_flags = native->guaranteed_flags & AMDF_MEMORY_BACKING_FLAGS,
      .supported_flags = native->supported_flags & AMDF_MEMORY_BACKING_FLAGS,
      .availability = native->availability,
      .allocation = native->allocation,
      .registration = native->registration,
      .import = native->import,
      .host_mapping = native->host_mapping,
      .external_memory_support_count = native->external_memory_support_count,
  };
  memcpy(profile->external_memory_support, native->external_memory_support,
         sizeof(profile->external_memory_support));
}

// Registers the same caller pages, or imports the same external backing, on
// every consumer. Ordinary GPU registrations share one native handle and VA;
// other consumers obtain independent references to the same caller pages.
// This is metadata selection only; no resource is acquired.
static amdf_status_t amdf_memory_scope_select_acquisition(
    amdf_memory_scope_t* scope, const amdf_memory_access_query_t* query,
    amdf_memory_class_t memory_class, amdf_memory_profile_roles_t role,
    const amdf_external_memory_support_t* transport,
    amdf_memory_scope_plan_t* plan, bool* out_found) {
  bool found = true;
  uint32_t gpu_owner = UINT32_MAX;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; amdf_status_is_ok(status) && found && i < query->count;
       ++i) {
    // Local import admits consumers on the scope's physical endpoint. Peer
    // allocation/import protocols require a separate native reach contract.
    if (scope->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL &&
        !amdf_endpoint_id_is_equal(
            &amdf_endpoint_get_cached_info(query->accesses[i].device->endpoint)
                 ->id,
            &amdf_endpoint_get_cached_info(scope->owner.endpoint)->id)) {
      found = false;
      continue;
    }
    status = amdf_memory_access_find_profile(query, i, memory_class, role,
                                             transport, NULL,
                                             &plan->native_profiles[i], &found);
    if (amdf_status_is_ok(status) && found &&
        role == AMDF_MEMORY_PROFILE_ROLE_REGISTER &&
        (plan->native_profiles[i].address_kinds &
         (UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU)) != 0) {
      if (gpu_owner == UINT32_MAX) {
        gpu_owner = i;
      } else {
        const amdf_memory_native_profile_t* owner =
            &plan->native_profiles[gpu_owner];
        // Independent GPU registrations reserve different ordinary addresses.
        // Only a shared native construction contract can promise one pointer.
        found =
            owner->construction.query_access != NULL &&
            owner->construction.query_access ==
                plan->native_profiles[i].construction.query_access &&
            amdf_memory_access_query_requirements(query, gpu_owner)->access ==
                amdf_memory_access_query_requirements(query, i)->access &&
            owner->construction.query_access(owner, &plan->native_profiles[i],
                                             &plan->native_profiles[i]) &&
            amdf_memory_native_profile_supports_access(
                &plan->native_profiles[i],
                amdf_memory_access_query_requirements(query, i));
        if (found) {
          plan->backing_access_ordinal = gpu_owner;
          plan->coordinated.ordinals[0] = gpu_owner;
          plan->native_owner_ordinals[i] = gpu_owner;
          plan->coordinated.ordinals[plan->coordinated.count++] = i;
        }
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_found = found;
  }
  return status;
}

// A compatible native group obtains one backing and its participating mappings
// together. Other consumers import references through a qualified transport.
// Both choices are metadata contracts, not native implementation identities.
static amdf_status_t amdf_memory_scope_select_allocation(
    amdf_memory_scope_t* scope, const amdf_memory_access_query_t* query,
    amdf_memory_class_t memory_class, amdf_memory_scope_plan_t* plan,
    bool* out_found) {
  bool found = false;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t source = 0;
       amdf_status_is_ok(status) && !found && source < query->count; ++source) {
    if (scope->kind == AMDF_MEMORY_SCOPE_KIND_LOCAL &&
        !amdf_endpoint_id_is_equal(
            &amdf_endpoint_get_cached_info(
                 amdf_memory_access_query_endpoint(query, source))
                 ->id,
            &amdf_endpoint_get_cached_info(scope->owner.endpoint)->id)) {
      continue;
    }
    bool source_found = false;
    amdf_memory_native_profile_t source_profile;
    status = amdf_memory_access_find_profile(
        query, source, memory_class, AMDF_MEMORY_PROFILE_ROLE_CREATE, NULL,
        NULL, &source_profile, &source_found);
    if (!amdf_status_is_ok(status) || !source_found) {
      continue;
    }
    plan->native_profiles[source] = source_profile;
    plan->backing_access_ordinal = source;
    plan->coordinated.count = 1;
    plan->coordinated.ordinals[0] = source;
    for (uint32_t i = 0; i < query->count; ++i) {
      plan->native_owner_ordinals[i] = i;
    }
    for (uint32_t i = 0;
         amdf_status_is_ok(status) &&
         source_profile.construction.query_access != NULL && i < query->count;
         ++i) {
      if (i == source ||
          amdf_memory_access_query_requirements(query, i)->access !=
              amdf_memory_access_query_requirements(query, source)->access) {
        continue;
      }
      bool member_found = false;
      status = amdf_memory_access_find_profile(
          query, i, memory_class, AMDF_MEMORY_PROFILE_ROLE_CREATE, NULL,
          &source_profile, &plan->native_profiles[i], &member_found);
      if (amdf_status_is_ok(status) && member_found) {
        plan->native_owner_ordinals[i] = source;
        plan->coordinated.ordinals[plan->coordinated.count++] = i;
      }
    }
    if (amdf_status_is_ok(status) && plan->coordinated.count == query->count) {
      found = true;
      continue;
    }
    for (uint32_t t = 0; amdf_status_is_ok(status) && !found &&
                         t < source_profile.external_memory_support_count;
         ++t) {
      const amdf_external_memory_support_t* transport =
          &source_profile.external_memory_support[t];
      if ((transport->flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT) == 0) {
        continue;
      }
      bool consumers_found = true;
      for (uint32_t i = 0;
           amdf_status_is_ok(status) && consumers_found && i < query->count;
           ++i) {
        if (plan->native_owner_ordinals[i] == source) {
          continue;
        }
        status = amdf_memory_access_find_profile(
            query, i, memory_class, AMDF_MEMORY_PROFILE_ROLE_IMPORT, transport,
            NULL, &plan->native_profiles[i], &consumers_found);
      }
      if (amdf_status_is_ok(status) && consumers_found) {
        plan->native_profiles[source] = source_profile;
        plan->backing_access_ordinal = source;
        plan->shared_external_type = transport->type;
        found = true;
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_found = found;
  }
  return status;
}

// Preserves the backing producer's physical geometry and host view. Other
// consumers constrain accepted requests and their own address alignment, not
// the number of physical allocations or the host mapping's cacheability.
static bool amdf_memory_scope_merge_profile(amdf_memory_profile_roles_t role,
                                            amdf_memory_scope_plan_t* plan) {
  amdf_memory_profile_t* profile = &plan->profile;
  const bool host_backing = plan->backing_access_ordinal == plan->access_count;
  const bool host_registration =
      host_backing && plan->shared_external_type == 0;
  amdf_memory_construction_capabilities_t* construction =
      role == AMDF_MEMORY_PROFILE_ROLE_CREATE     ? &profile->allocation
      : role == AMDF_MEMORY_PROFILE_ROLE_REGISTER ? &profile->registration
                                                  : &profile->import;
  if (plan->shared_external_type != 0 && !host_backing) {
    const amdf_external_memory_support_t* transport =
        amdf_memory_native_profile_find_transport(
            &plan->native_profiles[plan->backing_access_ordinal],
            plan->shared_external_type,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT);
    if (!amdf_memory_merge_granularity(
            construction->byte_length_granularity,
            transport->byte_length_alignment,
            &construction->byte_length_granularity)) {
      return false;
    }
    if (transport->maximum_byte_length != 0) {
      construction->maximum_byte_length = amdf_memory_minimum(
          construction->maximum_byte_length, transport->maximum_byte_length);
    }
  }
  for (uint32_t i = 0; i < plan->access_count; ++i) {
    if (amdf_status_is_ok(profile->availability)) {
      profile->availability = plan->native_profiles[i].availability;
    }
    if (i == plan->backing_access_ordinal) {
      continue;
    }
    const amdf_memory_native_profile_t* consumer = &plan->native_profiles[i];
    const bool coordinated =
        role == AMDF_MEMORY_PROFILE_ROLE_CREATE && !host_backing &&
        plan->native_owner_ordinals[i] == plan->backing_access_ordinal;
    amdf_memory_construction_capabilities_t consumer_construction =
        coordinated ? consumer->allocation
        : host_registration || role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
            ? consumer->registration
            : consumer->import;
    if (host_registration) {
      // The platform creates ordinary write-back pages; device registration
      // constrains their geometry without taking over physical allocation.
      if (consumer_construction.registered_host_cacheability !=
              AMDF_HOST_CACHEABILITY_WRITE_BACK ||
          consumer_construction.registered_host_pointer_alignment >
              plan->native_profiles[plan->backing_access_ordinal]
                  .allocation.minimum_alignment ||
          plan->native_profiles[plan->backing_access_ordinal]
                      .allocation.minimum_alignment %
                  consumer_construction.native_byte_length_granularity !=
              0 ||
          construction->native_byte_length_granularity %
                  consumer_construction.native_byte_length_granularity !=
              0) {
        return false;
      }
      consumer_construction.registered_host_cacheability =
          AMDF_HOST_CACHEABILITY_UNKNOWN;
      consumer_construction.registered_host_pointer_alignment = 0;
    }
    if (!amdf_memory_merge_construction(&consumer_construction, construction)) {
      return false;
    }
    if (role == AMDF_MEMORY_PROFILE_ROLE_CREATE && !coordinated &&
        !host_registration) {
      const amdf_external_memory_support_t* support =
          amdf_memory_native_profile_find_transport(
              consumer, plan->shared_external_type,
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT);
      if (!amdf_memory_merge_granularity(
              construction->byte_length_granularity,
              support->byte_length_alignment,
              &construction->byte_length_granularity)) {
        return false;
      }
      if (support->maximum_byte_length != 0) {
        construction->maximum_byte_length = amdf_memory_minimum(
            construction->maximum_byte_length, support->maximum_byte_length);
      }
    }
  }
  if (plan->coordinated.count > 1) {
    uint64_t minimum_address = 0;
    uint64_t maximum_address = UINT64_MAX;
    for (uint32_t i = 0; i < plan->coordinated.count; ++i) {
      const amdf_memory_address_capabilities_t* address =
          &plan->native_profiles[plan->coordinated.ordinals[i]].device_address;
      if (address->minimum_address > minimum_address) {
        minimum_address = address->minimum_address;
      }
      if (address->maximum_address < maximum_address) {
        maximum_address = address->maximum_address;
      }
    }
    if (minimum_address > maximum_address) {
      return false;
    }
    const uint64_t span_minus_one = maximum_address - minimum_address;
    if (span_minus_one < construction->maximum_byte_length) {
      construction->maximum_byte_length = span_minus_one + 1;
    }
  }
  // An absent optional provider may have no discoverable allocation limit.
  // Preserve its actionable availability instead of hiding the profile.
  return !amdf_status_is_ok(profile->availability) ||
         construction->byte_length_granularity <=
             construction->maximum_byte_length;
}

// External import is a joint contract too. Only transports accepted by every
// selected consumer are published, with the intersection of their limits.
static bool amdf_memory_scope_merge_import_support(
    amdf_memory_scope_plan_t* plan) {
  amdf_memory_profile_t* profile = &plan->profile;
  uint32_t count = 0;
  for (uint32_t t = 0; t < profile->external_memory_support_count; ++t) {
    amdf_external_memory_support_t support =
        profile->external_memory_support[t];
    bool supported =
        (support.flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT) != 0;
    for (uint32_t i = 1; supported && i < plan->access_count; ++i) {
      const amdf_external_memory_support_t* consumer =
          amdf_memory_native_profile_find_transport(
              &plan->native_profiles[i], support.type,
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT);
      if (consumer == NULL || !amdf_external_memory_provenance_is_equal(
                                  &support.provenance, &consumer->provenance)) {
        supported = false;
        continue;
      }
      support.flags &= consumer->flags;
      if ((support.flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET) !=
          0) {
        supported = amdf_memory_merge_granularity(
            support.source_offset_alignment, consumer->source_offset_alignment,
            &support.source_offset_alignment);
      } else {
        support.source_offset_alignment = 0;
      }
      supported = supported &&
                  amdf_memory_merge_granularity(support.byte_length_alignment,
                                                consumer->byte_length_alignment,
                                                &support.byte_length_alignment);
      if (consumer->maximum_byte_length != 0 &&
          (support.maximum_byte_length == 0 ||
           consumer->maximum_byte_length < support.maximum_byte_length)) {
        support.maximum_byte_length = consumer->maximum_byte_length;
      }
    }
    if (supported) {
      profile->external_memory_support[count++] = support;
    }
  }
  memset(&profile->external_memory_support[count], 0,
         (AMDF_MEMORY_PROFILE_EXTERNAL_SUPPORT_CAPACITY - count) *
             sizeof(profile->external_memory_support[0]));
  profile->external_memory_support_count = count;
  return count != 0;
}

amdf_status_t amdf_memory_scope_plan_initialize(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal,
    const amdf_memory_access_query_t* query,
    const amdf_external_memory_t* external_memory,
    amdf_memory_scope_plan_t* out_plan) {
  if (profile_ordinal >= amdf_memory_scope_profile_count(scope)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_status_t status = amdf_memory_access_query_validate(scope, query);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (query->count == 0 && (scope->kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM ||
                            profile_ordinal == 2 || profile_ordinal == 4)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_instance_t* instance = amdf_memory_scope_instance(scope);
  amdf_memory_scope_plan_t plan = {
      .scope = scope,
      .host_allocator = instance->host_allocator,
      .access_count = query->count,
  };
  const bool host_backing =
      query->count == 0 ||
      (scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM && profile_ordinal >= 3);
  const size_t count = (size_t)query->count + (host_backing ? 1 : 0);
  const size_t plan_stride =
      sizeof(*plan.native_profiles) + 2 * sizeof(*plan.native_owner_ordinals);
  if (count > SIZE_MAX / plan_stride) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_calloc(plan.host_allocator, count * plan_stride,
                       amdf_alignof(amdf_memory_native_profile_t),
                       (void**)&plan.native_profiles);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  plan.native_owner_ordinals = (uint32_t*)(plan.native_profiles + count);
  plan.coordinated.ordinals = plan.native_owner_ordinals + count;
  plan.coordinated.count = 1;
  plan.coordinated.ordinals[0] = 0;
  for (uint32_t i = 0; i < query->count; ++i) {
    plan.native_owner_ordinals[i] = i;
  }
  const amdf_memory_class_t memory_class =
      scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM ? AMDF_MEMORY_CLASS_SYSTEM
                                                   : AMDF_MEMORY_CLASS_LOCAL;
  const amdf_memory_profile_roles_t role =
      amdf_memory_scope_profile_role(scope, profile_ordinal);
  plan.acquisition_role = role;
  bool found = true;
  if (scope->kind == AMDF_MEMORY_SCOPE_KIND_PRIVATE) {
    amdf_device_t* owner_device = scope->owner.private_storage.device;
    found = query->count == 1 && query->accesses[0].device == owner_device;
    if (found) {
      scope->owner.private_storage.vtable->query_profile(
          scope, &plan.native_profiles[0]);
      found = amdf_memory_native_profile_supports_access(
          &plan.native_profiles[0],
          amdf_memory_access_query_requirements(query, 0));
    }
  } else if (host_backing) {
    const amdf_platform_host_export_t* service =
        profile_ordinal == 4
            ? amdf_platform_instance_host_export(instance->platform)
            : NULL;
    if (profile_ordinal == 4 && service == NULL) {
      found = false;
    } else if (query->count != 0) {
      const amdf_external_memory_support_t transport = {
          .type = service != NULL ? service->type : 0,
      };
      status = amdf_memory_scope_select_acquisition(
          scope, query, AMDF_MEMORY_CLASS_SYSTEM,
          service != NULL ? AMDF_MEMORY_NATIVE_PROFILE_ROLE_HOST_IMPORT
                          : AMDF_MEMORY_PROFILE_ROLE_REGISTER,
          service != NULL ? &transport : NULL, &plan, &found);
    }
    if (amdf_status_is_ok(status) && found) {
      plan.backing_access_ordinal = query->count;
      plan.native_profiles[plan.backing_access_ordinal] =
          amdf_memory_host_profile(
              instance, role == AMDF_MEMORY_PROFILE_ROLE_CREATE ? 0 : 1);
      if (service != NULL) {
        amdf_memory_native_profile_t* backing =
            &plan.native_profiles[plan.backing_access_ordinal];
        backing->availability = service->availability;
        backing->guaranteed_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
        backing->allocation.byte_length_granularity =
            backing->allocation.native_byte_length_granularity;
        backing->allocation.maximum_byte_length =
            amdf_memory_minimum(backing->allocation.maximum_byte_length,
                                service->maximum_byte_length);
        plan.shared_external_type = service->type;
      }
    }
  } else if (role == AMDF_MEMORY_PROFILE_ROLE_CREATE) {
    status = amdf_memory_scope_select_allocation(scope, query, memory_class,
                                                 &plan, &found);
  } else {
    const amdf_external_memory_support_t transport = {
        .type = external_memory == NULL ? 0 : external_memory->type,
        .provenance = external_memory == NULL
                          ? (amdf_external_memory_provenance_t){{0, 0}}
                          : external_memory->provenance,
    };
    status = amdf_memory_scope_select_acquisition(
        scope, query, memory_class, role,
        external_memory == NULL ? NULL : &transport, &plan, &found);
  }
  if (amdf_status_is_ok(status) && !found) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status)) {
    amdf_memory_scope_set_backing_profile(
        profile_ordinal, &plan.native_profiles[plan.backing_access_ordinal],
        &plan.profile);
    if (!amdf_memory_scope_merge_profile(role, &plan) ||
        (role == AMDF_MEMORY_PROFILE_ROLE_IMPORT &&
         !amdf_memory_scope_merge_import_support(&plan))) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }
  if (amdf_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    amdf_free(plan.host_allocator, plan.native_profiles);
  }
  return status;
}

static amdf_status_t amdf_memory_scope_query_access_profile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal,
    const amdf_memory_access_query_t* query, amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities) {
  if (scope == NULL || (query->count != 0 && out_access_capabilities == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_output(
      out_profile, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE, sizeof(*out_profile));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < query->count; ++i) {
    status = amdf_structure_validate_output(
        &out_access_capabilities[i],
        AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        sizeof(out_access_capabilities[i]));
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(scope, profile_ordinal, query,
                                             NULL, &plan);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  plan.profile.structure_size = out_profile->structure_size;
  plan.profile.next = out_profile->next;
  *out_profile = plan.profile;
  for (uint32_t i = 0; i < query->count; ++i) {
    const amdf_memory_native_profile_t* native = &plan.native_profiles[i];
    out_access_capabilities[i] = (amdf_memory_access_capabilities_t){
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = out_access_capabilities[i].structure_size,
        .next = out_access_capabilities[i].next,
        .guaranteed_access = native->guaranteed_device_access,
        .supported_access = native->supported_device_access,
        .guaranteed_flags = native->guaranteed_flags & AMDF_MEMORY_ACCESS_FLAGS,
        .supported_flags = native->supported_flags & AMDF_MEMORY_ACCESS_FLAGS,
        .atomic_operations_32 = native->atomic_operations_32,
        .atomic_operations_64 = native->atomic_operations_64,
        .device_address = native->device_address,
        .address_kinds = native->address_kinds,
    };
  }
  amdf_memory_scope_plan_deinitialize(&plan);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_memory_scope_query_device_profile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities) {
  const amdf_memory_access_query_t query = {
      .count = access_count,
      .accesses = accesses,
  };
  return amdf_memory_scope_query_access_profile(
      scope, profile_ordinal, &query, out_profile, out_access_capabilities);
}

void amdf_memory_scope_plan_deinitialize(amdf_memory_scope_plan_t* plan) {
  amdf_free(plan->host_allocator, plan->native_profiles);
}

// Native access flags are derived once per cold owner preparation. Coordinated
// consumers share the backing owner's cache mode as well as its allocation.
amdf_memory_flags_t amdf_memory_scope_plan_required_access_flags(
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_device_access_t* accesses, uint32_t access_ordinal) {
  const uint32_t owner = plan->native_owner_ordinals[access_ordinal];
  const uint32_t coordinated_owner = plan->coordinated.ordinals[0];
  const uint32_t count =
      owner == coordinated_owner ? plan->coordinated.count : 1;
  amdf_memory_flags_t flags = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t ordinal = owner == coordinated_owner
                                 ? plan->coordinated.ordinals[i]
                                 : access_ordinal;
    const amdf_memory_access_requirements_t* requirements =
        &accesses[ordinal].requirements;
    flags |= requirements->flags;
    if (requirements->address_kinds != 0) {
      flags |= AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    }
  }
  return flags;
}

amdf_status_t amdf_memory_scope_plan_validate_registration(
    const amdf_memory_scope_plan_t* plan,
    amdf_host_cacheability_t registered_host_cacheability) {
  if (plan->acquisition_role != AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
    return registered_host_cacheability == AMDF_HOST_CACHEABILITY_UNKNOWN
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  switch (registered_host_cacheability) {
    case AMDF_HOST_CACHEABILITY_UNKNOWN:
    case AMDF_HOST_CACHEABILITY_WRITE_BACK:
    case AMDF_HOST_CACHEABILITY_WRITE_COMBINED:
    case AMDF_HOST_CACHEABILITY_UNCACHED:
      break;
    default:
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (registered_host_cacheability == AMDF_HOST_CACHEABILITY_UNKNOWN ||
      registered_host_cacheability !=
          plan->profile.registration.registered_host_cacheability) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_memory_profile_validate_site(
    const amdf_memory_scope_plan_t* plan, amdf_memory_flags_t required_flags,
    const amdf_memory_profile_site_t* site) {
  if (site->reserved != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (site->kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    return site->value.device.access_ordinal < plan->access_count
               ? AMDF_STATUS_OK
               : amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if ((site->kind != AMDF_MEMORY_SITE_KIND_HOST &&
       site->kind != AMDF_MEMORY_SITE_KIND_FILE_IO) ||
      site->value.host_access == 0 ||
      (site->value.host_access &
       ~(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE)) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((required_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) == 0 ||
      (plan->profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) == 0 ||
      (site->value.host_access &
       ~plan->profile.host_mapping.supported_access) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (site->kind == AMDF_MEMORY_SITE_KIND_FILE_IO &&
      (plan->profile.host_mapping.features &
       AMDF_HOST_MAPPING_FEATURE_FILE_IO) == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static amdf_memory_flags_t amdf_memory_profile_access_flags(
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_profile_pair_query_t* query, uint32_t ordinal) {
  return plan->native_profiles[ordinal].guaranteed_flags |
         query->required_flags |
         amdf_memory_scope_plan_required_access_flags(plan, query->accesses,
                                                      ordinal);
}

static amdf_status_t amdf_memory_profile_describe_site(
    const amdf_memory_scope_plan_t* plan,
    const amdf_memory_profile_pair_query_t* query,
    const amdf_memory_profile_site_t* site,
    const amdf_memory_profile_site_t* peer,
    const amdf_memory_site_description_t* peer_description,
    amdf_memory_site_description_t* out_description) {
  if (site->kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    const uint32_t ordinal = site->value.device.access_ordinal;
    const amdf_memory_native_profile_t* native =
        &plan->native_profiles[ordinal];
    if (native->visibility.describe_site == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family),
    };
    const amdf_status_t status = amdf_endpoint_query_queue_family_info(
        query->accesses[ordinal].device->endpoint,
        site->value.device.queue_family_ordinal, &family);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    const amdf_memory_site_query_t local_query = {
        .access = query->accesses[ordinal].requirements.access,
        .flags = amdf_memory_profile_access_flags(plan, query, ordinal),
        .atomic_operations_32 = native->atomic_operations_32,
        .atomic_operations_64 = native->atomic_operations_64,
        .queue_family_info = &family,
    };
    return native->visibility.describe_site(&local_query, out_description);
  }

  if (site->kind == AMDF_MEMORY_SITE_KIND_FILE_IO) {
    *out_description = amdf_memory_describe_file_site(site->value.host_access);
    return AMDF_STATUS_OK;
  }

  amdf_memory_host_description_t host;
  if (plan->access_count == 0 ||
      (plan->backing_access_ordinal == plan->access_count &&
       peer->kind != AMDF_MEMORY_SITE_KIND_DEVICE)) {
    // Host-only access has no device visibility edge and no native exporter.
    // REGISTER admission has already established ordinary write-back pages.
    host = (amdf_memory_host_description_t){
        .cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK,
        .flush = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
        .invalidate = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
    };
  } else if (plan->backing_access_ordinal == plan->access_count) {
    // Device registration does not own the CPU view or its cache policy.
    const amdf_status_t status = amdf_host_memory_query_description(&host);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  } else {
    const uint32_t ordinal = plan->backing_access_ordinal;
    const amdf_memory_native_profile_t* native =
        &plan->native_profiles[ordinal];
    if (native->visibility.describe_host == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    host = native->visibility.describe_host(
        native->visibility.data, query->external_memory_type,
        amdf_memory_profile_access_flags(plan, query, ordinal));
  }
  const bool coherent = peer->kind != AMDF_MEMORY_SITE_KIND_DEVICE ||
                        (amdf_memory_profile_access_flags(
                             plan, query, peer->value.device.access_ordinal) &
                         AMDF_MEMORY_FLAG_HOST_COHERENT) != 0;
  *out_description = amdf_memory_describe_host_site(
      &host, site->value.host_access, coherent,
      peer->kind == AMDF_MEMORY_SITE_KIND_DEVICE ? peer_description : NULL);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_memory_scope_query_pair_info(
    amdf_memory_scope_t* scope, const amdf_memory_profile_pair_query_t* query,
    amdf_memory_pair_info_t* out_info) {
  if (scope == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_input(
      query, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY, sizeof(*query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO, sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const bool imported =
      amdf_memory_scope_profile_role(scope, query->memory_profile_ordinal) ==
      AMDF_MEMORY_PROFILE_ROLE_IMPORT;
  const bool opaque =
      query->external_memory_type == AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD ||
      query->external_memory_type == AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS;
  const bool has_provenance = amdf_external_memory_provenance_is_valid(
      &query->external_memory_provenance);
  if (query->reserved != 0 || query->external_memory_reserved != 0 ||
      (query->required_flags & ~AMDF_MEMORY_BACKING_FLAGS) != 0 ||
      (imported &&
       (query->external_memory_type == 0 ||
        query->external_memory_type > AMDF_EXTERNAL_MEMORY_TYPE_COUNT ||
        opaque != has_provenance)) ||
      (!imported && (query->external_memory_type != 0 || has_provenance))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_memory_access_query_t accesses = {
      .count = query->access_count,
      .accesses = query->accesses,
  };
  const amdf_external_memory_t transport = {
      .type = query->external_memory_type,
      .provenance = query->external_memory_provenance,
  };
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(
      scope, query->memory_profile_ordinal, &accesses,
      imported ? &transport : NULL, &plan);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = plan.profile.availability;
  if (amdf_status_is_ok(status) &&
      (query->required_flags & ~plan.profile.supported_flags) != 0) {
    status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_scope_plan_validate_registration(
        &plan, query->registered_host_cacheability);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_profile_validate_site(&plan, query->required_flags,
                                               &query->producer);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_profile_validate_site(&plan, query->required_flags,
                                               &query->consumer);
  }
  // Match concrete pair ordering: the device supplies any SYSTEM fact before
  // the host view is described, while composition retains the query direction.
  const bool consumer_first =
      query->producer.kind != AMDF_MEMORY_SITE_KIND_DEVICE;
  const amdf_memory_profile_site_t* first_site =
      consumer_first ? &query->consumer : &query->producer;
  const amdf_memory_profile_site_t* second_site =
      consumer_first ? &query->producer : &query->consumer;
  amdf_memory_site_description_t first;
  amdf_memory_site_description_t second;
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_profile_describe_site(&plan, query, first_site,
                                               second_site, NULL, &first);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_memory_profile_describe_site(&plan, query, second_site,
                                               first_site, &first, &second);
  }
  if (amdf_status_is_ok(status)) {
    status =
        amdf_memory_pair_compose(consumer_first ? &second : &first,
                                 consumer_first ? &first : &second, out_info);
  }
  amdf_memory_scope_plan_deinitialize(&plan);
  return status;
}
