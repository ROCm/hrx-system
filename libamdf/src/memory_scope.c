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
#include "libamdf/src/instance.h"
#include "libamdf/src/structure.h"

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
  if (capacity != 0) scopes[0] = &instance->system_memory_scope;
  *out_count = 1;
  return capacity == 0 ? amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL)
                       : AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_endpoint_enumerate_memory_scopes(
    amdf_endpoint_t* endpoint, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count) {
  if (endpoint == NULL || out_count == NULL ||
      (capacity != 0 && scopes == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  bool available = false;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t ordinal = 0; amdf_status_is_ok(status) && !available;
       ++ordinal) {
    amdf_memory_native_profile_t profile;
    status = amdf_endpoint_query_memory_profile(endpoint, ordinal, &profile);
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
      scopes[0] = amdf_endpoint_local_memory_scope(endpoint);
    }
    *out_count = available ? 1 : 0;
    if (available && capacity == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    }
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_device_enumerate_memory_scopes(
    amdf_device_t* device, uint32_t capacity, amdf_memory_scope_t** scopes,
    uint32_t* out_count) {
  if (device == NULL || out_count == NULL ||
      (capacity != 0 && scopes == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *out_count = 0;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_memory_scope_query_info(
    amdf_memory_scope_t* scope, amdf_memory_scope_info_t* out_info) {
  if (scope == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO, sizeof(*out_info));
  if (!amdf_status_is_ok(status)) return status;
  amdf_memory_scope_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = out_info->structure_size,
      .next = out_info->next,
      .kind = scope->kind,
      .memory_profile_count =
          scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM ? 3 : 1,
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
  return query->kind == AMDF_MEMORY_ACCESS_QUERY_EXPECTED
             ? &query->accesses.endpoints[index].requirements
             : &query->accesses.devices[index].requirements;
}

static amdf_endpoint_t* amdf_memory_access_query_endpoint(
    const amdf_memory_access_query_t* query, uint32_t index) {
  return query->kind == AMDF_MEMORY_ACCESS_QUERY_EXPECTED
             ? query->accesses.endpoints[index].endpoint
             : query->accesses.devices[index].device->endpoint;
}

static amdf_status_t amdf_memory_access_query_native_profile(
    const amdf_memory_access_query_t* query, uint32_t index,
    uint32_t profile_ordinal, amdf_memory_native_profile_t* out_profile) {
  if (query->kind == AMDF_MEMORY_ACCESS_QUERY_EXPECTED) {
    return amdf_endpoint_query_memory_profile(
        query->accesses.endpoints[index].endpoint, profile_ordinal,
        out_profile);
  }
  amdf_device_t* device = query->accesses.devices[index].device;
  return device->vtable->query_memory_profile(device, profile_ordinal,
                                              out_profile);
}

static amdf_status_t amdf_memory_access_query_validate(
    amdf_memory_scope_t* scope, const amdf_memory_access_query_t* query) {
  if (query->count != 0 && (query->kind == AMDF_MEMORY_ACCESS_QUERY_EXPECTED
                                ? query->accesses.endpoints == NULL
                                : query->accesses.devices == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_instance_t* instance = amdf_memory_scope_instance(scope);
  for (uint32_t i = 0; i < query->count; ++i) {
    if (query->kind == AMDF_MEMORY_ACCESS_QUERY_EXPECTED) {
      amdf_endpoint_t* endpoint = query->accesses.endpoints[i].endpoint;
      if (endpoint == NULL ||
          amdf_endpoint_get_instance(endpoint) != instance) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    } else {
      amdf_device_t* device = query->accesses.devices[i].device;
      if (device == NULL || device->provider_instance != instance) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      for (uint32_t j = 0; j < i; ++j) {
        if (device == query->accesses.devices[j].device) {
          return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
        }
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
    if (!amdf_status_is_ok(status)) continue;
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
    if (found) *out_profile = profile;
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
  if (quotient > UINT64_MAX / b) return false;
  *out_value = quotient * b;
  return true;
}

static bool amdf_memory_merge_construction(
    const amdf_memory_construction_capabilities_t* consumer,
    amdf_memory_construction_capabilities_t* backing) {
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
                                       &backing->byte_length_granularity) &&
         backing->byte_length_granularity <= backing->maximum_byte_length;
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
    profile.roles |= AMDF_MEMORY_PROFILE_ROLE_CREATE;
    profile.allocation = (amdf_memory_construction_capabilities_t){
        .maximum_byte_length = maximum_byte_length,
        .byte_length_granularity = 1,
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
    const amdf_memory_access_query_t* query, amdf_memory_class_t memory_class,
    amdf_memory_profile_roles_t role,
    const amdf_external_memory_support_t* transport,
    amdf_memory_scope_plan_t* plan, bool* out_found) {
  bool found = true;
  uint32_t gpu_owner = UINT32_MAX;
  amdf_status_t status = AMDF_STATUS_OK;
  for (uint32_t i = 0; amdf_status_is_ok(status) && found && i < query->count;
       ++i) {
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
          plan->backing_access_ordinals[0] = gpu_owner;
          plan->native_owner_ordinals[i] = gpu_owner;
          plan->backing_access_ordinals[plan->backing_access_count++] = i;
        }
      }
    }
  }
  if (amdf_status_is_ok(status)) *out_found = found;
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
    if (!amdf_status_is_ok(status) || !source_found) continue;
    plan->native_profiles[source] = source_profile;
    plan->backing_access_ordinal = source;
    plan->backing_access_count = 1;
    plan->backing_access_ordinals[0] = source;
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
        plan->backing_access_ordinals[plan->backing_access_count++] = i;
      }
    }
    if (amdf_status_is_ok(status) &&
        plan->backing_access_count == query->count) {
      found = true;
      continue;
    }
    for (uint32_t t = 0; amdf_status_is_ok(status) && !found &&
                         t < source_profile.external_memory_support_count;
         ++t) {
      const amdf_external_memory_support_t* transport =
          &source_profile.external_memory_support[t];
      if ((transport->flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT) == 0)
        continue;
      bool consumers_found = true;
      for (uint32_t i = 0;
           amdf_status_is_ok(status) && consumers_found && i < query->count;
           ++i) {
        if (plan->native_owner_ordinals[i] == source) continue;
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
  if (amdf_status_is_ok(status)) *out_found = found;
  return status;
}

// Preserves the backing producer's physical geometry and host view. Other
// consumers constrain accepted requests and their own address alignment, not
// the number of physical allocations or the host mapping's cacheability.
static bool amdf_memory_scope_merge_profile(amdf_memory_profile_roles_t role,
                                            amdf_memory_scope_plan_t* plan) {
  amdf_memory_profile_t* profile = &plan->profile;
  amdf_memory_construction_capabilities_t* construction =
      role == AMDF_MEMORY_PROFILE_ROLE_CREATE     ? &profile->allocation
      : role == AMDF_MEMORY_PROFILE_ROLE_REGISTER ? &profile->registration
                                                  : &profile->import;
  if (plan->shared_external_type != 0) {
    const amdf_external_memory_support_t* transport =
        amdf_memory_native_profile_find_transport(
            &plan->native_profiles[plan->backing_access_ordinal],
            plan->shared_external_type,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT);
    if (!amdf_memory_merge_granularity(construction->byte_length_granularity,
                                       transport->byte_length_alignment,
                                       &construction->byte_length_granularity))
      return false;
    if (transport->maximum_byte_length != 0) {
      construction->maximum_byte_length = amdf_memory_minimum(
          construction->maximum_byte_length, transport->maximum_byte_length);
    }
  }
  for (uint32_t i = 0; i < plan->access_count; ++i) {
    if (i == plan->backing_access_ordinal) continue;
    const amdf_memory_native_profile_t* consumer = &plan->native_profiles[i];
    const bool coordinated =
        role == AMDF_MEMORY_PROFILE_ROLE_CREATE &&
        plan->native_owner_ordinals[i] == plan->backing_access_ordinal;
    const amdf_memory_construction_capabilities_t* consumer_construction =
        coordinated                                 ? &consumer->allocation
        : role == AMDF_MEMORY_PROFILE_ROLE_REGISTER ? &consumer->registration
                                                    : &consumer->import;
    if (!amdf_memory_merge_construction(consumer_construction, construction))
      return false;
    if (role == AMDF_MEMORY_PROFILE_ROLE_CREATE && !coordinated) {
      const amdf_external_memory_support_t* support =
          amdf_memory_native_profile_find_transport(
              consumer, plan->shared_external_type,
              AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT);
      if (!amdf_memory_merge_granularity(
              construction->byte_length_granularity,
              support->byte_length_alignment,
              &construction->byte_length_granularity))
        return false;
      if (support->maximum_byte_length != 0) {
        construction->maximum_byte_length = amdf_memory_minimum(
            construction->maximum_byte_length, support->maximum_byte_length);
      }
    }
  }
  if (plan->backing_access_count > 1) {
    uint64_t minimum_address = 0;
    uint64_t maximum_address = UINT64_MAX;
    for (uint32_t i = 0; i < plan->backing_access_count; ++i) {
      const amdf_memory_address_capabilities_t* address =
          &plan->native_profiles[plan->backing_access_ordinals[i]]
               .device_address;
      if (address->minimum_address > minimum_address) {
        minimum_address = address->minimum_address;
      }
      if (address->maximum_address < maximum_address) {
        maximum_address = address->maximum_address;
      }
    }
    if (minimum_address > maximum_address) return false;
    const uint64_t span_minus_one = maximum_address - minimum_address;
    if (span_minus_one < construction->maximum_byte_length) {
      construction->maximum_byte_length = span_minus_one + 1;
    }
  }
  return construction->byte_length_granularity <=
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
    if (supported) profile->external_memory_support[count++] = support;
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
  if (profile_ordinal >=
      (scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM ? 3u : 1u)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  amdf_status_t status = amdf_memory_access_query_validate(scope, query);
  if (!amdf_status_is_ok(status)) return status;
  if (query->count == 0 &&
      (scope->kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM || profile_ordinal == 2)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_instance_t* instance = amdf_memory_scope_instance(scope);
  amdf_memory_scope_plan_t plan = {
      .scope = scope,
      .host_allocator = instance->host_allocator,
      .access_count = query->count,
  };
  const size_t count = query->count == 0 ? 1 : query->count;
  const size_t plan_stride =
      sizeof(*plan.native_profiles) + 2 * sizeof(*plan.native_owner_ordinals);
  if (count > SIZE_MAX / plan_stride) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_calloc(plan.host_allocator, count * plan_stride,
                       amdf_alignof(amdf_memory_native_profile_t),
                       (void**)&plan.native_profiles);
  if (!amdf_status_is_ok(status)) return status;
  plan.native_owner_ordinals = (uint32_t*)(plan.native_profiles + count);
  plan.backing_access_ordinals = plan.native_owner_ordinals + count;
  plan.backing_access_count = 1;
  plan.backing_access_ordinals[0] = 0;
  for (uint32_t i = 0; i < query->count; ++i) {
    plan.native_owner_ordinals[i] = i;
  }
  const amdf_memory_class_t memory_class =
      scope->kind == AMDF_MEMORY_SCOPE_KIND_SYSTEM ? AMDF_MEMORY_CLASS_SYSTEM
                                                   : AMDF_MEMORY_CLASS_LOCAL;
  const amdf_memory_profile_roles_t role =
      profile_ordinal == 0   ? AMDF_MEMORY_PROFILE_ROLE_CREATE
      : profile_ordinal == 1 ? AMDF_MEMORY_PROFILE_ROLE_REGISTER
                             : AMDF_MEMORY_PROFILE_ROLE_IMPORT;
  bool found = true;
  if (scope->kind == AMDF_MEMORY_SCOPE_KIND_PRIVATE) {
    amdf_device_t* owner_device = scope->owner.private_storage.device;
    found =
        query->count == 1 &&
        (query->kind == AMDF_MEMORY_ACCESS_QUERY_LIVE
             ? query->accesses.devices[0].device == owner_device
             : query->accesses.endpoints[0].endpoint == owner_device->endpoint);
    if (found) {
      scope->owner.private_storage.vtable->query_profile(
          scope, &plan.native_profiles[0]);
      found = amdf_memory_native_profile_supports_access(
          &plan.native_profiles[0],
          amdf_memory_access_query_requirements(query, 0));
    }
  } else if (query->count == 0) {
    plan.native_profiles[0] =
        amdf_memory_host_profile(instance, profile_ordinal);
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
        query, memory_class, role, external_memory == NULL ? NULL : &transport,
        &plan, &found);
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
  if (!amdf_status_is_ok(status)) return status;
  for (uint32_t i = 0; amdf_status_is_ok(status) && i < query->count; ++i) {
    status = amdf_structure_validate_output(
        &out_access_capabilities[i],
        AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        sizeof(out_access_capabilities[i]));
  }
  if (!amdf_status_is_ok(status)) return status;
  amdf_memory_scope_plan_t plan;
  status = amdf_memory_scope_plan_initialize(scope, profile_ordinal, query,
                                             NULL, &plan);
  if (!amdf_status_is_ok(status)) return status;
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

amdf_status_t AMDF_CALL amdf_memory_scope_query_profile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal, uint32_t access_count,
    const amdf_memory_endpoint_access_t* accesses,
    amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities) {
  const amdf_memory_access_query_t query = {
      .kind = AMDF_MEMORY_ACCESS_QUERY_EXPECTED,
      .count = access_count,
      .accesses.endpoints = accesses,
  };
  return amdf_memory_scope_query_access_profile(
      scope, profile_ordinal, &query, out_profile, out_access_capabilities);
}

amdf_status_t AMDF_CALL amdf_memory_scope_query_device_profile(
    amdf_memory_scope_t* scope, uint32_t profile_ordinal, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    amdf_memory_profile_t* out_profile,
    amdf_memory_access_capabilities_t* out_access_capabilities) {
  const amdf_memory_access_query_t query = {
      .kind = AMDF_MEMORY_ACCESS_QUERY_LIVE,
      .count = access_count,
      .accesses.devices = accesses,
  };
  return amdf_memory_scope_query_access_profile(
      scope, profile_ordinal, &query, out_profile, out_access_capabilities);
}

void amdf_memory_scope_plan_deinitialize(amdf_memory_scope_plan_t* plan) {
  amdf_free(plan->host_allocator, plan->native_profiles);
}
