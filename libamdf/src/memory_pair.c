// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/memory_pair.h"

amdf_memory_atomic_capabilities_t amdf_memory_atomic_capabilities_filter(
    amdf_memory_atomic_capabilities_t operations, amdf_memory_access_t access) {
  const bool read = (access & AMDF_MEMORY_ACCESS_READ) != 0;
  const bool write = (access & AMDF_MEMORY_ACCESS_WRITE) != 0;
  const amdf_atomic_operations_t mask =
      read && write ? UINT64_MAX
      : read        ? AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_WAIT
      : write       ? AMDF_ATOMIC_OPERATION_STORE
                    : 0;
  return (amdf_memory_atomic_capabilities_t){
      .device_scope_32 = operations.device_scope_32 & mask,
      .device_scope_64 = operations.device_scope_64 & mask,
      .fabric_scope_32 = operations.fabric_scope_32 & mask,
      .fabric_scope_64 = operations.fabric_scope_64 & mask,
      .system_scope_32 = operations.system_scope_32 & mask,
      .system_scope_64 = operations.system_scope_64 & mask,
  };
}

// Host visibility depends on the exact peer, not the resource's other accesses.
// Native API operations remain required even when CPU lines are coherent: they
// can also publish an allocation to the native device driver.
static amdf_cache_transition_t amdf_memory_host_transition(
    const amdf_cache_transition_t* available,
    amdf_host_memory_properties_t properties, bool coherent) {
  if (available->kind != AMDF_CACHE_TRANSITION_KIND_UNKNOWN && coherent &&
      (properties & AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK) != 0 &&
      available->executor != AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API) {
    return (amdf_cache_transition_t){.kind = AMDF_CACHE_TRANSITION_KIND_NONE};
  }
  return *available;
}

amdf_memory_site_description_t amdf_memory_describe_host_site(
    const amdf_memory_host_description_t* host, amdf_memory_map_flags_t access,
    bool coherent, const amdf_memory_site_description_t* device_peer) {
  amdf_memory_site_description_t description = {0};
  if ((access & AMDF_MEMORY_MAP_FLAG_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((access & AMDF_MEMORY_MAP_FLAG_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  description.release =
      amdf_memory_host_transition(&host->flush, host->properties, coherent);
  description.acquire = amdf_memory_host_transition(&host->invalidate,
                                                    host->properties, coherent);
  if (description.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN;
  }
  if (description.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN;
  }
  if ((host->properties & AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK) != 0 &&
      device_peer != NULL &&
      amdf_memory_compatibility_domain_is_valid(&device_peer->atomic_domain)) {
    if (device_peer->atomic_reach.scope_32 == AMDF_ATOMIC_SCOPE_SYSTEM) {
      description.atomic_reach.scope_32 = AMDF_ATOMIC_SCOPE_SYSTEM;
    }
    if (device_peer->atomic_reach.scope_64 == AMDF_ATOMIC_SCOPE_SYSTEM) {
      description.atomic_reach.scope_64 = AMDF_ATOMIC_SCOPE_SYSTEM;
    }
    if (description.atomic_reach.scope_32 != AMDF_ATOMIC_SCOPE_NONE ||
        description.atomic_reach.scope_64 != AMDF_ATOMIC_SCOPE_NONE) {
      description.atomic_domain = device_peer->atomic_domain;
    }
    // CPU loads/stores qualify the publication/observation edge. CPU RMW is
    // a separate relationship and is not inferred from GPU RMW support.
    const amdf_atomic_operations_t host_operations =
        AMDF_ATOMIC_OPERATION_LOAD | AMDF_ATOMIC_OPERATION_STORE |
        AMDF_ATOMIC_OPERATION_WAIT;
    description.atomic_operations = amdf_memory_atomic_capabilities_filter(
        (amdf_memory_atomic_capabilities_t){
            .system_scope_32 =
                description.atomic_reach.scope_32 == AMDF_ATOMIC_SCOPE_SYSTEM
                    ? host_operations
                    : 0,
            .system_scope_64 =
                description.atomic_reach.scope_64 == AMDF_ATOMIC_SCOPE_SYSTEM
                    ? host_operations
                    : 0,
        },
        access);
  }
  return description;
}

amdf_memory_site_description_t amdf_memory_describe_file_site(
    amdf_memory_map_flags_t access) {
  amdf_memory_site_description_t description = {
      .capabilities = AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN |
                      AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN,
      .release = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
      .acquire = {.kind = AMDF_CACHE_TRANSITION_KIND_NONE},
  };
  if ((access & AMDF_MEMORY_MAP_FLAG_READ) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_READ;
  }
  if ((access & AMDF_MEMORY_MAP_FLAG_WRITE) != 0) {
    description.capabilities |= AMDF_MEMORY_SITE_CAPABILITY_WRITE;
  }
  return description;
}

amdf_status_t amdf_memory_pair_compose(
    const amdf_memory_site_description_t* producer,
    const amdf_memory_site_description_t* consumer,
    amdf_memory_pair_info_t* out_info) {
  if ((producer->capabilities & AMDF_MEMORY_SITE_CAPABILITY_WRITE) == 0 ||
      (consumer->capabilities & AMDF_MEMORY_SITE_CAPABILITY_READ) == 0 ||
      producer->release.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN ||
      consumer->acquire.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  amdf_memory_pair_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = out_info->structure_size,
      .next = out_info->next,
      .flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
      .release = producer->release,
      .acquire = consumer->acquire,
  };
  if ((producer->capabilities & AMDF_MEMORY_SITE_CAPABILITY_MAPPING_SOURCE) !=
          0 &&
      (consumer->capabilities & AMDF_MEMORY_SITE_CAPABILITY_MAPPING_TARGET) !=
          0 &&
      amdf_memory_compatibility_domain_is_valid(&producer->mapping_domain) &&
      amdf_memory_compatibility_domain_is_equal(&producer->mapping_domain,
                                                &consumer->mapping_domain)) {
    info.flags |= AMDF_MEMORY_PAIR_FLAG_MAPPING_SOURCE;
  }
  if (amdf_memory_compatibility_domain_is_valid(&producer->atomic_domain) &&
      amdf_memory_compatibility_domain_is_equal(&producer->atomic_domain,
                                                &consumer->atomic_domain)) {
    info.atomic_reach.scope_32 =
        producer->atomic_reach.scope_32 < consumer->atomic_reach.scope_32
            ? producer->atomic_reach.scope_32
            : consumer->atomic_reach.scope_32;
    info.atomic_reach.scope_64 =
        producer->atomic_reach.scope_64 < consumer->atomic_reach.scope_64
            ? producer->atomic_reach.scope_64
            : consumer->atomic_reach.scope_64;
  }
  const bool shared_device_domain =
      amdf_memory_compatibility_domain_is_valid(
          &producer->device_atomic_domain) &&
      amdf_memory_compatibility_domain_is_equal(
          &producer->device_atomic_domain, &consumer->device_atomic_domain);
  if (shared_device_domain) {
    if (info.atomic_reach.scope_32 == AMDF_ATOMIC_SCOPE_NONE &&
        producer->atomic_operations.device_scope_32 != 0 &&
        consumer->atomic_operations.device_scope_32 != 0) {
      info.atomic_reach.scope_32 = AMDF_ATOMIC_SCOPE_DEVICE;
    }
    if (info.atomic_reach.scope_64 == AMDF_ATOMIC_SCOPE_NONE &&
        producer->atomic_operations.device_scope_64 != 0 &&
        consumer->atomic_operations.device_scope_64 != 0) {
      info.atomic_reach.scope_64 = AMDF_ATOMIC_SCOPE_DEVICE;
    }
  }
  if ((producer->capabilities &
       AMDF_MEMORY_SITE_CAPABILITY_RELEASE_COST_KNOWN) != 0 &&
      (consumer->capabilities &
       AMDF_MEMORY_SITE_CAPABILITY_ACQUIRE_COST_KNOWN) != 0) {
    if (producer->release_fixed_cost_nanoseconds >
        UINT64_MAX - consumer->acquire_fixed_cost_nanoseconds) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    info.flags |= AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN;
    info.estimated_fixed_cost_nanoseconds =
        producer->release_fixed_cost_nanoseconds +
        consumer->acquire_fixed_cost_nanoseconds;
  }
  if (out_info->next != NULL) {
    amdf_memory_pair_atomics_t* atomics = out_info->next;
    const amdf_memory_atomic_capabilities_t* producer_operations =
        &producer->atomic_operations;
    const amdf_memory_atomic_capabilities_t* consumer_operations =
        &consumer->atomic_operations;
    // Each cell is admitted as a complete Cartesian relation. A missing side
    // clears both outputs, so a local capability never appears as a pair fact.
#define AMDF_COPY_ATOMIC_CELL(field, compatible)                            \
  atomics->producer.field = (compatible) && consumer_operations->field != 0 \
                                ? producer_operations->field                \
                                : 0;                                        \
  atomics->consumer.field = (compatible) && producer_operations->field != 0 \
                                ? consumer_operations->field                \
                                : 0
    AMDF_COPY_ATOMIC_CELL(device_scope_32, shared_device_domain);
    AMDF_COPY_ATOMIC_CELL(device_scope_64, shared_device_domain);
    AMDF_COPY_ATOMIC_CELL(fabric_scope_32, info.atomic_reach.scope_32 >=
                                               AMDF_ATOMIC_SCOPE_FABRIC);
    AMDF_COPY_ATOMIC_CELL(fabric_scope_64, info.atomic_reach.scope_64 >=
                                               AMDF_ATOMIC_SCOPE_FABRIC);
    AMDF_COPY_ATOMIC_CELL(system_scope_32, info.atomic_reach.scope_32 ==
                                               AMDF_ATOMIC_SCOPE_SYSTEM);
    AMDF_COPY_ATOMIC_CELL(system_scope_64, info.atomic_reach.scope_64 ==
                                               AMDF_ATOMIC_SCOPE_SYSTEM);
#undef AMDF_COPY_ATOMIC_CELL
  }
  *out_info = info;
  return AMDF_STATUS_OK;
}
