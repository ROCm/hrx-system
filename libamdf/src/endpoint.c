// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/endpoint.h"

#include <stddef.h>
#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/child_tracker.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/memory_scope.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/structure.h"

struct amdf_endpoint_t {
  // Instance borrowed for the lifetime of this endpoint.
  amdf_instance_t* instance;
  // Platform query handle owned by this endpoint.
  amdf_platform_endpoint_t* platform;
  // Immutable properties cached while opening the platform endpoint.
  amdf_endpoint_info_t info;
  // Implementation selected before engine profile qualification.
  struct {
    // Immutable engine operations borrowed for this endpoint's lifetime.
    const void* operations;
    // Implementation-owned endpoint context borrowed for the same lifetime.
    void* context;
  } engine_binding;
  // Complete immutable metadata allocation owned after provider qualification.
  void* engine_profile;
  // Cached result of the engine-profile qualification attempt.
  amdf_status_t engine_profile_status;
  // Whether an engine extension resolved `engine_profile_status`.
  bool engine_profile_resolved;
  // Borrowed physical-local descriptor, published only for supported storage.
  amdf_memory_scope_t local_memory_scope;
  // Whether explicit device creation has published the immutable local scope.
  // Accessed only under the instance's native-state lock.
  bool local_memory_scope_published;
  // Immutable endpoint-local native queue families.
  struct {
    // Records owned inline for the lifetime of this endpoint.
    amdf_queue_family_info_t values[AMDF_ENDPOINT_QUEUE_FAMILY_CAPACITY];
    // Number of records in `values`.
    uint32_t count;
  } queue_families;
  // Number of materialized devices borrowing this endpoint.
  amdf_child_tracker_t children;
};

static amdf_status_t amdf_endpoint_allocate(amdf_instance_t* instance,
                                            amdf_endpoint_t** out_endpoint) {
  const amdf_allocator_t host_allocator =
      amdf_instance_host_allocator(instance);
  amdf_endpoint_t* endpoint = NULL;
  amdf_status_t status =
      amdf_calloc(host_allocator, sizeof(*endpoint),
                  amdf_alignof(amdf_endpoint_t), (void**)&endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  amdf_child_tracker_initialize(&endpoint->children);
  endpoint->local_memory_scope.kind = AMDF_MEMORY_SCOPE_KIND_LOCAL;
  endpoint->local_memory_scope.owner.endpoint = endpoint;
  status = amdf_instance_register_endpoint(instance);
  if (amdf_status_is_ok(status)) {
    endpoint->instance = instance;
    endpoint->info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint->info.structure_size = sizeof(endpoint->info);
    *out_endpoint = endpoint;
  } else {
    amdf_free(host_allocator, endpoint);
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_endpoint_open(amdf_instance_t* instance,
                                           const amdf_endpoint_id_t* id,
                                           amdf_endpoint_t** out_endpoint) {
  if (instance == NULL || id == NULL || out_endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_endpoint_t* endpoint = NULL;
  amdf_status_t status = amdf_endpoint_allocate(instance, &endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_platform_endpoint_open(amdf_instance_platform(instance), id,
                                       &endpoint->platform, &endpoint->info);
  if (amdf_status_is_ok(status)) {
    endpoint->info.queue_family_count = 0;
    *out_endpoint = endpoint;
  } else {
    amdf_instance_unregister_endpoint(instance);
    amdf_free(amdf_instance_host_allocator(instance), endpoint);
  }
  return status;
}

amdf_status_t amdf_endpoint_create(amdf_instance_t* instance,
                                   const amdf_endpoint_info_t* info,
                                   const void* binding, void* context,
                                   amdf_endpoint_t** out_endpoint) {
  amdf_endpoint_t* endpoint = NULL;
  const amdf_status_t status = amdf_endpoint_allocate(instance, &endpoint);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  endpoint->info = *info;
  endpoint->info.queue_family_count = 0;
  amdf_endpoint_set_engine_binding(endpoint, binding, context);
  *out_endpoint = endpoint;
  return AMDF_STATUS_OK;
}

void amdf_endpoint_set_queue_families(
    amdf_endpoint_t* endpoint, uint32_t queue_family_count,
    const amdf_queue_family_info_t* queue_families) {
  amdf_assert(queue_family_count <= AMDF_ENDPOINT_QUEUE_FAMILY_CAPACITY);
  for (uint32_t i = 0; i < queue_family_count; ++i) {
    endpoint->queue_families.values[i] = queue_families[i];
  }
  endpoint->queue_families.count = queue_family_count;
  endpoint->info.queue_family_count = queue_family_count;
}

const amdf_endpoint_info_t* amdf_endpoint_get_cached_info(
    const amdf_endpoint_t* endpoint) {
  return &endpoint->info;
}

amdf_memory_scope_t* amdf_endpoint_local_memory_scope(
    amdf_endpoint_t* endpoint) {
  return &endpoint->local_memory_scope;
}

void amdf_endpoint_publish_local_memory_scope(
    amdf_endpoint_t* endpoint, const amdf_memory_scope_t* scope) {
  if (!endpoint->local_memory_scope_published) {
    endpoint->local_memory_scope = *scope;
    endpoint->local_memory_scope.kind = AMDF_MEMORY_SCOPE_KIND_LOCAL;
    endpoint->local_memory_scope.owner.endpoint = endpoint;
    endpoint->local_memory_scope_published = true;
  }
}

amdf_platform_endpoint_t* amdf_endpoint_get_platform(
    amdf_endpoint_t* endpoint) {
  return endpoint->platform;
}

void amdf_endpoint_set_engine_binding(amdf_endpoint_t* endpoint,
                                      const void* binding, void* context) {
  endpoint->engine_binding.operations = binding;
  endpoint->engine_binding.context = context;
}

const void* amdf_endpoint_get_engine_binding(const amdf_endpoint_t* endpoint) {
  return endpoint->engine_binding.operations;
}

void* amdf_endpoint_get_engine_context(const amdf_endpoint_t* endpoint) {
  return endpoint->engine_binding.context;
}

amdf_allocator_t amdf_endpoint_host_allocator(const amdf_endpoint_t* endpoint) {
  return amdf_instance_host_allocator(endpoint->instance);
}

amdf_instance_t* amdf_endpoint_get_instance(const amdf_endpoint_t* endpoint) {
  return endpoint->instance;
}

void amdf_endpoint_store_engine_profile(amdf_endpoint_t* endpoint,
                                        void* profile) {
  amdf_assert(!endpoint->engine_profile_resolved);
  endpoint->engine_profile = profile;
  endpoint->engine_profile_status = AMDF_STATUS_OK;
  endpoint->engine_profile_resolved = true;
}

void amdf_endpoint_store_engine_profile_error(amdf_endpoint_t* endpoint,
                                              amdf_status_t status) {
  amdf_assert(!endpoint->engine_profile_resolved);
  amdf_assert(!amdf_status_is_ok(status));
  endpoint->engine_profile_status = status;
  endpoint->engine_profile_resolved = true;
}

amdf_status_t amdf_endpoint_query_engine_profile(
    const amdf_endpoint_t* endpoint, amdf_engine_kind_t expected_engine_kind,
    const void** out_profile) {
  if (endpoint->info.engine_kind != expected_engine_kind ||
      !endpoint->engine_profile_resolved) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (!amdf_status_is_ok(endpoint->engine_profile_status)) {
    return endpoint->engine_profile_status;
  }
  *out_profile = endpoint->engine_profile;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_endpoint_register_device(amdf_endpoint_t* endpoint) {
  return amdf_child_tracker_register(&endpoint->children);
}

void amdf_endpoint_unregister_device(amdf_endpoint_t* endpoint) {
  amdf_child_tracker_unregister(&endpoint->children);
}

amdf_status_t AMDF_CALL amdf_endpoint_query_info(
    amdf_endpoint_t* endpoint, amdf_endpoint_info_t* out_info) {
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_header(
      out_info, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      (uint32_t)sizeof(amdf_endpoint_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  if (out_info->next != NULL) {
    amdf_structure_type_t type;
    memcpy(&type, out_info->next, sizeof(type));
    switch (type) {
      case AMDF_STRUCTURE_TYPE_ENDPOINT_HOST_LOCALITY: {
        amdf_endpoint_host_locality_t* locality = out_info->next;
        status =
            amdf_structure_validate_output(locality, type, sizeof(*locality));
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        locality->numa_node =
            endpoint->platform != NULL
                ? amdf_platform_endpoint_host_numa_node(endpoint->platform)
                : UINT32_MAX;
        break;
      }
      case AMDF_STRUCTURE_TYPE_ENDPOINT_PCI_ADDRESS: {
        amdf_endpoint_pci_address_t* address = out_info->next;
        status =
            amdf_structure_validate_output(address, type, sizeof(*address));
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        const amdf_endpoint_pci_address_t unavailable = {0};
        const amdf_endpoint_pci_address_t* cached =
            endpoint->platform != NULL
                ? amdf_platform_endpoint_pci_address(endpoint->platform)
                : &unavailable;
        address->available = cached->available;
        address->domain = cached->domain;
        address->bus = cached->bus;
        address->device = cached->device;
        address->function = cached->function;
        break;
      }
      default:
        return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  }

  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = endpoint->info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_endpoint_query_queue_family_info(
    amdf_endpoint_t* endpoint, uint32_t queue_family_ordinal,
    amdf_queue_family_info_t* out_info) {
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
      (uint32_t)sizeof(amdf_queue_family_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (queue_family_ordinal >= endpoint->queue_families.count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_queue_family_info_t* source_info =
      &endpoint->queue_families.values[queue_family_ordinal];
  const uint32_t structure_size = out_info->structure_size;
  void* const next = out_info->next;
  *out_info = *source_info;
  out_info->structure_size = structure_size;
  out_info->next = next;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_endpoint_close(amdf_endpoint_t* endpoint) {
  if (endpoint == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (amdf_child_tracker_count(&endpoint->children) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status =
      endpoint->platform != NULL
          ? amdf_platform_endpoint_close(endpoint->platform)
          : AMDF_STATUS_OK;
  if (amdf_status_is_ok(status)) {
    const amdf_allocator_t host_allocator =
        amdf_endpoint_host_allocator(endpoint);
    amdf_instance_unregister_endpoint(endpoint->instance);
    amdf_free(host_allocator, endpoint->engine_profile);
    amdf_free(host_allocator, endpoint);
  }
  return status;
}
