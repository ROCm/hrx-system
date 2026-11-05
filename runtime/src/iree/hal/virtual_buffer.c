// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/virtual_buffer.h"

#include <stdlib.h>

#include "iree/hal/device_group.h"
#include "iree/hal/memory_backend.h"
#include "iree/hal/queue.h"

static iree_status_t iree_hal_virtual_buffer_validate_sources(
    iree_hal_device_group_t* group, iree_hal_virtual_buffer_sources_t sources) {
  if (!sources.pool_count || !sources.pools) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation requires mapping-source pools");
  }
  const void* domain = iree_hal_device_group_memory_domain(group);
  for (iree_host_size_t i = 0; i < sources.pool_count; ++i) {
    const iree_hal_memory_contract_t* contract =
        sources.pools[i] ? sources.pools[i]->memory_contract : NULL;
    if (!contract || contract->domain != domain) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source pool must belong to reservation group");
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_query_capabilities(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_virtual_buffer_sources_t sources, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation group is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_pool_scope_validate(group, scope));
  IREE_RETURN_IF_ERROR(
      iree_hal_virtual_buffer_validate_sources(group, sources));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < iree_hal_device_group_device_count(group) &&
       iree_status_is_ok(status) && !out_capabilities->max_byte_length;
       ++i) {
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (backend && backend->query_virtual_buffer) {
      status = backend->query_virtual_buffer(backend, group, scope, sources,
                                             host_allocator, out_capabilities);
    }
  }
  if (!iree_status_is_ok(status)) {
    memset(out_capabilities, 0, sizeof(*out_capabilities));
  }
  return status;
}

static iree_status_t iree_hal_virtual_buffer_create_native(
    void* user_data, iree_hal_device_group_t* group,
    const iree_hal_pool_scope_t* scope,
    const iree_hal_virtual_buffer_sources_t* sources,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group) &&
                               iree_status_is_ok(status) && !*out_buffer;
       ++i) {
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (backend && backend->create_virtual_buffer) {
      status =
          backend->create_virtual_buffer(backend, group, *scope, *sources,
                                         options, host_allocator, out_buffer);
    }
  }
  if (iree_status_is_ok(status) && !*out_buffer) {
    status =
        iree_make_status(IREE_STATUS_UNAVAILABLE,
                         "no memory backend supports this reservation scope");
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_create(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_hal_virtual_buffer_sources_t sources,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (!group || !options || !options->byte_length ||
      (options->flags & ~IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS) ||
      !iree_device_size_is_valid_alignment(options->min_alignment) ||
      !iree_device_size_is_valid_alignment(options->mapping_granularity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid reservation group, geometry, or flags");
  }
  IREE_RETURN_IF_ERROR(iree_hal_pool_scope_validate(group, scope));
  IREE_RETURN_IF_ERROR(
      iree_hal_virtual_buffer_validate_sources(group, sources));
  const iree_hal_virtual_buffer_create_callback_t leaf = {
      .fn = iree_hal_virtual_buffer_create_native,
  };
  return iree_hal_device_group_create_virtual_buffer(
      group, &scope, &sources, options, leaf, host_allocator, out_buffer);
}

//===----------------------------------------------------------------------===//
// Immutable accepted-mapping contracts
//===----------------------------------------------------------------------===//

static iree_hal_memory_access_t iree_hal_virtual_buffer_site_access(
    const iree_hal_memory_contract_t* contract, uint32_t id) {
  return id == 1 ? contract->host.access
                 : iree_hal_buffer_usage_access(contract->scopes[id].usage);
}

static iree_hal_memory_pair_info_t iree_hal_virtual_buffer_pair_info(
    const iree_hal_memory_contract_t* contract, uint32_t producer,
    uint32_t consumer) {
  return iree_hal_memory_transition_query_info(
      (iree_hal_memory_transition_table_t){contract},
      (iree_hal_memory_transition_pair_t){
          (producer << contract->transition_row_shift) | consumer});
}

// Checks facts needed even before the target has captured its final guarantees.
// Native schema and class eligibility are independent of physical allocation.
static bool iree_hal_virtual_buffer_source_matches_schema(
    const iree_hal_memory_contract_t* target,
    iree_hal_memory_backend_type_t source_type, uint64_t native_type_mask,
    const iree_hal_memory_contract_t* source) {
  if (!source || source->domain != target->domain ||
      !source->mapping_source.requirements.minimum_granularity ||
      source->mapping_source.type != source_type ||
      (source->mapping_source.native_type_mask & ~native_type_mask)) {
    return false;
  }
  return true;
}

static bool iree_hal_virtual_buffer_source_has_site(
    const iree_hal_memory_contract_t* source, uint32_t id) {
  return iree_any_bit_set(source->scopes[id].interfaces,
                          1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE);
}

static bool iree_hal_virtual_buffer_source_covers_site(
    const iree_hal_memory_contract_t* target,
    const iree_hal_memory_contract_t* source, uint32_t id) {
  const iree_hal_memory_contract_t* alias =
      iree_hal_memory_contract_mapping_alias(source);
  return iree_hal_virtual_buffer_source_has_site(source, id) &&
         iree_all_bits_set(alias->scopes[id].usage, target->scopes[id].usage) &&
         (id != 1 ||
          iree_all_bits_set(alias->host.access, target->host.access));
}

typedef struct iree_hal_virtual_buffer_transition_query_t {
  // Borrowed for synchronous construction; never retained by the result.
  iree_hal_virtual_buffer_sources_t sources;
  // False if source recipes cannot be executed by one common alias protocol.
  bool supported;
} iree_hal_virtual_buffer_transition_query_t;

static bool iree_hal_virtual_buffer_join_recipe(
    iree_hal_memory_transition_recipe_info_t* result,
    const iree_hal_memory_transition_recipe_info_t* value) {
  if (result->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN ||
      value->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN) {
    // A source without a qualified transition cannot contribute a promise.
    *result = (iree_hal_memory_transition_recipe_info_t){0};
    return true;
  }
  return iree_hal_memory_transition_recipe_join(result, value);
}

static iree_status_t iree_hal_virtual_buffer_query_transition(
    void* user_data, uint32_t producer, uint32_t consumer,
    iree_hal_memory_pair_info_t* out_info) {
  iree_hal_virtual_buffer_transition_query_t* query = user_data;
  bool found_source = false;
  for (iree_host_size_t i = 0; i < query->sources.pool_count; ++i) {
    const iree_hal_memory_contract_t* source =
        query->sources.pools[i]->memory_contract;
    if (!iree_hal_virtual_buffer_source_has_site(source, producer) ||
        !iree_hal_virtual_buffer_source_has_site(source, consumer)) {
      continue;
    }
    const iree_hal_memory_pair_info_t value = iree_hal_virtual_buffer_pair_info(
        iree_hal_memory_contract_mapping_alias(source), producer, consumer);
    if (!found_source) {
      *out_info = value;
      found_source = true;
      continue;
    }
    out_info->flags &= value.flags;
    out_info->atomic_reach.scope_32 =
        iree_min(out_info->atomic_reach.scope_32, value.atomic_reach.scope_32);
    out_info->atomic_reach.scope_64 =
        iree_min(out_info->atomic_reach.scope_64, value.atomic_reach.scope_64);
    out_info->producer_atomics =
        iree_hal_atomic_operation_capabilities_intersect(
            out_info->producer_atomics, value.producer_atomics);
    out_info->consumer_atomics =
        iree_hal_atomic_operation_capabilities_intersect(
            out_info->consumer_atomics, value.consumer_atomics);
    out_info->estimated_fixed_cost_nanoseconds =
        iree_max(out_info->estimated_fixed_cost_nanoseconds,
                 value.estimated_fixed_cost_nanoseconds);
    query->supported &=
        iree_hal_virtual_buffer_join_recipe(&out_info->release, &value.release);
    query->supported &=
        iree_hal_virtual_buffer_join_recipe(&out_info->acquire, &value.acquire);
  }
  if (!found_source) {
    // No declared class can back both sites. Leave shared reach absent so the
    // exact pair and any participating wildcard are explicitly unsupported.
    *out_info = (iree_hal_memory_pair_info_t){0};
  }
  return iree_ok_status();
}

static bool iree_hal_virtual_buffer_supports_atomics(
    uint32_t available, iree_hal_atomic_operation_capabilities_t required) {
  iree_hal_memory_scope_access_t requested = {0};
  iree_hal_memory_scope_access_set_atomic_operations(&requested, required);
  return iree_all_bits_set(available, requested.atomic_operations);
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_initialize_contract(
    iree_hal_pool_scope_t scope, iree_hal_virtual_buffer_sources_t sources,
    iree_hal_memory_backend_type_t source_type,
    uint64_t source_native_type_mask, iree_hal_memory_contract_t* contract,
    iree_hal_virtual_buffer_capabilities_t* capabilities) {
  bool supported = true;
  for (iree_host_size_t i = 0; i < sources.pool_count && supported; ++i) {
    const iree_hal_memory_contract_t* source =
        sources.pools[i]->memory_contract;
    // Preserve an enforced physical domain only when every admitted source
    // shares it. VA reservation alone supplies no physical placement guarantee.
    if (i == 0) {
      contract->placement = source->placement;
    } else if (contract->placement.mode != source->placement.mode ||
               contract->placement.node != source->placement.node) {
      contract->placement = (iree_hal_pool_placement_t){0};
    }
    supported = iree_hal_virtual_buffer_source_matches_schema(
        contract, source_type, source_native_type_mask, source);
    if (!supported) {
      continue;
    }
    bool participates = false;
    for (uint32_t id = 1; id < contract->scope_count; ++id) {
      if (contract->scopes[id].interfaces &&
          iree_hal_virtual_buffer_source_has_site(source, id)) {
        participates = true;
        supported &=
            iree_hal_virtual_buffer_source_covers_site(contract, source, id);
      }
    }
    supported &= participates;
    const iree_device_size_t granularity =
        source->mapping_source.requirements.minimum_granularity;
    capabilities->mapping_granularity =
        iree_max(capabilities->mapping_granularity, granularity);
    capabilities->reservation_granularity =
        iree_max(capabilities->reservation_granularity, granularity);
  }
  for (uint32_t id = 1; id < contract->scope_count && supported; ++id) {
    if (!contract->scopes[id].interfaces) {
      continue;
    }
    bool covered = false;
    for (iree_host_size_t i = 0; i < sources.pool_count; ++i) {
      const iree_hal_memory_contract_t* source =
          sources.pools[i]->memory_contract;
      if (!iree_hal_virtual_buffer_source_has_site(source, id)) {
        continue;
      }
      covered = true;
      const iree_hal_memory_contract_t* alias =
          iree_hal_memory_contract_mapping_alias(source);
      if (id == 1) {
        contract->host.properties &= alias->host.properties;
      }
      contract->scopes[id].atomic_operations &=
          alias->scopes[id].atomic_operations;
      contract->zero_compute_atomic_operations[id] &=
          alias->zero_compute_atomic_operations[id];
    }
    supported &= covered;
  }
  supported &=
      capabilities->reservation_granularity <= capabilities->max_byte_length &&
      capabilities->mapping_granularity <= capabilities->max_alignment;
  iree_hal_memory_contract_finalize_atomic_operations(contract);
  supported &= iree_hal_virtual_buffer_supports_atomics(
      contract->scopes[1].atomic_operations, scope.host.atomics);
  supported &=
      iree_all_bits_set(contract->host.properties, scope.host.properties);
  for (iree_host_size_t i = 0; i < scope.family_count && supported; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    const uint32_t id = access->family->memory.queue_scope_id;
    supported &=
        iree_hal_virtual_buffer_supports_atomics(
            contract->scopes[id].atomic_operations, access->atomics.queue) &&
        iree_hal_virtual_buffer_supports_atomics(
            contract->scopes[id + 1].atomic_operations,
            access->atomics.program);
    supported &= iree_hal_virtual_buffer_supports_atomics(
        contract->zero_compute_atomic_operations[id],
        access->atomics.queue_zero_compute);
  }
  iree_hal_virtual_buffer_transition_query_t query = {sources, supported};
  iree_status_t status = iree_ok_status();
  if (supported) {
    status = iree_hal_memory_contract_initialize_transitions(
        contract, iree_hal_virtual_buffer_query_transition, &query);
  }
  for (iree_host_size_t i = 0;
       i < scope.family_count && query.supported && iree_status_is_ok(status);
       ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    if (!(access->requirements &
          IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST)) {
      continue;
    }
    const uint32_t id = access->family->memory.queue_scope_id;
    const iree_hal_memory_access_t device_access =
        iree_hal_buffer_usage_access(access->usage);
    if ((scope.host.access & IREE_HAL_MEMORY_ACCESS_WRITE) &&
        (device_access & IREE_HAL_MEMORY_ACCESS_READ)) {
      const iree_hal_memory_pair_info_t pair =
          iree_hal_virtual_buffer_pair_info(contract, 1, id);
      query.supported &=
          pair.release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    }
    if ((scope.host.access & IREE_HAL_MEMORY_ACCESS_READ) &&
        (device_access & IREE_HAL_MEMORY_ACCESS_WRITE)) {
      const iree_hal_memory_pair_info_t pair =
          iree_hal_virtual_buffer_pair_info(contract, id, 1);
      query.supported &=
          pair.acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    }
  }
  if (!query.supported || !iree_status_is_ok(status)) {
    *capabilities = (iree_hal_virtual_buffer_capabilities_t){0};
  }
  return status;
}

static bool iree_hal_virtual_buffer_recipe_covers(
    const iree_hal_memory_transition_recipe_info_t* target,
    const iree_hal_memory_transition_recipe_info_t* source) {
  if (target->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN) {
    return true;
  }
  iree_hal_memory_transition_recipe_info_t combined = *target;
  return iree_hal_memory_transition_recipe_join(&combined, source) &&
         combined.kind == target->kind &&
         combined.executor == target->executor &&
         combined.operation == target->operation &&
         combined.range_granularity == target->range_granularity &&
         combined.host.instruction == target->host.instruction &&
         combined.host.fence_before == target->host.fence_before &&
         combined.host.fence_after == target->host.fence_after;
}

static bool iree_hal_virtual_buffer_source_is_compatible(
    const iree_hal_virtual_buffer_t* target, uint32_t scope_id,
    const iree_hal_memory_contract_t* source) {
  const iree_hal_memory_contract_t* contract = target->buffer.memory.contract;
  if (contract->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED &&
      (source->placement.mode != IREE_HAL_POOL_PLACEMENT_REQUIRED ||
       source->placement.node != contract->placement.node)) {
    return false;
  }
  if (!iree_hal_virtual_buffer_source_matches_schema(
          contract, target->source_type, target->source_native_type_mask,
          source)) {
    return false;
  }
  const iree_hal_memory_contract_t* alias =
      iree_hal_memory_contract_mapping_alias(source);
  for (uint32_t id = 1; id < contract->scope_count; ++id) {
    if (!contract->scopes[id].interfaces) {
      continue;
    }
    if (!iree_hal_virtual_buffer_source_has_site(source, id)) {
      if (!scope_id ||
          target->projections[id] == target->projections[scope_id]) {
        return false;
      }
      continue;
    }
    // Preserve every applicable guarantee, including aliases the caller may
    // install later. Compatibility never depends on currently mapped ranges.
    if (!iree_hal_virtual_buffer_source_covers_site(contract, source, id) ||
        (id == 1 && !iree_all_bits_set(alias->host.properties,
                                       contract->host.properties)) ||
        !iree_all_bits_set(alias->scopes[id].atomic_operations,
                           contract->scopes[id].atomic_operations) ||
        !iree_all_bits_set(alias->zero_compute_atomic_operations[id],
                           contract->zero_compute_atomic_operations[id])) {
      return false;
    }
    if (target->source_is_compatible &&
        !target->source_is_compatible(target, source, id)) {
      return false;
    }
  }
  for (uint32_t p = 1; p < contract->scope_count; ++p) {
    if (!iree_hal_virtual_buffer_source_has_site(source, p) ||
        !(iree_hal_virtual_buffer_site_access(contract, p) &
          IREE_HAL_MEMORY_ACCESS_WRITE)) {
      continue;
    }
    for (uint32_t c = 1; c < contract->scope_count; ++c) {
      if (!iree_hal_virtual_buffer_source_has_site(source, c) ||
          !(iree_hal_virtual_buffer_site_access(contract, c) &
            IREE_HAL_MEMORY_ACCESS_READ)) {
        continue;
      }
      const iree_hal_memory_pair_info_t required =
          iree_hal_virtual_buffer_pair_info(contract, p, c);
      const iree_hal_memory_pair_info_t candidate =
          iree_hal_virtual_buffer_pair_info(alias, p, c);
      const uint32_t guarantees =
          IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
          IREE_HAL_MEMORY_PAIR_MAPPING_SOURCE;
      if (!iree_all_bits_set(candidate.flags, required.flags & guarantees) ||
          candidate.atomic_reach.scope_32 < required.atomic_reach.scope_32 ||
          candidate.atomic_reach.scope_64 < required.atomic_reach.scope_64 ||
          !iree_hal_atomic_operation_capabilities_support(
              candidate.producer_atomics, required.producer_atomics) ||
          !iree_hal_atomic_operation_capabilities_support(
              candidate.consumer_atomics, required.consumer_atomics) ||
          !iree_hal_virtual_buffer_recipe_covers(&required.release,
                                                 &candidate.release) ||
          !iree_hal_virtual_buffer_recipe_covers(&required.acquire,
                                                 &candidate.acquire)) {
        return false;
      }
    }
  }
  return true;
}

IREE_API_EXPORT bool iree_hal_virtual_buffer_query_pool_compatibility(
    const iree_hal_buffer_t* buffer, iree_hal_memory_scope_t scope,
    const iree_hal_pool_t* pool) {
  if (!buffer || !pool) {
    return false;
  }
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  if (!contract || scope.domain != contract->domain ||
      scope.id >= contract->scope_count ||
      (scope.id && !contract->scopes[scope.id].interfaces)) {
    return false;
  }
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(buffer);
  return view.root && iree_hal_virtual_buffer_source_is_compatible(
                          view.root, scope.id, pool->memory_contract);
}

IREE_API_EXPORT iree_hal_virtual_buffer_view_t
iree_hal_buffer_virtual_view(const iree_hal_buffer_t* buffer) {
  const iree_hal_buffer_t* allocated = buffer->allocated_buffer;
  const iree_hal_buffer_vtable_t* vtable =
      (const iree_hal_buffer_vtable_t*)allocated->resource.vtable;
  iree_hal_virtual_buffer_view_t view =
      vtable->query_virtual ? vtable->query_virtual(allocated)
                            : (iree_hal_virtual_buffer_view_t){0};
  if (view.root && allocated != buffer) {
    view.byte_offset += buffer->byte_offset - allocated->byte_offset;
  }
  return view;
}

IREE_API_EXPORT iree_hal_virtual_buffer_view_t
iree_hal_virtual_buffer_query_root(const iree_hal_buffer_t* buffer) {
  return (iree_hal_virtual_buffer_view_t){
      .root = (iree_hal_virtual_buffer_t*)buffer,
  };
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_query_info(
    const iree_hal_buffer_t* buffer, iree_hal_virtual_buffer_info_t* out_info) {
  *out_info = (iree_hal_virtual_buffer_info_t){0};
  if (!buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
  }
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(buffer);
  if (!view.root) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffer has no committed virtual mapping target");
  }
  *out_info = (iree_hal_virtual_buffer_info_t){
      .mapping_byte_offset = view.byte_offset,
      .mapping_granularity = view.root->mapping_granularity,
      .operations = view.root->operations,
      .projection_count = view.root->projection_count,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_virtual_buffer_validate_scope(
    const iree_hal_buffer_t* buffer, iree_hal_memory_scope_t scope) {
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  if (!contract || scope.domain != contract->domain ||
      scope.id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scope does not belong to the buffer's group");
  }
  if (scope.id && !contract->scopes[scope.id].interfaces) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "scope is not admitted by the buffer view");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_resolve_projection(
    const iree_hal_buffer_t* buffer, iree_hal_memory_scope_t scope,
    uint32_t* out_projection) {
  *out_projection = IREE_HAL_MEMORY_SCOPE_INVALID;
  if (!buffer || !scope.id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer and concrete scope are required");
  }
  IREE_RETURN_IF_ERROR(iree_hal_virtual_buffer_validate_scope(buffer, scope));
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(buffer);
  if (!view.root) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffer has no committed virtual mapping target");
  }
  *out_projection = view.root->projections[scope.id];
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_detach(
    iree_hal_buffer_t* buffer, iree_hal_memory_scope_t scope) {
  if (!buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_virtual_buffer_validate_scope(buffer, scope));
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(buffer);
  if (!view.root) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffer has no committed virtual mapping target");
  }
  if (view.byte_offset ||
      buffer->byte_length != view.root->buffer.byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "full detach requires the complete native target");
  }
  iree_hal_buffer_t* allocated = buffer->allocated_buffer;
  const iree_hal_buffer_vtable_t* vtable =
      (const iree_hal_buffer_vtable_t*)allocated->resource.vtable;
  if (!vtable->detach_mappings) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffer does not support host-completed detach");
  }
  const iree_hal_memory_contract_t* contract = buffer->memory.contract;
  const uint32_t begin = scope.id ? scope.id : 1;
  const uint32_t end = scope.id ? begin + 1 : contract->scope_count;
  iree_status_t status = iree_ok_status();
  for (uint32_t id = begin; id < end && iree_status_is_ok(status); ++id) {
    if (!contract->scopes[id].interfaces) {
      continue;
    }
    const uint32_t projection = view.root->projections[id];
    bool visited = false;
    for (uint32_t previous = begin; previous < id; ++previous) {
      visited |= contract->scopes[previous].interfaces &&
                 view.root->projections[previous] == projection;
    }
    if (!visited) {
      status = vtable->detach_mappings(allocated, id);
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_export_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length,
    iree_hal_external_storage_class_t requested_class,
    iree_hal_external_storage_t* out_storage,
    iree_hal_buffer_range_t* out_logical_range) {
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "virtual reservations have no external storage payload");
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_map_range(
    iree_hal_buffer_t* buffer, iree_hal_mapping_mode_t mode,
    iree_hal_memory_access_t access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t offset, iree_device_size_t length,
    iree_hal_buffer_mapping_t* mapping) {
  return iree_hal_buffer_native_host_span(buffer, offset, length,
                                          &mapping->contents);
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_unmap_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_mapping_t* mapping) {
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_invalidate_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  iree_atomic_thread_fence(iree_memory_order_acquire);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_flush_range(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  iree_atomic_thread_fence(iree_memory_order_release);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_virtual_buffer_initialize(
    iree_hal_memory_contract_t* contract,
    const iree_hal_buffer_native_binding_t* bindings,
    const iree_hal_buffer_backing_facts_t* backing,
    iree_hal_memory_type_t memory_type, iree_device_size_t byte_length,
    iree_device_size_t mapping_granularity,
    iree_hal_memory_mapping_operations_t operations,
    iree_hal_memory_backend_type_t source_type,
    const iree_hal_buffer_vtable_t* vtable, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_t* reservation) {
  iree_hal_buffer_initialize(
      (iree_hal_buffer_placement_t){0}, &reservation->buffer, byte_length, 0,
      byte_length, memory_type, contract->allowed_access,
      contract->allowed_usage, vtable, &reservation->buffer);
  reservation->buffer.memory.contract = contract;
  iree_hal_memory_contract_retain(contract);
  reservation->buffer.memory.bindings = bindings;
  reservation->buffer.memory.backing = backing;
  reservation->buffer.host_binding_index =
      contract->binding_layout.host_binding_index;
  reservation->host_allocator = host_allocator;
  reservation->mapping_granularity = mapping_granularity;
  reservation->operations = operations;
  reservation->source_type = source_type;
  reservation->source_native_type_mask = 0;
  reservation->source_is_compatible = NULL;
}

//===----------------------------------------------------------------------===//
// Captured mapping transaction
//===----------------------------------------------------------------------===//

typedef struct iree_hal_memory_mapping_resolved_t {
  // Canonical native mapping target borrowed through the original view.
  iree_hal_virtual_buffer_t* target;
  // Native page-table projection within the target.
  uint32_t projection;
  // Native update in root coordinates.
  iree_hal_memory_mapping_update_t update;
} iree_hal_memory_mapping_resolved_t;

typedef struct iree_hal_memory_mapping_batch_storage_t {
  // Queue-owned resource and normalized entries.
  iree_hal_memory_mapping_batch_t base;
  // Allocator for the complete capture and resolution scratch.
  iree_allocator_t host_allocator;
  // Original-view retention policy.
  iree_hal_queue_map_flags_t flags;
  // Number of captured public descriptors.
  iree_host_size_t mapping_count;
  // Original views and view-relative ranges, retained until completion.
  iree_hal_memory_mapping_t* mappings;
  // Scratch sorted by canonical target, projection and native offset.
  iree_hal_memory_mapping_resolved_t* resolved;
  // Contiguous native update vectors referenced by the public entries.
  iree_hal_memory_mapping_update_t* updates;
} iree_hal_memory_mapping_batch_storage_t;

static void iree_hal_memory_mapping_batch_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_memory_mapping_batch_storage_t* batch =
      (iree_hal_memory_mapping_batch_storage_t*)resource;
  if (!iree_any_bit_set(batch->flags, IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS)) {
    for (iree_host_size_t i = 0; i < batch->mapping_count; ++i) {
      iree_hal_buffer_release(batch->mappings[i].source);
      iree_hal_buffer_release(batch->mappings[i].previous_source);
      iree_hal_buffer_release(batch->mappings[i].target);
    }
  }
  iree_allocator_free(batch->host_allocator, batch);
}

static const iree_hal_resource_vtable_t iree_hal_memory_mapping_batch_vtable = {
    .destroy = iree_hal_memory_mapping_batch_destroy,
};

static iree_status_t iree_hal_memory_mapping_validate_source(
    const iree_hal_memory_mapping_t* mapping, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_memory_access_t access,
    const void* domain) {
  const iree_hal_memory_contract_t* contract = mapping->target->memory.contract;
  const iree_hal_buffer_range_t range = mapping->target_range;
  if (!source_buffer) {
    if (source_offset || access) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "removal requires zero source offset and access");
    }
  } else {
    if (iree_any_bit_set(access, ~IREE_HAL_MEMORY_ACCESS_ALL) ||
        !iree_all_bits_set(source_buffer->allowed_access, access) ||
        !iree_all_bits_set(mapping->target->allowed_access, access)) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "mapping access exceeds source or target permissions");
    }
    if (source_offset > source_buffer->byte_length ||
        range.byte_length > source_buffer->byte_length - source_offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "mapping source range is outside its logical view");
    }
    const iree_hal_memory_contract_t* source = source_buffer->memory.contract;
    if (!source || source->domain != domain) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "source has no mapping contract in this group");
    }
    const uint32_t begin = mapping->scope.id ? mapping->scope.id : 1;
    const uint32_t end = mapping->scope.id ? begin + 1 : contract->scope_count;
    const iree_hal_memory_contract_t* alias =
        iree_hal_memory_contract_mapping_alias(source);
    for (uint32_t id = begin; id < end; ++id) {
      if (contract->scopes[id].interfaces &&
          (!iree_all_bits_set(alias->scopes[id].atomic_operations,
                              contract->scopes[id].atomic_operations) ||
           !iree_all_bits_set(alias->zero_compute_atomic_operations[id],
                              contract->zero_compute_atomic_operations[id]))) {
        return iree_make_status(
            IREE_STATUS_PERMISSION_DENIED,
            "source lacks target atomic guarantees at scope %u", id);
      }
      if (contract->scopes[id].interfaces &&
          source->scopes[id]
                  .bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE] ==
              IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
        return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                                "source is not remappable at target scope %u",
                                id);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_memory_mapping_validate(
    const iree_hal_memory_mapping_t* mapping, const void* domain) {
  if (!mapping->target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mapping target is null");
  }
  const iree_hal_memory_contract_t* contract = mapping->target->memory.contract;
  if (!contract || contract->domain != domain) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "mapping target and queue must belong to one group");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_virtual_buffer_validate_scope(mapping->target, mapping->scope));
  const iree_hal_buffer_vtable_t* vtable =
      (const iree_hal_buffer_vtable_t*)
          mapping->target->allocated_buffer->resource.vtable;
  if (!vtable->query_virtual) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "mapping target is not a virtual buffer");
  }
  if ((unsigned)mapping->operation >
          IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT ||
      (mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND &&
       (mapping->source || !mapping->previous_source)) ||
      ((mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_BIND ||
        mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT) &&
       mapping->previous_source) ||
      ((mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_BIND ||
        mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT) &&
       !mapping->source)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mapping operation has an invalid source");
  }
  const iree_hal_buffer_range_t range = mapping->target_range;
  if (!range.byte_length || range.byte_offset > mapping->target->byte_length ||
      range.byte_length > mapping->target->byte_length - range.byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "mapping target range is outside its logical view");
  }
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_validate_source(
      mapping, mapping->source, mapping->source_offset, mapping->access,
      domain));
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_validate_source(
      mapping, mapping->previous_source, mapping->previous_source_offset,
      IREE_HAL_MEMORY_ACCESS_NONE, domain));
  return iree_ok_status();
}

// Capture qualifies every currently available target. Commit requires all
// targets and native sources after the allocation dependencies have completed.
typedef enum iree_hal_memory_mapping_phase_e {
  IREE_HAL_MEMORY_MAPPING_PHASE_CAPTURE,
  IREE_HAL_MEMORY_MAPPING_PHASE_COMMIT,
} iree_hal_memory_mapping_phase_t;

static iree_status_t iree_hal_memory_mapping_resolve(
    const iree_hal_memory_mapping_t* mapping,
    iree_hal_memory_mapping_phase_t phase,
    iree_hal_memory_mapping_batch_storage_t* batch) {
  const iree_hal_virtual_buffer_view_t view =
      iree_hal_buffer_virtual_view(mapping->target);
  if (!view.root) {
    if (phase == IREE_HAL_MEMORY_MAPPING_PHASE_CAPTURE) {
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "mapping target has no committed virtual backing");
  }
  iree_hal_virtual_buffer_t* target = view.root;
  const iree_device_size_t target_offset =
      view.byte_offset + mapping->target_range.byte_offset;
  if (!iree_device_size_has_alignment(target_offset,
                                      target->mapping_granularity) ||
      (!iree_device_size_has_alignment(mapping->target_range.byte_length,
                                       target->mapping_granularity) &&
       mapping->target_range.byte_length !=
           target->buffer.byte_length - target_offset)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "mapping range is misaligned in native coordinates");
  }
  if (!(target->operations & (1u << mapping->operation))) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "virtual buffer does not support mapping operation %u",
        (unsigned)mapping->operation);
  }
  const iree_hal_memory_contract_t* contract = mapping->target->memory.contract;
  iree_hal_buffer_t* source_buffers[] = {
      mapping->source,
      mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT
          ? mapping->source
          : mapping->previous_source,
  };
  const iree_device_size_t source_offsets[] = {
      mapping->source_offset,
      mapping->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT
          ? mapping->source_offset
          : mapping->previous_source_offset,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_buffers); ++i) {
    if (source_buffers[i] &&
        !iree_hal_virtual_buffer_source_is_compatible(
            target, mapping->scope.id, source_buffers[i]->memory.contract)) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "source does not preserve the virtual buffer mapping contract");
    }
  }
  const uint32_t begin = mapping->scope.id ? mapping->scope.id : 1;
  const uint32_t end = mapping->scope.id ? begin + 1 : contract->scope_count;
  const iree_host_size_t first = batch->base.update_count;
  iree_status_t status = iree_ok_status();
  for (uint32_t id = begin; id < end && iree_status_is_ok(status); ++id) {
    if (!contract->scopes[id].interfaces) {
      continue;
    }
    const uint32_t projection = target->projections[id];
    bool visited = false;
    for (iree_host_size_t i = first; i < batch->base.update_count; ++i) {
      visited |= batch->resolved[i].projection == projection;
    }
    if (visited) {
      continue;
    }
    iree_hal_memory_mapping_resolved_t* resolved =
        &batch->resolved[batch->base.update_count++];
    *resolved = (iree_hal_memory_mapping_resolved_t){
        .target = target,
        .projection = projection,
        .update = {.scope_id = id,
                   .target_range = {target_offset,
                                    mapping->target_range.byte_length},
                   .access = mapping->access,
                   .operation = mapping->operation},
    };
    for (iree_host_size_t source_index = 0;
         source_index < IREE_ARRAYSIZE(source_buffers) &&
         iree_status_is_ok(status) &&
         phase == IREE_HAL_MEMORY_MAPPING_PHASE_COMMIT;
         ++source_index) {
      iree_hal_buffer_t* source_buffer = source_buffers[source_index];
      if (!source_buffer) {
        continue;
      }
      const iree_hal_memory_contract_t* source_contract =
          source_buffer->memory.contract;
      const iree_device_size_t source_offset = source_offsets[source_index];
      const iree_hal_buffer_memory_view_t memory =
          iree_hal_buffer_memory_view(source_buffer);
      if (!memory.bindings) {
        status =
            iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "mapping source allocation has not committed");
      } else {
        const iree_hal_buffer_native_binding_slot_t slot = {
            .index = source_contract->scopes[id]
                         .bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE],
            .type = IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
        };
        iree_hal_buffer_native_binding_t source =
            iree_hal_buffer_memory_native_binding(&memory, slot);
        if (!source.mapping_source.descriptor ||
            source.mapping_source.offset > UINT64_MAX - source_offset ||
            !iree_device_size_has_alignment(
                source.mapping_source.offset + source_offset,
                source.mapping_source.granularity) ||
            !iree_device_size_has_alignment(
                mapping->target_range.byte_length,
                source.mapping_source.granularity)) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "native mapping source is unavailable or misaligned");
        } else {
          source.mapping_source.offset += source_offset;
          if (source_index == 0) {
            resolved->update.source = source;
          } else {
            resolved->update.previous_source = source;
          }
        }
      }
    }
  }
  return status;
}

static int iree_hal_memory_mapping_compare(const void* lhs, const void* rhs) {
  const iree_hal_memory_mapping_resolved_t* a = lhs;
  const iree_hal_memory_mapping_resolved_t* b = rhs;
  if (a->target != b->target) {
    return (uintptr_t)a->target < (uintptr_t)b->target ? -1 : 1;
  }
  if (a->projection != b->projection) {
    return a->projection < b->projection ? -1 : 1;
  }
  return a->update.target_range.byte_offset < b->update.target_range.byte_offset
             ? -1
         : a->update.target_range.byte_offset >
                 b->update.target_range.byte_offset
             ? 1
             : 0;
}

// Original views retain ownership independently of the native update vector.
// Only committed source bindings identify contiguous bytes of one native owner;
// the null source scratch populated during capture does not identify removals.
static bool iree_hal_memory_mapping_updates_are_contiguous(
    const iree_hal_memory_mapping_update_t* previous,
    const iree_hal_memory_mapping_update_t* current) {
  return previous->scope_id == current->scope_id &&
         previous->operation == current->operation &&
         previous->access == current->access &&
         previous->target_range.byte_offset +
                 previous->target_range.byte_length ==
             current->target_range.byte_offset &&
         previous->previous_source.mapping_source.descriptor ==
             current->previous_source.mapping_source.descriptor &&
         previous->previous_source.mapping_source.granularity ==
             current->previous_source.mapping_source.granularity &&
         (!previous->previous_source.mapping_source.descriptor ||
          previous->previous_source.mapping_source.offset +
                  previous->target_range.byte_length ==
              current->previous_source.mapping_source.offset) &&
         previous->source.mapping_source.descriptor ==
             current->source.mapping_source.descriptor &&
         previous->source.mapping_source.granularity ==
             current->source.mapping_source.granularity &&
         (!previous->source.mapping_source.descriptor ||
          previous->source.mapping_source.offset +
                  previous->target_range.byte_length ==
              current->source.mapping_source.offset);
}

static iree_status_t iree_hal_memory_mapping_batch_resolve_phase(
    iree_hal_memory_mapping_batch_storage_t* batch,
    iree_hal_memory_mapping_phase_t phase) {
  batch->base.update_count = 0;
  batch->base.entry_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < batch->mapping_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_memory_mapping_resolve(&batch->mappings[i], phase, batch);
  }
  if (iree_status_is_ok(status)) {
    qsort(batch->resolved, batch->base.update_count, sizeof(batch->resolved[0]),
          iree_hal_memory_mapping_compare);
    for (iree_host_size_t i = 1;
         i < batch->base.update_count && iree_status_is_ok(status); ++i) {
      const iree_hal_memory_mapping_resolved_t* previous =
          &batch->resolved[i - 1];
      const iree_hal_memory_mapping_resolved_t* current = &batch->resolved[i];
      if (previous->target == current->target &&
          previous->projection == current->projection &&
          previous->update.target_range.byte_offset +
                  previous->update.target_range.byte_length >
              current->update.target_range.byte_offset) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "mapping batch overlaps in one native projection");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_memory_mapping_batch_entry_t* entries =
        (iree_hal_memory_mapping_batch_entry_t*)batch->base.entries;
    const iree_host_size_t resolved_count = batch->base.update_count;
    batch->base.update_count = 0;
    for (iree_host_size_t i = 0; i < resolved_count; ++i) {
      const iree_hal_memory_mapping_resolved_t* resolved = &batch->resolved[i];
      if (!i || resolved->target != batch->resolved[i - 1].target ||
          resolved->projection != batch->resolved[i - 1].projection) {
        entries[batch->base.entry_count++] =
            (iree_hal_memory_mapping_batch_entry_t){
                .target = resolved->target,
                .projection = resolved->projection,
                .updates = &batch->updates[batch->base.update_count],
            };
      }
      iree_hal_memory_mapping_batch_entry_t* entry =
          &entries[batch->base.entry_count - 1];
      if (phase == IREE_HAL_MEMORY_MAPPING_PHASE_COMMIT &&
          entry->update_count &&
          iree_hal_memory_mapping_updates_are_contiguous(
              &batch->updates[batch->base.update_count - 1],
              &resolved->update)) {
        batch->updates[batch->base.update_count - 1].target_range.byte_length +=
            resolved->update.target_range.byte_length;
      } else {
        batch->updates[batch->base.update_count++] = resolved->update;
        ++entry->update_count;
      }
    }
  }
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_memory_mapping_batch_resolve(iree_hal_memory_mapping_batch_t* batch) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_memory_mapping_batch_resolve_phase(
      (iree_hal_memory_mapping_batch_storage_t*)batch,
      IREE_HAL_MEMORY_MAPPING_PHASE_COMMIT);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_memory_mapping_batch_create(
    iree_hal_queue_t* queue, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings, iree_hal_queue_map_flags_t flags,
    iree_allocator_t host_allocator,
    iree_hal_memory_mapping_batch_t** out_batch) {
  *out_batch = NULL;
  if (!mapping_count) {
    return iree_ok_status();
  }
  const void* domain = iree_hal_queue_family(queue)->memory.domain;
  iree_host_size_t capacity = 0;
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_memory_mapping_validate(&mappings[i], domain));
    const iree_host_size_t projection_bound =
        mappings[i].scope.id
            ? 1
            : mappings[i].target->memory.contract->scope_count - 1;
    if (!iree_host_size_checked_add(capacity, projection_bound, &capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "mapping count overflows");
    }
  }
  iree_host_size_t total_size = 0;
  iree_host_size_t mappings_offset = 0;
  iree_host_size_t resolved_offset = 0;
  iree_host_size_t updates_offset = 0;
  iree_host_size_t entries_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_memory_mapping_batch_storage_t), &total_size,
      IREE_STRUCT_FIELD(mapping_count, iree_hal_memory_mapping_t,
                        &mappings_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_memory_mapping_resolved_t,
                        &resolved_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_memory_mapping_update_t,
                        &updates_offset),
      IREE_STRUCT_FIELD(capacity, iree_hal_memory_mapping_batch_entry_t,
                        &entries_offset)));
  iree_hal_memory_mapping_batch_storage_t* batch = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&batch));
  iree_hal_resource_initialize(&iree_hal_memory_mapping_batch_vtable,
                               &batch->base.resource);
  batch->host_allocator = host_allocator;
  batch->flags = flags;
  batch->mapping_count = mapping_count;
  batch->base.capacity = capacity;
  batch->mappings =
      (iree_hal_memory_mapping_t*)((uint8_t*)batch + mappings_offset);
  batch->resolved =
      (iree_hal_memory_mapping_resolved_t*)((uint8_t*)batch + resolved_offset);
  batch->updates =
      (iree_hal_memory_mapping_update_t*)((uint8_t*)batch + updates_offset);
  batch->base.entries =
      (iree_hal_memory_mapping_batch_entry_t*)((uint8_t*)batch +
                                               entries_offset);
  memcpy(batch->mappings, mappings, mapping_count * sizeof(*mappings));
  if (!iree_any_bit_set(flags, IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS)) {
    for (iree_host_size_t i = 0; i < mapping_count; ++i) {
      iree_hal_buffer_retain(mappings[i].target);
      iree_hal_buffer_retain(mappings[i].source);
      iree_hal_buffer_retain(mappings[i].previous_source);
    }
  }
  iree_status_t status = iree_hal_memory_mapping_batch_resolve_phase(
      batch, IREE_HAL_MEMORY_MAPPING_PHASE_CAPTURE);
  if (iree_status_is_ok(status)) {
    *out_batch = &batch->base;
  } else {
    iree_hal_memory_mapping_batch_release(&batch->base);
  }
  return status;
}

static iree_status_t iree_hal_memory_mapping_batch_execute(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  iree_hal_memory_mapping_batch_t* batch = user_data;
  if (!batch) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_memory_mapping_batch_resolve(batch);
  for (iree_host_size_t i = 0;
       i < batch->entry_count && iree_status_is_ok(status); ++i) {
    const iree_hal_buffer_vtable_t* vtable =
        (const iree_hal_buffer_vtable_t*)batch->entries[i]
            .target->buffer.resource.vtable;
    if (!vtable->update_mappings) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "virtual buffer updates require a native mapping queue");
    }
  }
  for (iree_host_size_t i = 0;
       i < batch->entry_count && iree_status_is_ok(status); ++i) {
    const iree_hal_memory_mapping_batch_entry_t* entry = &batch->entries[i];
    iree_hal_buffer_t* target = &entry->target->buffer;
    const iree_hal_buffer_vtable_t* vtable =
        (const iree_hal_buffer_vtable_t*)target->resource.vtable;
    status = vtable->update_mappings(target, entry->projection,
                                     entry->update_count, entry->updates);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_memory_mapping_prepare(
    iree_hal_queue_t* queue, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings, iree_hal_queue_map_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_host_call_t* out_call) {
  *out_call = (iree_hal_host_call_t){0};
  iree_hal_memory_mapping_batch_t* batch = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_batch_create(
      queue, mapping_count, mappings, flags, host_allocator, &batch));
  *out_call = iree_hal_make_host_call_with_resource(
      iree_hal_memory_mapping_batch_execute, batch,
      batch ? &batch->resource : NULL);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_memory_mapping_queue_map(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_hal_host_call_t call;
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(
      iree_hal_queue_family_device(iree_hal_queue_family(queue)));
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_prepare(
      queue, mapping_count, mappings, flags, host_allocator, &call));
  const uint64_t args[4] = {0};
  iree_status_t status =
      iree_hal_queue_host_call(queue, waits, signals, call, args, 0);
  iree_hal_resource_release(call.resource);
  return status;
}
