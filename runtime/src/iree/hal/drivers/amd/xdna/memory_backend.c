// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/memory_backend.h"

#include "iree/base/internal/math.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"

typedef struct iree_hal_amd_xdna_slab_pool_plan_t {
  // Generic cold construction product.
  iree_hal_slab_pool_plan_t base;
  // Allocator for this temporary plan and its inline access array.
  iree_allocator_t host_allocator;
  // Borrowed HAL owner for materialized buffer placement.
  iree_hal_device_t* device;
  // Borrowed native owner and diagnostic sink.
  iree_hal_amd_xdna_context_t* context;
  // Qualified native allocation policy copied by the created provider.
  iree_hal_amd_xdna_slab_provider_options_t provider_options;
  // Existing native-owner progress resources captured during qualification.
  struct {
    // Placement-local capacity notification.
    iree_async_notification_t* notification;
    // Independent cold allocation and retirement owner.
    iree_hal_memory_maintenance_t* maintenance;
    // Sealed group's completion probe.
    iree_hal_pool_epoch_query_t epoch_query;
    // Sealed group's shared completion tracker.
    iree_async_frontier_tracker_t* tracker;
  } progress;
  // Owned immutable access facts retained by the created pool.
  iree_hal_memory_contract_t* contract;
  // Borrowed until synchronous creation copies the diagnostic name.
  iree_string_view_t trace_name;
  // Caller-ordered native consumers copied again by the created provider.
  amdf_memory_device_access_t accesses[];
} iree_hal_amd_xdna_slab_pool_plan_t;

static void iree_hal_amd_xdna_slab_pool_plan_destroy(
    iree_hal_slab_pool_plan_t* base) {
  iree_hal_amd_xdna_slab_pool_plan_t* plan =
      (iree_hal_amd_xdna_slab_pool_plan_t*)base;
  iree_hal_memory_contract_release(plan->contract);
  iree_allocator_free(plan->host_allocator, plan);
}

static iree_status_t iree_hal_amd_xdna_slab_pool_plan_create(
    iree_hal_slab_pool_plan_t* base, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  iree_hal_amd_xdna_slab_pool_plan_t* plan =
      (iree_hal_amd_xdna_slab_pool_plan_t*)base;
  iree_hal_slab_provider_t* provider = NULL;
  iree_status_t status = iree_hal_amd_xdna_slab_provider_create(
      plan->device, plan->context, &plan->provider_options, host_allocator,
      &provider);
  if (iree_status_is_ok(status)) {
    const iree_hal_passthrough_pool_options_t options = {
        .memory_contract = plan->contract,
        .epoch_query = plan->progress.epoch_query,
        .trace_name = plan->trace_name,
    };
    status = iree_hal_passthrough_pool_create(
        options, provider, plan->progress.notification, plan->progress.tracker,
        plan->progress.maintenance, host_allocator, out_pool);
  }
  iree_hal_slab_provider_release(provider);
  return status;
}

static const iree_hal_slab_pool_plan_vtable_t
    iree_hal_amd_xdna_slab_pool_plan_vtable = {
        .destroy = iree_hal_amd_xdna_slab_pool_plan_destroy,
        .create = iree_hal_amd_xdna_slab_pool_plan_create,
};

static amdf_memory_map_flags_t iree_hal_amd_xdna_memory_host_access(
    iree_hal_memory_access_t access) {
  amdf_memory_map_flags_t result = 0;
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
    result |= AMDF_MEMORY_MAP_FLAG_READ;
  }
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    result |= AMDF_MEMORY_MAP_FLAG_WRITE;
  }
  return result;
}

static iree_hal_buffer_usage_t iree_hal_amd_xdna_memory_mapping_usage(
    iree_hal_pool_host_access_t host) {
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (iree_any_bit_set(host.modes, IREE_HAL_MAPPING_MODE_SCOPED)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }
  if (iree_any_bit_set(host.modes, IREE_HAL_MAPPING_MODE_PERSISTENT)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  }
  return usage;
}

static bool iree_hal_amd_xdna_memory_supports_host_access(
    const amdf_memory_profile_t* profile, iree_hal_pool_host_access_t host) {
  if (!host.access) {
    return true;
  }
  const amdf_memory_map_flags_t access =
      iree_hal_amd_xdna_memory_host_access(host.access);
  return host.cacheability == IREE_HAL_HOST_CACHEABILITY_UNKNOWN &&
         iree_all_bits_set(profile->roles, AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
         iree_all_bits_set(profile->supported_flags,
                           AMDF_MEMORY_FLAG_HOST_VISIBLE) &&
         iree_all_bits_set(profile->host_mapping.supported_access, access);
}

// Every slab owns one persistent read/write mapping used by queue transfers
// and cache maintenance, independently of the public host mapping contract.
static bool iree_hal_amd_xdna_memory_supports_private_mapping(
    const amdf_memory_profile_t* profile) {
  return iree_all_bits_set(
      profile->host_mapping.supported_access,
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE);
}

static bool iree_hal_amd_xdna_memory_host_transition(
    const amdf_cache_transition_t* transition,
    amdf_host_cache_operation_t operation,
    iree_device_size_t* maintenance_alignment) {
  if (transition->kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
    return true;
  }
  if (transition->kind != AMDF_CACHE_TRANSITION_KIND_RANGE ||
      (transition->executor != AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT &&
       transition->executor != AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API) ||
      transition->host_operation != operation ||
      !iree_device_size_is_power_of_two(transition->range_granularity)) {
    return false;
  }
  *maintenance_alignment =
      iree_max(*maintenance_alignment, transition->range_granularity);
  return true;
}

static iree_status_t iree_hal_amd_xdna_memory_query_pair(
    const iree_hal_amd_xdna_memory_backend_t* owner,
    const amdf_memory_profile_pair_query_t* query,
    amdf_host_cache_operation_t host_operation,
    iree_device_size_t* maintenance_alignment, bool* out_supported,
    bool* out_no_op) {
  *out_supported = false;
  *out_no_op = false;
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  const amdf_status_t native_status =
      owner->context->api->memory_scope_query_pair_info(
          owner->context->data_source.scope, query, &pair);
  if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      native_status, "memory_scope_query_pair_info"));
  if (!iree_all_bits_set(pair.flags,
                         AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE)) {
    return iree_ok_status();
  }

  bool supported = false;
  if (host_operation == AMDF_HOST_CACHE_OPERATION_FLUSH) {
    supported = iree_hal_amd_xdna_memory_host_transition(
                    &pair.release, host_operation, maintenance_alignment) &&
                pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE;
    *out_no_op = pair.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE;
  } else if (host_operation == AMDF_HOST_CACHE_OPERATION_INVALIDATE) {
    supported = pair.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE &&
                iree_hal_amd_xdna_memory_host_transition(
                    &pair.acquire, host_operation, maintenance_alignment);
    *out_no_op = pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE;
  } else {
    supported = pair.release.kind == AMDF_CACHE_TRANSITION_KIND_NONE &&
                pair.acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE;
    *out_no_op = supported;
  }
  *out_supported = supported;
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_memory_qualify_pairs(
    const iree_hal_amd_xdna_memory_backend_t* owner,
    iree_hal_pool_scope_t scope, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    const uint32_t* family_access_ordinals,
    const iree_hal_amd_xdna_memory_backend_t* const* family_backends,
    uint32_t profile_ordinal, iree_device_size_t* out_maintenance_alignment,
    bool* out_host_coherent, bool* out_supported) {
  *out_maintenance_alignment = 1;
  *out_host_coherent = true;
  *out_supported = false;
  amdf_memory_profile_pair_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(query),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = access_count,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .accesses = accesses,
  };
  iree_status_t status = iree_ok_status();
  bool supported = true;
  for (iree_host_size_t i = 0;
       i < scope.family_count && supported && iree_status_is_ok(status); ++i) {
    const uint32_t access_ordinal = family_access_ordinals[i];
    const uint32_t family_ordinal =
        family_backends[i]->context->queue_family_ordinal;
    query.producer = (amdf_memory_profile_site_t){
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
        .value.host_access =
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    query.consumer = (amdf_memory_profile_site_t){
        .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
        .value.device =
            {
                .access_ordinal = access_ordinal,
                .queue_family_ordinal = family_ordinal,
            },
    };
    bool no_op = false;
    status = iree_hal_amd_xdna_memory_query_pair(
        owner, &query, AMDF_HOST_CACHE_OPERATION_FLUSH,
        out_maintenance_alignment, &supported, &no_op);
    *out_host_coherent &= no_op;
    if (!supported || !iree_status_is_ok(status)) {
      break;
    }
    query.producer = query.consumer;
    query.consumer = (amdf_memory_profile_site_t){
        .kind = AMDF_MEMORY_SITE_KIND_HOST,
        .value.host_access =
            AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    status = iree_hal_amd_xdna_memory_query_pair(
        owner, &query, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
        out_maintenance_alignment, &supported, &no_op);
    *out_host_coherent &= no_op;
  }
  for (iree_host_size_t i = 0;
       i < scope.family_count && supported && iree_status_is_ok(status); ++i) {
    for (iree_host_size_t j = 0;
         j < scope.family_count && supported && iree_status_is_ok(status);
         ++j) {
      query.producer = (amdf_memory_profile_site_t){
          .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
          .value.device =
              {
                  .access_ordinal = family_access_ordinals[i],
                  .queue_family_ordinal =
                      family_backends[i]->context->queue_family_ordinal,
              },
      };
      query.consumer = (amdf_memory_profile_site_t){
          .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
          .value.device =
              {
                  .access_ordinal = family_access_ordinals[j],
                  .queue_family_ordinal =
                      family_backends[j]->context->queue_family_ordinal,
              },
      };
      bool no_op = false;
      status = iree_hal_amd_xdna_memory_query_pair(
          owner, &query, AMDF_HOST_CACHE_OPERATION_NONE,
          out_maintenance_alignment, &supported, &no_op);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_supported = supported;
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_slab_pool_emit_plan(
    const iree_hal_amd_xdna_memory_backend_t* owner,
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    const uint32_t* family_access_ordinals,
    const iree_hal_amd_xdna_memory_backend_t* const* family_backends,
    const amdf_memory_profile_t* profile, const uint16_t* binding_types,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  if (profile->memory_class != AMDF_MEMORY_CLASS_SYSTEM ||
      !iree_all_bits_set(profile->roles,
                         AMDF_MEMORY_PROFILE_ROLE_CREATE |
                             AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) ||
      !iree_all_bits_set(profile->supported_flags,
                         AMDF_MEMORY_FLAG_HOST_VISIBLE) ||
      !iree_hal_amd_xdna_memory_supports_private_mapping(profile) ||
      !iree_hal_amd_xdna_memory_supports_host_access(profile, scope.host)) {
    return iree_ok_status();
  }

  iree_hal_pool_host_access_t host = scope.host;
  const iree_hal_pool_host_access_t preferred = options->preferences.host;
  const iree_hal_pool_host_access_t combined = {
      .access = host.access | preferred.access,
      .modes = host.modes | preferred.modes,
      .cacheability =
          preferred.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN
              ? preferred.cacheability
              : host.cacheability,
  };
  bool preferred_host_added = false;
  if (iree_hal_amd_xdna_memory_supports_host_access(profile, combined)) {
    host = combined;
    preferred_host_added =
        host.access != scope.host.access || host.modes != scope.host.modes;
  }

  iree_device_size_t maintenance_alignment = 1;
  bool host_coherent = false;
  bool supported = false;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_memory_qualify_pairs(
      owner, scope, access_count, accesses, family_access_ordinals,
      family_backends, profile->ordinal, &maintenance_alignment, &host_coherent,
      &supported));
  if (!supported) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    if (iree_any_bit_set(scope.families[i].requirements,
                         IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST) &&
        !host_coherent) {
      return iree_ok_status();
    }
  }

  iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (host_coherent) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    usage |= scope.families[i].usage;
  }
  const iree_hal_buffer_usage_t mapping_usage =
      iree_hal_amd_xdna_memory_mapping_usage(host);

  iree_host_size_t plan_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_slab_pool_plan_t), &plan_size,
      IREE_STRUCT_FIELD_FAM(access_count, amdf_memory_device_access_t)));
  iree_hal_amd_xdna_slab_pool_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, plan_size, (void**)&plan));
  memset(plan, 0, plan_size);
  plan->base.vtable = &iree_hal_amd_xdna_slab_pool_plan_vtable;
  plan->base.info = (iree_hal_slab_pool_plan_info_t){
      .preference = preferred_host_added ? 1024 : 0,
      .host = host,
  };
  plan->host_allocator = host_allocator;
  plan->device = owner->device;
  plan->context = owner->context;
  memcpy(plan->accesses, accesses, access_count * sizeof(*accesses));
  plan->provider_options = (iree_hal_amd_xdna_slab_provider_options_t){
      .scope = owner->context->data_source.scope,
      .profile = *profile,
      .access_count = access_count,
      .accesses = plan->accesses,
      .memory_type = memory_type,
      .supported_usage = usage | mapping_usage,
      .maintenance_alignment = maintenance_alignment,
  };
  plan->progress.notification = owner->notification;
  plan->progress.maintenance = owner->maintenance;
  plan->progress.epoch_query = owner->epoch_query;
  plan->progress.tracker =
      iree_hal_device_topology_info(owner->device)->frontier.tracker;
  plan->trace_name = options->trace_name;
  const uint16_t binding_count = (uint16_t)(access_count + 1);
  const iree_hal_buffer_binding_layout_t binding_layout = {
      .byte_length =
          (uint32_t)(binding_count * sizeof(iree_hal_buffer_native_binding_t)),
      .binding_count = binding_count,
      .host_binding_index = 0,
      .types = binding_types,
  };
  iree_status_t status = iree_hal_memory_contract_create(
      iree_hal_device_group_memory_domain(group),
      iree_hal_device_group_memory_scope_count(group), &binding_layout,
      host_allocator, &plan->contract);
  if (iree_status_is_ok(status)) {
    iree_hal_memory_contract_t* contract = plan->contract;
    contract->host = host;
    contract->placement = plan->base.info.placement;
    contract->buffer_params = (iree_hal_buffer_params_t){
        .usage = usage | mapping_usage,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = memory_type,
        .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
    };
    if (host.access) {
      contract->scopes[1].interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
      contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_HOST] = 0;
    }
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      const uint32_t queue_scope_id =
          scope.families[i].family->memory.queue_scope_id;
      const uint16_t binding_index = (uint16_t)(family_access_ordinals[i] + 1);
      for (uint32_t id = queue_scope_id; id <= queue_scope_id + 1; ++id) {
        contract->scopes[id].usage = scope.families[i].usage;
        contract->scopes[id].interfaces =
            1u << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
        contract->scopes[id].bindings[IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA] =
            binding_index;
      }
    }
    status = callback.fn(callback.user_data, &plan->base);
  } else {
    iree_hal_amd_xdna_slab_pool_plan_destroy(&plan->base);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_slab_pool_query(
    void* self, iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  (void)self;
  if (!scope.family_count ||
      options->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED ||
      (scope.host.access &&
       scope.host.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN) ||
      scope.family_count > UINT32_MAX) {
    return iree_ok_status();
  }

  iree_host_size_t binding_capacity = 0;
  if (!iree_host_size_checked_add(scope.family_count, 1, &binding_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "XDNA memory scope is too large");
  }
  iree_host_size_t scratch_size = 0;
  iree_host_size_t accesses_offset = 0;
  iree_host_size_t family_access_ordinals_offset = 0;
  iree_host_size_t family_backends_offset = 0;
  iree_host_size_t capabilities_offset = 0;
  iree_host_size_t binding_types_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scratch_size,
      IREE_STRUCT_FIELD(scope.family_count, amdf_memory_device_access_t,
                        &accesses_offset),
      IREE_STRUCT_FIELD(scope.family_count, uint32_t,
                        &family_access_ordinals_offset),
      IREE_STRUCT_FIELD(scope.family_count,
                        const iree_hal_amd_xdna_memory_backend_t*,
                        &family_backends_offset),
      IREE_STRUCT_FIELD(scope.family_count, amdf_memory_access_capabilities_t,
                        &capabilities_offset),
      IREE_STRUCT_FIELD(binding_capacity, uint16_t, &binding_types_offset)));
  uint8_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, scratch_size, (void**)&scratch));
  memset(scratch, 0, scratch_size);
  amdf_memory_device_access_t* accesses =
      (amdf_memory_device_access_t*)(scratch + accesses_offset);
  uint32_t* family_access_ordinals =
      (uint32_t*)(scratch + family_access_ordinals_offset);
  const iree_hal_amd_xdna_memory_backend_t** family_backends =
      (const iree_hal_amd_xdna_memory_backend_t**)(scratch +
                                                   family_backends_offset);
  amdf_memory_access_capabilities_t* capabilities =
      (amdf_memory_access_capabilities_t*)(scratch + capabilities_offset);
  uint16_t* binding_types = (uint16_t*)(scratch + binding_types_offset);

  bool supported = true;
  iree_host_size_t matched_family_count = 0;
  uint32_t access_count = 0;
  const iree_hal_amd_xdna_memory_backend_t* owner = NULL;
  for (iree_host_size_t device_index = 0;
       device_index < iree_hal_device_group_device_count(group) && supported;
       ++device_index) {
    iree_hal_device_t* device =
        iree_hal_device_group_device_at(group, device_index);
    for (iree_host_size_t family_index = 0;
         family_index < scope.family_count && supported; ++family_index) {
      if (iree_hal_queue_family_device(scope.families[family_index].family) !=
          device) {
        continue;
      }
      ++matched_family_count;
      const iree_hal_memory_backend_t* base =
          iree_hal_device_memory_backend(device);
      if (!base || base->type != IREE_HAL_MEMORY_BACKEND_AMDF ||
          !base->factory_count || !base->factories ||
          base->factories[0]->query != iree_hal_amd_xdna_slab_pool_query ||
          iree_any_bit_set(scope.families[family_index].usage,
                           ~(IREE_HAL_BUFFER_USAGE_TRANSFER |
                             IREE_HAL_BUFFER_USAGE_STORAGE)) ||
          (scope.families[family_index].interfaces &
           ~(UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA)) ||
          iree_any_bit_set(scope.families[family_index].requirements,
                           IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED)) {
        supported = false;
        break;
      }
      const iree_hal_amd_xdna_memory_backend_t* backend =
          (const iree_hal_amd_xdna_memory_backend_t*)base;
      if (!owner) {
        owner = backend;
      } else if (backend->context->api != owner->context->api ||
                 backend->context->data_source.scope !=
                     owner->context->data_source.scope) {
        supported = false;
        break;
      }
      uint32_t access_ordinal = 0;
      for (; access_ordinal < access_count; ++access_ordinal) {
        if (accesses[access_ordinal].device == backend->context->device) {
          break;
        }
      }
      if (access_ordinal == access_count) {
        accesses[access_count] = (amdf_memory_device_access_t){
            .device = backend->context->device,
            .requirements =
                {
                    .access =
                        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                    .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
                    .address_kinds = UINT64_C(1)
                                     << AMDF_MEMORY_ADDRESS_XDNA_DMA,
                },
        };
        ++access_count;
      }
      if (iree_any_bit_set(scope.families[family_index].requirements,
                           IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST)) {
        accesses[access_ordinal].requirements.flags |=
            AMDF_MEMORY_FLAG_HOST_COHERENT;
      }
      family_access_ordinals[family_index] = access_ordinal;
      family_backends[family_index] = backend;
    }
  }
  supported &= matched_family_count == scope.family_count && owner != NULL &&
               access_count < IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
  if (!supported) {
    iree_allocator_free(host_allocator, scratch);
    return iree_ok_status();
  }

  binding_types[0] = IREE_HAL_BUFFER_INTERFACE_HOST;
  for (uint32_t i = 0; i < access_count; ++i) {
    binding_types[i + 1] = IREE_HAL_BUFFER_INTERFACE_XDNA_SHIM_DMA;
  }
  amdf_memory_scope_info_t scope_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope_info),
  };
  iree_status_t status = IREE_HAL_AMD_STATUS_FROM_AMDF(
      owner->context->api->memory_scope_query_info(
          owner->context->data_source.scope, &scope_info),
      "memory_scope_query_info");
  for (uint32_t profile_ordinal = 0;
       profile_ordinal < scope_info.memory_profile_count &&
       iree_status_is_ok(status);
       ++profile_ordinal) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    for (uint32_t i = 0; i < access_count; ++i) {
      capabilities[i] = (amdf_memory_access_capabilities_t){
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities[i]),
      };
    }
    const amdf_status_t native_status =
        owner->context->api->memory_scope_query_device_profile(
            owner->context->data_source.scope, profile_ordinal, access_count,
            accesses, &profile, capabilities);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                           "memory_scope_query_device_profile");
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_xdna_slab_pool_emit_plan(
          owner, group, scope, options, access_count, accesses,
          family_access_ordinals, family_backends, &profile, binding_types,
          callback, host_allocator);
    }
  }
  iree_allocator_free(host_allocator, scratch);
  return status;
}

static const iree_hal_slab_pool_factory_t iree_hal_amd_xdna_slab_pool_factory =
    {
        .query = iree_hal_amd_xdna_slab_pool_query,
};
static const iree_hal_slab_pool_factory_t* const
    iree_hal_amd_xdna_slab_pool_factories[] = {
        &iree_hal_amd_xdna_slab_pool_factory,
};

void iree_hal_amd_xdna_memory_backend_initialize(
    iree_hal_device_t* device, iree_hal_amd_xdna_context_t* context,
    iree_async_notification_t* notification,
    iree_hal_memory_maintenance_t* maintenance,
    iree_hal_pool_epoch_query_t epoch_query,
    iree_hal_amd_xdna_memory_backend_t* out_backend) {
  *out_backend = (iree_hal_amd_xdna_memory_backend_t){
      .base =
          {
              .type = IREE_HAL_MEMORY_BACKEND_AMDF,
              .factory_count =
                  IREE_ARRAYSIZE(iree_hal_amd_xdna_slab_pool_factories),
              .factories = iree_hal_amd_xdna_slab_pool_factories,
          },
      .device = device,
      .context = context,
      .notification = notification,
      .maintenance = maintenance,
      .epoch_query = epoch_query,
  };
}
