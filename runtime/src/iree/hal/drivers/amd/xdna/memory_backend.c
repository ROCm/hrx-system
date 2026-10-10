// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/memory_backend.h"

#include "iree/base/internal/math.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/drivers/amd/xdna/memory_transition.h"
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

typedef enum iree_hal_amd_xdna_memory_site_kind_e {
  IREE_HAL_AMD_XDNA_MEMORY_SITE_UNKNOWN = 0,
  IREE_HAL_AMD_XDNA_MEMORY_SITE_HOST = 1,
  IREE_HAL_AMD_XDNA_MEMORY_SITE_QUEUE = 2,
} iree_hal_amd_xdna_memory_site_kind_t;

typedef struct iree_hal_amd_xdna_memory_transition_site_t {
  // Native interpretation of this exact HAL scope.
  iree_hal_amd_xdna_memory_site_kind_t kind;
  // Whether this exact queue requires coherent public host access.
  bool requires_host_coherent;
  // Prospective libamdf site when kind is HOST or QUEUE.
  amdf_memory_profile_site_t native;
} iree_hal_amd_xdna_memory_transition_site_t;

typedef struct iree_hal_amd_xdna_memory_transition_query_t {
  // Borrowed native query owner.
  const iree_hal_amd_xdna_memory_backend_t* owner;
  // Base construction facts completed with exact sites for each query.
  amdf_memory_profile_pair_query_t native;
  // Borrowed dense construction-time site interpretations.
  const iree_hal_amd_xdna_memory_transition_site_t* sites;
  // Largest explicit range granularity observed across admitted exact pairs.
  iree_device_size_t maintenance_alignment;
  // Whether every host/XDNA direction requires no host cache action.
  bool host_coherent;
  // Whether every admitted host/queue pair has a complete native route.
  bool complete;
} iree_hal_amd_xdna_memory_transition_query_t;

static void iree_hal_amd_xdna_memory_accumulate_pair(
    const iree_hal_amd_xdna_memory_transition_site_t* producer,
    const iree_hal_amd_xdna_memory_transition_site_t* consumer,
    const iree_hal_memory_pair_info_t* pair,
    iree_hal_amd_xdna_memory_transition_query_t* query) {
  if (!iree_all_bits_set(pair->flags,
                         IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE) ||
      pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN ||
      pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN) {
    query->complete = false;
    return;
  }
  if (pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE) {
    query->maintenance_alignment =
        iree_max(query->maintenance_alignment, pair->release.range_granularity);
  }
  if (pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_RANGE) {
    query->maintenance_alignment =
        iree_max(query->maintenance_alignment, pair->acquire.range_granularity);
  }
  if (producer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_HOST &&
      consumer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_QUEUE) {
    const bool coherent =
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    query->host_coherent &= coherent;
    query->complete &= !consumer->requires_host_coherent || coherent;
  } else if (producer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_QUEUE &&
             consumer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_HOST) {
    const bool coherent =
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    query->host_coherent &= coherent;
    query->complete &= !producer->requires_host_coherent || coherent;
  }
}

static iree_status_t iree_hal_amd_xdna_memory_query_transition(
    void* user_data, iree_hal_memory_scope_id_t producer_id,
    iree_hal_memory_scope_id_t consumer_id,
    iree_hal_memory_pair_info_t* out_info) {
  iree_hal_amd_xdna_memory_transition_query_t* query = user_data;
  const iree_hal_amd_xdna_memory_transition_site_t* producer =
      &query->sites[producer_id];
  const iree_hal_amd_xdna_memory_transition_site_t* consumer =
      &query->sites[consumer_id];
  if (producer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_UNKNOWN ||
      consumer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_UNKNOWN) {
    return iree_ok_status();
  }
  if (producer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_HOST &&
      consumer->kind == IREE_HAL_AMD_XDNA_MEMORY_SITE_HOST) {
    *out_info = (iree_hal_memory_pair_info_t){
        .flags = IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
                 IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN,
        .release = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
        .acquire = {.kind = IREE_HAL_MEMORY_TRANSITION_KIND_NONE},
        .atomic_reach = {.scope_32 = IREE_HAL_ATOMIC_REACH_SYSTEM,
                         .scope_64 = IREE_HAL_ATOMIC_REACH_SYSTEM},
    };
    return iree_ok_status();
  }

  query->native.producer = producer->native;
  query->native.consumer = consumer->native;
  amdf_memory_pair_info_t native = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(native),
  };
  const amdf_status_t native_status =
      query->owner->context->api->memory_scope_query_pair_info(
          query->owner->context->data_source.scope, &query->native, &native);
  if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    query->complete = false;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      native_status, "memory_scope_query_pair_info"));
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_xdna_memory_translate_pair(&native, out_info));
  iree_hal_amd_xdna_memory_accumulate_pair(producer, consumer, out_info, query);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_slab_pool_emit_plan(
    const iree_hal_amd_xdna_memory_backend_t* owner,
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options, uint32_t access_count,
    const amdf_memory_device_access_t* accesses,
    const uint32_t* family_access_ordinals,
    const iree_hal_amd_xdna_memory_transition_site_t* transition_sites,
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

  iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
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
      .maintenance_alignment = 1,
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
  bool plan_consumed = false;
  iree_status_t status = iree_hal_memory_contract_create(
      iree_hal_device_group_memory_domain(group),
      iree_hal_device_group_memory_scope_count(group), &binding_layout,
      host_allocator, &plan->contract);
  if (iree_status_is_ok(status)) {
    iree_hal_memory_contract_t* contract = plan->contract;
    contract->host = host;
    contract->placement = plan->base.info.placement;
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
    bool host_coherent = host.access != IREE_HAL_MEMORY_ACCESS_NONE;
    for (uint32_t i = 0; i < access_count; ++i) {
      host_coherent &= iree_all_bits_set(accesses[i].requirements.flags,
                                         AMDF_MEMORY_FLAG_HOST_COHERENT);
    }
    iree_hal_amd_xdna_memory_transition_query_t transition_query = {
        .owner = owner,
        .native =
            {
                .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
                .structure_size = sizeof(amdf_memory_profile_pair_query_t),
                .memory_profile_ordinal = profile->ordinal,
                .access_count = access_count,
                .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
                .accesses = accesses,
            },
        .sites = transition_sites,
        .maintenance_alignment = 1,
        .host_coherent = host_coherent,
        .complete = true,
    };
    status = iree_hal_memory_contract_initialize_transitions(
        contract, iree_hal_amd_xdna_memory_query_transition, &transition_query);
    const bool requirements_satisfied = transition_query.complete;
    if (iree_status_is_ok(status) && requirements_satisfied) {
      if (transition_query.host_coherent) {
        memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
      }
      plan->provider_options.memory_type = memory_type;
      plan->provider_options.maintenance_alignment =
          transition_query.maintenance_alignment;
      contract->buffer_params = (iree_hal_buffer_params_t){
          .usage = usage | mapping_usage,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = memory_type,
          .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      };
      plan_consumed = true;
      status = callback.fn(callback.user_data, &plan->base);
    } else if (iree_status_is_ok(status)) {
      iree_hal_amd_xdna_slab_pool_plan_destroy(&plan->base);
    }
  }
  if (!iree_status_is_ok(status) && !plan_consumed) {
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
  iree_host_size_t transition_sites_offset = 0;
  const uint32_t memory_scope_count =
      iree_hal_device_group_memory_scope_count(group);
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scratch_size,
      IREE_STRUCT_FIELD_ALIGNED(scope.family_count, amdf_memory_device_access_t,
                                iree_alignof(amdf_memory_device_access_t),
                                &accesses_offset),
      IREE_STRUCT_FIELD_ALIGNED(scope.family_count, uint32_t,
                                iree_alignof(uint32_t),
                                &family_access_ordinals_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          scope.family_count, const iree_hal_amd_xdna_memory_backend_t*,
          iree_alignof(const iree_hal_amd_xdna_memory_backend_t*),
          &family_backends_offset),
      IREE_STRUCT_FIELD_ALIGNED(scope.family_count,
                                amdf_memory_access_capabilities_t,
                                iree_alignof(amdf_memory_access_capabilities_t),
                                &capabilities_offset),
      IREE_STRUCT_FIELD_ALIGNED(binding_capacity, uint16_t,
                                iree_alignof(uint16_t), &binding_types_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          memory_scope_count, iree_hal_amd_xdna_memory_transition_site_t,
          iree_alignof(iree_hal_amd_xdna_memory_transition_site_t),
          &transition_sites_offset)));
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
  iree_hal_amd_xdna_memory_transition_site_t* transition_sites =
      (iree_hal_amd_xdna_memory_transition_site_t*)(scratch +
                                                    transition_sites_offset);
  transition_sites[1] = (iree_hal_amd_xdna_memory_transition_site_t){
      .kind = IREE_HAL_AMD_XDNA_MEMORY_SITE_HOST,
      .native =
          {
              .kind = AMDF_MEMORY_SITE_KIND_HOST,
              .value.host_access =
                  AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
          },
  };

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
      transition_sites[scope.families[family_index]
                           .family->memory.queue_scope_id] =
          (iree_hal_amd_xdna_memory_transition_site_t){
              .kind = IREE_HAL_AMD_XDNA_MEMORY_SITE_QUEUE,
              .requires_host_coherent = iree_any_bit_set(
                  scope.families[family_index].requirements,
                  IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST),
              .native =
                  {
                      .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
                      .value.device =
                          {
                              .access_ordinal = access_ordinal,
                              .queue_family_ordinal =
                                  backend->context->queue_family_ordinal,
                          },
                  },
          };
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
          family_access_ordinals, transition_sites, &profile, binding_types,
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
          },
      .device = device,
      .context = context,
      .notification = notification,
      .maintenance = maintenance,
      .epoch_query = epoch_query,
  };
  out_backend->factories[0] = &iree_hal_amd_xdna_slab_pool_factory;
  out_backend->base.factory_count = 1;
  out_backend->base.factories = out_backend->factories;
}
