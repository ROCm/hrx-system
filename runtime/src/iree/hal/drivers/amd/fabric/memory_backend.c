// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/fabric/memory_backend.h"

#include "iree/base/internal/math.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/fabric/memory_transition.h"
#include "iree/hal/drivers/amd/fabric/slab_provider.h"
#include "iree/hal/drivers/amd/status.h"
#include "iree/hal/memory/passthrough_pool.h"

typedef struct iree_hal_amd_fabric_slab_pool_plan_t {
  // Generic cold construction product.
  iree_hal_slab_pool_plan_t base;
  // Allocator for this temporary plan and its inline access array.
  iree_allocator_t host_allocator;
  // Borrowed HAL owner for materialized buffer placement.
  iree_hal_device_t* device;
  // Borrowed core API dominated by the group.
  const amdf_api_t* api;
  // Borrowed diagnostic destination captured from the placement owner.
  iree_hal_device_event_sink_t event_sink;
  // Qualified native allocation policy copied by the created provider.
  iree_hal_amd_fabric_slab_provider_options_t provider_options;
  // Whether this pool may acquire owned backing.
  iree_hal_slab_pool_owned_mode_t owned_mode;
  // Import policies borrowed through synchronous pool construction.
  const iree_hal_pool_import_class_t* import_classes;
  // Existing native-owner progress resources captured during qualification.
  struct {
    // Placement-local capacity notification.
    iree_async_notification_t* notification;
    // Native allocation domain shared by the placement owner's engines.
    iree_hal_amd_fabric_allocation_queue_t* allocation_queue;
    // Sealed group's completion probe.
    iree_hal_pool_epoch_query_t epoch_query;
    // Sealed group's shared completion tracker.
    iree_async_frontier_tracker_t* tracker;
  } progress;
  // Owned immutable access facts retained by the created pool.
  iree_hal_memory_contract_t* contract;
  // Borrowed until synchronous creation copies the diagnostic name.
  iree_string_view_t trace_name;
  // Canonical consumers followed by any private native allocation owner.
  amdf_memory_device_access_t accesses[];
} iree_hal_amd_fabric_slab_pool_plan_t;

static void iree_hal_amd_fabric_slab_pool_plan_destroy(
    iree_hal_slab_pool_plan_t* base) {
  iree_hal_amd_fabric_slab_pool_plan_t* plan =
      (iree_hal_amd_fabric_slab_pool_plan_t*)base;
  iree_hal_memory_contract_release(plan->contract);
  iree_allocator_free(plan->host_allocator, plan);
}

static iree_status_t iree_hal_amd_fabric_slab_pool_plan_create(
    iree_hal_slab_pool_plan_t* base, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  iree_hal_amd_fabric_slab_pool_plan_t* plan =
      (iree_hal_amd_fabric_slab_pool_plan_t*)base;
  iree_hal_slab_provider_t* provider = NULL;
  iree_status_t status = iree_hal_amd_fabric_slab_provider_create(
      plan->device, plan->api, plan->event_sink,
      plan->progress.allocation_queue, &plan->provider_options,
      iree_hal_device_host_allocator(plan->device), &provider);
  if (iree_status_is_ok(status)) {
    const iree_hal_passthrough_pool_options_t options = {
        .memory_contract = plan->contract,
        .epoch_query = plan->progress.epoch_query,
        .owned_mode = plan->owned_mode,
        .imports =
            {
                .class_count = plan->provider_options.import_count,
                .classes = plan->import_classes,
            },
        .trace_name = plan->trace_name,
    };
    status = iree_hal_passthrough_pool_create(
        options, provider, plan->progress.notification, plan->progress.tracker,
        host_allocator, out_pool);
  }
  iree_hal_slab_provider_release(provider);
  return status;
}

static const iree_hal_slab_pool_plan_vtable_t
    iree_hal_amd_fabric_slab_pool_plan_vtable = {
        .destroy = iree_hal_amd_fabric_slab_pool_plan_destroy,
        .create = iree_hal_amd_fabric_slab_pool_plan_create,
};

static amdf_memory_map_flags_t iree_hal_amd_fabric_memory_host_access(
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

static iree_hal_buffer_usage_t iree_hal_amd_fabric_memory_mapping_usage(
    iree_hal_pool_host_access_t host) {
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (iree_any_bit_set(host.modes, IREE_HAL_MAPPING_MODE_SCOPED)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }
  if (iree_any_bit_set(host.modes, IREE_HAL_MAPPING_MODE_PERSISTENT)) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  }
  if (host.access) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
             IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  }
  return usage;
}

static bool iree_hal_amd_fabric_memory_supports_host_access(
    const amdf_memory_profile_t* profile, iree_hal_pool_host_access_t host) {
  if (!host.access) {
    return true;
  }
  const amdf_memory_map_flags_t access =
      iree_hal_amd_fabric_memory_host_access(host.access);
  return host.cacheability == IREE_HAL_HOST_CACHEABILITY_UNKNOWN &&
         iree_all_bits_set(profile->roles, AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
         iree_all_bits_set(profile->supported_flags,
                           AMDF_MEMORY_FLAG_HOST_VISIBLE) &&
         iree_all_bits_set(profile->host_mapping.supported_access, access);
}

static bool iree_hal_amd_fabric_memory_supports_mapping(
    const amdf_memory_profile_t* profile, amdf_memory_map_flags_t host_access) {
  return !host_access ||
         (iree_all_bits_set(profile->roles,
                            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
          iree_all_bits_set(profile->supported_flags,
                            AMDF_MEMORY_FLAG_HOST_VISIBLE) &&
          iree_all_bits_set(profile->host_mapping.supported_access,
                            host_access));
}

static bool iree_hal_amd_fabric_memory_supports_exports(
    const amdf_memory_profile_t* profile,
    iree_hal_pool_export_classes_t exports) {
  for (iree_host_size_t i = 0; i < exports.class_count; ++i) {
    const iree_hal_external_storage_class_t storage_class = exports.classes[i];
    if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
      if (storage_class.provenance[0] || storage_class.provenance[1] ||
          !iree_hal_amd_fabric_memory_supports_mapping(
              profile,
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE)) {
        return false;
      }
    } else if (!iree_all_bits_set(profile->roles,
                                  AMDF_MEMORY_PROFILE_ROLE_EXPORT) ||
               !iree_all_bits_set(profile->supported_flags,
                                  AMDF_MEMORY_FLAG_SHAREABLE) ||
               !iree_hal_amd_fabric_memory_external_support(
                   profile, storage_class,
                   AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT)) {
      return false;
    }
  }
  return true;
}

static bool iree_hal_amd_fabric_memory_exports_host_pointer(
    iree_hal_pool_export_classes_t exports) {
  for (iree_host_size_t i = 0; i < exports.class_count; ++i) {
    if (exports.classes[i].type ==
        IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
      return true;
    }
  }
  return false;
}

static amdf_memory_flags_t iree_hal_amd_fabric_memory_export_flags(
    iree_hal_pool_export_classes_t exports) {
  for (iree_host_size_t i = 0; i < exports.class_count; ++i) {
    if (exports.classes[i].type !=
        IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
      return AMDF_MEMORY_FLAG_SHAREABLE;
    }
  }
  return 0;
}

static bool iree_hal_amd_fabric_memory_supports_import(
    const amdf_memory_profile_t* profile,
    iree_hal_external_storage_class_t storage_class) {
  if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
    return !storage_class.provenance[0] && !storage_class.provenance[1] &&
           iree_all_bits_set(profile->roles, AMDF_MEMORY_PROFILE_ROLE_REGISTER);
  }
  return iree_all_bits_set(profile->roles, AMDF_MEMORY_PROFILE_ROLE_IMPORT) &&
         iree_hal_amd_fabric_memory_external_support(
             profile, storage_class, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT);
}

typedef enum iree_hal_amd_fabric_memory_site_kind_e {
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_UNKNOWN = 0,
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST = 1,
  IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE = 2,
} iree_hal_amd_fabric_memory_site_kind_t;

typedef struct iree_hal_amd_fabric_memory_transition_site_t {
  // Native interpretation of this exact HAL scope.
  iree_hal_amd_fabric_memory_site_kind_t kind;
  // Whether this exact queue requires coherent public host access.
  bool requires_host_coherent;
  // Prospective libamdf site when kind is HOST or QUEUE.
  amdf_memory_profile_site_t native;
} iree_hal_amd_fabric_memory_transition_site_t;

typedef struct iree_hal_amd_fabric_memory_transition_query_t {
  // Borrowed native query owner.
  const iree_hal_amd_fabric_memory_backend_t* owner;
  // Base construction facts completed with exact sites for each query.
  amdf_memory_profile_pair_query_t native;
  // Complete correlated construction routes covered by the pool's table.
  const iree_hal_amd_fabric_slab_provider_options_t* provider_options;
  // Borrowed dense construction-time site interpretations.
  const iree_hal_amd_fabric_memory_transition_site_t* sites;
  // Largest explicit range granularity observed across admitted exact pairs.
  iree_device_size_t maintenance_alignment;
  // Whether every admitted host/device direction requires no host cache action.
  bool host_coherent;
  // Whether every admitted host/queue pair has a complete native route.
  bool complete;
} iree_hal_amd_fabric_memory_transition_query_t;

// Complete consumer requirements shared by all candidate storage scopes. This
// temporary construction state never escapes into a pool or submission path.
typedef struct iree_hal_amd_fabric_slab_pool_query_t {
  // Sealed group owning all native devices and topology coordinates.
  iree_hal_device_group_t* group;
  // Complete family and host requirements borrowed from the caller.
  iree_hal_pool_scope_t scope;
  // Storage acquisition, placement and preference policy.
  const iree_hal_slab_pool_options_t* options;
  // Requested topology node, borrowed during construction; NULL for automatic.
  const iree_hal_topology_node_t* placement_node;
  // Number of unique native consumers in canonical group order.
  uint32_t access_count;
  // Complete native access set, independent of the storage owner.
  const amdf_memory_device_access_t* accesses;
  // Exact HAL owners parallel to the complete native access array.
  iree_hal_device_t* const* access_devices;
  // Prepared binding slots for every requested family.
  const iree_hal_memory_scope_access_t* family_bindings;
  // Private host mapping permissions required by execution implementations.
  amdf_memory_map_flags_t private_host_access;
  // Backing flags required by the complete consumer set.
  amdf_memory_flags_t required_flags;
  // Number of prepared bindings, including the reserved host slot.
  uint16_t binding_count;
  // Native address requests in binding order.
  const iree_hal_amd_fabric_binding_request_t* requests;
  // Prospective native sites used to capture each candidate's transition table.
  iree_hal_amd_fabric_memory_transition_site_t* transition_sites;
  // Scratch output array for native consumer qualification.
  amdf_memory_access_capabilities_t* capabilities;
  // Native binding interface identifiers in binding order.
  const uint16_t* binding_types;
  // Consumes complete candidate plans during synchronous construction.
  iree_hal_slab_pool_plan_callback_t callback;
  // Allocator for temporary construction state and candidate plans.
  iree_allocator_t host_allocator;
} iree_hal_amd_fabric_slab_pool_query_t;

static void iree_hal_amd_fabric_memory_accumulate_pair(
    const iree_hal_amd_fabric_memory_transition_site_t* producer,
    const iree_hal_amd_fabric_memory_transition_site_t* consumer,
    const iree_hal_memory_pair_info_t* pair,
    iree_hal_amd_fabric_memory_transition_query_t* query) {
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
  if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST) {
    query->host_coherent &=
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  }
  if (consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST) {
    query->host_coherent &=
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  }
  if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
      consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE) {
    const bool coherent =
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    query->complete &= !consumer->requires_host_coherent || coherent;
  } else if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE &&
             consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST) {
    const bool coherent =
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
    query->complete &= !producer->requires_host_coherent || coherent;
  } else if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
             consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST &&
             (producer->requires_host_coherent ||
              consumer->requires_host_coherent)) {
    query->complete &=
        pair->release.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE &&
        pair->acquire.kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE;
  }
}

// Joins guarantees across construction routes, not across execution sites.
// A prepared host mapping executes its own cache recipe when routes differ.
// Device recipes must agree on an executor and operation; a range recipe does
// not establish that the same engine supports a global operation.
static bool iree_hal_amd_fabric_memory_join_transition(
    iree_hal_memory_transition_recipe_info_t* result,
    const iree_hal_memory_transition_recipe_info_t* value) {
  if (result->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN ||
      value->kind == IREE_HAL_MEMORY_TRANSITION_KIND_UNKNOWN) {
    return false;
  }
  if (result->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE ||
      value->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
    if (result->kind == IREE_HAL_MEMORY_TRANSITION_KIND_NONE) {
      *result = *value;
    }
    // A no-op route may have a different host cache class. Execute through
    // each mapping instead of applying another route's direct instructions.
    if (result->operation == IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH ||
        result->operation ==
            IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE) {
      result->executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
      memset(&result->host, 0, sizeof(result->host));
    }
    return true;
  }
  if (result->operation != value->operation) {
    return false;
  }
  const bool host =
      result->operation == IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH ||
      result->operation == IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_INVALIDATE;
  if (host) {
    // A contract with both ranged and global native routes promises the
    // requested range. The selected mapping may perform broader maintenance.
    if (result->kind != value->kind) {
      result->kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE;
    }
    if (result->executor != value->executor ||
        result->host.instruction != value->host.instruction ||
        result->host.fence_before != value->host.fence_before ||
        result->host.fence_after != value->host.fence_after) {
      result->executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_API;
      memset(&result->host, 0, sizeof(result->host));
    }
  } else if (result->kind != value->kind ||
             result->executor != value->executor) {
    return false;
  }
  result->range_granularity =
      iree_max(result->range_granularity, value->range_granularity);
  return true;
}

static iree_status_t iree_hal_amd_fabric_memory_query_transition(
    void* user_data, iree_hal_memory_scope_id_t producer_id,
    iree_hal_memory_scope_id_t consumer_id,
    iree_hal_memory_pair_info_t* out_info) {
  iree_hal_amd_fabric_memory_transition_query_t* query = user_data;
  const iree_hal_amd_fabric_memory_transition_site_t* producer =
      &query->sites[producer_id];
  const iree_hal_amd_fabric_memory_transition_site_t* consumer =
      &query->sites[consumer_id];
  if (producer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_UNKNOWN ||
      consumer->kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_UNKNOWN) {
    return iree_ok_status();
  }
  query->native.producer = producer->native;
  query->native.consumer = consumer->native;
  const iree_hal_amd_fabric_slab_provider_options_t* options =
      query->provider_options;
  const iree_host_size_t owned_count =
      iree_all_bits_set(options->memory.profile.roles,
                        AMDF_MEMORY_PROFILE_ROLE_CREATE)
          ? 1
          : 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < owned_count + options->import_count && iree_status_is_ok(status);
       ++i) {
    const amdf_memory_profile_t* profile =
        i < owned_count ? &options->memory.profile
                        : &options->imports[i - owned_count].profile;
    query->native.memory_profile_ordinal = profile->ordinal;
    query->native.required_flags =
        i < owned_count ? options->memory.required_flags
                        : options->imports[i - owned_count].required_flags;
    query->native.registered_host_cacheability =
        profile->registration.registered_host_cacheability;
    query->native.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_NONE;
    query->native.external_memory_provenance =
        (amdf_external_memory_provenance_t){{0, 0}};
    if (i >= owned_count &&
        options->imports[i - owned_count].policy.storage_class.type !=
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER) {
      const iree_hal_external_storage_class_t storage_class =
          options->imports[i - owned_count].policy.storage_class;
      query->native.external_memory_type =
          iree_hal_amd_fabric_memory_external_type(storage_class.type);
      query->native.external_memory_provenance =
          (amdf_external_memory_provenance_t){
              {storage_class.provenance[0], storage_class.provenance[1]}};
    }
    amdf_memory_pair_info_t native = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(native),
    };
    const amdf_status_t native_status =
        query->owner->native.api->memory_scope_query_pair_info(
            options->memory.scope, &query->native, &native);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      query->complete = false;
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                           "memory_scope_query_pair_info");
    iree_hal_memory_pair_info_t info = {0};
    if (iree_status_is_ok(status)) {
      status = iree_hal_amd_fabric_memory_translate_pair(&native, &info);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_amd_fabric_memory_accumulate_pair(producer, consumer, &info,
                                                 query);
      if (!i) {
        *out_info = info;
      } else {
        out_info->flags &= info.flags;
        out_info->atomic_reach.scope_32 = iree_min(
            out_info->atomic_reach.scope_32, info.atomic_reach.scope_32);
        out_info->atomic_reach.scope_64 = iree_min(
            out_info->atomic_reach.scope_64, info.atomic_reach.scope_64);
        out_info->estimated_fixed_cost_nanoseconds =
            iree_max(out_info->estimated_fixed_cost_nanoseconds,
                     info.estimated_fixed_cost_nanoseconds);
        query->complete &= iree_hal_amd_fabric_memory_join_transition(
            &out_info->release, &info.release);
        query->complete &= iree_hal_amd_fabric_memory_join_transition(
            &out_info->acquire, &info.acquire);
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_amd_fabric_slab_pool_emit_plan(
    const iree_hal_amd_fabric_memory_backend_t* owner,
    const iree_hal_amd_fabric_memory_storage_t* storage,
    iree_hal_pool_placement_t placement,
    const iree_hal_amd_fabric_slab_pool_query_t* query,
    const amdf_memory_profile_t* profile,
    const iree_hal_amd_fabric_import_route_t* imports) {
  iree_hal_memory_type_t memory_type = storage->memory_type;
  amdf_memory_placement_t native_placement = {0};
  if (profile && query->placement_node &&
      query->placement_node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
      profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM &&
      iree_any_bit_set(profile->allocation.placement_kinds,
                       UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA)) {
    placement = (iree_hal_pool_placement_t){
        .mode = IREE_HAL_POOL_PLACEMENT_REQUIRED,
        .node = query->placement_node->ordinal,
    };
    native_placement = (amdf_memory_placement_t){
        .kind = AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA,
        .node = query->placement_node->local_ordinal,
    };
    // An explicit OS node can be remote from the native consumers. Profile
    // qualification establishes access and visibility, not device locality.
    memory_type &= ~IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  }
  if (query->options->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED &&
      (placement.mode != IREE_HAL_POOL_PLACEMENT_REQUIRED ||
       placement.node != query->options->placement.node)) {
    return iree_ok_status();
  }
  uint16_t binding_count = query->binding_count;
  if (profile &&
      (!iree_all_bits_set(profile->roles, AMDF_MEMORY_PROFILE_ROLE_CREATE) ||
       !iree_all_bits_set(profile->supported_flags, query->required_flags) ||
       !iree_hal_amd_fabric_memory_supports_exports(
           profile, query->options->owned.exports) ||
       !iree_hal_amd_fabric_memory_supports_mapping(
           profile, query->private_host_access) ||
       !iree_hal_amd_fabric_memory_supports_host_access(profile,
                                                        query->scope.host))) {
    return iree_ok_status();
  }
  amdf_memory_profile_t owned_profile = {0};
  if (profile) {
    owned_profile = *profile;
    if (!iree_hal_amd_fabric_memory_constrain_profile(
            (iree_hal_external_storage_class_t){0},
            query->options->owned.exports, &owned_profile)) {
      return iree_ok_status();
    }
    profile = &owned_profile;
  }

  iree_hal_pool_host_access_t host = query->scope.host;
  const iree_hal_pool_host_access_t preferred =
      query->options->preferences.host;
  const iree_hal_pool_host_access_t combined = {
      .access = host.access | preferred.access,
      .modes = host.modes | preferred.modes,
      .cacheability =
          preferred.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN
              ? preferred.cacheability
              : host.cacheability,
  };
  bool preferred_host_added = false;
  bool supports_preferred =
      !profile ||
      iree_hal_amd_fabric_memory_supports_host_access(profile, combined);
  for (iree_host_size_t i = 0; i < query->options->imports.class_count; ++i) {
    supports_preferred &= iree_hal_amd_fabric_memory_supports_host_access(
        &imports[i].profile, combined);
  }
  if (supports_preferred) {
    host = combined;
    preferred_host_added = host.access != query->scope.host.access ||
                           host.modes != query->scope.host.modes;
  }

  const amdf_memory_map_flags_t host_access =
      query->private_host_access |
      iree_hal_amd_fabric_memory_host_access(host.access);
  const uint16_t binding_offset = host_access ? 0 : 1;
  const uint16_t host_binding_index =
      host_access ? 0 : IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
  binding_count -= binding_offset;
  query->transition_sites[1] =
      (iree_hal_amd_fabric_memory_transition_site_t){0};
  if (host_access) {
    query->transition_sites[1].kind = IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST;
    query->transition_sites[1].native.kind = AMDF_MEMORY_SITE_KIND_HOST;
    query->transition_sites[1].native.value.host_access = host_access;
  }
  memory_type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (host_access) {
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  }
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  for (iree_host_size_t i = 0; i < query->scope.family_count; ++i) {
    usage |= query->scope.families[i].usage;
  }
  const iree_hal_buffer_usage_t mapping_usage =
      iree_hal_amd_fabric_memory_mapping_usage(host);
  bool exports_enabled = query->options->owned.exports.class_count != 0;
  for (iree_host_size_t i = 0; i < query->options->imports.class_count; ++i) {
    exports_enabled |= imports[i].policy.exports.class_count != 0;
  }
  if (exports_enabled) {
    usage |= IREE_HAL_BUFFER_USAGE_SHARING_EXPORT;
  }

  iree_host_size_t plan_size = 0;
  iree_host_size_t access_devices_offset = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t imports_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_fabric_slab_pool_plan_t), &plan_size,
      IREE_STRUCT_FIELD_FAM(query->access_count, amdf_memory_device_access_t),
      IREE_STRUCT_FIELD(query->access_count, iree_hal_device_t*,
                        &access_devices_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          binding_count, iree_hal_amd_fabric_binding_request_t,
          iree_alignof(iree_hal_amd_fabric_binding_request_t),
          &bindings_offset),
      IREE_STRUCT_FIELD(query->options->imports.class_count,
                        iree_hal_amd_fabric_import_route_t, &imports_offset)));
  iree_hal_amd_fabric_slab_pool_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(query->host_allocator, plan_size, (void**)&plan));
  memset(plan, 0, plan_size);
  plan->base.vtable = &iree_hal_amd_fabric_slab_pool_plan_vtable;
  plan->base.info = (iree_hal_slab_pool_plan_info_t){
      .preference =
          (preferred_host_added ? 1024 : 0) +
          (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
               ? 512
               : 0),
      .placement = placement,
      .host = host,
  };
  plan->host_allocator = query->host_allocator;
  plan->device = owner->device;
  plan->api = owner->native.api;
  plan->event_sink = owner->native.event_sink;
  memcpy(plan->accesses, query->accesses,
         query->access_count * sizeof(*query->accesses));
  iree_hal_device_t** access_devices =
      (iree_hal_device_t**)((uint8_t*)plan + access_devices_offset);
  memcpy(access_devices, query->access_devices,
         query->access_count * sizeof(*access_devices));
  iree_hal_amd_fabric_binding_request_t* binding_requests =
      (iree_hal_amd_fabric_binding_request_t*)((uint8_t*)plan +
                                               bindings_offset);
  memcpy(binding_requests, query->requests + binding_offset,
         binding_count * sizeof(*query->requests));
  iree_hal_amd_fabric_import_route_t* import_routes =
      (iree_hal_amd_fabric_import_route_t*)((uint8_t*)plan + imports_offset);
  if (query->options->imports.class_count) {
    memcpy(import_routes, imports,
           query->options->imports.class_count * sizeof(*imports));
    for (iree_host_size_t i = 0; i < query->options->imports.class_count; ++i) {
      import_routes[i].required_flags =
          query->required_flags |
          (host_access ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0) |
          iree_hal_amd_fabric_memory_export_flags(
              import_routes[i].policy.exports);
    }
  }
  plan->provider_options = (iree_hal_amd_fabric_slab_provider_options_t){
      .trace_domain =
          profile && profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM
              ? (native_placement.kind == AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA
                     ? "libamdf-numa"
                     : "libamdf-system")
              : owner->native.trace_device_domain,
      .trace_ordinal =
          profile && profile->memory_class == AMDF_MEMORY_CLASS_SYSTEM
              ? (native_placement.kind == AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA
                     ? native_placement.node
                     : UINT32_MAX)
              : owner->native.trace_device_ordinal,
      .access_devices = access_devices,
      .memory =
          {
              .scope = storage->scope,
              .profile = profile ? *profile : (amdf_memory_profile_t){0},
              .placement = native_placement,
              .access_count = query->access_count,
              .accesses = plan->accesses,
              .required_flags =
                  query->required_flags |
                  (host_access ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0) |
                  iree_hal_amd_fabric_memory_export_flags(
                      query->options->owned.exports),
              .host_access = host_access,
              .exports = query->options->owned.exports,
          },
      .import_count = query->options->imports.class_count,
      .imports = import_routes,
      .binding_count = binding_count,
      .host_binding_index = host_binding_index,
      .bindings = binding_requests,
      .memory_type = memory_type,
      .supported_usage = usage | mapping_usage,
      .maintenance_alignment = 1,
  };
  plan->owned_mode = query->options->owned.mode;
  plan->import_classes = query->options->imports.classes;
  plan->progress.notification = owner->notification;
  plan->progress.allocation_queue = owner->allocation_queue;
  plan->progress.epoch_query = owner->epoch_query;
  plan->progress.tracker =
      iree_hal_device_topology_info(owner->device)->frontier.tracker;
  plan->trace_name = query->options->trace_name;
  const iree_hal_buffer_binding_layout_t binding_layout = {
      .byte_length =
          (uint32_t)(binding_count * sizeof(iree_hal_buffer_native_binding_t)),
      .binding_count = binding_count,
      .host_binding_index = host_binding_index,
      .types = query->binding_types + binding_offset,
  };
  bool plan_consumed = false;
  iree_status_t status = iree_hal_memory_contract_create(
      iree_hal_device_group_memory_domain(query->group),
      iree_hal_device_group_memory_scope_count(query->group), &binding_layout,
      query->host_allocator, &plan->contract);
  if (iree_status_is_ok(status)) {
    iree_hal_memory_contract_t* contract = plan->contract;
    contract->host = host;
    contract->placement = plan->base.info.placement;
    for (uint16_t i = 0; i < binding_count; ++i) {
      if (binding_requests[i].type ==
          IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE) {
        contract->mapping_source.type = IREE_HAL_MEMORY_BACKEND_AMDF;
        contract->mapping_source.requirements.minimum_granularity =
            iree_max(contract->mapping_source.requirements.minimum_granularity,
                     binding_requests[i].mapping_granularity);
        if (host.access &&
            binding_requests[i].address_kind == AMDF_MEMORY_ADDRESS_HOST) {
          contract->scopes[1].interfaces |=
              1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
          contract->scopes[1]
              .bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE] = i;
        }
      }
    }
    contract->mapping_source.requirements.recommended_granularity =
        contract->mapping_source.requirements.minimum_granularity;
    if (host.access) {
      contract->scopes[1].interfaces |= 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
      contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_HOST] = 0;
    }
    for (iree_host_size_t i = 0; i < query->scope.family_count; ++i) {
      const uint32_t queue_scope_id =
          query->scope.families[i].family->memory.queue_scope_id;
      iree_hal_memory_scope_access_t bindings = query->family_bindings[i];
      for (uint32_t type = 0; type < IREE_HAL_BUFFER_INTERFACE_COUNT; ++type) {
        if (iree_any_bit_set(bindings.interfaces, 1u << type)) {
          bindings.bindings[type] -= binding_offset;
        }
      }
      contract->scopes[queue_scope_id] = bindings;
      contract->scopes[queue_scope_id + 1] = bindings;
    }
    bool host_coherent = host.access != IREE_HAL_MEMORY_ACCESS_NONE;
    for (iree_host_size_t i = 0; i < query->scope.family_count; ++i) {
      host_coherent |=
          query
              ->transition_sites[query->scope.families[i]
                                     .family->memory.queue_scope_id]
              .kind == IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST;
    }
    iree_hal_amd_fabric_memory_transition_query_t transition_query = {
        .owner = owner,
        .native =
            {
                .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
                .structure_size = sizeof(amdf_memory_profile_pair_query_t),
                .access_count = query->access_count,
                .required_flags =
                    host_access ? AMDF_MEMORY_FLAG_HOST_VISIBLE : 0,
                .accesses = query->accesses,
            },
        .provider_options = &plan->provider_options,
        .sites = query->transition_sites,
        .maintenance_alignment = 1,
        .host_coherent = host_coherent,
        .complete = true,
    };
    status = iree_hal_memory_contract_initialize_transitions(
        contract, iree_hal_amd_fabric_memory_query_transition,
        &transition_query);
    // Private CPU transfers need the same cache guarantees as public host
    // access, without granting callers a host mapping or adding a public scope.
    for (iree_host_size_t i = 0;
         (query->private_host_access &
          ~iree_hal_amd_fabric_memory_host_access(host.access)) &&
         i < query->scope.family_count && iree_status_is_ok(status);
         ++i) {
      const uint32_t queue_scope_id =
          query->scope.families[i].family->memory.queue_scope_id;
      const iree_hal_memory_access_t access =
          iree_hal_buffer_usage_access(query->scope.families[i].usage);
      iree_hal_memory_pair_info_t info = {0};
      if (iree_any_bit_set(query->private_host_access,
                           AMDF_MEMORY_MAP_FLAG_WRITE) &&
          iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
        status = iree_hal_amd_fabric_memory_query_transition(
            &transition_query, 1, queue_scope_id, &info);
      }
      if (iree_status_is_ok(status) &&
          iree_any_bit_set(query->private_host_access,
                           AMDF_MEMORY_MAP_FLAG_READ) &&
          iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        status = iree_hal_amd_fabric_memory_query_transition(
            &transition_query, queue_scope_id, 1, &info);
      }
    }
    const bool requirements_satisfied = transition_query.complete;
    if (iree_status_is_ok(status) && requirements_satisfied) {
      if (transition_query.host_coherent) {
        memory_type |= IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
      }
      plan->provider_options.memory_type = memory_type;
      plan->provider_options.maintenance_alignment =
          transition_query.maintenance_alignment;
      contract->allowed_access =
          iree_hal_buffer_usage_access(usage) | host.access;
      contract->allowed_usage = usage | mapping_usage;
      plan_consumed = true;
      status = query->callback.fn(query->callback.user_data, &plan->base);
    } else if (iree_status_is_ok(status)) {
      iree_hal_amd_fabric_slab_pool_plan_destroy(&plan->base);
    }
  }
  if (!iree_status_is_ok(status) && !plan_consumed) {
    iree_hal_amd_fabric_slab_pool_plan_destroy(&plan->base);
  }
  return status;
}

static iree_status_t iree_hal_amd_fabric_slab_pool_query_storage(
    const iree_hal_amd_fabric_memory_backend_t* owner,
    const iree_hal_amd_fabric_memory_storage_t* storage,
    iree_hal_pool_placement_t placement,
    const iree_hal_amd_fabric_slab_pool_query_t* query) {
  amdf_memory_scope_info_t scope_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope_info),
  };
  IREE_RETURN_IF_ERROR(IREE_HAL_AMD_STATUS_FROM_AMDF(
      owner->native.api->memory_scope_query_info(storage->scope, &scope_info),
      "memory_scope_query_info"));
  iree_host_size_t routes_size = 0;
  iree_host_size_t profiles_offset = 0;
  iree_host_size_t imports_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &routes_size,
      IREE_STRUCT_FIELD(scope_info.memory_profile_count, amdf_memory_profile_t,
                        &profiles_offset),
      IREE_STRUCT_FIELD(query->options->imports.class_count,
                        iree_hal_amd_fabric_import_route_t, &imports_offset)));
  uint8_t* routes = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(query->host_allocator, routes_size,
                                             (void**)&routes));
  amdf_memory_profile_t* profiles =
      routes ? (amdf_memory_profile_t*)(routes + profiles_offset) : NULL;
  iree_hal_amd_fabric_import_route_t* imports =
      routes ? (iree_hal_amd_fabric_import_route_t*)(routes + imports_offset)
             : NULL;
  bool supported = true;
  iree_status_t status = iree_ok_status();
  for (uint32_t profile_ordinal = 0;
       profile_ordinal < scope_info.memory_profile_count &&
       iree_status_is_ok(status);
       ++profile_ordinal) {
    amdf_memory_profile_t* profile = &profiles[profile_ordinal];
    *profile = (amdf_memory_profile_t){
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(*profile),
    };
    for (uint32_t i = 0; i < query->access_count; ++i) {
      query->capabilities[i] = (amdf_memory_access_capabilities_t){
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(query->capabilities[i]),
      };
    }
    const amdf_status_t native_status =
        owner->native.api->memory_scope_query_device_profile(
            storage->scope, profile_ordinal, query->access_count,
            query->accesses, profile, query->capabilities);
    if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(native_status,
                                           "memory_scope_query_device_profile");
  }
  for (iree_host_size_t i = 0; i < query->options->imports.class_count &&
                               supported && iree_status_is_ok(status);
       ++i) {
    const iree_hal_pool_import_class_t* policy =
        &query->options->imports.classes[i];
    const iree_hal_external_storage_class_t storage_class =
        policy->storage_class;
    supported = false;
    for (uint32_t p = 0; p < scope_info.memory_profile_count; ++p) {
      const amdf_memory_profile_t* profile = &profiles[p];
      if (iree_all_bits_set(profile->supported_flags, query->required_flags) &&
          iree_hal_amd_fabric_memory_supports_import(profile, storage_class) &&
          iree_hal_amd_fabric_memory_supports_exports(profile,
                                                      policy->exports) &&
          iree_hal_amd_fabric_memory_supports_mapping(
              profile, query->private_host_access) &&
          iree_hal_amd_fabric_memory_supports_host_access(profile,
                                                          query->scope.host)) {
        imports[i] = (iree_hal_amd_fabric_import_route_t){
            .policy = *policy,
            .profile = *profile,
        };
        supported = iree_hal_amd_fabric_memory_constrain_profile(
            storage_class, policy->exports, &imports[i].profile);
        if (supported) {
          break;
        }
      }
    }
  }
  if (supported && iree_status_is_ok(status)) {
    if (query->options->owned.mode == IREE_HAL_SLAB_POOL_IMPORT_ONLY) {
      status = iree_hal_amd_fabric_slab_pool_emit_plan(
          owner, storage, placement, query, NULL, imports);
    } else {
      for (uint32_t p = 0;
           p < scope_info.memory_profile_count && iree_status_is_ok(status);
           ++p) {
        status = iree_hal_amd_fabric_slab_pool_emit_plan(
            owner, storage, placement, query, &profiles[p], imports);
      }
    }
  }
  iree_allocator_free(query->host_allocator, routes);
  return status;
}

// Captures enforced native physical affinity. Host NUMA placement is qualified
// separately against each allocation profile, independently of this affinity.
static iree_hal_pool_placement_t iree_hal_amd_fabric_memory_storage_placement(
    const iree_hal_amd_fabric_memory_backend_t* owner,
    const iree_hal_amd_fabric_memory_storage_t* storage,
    iree_hal_device_group_t* group) {
  if (storage->physical_device_affinity) {
    const uint32_t device_ordinal =
        iree_hal_device_topology_info(owner->device)->topology_index;
    const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
    for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
         ++i) {
      const iree_hal_topology_node_t* node =
          iree_hal_topology_node_at(topology, i);
      if (node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
          node->device_ordinal == device_ordinal &&
          node->physical_device_affinity == storage->physical_device_affinity) {
        return (iree_hal_pool_placement_t){
            .mode = IREE_HAL_POOL_PLACEMENT_REQUIRED,
            .node = node->ordinal,
        };
      }
    }
  }
  return (iree_hal_pool_placement_t){0};
}

static iree_status_t iree_hal_amd_fabric_slab_pool_query(
    void* self, iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  (void)self;
  if ((scope.host.access &&
       scope.host.cacheability != IREE_HAL_HOST_CACHEABILITY_UNKNOWN) ||
      scope.family_count >= UINT32_MAX) {
    return iree_ok_status();
  }

  const uint32_t memory_scope_count =
      iree_hal_device_group_memory_scope_count(group);
  iree_host_size_t binding_capacity = 0;
  if (!iree_host_size_checked_mul(scope.family_count,
                                  IREE_HAL_BUFFER_INTERFACE_COUNT,
                                  &binding_capacity) ||
      !iree_host_size_checked_add(binding_capacity, 2, &binding_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMD memory scope is too large");
  }
  iree_host_size_t scratch_size = 0;
  iree_host_size_t accesses_offset = 0, family_bindings_offset = 0;
  iree_host_size_t access_devices_offset = 0;
  iree_host_size_t backends_offset = 0;
  iree_host_size_t capabilities_offset = 0, binding_types_offset = 0;
  iree_host_size_t requests_offset = 0, transition_sites_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &scratch_size,
      IREE_STRUCT_FIELD(iree_hal_device_group_device_count(group),
                        const iree_hal_amd_fabric_memory_backend_t*,
                        &backends_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          scope.family_count + 1, amdf_memory_device_access_t,
          iree_alignof(amdf_memory_device_access_t), &accesses_offset),
      IREE_STRUCT_FIELD(scope.family_count + 1, iree_hal_device_t*,
                        &access_devices_offset),
      IREE_STRUCT_FIELD_ALIGNED(scope.family_count,
                                iree_hal_memory_scope_access_t,
                                iree_alignof(iree_hal_memory_scope_access_t),
                                &family_bindings_offset),
      IREE_STRUCT_FIELD_ALIGNED(scope.family_count + 1,
                                amdf_memory_access_capabilities_t,
                                iree_alignof(amdf_memory_access_capabilities_t),
                                &capabilities_offset),
      IREE_STRUCT_FIELD_ALIGNED(binding_capacity, uint16_t,
                                iree_alignof(uint16_t), &binding_types_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          binding_capacity, iree_hal_amd_fabric_binding_request_t,
          iree_alignof(iree_hal_amd_fabric_binding_request_t),
          &requests_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          memory_scope_count, iree_hal_amd_fabric_memory_transition_site_t,
          iree_alignof(iree_hal_amd_fabric_memory_transition_site_t),
          &transition_sites_offset)));
  uint8_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, scratch_size, (void**)&scratch));
  const iree_hal_amd_fabric_memory_backend_t** backends =
      (const iree_hal_amd_fabric_memory_backend_t**)(scratch + backends_offset);
  amdf_memory_device_access_t* accesses =
      (amdf_memory_device_access_t*)(scratch + accesses_offset);
  iree_hal_device_t** access_devices =
      (iree_hal_device_t**)(scratch + access_devices_offset);
  iree_hal_memory_scope_access_t* family_bindings =
      (iree_hal_memory_scope_access_t*)(scratch + family_bindings_offset);
  amdf_memory_access_capabilities_t* capabilities =
      (amdf_memory_access_capabilities_t*)(scratch + capabilities_offset);
  uint16_t* binding_types = (uint16_t*)(scratch + binding_types_offset);
  iree_hal_amd_fabric_binding_request_t* requests =
      (iree_hal_amd_fabric_binding_request_t*)(scratch + requests_offset);
  iree_hal_amd_fabric_memory_transition_site_t* transition_sites =
      (iree_hal_amd_fabric_memory_transition_site_t*)(scratch +
                                                      transition_sites_offset);
  binding_types[0] = IREE_HAL_BUFFER_INTERFACE_HOST;
  uint32_t binding_count = 1;
  bool supported = true;
  iree_host_size_t matched_family_count = 0;
  uint32_t access_count = 0;
  uint32_t host_source_index = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
  amdf_memory_flags_t required_flags = 0;
  amdf_memory_map_flags_t private_host_access = 0;
  bool requires_host_export =
      iree_hal_amd_fabric_memory_exports_host_pointer(options->owned.exports);
  for (iree_host_size_t i = 0; i < options->imports.class_count; ++i) {
    requires_host_export |= iree_hal_amd_fabric_memory_exports_host_pointer(
        options->imports.classes[i].exports);
  }
  if (requires_host_export) {
    private_host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  }
  const iree_hal_amd_fabric_memory_backend_t* owner = NULL;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t device_index = 0;
       device_index < iree_hal_device_group_device_count(group) && supported;
       ++device_index) {
    iree_hal_device_t* device =
        iree_hal_device_group_device_at(group, device_index);
    for (iree_host_size_t family_index = 0;
         family_index < scope.family_count && supported; ++family_index) {
      const iree_hal_pool_family_access_t* requested =
          &scope.families[family_index];
      if (iree_hal_queue_family_device(requested->family) != device) {
        continue;
      }
      ++matched_family_count;
      const iree_hal_memory_backend_t* base =
          iree_hal_device_memory_backend(device);
      // Published native buffers carry no atomic operation capabilities. A
      // construction request must not promise more than those buffers expose.
      const iree_hal_atomic_operation_capabilities_t atomic_operations = {0};
      if (!base ||
          !iree_hal_atomic_operation_capabilities_support(
              atomic_operations, requested->atomic_operations) ||
          iree_any_bit_set(requested->requirements,
                           IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED)) {
        supported = false;
        break;
      }
      iree_hal_memory_scope_access_t* bindings = &family_bindings[family_index];
      bindings->usage = requested->usage;
      for (uint32_t type = 0; type < IREE_HAL_BUFFER_INTERFACE_COUNT; ++type) {
        bindings->bindings[type] = IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
      }
      if (base->type == IREE_HAL_MEMORY_BACKEND_HOST) {
        if (!iree_all_bits_set(
                iree_hal_queue_family_spec(requested->family)->flags,
                IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_HOST_MEMORY_BARRIERS) ||
            (requested->interfaces &
             ~((UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_HOST) |
               (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE)))) {
          supported = false;
          break;
        }
        const amdf_memory_map_flags_t access =
            iree_hal_amd_fabric_memory_host_access(
                iree_hal_buffer_usage_access(requested->usage));
        if (!access) {
          supported = false;
          break;
        }
        private_host_access |= access;
        bindings->interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
        bindings->bindings[IREE_HAL_BUFFER_INTERFACE_HOST] = 0;
        if (requested->interfaces &
            (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE)) {
          required_flags |= AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE;
          if (host_source_index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
            host_source_index = binding_count++;
            requests[host_source_index] =
                (iree_hal_amd_fabric_binding_request_t){
                    .type = IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
                    .address_kind = AMDF_MEMORY_ADDRESS_HOST,
                };
            binding_types[host_source_index] =
                IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
          }
          bindings->interfaces |= 1u
                                  << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
          bindings->bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE] =
              (uint16_t)host_source_index;
        }
        const uint32_t queue_scope_id =
            requested->family->memory.queue_scope_id;
        transition_sites[queue_scope_id] =
            (iree_hal_amd_fabric_memory_transition_site_t){
                .kind = IREE_HAL_AMD_FABRIC_MEMORY_SITE_HOST,
                .requires_host_coherent = iree_any_bit_set(
                    requested->requirements,
                    IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST),
                .native = {.kind = AMDF_MEMORY_SITE_KIND_HOST,
                           .value.host_access = access},
            };
        transition_sites[queue_scope_id + 1] = transition_sites[queue_scope_id];
        continue;
      }
      if (base->type != IREE_HAL_MEMORY_BACKEND_AMDF || !base->factory_count ||
          !base->factories ||
          base->factories[0]->query != iree_hal_amd_fabric_slab_pool_query) {
        supported = false;
        break;
      }
      const iree_hal_amd_fabric_memory_backend_t* backend =
          (const iree_hal_amd_fabric_memory_backend_t*)base;
      if (!owner) {
        owner = backend;
      } else if (backend->native.api != owner->native.api ||
                 backend->native.instance != owner->native.instance) {
        supported = false;
        break;
      }
      const iree_hal_amd_fabric_memory_family_t* family =
          &backend->native
               .families[iree_hal_queue_family_ordinal(requested->family)];
      if (iree_any_bit_set(requested->usage, ~family->supported_usage)) {
        supported = false;
        break;
      }
      private_host_access |= iree_hal_amd_fabric_memory_host_access(
          iree_hal_buffer_usage_access(requested->usage & family->host_usage));
      uint32_t access_ordinal = 0;
      for (; access_ordinal < access_count; ++access_ordinal) {
        if (accesses[access_ordinal].device == backend->native.device) {
          break;
        }
      }
      if (access_ordinal == access_count) {
        backends[access_count] = backend;
        access_devices[access_count] = device;
        accesses[access_count++].device = backend->native.device;
      }
      amdf_memory_access_requirements_t* requirements =
          &accesses[access_ordinal].requirements;
      requirements->access |= family->minimum_access;
      const iree_hal_memory_access_t access =
          iree_hal_buffer_usage_access(requested->usage);
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
        requirements->access |= AMDF_MEMORY_ACCESS_READ;
      }
      if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        requirements->access |= AMDF_MEMORY_ACCESS_WRITE;
      }
      if (iree_any_bit_set(requested->requirements,
                           IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST)) {
        requirements->flags |= AMDF_MEMORY_FLAG_HOST_COHERENT;
      }
      for (uint32_t i = 0; i < family->binding_count; ++i) {
        const iree_hal_amd_fabric_memory_binding_t binding =
            family->bindings[i];
        requirements->flags |= AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
        requirements->address_kinds |= UINT64_C(1) << binding.address_kind;
        uint32_t index = 1;
        for (; index < binding_count; ++index) {
          if (requests[index].access_ordinal == access_ordinal &&
              requests[index].address_kind == binding.address_kind &&
              binding_types[index] == binding.type) {
            break;
          }
        }
        if (index == binding_count) {
          requests[index] = (iree_hal_amd_fabric_binding_request_t){
              .type = binding.type,
              .access_ordinal = access_ordinal,
              .address_kind = binding.address_kind};
          binding_types[index] = binding.type;
          ++binding_count;
        }
        bindings->interfaces |= 1u << binding.type;
        bindings->bindings[binding.type] = (uint16_t)index;
      }
      if (requested->interfaces &
          (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE)) {
        amdf_memory_reservation_capabilities_t capabilities = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
            .structure_size = sizeof(capabilities),
        };
        const amdf_status_t native_status =
            backend->native.api->device_query_memory_reservation_capabilities(
                backend->native.device, backend->native.address_domain_ordinal,
                &capabilities);
        if (native_status ==
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          supported = false;
          break;
        }
        if (!amdf_status_is_ok(native_status)) {
          status = IREE_HAL_AMD_STATUS_FROM_AMDF(
              native_status, "device_query_memory_reservation_capabilities");
          supported = false;
          break;
        }
        if (capabilities.mapping_granularity > UINT32_MAX) {
          supported = false;
          break;
        }
        requirements->flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
        requirements->access |= capabilities.required_access;
        required_flags |= capabilities.required_source_flags;
        uint32_t index = 1;
        for (; index < binding_count; ++index) {
          if (requests[index].access_ordinal == access_ordinal &&
              requests[index].address_kind != AMDF_MEMORY_ADDRESS_HOST &&
              binding_types[index] ==
                  IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE) {
            break;
          }
        }
        if (index == binding_count) {
          requests[index] = (iree_hal_amd_fabric_binding_request_t){
              .type = IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
              .access_ordinal = access_ordinal,
              .mapping_granularity = (uint32_t)capabilities.mapping_granularity,
          };
          binding_types[index] = IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
          ++binding_count;
        }
        bindings->interfaces |= 1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
        bindings->bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE] =
            (uint16_t)index;
      }
      if (requested->interfaces & ~((uint64_t)bindings->interfaces)) {
        supported = false;
        break;
      }
      transition_sites[requested->family->memory.queue_scope_id] =
          (iree_hal_amd_fabric_memory_transition_site_t){
              .kind = IREE_HAL_AMD_FABRIC_MEMORY_SITE_QUEUE,
              .requires_host_coherent = iree_any_bit_set(
                  requested->requirements,
                  IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST),
              .native = {.kind = AMDF_MEMORY_SITE_KIND_DEVICE,
                         .value.device = {.access_ordinal = access_ordinal,
                                          .queue_family_ordinal =
                                              family->queue_family_ordinal}},
          };
    }
  }
  supported &= matched_family_count == scope.family_count &&
               binding_count < IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE;
  if (!supported) {
    iree_allocator_free(host_allocator, scratch);
    return status;
  }

  const iree_hal_amd_fabric_slab_pool_query_t query = {
      .group = group,
      .scope = scope,
      .options = options,
      .placement_node =
          options->placement.mode == IREE_HAL_POOL_PLACEMENT_AUTOMATIC
              ? NULL
              : iree_hal_topology_node_at(iree_hal_device_group_topology(group),
                                          options->placement.node),
      .access_count = access_count,
      .accesses = accesses,
      .access_devices = access_devices,
      .family_bindings = family_bindings,
      .private_host_access = private_host_access,
      .required_flags = required_flags,
      .binding_count = (uint16_t)binding_count,
      .requests = requests,
      .transition_sites = transition_sites,
      .capabilities = capabilities,
      .binding_types = binding_types,
      .callback = callback,
      .host_allocator = host_allocator,
  };
  // Keep consumer-owned storage first for automatic placement. Other devices
  // in the sealed group may own reachable storage without exposing a family in
  // this pool. Host-only consumers can qualify against any native instance.
  iree_host_size_t backend_count = access_count;
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group);
       ++i) {
    const iree_hal_memory_backend_t* base = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (!base || base->type != IREE_HAL_MEMORY_BACKEND_AMDF ||
        !base->factory_count || !base->factories ||
        base->factories[0]->query != iree_hal_amd_fabric_slab_pool_query) {
      continue;
    }
    const iree_hal_amd_fabric_memory_backend_t* backend =
        (const iree_hal_amd_fabric_memory_backend_t*)base;
    if (owner && (backend->native.api != owner->native.api ||
                  backend->native.instance != owner->native.instance)) {
      continue;
    }
    bool captured = false;
    for (iree_host_size_t j = 0; j < backend_count; ++j) {
      captured |= backends[j] == backend;
    }
    if (!captured) {
      backends[backend_count++] = backend;
    }
  }
  amdf_memory_access_t private_owner_access = 0;
  for (uint32_t i = 0; i < access_count; ++i) {
    private_owner_access |= accesses[i].requirements.access;
  }
  // Native-backed CPU-only routes need owner permissions for their host view.
  if (!access_count &&
      (iree_any_bit_set(private_host_access, AMDF_MEMORY_MAP_FLAG_READ) ||
       iree_any_bit_set(scope.host.access, IREE_HAL_MEMORY_ACCESS_READ))) {
    private_owner_access |= AMDF_MEMORY_ACCESS_READ;
  }
  if (!access_count &&
      (iree_any_bit_set(private_host_access, AMDF_MEMORY_MAP_FLAG_WRITE) ||
       iree_any_bit_set(scope.host.access, IREE_HAL_MEMORY_ACCESS_WRITE))) {
    private_owner_access |= AMDF_MEMORY_ACCESS_WRITE;
  }
  for (iree_host_size_t i = 0; i < backend_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_amd_fabric_memory_backend_t* backend = backends[i];
    if (host_source_index != IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
      amdf_memory_reservation_capabilities_t host_capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
          .structure_size = sizeof(host_capabilities),
      };
      const amdf_status_t native_status =
          backend->native.api
              ->instance_query_host_memory_reservation_capabilities(
                  backend->native.instance, &host_capabilities);
      if (native_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      status = IREE_HAL_AMD_STATUS_FROM_AMDF(
          native_status, "instance_query_host_memory_reservation_capabilities");
      if (!iree_status_is_ok(status) ||
          host_capabilities.mapping_granularity > UINT32_MAX) {
        continue;
      }
      requests[host_source_index].mapping_granularity =
          (uint32_t)host_capabilities.mapping_granularity;
    }
    for (iree_host_size_t j = 0;
         j < backend->native.storage_count && iree_status_is_ok(status); ++j) {
      iree_hal_amd_fabric_memory_storage_t storage = backend->native.storage[j];
      // Shared SYSTEM scopes have one plan regardless of how many engines
      // publish them. Its locality guarantees must hold for every consumer.
      bool queried = false;
      if (!storage.physical_device_affinity) {
        for (iree_host_size_t k = 0; k < backend_count; ++k) {
          bool locality_known = false;
          for (iree_host_size_t m = 0; m < backends[k]->native.storage_count;
               ++m) {
            const iree_hal_amd_fabric_memory_storage_t* other =
                &backends[k]->native.storage[m];
            if (other->scope == storage.scope) {
              locality_known = true;
              if (k < access_count) {
                storage.memory_type &= other->memory_type;
              }
              queried |= k < i || (k == i && m < j);
            }
          }
          // Native profile qualification proves access, not locality. A
          // consumer that publishes no facts for this scope cannot establish
          // that shared system memory is local to its device.
          if (k < access_count && !locality_known) {
            storage.memory_type &= ~IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
          }
        }
      }
      if (queried) {
        continue;
      }
      const iree_hal_pool_placement_t placement =
          iree_hal_amd_fabric_memory_storage_placement(backend, &storage,
                                                       group);
      if (options->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED &&
          !(query.placement_node->kind ==
                IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
            !storage.physical_device_affinity) &&
          (placement.mode != IREE_HAL_POOL_PLACEMENT_REQUIRED ||
           placement.node != options->placement.node)) {
        continue;
      }
      if (!storage.physical_device_affinity && !access_count) {
        // Platform-owned host pages need no native device attachment. Keep a
        // separate device-owned candidate for native exports and imports that
        // the zero-consumer route cannot provide.
        iree_hal_amd_fabric_memory_storage_t host_storage = storage;
        host_storage.memory_type &= ~IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
        status = iree_hal_amd_fabric_slab_pool_query_storage(
            backend, &host_storage, placement, &query);
      }
      iree_hal_amd_fabric_slab_pool_query_t storage_query = query;
      const iree_hal_amd_fabric_memory_backend_t* allocation_owner = backend;
      if (storage.physical_device_affinity || !access_count) {
        uint32_t owner_ordinal = 0;
        for (; owner_ordinal < access_count; ++owner_ordinal) {
          if (accesses[owner_ordinal].device == backend->native.device) {
            break;
          }
        }
        if (owner_ordinal == access_count) {
          // LOCAL construction needs a live owner attachment. Its native
          // permissions are qualified with the consumers, but no public family
          // or binding is added. Appending preserves captured access ordinals.
          accesses[access_count] = (amdf_memory_device_access_t){
              .device = backend->native.device,
              .requirements =
                  {
                      .access = private_owner_access | storage.minimum_access,
                  },
          };
          access_devices[access_count] = backend->device;
          ++storage_query.access_count;
        }
      } else {
        // SYSTEM construction chooses from the actual native consumers.
        allocation_owner = backends[0];
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_amd_fabric_slab_pool_query_storage(
            allocation_owner, &storage, placement, &storage_query);
      }
    }
  }
  iree_allocator_free(host_allocator, scratch);
  return status;
}

static const iree_hal_slab_pool_factory_t
    iree_hal_amd_fabric_slab_pool_factory = {
        .query = iree_hal_amd_fabric_slab_pool_query,
};

void iree_hal_amd_fabric_memory_backend_initialize(
    iree_hal_device_t* device,
    const iree_hal_amd_fabric_memory_backend_options_t* options,
    iree_async_notification_t* notification,
    iree_hal_amd_fabric_allocation_queue_t* allocation_queue,
    iree_hal_pool_epoch_query_t epoch_query,
    iree_hal_amd_fabric_memory_backend_t* out_backend) {
  static const iree_hal_slab_pool_factory_t* const factories[] = {
      &iree_hal_amd_fabric_slab_pool_factory,
  };
  *out_backend = (iree_hal_amd_fabric_memory_backend_t){
      .base =
          {
              .type = IREE_HAL_MEMORY_BACKEND_AMDF,
              .factory_count = IREE_ARRAYSIZE(factories),
              .factories = factories,
              .query_virtual_buffer =
                  iree_hal_amd_fabric_virtual_buffer_query_capabilities,
              .create_virtual_buffer =
                  iree_hal_amd_fabric_virtual_buffer_create,
          },
      .device = device,
      .native = *options,
      .notification = notification,
      .allocation_queue = allocation_queue,
      .epoch_query = epoch_query,
  };
}
