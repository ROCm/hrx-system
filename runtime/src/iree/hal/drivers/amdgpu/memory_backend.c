// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/memory_backend.h"

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/asan_state.h"
#include "iree/hal/drivers/amdgpu/atomic_memory.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"

typedef struct iree_hal_amdgpu_slab_pool_plan_t {
  // Generic cold construction product.
  iree_hal_slab_pool_plan_t base;
  // Allocator for this temporary plan and its inline agent array.
  iree_allocator_t host_allocator;
  // Borrowed HAL owner for placement and profiling.
  iree_hal_device_t* device;
  // Borrowed native attachment service.
  iree_hal_allocator_t* allocator;
  // Borrowed native allocation dispatch table.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  // Selected native pool, access grants and sanitizer policy.
  iree_hal_amdgpu_slab_provider_options_t provider_options;
  // Owner's physical ordinal used for native profiling.
  iree_host_size_t physical_ordinal;
  // Existing physical-device wrapper storage.
  iree_hal_amdgpu_buffer_pool_t* buffer_pool;
  // Existing placement-local cold progress owners.
  struct {
    // Native owner's capacity notification.
    iree_async_notification_t* notification;
    // Sealed group's completion probe.
    iree_hal_pool_epoch_query_t epoch_query;
    // Sealed group's shared completion tracker.
    iree_async_frontier_tracker_t* tracker;
  } progress;
  // Owned immutable exact-family and public mapping contract.
  iree_hal_memory_contract_t* contract;
  // Device-selected sanitizer allocation policy for this native source.
  iree_hal_asan_pool_options_t asan;
  // Whether the created pool may allocate new HSA slabs.
  iree_hal_slab_pool_owned_mode_t owned_mode;
  // Constructor-borrowed import classes copied by synchronous pool creation.
  struct {
    // Number of admitted input classes.
    iree_host_size_t class_count;
    // Source classes and correlated export requirements.
    const iree_hal_pool_import_class_t* classes;
  } imports;
  // Borrowed until synchronous creation copies the diagnostic name.
  iree_string_view_t trace_name;
  // Physical HSA domain name, copied by the native provider at construction.
  char physical_trace_name[64];
  // Unique native agents selected for this source, copied by the provider.
  hsa_agent_t agents[];
} iree_hal_amdgpu_slab_pool_plan_t;

static void iree_hal_amdgpu_slab_pool_plan_destroy(
    iree_hal_slab_pool_plan_t* base) {
  iree_hal_amdgpu_slab_pool_plan_t* plan =
      (iree_hal_amdgpu_slab_pool_plan_t*)base;
  iree_hal_memory_contract_release(plan->contract);
  iree_allocator_free(plan->host_allocator, plan);
}

static iree_status_t iree_hal_amdgpu_slab_pool_plan_create(
    iree_hal_slab_pool_plan_t* base, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  iree_hal_amdgpu_slab_pool_plan_t* plan =
      (iree_hal_amdgpu_slab_pool_plan_t*)base;
  iree_hal_slab_provider_t* provider = NULL;
  iree_status_t status = iree_hal_amdgpu_slab_provider_create(
      plan->device, plan->allocator, plan->libhsa, plan->provider_options,
      plan->physical_ordinal, plan->buffer_pool,
      iree_make_cstring_view(plan->physical_trace_name),
      iree_hal_device_host_allocator(plan->device), &provider);
  if (iree_status_is_ok(status)) {
    const iree_hal_passthrough_pool_options_t options = {
        .memory_contract = plan->contract,
        .epoch_query = plan->progress.epoch_query,
        .asan = plan->asan,
        .owned_mode = plan->owned_mode,
        .imports =
            {
                .class_count = plan->imports.class_count,
                .classes = plan->imports.classes,
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
    iree_hal_amdgpu_slab_pool_plan_vtable = {
        .destroy = iree_hal_amdgpu_slab_pool_plan_destroy,
        .create = iree_hal_amdgpu_slab_pool_plan_create,
};

// Native pool selection does not expose a host page cacheability guarantee.
static bool iree_hal_amdgpu_memory_supports_host_access(
    iree_hal_memory_type_t memory_type, iree_hal_pool_host_access_t host) {
  return !host.access ||
         (iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
          host.cacheability == IREE_HAL_HOST_CACHEABILITY_UNKNOWN);
}

static iree_hal_buffer_usage_t iree_hal_amdgpu_memory_mapping_usage(
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

static iree_hal_buffer_usage_t iree_hal_amdgpu_pointer_import_supported_usage(
    bool host_visible) {
  const iree_hal_buffer_usage_t sharing_usage =
      IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE |
      IREE_HAL_BUFFER_USAGE_SHARING_CONCURRENT |
      IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE;
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                  IREE_HAL_BUFFER_USAGE_DISPATCH |
                                  sharing_usage;
  if (host_visible) {
    usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
             IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
             IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
             IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
             IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE;
  }
  return usage;
}

static bool iree_hal_amdgpu_pointer_storage_class_is_supported(
    iree_hal_external_storage_class_t storage_class) {
  return (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER ||
          storage_class.type ==
              IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS) &&
         storage_class.provenance[0] == 0 && storage_class.provenance[1] == 0;
}

static bool iree_hal_amdgpu_exports_are_supported(
    iree_hal_pool_export_classes_t exports,
    iree_hal_memory_type_t guaranteed_memory_type) {
  for (iree_host_size_t i = 0; i < exports.class_count; ++i) {
    const iree_hal_external_storage_class_t storage_class = exports.classes[i];
    if (!iree_hal_amdgpu_pointer_storage_class_is_supported(storage_class)) {
      return false;
    }
    if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER &&
        !iree_all_bits_set(guaranteed_memory_type,
                           IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
      return false;
    }
    if (storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS &&
        !iree_all_bits_set(guaranteed_memory_type,
                           IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
      return false;
    }
  }
  return true;
}

static bool iree_hal_amdgpu_resolve_external_route_properties(
    const iree_hal_slab_pool_options_t* options,
    iree_hal_memory_type_t owned_memory_type,
    iree_hal_buffer_usage_t owned_supported_usage,
    iree_hal_memory_type_t* out_memory_type,
    iree_hal_buffer_usage_t* out_supported_usage) {
  iree_hal_memory_type_t common_memory_type = UINT32_MAX;
  iree_hal_buffer_usage_t common_supported_usage = UINT32_MAX;
  if (options->owned.mode == IREE_HAL_SLAB_POOL_OWNED_ENABLED) {
    if (!iree_hal_amdgpu_exports_are_supported(options->owned.exports,
                                               owned_memory_type)) {
      return false;
    }
    common_memory_type = owned_memory_type;
    common_supported_usage = owned_supported_usage;
  }
  for (iree_host_size_t i = 0; i < options->imports.class_count; ++i) {
    const iree_hal_pool_import_class_t* import_class =
        &options->imports.classes[i];
    if (!iree_hal_amdgpu_pointer_storage_class_is_supported(
            import_class->storage_class)) {
      return false;
    }
    const bool host_visible = import_class->storage_class.type ==
                              IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER;
    const iree_hal_memory_type_t route_memory_type =
        host_visible ? IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                           IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
                           IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE
                     : IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    if (!iree_hal_amdgpu_exports_are_supported(import_class->exports,
                                               route_memory_type)) {
      return false;
    }
    common_memory_type &= route_memory_type;
    common_supported_usage &=
        iree_hal_amdgpu_pointer_import_supported_usage(host_visible);
  }
  *out_memory_type = common_memory_type;
  *out_supported_usage = common_supported_usage;
  return true;
}

static void iree_hal_amdgpu_slab_pool_plan_append_agent(
    iree_hal_amdgpu_slab_pool_plan_t* plan, hsa_agent_t agent) {
  for (uint32_t i = 0; i < plan->provider_options.access.agent_count; ++i) {
    if (plan->agents[i].handle == agent.handle) {
      return;
    }
  }
  plan->agents[plan->provider_options.access.agent_count++] = agent;
}

// Complete-scope atomic guarantees do not depend on a requesting queue's
// legacy device-local ordinal. Every materialized view has the same cells.
static iree_hal_amdgpu_atomic_memory_source_masks_t
iree_hal_amdgpu_memory_atomic_source_masks(
    iree_hal_amdgpu_atomic_memory_cell_flags_t cells) {
  return (iree_hal_amdgpu_atomic_memory_source_masks_t){
      .device_scope_32 =
          (cells & IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_32)
              ? UINT64_MAX
              : 0,
      .device_scope_64 =
          (cells & IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_DEVICE_SCOPE_64)
              ? UINT64_MAX
              : 0,
      .system_scope_32 =
          (cells & IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_32)
              ? UINT64_MAX
              : 0,
      .system_scope_64 =
          (cells & IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAG_SYSTEM_SCOPE_64)
              ? UINT64_MAX
              : 0,
  };
}

static uint32_t iree_hal_amdgpu_memory_preference(
    iree_hal_device_group_t* group, uint32_t device_ordinal,
    const iree_hal_amdgpu_physical_device_t* physical_device,
    hsa_amd_memory_pool_location_t location, iree_hal_memory_type_t memory_type,
    iree_hal_pool_scope_t scope, const iree_hal_slab_pool_options_t* options) {
  uint32_t preference = 0;
  if (options->placement.mode == IREE_HAL_POOL_PLACEMENT_PREFERRED) {
    const iree_hal_topology_node_t* node = iree_hal_topology_node_at(
        iree_hal_device_group_topology(group), options->placement.node);
    if ((location == HSA_AMD_MEMORY_POOL_LOCATION_CPU &&
         node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_HOST_NUMA &&
         node->local_ordinal == physical_device->host_numa_node) ||
        (location == HSA_AMD_MEMORY_POOL_LOCATION_GPU &&
         node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
         node->device_ordinal == device_ordinal &&
         node->local_ordinal == physical_device->device_ordinal)) {
      preference += 2048;
    }
  }
  if (options->preferences.host.access &&
      iree_hal_amdgpu_memory_supports_host_access(memory_type,
                                                  options->preferences.host)) {
    preference += 1024;
  }
  if (scope.family_count && location == HSA_AMD_MEMORY_POOL_LOCATION_GPU) {
    preference += 512;
    if (!scope.host.access &&
        !iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      preference += 256;
    }
  }
  if (!scope.family_count && location == HSA_AMD_MEMORY_POOL_LOCATION_CPU) {
    preference += 512;
  }
  return preference;
}

static iree_status_t iree_hal_amdgpu_slab_pool_query_memory(
    const iree_hal_amdgpu_memory_backend_t* backend,
    iree_hal_device_group_t* group, uint32_t device_ordinal,
    iree_hal_amdgpu_physical_device_t* physical_device,
    hsa_amd_memory_pool_t memory_pool, uint32_t allocation_flags,
    bool requires_uncached, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  if (!memory_pool.handle) {
    return iree_ok_status();
  }
  bool mapping_source = false;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    mapping_source |=
        (scope.families[i].interfaces &
         (UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE)) != 0;
  }
  if (mapping_source &&
      (options->owned.mode != IREE_HAL_SLAB_POOL_OWNED_ENABLED ||
       options->imports.class_count || options->owned.exports.class_count ||
       requires_uncached ||
       !iree_hal_allocator_supports_virtual_memory(backend->allocator))) {
    return iree_ok_status();
  }
  iree_hal_amdgpu_vmem_granularity_t mapping_granularity = {0};
  if (mapping_source) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_vmem_query_alloc_granularity(
        backend->libhsa, memory_pool, &mapping_granularity));
    if (mapping_granularity.minimum > UINT32_MAX) {
      return iree_ok_status();
    }
  }
  uint32_t global_flags = 0;
  hsa_amd_memory_pool_location_t location;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(backend->libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flags));
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_get_info(
      IREE_LIBHSA(backend->libhsa), memory_pool,
      HSA_AMD_MEMORY_POOL_INFO_LOCATION, &location));
  if (location != HSA_AMD_MEMORY_POOL_LOCATION_CPU &&
      location != HSA_AMD_MEMORY_POOL_LOCATION_GPU) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HSA reported unknown memory pool location %u",
                            (uint32_t)location);
  }
  iree_hal_amdgpu_slab_provider_memory_pool_properties_t properties;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_slab_provider_query_memory_pool_properties(
          backend->libhsa, memory_pool, &properties));
  const bool owned_host_visible =
      !requires_uncached &&
      iree_any_bit_set(global_flags,
                       HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
      (location == HSA_AMD_MEMORY_POOL_LOCATION_CPU ||
       physical_device->memory_system.svm.direct_host_access);
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  if (location == HSA_AMD_MEMORY_POOL_LOCATION_CPU) {
    memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    if (owned_host_visible) {
      memory_type |= IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    }
  }
  if (owned_host_visible) {
    memory_type |=
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_COHERENT;
  }
  if (requires_uncached) {
    memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                  IREE_HAL_MEMORY_TYPE_DEVICE_UNCACHED;
    properties.supported_usage &=
        ~(IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
          IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
          IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
          IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
          IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE);
  }
  iree_hal_memory_type_t exposed_memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  iree_hal_buffer_usage_t exposed_supported_usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (!iree_hal_amdgpu_resolve_external_route_properties(
          options, memory_type, properties.supported_usage,
          &exposed_memory_type, &exposed_supported_usage) ||
      !iree_hal_amdgpu_memory_supports_host_access(exposed_memory_type,
                                                   scope.host)) {
    return iree_ok_status();
  }
  const bool exposed_host_visible =
      iree_any_bit_set(exposed_memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
  const bool exposed_host_coherent =
      iree_any_bit_set(exposed_memory_type, IREE_HAL_MEMORY_TYPE_HOST_COHERENT);
  const uint32_t interfaces =
      (1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS) |
      (exposed_host_visible ? (1u << IREE_HAL_BUFFER_INTERFACE_HOST) : 0) |
      (mapping_source ? (1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE) : 0);
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    const iree_hal_memory_backend_t* participant =
        iree_hal_device_memory_backend(
            iree_hal_queue_family_device(access->family));
    if ((participant->type == IREE_HAL_MEMORY_BACKEND_HOST &&
         !exposed_host_visible) ||
        (access->interfaces & ~((uint64_t)interfaces)) ||
        (!exposed_host_coherent &&
         iree_any_bit_set(access->requirements,
                          IREE_HAL_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST))) {
      return iree_ok_status();
    }
    usage |= access->usage;
  }
  if (!iree_all_bits_set(exposed_supported_usage, usage)) {
    return iree_ok_status();
  }
  iree_hal_pool_host_access_t host = scope.host;
  if (iree_hal_amdgpu_memory_supports_host_access(exposed_memory_type,
                                                  options->preferences.host)) {
    host.access |= options->preferences.host.access;
    host.modes |= options->preferences.host.modes;
  }
  // Host execution addresses are private native facts, independent of maps.
  const iree_host_size_t host_agent_count =
      options->owned.mode == IREE_HAL_SLAB_POOL_OWNED_ENABLED &&
              owned_host_visible
          ? backend->topology->cpu_agent_count
          : 0;
  const iree_host_size_t agent_capacity = scope.family_count + host_agent_count;
  iree_hal_amdgpu_slab_pool_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*plan) + agent_capacity * sizeof(hsa_agent_t),
      (void**)&plan));
  memset(plan, 0, sizeof(*plan) + agent_capacity * sizeof(hsa_agent_t));
  plan->base.vtable = &iree_hal_amdgpu_slab_pool_plan_vtable;
  plan->base.info.host = host;
  plan->base.info.preference = iree_hal_amdgpu_memory_preference(
      group, device_ordinal, physical_device, location, exposed_memory_type,
      scope, options);
  plan->host_allocator = host_allocator;
  plan->device = backend->device;
  plan->allocator = backend->allocator;
  plan->libhsa = backend->libhsa;
  plan->physical_ordinal = physical_device->device_ordinal;
  if (location == HSA_AMD_MEMORY_POOL_LOCATION_CPU &&
      physical_device->host_numa_node == UINT32_MAX) {
    iree_snprintf(plan->physical_trace_name, sizeof(plan->physical_trace_name),
                  "amdgpu-system");
  } else {
    iree_snprintf(plan->physical_trace_name, sizeof(plan->physical_trace_name),
                  location == HSA_AMD_MEMORY_POOL_LOCATION_CPU
                      ? "amdgpu-numa-%u"
                      : "amdgpu-gpu-%u",
                  location == HSA_AMD_MEMORY_POOL_LOCATION_CPU
                      ? physical_device->host_numa_node
                      : (uint32_t)physical_device->device_ordinal);
  }
  plan->buffer_pool = &physical_device->materialized_buffer_pool;
  plan->provider_options.allocation_queue = physical_device->allocation_queue;
  plan->progress.notification = physical_device->memory_notification;
  plan->progress.epoch_query = backend->epoch_query;
  plan->progress.tracker =
      iree_hal_device_topology_info(backend->device)->frontier.tracker;
  plan->trace_name = options->trace_name;
  plan->provider_options.memory_pool = memory_pool;
  plan->provider_options.owned.memory_type = memory_type;
  plan->provider_options.owned.supported_usage = properties.supported_usage;
  plan->provider_options.owned.allocation_flags = allocation_flags;
  plan->provider_options.exposed.memory_type = exposed_memory_type;
  plan->provider_options.exposed.supported_usage = exposed_supported_usage;
  plan->provider_options.exposed.allocation_alignment =
      options->imports.class_count ? 1 : 0;
  plan->provider_options.access.queue_family_affinity =
      IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  plan->provider_options.access.agents = plan->agents;
  plan->provider_options.asan_state = backend->asan_state;
  plan->asan = physical_device->asan_pool_options;
  plan->owned_mode = options->owned.mode;
  plan->imports.class_count = options->imports.class_count;
  plan->imports.classes = options->imports.classes;
  if (iree_hal_amdgpu_asan_state_is_enabled(backend->asan_state)) {
    plan->provider_options.flags =
        IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_SHADOW;
    if (allocation_flags == HSA_AMD_MEMORY_POOL_STANDARD_FLAG &&
        location == HSA_AMD_MEMORY_POOL_LOCATION_GPU &&
        iree_any_bit_set(global_flags,
                         HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)) {
      plan->provider_options.flags |=
          IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_ASAN_VMM;
    }
  }

  if (mapping_source) {
    plan->provider_options.flags |=
        IREE_HAL_AMDGPU_SLAB_PROVIDER_FLAG_MAPPING_SOURCE;
    plan->provider_options.vmem_memory_type =
        IREE_HAL_AMDGPU_VMEM_MEMORY_TYPE_PINNED_HOST;
  }

  bool supported = true;
  iree_hal_amdgpu_atomic_memory_cell_flags_t cells =
      options->owned.mode == IREE_HAL_SLAB_POOL_OWNED_ENABLED &&
              scope.family_count
          ? IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL
          : 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < scope.family_count && supported && iree_status_is_ok(status); ++i) {
    const iree_hal_queue_family_t* family = scope.families[i].family;
    const iree_hal_memory_backend_t* participant_base =
        iree_hal_device_memory_backend(iree_hal_queue_family_device(family));
    if (participant_base->type == IREE_HAL_MEMORY_BACKEND_HOST) {
      // Native CPU access is qualified for all host agents below. Execution
      // and public host mappings use the same coherent physical storage.
      continue;
    }
    const iree_hal_amdgpu_memory_backend_t* participant =
        (const iree_hal_amdgpu_memory_backend_t*)participant_base;
#if !IREE_HAL_AMDGPU_LIBHSA_STATIC
    // Handles are meaningful only inside the same loaded native runtime.
    if (participant->libhsa->hsa_init != backend->libhsa->hsa_init) {
      supported = false;
      break;
    }
#endif  // !IREE_HAL_AMDGPU_LIBHSA_STATIC
    // Instrumented executables require their own published shadow namespace.
    if (iree_hal_amdgpu_asan_state_is_enabled(participant->asan_state) &&
        participant->asan_state != backend->asan_state) {
      supported = false;
      break;
    }
    const hsa_agent_t agent =
        participant->topology->gpu_agents[family->ordinal];
    if (options->owned.mode == IREE_HAL_SLAB_POOL_OWNED_ENABLED) {
      hsa_amd_memory_pool_access_t access;
      status = iree_hsa_amd_agent_memory_pool_get_info(
          IREE_LIBHSA(backend->libhsa), agent, memory_pool,
          HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
      if (iree_status_is_ok(status)) {
        supported = access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
        iree_hal_amdgpu_atomic_memory_cell_flags_t source_cells = 0;
        status = iree_hal_amdgpu_atomic_memory_query_source_cells(
            backend->libhsa, agent, memory_pool, global_flags, allocation_flags,
            access, &source_cells);
        cells &= source_cells;
      }
    }
    if (iree_status_is_ok(status) && supported) {
      iree_hal_amdgpu_slab_pool_plan_append_agent(plan, agent);
    }
  }
  for (iree_host_size_t i = 0;
       i < host_agent_count && supported && iree_status_is_ok(status); ++i) {
    const hsa_agent_t agent = backend->topology->cpu_agents[i];
    hsa_amd_memory_pool_access_t access;
    status = iree_hsa_amd_agent_memory_pool_get_info(
        IREE_LIBHSA(backend->libhsa), agent, memory_pool,
        HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
    if (iree_status_is_ok(status)) {
      switch (access) {
        case HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:
          supported = false;
          break;
        case HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:
        case HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:
          iree_hal_amdgpu_slab_pool_plan_append_agent(plan, agent);
          break;
        default:
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "HSA reported unknown host pool access %u",
                                    (uint32_t)access);
          break;
      }
    }
  }
  if (iree_status_is_ok(status) && supported) {
    const iree_hal_atomic_operation_capabilities_t atomic_operations =
        options->imports.class_count
            ? (iree_hal_atomic_operation_capabilities_t){0}
            : iree_hal_amdgpu_atomic_memory_expand_capabilities(cells);
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      if (!iree_hal_atomic_operation_capabilities_support(
              atomic_operations, scope.families[i].atomic_operations)) {
        supported = false;
        break;
      }
    }
    plan->provider_options.access.atomic_source_masks =
        iree_hal_amdgpu_memory_atomic_source_masks(cells);
    plan->provider_options.exposed.atomic_operations = atomic_operations;
    if (supported) {
      status = iree_hal_memory_contract_create(
          iree_hal_device_group_memory_domain(group),
          iree_hal_device_group_memory_scope_count(group),
          iree_hal_amdgpu_buffer_binding_layout(), host_allocator,
          &plan->contract);
    }
  }
  if (iree_status_is_ok(status) && supported) {
    iree_hal_memory_contract_t* contract = plan->contract;
    contract->host = host;
    if (mapping_source) {
      contract->mapping_source.type = IREE_HAL_MEMORY_BACKEND_ROCR;
      contract->mapping_source.requirements =
          (iree_hal_pool_mapping_requirements_t){
              .minimum_granularity = mapping_granularity.minimum,
              .recommended_granularity = mapping_granularity.recommended,
              .flags = IREE_HAL_POOL_MAPPING_FLAG_WHOLE_ALLOCATION,
          };
    }
    contract->allowed_access =
        iree_hal_buffer_usage_access(usage) | host.access;
    contract->allowed_usage =
        usage | iree_hal_amdgpu_memory_mapping_usage(host);
    if (host.access) {
      contract->scopes[1].interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_HOST;
      contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_HOST] =
          IREE_HAL_AMDGPU_BUFFER_BINDING_HOST;
    }
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      const uint32_t queue_scope_id =
          scope.families[i].family->memory.queue_scope_id;
      for (uint32_t id = queue_scope_id; id <= queue_scope_id + 1; ++id) {
        contract->scopes[id].usage = scope.families[i].usage;
        contract->scopes[id].interfaces = interfaces;
        contract->scopes[id]
            .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] =
            IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS;
        if (mapping_source) {
          contract->scopes[id]
              .bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE] =
              IREE_HAL_AMDGPU_BUFFER_BINDING_MAPPING_SOURCE;
        }
        if (exposed_host_visible) {
          contract->scopes[id].bindings[IREE_HAL_BUFFER_INTERFACE_HOST] =
              IREE_HAL_AMDGPU_BUFFER_BINDING_HOST;
        }
      }
    }
    status = callback.fn(callback.user_data, &plan->base);
  } else {
    iree_hal_amdgpu_slab_pool_plan_destroy(&plan->base);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_slab_pool_query(
    void* self, iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  (void)self;
  // HSA pool selection is preferential; OS policy may move physical pages.
  // It cannot establish a hard physical-node residency guarantee.
  if (options->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED) {
    return iree_ok_status();
  }
  bool has_native_family = false;
  bool requires_uncached = false;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_queue_family_device(access->family));
    if (!backend || (backend->type != IREE_HAL_MEMORY_BACKEND_ROCR &&
                     backend->type != IREE_HAL_MEMORY_BACKEND_HOST)) {
      return iree_ok_status();
    }
    has_native_family |= backend->type == IREE_HAL_MEMORY_BACKEND_ROCR;
    requires_uncached |= iree_any_bit_set(
        access->requirements, IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED);
  }
  if (scope.family_count && !has_native_family) {
    return iree_ok_status();
  }
  if (requires_uncached &&
      (options->owned.mode != IREE_HAL_SLAB_POOL_OWNED_ENABLED ||
       options->imports.class_count != 0)) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group) &&
                               iree_status_is_ok(status);
       ++i) {
    const iree_hal_memory_backend_t* base = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (!base || base->type != IREE_HAL_MEMORY_BACKEND_ROCR) {
      continue;
    }
    const iree_hal_amdgpu_memory_backend_t* backend =
        (const iree_hal_amdgpu_memory_backend_t*)base;
    for (iree_host_size_t j = 0;
         j < backend->topology->gpu_agent_count && iree_status_is_ok(status);
         ++j) {
      iree_hal_amdgpu_physical_device_t* physical_device =
          backend->physical_devices[j];
      if (requires_uncached) {
        hsa_amd_memory_pool_t memory_pool = {0};
        uint32_t allocation_flags = 0;
        if (iree_hal_amdgpu_allocator_query_uncached_memory_pool(
                backend->allocator, (uint32_t)physical_device->device_ordinal,
                &memory_pool, &allocation_flags)) {
          status = iree_hal_amdgpu_slab_pool_query_memory(
              backend, group, (uint32_t)i, physical_device, memory_pool,
              allocation_flags, /*requires_uncached=*/true, scope, options,
              callback, host_allocator);
        }
        continue;
      }
      const hsa_amd_memory_pool_t pools[] = {
          physical_device->coarse_block_pools.large.memory_pool,
          physical_device->fine_block_pools.large.memory_pool,
          physical_device->host_memory_pools.fine_pool,
      };
      for (iree_host_size_t k = 0;
           k < IREE_ARRAYSIZE(pools) && iree_status_is_ok(status); ++k) {
        status = iree_hal_amdgpu_slab_pool_query_memory(
            backend, group, (uint32_t)i, physical_device, pools[k],
            HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
            /*requires_uncached=*/false, scope, options, callback,
            host_allocator);
      }
    }
  }
  return status;
}

static const iree_hal_slab_pool_factory_t iree_hal_amdgpu_slab_pool_factory = {
    .query = iree_hal_amdgpu_slab_pool_query,
};
static const iree_hal_slab_pool_factory_t* const
    iree_hal_amdgpu_slab_pool_factories[] = {
        &iree_hal_amdgpu_slab_pool_factory,
};

void iree_hal_amdgpu_memory_backend_initialize(
    iree_hal_amdgpu_memory_backend_t* backend) {
  backend->base = (iree_hal_memory_backend_t){
      .type = IREE_HAL_MEMORY_BACKEND_ROCR,
      .factory_count = IREE_ARRAYSIZE(iree_hal_amdgpu_slab_pool_factories),
      .factories = iree_hal_amdgpu_slab_pool_factories,
      .query_virtual_buffer = iree_hal_amdgpu_virtual_buffer_query_capabilities,
      .create_virtual_buffer = iree_hal_amdgpu_virtual_buffer_create,
  };
}
