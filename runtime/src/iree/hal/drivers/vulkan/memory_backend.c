// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/memory_backend.h"

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/vulkan/atomic.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/slab_provider.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/memory/passthrough_pool.h"

typedef struct iree_hal_vulkan_slab_pool_plan_t {
  // Generic cold construction product.
  iree_hal_slab_pool_plan_t base;
  // Allocator for temporary construction metadata.
  iree_allocator_t host_allocator;
  // Borrowed native allocation services; no topology is retained.
  iree_hal_vulkan_allocator_t* allocator;
  // Qualified native allocation policy for one memory type and family set.
  iree_hal_vulkan_slab_provider_options_t provider_options;
  // Borrowed native-owner progress resources captured during qualification.
  struct {
    // Existing placement-local capacity notification.
    iree_async_notification_t* notification;
    // Existing completion probe with a group-lifetime borrowed context.
    iree_hal_pool_epoch_query_t epoch_query;
    // Sealed group's shared completion tracker.
    iree_async_frontier_tracker_t* tracker;
  } progress;
  // Owned immutable access facts retained by the created source.
  iree_hal_memory_contract_t* contract;
  // Whether the created pool may allocate new Vulkan slabs.
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
} iree_hal_vulkan_slab_pool_plan_t;

static void iree_hal_vulkan_slab_pool_plan_destroy(
    iree_hal_slab_pool_plan_t* base) {
  iree_hal_vulkan_slab_pool_plan_t* plan =
      (iree_hal_vulkan_slab_pool_plan_t*)base;
  iree_hal_memory_contract_release(plan->contract);
  iree_allocator_free(plan->host_allocator, plan);
}

static iree_status_t iree_hal_vulkan_slab_pool_plan_create(
    iree_hal_slab_pool_plan_t* base, iree_allocator_t host_allocator,
    iree_hal_pool_t** out_pool) {
  iree_hal_vulkan_slab_pool_plan_t* plan =
      (iree_hal_vulkan_slab_pool_plan_t*)base;
  iree_hal_slab_provider_t* provider = NULL;
  iree_status_t status = iree_hal_vulkan_slab_provider_create(
      plan->allocator, plan->provider_options,
      iree_hal_device_host_allocator(plan->provider_options.parent_device),
      &provider);
  if (iree_status_is_ok(status)) {
    const iree_hal_passthrough_pool_options_t options = {
        .memory_contract = plan->contract,
        .epoch_query = plan->progress.epoch_query,
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
    iree_hal_vulkan_slab_pool_plan_vtable = {
        .destroy = iree_hal_vulkan_slab_pool_plan_destroy,
        .create = iree_hal_vulkan_slab_pool_plan_create,
};

// Vulkan does not distinguish write-combined from strongly uncached host
// mappings. Neither class can be promised solely from !HOST_CACHED.
static iree_hal_host_cacheability_t iree_hal_vulkan_memory_host_cacheability(
    VkMemoryPropertyFlags flags) {
  return iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
             ? IREE_HAL_HOST_CACHEABILITY_WRITE_BACK
             : IREE_HAL_HOST_CACHEABILITY_UNKNOWN;
}

static bool iree_hal_vulkan_memory_supports_host_access(
    VkMemoryPropertyFlags flags, iree_hal_pool_host_access_t host) {
  return !host.access ||
         (iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
          (host.cacheability == IREE_HAL_HOST_CACHEABILITY_UNKNOWN ||
           host.cacheability ==
               iree_hal_vulkan_memory_host_cacheability(flags)));
}

static iree_hal_buffer_usage_t iree_hal_vulkan_memory_mapping_usage(
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

static bool iree_hal_vulkan_host_pointer_class_is_supported(
    iree_hal_external_storage_class_t storage_class) {
  return storage_class.type == IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER &&
         storage_class.provenance[0] == 0 && storage_class.provenance[1] == 0;
}

static bool iree_hal_vulkan_external_storage_options_are_supported(
    const iree_hal_vulkan_memory_backend_t* backend,
    const iree_hal_slab_pool_options_t* options) {
  // Vulkan buffers do not yet implement any external export route.
  if (options->owned.exports.class_count) {
    return false;
  }
  if (options->imports.class_count &&
      !iree_hal_vulkan_allocator_supports_host_allocation_import(
          backend->allocator)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < options->imports.class_count; ++i) {
    const iree_hal_pool_import_class_t* import_class =
        &options->imports.classes[i];
    if (!iree_hal_vulkan_host_pointer_class_is_supported(
            import_class->storage_class) ||
        import_class->exports.class_count) {
      return false;
    }
  }
  return true;
}

static bool iree_hal_vulkan_memory_properties_support_slab_storage(
    VkMemoryPropertyFlags flags) {
  return !iree_any_bit_set(flags,
                           VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |
                               VK_MEMORY_PROPERTY_PROTECTED_BIT |
                               VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
                               VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD);
}

// Returns external-buffer memory types that preserve the selected pool memory
// guarantees. Vulkan may expose separate type indices for ordinary and external
// resources even when their HAL-visible properties are identical.
static uint32_t iree_hal_vulkan_compatible_import_memory_type_bits(
    const iree_hal_vulkan_memory_backend_t* backend,
    uint32_t external_memory_type_bits,
    iree_hal_memory_type_t pool_memory_type) {
  iree_hal_memory_type_t required_memory_type =
      pool_memory_type &
      ~(IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_OPTIMAL);
  required_memory_type |=
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;

  uint32_t compatible_memory_type_bits = 0;
  const VkPhysicalDeviceMemoryProperties* memory =
      &backend->physical_device->memory_properties2.memoryProperties;
  for (uint32_t i = 0; i < memory->memoryTypeCount; ++i) {
    if (!(external_memory_type_bits & (1u << i))) {
      continue;
    }
    const VkMemoryPropertyFlags flags = memory->memoryTypes[i].propertyFlags;
    if (!iree_hal_vulkan_memory_properties_support_slab_storage(flags)) {
      continue;
    }
    const iree_hal_memory_type_t memory_type =
        iree_hal_vulkan_memory_type_from_properties(
            backend->physical_device->properties2.properties.deviceType, flags);
    if (iree_all_bits_set(memory_type, required_memory_type)) {
      compatible_memory_type_bits |= 1u << i;
    }
  }
  return compatible_memory_type_bits;
}

// Reports device-local placement only when this native owner selects discrete
// device memory. Host/UMA allocation does not enforce a particular NUMA node.
static iree_hal_pool_placement_t iree_hal_vulkan_memory_placement(
    const iree_hal_vulkan_memory_backend_t* backend,
    iree_hal_device_group_t* group, uint32_t device_ordinal,
    VkMemoryPropertyFlags flags) {
  if (backend->physical_device->properties2.properties.deviceType !=
          VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
      !iree_any_bit_set(flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    return (iree_hal_pool_placement_t){0};
  }
  const iree_hal_topology_t* topology = iree_hal_device_group_topology(group);
  for (iree_host_size_t i = 0; i < iree_hal_topology_node_count(topology);
       ++i) {
    const iree_hal_topology_node_t* node =
        iree_hal_topology_node_at(topology, i);
    if (node->device_ordinal == device_ordinal &&
        node->kind == IREE_HAL_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE) {
      return (iree_hal_pool_placement_t){
          .mode = IREE_HAL_POOL_PLACEMENT_REQUIRED,
          .node = node->ordinal,
      };
    }
  }
  return (iree_hal_pool_placement_t){0};
}

static uint32_t iree_hal_vulkan_memory_preference(
    VkMemoryPropertyFlags flags, iree_hal_memory_type_t memory_type,
    iree_host_size_t family_count, iree_hal_pool_host_access_t host,
    iree_hal_pool_host_access_t preferred_host) {
  uint32_t preference = 0;
  if (preferred_host.access &&
      iree_hal_vulkan_memory_supports_host_access(flags, preferred_host)) {
    preference += 1024;
  }
  if (family_count &&
      iree_any_bit_set(flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    preference += 512;
    if (!host.access &&
        !iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      preference += 256;
    }
  }
  if (!family_count &&
      iree_any_bit_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_LOCAL)) {
    preference += 512;
  }
  if (iree_any_bit_set(host.access, IREE_HAL_MEMORY_ACCESS_READ) &&
      iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
    preference += 128;
  }
  if (iree_any_bit_set(host.access, IREE_HAL_MEMORY_ACCESS_WRITE) &&
      !iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
    preference += 64;
  }
  if (iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
    preference += 8;
  }
  return preference;
}

static iree_status_t iree_hal_vulkan_slab_pool_query_owner(
    const iree_hal_vulkan_memory_backend_t* backend,
    iree_hal_device_group_t* group, uint32_t device_ordinal,
    iree_hal_pool_scope_t scope, const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  if (!iree_hal_vulkan_external_storage_options_are_supported(backend,
                                                              options)) {
    return iree_ok_status();
  }
  uint32_t interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER;
  if (iree_any_bit_set(
          backend->enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)) {
    interfaces |= 1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  }
  const iree_hal_atomic_operation_capabilities_t atomic_operations =
      iree_hal_vulkan_atomic_capabilities(backend->enabled_features).operations;
  bool mapping_source = false;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    mapping_source |=
        iree_any_bit_set(scope.families[i].interfaces,
                         1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE);
  }
  if (mapping_source) {
    if (!iree_all_bits_set(
            backend->enabled_features.general,
            IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED) ||
        options->imports.class_count) {
      // Imported VkDeviceMemory requires matching external handle types on
      // every sparse target at creation. This source format admits owned
      // dense allocations only.
      return iree_ok_status();
    }
    interfaces |= 1u << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_NONE;
  iree_hal_queue_family_affinity_t families = 0;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    if (iree_hal_device_memory_backend(
            iree_hal_queue_family_device(access->family)) != &backend->base ||
        (access->interfaces & ~((uint64_t)interfaces)) ||
        access->requirements ||
        !iree_hal_atomic_operation_capabilities_support(
            atomic_operations, access->atomic_operations)) {
      // HOST_COHERENT alone does not establish automatic device visibility.
      // This native owner does not enable device_coherent_memory, so neither
      // full host/device coherence nor an uncached device route is available.
      return iree_ok_status();
    }
    usage |= access->usage;
    families |= iree_hal_make_queue_family_affinity(access->family->ordinal);
  }
  // Host-only sources still need a native resource creation family. This is
  // private allocation metadata and grants no family access in the contract.
  if (!families) {
    families = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  }

  // Native storage supports both storage and uniform bindings independently
  // of public family permissions, including a scope requesting only a native
  // address. Uniform usage makes Vulkan include its uniform-buffer alignment
  // in the native requirements even when the public scope only grants
  // transfers; storage-buffer alignment alone can be smaller.
  // The provider uses these same usage bits when allocating each slab.
  const iree_hal_buffer_usage_t native_usage =
      usage | IREE_HAL_BUFFER_USAGE_STORAGE |
      IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ;
  const VkBufferCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 4,
      .usage = iree_hal_vulkan_buffer_usage_from_hal(backend->enabled_features,
                                                     native_usage),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VkDeviceBufferMemoryRequirements query = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pCreateInfo = &create_info,
  };
  VkMemoryDedicatedRequirements dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
  };
  VkMemoryRequirements2 requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = &dedicated,
  };
  // For fixed flags and usage, buffer memory types and alignment are
  // independent of size and sharing mode. No resource or payload is acquired.
  iree_vkGetDeviceBufferMemoryRequirements(IREE_VULKAN_DEVICE(backend->syms),
                                           backend->logical_device, &query,
                                           &requirements);

  iree_device_size_t min_alignment = requirements.memoryRequirements.alignment;
  iree_device_size_t mapping_granularity = 0;
  if (mapping_source) {
    if (dedicated.requiresDedicatedAllocation) {
      return iree_ok_status();
    }
    const VkMemoryRequirements sparse_requirements =
        iree_hal_vulkan_sparse_buffer_query_requirements(
            backend->syms, backend->logical_device, backend->enabled_features,
            native_usage);
    if (!sparse_requirements.alignment ||
        sparse_requirements.alignment > UINT32_MAX) {
      return iree_ok_status();
    }
    mapping_granularity = sparse_requirements.alignment;
    min_alignment = iree_max(min_alignment, mapping_granularity);
    requirements.memoryRequirements.memoryTypeBits &=
        sparse_requirements.memoryTypeBits;
  }
  uint32_t external_memory_type_bits = 0;
  if (options->imports.class_count) {
    const VkExternalMemoryBufferCreateInfo external_create_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    VkBufferCreateInfo external_buffer_create_info = create_info;
    external_buffer_create_info.pNext = &external_create_info;
    const VkDeviceBufferMemoryRequirements external_query = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
        .pCreateInfo = &external_buffer_create_info,
    };
    VkMemoryRequirements2 external_requirements = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
    };
    iree_vkGetDeviceBufferMemoryRequirements(
        IREE_VULKAN_DEVICE(backend->syms), backend->logical_device,
        &external_query, &external_requirements);
    external_memory_type_bits =
        external_requirements.memoryRequirements.memoryTypeBits;
    min_alignment = iree_max(
        min_alignment,
        (iree_device_size_t)external_requirements.memoryRequirements.alignment);

    const VkDeviceSize import_alignment =
        backend->physical_device->external_memory_host_properties
            .minImportedHostPointerAlignment;
    if (import_alignment > IREE_DEVICE_SIZE_MAX ||
        (import_alignment && !iree_device_size_is_valid_alignment(
                                 (iree_device_size_t)import_alignment))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan reported invalid minImportedHostPointerAlignment %" PRIu64,
          (uint64_t)import_alignment);
    }
    min_alignment =
        iree_max(min_alignment, (iree_device_size_t)import_alignment);
  }

  const VkPhysicalDeviceMemoryProperties* memory =
      &backend->physical_device->memory_properties2.memoryProperties;
  const VkPhysicalDeviceLimits* limits =
      &backend->physical_device->properties2.properties.limits;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < memory->memoryTypeCount && iree_status_is_ok(status);
       ++i) {
    const VkMemoryPropertyFlags flags = memory->memoryTypes[i].propertyFlags;
    if (!(requirements.memoryRequirements.memoryTypeBits & (1u << i)) ||
        !iree_hal_vulkan_memory_properties_support_slab_storage(flags) ||
        (options->imports.class_count &&
         !iree_any_bit_set(flags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ||
        !iree_hal_vulkan_memory_supports_host_access(flags, scope.host)) {
      continue;
    }
    // External host storage already has backing. An owned-allocation candidate
    // cannot impose its physical placement on imports admitted by the same
    // pool; the actual pointer's native memory type is resolved at attachment.
    iree_hal_pool_placement_t placement =
        options->imports.class_count
            ? (iree_hal_pool_placement_t){0}
            : iree_hal_vulkan_memory_placement(backend, group, device_ordinal,
                                               flags);
    if (options->placement.mode == IREE_HAL_POOL_PLACEMENT_REQUIRED &&
        (placement.mode != IREE_HAL_POOL_PLACEMENT_REQUIRED ||
         placement.node != options->placement.node)) {
      continue;
    }
    iree_hal_memory_type_t memory_type =
        iree_hal_vulkan_memory_type_from_properties(
            backend->physical_device->properties2.properties.deviceType, flags);
    if (options->imports.class_count) {
      memory_type &= ~(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                       IREE_HAL_MEMORY_TYPE_HOST_LOCAL);
    }
    const uint32_t import_memory_type_bits =
        options->imports.class_count
            ? iree_hal_vulkan_compatible_import_memory_type_bits(
                  backend, external_memory_type_bits, memory_type)
            : 0;
    if (options->imports.class_count && !import_memory_type_bits) {
      continue;
    }
    iree_hal_pool_host_access_t host = scope.host;
    if (iree_hal_vulkan_memory_supports_host_access(
            flags, options->preferences.host)) {
      host.access |= options->preferences.host.access;
      host.modes |= options->preferences.host.modes;
    }
    if (host.access) {
      host.cacheability = iree_hal_vulkan_memory_host_cacheability(flags);
    }
    const iree_hal_buffer_usage_t mapping_usage =
        iree_hal_vulkan_memory_mapping_usage(host);

    iree_hal_vulkan_slab_pool_plan_t* plan = NULL;
    status =
        iree_allocator_malloc(host_allocator, sizeof(*plan), (void**)&plan);
    if (!iree_status_is_ok(status)) {
      break;
    }
    plan->base.vtable = &iree_hal_vulkan_slab_pool_plan_vtable;
    plan->base.info = (iree_hal_slab_pool_plan_info_t){
        .placement = placement,
        .preference = iree_hal_vulkan_memory_preference(
            flags, memory_type, scope.family_count, host,
            options->preferences.host),
        .host = host,
    };
    plan->host_allocator = host_allocator;
    plan->allocator = backend->allocator;
    plan->provider_options = (iree_hal_vulkan_slab_provider_options_t){
        .parent_device = backend->device,
        .allocation_queue = backend->allocation_queue,
        .syms = backend->syms,
        .logical_device = backend->logical_device,
        .memory_type_index = i,
        .import_memory_type_bits = import_memory_type_bits,
        .memory_type = memory_type,
        .supported_usage = native_usage | mapping_usage,
        .atomic_operations = atomic_operations,
        .queue_family_affinity_mask = families,
        .min_alignment = min_alignment,
        .mapping_granularity = mapping_granularity,
        .non_coherent_atom_size = limits->nonCoherentAtomSize,
    };
    plan->progress.notification = backend->notification;
    plan->progress.epoch_query = backend->epoch_query;
    plan->progress.tracker =
        iree_hal_device_topology_info(backend->device)->frontier.tracker;
    plan->owned_mode = options->owned.mode;
    plan->imports.class_count = options->imports.class_count;
    plan->imports.classes = options->imports.classes;
    plan->trace_name = options->trace_name;
    status = iree_hal_memory_contract_create(
        iree_hal_device_group_memory_domain(group),
        iree_hal_device_group_memory_scope_count(group),
        iree_hal_vulkan_buffer_binding_layout(), host_allocator,
        &plan->contract);
    if (iree_status_is_ok(status)) {
      iree_hal_memory_contract_t* contract = plan->contract;
      contract->host = host;
      contract->placement = placement;
      contract->allowed_access =
          iree_hal_buffer_usage_access(usage) | host.access;
      contract->allowed_usage = usage | mapping_usage;
      if (mapping_source) {
        contract->mapping_source.type = IREE_HAL_MEMORY_BACKEND_VULKAN;
        contract->mapping_source.requirements =
            (iree_hal_pool_mapping_requirements_t){
                .minimum_granularity = mapping_granularity,
                .recommended_granularity = mapping_granularity,
            };
      }
      for (iree_host_size_t j = 0; j < scope.family_count; ++j) {
        const uint32_t queue_scope_id =
            scope.families[j].family->memory.queue_scope_id;
        for (uint32_t id = queue_scope_id; id <= queue_scope_id + 1; ++id) {
          iree_hal_memory_scope_access_t* access = &contract->scopes[id];
          access->usage = scope.families[j].usage;
          access->interfaces = interfaces;
          access->bindings[IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER] =
              IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE;
          if (interfaces & (1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS)) {
            access->bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] =
                IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS;
          }
          if (mapping_source) {
            access->bindings[IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE] =
                IREE_HAL_VULKAN_BUFFER_BINDING_MAPPING_SOURCE;
          }
        }
      }
      status = callback.fn(callback.user_data, &plan->base);
    } else {
      iree_hal_vulkan_slab_pool_plan_destroy(&plan->base);
    }
  }
  return status;
}

static iree_status_t iree_hal_vulkan_slab_pool_query(
    void* self, iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    const iree_hal_slab_pool_options_t* options,
    iree_hal_slab_pool_plan_callback_t callback,
    iree_allocator_t host_allocator) {
  (void)self;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group) &&
                               iree_status_is_ok(status);
       ++i) {
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (backend && backend->type == IREE_HAL_MEMORY_BACKEND_VULKAN) {
      status = iree_hal_vulkan_slab_pool_query_owner(
          (const iree_hal_vulkan_memory_backend_t*)backend, group, (uint32_t)i,
          scope, options, callback, host_allocator);
    }
  }
  return status;
}

static const iree_hal_slab_pool_factory_t iree_hal_vulkan_slab_pool_factory = {
    .query = iree_hal_vulkan_slab_pool_query,
};
static const iree_hal_slab_pool_factory_t* const
    iree_hal_vulkan_slab_pool_factories[] = {
        &iree_hal_vulkan_slab_pool_factory,
};

void iree_hal_vulkan_memory_backend_initialize(
    iree_hal_vulkan_memory_backend_t* backend) {
  backend->base = (iree_hal_memory_backend_t){
      .type = IREE_HAL_MEMORY_BACKEND_VULKAN,
      .factory_count = IREE_ARRAYSIZE(iree_hal_vulkan_slab_pool_factories),
      .factories = iree_hal_vulkan_slab_pool_factories,
      .query_virtual_buffer = iree_hal_vulkan_virtual_buffer_query_capabilities,
      .create_virtual_buffer = iree_hal_vulkan_virtual_buffer_create,
  };
}
