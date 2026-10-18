// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "iree/hal/drivers/vulkan/atomic.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/memory_backend.h"
#include "iree/hal/drivers/vulkan/queue.h"
#include "iree/hal/drivers/vulkan/sparse_buffer.h"

typedef struct iree_hal_vulkan_virtual_buffer_t {
  // Ordinary buffer resource owning the sparse address interval.
  iree_hal_virtual_buffer_t base;
  // Device-owned native namespace and group progress services.
  const iree_hal_vulkan_memory_backend_t* backend;
  // Inline native resource facts borrowed by the stable view.
  iree_hal_vulkan_buffer_native_t native;
  // Virtual capacity and placement-local progress, without physical backing.
  iree_hal_buffer_backing_facts_t backing;
  // Memory types admitted by the sparse resource's fixed creation parameters.
  uint32_t memory_type_bits;
  // Terminal native publication failure, shared across submitting queues.
  iree_atomic_int32_t failure_code;
  // Native sparse page unit, independent of a larger requested mapping unit.
  VkDeviceSize mapping_alignment;
  // Scope-indexed aliases of this resource's one native page table.
  uint32_t projections[];
} iree_hal_vulkan_virtual_buffer_t;

static void iree_hal_vulkan_virtual_buffer_destroy(iree_hal_buffer_t* base) {
  iree_hal_vulkan_virtual_buffer_t* buffer =
      (iree_hal_vulkan_virtual_buffer_t*)base;
  iree_vkDestroyBuffer(
      IREE_VULKAN_DEVICE(buffer->backend->syms),
      buffer->backend->logical_device,
      (VkBuffer)buffer->native.bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
          .vulkan.buffer,
      /*pAllocator=*/NULL);
  iree_hal_memory_contract_release(base->memory.contract);
  iree_allocator_free(buffer->base.host_allocator, buffer);
}

static iree_status_t iree_hal_vulkan_virtual_buffer_detach(
    iree_hal_buffer_t* base, uint32_t scope_id) {
  iree_hal_vulkan_virtual_buffer_t* buffer =
      (iree_hal_vulkan_virtual_buffer_t*)base;
  const VkSparseMemoryBind bind = {
      .resourceOffset = base->memory.binding_offset,
      .size = base->byte_length,
  };
  const VkSparseBufferMemoryBindInfo buffer_bind = {
      .buffer = (VkBuffer)buffer->native
                    .bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
                    .vulkan.buffer,
      .bindCount = 1,
      .pBinds = &bind,
  };
  // Full removal is independent of the sticky mapping error and preserves it.
  // The caller has retired every prior use and accepted mapping submission.
  iree_status_t status = iree_hal_vulkan_queue_submit_sparse_bind_sync(
      buffer->backend->sparse_binding_queue, 1, &buffer_bind);
  if (!iree_status_is_ok(status)) {
    int32_t expected = IREE_STATUS_OK;
    iree_atomic_compare_exchange_strong(
        &buffer->failure_code, &expected, iree_status_code(status),
        iree_memory_order_release, iree_memory_order_relaxed);
  }
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_vulkan_virtual_buffer_vtable = {
    .destroy = iree_hal_vulkan_virtual_buffer_destroy,
    .export_range = iree_hal_virtual_buffer_export_range,
    .map_range = iree_hal_virtual_buffer_map_range,
    .unmap_range = iree_hal_virtual_buffer_unmap_range,
    .invalidate_range = iree_hal_virtual_buffer_invalidate_range,
    .flush_range = iree_hal_virtual_buffer_flush_range,
    .query_virtual = iree_hal_virtual_buffer_query_root,
    .detach_mappings = iree_hal_vulkan_virtual_buffer_detach,
};

bool iree_hal_vulkan_virtual_buffer_isa(iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(buffer, &iree_hal_vulkan_virtual_buffer_vtable);
}

typedef struct iree_hal_vulkan_reservation_plan_t {
  // Zero geometry denotes an unsupported complete scope.
  iree_hal_virtual_buffer_capabilities_t capabilities;
  // Native creation usage, including the driver's binding requirements.
  iree_hal_buffer_usage_t native_usage;
  // Interfaces supplied at every admitted queue and program scope.
  uint32_t interfaces;
  // Immutable native geometry and compatible physical memory types.
  VkMemoryRequirements requirements;
  // Number of distinct native access families.
  uint32_t family_count;
  // Deduplicated native families used for concurrent resource sharing.
  uint32_t families[IREE_HAL_MAX_QUEUE_FAMILIES];
} iree_hal_vulkan_reservation_plan_t;

// Cold scope qualification uses only the sealed group's native owners. The
// query and constructor consume the same plan without acquiring scratch
// buffers.
static void iree_hal_vulkan_virtual_buffer_prepare(
    const iree_hal_vulkan_memory_backend_t* backend,
    iree_hal_pool_scope_t scope, iree_hal_vulkan_reservation_plan_t* plan) {
  *plan = (iree_hal_vulkan_reservation_plan_t){0};
  if (!scope.family_count || scope.host.access ||
      !backend->sparse_binding_queue ||
      !iree_all_bits_set(
          backend->enabled_features.general,
          IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED)) {
    return;
  }
  plan->interfaces = 1u << IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER;
  const bool has_device_address =
      iree_any_bit_set(backend->enabled_features.general,
                       IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES);
  if (has_device_address) {
    plan->interfaces |= 1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  }
  const iree_hal_atomic_operation_capabilities_t atomic_operations =
      iree_hal_vulkan_atomic_capabilities(backend->enabled_features).operations;
  plan->native_usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                       IREE_HAL_BUFFER_USAGE_DISPATCH_UNIFORM_READ;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    if (iree_hal_device_memory_backend(
            iree_hal_queue_family_device(access->family)) != &backend->base ||
        (access->interfaces & ~((uint64_t)plan->interfaces)) ||
        access->requirements ||
        !iree_hal_atomic_operation_capabilities_support(
            atomic_operations, access->atomic_operations)) {
      return;
    }
    const iree_hal_vulkan_queue_t* queue =
        (const iree_hal_vulkan_queue_t*)iree_hal_device_queue(
            backend->device, access->family->ordinal, 0);
    if (!queue) {
      return;
    }
    uint32_t j = 0;
    for (; j < plan->family_count; ++j) {
      if (plan->families[j] == queue->queue_family_index) {
        break;
      }
    }
    if (j == plan->family_count) {
      plan->families[plan->family_count++] = queue->queue_family_index;
    }
    plan->native_usage |= access->usage;
  }
  plan->requirements = iree_hal_vulkan_sparse_buffer_query_requirements(
      backend->syms, backend->logical_device, backend->enabled_features,
      plan->native_usage);
  const iree_device_size_t granularity = plan->requirements.alignment;
  const iree_device_size_t max_length =
      iree_min(backend->physical_device->properties2.properties.limits
                   .sparseAddressSpaceSize,
               backend->physical_device->maintenance4_properties.maxBufferSize);
  if (!granularity || granularity > max_length ||
      !plan->requirements.memoryTypeBits) {
    return;
  }
  plan->capabilities = (iree_hal_virtual_buffer_capabilities_t){
      .max_byte_length = max_length - max_length % granularity,
      .reservation_granularity = granularity,
      .mapping_granularity = granularity,
      .max_alignment =
          has_device_address
              ? UINT64_C(1)
                    << (63 - iree_math_count_leading_zeros_u64(max_length))
              : granularity,
      .operations = IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL,
  };
}

iree_status_t iree_hal_vulkan_virtual_buffer_query_capabilities(
    const iree_hal_memory_backend_t* base, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities) {
  iree_hal_vulkan_reservation_plan_t plan;
  iree_hal_vulkan_virtual_buffer_prepare(
      (const iree_hal_vulkan_memory_backend_t*)base, scope, &plan);
  *out_capabilities = plan.capabilities;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_virtual_buffer_create(
    const iree_hal_memory_backend_t* base, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  const iree_hal_vulkan_memory_backend_t* backend =
      (const iree_hal_vulkan_memory_backend_t*)base;
  iree_hal_vulkan_reservation_plan_t plan;
  iree_hal_vulkan_virtual_buffer_prepare(backend, scope, &plan);
  if (!plan.capabilities.max_byte_length) {
    return iree_ok_status();
  }
  const iree_device_size_t alignment = iree_max(
      options->min_alignment, plan.capabilities.reservation_granularity);
  const iree_device_size_t granularity = iree_max(
      options->mapping_granularity, plan.capabilities.mapping_granularity);
  // Sparse addresses already satisfy the native alignment. Larger requested
  // alignments use an interior view without committing any physical storage.
  iree_device_size_t native_length = 0;
  if (options->byte_length % granularity ||
      alignment > plan.capabilities.max_alignment ||
      !iree_device_size_checked_add(
          options->byte_length,
          alignment - plan.capabilities.reservation_granularity,
          &native_length) ||
      native_length > plan.capabilities.max_byte_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "reservation exceeds Vulkan address geometry");
  }

  const uint32_t scope_count = iree_hal_device_group_memory_scope_count(group);
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(iree_hal_vulkan_virtual_buffer_t), &total_size,
                         IREE_STRUCT_FIELD(scope_count, uint32_t, NULL)));
  iree_hal_vulkan_virtual_buffer_t* reservation = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&reservation));
  reservation->backend = backend;
  reservation->memory_type_bits = plan.requirements.memoryTypeBits;
  reservation->failure_code = IREE_ATOMIC_VAR_INIT(IREE_STATUS_OK);
  reservation->mapping_alignment = plan.requirements.alignment;
  for (uint32_t i = 0; i < scope_count; ++i) {
    reservation->projections[i] = IREE_HAL_MEMORY_SCOPE_INVALID;
  }
  iree_hal_memory_contract_t* contract = NULL;
  iree_status_t status = iree_hal_memory_contract_create(
      iree_hal_device_group_memory_domain(group), scope_count,
      iree_hal_vulkan_buffer_binding_layout(), host_allocator, &contract);
  VkBuffer handle = VK_NULL_HANDLE;
  if (iree_status_is_ok(status)) {
    const VkBufferCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
                 VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                 VK_BUFFER_CREATE_SPARSE_ALIASED_BIT,
        .size = native_length,
        .usage = iree_hal_vulkan_buffer_usage_from_hal(
            backend->enabled_features, plan.native_usage),
        .sharingMode = plan.family_count > 1 ? VK_SHARING_MODE_CONCURRENT
                                             : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = plan.family_count > 1 ? plan.family_count : 0,
        .pQueueFamilyIndices = plan.family_count > 1 ? plan.families : NULL,
    };
    status = iree_vkCreateBuffer(IREE_VULKAN_DEVICE(backend->syms),
                                 backend->logical_device, &create_info,
                                 /*pAllocator=*/NULL, &handle);
  }
  if (iree_status_is_ok(status)) {
    VkDeviceAddress address = 0;
    if (plan.interfaces & (1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS)) {
      const VkBufferDeviceAddressInfo address_info = {
          .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
          .buffer = handle,
      };
      address =
          iree_vkGetBufferDeviceAddress(IREE_VULKAN_DEVICE(backend->syms),
                                        backend->logical_device, &address_info);
    }
    const VkDeviceSize offset = (0 - address) & (alignment - 1);
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      const iree_hal_pool_family_access_t* family = &scope.families[i];
      contract->allowed_usage |= family->usage;
      const uint32_t id = family->family->memory.queue_scope_id;
      iree_hal_memory_scope_access_t* site = &contract->scopes[id];
      site->usage = family->usage;
      site->interfaces = plan.interfaces;
      site->bindings[IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER] =
          IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE;
      if (plan.interfaces & (1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS)) {
        site->bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] =
            IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS;
      }
      contract->scopes[id + 1] = *site;
      reservation->projections[id] = 0;
      reservation->projections[id + 1] = 0;
    }
    contract->allowed_access =
        iree_hal_buffer_usage_access(contract->allowed_usage);
    reservation->backing = (iree_hal_buffer_backing_facts_t){
        .notification = backend->notification,
        .tracker =
            iree_hal_device_topology_info(backend->device)->frontier.tracker,
        .maintenance_range = {.byte_length = options->byte_length},
        .allocation_alignment = alignment,
        .maintenance_alignment = 1,
    };
    iree_hal_virtual_buffer_initialize(
        contract, reservation->native.bindings, &reservation->backing,
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, options->byte_length, granularity,
        plan.capabilities.operations, IREE_HAL_MEMORY_BACKEND_VULKAN,
        &iree_hal_vulkan_virtual_buffer_vtable, host_allocator,
        &reservation->base);
    iree_hal_vulkan_buffer_initialize_bindings(
        &reservation->base.buffer, VK_NULL_HANDLE, handle, native_length,
        address, offset, &reservation->native);
    reservation->base.projection_count = 1;
    reservation->base.projections = reservation->projections;
    *out_buffer = &reservation->base.buffer;
  } else {
    iree_allocator_free(host_allocator, reservation);
  }
  iree_hal_memory_contract_release(contract);
  return status;
}

iree_status_t iree_hal_vulkan_memory_mapping_batch_validate(
    const iree_hal_memory_mapping_batch_t* batch, VkDevice logical_device) {
  if (batch->entry_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan mapping buffer count exceeds uint32");
  }
  for (iree_host_size_t i = 0; i < batch->entry_count; ++i) {
    const iree_hal_memory_mapping_batch_entry_t* entry = &batch->entries[i];
    if (!iree_hal_vulkan_virtual_buffer_isa(&entry->target->buffer)) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "Vulkan queue requires Vulkan reservations");
    }
    const iree_hal_vulkan_virtual_buffer_t* reservation =
        (const iree_hal_vulkan_virtual_buffer_t*)entry->target;
    if (reservation->backend->logical_device != logical_device) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "reservation belongs to another Vulkan device");
    }
    if (entry->update_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan mapping update count exceeds uint32");
    }
    for (iree_host_size_t j = 0; j < entry->update_count; ++j) {
      if (entry->updates[j].access != IREE_HAL_MEMORY_ACCESS_NONE &&
          entry->updates[j].access != IREE_HAL_MEMORY_ACCESS_ALL) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "Vulkan sparse bindings support NONE or READ|WRITE access");
      }
    }
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_memory_mapping_batch_resolve(
    iree_hal_memory_mapping_batch_t* batch, VkDevice logical_device,
    VkSparseBufferMemoryBindInfo* buffers, VkSparseMemoryBind* binds) {
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_batch_resolve(batch));
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_memory_mapping_batch_validate(batch, logical_device));
  for (iree_host_size_t i = 0; i < batch->entry_count; ++i) {
    const iree_hal_memory_mapping_batch_entry_t* entry = &batch->entries[i];
    const iree_hal_vulkan_virtual_buffer_t* reservation =
        (const iree_hal_vulkan_virtual_buffer_t*)entry->target;
    const iree_status_code_t failure_code =
        (iree_status_code_t)iree_atomic_load(&reservation->failure_code,
                                             iree_memory_order_acquire);
    if (failure_code != IREE_STATUS_OK) {
      return iree_make_status(failure_code,
                              "Vulkan mapping updates are unavailable after "
                              "a failed native publication");
    }
    buffers[i] = (VkSparseBufferMemoryBindInfo){
        .buffer = (VkBuffer)reservation->native
                      .bindings[IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE]
                      .vulkan.buffer,
        .bindCount = (uint32_t)entry->update_count,
        .pBinds = binds,
    };
    for (iree_host_size_t j = 0; j < entry->update_count; ++j) {
      const iree_hal_memory_mapping_update_t* update = &entry->updates[j];
      VkDeviceMemory memory = VK_NULL_HANDLE;
      VkDeviceSize memory_offset = 0;
      if (update->access) {
        const iree_hal_vulkan_mapping_source_info_t source =
            iree_hal_vulkan_buffer_mapping_source_info(
                update->source.mapping_source.descriptor);
        memory_offset = update->source.mapping_source.offset;
        if (source.logical_device != reservation->backend->logical_device ||
            !(reservation->memory_type_bits &
              (1u << source.memory_type_index))) {
          return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "mapping source has an incompatible Vulkan "
                                  "device or memory type");
        }
        if (memory_offset % reservation->mapping_alignment ||
            memory_offset > source.byte_length ||
            update->target_range.byte_length >
                source.byte_length - memory_offset) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "mapping source is misaligned or exceeds its "
                                  "Vulkan allocation extent");
        }
        memory = source.memory;
      }
      binds[j] = (VkSparseMemoryBind){
          .resourceOffset = reservation->base.buffer.memory.binding_offset +
                            update->target_range.byte_offset,
          .size = update->target_range.byte_length,
          .memory = memory,
          .memoryOffset = memory_offset,
      };
    }
    binds += entry->update_count;
  }
  return iree_ok_status();
}

void iree_hal_vulkan_memory_mapping_batch_fail(
    iree_hal_memory_mapping_batch_t* batch, iree_status_code_t failure_code) {
  for (iree_host_size_t i = 0; i < batch->entry_count; ++i) {
    iree_hal_vulkan_virtual_buffer_t* reservation =
        (iree_hal_vulkan_virtual_buffer_t*)batch->entries[i].target;
    iree_atomic_store(&reservation->failure_code, failure_code,
                      iree_memory_order_release);
  }
}
