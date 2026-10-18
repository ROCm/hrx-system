// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_group.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/memory_backend.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/virtual_memory.h"

// ROCr shares one virtual address map across the admitted agents. Native
// updates are cold host calls; the extent inventory exists only because ROCr
// cannot detach a sparse reservation without knowing its installed ranges.
typedef struct iree_hal_amdgpu_virtual_buffer_t {
  // Ordinary resource owning the stable addresses and mapping state.
  iree_hal_virtual_buffer_t base;
  // Borrowed device-lifetime native namespace and publication service.
  const iree_hal_amdgpu_memory_backend_t* backend;
  // Owned scalar reservation used by the retained HIP VMM implementation.
  iree_hal_buffer_t* reservation;
  // Immutable native addresses copied from the scalar reservation.
  iree_hal_amdgpu_buffer_native_t native;
  // Virtual geometry and group progress services for child views.
  iree_hal_buffer_backing_facts_t backing;
  // Serializes native updates and their installed-range inventory.
  iree_slim_mutex_t mutex;
  // First native update failure; checked removal remains available afterward.
  iree_status_code_t failure_code;
  // Allocated entries in ranges; additions reserve storage before mutation.
  iree_host_size_t range_capacity;
  // Number of sorted, nonoverlapping installed ranges.
  iree_host_size_t range_count;
  // Native mappings only; source lifetime remains the caller's obligation.
  iree_hal_buffer_range_t* ranges;
  // Exact device-local families sharing this native projection.
  iree_hal_queue_family_affinity_t affinity;
  // Scope-indexed aliases of the shared native address map.
  uint32_t projections[];
} iree_hal_amdgpu_virtual_buffer_t;

static iree_status_t iree_hal_amdgpu_virtual_buffer_remove(
    iree_hal_amdgpu_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_unmap(
      buffer->backend->allocator, buffer->reservation, offset, length));
  const iree_device_size_t end = offset + length;
  iree_host_size_t i = 0;
  while (i < buffer->range_count) {
    iree_hal_buffer_range_t* range = &buffer->ranges[i];
    const iree_device_size_t range_end =
        range->byte_offset + range->byte_length;
    if (range_end <= offset || range->byte_offset >= end) {
      ++i;
    } else {
      memmove(range, range + 1, (buffer->range_count - i - 1) * sizeof(*range));
      --buffer->range_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_virtual_buffer_detach(
    iree_hal_buffer_t* base, uint32_t scope_id) {
  (void)scope_id;
  iree_hal_amdgpu_virtual_buffer_t* buffer =
      (iree_hal_amdgpu_virtual_buffer_t*)base;
  iree_slim_mutex_lock(&buffer->mutex);
  iree_status_t status = iree_ok_status();
  while (buffer->range_count && iree_status_is_ok(status)) {
    const iree_hal_buffer_range_t range =
        buffer->ranges[buffer->range_count - 1];
    status = iree_hal_allocator_virtual_memory_unmap(
        buffer->backend->allocator, buffer->reservation, range.byte_offset,
        range.byte_length);
    if (iree_status_is_ok(status)) {
      --buffer->range_count;
    }
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return status;
}

static void iree_hal_amdgpu_virtual_buffer_destroy(iree_hal_buffer_t* base) {
  iree_hal_amdgpu_virtual_buffer_t* buffer =
      (iree_hal_amdgpu_virtual_buffer_t*)base;
  iree_status_t status = iree_hal_amdgpu_virtual_buffer_detach(base, 0);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer->reservation);
  }
  // A failed native detach leaves its addresses reserved. Reusing them could
  // alias live native mappings, so cleanup failure deliberately leaks them.
  IREE_ASSERT(iree_status_is_ok(status), "ROCr virtual buffer detach failed");
  iree_status_ignore(status);
  iree_slim_mutex_deinitialize(&buffer->mutex);
  iree_allocator_free(buffer->base.host_allocator, buffer->ranges);
  iree_hal_memory_contract_release(base->memory.contract);
  iree_allocator_free(buffer->base.host_allocator, buffer);
}

static iree_status_t iree_hal_amdgpu_virtual_buffer_update(
    iree_hal_buffer_t* base, uint32_t projection, iree_host_size_t count,
    const iree_hal_memory_mapping_update_t* updates) {
  (void)projection;
  iree_hal_amdgpu_virtual_buffer_t* buffer =
      (iree_hal_amdgpu_virtual_buffer_t*)base;
  iree_slim_mutex_lock(&buffer->mutex);
  iree_status_t status = iree_status_from_code(buffer->failure_code);
  // ROCr maps and protects complete handles, and unmaps complete bindings.
  // Reject unsupported public ranges before this vector's native mutations.
  iree_host_size_t additions = 0;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    if (updates[i].operation == IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) {
      ++additions;
    } else if (updates[i].operation ==
               IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND) {
      const iree_device_size_t begin = updates[i].target_range.byte_offset;
      const iree_device_size_t end =
          begin + updates[i].target_range.byte_length;
      iree_device_size_t covered = begin;
      for (iree_host_size_t j = 0; j < buffer->range_count && covered < end;
           ++j) {
        const iree_hal_buffer_range_t range = buffer->ranges[j];
        if (range.byte_offset == covered &&
            range.byte_length <= end - covered) {
          covered += range.byte_length;
        }
      }
      if (covered != end) {
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "ROCr unbinding requires complete contiguous native mappings");
      }
    }
    if (!updates[i].source.mapping_source.descriptor) {
      continue;
    }
    const iree_hal_amdgpu_vmem_source_t* source =
        updates[i].source.mapping_source.descriptor;
    if (updates[i].source.mapping_source.offset ||
        updates[i].target_range.byte_length != source->byte_length) {
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "ROCr mappings require a complete source allocation");
    }
  }
  iree_host_size_t capacity = 0;
  if (iree_status_is_ok(status) &&
      (!iree_host_size_checked_add(buffer->range_count, additions, &capacity) ||
       capacity > IREE_HOST_SIZE_MAX / sizeof(*buffer->ranges))) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "ROCr mapping inventory is too large");
  }
  if (iree_status_is_ok(status) && capacity > buffer->range_capacity) {
    status = iree_allocator_realloc(buffer->base.host_allocator,
                                    capacity * sizeof(*buffer->ranges),
                                    (void**)&buffer->ranges);
    if (iree_status_is_ok(status)) {
      buffer->range_capacity = capacity;
    }
  }
  bool mutated = false;
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const iree_hal_memory_mapping_update_t* update = &updates[i];
    const iree_hal_buffer_range_t range = update->target_range;
    if (update->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND) {
      status = iree_hal_amdgpu_virtual_buffer_remove(buffer, range.byte_offset,
                                                     range.byte_length);
      mutated |= iree_status_is_ok(status);
      continue;
    }
    if (update->operation == IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) {
      const iree_hal_amdgpu_vmem_source_t* source =
          update->source.mapping_source.descriptor;
      void* address =
          (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(buffer->reservation) +
          range.byte_offset;
      status =
          iree_hsa_amd_vmem_map(IREE_LIBHSA(buffer->backend->libhsa), address,
                                range.byte_length, 0, source->handle, 0);
      if (iree_status_is_ok(status)) {
        iree_host_size_t index = 0;
        while (index < buffer->range_count &&
               buffer->ranges[index].byte_offset < range.byte_offset) {
          ++index;
        }
        memmove(buffer->ranges + index + 1, buffer->ranges + index,
                (buffer->range_count - index) * sizeof(*buffer->ranges));
        buffer->ranges[index] = range;
        ++buffer->range_count;
        mutated = true;
      }
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_memory_protection_t protection =
          ((update->access & IREE_HAL_MEMORY_ACCESS_READ)
               ? IREE_HAL_MEMORY_PROTECTION_READ
               : 0) |
          ((update->access & IREE_HAL_MEMORY_ACCESS_WRITE)
               ? IREE_HAL_MEMORY_PROTECTION_WRITE
               : 0);
      status = iree_hal_allocator_virtual_memory_protect(
          buffer->backend->allocator, buffer->reservation, range.byte_offset,
          range.byte_length, buffer->affinity,
          IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE, protection);
      mutated = true;
    }
  }
  if (mutated && !iree_status_is_ok(status) && !buffer->failure_code) {
    buffer->failure_code = iree_status_code(status);
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_virtual_buffer_vtable = {
    .destroy = iree_hal_amdgpu_virtual_buffer_destroy,
    .export_range = iree_hal_virtual_buffer_export_range,
    .map_range = iree_hal_virtual_buffer_map_range,
    .unmap_range = iree_hal_virtual_buffer_unmap_range,
    .invalidate_range = iree_hal_virtual_buffer_invalidate_range,
    .flush_range = iree_hal_virtual_buffer_flush_range,
    .query_virtual = iree_hal_virtual_buffer_query_root,
    .update_mappings = iree_hal_amdgpu_virtual_buffer_update,
    .detach_mappings = iree_hal_amdgpu_virtual_buffer_detach,
};

static iree_status_t iree_hal_amdgpu_virtual_buffer_prepare(
    const iree_hal_amdgpu_memory_backend_t* backend,
    iree_hal_pool_scope_t scope, iree_hal_queue_family_affinity_t* out_affinity,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities) {
  *out_affinity = 0;
  *out_capabilities = (iree_hal_virtual_buffer_capabilities_t){0};
  if (!scope.family_count || scope.host.access ||
      !iree_hal_allocator_supports_virtual_memory(backend->allocator)) {
    return iree_ok_status();
  }
  const iree_hal_atomic_operation_capabilities_t no_atomics = {0};
  iree_hal_buffer_usage_t usage = 0;
  for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
    const iree_hal_pool_family_access_t* access = &scope.families[i];
    if (iree_hal_queue_family_device(access->family) != backend->device ||
        access->requirements ||
        (access->interfaces &
         ~(UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS)) ||
        !iree_hal_atomic_operation_capabilities_support(
            no_atomics, access->atomic_operations)) {
      return iree_ok_status();
    }
    *out_affinity |= UINT64_C(1) << access->family->ordinal;
    usage |= access->usage;
  }
  const iree_hal_physical_memory_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = usage,
      .queue_family_affinity = *out_affinity,
  };
  iree_device_size_t minimum = 0, recommended = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_virtual_memory_query_granularity(
      backend->allocator, params, &minimum, &recommended));
  *out_capabilities = (iree_hal_virtual_buffer_capabilities_t){
      .max_byte_length = IREE_HOST_SIZE_MAX - IREE_HOST_SIZE_MAX % minimum,
      .reservation_granularity = minimum,
      .mapping_granularity = minimum,
      .max_alignment = minimum,
      .operations = (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) |
                    (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND) |
                    (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT),
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_virtual_buffer_query_capabilities(
    const iree_hal_memory_backend_t* base, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope, iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities) {
  (void)group;
  (void)host_allocator;
  iree_hal_queue_family_affinity_t affinity;
  return iree_hal_amdgpu_virtual_buffer_prepare(
      (const iree_hal_amdgpu_memory_backend_t*)base, scope, &affinity,
      out_capabilities);
}

iree_status_t iree_hal_amdgpu_virtual_buffer_create(
    const iree_hal_memory_backend_t* base, iree_hal_device_group_t* group,
    iree_hal_pool_scope_t scope,
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  const iree_hal_amdgpu_memory_backend_t* backend =
      (const iree_hal_amdgpu_memory_backend_t*)base;
  iree_hal_queue_family_affinity_t affinity;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_virtual_buffer_prepare(
      backend, scope, &affinity, &capabilities));
  if (!capabilities.max_byte_length) {
    return iree_ok_status();
  }
  if (options->min_alignment > capabilities.max_alignment) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "ROCr reservation alignment exceeds native guarantee");
  }
  const iree_device_size_t granularity =
      iree_max(options->mapping_granularity, capabilities.mapping_granularity);
  if (options->byte_length % granularity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ROCr reservation length is not granule aligned");
  }
  const uint32_t scope_count = iree_hal_device_group_memory_scope_count(group);
  iree_hal_memory_contract_t* contract = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_memory_contract_create(
      iree_hal_device_group_memory_domain(group), scope_count,
      iree_hal_amdgpu_buffer_binding_layout(), host_allocator, &contract));
  iree_hal_amdgpu_virtual_buffer_t* buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*buffer) + scope_count * sizeof(uint32_t),
      (void**)&buffer);
  if (iree_status_is_ok(status)) {
    buffer->backend = backend;
    buffer->affinity = affinity;
    status = iree_hal_allocator_virtual_memory_reserve(
        backend->allocator, affinity, options->byte_length,
        &buffer->reservation);
  }
  if (iree_status_is_ok(status)) {
    memset(buffer->projections, 0xFF, scope_count * sizeof(uint32_t));
    buffer->native = *iree_hal_amdgpu_buffer_native(buffer->reservation);
    // Virtual reservations promise only the atomic capabilities explicitly
    // qualified for arbitrary mapped sources; this adapter admits none.
    buffer->native.atomic_memory_cells = 0;
    for (iree_host_size_t i = 0; i < scope.family_count; ++i) {
      const iree_hal_pool_family_access_t* access = &scope.families[i];
      contract->allowed_usage |= access->usage;
      const uint32_t id = access->family->memory.queue_scope_id;
      contract->scopes[id].usage = access->usage;
      contract->scopes[id].interfaces =
          1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
      contract->scopes[id].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] =
          IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS;
      contract->scopes[id + 1] = contract->scopes[id];
      buffer->projections[id] = buffer->projections[id + 1] = 0;
    }
    contract->allowed_access =
        iree_hal_buffer_usage_access(contract->allowed_usage);
    buffer->backing = (iree_hal_buffer_backing_facts_t){
        .notification = backend->physical_devices[0]->memory_notification,
        .tracker =
            iree_hal_device_topology_info(backend->device)->frontier.tracker,
        .allocation_alignment = capabilities.reservation_granularity,
        .maintenance_alignment = 1,
        .maintenance_range = {.byte_length = options->byte_length},
    };
    iree_slim_mutex_initialize(&buffer->mutex);
    iree_hal_virtual_buffer_initialize(
        contract, buffer->native.bindings, &buffer->backing,
        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE, options->byte_length, granularity,
        capabilities.operations, IREE_HAL_MEMORY_BACKEND_ROCR,
        &iree_hal_amdgpu_virtual_buffer_vtable, host_allocator, &buffer->base);
    buffer->base.projection_count = 1;
    buffer->base.projections = buffer->projections;
    *out_buffer = &buffer->base.buffer;
  } else {
    iree_allocator_free(host_allocator, buffer);
  }
  iree_hal_memory_contract_release(contract);
  return status;
}

static iree_status_t iree_hal_amdgpu_memory_mapping_execute(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  (void)args;
  (void)context;
  iree_hal_memory_mapping_batch_t* batch = user_data;
  if (!batch) {
    return iree_ok_status();
  }
  iree_status_t status = iree_hal_memory_mapping_batch_resolve(batch);
  for (iree_host_size_t i = 0;
       i < batch->entry_count && iree_status_is_ok(status); ++i) {
    if (!iree_hal_resource_is(&batch->entries[i].target->buffer,
                              &iree_hal_amdgpu_virtual_buffer_vtable)) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "ROCr queues require ROCr virtual buffers");
    }
  }
  for (iree_host_size_t i = 0;
       i < batch->entry_count && iree_status_is_ok(status); ++i) {
    const iree_hal_memory_mapping_batch_entry_t* entry = &batch->entries[i];
    status = iree_hal_amdgpu_virtual_buffer_update(
        &entry->target->buffer, entry->projection, entry->update_count,
        entry->updates);
  }
  if (iree_status_is_ok(status) && batch->entry_count) {
    const iree_hal_amdgpu_memory_backend_t* backend =
        ((iree_hal_amdgpu_virtual_buffer_t*)batch->entries[0].target)->backend;
    status = iree_hal_amdgpu_virtual_memory_publish(
        backend->libhsa,
        backend->physical_devices[0]->coarse_block_pools.large.memory_pool,
        backend->topology->gpu_agent_count, backend->topology->gpu_agents);
    if (!iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < batch->entry_count; ++i) {
        iree_hal_amdgpu_virtual_buffer_t* buffer =
            (iree_hal_amdgpu_virtual_buffer_t*)batch->entries[i].target;
        iree_slim_mutex_lock(&buffer->mutex);
        if (!buffer->failure_code) {
          buffer->failure_code = iree_status_code(status);
        }
        iree_slim_mutex_unlock(&buffer->mutex);
      }
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_memory_mapping_queue_map(
    iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_host_size_t mapping_count,
    const iree_hal_memory_mapping_t* mappings,
    iree_hal_queue_map_flags_t flags) {
  iree_hal_memory_mapping_batch_t* batch = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_memory_mapping_batch_create(
      queue, mapping_count, mappings, flags,
      iree_hal_device_host_allocator(
          iree_hal_queue_family_device(iree_hal_queue_family(queue))),
      &batch));
  const iree_hal_host_call_t call = iree_hal_make_host_call_with_resource(
      iree_hal_amdgpu_memory_mapping_execute, batch,
      batch ? &batch->resource : NULL);
  const uint64_t args[4] = {0};
  iree_status_t status =
      iree_hal_queue_host_call(queue, waits, signals, call, args, 0);
  iree_hal_memory_mapping_batch_release(batch);
  return status;
}
