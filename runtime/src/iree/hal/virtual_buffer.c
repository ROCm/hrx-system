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

IREE_API_EXPORT iree_status_t iree_hal_virtual_buffer_query_capabilities(
    iree_hal_device_group_t* group, iree_hal_pool_scope_t scope,
    iree_allocator_t host_allocator,
    iree_hal_virtual_buffer_capabilities_t* out_capabilities) {
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  if (!group) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reservation group is null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_pool_scope_validate(group, scope));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < iree_hal_device_group_device_count(group) &&
       iree_status_is_ok(status) && !out_capabilities->max_byte_length;
       ++i) {
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (backend && backend->query_virtual_buffer) {
      status = backend->query_virtual_buffer(backend, group, scope,
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
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iree_hal_device_group_device_count(group) &&
                               iree_status_is_ok(status) && !*out_buffer;
       ++i) {
    const iree_hal_memory_backend_t* backend = iree_hal_device_memory_backend(
        iree_hal_device_group_device_at(group, i));
    if (backend && backend->create_virtual_buffer) {
      status = backend->create_virtual_buffer(backend, group, *scope, options,
                                              host_allocator, out_buffer);
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
    const iree_hal_virtual_buffer_options_t* options,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  if (!group || !options || !options->byte_length ||
      !iree_device_size_is_valid_alignment(options->min_alignment) ||
      !iree_device_size_is_valid_alignment(options->mapping_granularity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid reservation group or geometry");
  }
  IREE_RETURN_IF_ERROR(iree_hal_pool_scope_validate(group, scope));
  const iree_hal_virtual_buffer_create_callback_t leaf = {
      .fn = iree_hal_virtual_buffer_create_native,
  };
  return iree_hal_device_group_create_virtual_buffer(
      group, &scope, options, leaf, host_allocator, out_buffer);
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
      iree_hal_buffer_release(batch->mappings[i].target);
    }
  }
  iree_allocator_free(batch->host_allocator, batch);
}

static const iree_hal_resource_vtable_t iree_hal_memory_mapping_batch_vtable = {
    .destroy = iree_hal_memory_mapping_batch_destroy,
};

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
       mapping->source) ||
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
  if (!mapping->source) {
    if (mapping->source_offset || mapping->access) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "removal requires zero source offset and access");
    }
  } else {
    if (iree_any_bit_set(mapping->access, ~IREE_HAL_MEMORY_ACCESS_ALL) ||
        !iree_all_bits_set(mapping->source->allowed_access, mapping->access) ||
        !iree_all_bits_set(mapping->target->allowed_access, mapping->access)) {
      return iree_make_status(
          IREE_STATUS_PERMISSION_DENIED,
          "mapping access exceeds source or target permissions");
    }
    if (mapping->source_offset > mapping->source->byte_length ||
        range.byte_length >
            mapping->source->byte_length - mapping->source_offset) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "mapping source range is outside its logical view");
    }
    const iree_hal_memory_contract_t* source = mapping->source->memory.contract;
    if (!source || source->domain != domain) {
      return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                              "source has no mapping contract in this group");
    }
    const uint32_t begin = mapping->scope.id ? mapping->scope.id : 1;
    const uint32_t end = mapping->scope.id ? begin + 1 : contract->scope_count;
    for (uint32_t id = begin; id < end; ++id) {
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
  const iree_hal_memory_contract_t* source_contract =
      mapping->source ? mapping->source->memory.contract : NULL;
  if (source_contract &&
      source_contract->mapping_source.type != target->source_type) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "source and target use different native mapping formats");
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
    if (mapping->source && phase == IREE_HAL_MEMORY_MAPPING_PHASE_COMMIT) {
      const iree_hal_buffer_memory_view_t memory =
          iree_hal_buffer_memory_view(mapping->source);
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
            source.mapping_source.offset >
                UINT64_MAX - mapping->source_offset ||
            !iree_device_size_has_alignment(
                source.mapping_source.offset + mapping->source_offset,
                source.mapping_source.granularity) ||
            !iree_device_size_has_alignment(
                mapping->target_range.byte_length,
                source.mapping_source.granularity)) {
          status = iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "native mapping source is unavailable or misaligned");
        } else {
          source.mapping_source.offset += mapping->source_offset;
          resolved->update.source = source;
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
    for (iree_host_size_t i = 0; i < batch->base.update_count; ++i) {
      const iree_hal_memory_mapping_resolved_t* resolved = &batch->resolved[i];
      batch->updates[i] = resolved->update;
      if (!i || resolved->target != batch->resolved[i - 1].target ||
          resolved->projection != batch->resolved[i - 1].projection) {
        entries[batch->base.entry_count++] =
            (iree_hal_memory_mapping_batch_entry_t){
                .target = resolved->target,
                .projection = resolved->projection,
                .updates = &batch->updates[i],
            };
      }
      ++entries[batch->base.entry_count - 1].update_count;
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
