// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/device_spec_builder.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_device_spec_builder_t
//===----------------------------------------------------------------------===//

struct iree_hal_device_spec_builder_storage_t {
  // Parameters assembled from copied builder facets.
  iree_hal_device_spec_params_t params;
  // Builder-owned logical device identity facet.
  iree_hal_device_identity_spec_t identity;
  // Builder-owned memory capability facet.
  iree_hal_device_memory_spec_t memory;
  // Builder-owned virtual memory capability facet.
  iree_hal_device_virtual_memory_spec_t virtual_memory;
  // Builder-owned queue capability facet.
  iree_hal_device_queue_spec_t queues;
  // Builder-owned dispatch capability facet.
  iree_hal_device_dispatch_spec_t dispatch;
  // Builder-owned timing and profiling capability facet.
  iree_hal_device_timing_spec_t timing;
  // Builder-owned executable capability facet.
  iree_hal_device_executable_spec_t executables;
  // Builder-owned sanitizer configuration facet.
  iree_hal_device_sanitizer_spec_t sanitizer;
  // Builder-owned physical device identity records.
  iree_hal_physical_device_spec_t* physical_devices;
  // Builder-owned memory heap records.
  iree_hal_memory_heap_spec_t* memory_heaps;
  // Builder-owned memory type records.
  iree_hal_memory_type_spec_t* memory_types;
  // Builder-owned external buffer handle records.
  iree_hal_external_buffer_handle_spec_t* external_buffer_handles;
  // Builder-owned virtual memory class records.
  iree_hal_virtual_memory_class_spec_t* virtual_memory_classes;
  // Builder-owned queue family records.
  iree_hal_queue_family_spec_t* queue_families;
  // Builder-owned external timepoint handle records.
  iree_hal_external_timepoint_handle_spec_t* external_timepoint_handles;
  // Builder-owned executable target records.
  iree_hal_executable_target_t* executable_targets;
  // Builder-owned driver-local extension facets.
  iree_hal_device_spec_facet_t* facets;
};

static iree_status_t iree_hal_device_spec_builder_ensure_storage(
    iree_hal_device_spec_builder_t* builder) {
  if (builder->storage) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(builder->host_allocator,
                                             sizeof(*builder->storage),
                                             (void**)&builder->storage));
  memset(builder->storage, 0, sizeof(*builder->storage));
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_builder_copy_array(
    iree_allocator_t host_allocator, iree_host_size_t count,
    iree_host_size_t element_size, const void* source, void** out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = NULL;
  if (!count) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(host_allocator, count,
                                                   element_size, out_target));
  memcpy(*out_target, source, count * element_size);
  return iree_ok_status();
}

static iree_status_t iree_hal_device_spec_builder_copy_string(
    iree_allocator_t host_allocator, iree_string_view_t source,
    iree_string_view_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  char* target = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, source.size, (void**)&target));
  memcpy(target, source.data, source.size);
  *out_target = iree_make_string_view(target, source.size);
  return iree_ok_status();
}

static void iree_hal_device_spec_builder_free_string(
    iree_allocator_t host_allocator, iree_string_view_t value) {
  iree_allocator_free(host_allocator, (void*)value.data);
}

static iree_status_t iree_hal_device_spec_builder_copy_bytes(
    iree_allocator_t host_allocator, iree_const_byte_span_t source,
    iree_const_byte_span_t* out_target) {
  IREE_ASSERT_ARGUMENT(out_target);
  *out_target = iree_const_byte_span_empty();
  if (iree_const_byte_span_is_empty(source)) return iree_ok_status();
  uint8_t* target = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, source.data_length, (void**)&target));
  memcpy(target, source.data, source.data_length);
  *out_target = iree_make_const_byte_span(target, source.data_length);
  return iree_ok_status();
}

static void iree_hal_device_spec_builder_free_bytes(
    iree_allocator_t host_allocator, iree_const_byte_span_t value) {
  iree_allocator_free(host_allocator, (void*)value.data);
}

static void iree_hal_device_spec_builder_reset_identity(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  iree_hal_device_identity_spec_t* identity = &builder->storage->identity;
  iree_hal_device_spec_builder_free_string(host_allocator,
                                           identity->logical_device_id);
  iree_hal_device_spec_builder_free_string(host_allocator,
                                           identity->display_name);
  iree_hal_device_spec_builder_free_string(host_allocator, identity->driver_id);
  iree_hal_device_spec_builder_free_string(host_allocator,
                                           identity->driver_version);
  iree_hal_device_spec_builder_free_string(host_allocator,
                                           identity->backend_id);
  iree_hal_device_spec_builder_free_string(host_allocator,
                                           identity->device_path);
  iree_hal_device_spec_builder_free_string(host_allocator,
                                           identity->vendor_name);
  if (builder->storage->physical_devices) {
    for (iree_host_size_t i = 0; i < identity->physical_device_count; ++i) {
      iree_hal_device_spec_builder_free_string(
          host_allocator,
          builder->storage->physical_devices[i].identity.display_name);
      iree_hal_device_spec_builder_free_string(
          host_allocator,
          builder->storage->physical_devices[i].identity.backend_path);
    }
  }
  iree_allocator_free(host_allocator, builder->storage->physical_devices);
  builder->storage->physical_devices = NULL;
  memset(identity, 0, sizeof(*identity));
  builder->storage->params.identity = NULL;
}

static void iree_hal_device_spec_builder_reset_memory(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  if (builder->storage->memory_heaps) {
    for (iree_host_size_t i = 0; i < builder->storage->memory.heap_count; ++i) {
      iree_hal_device_spec_builder_free_string(
          host_allocator, builder->storage->memory_heaps[i].name);
    }
  }
  iree_allocator_free(host_allocator, builder->storage->memory_heaps);
  iree_allocator_free(host_allocator, builder->storage->memory_types);
  iree_allocator_free(host_allocator,
                      builder->storage->external_buffer_handles);
  builder->storage->memory_heaps = NULL;
  builder->storage->memory_types = NULL;
  builder->storage->external_buffer_handles = NULL;
  memset(&builder->storage->memory, 0, sizeof(builder->storage->memory));
  builder->storage->params.memory = NULL;
}

static void iree_hal_device_spec_builder_reset_virtual_memory(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  iree_allocator_free(builder->host_allocator,
                      builder->storage->virtual_memory_classes);
  builder->storage->virtual_memory_classes = NULL;
  memset(&builder->storage->virtual_memory, 0,
         sizeof(builder->storage->virtual_memory));
  builder->storage->params.virtual_memory = NULL;
}

static void iree_hal_device_spec_builder_reset_queues(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  if (builder->storage->queue_families) {
    for (iree_host_size_t i = 0; i < builder->storage->queues.family_count;
         ++i) {
      iree_hal_device_spec_builder_free_string(
          host_allocator, builder->storage->queue_families[i].name);
    }
  }
  iree_allocator_free(host_allocator, builder->storage->queue_families);
  iree_allocator_free(host_allocator,
                      builder->storage->external_timepoint_handles);
  builder->storage->queue_families = NULL;
  builder->storage->external_timepoint_handles = NULL;
  memset(&builder->storage->queues, 0, sizeof(builder->storage->queues));
  builder->storage->params.queues = NULL;
}

static void iree_hal_device_spec_builder_reset_dispatch(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  memset(&builder->storage->dispatch, 0, sizeof(builder->storage->dispatch));
  builder->storage->params.dispatch = NULL;
}

static void iree_hal_device_spec_builder_reset_timing(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  memset(&builder->storage->timing, 0, sizeof(builder->storage->timing));
  builder->storage->params.timing = NULL;
}

static void iree_hal_device_spec_builder_reset_executables(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  if (builder->storage->executable_targets) {
    for (iree_host_size_t i = 0; i < builder->storage->executables.target_count;
         ++i) {
      iree_hal_executable_target_t* target =
          &builder->storage->executable_targets[i];
      iree_hal_device_spec_builder_free_string(host_allocator, target->family);
      iree_hal_device_spec_builder_free_string(host_allocator,
                                               target->target_key);
    }
  }
  iree_allocator_free(host_allocator, builder->storage->executable_targets);
  builder->storage->executable_targets = NULL;
  memset(&builder->storage->executables, 0,
         sizeof(builder->storage->executables));
  builder->storage->params.executables = NULL;
}

static void iree_hal_device_spec_builder_reset_sanitizer(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  memset(&builder->storage->sanitizer, 0, sizeof(builder->storage->sanitizer));
  builder->storage->params.sanitizer = NULL;
}

static void iree_hal_device_spec_builder_reset_facets(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder->storage) return;
  iree_allocator_t host_allocator = builder->host_allocator;
  if (builder->storage->facets) {
    for (iree_host_size_t i = 0; i < builder->storage->params.facet_count;
         ++i) {
      iree_hal_device_spec_builder_free_string(
          host_allocator, builder->storage->facets[i].schema_id);
      iree_hal_device_spec_builder_free_bytes(
          host_allocator, builder->storage->facets[i].payload);
    }
  }
  iree_allocator_free(host_allocator, builder->storage->facets);
  builder->storage->facets = NULL;
  builder->storage->params.facet_count = 0;
  builder->storage->params.facets = NULL;
}

static void iree_hal_device_spec_builder_reset_all(
    iree_hal_device_spec_builder_t* builder) {
  iree_hal_device_spec_builder_reset_identity(builder);
  iree_hal_device_spec_builder_reset_memory(builder);
  iree_hal_device_spec_builder_reset_virtual_memory(builder);
  iree_hal_device_spec_builder_reset_queues(builder);
  iree_hal_device_spec_builder_reset_dispatch(builder);
  iree_hal_device_spec_builder_reset_timing(builder);
  iree_hal_device_spec_builder_reset_executables(builder);
  iree_hal_device_spec_builder_reset_sanitizer(builder);
  iree_hal_device_spec_builder_reset_facets(builder);
}

void iree_hal_device_spec_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_device_spec_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(out_builder);
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->host_allocator = host_allocator;
}

void iree_hal_device_spec_builder_deinitialize(
    iree_hal_device_spec_builder_t* builder) {
  if (!builder) return;
  iree_hal_device_spec_builder_reset_all(builder);
  iree_allocator_free(builder->host_allocator, builder->storage);
  memset(builder, 0, sizeof(*builder));
}

iree_status_t iree_hal_device_spec_builder_set_identity(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_identity_spec_t* identity) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(identity);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_identity(builder);

  iree_status_t status = iree_ok_status();
  iree_hal_device_identity_spec_t* target = &builder->storage->identity;
  *target = *identity;
  status = iree_hal_device_spec_builder_copy_string(builder->host_allocator,
                                                    identity->logical_device_id,
                                                    &target->logical_device_id);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, identity->display_name, &target->display_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, identity->driver_id, &target->driver_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(builder->host_allocator,
                                                      identity->driver_version,
                                                      &target->driver_version);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, identity->backend_id, &target->backend_id);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, identity->device_path, &target->device_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, identity->vendor_name, &target->vendor_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_array(
        builder->host_allocator, identity->physical_device_count,
        sizeof(*builder->storage->physical_devices), identity->physical_devices,
        (void**)&builder->storage->physical_devices);
  }
  target->physical_devices = builder->storage->physical_devices;
  for (iree_host_size_t i = 0;
       i < target->physical_device_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator,
        identity->physical_devices[i].identity.display_name,
        &builder->storage->physical_devices[i].identity.display_name);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_builder_copy_string(
          builder->host_allocator,
          identity->physical_devices[i].identity.backend_path,
          &builder->storage->physical_devices[i].identity.backend_path);
    }
  }
  if (iree_status_is_ok(status)) {
    builder->storage->params.identity = target;
  } else {
    iree_hal_device_spec_builder_reset_identity(builder);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_set_memory(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_memory_spec_t* memory) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(memory);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_memory(builder);

  iree_status_t status = iree_ok_status();
  builder->storage->memory = *memory;
  status = iree_hal_device_spec_builder_copy_array(
      builder->host_allocator, memory->heap_count,
      sizeof(*builder->storage->memory_heaps), memory->heaps,
      (void**)&builder->storage->memory_heaps);
  builder->storage->memory.heaps = builder->storage->memory_heaps;
  for (iree_host_size_t i = 0;
       i < memory->heap_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, memory->heaps[i].name,
        &builder->storage->memory_heaps[i].name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_array(
        builder->host_allocator, memory->memory_type_count,
        sizeof(*builder->storage->memory_types), memory->memory_types,
        (void**)&builder->storage->memory_types);
  }
  builder->storage->memory.memory_types = builder->storage->memory_types;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_array(
        builder->host_allocator, memory->external_buffer_handle_count,
        sizeof(*builder->storage->external_buffer_handles),
        memory->external_buffer_handles,
        (void**)&builder->storage->external_buffer_handles);
  }
  builder->storage->memory.external_buffer_handles =
      builder->storage->external_buffer_handles;
  if (iree_status_is_ok(status)) {
    builder->storage->params.memory = &builder->storage->memory;
  } else {
    iree_hal_device_spec_builder_reset_memory(builder);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_set_virtual_memory(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_virtual_memory_spec_t* virtual_memory) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(virtual_memory);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_virtual_memory(builder);

  iree_status_t status = iree_ok_status();
  builder->storage->virtual_memory = *virtual_memory;
  status = iree_hal_device_spec_builder_copy_array(
      builder->host_allocator, virtual_memory->class_count,
      sizeof(*builder->storage->virtual_memory_classes),
      virtual_memory->classes,
      (void**)&builder->storage->virtual_memory_classes);
  builder->storage->virtual_memory.classes =
      builder->storage->virtual_memory_classes;
  if (iree_status_is_ok(status)) {
    builder->storage->params.virtual_memory = &builder->storage->virtual_memory;
  } else {
    iree_hal_device_spec_builder_reset_virtual_memory(builder);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_set_queues(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_queue_spec_t* queues) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(queues);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_queues(builder);

  iree_status_t status = iree_ok_status();
  builder->storage->queues = *queues;
  status = iree_hal_device_spec_builder_copy_array(
      builder->host_allocator, queues->family_count,
      sizeof(*builder->storage->queue_families), queues->families,
      (void**)&builder->storage->queue_families);
  builder->storage->queues.families = builder->storage->queue_families;
  for (iree_host_size_t i = 0;
       i < queues->family_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, queues->families[i].name,
        &builder->storage->queue_families[i].name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_array(
        builder->host_allocator, queues->external_timepoint_handle_count,
        sizeof(*builder->storage->external_timepoint_handles),
        queues->external_timepoint_handles,
        (void**)&builder->storage->external_timepoint_handles);
  }
  builder->storage->queues.external_timepoint_handles =
      builder->storage->external_timepoint_handles;
  if (iree_status_is_ok(status)) {
    builder->storage->params.queues = &builder->storage->queues;
  } else {
    iree_hal_device_spec_builder_reset_queues(builder);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_set_dispatch(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_dispatch_spec_t* dispatch) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(dispatch);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_dispatch(builder);
  builder->storage->dispatch = *dispatch;
  builder->storage->params.dispatch = &builder->storage->dispatch;
  return iree_ok_status();
}

iree_status_t iree_hal_device_spec_builder_set_timing(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_timing_spec_t* timing) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(timing);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_timing(builder);
  builder->storage->timing = *timing;
  builder->storage->params.timing = &builder->storage->timing;
  return iree_ok_status();
}

iree_status_t iree_hal_device_spec_builder_set_executables(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_executable_spec_t* executables) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(executables);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_executables(builder);

  iree_status_t status = iree_ok_status();
  builder->storage->executables = *executables;
  status = iree_hal_device_spec_builder_copy_array(
      builder->host_allocator, executables->target_count,
      sizeof(*builder->storage->executable_targets), executables->targets,
      (void**)&builder->storage->executable_targets);
  builder->storage->executables.targets = builder->storage->executable_targets;
  for (iree_host_size_t i = 0;
       i < executables->target_count && iree_status_is_ok(status); ++i) {
    const iree_hal_executable_target_t* source = &executables->targets[i];
    iree_hal_executable_target_t* target =
        &builder->storage->executable_targets[i];
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, source->family, &target->family);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_spec_builder_copy_string(
          builder->host_allocator, source->target_key, &target->target_key);
    }
  }
  if (iree_status_is_ok(status)) {
    builder->storage->params.executables = &builder->storage->executables;
  } else {
    iree_hal_device_spec_builder_reset_executables(builder);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_add_executable_target(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_executable_target_t* target) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(target);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));

  iree_hal_device_executable_spec_t* executables =
      &builder->storage->executables;
  for (iree_host_size_t i = 0; i < executables->target_count; ++i) {
    iree_hal_executable_target_t* existing_target =
        &builder->storage->executable_targets[i];
    if (existing_target->kind != target->kind ||
        !iree_string_view_equal(existing_target->family, target->family) ||
        !iree_string_view_equal(existing_target->target_key,
                                target->target_key)) {
      continue;
    }
    if (existing_target->priority != target->priority) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "duplicate executable target '%.*s:%.*s' has priorities %" PRIu32
          " and %" PRIu32,
          (int)target->family.size, target->family.data,
          (int)target->target_key.size, target->target_key.data,
          existing_target->priority, target->priority);
    }
    existing_target->physical_device_affinity |=
        target->physical_device_affinity;
    existing_target->flags |= target->flags;
    return iree_ok_status();
  }

  if (executables->target_count == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec executable target count overflow");
  }
  const iree_host_size_t new_target_count = executables->target_count + 1;
  iree_hal_executable_target_t* new_targets = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(builder->host_allocator, new_target_count,
                                  sizeof(*new_targets), (void**)&new_targets));
  if (executables->target_count != 0) {
    memcpy(new_targets, builder->storage->executable_targets,
           executables->target_count * sizeof(*new_targets));
  }

  iree_hal_executable_target_t* new_target =
      &new_targets[executables->target_count];
  *new_target = *target;
  new_target->family = iree_string_view_empty();
  new_target->target_key = iree_string_view_empty();
  iree_status_t status = iree_hal_device_spec_builder_copy_string(
      builder->host_allocator, target->family, &new_target->family);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_string(
        builder->host_allocator, target->target_key, &new_target->target_key);
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(builder->host_allocator,
                        builder->storage->executable_targets);
    builder->storage->executable_targets = new_targets;
    executables->target_count = new_target_count;
    executables->targets = new_targets;
    builder->storage->params.executables = executables;
  } else {
    iree_hal_device_spec_builder_free_string(builder->host_allocator,
                                             new_target->family);
    iree_hal_device_spec_builder_free_string(builder->host_allocator,
                                             new_target->target_key);
    iree_allocator_free(builder->host_allocator, new_targets);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_set_sanitizer(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_sanitizer_spec_t* sanitizer) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(sanitizer);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));
  iree_hal_device_spec_builder_reset_sanitizer(builder);
  builder->storage->sanitizer = *sanitizer;
  builder->storage->params.sanitizer = &builder->storage->sanitizer;
  return iree_ok_status();
}

iree_status_t iree_hal_device_spec_builder_add_facet(
    iree_hal_device_spec_builder_t* builder,
    const iree_hal_device_spec_facet_t* facet) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(facet);
  IREE_RETURN_IF_ERROR(iree_hal_device_spec_builder_ensure_storage(builder));

  iree_host_size_t old_count = builder->storage->params.facet_count;
  if (old_count == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device spec facet count overflow");
  }
  iree_hal_device_spec_facet_t* new_facets = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(builder->host_allocator, old_count + 1,
                                  sizeof(*new_facets), (void**)&new_facets));
  if (old_count) {
    memcpy(new_facets, builder->storage->facets,
           old_count * sizeof(*new_facets));
  }

  iree_status_t status = iree_ok_status();
  iree_hal_device_spec_facet_t* target = &new_facets[old_count];
  memset(target, 0, sizeof(*target));
  target->schema_version = facet->schema_version;
  status = iree_hal_device_spec_builder_copy_string(
      builder->host_allocator, facet->schema_id, &target->schema_id);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_copy_bytes(
        builder->host_allocator, facet->payload, &target->payload);
  }
  if (iree_status_is_ok(status)) {
    iree_allocator_free(builder->host_allocator, builder->storage->facets);
    builder->storage->facets = new_facets;
    builder->storage->params.facet_count = old_count + 1;
    builder->storage->params.facets = builder->storage->facets;
  } else {
    iree_hal_device_spec_builder_free_string(builder->host_allocator,
                                             target->schema_id);
    iree_hal_device_spec_builder_free_bytes(builder->host_allocator,
                                            target->payload);
    iree_allocator_free(builder->host_allocator, new_facets);
  }
  return status;
}

iree_status_t iree_hal_device_spec_builder_finalize(
    iree_hal_device_spec_builder_t* builder,
    iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_spec);
  const iree_hal_device_spec_params_t* params =
      builder->storage ? &builder->storage->params : NULL;
  return iree_hal_device_spec_create(params, builder->host_allocator, out_spec);
}

iree_status_t iree_hal_device_spec_create_minimal(
    iree_string_view_t logical_device_id, iree_string_view_t display_name,
    iree_string_view_t driver_id, iree_string_view_t backend_id,
    iree_allocator_t host_allocator, iree_hal_device_spec_t** out_spec) {
  IREE_ASSERT_ARGUMENT(out_spec);
  *out_spec = NULL;

  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(host_allocator, &builder);
  iree_hal_device_identity_spec_t identity = {
      .logical_device_id = logical_device_id,
      .display_name = display_name,
      .driver_id = driver_id,
      .backend_id = backend_id,
      .flags = IREE_HAL_DEVICE_IDENTITY_FLAG_NONE,
  };
  iree_status_t status =
      iree_hal_device_spec_builder_set_identity(&builder, &identity);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, out_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  return status;
}
