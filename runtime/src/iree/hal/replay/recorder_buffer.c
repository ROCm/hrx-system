// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_buffer.h"

#include <string.h>

#include "iree/hal/buffer_allocation.h"
#include "iree/hal/replay/recorder_record.h"

#define IREE_HAL_REPLAY_VTABLE_DISPATCH(resource, type_prefix, method_name) \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource))  \
       ->vtable)                                                            \
      ->method_name

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_buffer_t
//===----------------------------------------------------------------------===//

struct iree_hal_replay_recorder_buffer_t {
  // HAL buffer header for the recording wrapper buffer.
  iree_hal_buffer_t base;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying buffer receiving forwarded HAL calls.
  iree_hal_buffer_t* base_buffer;
  // Session-local device object id associated with this buffer.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this buffer.
  iree_hal_replay_object_id_t buffer_id;
  // Recorder allocator ID for virtual reservations, or NONE for ordinary
  // buffers.
  iree_hal_replay_object_id_t virtual_memory_allocator_id;
  // One scoped host mapping owned by this proxy, or zero for an ordinary view.
  struct {
    // Session-local identity independent from the original buffer's lifetime.
    iree_hal_replay_object_id_t id;
    // Borrowed caller mapping valid until its consuming unmap.
    iree_hal_buffer_mapping_t* value;
    // Offset in the original captured buffer at the mapped range's start.
    iree_device_size_t recorded_origin;
    // Offset in the native mapped buffer at the same byte.
    iree_device_size_t native_origin;
  } mapping;
};

static const iree_hal_buffer_vtable_t iree_hal_replay_recorder_buffer_vtable;
static const iree_hal_buffer_vtable_t
    iree_hal_replay_recorder_allocated_buffer_vtable;

static bool iree_hal_replay_recorder_buffer_isa(
    iree_hal_buffer_t* base_buffer) {
  return iree_hal_resource_is(base_buffer,
                              &iree_hal_replay_recorder_buffer_vtable) ||
         iree_hal_resource_is(
             base_buffer, &iree_hal_replay_recorder_allocated_buffer_vtable);
}

static iree_hal_replay_recorder_buffer_t* iree_hal_replay_recorder_buffer_cast(
    iree_hal_buffer_t* base_buffer) {
  IREE_ASSERT(iree_hal_replay_recorder_buffer_isa(base_buffer));
  return (iree_hal_replay_recorder_buffer_t*)base_buffer;
}

static iree_device_size_t iree_hal_replay_recorder_buffer_recorded_offset(
    const iree_hal_replay_recorder_buffer_t* buffer,
    iree_device_size_t native_offset) {
  return buffer->mapping.id
             ? buffer->mapping.recorded_origin + native_offset -
                   buffer->mapping.native_origin
             : native_offset - iree_hal_buffer_byte_offset(&buffer->base);
}

void iree_hal_replay_recorder_buffer_make_object_payload(
    iree_hal_buffer_t* base_buffer,
    iree_hal_replay_buffer_object_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(base_buffer);
  out_payload->allocation_size = iree_hal_buffer_allocation_size(base_buffer);
  out_payload->byte_offset = iree_hal_buffer_byte_offset(base_buffer);
  out_payload->byte_length = iree_hal_buffer_byte_length(base_buffer);
  out_payload->queue_family_affinity = placement.queue_family_affinity;
  out_payload->placement_flags = placement.flags;
  out_payload->memory_type = iree_hal_buffer_memory_type(base_buffer);
  out_payload->allowed_access = iree_hal_buffer_allowed_access(base_buffer);
  out_payload->allowed_usage = iree_hal_buffer_allowed_usage(base_buffer);
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_buffer_id_or_none(
    iree_hal_buffer_t* buffer) {
  if (!buffer) {
    return IREE_HAL_REPLAY_OBJECT_ID_NONE;
  }
  if (iree_hal_replay_recorder_buffer_isa(buffer)) {
    return iree_hal_replay_recorder_buffer_cast(buffer)->buffer_id;
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    return iree_hal_replay_recorder_buffer_cast(allocated_buffer)->buffer_id;
  }
  return IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_find_buffer_id(
    iree_hal_replay_recorder_t* recorder, iree_hal_buffer_t* buffer) {
  if (!buffer) {
    return IREE_HAL_REPLAY_OBJECT_ID_NONE;
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    return IREE_HAL_REPLAY_OBJECT_ID_NONE;
  }
  iree_hal_replay_recorder_buffer_t* replay_buffer =
      iree_hal_replay_recorder_buffer_cast(allocated_buffer);
  return replay_buffer->recorder == recorder ? replay_buffer->buffer_id
                                             : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

void iree_hal_replay_recorder_buffer_ref_make_payload(
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_replay_buffer_ref_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->offset = buffer_ref.offset;
  out_payload->length = buffer_ref.length;
  if (buffer_ref.buffer) {
    out_payload->buffer_id =
        iree_hal_replay_recorder_buffer_id_or_none(buffer_ref.buffer);
    out_payload->offset += iree_hal_buffer_byte_offset(buffer_ref.buffer);
    iree_hal_buffer_t* root =
        iree_hal_buffer_allocated_buffer(buffer_ref.buffer);
    if (iree_hal_replay_recorder_buffer_isa(root)) {
      out_payload->offset = iree_hal_replay_recorder_buffer_recorded_offset(
          iree_hal_replay_recorder_buffer_cast(root), out_payload->offset);
    }
  } else {
    out_payload->buffer_slot = buffer_ref.buffer_slot;
  }
}

iree_status_t iree_hal_replay_recorder_buffer_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t buffer_id,
    iree_hal_replay_object_id_t virtual_memory_allocator_id,
    iree_hal_device_t* placement_device, iree_hal_buffer_t* base_buffer,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);

  if (iree_hal_replay_recorder_buffer_isa(base_buffer)) {
    iree_hal_buffer_retain(base_buffer);
    *out_buffer = base_buffer;
    return iree_ok_status();
  }

  iree_hal_replay_recorder_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_buffer_allocate_proxy(host_allocator, &buffer));
  *out_buffer = iree_hal_replay_recorder_buffer_initialize_proxy(
      recorder, device_id, buffer_id, virtual_memory_allocator_id,
      placement_device, base_buffer, host_allocator, buffer);
  return iree_ok_status();
}

iree_status_t iree_hal_replay_recorder_buffer_allocate_proxy(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  iree_hal_replay_recorder_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  *out_buffer = buffer;
  return iree_ok_status();
}

void iree_hal_replay_recorder_buffer_free_proxy(
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_buffer_t* buffer) {
  iree_allocator_free(host_allocator, buffer);
}

iree_hal_buffer_t* iree_hal_replay_recorder_buffer_initialize_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t buffer_id,
    iree_hal_replay_object_id_t virtual_memory_allocator_id,
    iree_hal_device_t* placement_device, iree_hal_buffer_t* base_buffer,
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_FALSE(iree_hal_replay_recorder_buffer_isa(base_buffer));

  iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(base_buffer);
  if (placement.device) {
    placement.device = placement_device;
  }
  iree_hal_buffer_initialize(
      placement, &buffer->base, iree_hal_buffer_allocation_size(base_buffer),
      iree_hal_buffer_byte_offset(base_buffer),
      iree_hal_buffer_byte_length(base_buffer),
      iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allowed_access(base_buffer),
      iree_hal_buffer_allowed_usage(base_buffer),
      iree_hal_buffer_allocation_vtable(base_buffer)
          ? &iree_hal_replay_recorder_allocated_buffer_vtable
          : &iree_hal_replay_recorder_buffer_vtable,
      &buffer->base);
  buffer->base.memory = base_buffer->memory;
  buffer->base.host_binding_index = base_buffer->host_binding_index;
  buffer->host_allocator = host_allocator;
  buffer->recorder = recorder;
  iree_hal_replay_recorder_retain(buffer->recorder);
  buffer->base_buffer = base_buffer;
  iree_hal_buffer_retain(buffer->base_buffer);
  buffer->device_id = device_id;
  buffer->buffer_id = buffer_id;
  buffer->virtual_memory_allocator_id = virtual_memory_allocator_id;

  return &buffer->base;
}

iree_status_t iree_hal_replay_recorder_buffer_resolve_virtual_memory(
    iree_hal_buffer_t* buffer, iree_hal_replay_recorder_t* expected_recorder,
    iree_hal_replay_object_id_t expected_allocator_id,
    iree_hal_replay_object_id_t* out_buffer_id,
    iree_hal_buffer_t** out_base_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer_id);
  IREE_ASSERT_ARGUMENT(out_base_buffer);
  *out_buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  *out_base_buffer = NULL;
  if (IREE_UNLIKELY(!buffer || !iree_hal_replay_recorder_buffer_isa(buffer))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "virtual memory reservation was not created by this replay recorder "
        "allocator");
  }
  iree_hal_replay_recorder_buffer_t* replay_buffer =
      iree_hal_replay_recorder_buffer_cast(buffer);
  if (IREE_UNLIKELY(replay_buffer->recorder != expected_recorder ||
                    replay_buffer->virtual_memory_allocator_id !=
                        expected_allocator_id ||
                    !replay_buffer->base_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "virtual memory reservation belongs to another replay recorder "
        "allocator or was already released");
  }
  *out_buffer_id = replay_buffer->buffer_id;
  *out_base_buffer = replay_buffer->base_buffer;
  return iree_ok_status();
}

iree_hal_buffer_t* iree_hal_replay_recorder_buffer_base_or_self(
    iree_hal_buffer_t* buffer) {
  return iree_hal_replay_recorder_buffer_isa(buffer)
             ? iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer
             : buffer;
}

// Resolves the native root beneath any internal replay identity adapters.
// Offsets remain absolute within the allocation and are handled by callers.
static iree_hal_buffer_t*
iree_hal_replay_recorder_buffer_unwrap_allocated_buffer(
    iree_hal_buffer_t* buffer) {
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  while (iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    iree_hal_replay_recorder_buffer_t* replay_buffer =
        iree_hal_replay_recorder_buffer_cast(allocated_buffer);
    allocated_buffer =
        iree_hal_buffer_allocated_buffer(replay_buffer->base_buffer);
  }
  return allocated_buffer;
}

void iree_hal_replay_recorder_buffer_consume_virtual_memory(
    iree_hal_buffer_t* buffer) {
  iree_hal_replay_recorder_buffer_t* replay_buffer =
      iree_hal_replay_recorder_buffer_cast(buffer);
  replay_buffer->base_buffer = NULL;
  iree_hal_buffer_release(buffer);
}

iree_status_t iree_hal_replay_recorder_buffer_unwrap_for_call(
    iree_hal_buffer_t* buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_base_buffer,
    iree_hal_buffer_t** out_temporary_buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_base_buffer);
  IREE_ASSERT_ARGUMENT(out_temporary_buffer);
  *out_base_buffer = NULL;
  *out_temporary_buffer = NULL;

  if (iree_hal_replay_recorder_buffer_isa(buffer)) {
    return iree_hal_replay_recorder_buffer_unwrap_for_call(
        iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer,
        host_allocator, out_base_buffer, out_temporary_buffer);
  }

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    *out_base_buffer = buffer;
    return iree_ok_status();
  }

  iree_hal_replay_recorder_buffer_t* replay_allocated_buffer =
      iree_hal_replay_recorder_buffer_cast(allocated_buffer);
  iree_hal_buffer_t* base_allocated_buffer =
      iree_hal_replay_recorder_buffer_unwrap_allocated_buffer(
          replay_allocated_buffer->base_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(
      base_allocated_buffer, iree_hal_buffer_byte_offset(buffer),
      iree_hal_buffer_byte_length(buffer), host_allocator,
      out_temporary_buffer));
  *out_base_buffer = *out_temporary_buffer;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_recorder_buffer_ref_unwrap_for_call(
    iree_hal_buffer_ref_t* inout_ref) {
  IREE_ASSERT_ARGUMENT(inout_ref);

  // Binding tables may contain unused slots. Preserve them while unwrapping
  // the populated slots for the underlying device call.
  if (!inout_ref->buffer) {
    return iree_ok_status();
  }

  if (iree_hal_replay_recorder_buffer_isa(inout_ref->buffer)) {
    inout_ref->buffer =
        iree_hal_replay_recorder_buffer_cast(inout_ref->buffer)->base_buffer;
    return iree_hal_replay_recorder_buffer_ref_unwrap_for_call(inout_ref);
  }

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(inout_ref->buffer);
  if (!iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    return iree_ok_status();
  }

  iree_device_size_t base_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(inout_ref->buffer), inout_ref->offset,
          &base_offset))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay buffer reference offset overflow");
  }
  iree_hal_replay_recorder_buffer_t* replay_allocated_buffer =
      iree_hal_replay_recorder_buffer_cast(allocated_buffer);
  inout_ref->buffer = iree_hal_replay_recorder_buffer_unwrap_allocated_buffer(
      replay_allocated_buffer->base_buffer);
  inout_ref->offset = base_offset;
  return iree_ok_status();
}

void iree_hal_replay_recorder_buffer_release_temporary(
    iree_hal_buffer_t* temporary_buffer) {
  iree_hal_buffer_release(temporary_buffer);
}

static iree_status_t iree_hal_replay_recorder_buffer_begin_operation(
    iree_hal_replay_recorder_buffer_t* buffer,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      buffer->recorder, buffer->device_id, buffer->buffer_id,
      buffer->mapping.id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, operation_code,
      payload_type, out_pending_record);
}

static bool iree_hal_replay_recorder_buffer_mapping_has_write_access(
    const iree_hal_buffer_mapping_t* mapping) {
  return iree_all_bits_set(mapping->impl.allowed_access,
                           IREE_HAL_MEMORY_ACCESS_WRITE);
}

static void iree_hal_replay_recorder_buffer_make_range_data_payload(
    const iree_hal_replay_recorder_buffer_t* buffer,
    const iree_hal_buffer_mapping_t* mapping,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_replay_buffer_range_data_payload_t* out_payload,
    iree_const_byte_span_t* out_data) {
  // The public mapping API has already bounded this range within the map.
  memset(out_payload, 0, sizeof(*out_payload));
  const iree_host_size_t mapped_data_offset =
      (iree_host_size_t)(local_byte_offset - mapping->impl.byte_offset);
  out_payload->byte_offset = iree_hal_replay_recorder_buffer_recorded_offset(
      buffer, local_byte_offset);
  out_payload->byte_length = local_byte_length;
  out_payload->data_length = local_byte_length;
  out_payload->memory_access = mapping->impl.allowed_access;
  out_payload->map_flags = mapping->impl.flags;
  *out_data = iree_make_const_byte_span(
      mapping->contents.data ? mapping->contents.data + mapped_data_offset
                             : NULL,
      (iree_host_size_t)local_byte_length);
}

static void iree_hal_replay_recorder_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_release(buffer->base_buffer);
  iree_hal_replay_recorder_release(buffer->recorder);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_replay_recorder_buffer_export_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_hal_external_storage_class_t requested_class,
    iree_hal_external_storage_t* out_storage,
    iree_hal_buffer_range_t* out_logical_range) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  if (buffer->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    return IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                           export_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length,
        requested_class, out_storage, out_logical_range);
  }
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_EXPORT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  iree_hal_replay_recorder_mark_unsupported(&pending_record);
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                      export_range)(
          buffer->base_buffer, local_byte_offset, local_byte_length,
          requested_class, out_storage, out_logical_range));
}

static iree_status_t iree_hal_replay_recorder_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access, iree_hal_buffer_map_flags_t flags,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  if (buffer->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    return IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                           map_range)(
        buffer->base_buffer, mapping_mode, memory_access, flags,
        local_byte_offset, local_byte_length, mapping);
  }
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = iree_hal_replay_recorder_buffer_recorded_offset(
          buffer, local_byte_offset),
      .byte_length = local_byte_length,
      .mapping_mode = mapping_mode,
      .memory_access = memory_access,
      .map_flags = flags,
  };
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span((const uint8_t*)&payload, sizeof(payload));
  iree_hal_replay_object_id_t mapping_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (!mapping->impl.is_persistent) {
    IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
        buffer->recorder, &mapping_id));
  }
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      buffer->recorder, buffer->device_id, buffer->buffer_id, mapping_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE, &pending_record));
  iree_status_t status = iree_ok_status();
  if (mapping->impl.is_persistent &&
      iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL replay capture does not support persistent write mappings");
  }
  iree_hal_replay_recorder_buffer_t* mapped_proxy = NULL;
  if (iree_status_is_ok(status) && !mapping->impl.is_persistent) {
    status = iree_hal_replay_recorder_buffer_allocate_proxy(
        buffer->host_allocator, &mapped_proxy);
  }
  iree_hal_buffer_t* original_mapping_buffer = mapping->buffer;
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer,
                                             iree_hal_buffer, map_range)(
        buffer->base_buffer, mapping_mode, memory_access, flags,
        local_byte_offset, local_byte_length, mapping);
  }
  if (iree_status_is_ok(status) && mapped_proxy) {
    // A scoped map owns its actual native storage independently from the
    // original view. Its proxy also gives flush/unmap an exact capture identity
    // when several maps overlap or the original view has been decommitted.
    iree_hal_buffer_t* retained_buffer = mapping->buffer;
    iree_hal_buffer_t* native_mapped_buffer =
        retained_buffer == original_mapping_buffer ? buffer->base_buffer
                                                   : retained_buffer;
    mapping->buffer = iree_hal_replay_recorder_buffer_initialize_proxy(
        buffer->recorder, buffer->device_id, buffer->buffer_id,
        IREE_HAL_REPLAY_OBJECT_ID_NONE, NULL, native_mapped_buffer,
        buffer->host_allocator, mapped_proxy);
    mapped_proxy->mapping.id = mapping_id;
    mapped_proxy->mapping.value = mapping;
    mapped_proxy->mapping.recorded_origin = payload.byte_offset;
    mapped_proxy->mapping.native_origin = mapping->impl.byte_offset;
    iree_hal_buffer_release(retained_buffer);
    mapped_proxy = NULL;
  }
  iree_hal_replay_recorder_buffer_free_proxy(buffer->host_allocator,
                                             mapped_proxy);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static iree_status_t iree_hal_replay_recorder_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  if (buffer->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    return IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                           unmap_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length, mapping);
  }
  const bool capture_data =
      iree_hal_replay_recorder_buffer_mapping_has_write_access(mapping);
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = iree_hal_replay_recorder_buffer_recorded_offset(
          buffer, local_byte_offset),
      .byte_length = local_byte_length,
      .memory_access = mapping->impl.allowed_access,
      .map_flags = mapping->impl.flags,
  };
  iree_hal_replay_buffer_range_data_payload_t data_payload;
  iree_const_byte_span_t data_span = iree_make_const_byte_span(NULL, 0);
  uint8_t* data_snapshot = NULL;
  iree_status_t status = iree_ok_status();
  if (capture_data) {
    iree_hal_replay_recorder_buffer_make_range_data_payload(
        buffer, mapping, local_byte_offset, local_byte_length, &data_payload,
        &data_span);
    if (data_span.data_length > 0) {
      status =
          iree_allocator_malloc(buffer->host_allocator, data_span.data_length,
                                (void**)&data_snapshot);
      if (iree_status_is_ok(status)) {
        memcpy(data_snapshot, data_span.data, data_span.data_length);
        data_span =
            iree_make_const_byte_span(data_snapshot, data_span.data_length);
      }
    }
  }
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(
          capture_data ? (const uint8_t*)&data_payload
                       : (const uint8_t*)&payload,
          capture_data ? sizeof(data_payload) : sizeof(payload)),
      data_span,
  };
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_try_begin_operation(
        buffer->recorder, buffer->device_id, buffer->buffer_id,
        buffer->mapping.id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
        IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE,
        capture_data ? IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA
                     : IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE,
        &pending_record);
  } else {
    iree_hal_replay_recorder_fail(buffer->recorder, iree_status_code(status));
    // The capture failure is now owned by the recorder and reported at close.
    iree_status_free(status);
  }
  // Public unmap consumes the mapping on every path. Capture cannot prevent
  // native teardown or replace its status with an observation failure.
  status = IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                           unmap_range)(
      buffer->base_buffer, local_byte_offset, local_byte_length, mapping);
  status = iree_hal_replay_recorder_end_passthrough_operation_with_payload(
      &pending_record, status, capture_data ? 2 : 1, iovecs);
  iree_allocator_free(buffer->host_allocator, data_snapshot);
  return status;
}

static iree_status_t iree_hal_replay_recorder_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  if (buffer->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    return IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                           invalidate_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length);
  }
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = iree_hal_replay_recorder_buffer_recorded_offset(
          buffer, local_byte_offset),
      .byte_length = local_byte_length,
  };
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span((const uint8_t*)&payload, sizeof(payload));
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE, &pending_record));
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      buffer->base_buffer, iree_hal_buffer, invalidate_range)(
      buffer->base_buffer, local_byte_offset, local_byte_length);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static iree_status_t iree_hal_replay_recorder_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  if (buffer->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    return IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer, iree_hal_buffer,
                                           flush_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length);
  }
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = iree_hal_replay_recorder_buffer_recorded_offset(
          buffer, local_byte_offset),
      .byte_length = local_byte_length,
  };
  iree_hal_replay_buffer_range_data_payload_t data_payload;
  iree_const_byte_span_t data_span = iree_make_const_byte_span(NULL, 0);
  uint8_t* data_snapshot = NULL;
  iree_hal_replay_recorder_buffer_make_range_data_payload(
      buffer, buffer->mapping.value, local_byte_offset, local_byte_length,
      &data_payload, &data_span);
  iree_status_t status = iree_ok_status();
  if (data_span.data_length > 0) {
    status = iree_allocator_malloc(
        buffer->host_allocator, data_span.data_length, (void**)&data_snapshot);
    if (iree_status_is_ok(status)) {
      memcpy(data_snapshot, data_span.data, data_span.data_length);
      data_span =
          iree_make_const_byte_span(data_snapshot, data_span.data_length);
    }
  }
  const bool capture_data = iree_status_is_ok(status);
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(
          capture_data ? (const uint8_t*)&data_payload
                       : (const uint8_t*)&payload,
          capture_data ? sizeof(data_payload) : sizeof(payload)),
      data_span,
  };
  iree_hal_replay_pending_record_t pending_record;
  iree_status_t begin_status = iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE,
      capture_data ? IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE,
      &pending_record);
  if (!iree_status_is_ok(begin_status)) {
    iree_allocator_free(buffer->host_allocator, data_snapshot);
    return iree_status_join(begin_status, status);
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer,
                                             iree_hal_buffer, flush_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length);
  }
  status = iree_hal_replay_recorder_end_operation_with_payload(
      &pending_record, status, capture_data ? 2 : 1, iovecs);
  iree_allocator_free(buffer->host_allocator, data_snapshot);
  return status;
}

static iree_hal_buffer_memory_view_t
iree_hal_replay_recorder_buffer_query_memory(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_replay_recorder_buffer_t* buffer =
      (const iree_hal_replay_recorder_buffer_t*)base_buffer;
  return iree_hal_buffer_memory_view(buffer->base_buffer);
}

static iree_hal_virtual_buffer_view_t
iree_hal_replay_recorder_buffer_query_virtual(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_replay_recorder_buffer_t* buffer =
      (const iree_hal_replay_recorder_buffer_t*)base_buffer;
  return iree_hal_buffer_virtual_view(buffer->base_buffer);
}

static iree_status_t iree_hal_replay_recorder_buffer_detach_mappings(
    iree_hal_buffer_t* base_buffer, uint32_t scope_id) {
  iree_hal_replay_recorder_buffer_t* buffer =
      (iree_hal_replay_recorder_buffer_t*)base_buffer;
  iree_hal_replay_pending_record_t pending_record = {0};
  iree_hal_replay_recorder_try_begin_operation(
      buffer->recorder, buffer->device_id, buffer->buffer_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      IREE_HAL_REPLAY_OPERATION_CODE_VIRTUAL_BUFFER_DETACH,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_VIRTUAL_BUFFER_DETACH, &pending_record);
  const iree_hal_memory_scope_t scope = {
      .domain = base_buffer->memory.contract->domain, .id = scope_id};
  iree_status_t status =
      iree_hal_virtual_buffer_detach(buffer->base_buffer, scope);
  const iree_hal_replay_virtual_buffer_detach_payload_t payload = {
      .scope_id = scope_id,
  };
  const iree_const_byte_span_t iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));
  return iree_hal_replay_recorder_end_passthrough_operation_with_payload(
      &pending_record, status, 1, &iovec);
}

static iree_hal_buffer_allocation_profile_t
iree_hal_replay_recorder_buffer_allocation_profile(iree_hal_buffer_t* buffer) {
  return iree_hal_buffer_allocation_profile(
      iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer);
}

static iree_status_t iree_hal_replay_recorder_buffer_begin_dealloca(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool) {
  return iree_hal_buffer_allocation_begin_dealloca(
      iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer, out_pool);
}

static void iree_hal_replay_recorder_buffer_abort_dealloca(
    iree_hal_buffer_t* buffer) {
  iree_hal_buffer_allocation_abort_dealloca(
      iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer);
}

static void iree_hal_replay_recorder_buffer_take_dealloca_reservation(
    iree_hal_buffer_t* buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_buffer_allocation_take_dealloca_reservation(
      iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer, out_pool,
      out_reservation);
}

static void iree_hal_replay_recorder_buffer_decommit(
    iree_hal_buffer_t* buffer) {
  iree_hal_buffer_allocation_decommit(
      iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer);
}

static const iree_hal_buffer_allocation_vtable_t
    iree_hal_replay_recorder_buffer_allocation_vtable = {
        .profile = iree_hal_replay_recorder_buffer_allocation_profile,
        .begin_dealloca = iree_hal_replay_recorder_buffer_begin_dealloca,
        .abort_dealloca = iree_hal_replay_recorder_buffer_abort_dealloca,
        .take_dealloca_reservation =
            iree_hal_replay_recorder_buffer_take_dealloca_reservation,
        .decommit = iree_hal_replay_recorder_buffer_decommit,
};

static const iree_hal_buffer_vtable_t iree_hal_replay_recorder_buffer_vtable = {
    .destroy = iree_hal_replay_recorder_buffer_destroy,
    .export_range = iree_hal_replay_recorder_buffer_export_range,
    .map_range = iree_hal_replay_recorder_buffer_map_range,
    .unmap_range = iree_hal_replay_recorder_buffer_unmap_range,
    .invalidate_range = iree_hal_replay_recorder_buffer_invalidate_range,
    .flush_range = iree_hal_replay_recorder_buffer_flush_range,
    .query_memory = iree_hal_replay_recorder_buffer_query_memory,
    .query_virtual = iree_hal_replay_recorder_buffer_query_virtual,
    .detach_mappings = iree_hal_replay_recorder_buffer_detach_mappings,
};

static const iree_hal_buffer_vtable_t
    iree_hal_replay_recorder_allocated_buffer_vtable = {
        .destroy = iree_hal_replay_recorder_buffer_destroy,
        .export_range = iree_hal_replay_recorder_buffer_export_range,
        .map_range = iree_hal_replay_recorder_buffer_map_range,
        .unmap_range = iree_hal_replay_recorder_buffer_unmap_range,
        .invalidate_range = iree_hal_replay_recorder_buffer_invalidate_range,
        .flush_range = iree_hal_replay_recorder_buffer_flush_range,
        .query_memory = iree_hal_replay_recorder_buffer_query_memory,
        .query_virtual = iree_hal_replay_recorder_buffer_query_virtual,
        .detach_mappings = iree_hal_replay_recorder_buffer_detach_mappings,
        .allocation = &iree_hal_replay_recorder_buffer_allocation_vtable,
};
