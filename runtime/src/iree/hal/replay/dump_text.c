// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump_text.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/replay/digest.h"
#include "iree/hal/replay/dump_layout.h"

static iree_status_t iree_hal_replay_dump_append_text_buffer_ref(
    iree_string_builder_t* builder, const char* label,
    const iree_hal_replay_buffer_ref_payload_t* buffer_ref) {
  return iree_string_builder_append_format(
      builder,
      " %s={buffer_id=%" PRIu64 " offset=%" PRIu64 " length=%" PRIu64
      " slot=%" PRIu32 "}",
      label, buffer_ref->buffer_id, buffer_ref->offset, buffer_ref->length,
      buffer_ref->buffer_slot);
}

static iree_status_t iree_hal_replay_dump_append_text_semaphores(
    iree_string_builder_t* builder, const char* label,
    const iree_hal_replay_semaphore_timepoint_payload_t* semaphores,
    iree_host_size_t semaphore_count) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, " %s=[", label));
  for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{semaphore_id=%" PRIu64 " value=%" PRIu64 "}",
        semaphores[i].semaphore_id, semaphores[i].value));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t iree_hal_replay_dump_append_text_buffer_refs(
    iree_string_builder_t* builder, const char* label,
    const iree_hal_replay_buffer_ref_payload_t* buffer_refs,
    iree_host_size_t buffer_ref_count) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, " %s=[", label));
  for (iree_host_size_t i = 0; i < buffer_ref_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "{buffer_id=%" PRIu64 " offset=%" PRIu64 " length=%" PRIu64
        " slot=%" PRIu32 "}",
        buffer_refs[i].buffer_id, buffer_refs[i].offset, buffer_refs[i].length,
        buffer_refs[i].buffer_slot));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t iree_hal_replay_dump_append_text_atomic_wait_params(
    iree_string_builder_t* builder,
    const iree_hal_replay_atomic_wait_params_payload_t* params) {
  return iree_string_builder_append_format(
      builder,
      " value=0x%016" PRIx64 " mask=0x%016" PRIx64 " flags=0x%08" PRIx32
      " width=%" PRIu8 " condition=%s(%" PRIu8 ")",
      params->value, params->mask, params->flags, params->width,
      iree_hal_replay_dump_atomic_wait_condition_string(params->condition),
      params->condition);
}

static iree_status_t iree_hal_replay_dump_append_text_atomic_store_params(
    iree_string_builder_t* builder,
    const iree_hal_replay_atomic_store_params_payload_t* params) {
  return iree_string_builder_append_format(
      builder, " value=0x%016" PRIx64 " flags=0x%08" PRIx32 " width=%" PRIu8,
      params->value, params->flags, params->width);
}

static iree_status_t iree_hal_replay_dump_append_text_atomic_rmw_params(
    iree_string_builder_t* builder,
    const iree_hal_replay_atomic_rmw_params_payload_t* params) {
  return iree_string_builder_append_format(
      builder,
      " operand=0x%016" PRIx64 " flags=0x%08" PRIx32 " width=%" PRIu8
      " operation=%s(%" PRIu8 ")",
      params->operand, params->flags, params->width,
      iree_hal_replay_dump_atomic_rmw_operation_string(params->operation),
      params->operation);
}

static iree_status_t iree_hal_replay_dump_append_text_queue_semaphores(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dump_queue_payload_layout_t* layout,
    iree_host_size_t wait_semaphore_count,
    iree_host_size_t signal_semaphore_count) {
  const uint8_t* payload_data = record->payload.data;
  const uint8_t* wait_data = payload_data + layout->wait_payloads_offset;
  const uint8_t* signal_data = payload_data + layout->signal_payloads_offset;
  const iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)wait_data;
  const iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)signal_data;
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_semaphores(
      builder, "wait_semaphores", wait_payloads, wait_semaphore_count));
  return iree_hal_replay_dump_append_text_semaphores(
      builder, "signal_semaphores", signal_payloads, signal_semaphore_count);
}

static iree_status_t iree_hal_replay_dump_append_text_queue_atomic_header(
    iree_string_builder_t* builder,
    const iree_hal_replay_file_range_t* payload_range,
    const iree_hal_replay_dump_queue_payload_layout_t* layout,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count) {
  return iree_string_builder_append_format(
      builder,
      " wait_count=%" PRIu64 " signal_count=%" PRIu64 " wait_range=[%" PRIu64
      ", +%" PRIhsz "] signal_range=[%" PRIu64 ", +%" PRIhsz "]",
      wait_semaphore_count, signal_semaphore_count,
      payload_range->offset + layout->wait_payloads_offset,
      layout->wait_payloads_size,
      payload_range->offset + layout->signal_payloads_offset,
      layout->signal_payloads_size);
}

static iree_status_t iree_hal_replay_dump_append_text_queue_transfer_operations(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range,
    const iree_hal_replay_queue_transfer_payload_t* payload,
    const iree_hal_replay_dump_queue_transfer_layout_t* layout) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, " operations=["));
  for (iree_host_size_t i = 0; i < payload->operation_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    iree_hal_replay_queue_transfer_operation_payload_t operation;
    IREE_RETURN_IF_ERROR(iree_hal_replay_dump_read_queue_transfer_operation(
        record, layout, i, &operation));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(
            builder,
            "{type=%s(%" PRIu32 ") flags=0x%016" PRIx64
            " source={buffer_id=%" PRIu64 " offset=%" PRIu64 " length=%" PRIu64
            " slot=%" PRIu32 "}"
            " target={buffer_id=%" PRIu64 " offset=%" PRIu64 " length=%" PRIu64
            " slot=%" PRIu32 "}"
            " data_range=[%" PRIu64 ", +%" PRIu64 "]}",
            iree_hal_replay_dump_queue_transfer_operation_type_string(
                operation.type),
            operation.type, operation.flags, operation.source_ref.buffer_id,
            operation.source_ref.offset, operation.source_ref.length,
            operation.source_ref.buffer_slot, operation.target_ref.buffer_id,
            operation.target_ref.offset, operation.target_ref.length,
            operation.target_ref.buffer_slot,
            payload_range->offset + layout->data_offset + operation.data_offset,
            operation.data_length));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t iree_hal_replay_dump_append_text_payload(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range) {
  switch (record->header.payload_type) {
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE: {
      iree_string_view_t scope_name;
      IREE_RETURN_IF_ERROR(
          iree_hal_replay_dump_scope_name(record, &scope_name));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_cstring(builder, " name=\""));
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_string(builder, scope_name));
      return iree_string_builder_append_cstring(builder, "\"");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_object_payload_t)));
      iree_hal_replay_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " allocation_size=%" PRIu64 " byte_offset=%" PRIu64
          " byte_length=%" PRIu64 " queue_family_affinity=%" PRIu64
          " placement_flags=0x%08" PRIx32 " memory_type=0x%08" PRIx32
          " allowed_usage=0x%08" PRIx32 " allowed_access=0x%04" PRIx16,
          payload.allocation_size, payload.byte_offset, payload.byte_length,
          payload.queue_family_affinity, payload.placement_flags,
          payload.memory_type, payload.allowed_usage, payload.allowed_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_allocator_allocate_buffer_payload_t)));
      iree_hal_replay_allocator_allocate_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " allocation_size=%" PRIu64 " queue_family_affinity=%" PRIu64
          " min_alignment=%" PRIu64 " usage=0x%08" PRIx32 " type=0x%08" PRIx32
          " access=0x%04" PRIx16,
          payload.allocation_size, payload.queue_family_affinity,
          payload.min_alignment, payload.usage, payload.type, payload.access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_IMPORT_BUFFER: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_allocator_import_buffer_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay import buffer payload is short");
      }
      iree_hal_replay_allocator_import_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.data_length >
          record->payload.data_length -
              sizeof(iree_hal_replay_allocator_import_buffer_payload_t)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay import buffer data extends past record");
      }
      const uint64_t data_offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_allocator_import_buffer_payload_t);
      return iree_string_builder_append_format(
          builder,
          " allocation_size=%" PRIu64 " queue_family_affinity=%" PRIu64
          " min_alignment=%" PRIu64 " usage=0x%08" PRIx32 " type=0x%08" PRIx32
          " access=0x%04" PRIx16 " external_type=%" PRIu32
          " external_flags=0x%08" PRIx32 " data_range=[%" PRIu64 ", +%" PRIu64
          "]",
          payload.allocation.allocation_size,
          payload.allocation.queue_family_affinity,
          payload.allocation.min_alignment, payload.allocation.usage,
          payload.allocation.type, payload.allocation.access,
          payload.external_type, payload.external_flags, data_offset,
          payload.data_length);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_range_payload_t)));
      iree_hal_replay_buffer_range_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " byte_offset=%" PRIu64 " byte_length=%" PRIu64
          " mapping_mode=0x%08" PRIx32 " memory_access=0x%04" PRIx16,
          payload.byte_offset, payload.byte_length, payload.mapping_mode,
          payload.memory_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_buffer_range_data_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay buffer range data payload is short");
      }
      iree_hal_replay_buffer_range_data_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.data_length >
          record->payload.data_length -
              sizeof(iree_hal_replay_buffer_range_data_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay buffer range data extends past record");
      }
      const uint64_t data_offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_buffer_range_data_payload_t);
      return iree_string_builder_append_format(
          builder,
          " byte_offset=%" PRIu64 " byte_length=%" PRIu64
          " data_range=[%" PRIu64 ", +%" PRIu64
          "]"
          " memory_access=0x%04" PRIx16,
          payload.byte_offset, payload.byte_length, data_offset,
          payload.data_length, payload.memory_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_FAMILY_COMMAND_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(
              iree_hal_replay_queue_family_command_buffer_object_payload_t)));
      iree_hal_replay_queue_family_command_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.reserved0 != 0) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay command buffer reserved fields must be zero");
      }
      return iree_string_builder_append_format(
          builder,
          " mode=0x%08" PRIx32 " categories=0x%08" PRIx32
          " queue_family_ordinal=%" PRIu32 " binding_capacity=%" PRIu64,
          payload.mode, payload.command_categories,
          payload.queue_family_ordinal, payload.binding_capacity);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_LOAD: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_executable_load_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable load payload is short");
      }
      iree_hal_replay_executable_load_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_executable_load_ranges_t ranges;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_compute_executable_load_ranges(
          record, &payload, &ranges));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " queue_family_ordinal=%" PRIu32 " load_flags=0x%08" PRIx32
          " target_kind=%" PRIu32 " target_flags=0x%08" PRIx32
          " target_affinity=%" PRIu64 " family_range=[%" PRIu64 ", +%" PRIu32
          "]"
          " key_range=[%" PRIu64 ", +%" PRIu32 "] data_range=[%" PRIu64
          ", +%" PRIu64 "] constants_range=[%" PRIu64 ", +%" PRIhsz
          "] metadata_range=[%" PRIu64 ", +%" PRIu32 "]",
          payload.queue_family_ordinal, payload.load_flags, payload.target_kind,
          payload.target_flags, payload.target_physical_device_affinity,
          payload_range->offset + ranges.target_family_offset,
          payload.target_family_length,
          payload_range->offset + ranges.target_key_offset,
          payload.target_key_length, payload_range->offset + ranges.data_offset,
          payload.executable_data_length,
          payload_range->offset + ranges.constants_offset,
          ranges.constant_bytes, payload_range->offset + ranges.metadata_offset,
          payload.executable_metadata_length));
      iree_hal_replay_executable_metadata_header_t metadata_header;
      bool has_metadata = false;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_read_executable_metadata_header(
          record, &payload, &ranges, &has_metadata, &metadata_header));
      if (has_metadata) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            " metadata_functions=%" PRIu64 " metadata_parameters=%" PRIu64
            " metadata_function_name_storage_length=%" PRIu64,
            metadata_header.function_count, metadata_header.parameter_count,
            metadata_header.function_name_storage_length));
      }
      return iree_ok_status();
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_semaphore_object_payload_t)));
      iree_hal_replay_semaphore_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " queue_family_affinity=%" PRIu64 " initial_value=%" PRIu64
          " flags=0x%016" PRIx64,
          payload.queue_family_affinity, payload.initial_value, payload.flags);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_PROVISIONED_QUEUE_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_provisioned_queue_object_payload_t)));
      iree_hal_replay_provisioned_queue_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (IREE_UNLIKELY(payload.reserved0 != 0)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay provisioned queue reserved fields must be zero");
      }
      return iree_string_builder_append_format(
          builder, " family_ordinal=%" PRIu32 " queue_ordinal=%" PRIu32,
          payload.family_ordinal, payload.queue_ordinal);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_file_object_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay file object payload is short");
      }
      iree_hal_replay_file_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.reference_length > IREE_HOST_SIZE_MAX ||
          sizeof(payload) + (iree_host_size_t)payload.reference_length !=
              record->payload.data_length) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay file object payload length mismatch");
      }
      return iree_string_builder_append_format(
          builder,
          " queue_family_affinity=%" PRIu64 " file_length=%" PRIu64
          " access=0x%08" PRIx32 " flags=0x%08" PRIx32 " handle_type=%" PRIu32
          " reference_type=%s(%" PRIu32 ") file_device=%" PRIu64
          " file_inode=%" PRIu64 " file_mtime_ns=%" PRIu64
          " validation_type=%s(%" PRIu32 ") digest_type=%" PRIu32
          " digest_fnv1a64=0x%016" PRIx64 " reference_range=[%" PRIu64
          ", +%" PRIu64 "]",
          payload.queue_family_affinity, payload.file_length, payload.access,
          payload.flags, payload.handle_type,
          iree_hal_replay_dump_file_reference_type_string(
              payload.reference_type),
          payload.reference_type, payload.file_device, payload.file_inode,
          payload.file_mtime_ns,
          iree_hal_replay_dump_file_validation_type_string(
              payload.validation_type),
          payload.validation_type, (uint32_t)payload.digest_type,
          iree_hal_replay_digest_load_fnv1a64(payload.digest),
          payload_range->offset + sizeof(payload), payload.reference_length);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_dispatch_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay dispatch payload is short");
      }
      iree_hal_replay_dispatch_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t constants_offset = 0;
      iree_host_size_t bindings_offset = 0;
      iree_host_size_t bindings_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_dispatch_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &constants_offset, &bindings_offset, &bindings_size));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " executable_id=%" PRIu64 " function_ordinal=%" PRIu32
          " flags=0x%08" PRIx32 " workgroup_count=[%" PRIu32 ",%" PRIu32
          ",%" PRIu32 "] workgroup_size=[%" PRIu32 ",%" PRIu32 ",%" PRIu32
          "] wait_count=%" PRIu64 " signal_count=%" PRIu64
          " constants_range=[%" PRIu64 ", +%" PRIu64
          "] bindings_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.executable_id, payload.function_ordinal, payload.flags,
          payload.workgroup_count[0], payload.workgroup_count[1],
          payload.workgroup_count[2], payload.workgroup_size[0],
          payload.workgroup_size[1], payload.workgroup_size[2],
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload_range->offset + constants_offset, payload.constants_length,
          payload_range->offset + bindings_offset, bindings_size));
      return iree_hal_replay_dump_append_text_buffer_ref(
          builder, "workgroup_count_ref", &payload.workgroup_count_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_BARRIER: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_barrier_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue barrier payload is short");
      }
      iree_hal_replay_queue_barrier_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.reserved0 != 0) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay queue barrier reserved fields must be zero");
      }
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " flags=0x%016" PRIx64 " wait_count=%" PRIu64 " signal_count=%" PRIu64
          " wait_range=[%" PRIu64 ", +%" PRIhsz "] signal_range=[%" PRIu64
          ", +%" PRIhsz "]",
          payload.flags, payload.wait_semaphore_count,
          payload.signal_semaphore_count,
          payload_range->offset + layout.wait_payloads_offset,
          layout.wait_payloads_size,
          payload_range->offset + layout.signal_payloads_offset,
          layout.signal_payloads_size));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_EXECUTE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_execute_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue execute payload is short");
      }
      iree_hal_replay_queue_execute_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t bindings_offset = 0;
      iree_host_size_t bindings_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_execute_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &bindings_offset, &bindings_size));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " command_buffer_id=%" PRIu64 " flags=0x%016" PRIx64
          " wait_count=%" PRIu64 " signal_count=%" PRIu64
          " wait_range=[%" PRIu64 ", +%" PRIhsz "] signal_range=[%" PRIu64
          ", +%" PRIhsz "] bindings_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.command_buffer_id, payload.flags,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload_range->offset + wait_offset, wait_size,
          payload_range->offset + signal_offset, signal_size,
          payload_range->offset + bindings_offset, bindings_size));
      const iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads =
          (const iree_hal_replay_semaphore_timepoint_payload_t*)(record->payload
                                                                     .data +
                                                                 wait_offset);
      const iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads =
          (const iree_hal_replay_semaphore_timepoint_payload_t*)(record->payload
                                                                     .data +
                                                                 signal_offset);
      const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
          (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                        bindings_offset);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_semaphores(
          builder, "wait_semaphores", wait_payloads,
          (iree_host_size_t)payload.wait_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_semaphores(
          builder, "signal_semaphores", signal_payloads,
          (iree_host_size_t)payload.signal_semaphore_count));
      return iree_hal_replay_dump_append_text_buffer_refs(
          builder, "bindings", binding_payloads,
          (iree_host_size_t)payload.binding_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TRANSFER: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_transfer_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue transfer payload is short");
      }
      iree_hal_replay_queue_transfer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_transfer_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_transfer_layout(
          record, &payload, &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " wait_count=%" PRIu64 " signal_count=%" PRIu64
          " operation_count=%" PRIu64 " data_length=%" PRIu64
          " wait_range=[%" PRIu64 ", +%" PRIhsz "] signal_range=[%" PRIu64
          ", +%" PRIhsz "] operations_range=[%" PRIu64 ", +%" PRIhsz
          "] data_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload.operation_count, payload.data_length,
          payload_range->offset + layout.queue.wait_payloads_offset,
          layout.queue.wait_payloads_size,
          payload_range->offset + layout.queue.signal_payloads_offset,
          layout.queue.signal_payloads_size,
          payload_range->offset + layout.operation_payloads_offset,
          layout.operation_payloads_size,
          payload_range->offset + layout.data_offset, layout.data_size));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout.queue,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count));
      return iree_hal_replay_dump_append_text_queue_transfer_operations(
          builder, record, payload_range, &payload, &layout);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_read_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay exact queue read payload is short");
      }
      iree_hal_replay_queue_read_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.captured_data_length,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_file_id=%" PRIu64 " source_offset=%" PRIu64
          " flags=0x%016" PRIx64 " captured_data_length=%" PRIu64
          " wait_count=%" PRIu64 " signal_count=%" PRIu64
          " wait_range=[%" PRIu64 ", +%" PRIhsz "] signal_range=[%" PRIu64
          ", +%" PRIhsz "] captured_data_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.source_file_id, payload.source_offset, payload.flags,
          payload.captured_data_length, payload.wait_semaphore_count,
          payload.signal_semaphore_count,
          payload_range->offset + layout.wait_payloads_offset,
          layout.wait_payloads_size,
          payload_range->offset + layout.signal_payloads_offset,
          layout.signal_payloads_size,
          payload_range->offset + layout.trailing_payload_offset,
          layout.trailing_payload_size));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_WRITE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_write_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay exact queue write payload is short");
      }
      iree_hal_replay_queue_write_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " target_file_id=%" PRIu64 " target_offset=%" PRIu64
          " flags=0x%016" PRIx64 " wait_count=%" PRIu64 " signal_count=%" PRIu64
          " wait_range=[%" PRIu64 ", +%" PRIhsz "] signal_range=[%" PRIu64
          ", +%" PRIhsz "]",
          payload.target_file_id, payload.target_offset, payload.flags,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload_range->offset + layout.wait_payloads_offset,
          layout.wait_payloads_size,
          payload_range->offset + layout.signal_payloads_offset,
          layout.signal_payloads_size));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "source_ref", &payload.source_ref));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ALLOCA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_alloca_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay exact queue alloca payload is short");
      }
      iree_hal_replay_queue_alloca_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t requests_size = 0;
      if (payload.reserved0 != 0 ||
          payload.request_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.request_count,
              sizeof(iree_hal_replay_queue_alloca_request_payload_t),
              &requests_size)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay exact queue alloca payload is invalid");
      }
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, requests_size, &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " request_count=%" PRIu64 " wait_count=%" PRIu64
          " signal_count=%" PRIu64 " wait_range=[%" PRIu64 ", +%" PRIhsz
          "] signal_range=[%" PRIu64 ", +%" PRIhsz "] request_range=[%" PRIu64
          ", +%" PRIhsz "]",
          payload.request_count, payload.wait_semaphore_count,
          payload.signal_semaphore_count,
          payload_range->offset + layout.wait_payloads_offset,
          layout.wait_payloads_size,
          payload_range->offset + layout.signal_payloads_offset,
          layout.signal_payloads_size,
          payload_range->offset + layout.trailing_payload_offset,
          layout.trailing_payload_size));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_DEALLOCA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_dealloca_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay exact queue dealloca payload is short");
      }
      iree_hal_replay_queue_dealloca_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t buffer_ids_size = 0;
      if (payload.reserved0 != 0 || payload.buffer_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.buffer_count,
                                      sizeof(iree_hal_replay_object_id_t),
                                      &buffer_ids_size)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay exact queue dealloca payload is invalid");
      }
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, buffer_ids_size, &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " buffer_count=%" PRIu64 " wait_count=%" PRIu64
          " signal_count=%" PRIu64 " wait_range=[%" PRIu64 ", +%" PRIhsz
          "] signal_range=[%" PRIu64 ", +%" PRIhsz "] buffer_id_range=[%" PRIu64
          ", +%" PRIhsz "]",
          payload.buffer_count, payload.wait_semaphore_count,
          payload.signal_semaphore_count,
          payload_range->offset + layout.wait_payloads_offset,
          layout.wait_payloads_size,
          payload_range->offset + layout.signal_payloads_offset,
          layout.signal_payloads_size,
          payload_range->offset + layout.trailing_payload_offset,
          layout.trailing_payload_size));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_WAIT: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_atomic_wait_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue atomic wait payload is short");
      }
      iree_hal_replay_queue_atomic_wait_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_queue_atomic_header(
          builder, payload_range, &layout, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_atomic_wait_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_STORE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_atomic_store_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue atomic store payload is short");
      }
      iree_hal_replay_queue_atomic_store_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_queue_atomic_header(
          builder, payload_range, &layout, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_atomic_store_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_ATOMIC_RMW: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_atomic_rmw_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue atomic RMW payload is short");
      }
      iree_hal_replay_queue_atomic_rmw_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_queue_atomic_header(
          builder, payload_range, &layout, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_atomic_rmw_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_TIMESTAMP: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_timestamp_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue timestamp payload is short");
      }
      iree_hal_replay_queue_timestamp_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " flags=0x%016" PRIx64, payload.flags));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_queue_atomic_header(
          builder, payload_range, &layout, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_hal_replay_dump_append_text_queue_semaphores(
          builder, record, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_command_buffer_execution_barrier_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay execution barrier payload is short");
      }
      iree_hal_replay_command_buffer_execution_barrier_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t memory_offset = 0;
      iree_host_size_t memory_size = 0;
      iree_host_size_t buffer_offset = 0;
      iree_host_size_t buffer_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_execution_barrier_layout(
          record, &payload, &memory_offset, &memory_size, &buffer_offset,
          &buffer_size));
      return iree_string_builder_append_format(
          builder,
          " source_stage_mask=0x%016" PRIx64 " target_stage_mask=0x%016" PRIx64
          " flags=0x%016" PRIx64 " memory_count=%" PRIu64
          " buffer_count=%" PRIu64 " memory_barriers_range=[%" PRIu64
          ", +%" PRIhsz "] buffer_barriers_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.source_stage_mask, payload.target_stage_mask, payload.flags,
          payload.memory_barrier_count, payload.buffer_barrier_count,
          payload_range->offset + memory_offset, memory_size,
          payload_range->offset + buffer_offset, buffer_size);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_atomic_wait_payload_t)));
      iree_hal_replay_command_buffer_atomic_wait_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_stage_mask=0x%016" PRIx64 " target_stage_mask=0x%016" PRIx64,
          payload.source_stage_mask, payload.target_stage_mask));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_atomic_wait_params(
          builder, &payload.params));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target_ref",
                                                         &payload.target_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_atomic_store_payload_t)));
      iree_hal_replay_command_buffer_atomic_store_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_stage_mask=0x%016" PRIx64 " target_stage_mask=0x%016" PRIx64,
          payload.source_stage_mask, payload.target_stage_mask));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_atomic_store_params(
          builder, &payload.params));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target_ref",
                                                         &payload.target_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_command_buffer_atomic_rmw_payload_t)));
      iree_hal_replay_command_buffer_atomic_rmw_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " source_stage_mask=0x%016" PRIx64 " target_stage_mask=0x%016" PRIx64,
          payload.source_stage_mask, payload.target_stage_mask));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_atomic_rmw_params(
          builder, &payload.params));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target_ref",
                                                         &payload.target_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_command_buffer_fill_buffer_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay command buffer fill payload is short");
      }
      iree_hal_replay_command_buffer_fill_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.pattern_length > IREE_HOST_SIZE_MAX ||
          sizeof(payload) + (iree_host_size_t)payload.pattern_length !=
              record->payload.data_length) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay command buffer fill payload length mismatch");
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " flags=0x%08" PRIx32 " pattern_range=[%" PRIu64 ", +%" PRIu64 "]",
          payload.flags, payload_range->offset + sizeof(payload),
          payload.pattern_length));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target",
                                                         &payload.target_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_command_buffer_update_buffer_payload_t)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay command buffer update payload is short");
      }
      iree_hal_replay_command_buffer_update_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.data_length > IREE_HOST_SIZE_MAX ||
          sizeof(payload) + (iree_host_size_t)payload.data_length !=
              record->payload.data_length) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay command buffer update payload length mismatch");
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " flags=0x%08" PRIx32 " source_offset=%" PRIu64
          " data_range=[%" PRIu64 ", +%" PRIu64 "]",
          payload.flags, payload.source_offset,
          payload_range->offset + sizeof(payload), payload.data_length));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target",
                                                         &payload.target_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_copy_buffer_payload_t)));
      iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " flags=0x%08" PRIx32, payload.flags));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "source", &payload.source_ref));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target",
                                                         &payload.target_ref);
    }
    default:
      return iree_ok_status();
  }
}

iree_status_t iree_hal_replay_dump_emit_text_record(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t record_offset) {
  const iree_hal_replay_file_record_header_t* header = &record->header;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  @%" PRIhsz " #%" PRIu64 " %-11s dev=%" PRIu64 " obj=%" PRIu64
      " rel=%" PRIu64 " thread=%" PRIu64 " status=%s",
      record_offset, header->sequence_ordinal,
      iree_hal_replay_file_record_type_string(header->record_type),
      header->device_id, header->object_id, header->related_object_id,
      header->thread_id,
      iree_status_code_string((iree_status_code_t)header->status_code)));
  if (header->object_type != IREE_HAL_REPLAY_OBJECT_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " object=%s(%u)",
        iree_hal_replay_object_type_string(header->object_type),
        header->object_type));
  }
  if (header->operation_code != IREE_HAL_REPLAY_OPERATION_CODE_NONE) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " op=%s(%u)",
        iree_hal_replay_operation_code_string(header->operation_code),
        header->operation_code));
  }
  if (header->payload_type != IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE ||
      header->payload_length != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " payload=%s(%u) range=[%" PRIu64 ", +%" PRIu64 "]",
        iree_hal_replay_payload_type_string(header->payload_type),
        header->payload_type, payload_range->offset, payload_range->length));
    IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_payload(
        builder, record, payload_range));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_hal_replay_dump_emit(context, builder);
}

iree_status_t iree_hal_replay_dump_emit_text_file(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_header_t* header,
    const iree_hal_replay_dump_file_summary_t* summary) {
  const bool environment_referenced = summary->external_file_count != 0;
  const bool hermetic =
      !environment_referenced && summary->unknown_file_reference_count == 0;
  const bool strict_replay_supported =
      summary->unsupported_count == 0 &&
      summary->unknown_file_reference_count == 0 &&
      summary->unknown_file_validation_count == 0;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(
          builder,
          "IREE HAL replay v%u.%u\nfile_length: %" PRIu64
          "\nheader_length: %u\nsummary:\n"
          "  hermetic: %s\n"
          "  environment_referenced: %s\n"
          "  strict_replay_supported: %s\n"
          "  records: total=%" PRIu64 " objects=%" PRIu64 " operations=%" PRIu64
          " unsupported=%" PRIu64 "\n"
          "  scopes: begin=%" PRIu64 " end=%" PRIu64 "\n"
          "  files: total=%" PRIu64 " external=%" PRIu64 " inline=%" PRIu64
          " ranges=%" PRIu64 " unknown=%" PRIu64 "\n"
          "  file_bytes: external=%" PRIu64 " inline=%" PRIu64
          " ranges=%" PRIu64 " captured_reads=%" PRIu64 "\n"
          "  file_validation: identity=%" PRIu64 " digest=%" PRIu64
          " none=%" PRIu64 " unknown=%" PRIu64 "\nrecords:\n",
          header->version_major, header->version_minor, header->file_length,
          header->header_length, hermetic ? "yes" : "no",
          environment_referenced ? "yes" : "no",
          strict_replay_supported ? "yes" : "no", summary->record_count,
          summary->object_count, summary->operation_count,
          summary->unsupported_count, summary->scope_begin_count,
          summary->scope_end_count, summary->file_object_count,
          summary->external_file_count, summary->inline_file_count,
          summary->range_file_count, summary->unknown_file_reference_count,
          summary->external_file_total_length,
          summary->inline_file_total_length, summary->range_file_total_length,
          summary->captured_read_total_length,
          summary->identity_file_validation_count,
          summary->digest_file_validation_count,
          summary->no_file_validation_count,
          summary->unknown_file_validation_count));
  return iree_hal_replay_dump_emit(context, builder);
}
