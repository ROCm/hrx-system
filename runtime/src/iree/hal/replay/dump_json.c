// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump_json.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/replay/digest.h"
#include "iree/hal/replay/dump_layout.h"

static iree_status_t iree_hal_replay_dump_append_json_string_view(
    iree_string_builder_t* builder, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    const char c = value.data[i];
    switch (c) {
      case '\\':
      case '"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_format(builder, "\\%c", c));
        break;
      }
      case '\b': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\b"));
        break;
      }
      case '\f': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\f"));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\t"));
        break;
      }
      default: {
        if ((uint8_t)c < 0x20) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, "\\u%04x", (uint32_t)(uint8_t)c));
        } else {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_format(builder, "%c", c));
        }
        break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t iree_hal_replay_dump_append_json_string(
    iree_string_builder_t* builder, const char* value) {
  return iree_hal_replay_dump_append_json_string_view(
      builder, iree_make_cstring_view(value));
}

static iree_status_t iree_hal_replay_dump_append_json_file_range(
    iree_string_builder_t* builder, const char* field_name,
    const iree_hal_replay_file_range_t* range) {
  return iree_string_builder_append_format(
      builder,
      ",\"%s\":{\"offset\":%" PRIu64 ",\"length\":%" PRIu64
      ",\"uncompressed_length\":%" PRIu64
      ",\"compression_type\":%u"
      ",\"digest_type\":%u}",
      field_name, range->offset, range->length, range->uncompressed_length,
      (uint32_t)range->compression_type, (uint32_t)range->digest_type);
}

static iree_status_t iree_hal_replay_dump_append_json_buffer_ref(
    iree_string_builder_t* builder, const char* field_name,
    const iree_hal_replay_buffer_ref_payload_t* buffer_ref) {
  return iree_string_builder_append_format(
      builder,
      ",\"%s\":{\"buffer_id\":%" PRIu64 ",\"offset\":%" PRIu64
      ",\"length\":%" PRIu64 ",\"buffer_slot\":%" PRIu32 "}",
      field_name, buffer_ref->buffer_id, buffer_ref->offset, buffer_ref->length,
      buffer_ref->buffer_slot);
}

static iree_status_t iree_hal_replay_dump_append_json_semaphores(
    iree_string_builder_t* builder, const char* field_name,
    const iree_hal_replay_semaphore_timepoint_payload_t* semaphores,
    iree_host_size_t semaphore_count) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, ",\"%s\":[", field_name));
  for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "{\"semaphore_id\":%" PRIu64 ",\"value\":%" PRIu64 "}",
        semaphores[i].semaphore_id, semaphores[i].value));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t iree_hal_replay_dump_append_json_buffer_refs(
    iree_string_builder_t* builder, const char* field_name,
    const iree_hal_replay_buffer_ref_payload_t* buffer_refs,
    iree_host_size_t buffer_ref_count) {
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_format(builder, ",\"%s\":[", field_name));
  for (iree_host_size_t i = 0; i < buffer_ref_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ","));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "{\"buffer_id\":%" PRIu64 ",\"offset\":%" PRIu64 ",\"length\":%" PRIu64
        ",\"buffer_slot\":%" PRIu32 "}",
        buffer_refs[i].buffer_id, buffer_refs[i].offset, buffer_refs[i].length,
        buffer_refs[i].buffer_slot));
  }
  return iree_string_builder_append_cstring(builder, "]");
}

static iree_status_t iree_hal_replay_dump_append_json_atomic_wait_params(
    iree_string_builder_t* builder,
    const iree_hal_replay_atomic_wait_params_payload_t* params) {
  return iree_string_builder_append_format(
      builder,
      ",\"value\":%" PRIu64 ",\"mask\":%" PRIu64 ",\"flags\":%" PRIu32
      ",\"width\":%" PRIu8 ",\"condition\":%" PRIu8
      ",\"condition_name\":\"%s\"",
      params->value, params->mask, params->flags, params->width,
      params->condition,
      iree_hal_replay_dump_atomic_wait_condition_string(params->condition));
}

static iree_status_t iree_hal_replay_dump_append_json_atomic_store_params(
    iree_string_builder_t* builder,
    const iree_hal_replay_atomic_store_params_payload_t* params) {
  return iree_string_builder_append_format(
      builder, ",\"value\":%" PRIu64 ",\"flags\":%" PRIu32 ",\"width\":%" PRIu8,
      params->value, params->flags, params->width);
}

static iree_status_t iree_hal_replay_dump_append_json_atomic_rmw_params(
    iree_string_builder_t* builder,
    const iree_hal_replay_atomic_rmw_params_payload_t* params) {
  return iree_string_builder_append_format(
      builder,
      ",\"operand\":%" PRIu64 ",\"flags\":%" PRIu32 ",\"width\":%" PRIu8
      ",\"operation\":%" PRIu8 ",\"operation_name\":\"%s\"",
      params->operand, params->flags, params->width, params->operation,
      iree_hal_replay_dump_atomic_rmw_operation_string(params->operation));
}

static iree_status_t iree_hal_replay_dump_append_json_queue_atomic_header(
    iree_string_builder_t* builder, uint64_t queue_affinity,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count) {
  return iree_string_builder_append_format(
      builder,
      ",\"payload\":{\"queue_affinity\":%" PRIu64
      ",\"wait_semaphore_count\":%" PRIu64
      ",\"signal_semaphore_count\":%" PRIu64,
      queue_affinity, wait_semaphore_count, signal_semaphore_count);
}

static iree_status_t iree_hal_replay_dump_append_json_queue_semaphores(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range,
    const iree_hal_replay_dump_queue_payload_layout_t* layout,
    iree_host_size_t wait_semaphore_count,
    iree_host_size_t signal_semaphore_count) {
  iree_hal_replay_file_range_t wait_range =
      iree_hal_replay_dump_payload_subrange(payload_range,
                                            layout->wait_payloads_offset,
                                            layout->wait_payloads_size);
  iree_hal_replay_file_range_t signal_range =
      iree_hal_replay_dump_payload_subrange(payload_range,
                                            layout->signal_payloads_offset,
                                            layout->signal_payloads_size);
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
      builder, "wait_semaphores_range", &wait_range));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
      builder, "signal_semaphores_range", &signal_range));
  const uint8_t* payload_data = record->payload.data;
  const uint8_t* wait_data = payload_data + layout->wait_payloads_offset;
  const uint8_t* signal_data = payload_data + layout->signal_payloads_offset;
  const iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)wait_data;
  const iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)signal_data;
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_semaphores(
      builder, "wait_semaphores", wait_payloads, wait_semaphore_count));
  return iree_hal_replay_dump_append_json_semaphores(
      builder, "signal_semaphores", signal_payloads, signal_semaphore_count);
}

static iree_status_t iree_hal_replay_dump_append_json_payload(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range) {
  switch (record->header.payload_type) {
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE: {
      iree_string_view_t scope_name;
      IREE_RETURN_IF_ERROR(
          iree_hal_replay_dump_scope_name(record, &scope_name));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          builder, ",\"payload\":{\"name\":"));
      IREE_RETURN_IF_ERROR(
          iree_hal_replay_dump_append_json_string_view(builder, scope_name));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_object_payload_t)));
      iree_hal_replay_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"allocation_size\":%" PRIu64
          ",\"byte_offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
          ",\"queue_family_affinity\":%" PRIu64 ",\"placement_flags\":%" PRIu32
          ",\"memory_type\":%" PRIu32 ",\"allowed_usage\":%" PRIu32
          ",\"allowed_access\":%" PRIu16 "}",
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
          ",\"payload\":{\"allocation_size\":%" PRIu64
          ",\"queue_family_affinity\":%" PRIu64 ",\"min_alignment\":%" PRIu64
          ",\"usage\":%" PRIu32 ",\"type\":%" PRIu32 ",\"access\":%" PRIu16 "}",
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
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"allocation_size\":%" PRIu64
          ",\"queue_family_affinity\":%" PRIu64 ",\"min_alignment\":%" PRIu64
          ",\"usage\":%" PRIu32 ",\"type\":%" PRIu32 ",\"access\":%" PRIu16
          ",\"external_type\":%" PRIu32 ",\"external_flags\":%" PRIu32,
          payload.allocation.allocation_size,
          payload.allocation.queue_family_affinity,
          payload.allocation.min_alignment, payload.allocation.usage,
          payload.allocation.type, payload.allocation.access,
          payload.external_type, payload.external_flags));
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_file_range_empty();
      data_range.offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_allocator_import_buffer_payload_t);
      data_range.length = payload.data_length;
      data_range.uncompressed_length = payload.data_length;
      data_range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
      data_range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_range_payload_t)));
      iree_hal_replay_buffer_range_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"byte_offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
          ",\"mapping_mode\":%" PRIu32 ",\"memory_access\":%" PRIu16 "}",
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
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"byte_offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
          ",\"mapping_mode\":%" PRIu32 ",\"memory_access\":%" PRIu16,
          payload.byte_offset, payload.byte_length, payload.mapping_mode,
          payload.memory_access));
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_file_range_empty();
      data_range.offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_buffer_range_data_payload_t);
      data_range.length = payload.data_length;
      data_range.uncompressed_length = payload.data_length;
      data_range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
      data_range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_command_buffer_object_payload_t)));
      iree_hal_replay_command_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"mode\":%" PRIu32 ",\"command_categories\":%" PRIu32
          ",\"queue_affinity\":%" PRIu64 ",\"binding_capacity\":%" PRIu64 "}",
          payload.mode, payload.command_categories, payload.queue_affinity,
          payload.binding_capacity);
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
          ",\"payload\":{\"queue_affinity\":%" PRIu64 ",\"load_flags\":%" PRIu32
          ",\"target_kind\":%" PRIu32 ",\"target_flags\":%" PRIu32
          ",\"target_physical_device_affinity\":%" PRIu64
          ",\"target_family_length\":%" PRIu32 ",\"target_key_length\":%" PRIu32
          ",\"executable_data_length\":%" PRIu64 ",\"constant_count\":%" PRIu64
          ",\"executable_metadata_length\":%" PRIu32,
          payload.queue_affinity, payload.load_flags, payload.target_kind,
          payload.target_flags, payload.target_physical_device_affinity,
          payload.target_family_length, payload.target_key_length,
          payload.executable_data_length, payload.constant_count,
          payload.executable_metadata_length));
      iree_hal_replay_file_range_t target_family_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, ranges.target_family_offset,
              (iree_host_size_t)payload.target_family_length);
      iree_hal_replay_file_range_t target_key_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, ranges.target_key_offset,
              (iree_host_size_t)payload.target_key_length);
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, ranges.data_offset,
              (iree_host_size_t)payload.executable_data_length);
      iree_hal_replay_file_range_t constants_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, ranges.constants_offset, ranges.constant_bytes);
      iree_hal_replay_file_range_t metadata_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, ranges.metadata_offset,
              (iree_host_size_t)payload.executable_metadata_length);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "target_family_range", &target_family_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "target_key_range", &target_key_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "constants_range", &constants_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "metadata_range", &metadata_range));
      iree_hal_replay_executable_metadata_header_t metadata_header;
      bool has_metadata = false;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_read_executable_metadata_header(
          record, &payload, &ranges, &has_metadata, &metadata_header));
      if (has_metadata) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            ",\"metadata_function_count\":%" PRIu64
            ",\"metadata_parameter_count\":%" PRIu64
            ",\"metadata_function_name_storage_length\":%" PRIu64,
            metadata_header.function_count, metadata_header.parameter_count,
            metadata_header.function_name_storage_length));
      }
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_semaphore_object_payload_t)));
      iree_hal_replay_semaphore_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_family_affinity\":%" PRIu64
          ",\"initial_value\":%" PRIu64 ",\"flags\":%" PRIu64 "}",
          payload.queue_family_affinity, payload.initial_value, payload.flags);
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
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(
              builder,
              ",\"payload\":{\"queue_affinity\":%" PRIu64
              ",\"file_length\":%" PRIu64 ",\"file_device\":%" PRIu64
              ",\"file_inode\":%" PRIu64 ",\"file_mtime_ns\":%" PRIu64
              ",\"reference_length\":%" PRIu64 ",\"access\":%" PRIu32
              ",\"flags\":%" PRIu32 ",\"handle_type\":%" PRIu32
              ",\"reference_type\":%" PRIu32 ",\"reference_type_name\":\"%s\""
              ",\"validation_type\":%" PRIu32 ",\"validation_type_name\":\"%s\""
              ",\"digest_type\":%" PRIu32 ",\"digest_fnv1a64\":\"0x%016" PRIx64
              "\"",
              payload.queue_affinity, payload.file_length, payload.file_device,
              payload.file_inode, payload.file_mtime_ns,
              payload.reference_length, payload.access, payload.flags,
              payload.handle_type, payload.reference_type,
              iree_hal_replay_dump_file_reference_type_string(
                  payload.reference_type),
              payload.validation_type,
              iree_hal_replay_dump_file_validation_type_string(
                  payload.validation_type),
              (uint32_t)payload.digest_type,
              iree_hal_replay_digest_load_fnv1a64(payload.digest)));
      iree_hal_replay_file_range_t reference_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, sizeof(payload),
              (iree_host_size_t)payload.reference_length);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "reference_range", &reference_range));
      return iree_string_builder_append_cstring(builder, "}");
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
          ",\"payload\":{\"executable_id\":%" PRIu64
          ",\"queue_affinity\":%" PRIu64 ",\"function_ordinal\":%" PRIu32
          ",\"flags\":%" PRIu32 ",\"workgroup_count\":[%" PRIu32 ",%" PRIu32
          ",%" PRIu32 "],\"workgroup_size\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32
          "],\"dynamic_workgroup_local_memory\":%" PRIu32
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64
          ",\"constants_length\":%" PRIu64 ",\"binding_count\":%" PRIu64,
          payload.executable_id, payload.queue_affinity,
          payload.function_ordinal, payload.flags, payload.workgroup_count[0],
          payload.workgroup_count[1], payload.workgroup_count[2],
          payload.workgroup_size[0], payload.workgroup_size[1],
          payload.workgroup_size[2], payload.dynamic_workgroup_local_memory,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload.constants_length, payload.binding_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "workgroup_count_ref", &payload.workgroup_count_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range, wait_offset,
                                                wait_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range, signal_offset,
                                                signal_size);
      iree_hal_replay_file_range_t constants_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, constants_offset,
              (iree_host_size_t)payload.constants_length);
      iree_hal_replay_file_range_t bindings_range =
          iree_hal_replay_dump_payload_subrange(payload_range, bindings_offset,
                                                bindings_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "constants_range", &constants_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "bindings_range", &bindings_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_execute_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue execute payload is short");
      }
      iree_hal_replay_device_queue_execute_payload_t payload;
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
          ",\"payload\":{\"command_buffer_id\":%" PRIu64
          ",\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64 ",\"binding_count\":%" PRIu64,
          payload.command_buffer_id, payload.queue_affinity, payload.flags,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload.binding_count));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range, wait_offset,
                                                wait_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range, signal_offset,
                                                signal_size);
      iree_hal_replay_file_range_t bindings_range =
          iree_hal_replay_dump_payload_subrange(payload_range, bindings_offset,
                                                bindings_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "bindings_range", &bindings_range));
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
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_semaphores(
          builder, "wait_semaphores", wait_payloads,
          (iree_host_size_t)payload.wait_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_semaphores(
          builder, "signal_semaphores", signal_payloads,
          (iree_host_size_t)payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_refs(
          builder, "bindings", binding_payloads,
          (iree_host_size_t)payload.binding_count));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_alloca_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue alloca payload is short");
      }
      iree_hal_replay_device_queue_alloca_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"allocation_size\":%" PRIu64
          ",\"allocation_queue_family_affinity\":%" PRIu64
          ",\"min_alignment\":%" PRIu64 ",\"usage\":%" PRIu32
          ",\"type\":%" PRIu32 ",\"access\":%" PRIu16
          ",\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64,
          payload.allocation.allocation_size,
          payload.allocation.queue_family_affinity,
          payload.allocation.min_alignment, payload.allocation.usage,
          payload.allocation.type, payload.allocation.access,
          payload.queue_affinity, payload.flags, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_dealloca_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue dealloca payload is short");
      }
      iree_hal_replay_device_queue_dealloca_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64,
          payload.queue_affinity, payload.flags, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "buffer_ref", &payload.buffer_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_fill_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue fill payload is short");
      }
      iree_hal_replay_device_queue_fill_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.pattern_length, &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64 ",\"pattern_length\":%" PRIu64,
          payload.queue_affinity, payload.flags, payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.pattern_length));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      iree_hal_replay_file_range_t pattern_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.trailing_payload_offset,
                                                layout.trailing_payload_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "pattern_range", &pattern_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_update_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue update payload is short");
      }
      iree_hal_replay_device_queue_update_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.data_length, &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"source_offset\":%" PRIu64 ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64 ",\"data_length\":%" PRIu64,
          payload.queue_affinity, payload.flags, payload.source_offset,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload.data_length));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.trailing_payload_offset,
                                                layout.trailing_payload_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_copy_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue copy payload is short");
      }
      iree_hal_replay_device_queue_copy_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64,
          payload.queue_affinity, payload.flags, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "source_ref", &payload.source_ref));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_read_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue read payload is short");
      }
      iree_hal_replay_device_queue_read_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.captured_data_length,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"source_file_id\":%" PRIu64
          ",\"source_offset\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
          ",\"flags\":%" PRIu64 ",\"captured_data_length\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64,
          payload.source_file_id, payload.source_offset, payload.queue_affinity,
          payload.flags, payload.captured_data_length,
          payload.wait_semaphore_count, payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      iree_hal_replay_file_range_t captured_data_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.trailing_payload_offset,
                                                layout.trailing_payload_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "captured_data_range", &captured_data_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_write_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue write payload is short");
      }
      iree_hal_replay_device_queue_write_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"target_file_id\":%" PRIu64
          ",\"target_offset\":%" PRIu64 ",\"queue_affinity\":%" PRIu64
          ",\"flags\":%" PRIu64 ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64,
          payload.target_file_id, payload.target_offset, payload.queue_affinity,
          payload.flags, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "source_ref", &payload.source_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.wait_payloads_offset,
                                                layout.wait_payloads_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range,
                                                layout.signal_payloads_offset,
                                                layout.signal_payloads_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_atomic_wait_payload_t)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay device queue atomic wait payload is short");
      }
      iree_hal_replay_device_queue_atomic_wait_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_queue_atomic_header(
          builder, payload.queue_affinity, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_atomic_wait_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_queue_semaphores(
          builder, record, payload_range, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_atomic_store_payload_t)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay device queue atomic store payload is short");
      }
      iree_hal_replay_device_queue_atomic_store_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_queue_atomic_header(
          builder, payload.queue_affinity, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_atomic_store_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_queue_semaphores(
          builder, record, payload_range, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_atomic_rmw_payload_t)) {
        return iree_make_status(
            IREE_STATUS_DATA_LOSS,
            "replay device queue atomic RMW payload is short");
      }
      iree_hal_replay_device_queue_atomic_rmw_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_hal_replay_dump_queue_payload_layout_t layout;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_payload_layout(
          record, sizeof(payload), payload.wait_semaphore_count,
          payload.signal_semaphore_count, /*trailing_payload_length=*/0,
          &layout));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_queue_atomic_header(
          builder, payload.queue_affinity, payload.wait_semaphore_count,
          payload.signal_semaphore_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_atomic_rmw_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_queue_semaphores(
          builder, record, payload_range, &layout,
          (iree_host_size_t)payload.wait_semaphore_count,
          (iree_host_size_t)payload.signal_semaphore_count));
      return iree_string_builder_append_cstring(builder, "}");
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
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"source_stage_mask\":%" PRIu64
          ",\"target_stage_mask\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"memory_barrier_count\":%" PRIu64
          ",\"buffer_barrier_count\":%" PRIu64,
          payload.source_stage_mask, payload.target_stage_mask, payload.flags,
          payload.memory_barrier_count, payload.buffer_barrier_count));
      iree_hal_replay_file_range_t memory_range =
          iree_hal_replay_dump_payload_subrange(payload_range, memory_offset,
                                                memory_size);
      iree_hal_replay_file_range_t buffer_range =
          iree_hal_replay_dump_payload_subrange(payload_range, buffer_offset,
                                                buffer_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "memory_barriers_range", &memory_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "buffer_barriers_range", &buffer_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_atomic_wait_payload_t)));
      iree_hal_replay_command_buffer_atomic_wait_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"source_stage_mask\":%" PRIu64
          ",\"target_stage_mask\":%" PRIu64,
          payload.source_stage_mask, payload.target_stage_mask));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_atomic_wait_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_atomic_store_payload_t)));
      iree_hal_replay_command_buffer_atomic_store_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"source_stage_mask\":%" PRIu64
          ",\"target_stage_mask\":%" PRIu64,
          payload.source_stage_mask, payload.target_stage_mask));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_atomic_store_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_command_buffer_atomic_rmw_payload_t)));
      iree_hal_replay_command_buffer_atomic_rmw_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"source_stage_mask\":%" PRIu64
          ",\"target_stage_mask\":%" PRIu64,
          payload.source_stage_mask, payload.target_stage_mask));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_atomic_rmw_params(
          builder, &payload.params));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target_ref", &payload.target_ref));
      return iree_string_builder_append_cstring(builder, "}");
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
          ",\"payload\":{\"flags\":%" PRIu32 ",\"pattern_length\":%" PRIu64,
          payload.flags, payload.pattern_length));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target", &payload.target_ref));
      iree_hal_replay_file_range_t pattern_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, sizeof(payload),
              (iree_host_size_t)payload.pattern_length);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "pattern_range", &pattern_range));
      return iree_string_builder_append_cstring(builder, "}");
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
          ",\"payload\":{\"flags\":%" PRIu32 ",\"source_offset\":%" PRIu64
          ",\"data_length\":%" PRIu64,
          payload.flags, payload.source_offset, payload.data_length));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target", &payload.target_ref));
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, sizeof(payload),
              (iree_host_size_t)payload.data_length);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_copy_buffer_payload_t)));
      iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"payload\":{\"flags\":%" PRIu32, payload.flags));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "source", &payload.source_ref));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target", &payload.target_ref));
      return iree_string_builder_append_cstring(builder, "}");
    }
    default:
      return iree_string_builder_append_cstring(builder, ",\"payload\":null");
  }
}

iree_status_t iree_hal_replay_dump_emit_json_record(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t record_offset) {
  const iree_hal_replay_file_record_header_t* header = &record->header;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_file_record_type_string(header->record_type)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"file_offset\":%" PRIhsz ",\"record_length\":%" PRIu64
      ",\"payload_length\":%" PRIu64 ",\"sequence_ordinal\":%" PRIu64
      ",\"thread_id\":%" PRIu64 ",\"device_id\":%" PRIu64
      ",\"object_id\":%" PRIu64 ",\"related_object_id\":%" PRIu64
      ",\"record_type_code\":%u,\"record_flags\":%u",
      record_offset, header->record_length, header->payload_length,
      header->sequence_ordinal, header->thread_id, header->device_id,
      header->object_id, header->related_object_id, header->record_type,
      header->record_flags));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"object_type\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_object_type_string(header->object_type)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"object_type_code\":%u", header->object_type));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"operation\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_operation_code_string(header->operation_code)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"operation_code\":%u,\"status_code\":%u,\"status\":",
      header->operation_code, header->status_code));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder,
      iree_status_code_string((iree_status_code_t)header->status_code)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"payload_type\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_payload_type_string(header->payload_type)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"payload_type_code\":%u", header->payload_type));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
      builder, "payload_range", payload_range));
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_dump_append_json_payload(builder, record, payload_range));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}\n"));
  return iree_hal_replay_dump_emit(context, builder);
}

iree_status_t iree_hal_replay_dump_emit_json_file(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_header_t* header) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"kind\":\"file\",\"version_major\":%u,\"version_minor\":%u"
      ",\"header_length\":%u,\"flags\":%u,\"file_length\":%" PRIu64 "}\n",
      header->version_major, header->version_minor, header->header_length,
      header->flags, header->file_length));
  return iree_hal_replay_dump_emit(context, builder);
}

iree_status_t iree_hal_replay_dump_emit_json_summary(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_dump_file_summary_t* summary) {
  const bool environment_referenced = summary->external_file_count != 0;
  const bool hermetic =
      !environment_referenced && summary->unknown_file_reference_count == 0;
  const bool strict_replay_supported =
      summary->unsupported_count == 0 &&
      summary->unknown_file_reference_count == 0 &&
      summary->unknown_file_validation_count == 0;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"kind\":\"summary\",\"hermetic\":%s,\"environment_referenced\":%s"
      ",\"strict_replay_supported\":%s"
      ",\"record_count\":%" PRIu64 ",\"object_count\":%" PRIu64
      ",\"operation_count\":%" PRIu64 ",\"unsupported_count\":%" PRIu64
      ",\"scope_begin_count\":%" PRIu64 ",\"scope_end_count\":%" PRIu64
      ",\"file_object_count\":%" PRIu64 ",\"external_file_count\":%" PRIu64
      ",\"inline_file_count\":%" PRIu64 ",\"range_file_count\":%" PRIu64
      ",\"unknown_file_reference_count\":%" PRIu64
      ",\"external_file_total_length\":%" PRIu64
      ",\"inline_file_total_length\":%" PRIu64
      ",\"range_file_total_length\":%" PRIu64
      ",\"captured_read_total_length\":%" PRIu64
      ",\"file_validation\":{\"identity\":%" PRIu64 ",\"digest\":%" PRIu64
      ",\"none\":%" PRIu64 ",\"unknown\":%" PRIu64 "}}\n",
      hermetic ? "true" : "false", environment_referenced ? "true" : "false",
      strict_replay_supported ? "true" : "false", summary->record_count,
      summary->object_count, summary->operation_count,
      summary->unsupported_count, summary->scope_begin_count,
      summary->scope_end_count, summary->file_object_count,
      summary->external_file_count, summary->inline_file_count,
      summary->range_file_count, summary->unknown_file_reference_count,
      summary->external_file_total_length, summary->inline_file_total_length,
      summary->range_file_total_length, summary->captured_read_total_length,
      summary->identity_file_validation_count,
      summary->digest_file_validation_count, summary->no_file_validation_count,
      summary->unknown_file_validation_count));
  return iree_hal_replay_dump_emit(context, builder);
}
