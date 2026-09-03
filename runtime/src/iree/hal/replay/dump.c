// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump.h"

#include <string.h>

#include "iree/hal/replay/dump_json.h"
#include "iree/hal/replay/dump_layout.h"
#include "iree/hal/replay/dump_state.h"
#include "iree/hal/replay/dump_text.h"

static iree_status_t iree_hal_replay_dump_summary_scan_file_object(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_dump_file_summary_t* summary) {
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

  ++summary->file_object_count;
  switch (payload.reference_type) {
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH:
      ++summary->external_file_count;
      summary->external_file_total_length += payload.file_length;
      break;
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES:
      ++summary->inline_file_count;
      summary->inline_file_total_length += payload.reference_length;
      break;
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_CAPTURED_RANGES:
      ++summary->range_file_count;
      summary->range_file_total_length += payload.file_length;
      break;
    default:
      ++summary->unknown_file_reference_count;
      break;
  }
  switch (payload.validation_type) {
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_NONE:
      ++summary->no_file_validation_count;
      break;
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_IDENTITY:
      ++summary->identity_file_validation_count;
      break;
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST:
      ++summary->digest_file_validation_count;
      break;
    default:
      ++summary->unknown_file_validation_count;
      break;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_dump_summary_scan_operation(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_dump_file_summary_t* summary) {
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN:
      ++summary->scope_begin_count;
      break;
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END:
      ++summary->scope_end_count;
      break;
    default:
      break;
  }
  switch (record->header.payload_type) {
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_read_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue read payload is short");
      }
      iree_hal_replay_device_queue_read_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      summary->captured_read_total_length += payload.captured_data_length;
      break;
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_QUEUE_READ: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_queue_read_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay exact queue read payload is short");
      }
      iree_hal_replay_queue_read_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      summary->captured_read_total_length += payload.captured_data_length;
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_dump_scan_summary(
    iree_const_byte_span_t valid_contents, iree_host_size_t record_offset,
    iree_hal_replay_dump_file_summary_t* out_summary) {
  memset(out_summary, 0, sizeof(*out_summary));
  uint64_t expected_sequence_ordinal = 0;
  while (record_offset < valid_contents.data_length) {
    iree_hal_replay_file_record_t record;
    IREE_RETURN_IF_ERROR(iree_hal_replay_file_parse_record(
        valid_contents, record_offset, &record, &record_offset));
    if (record.header.sequence_ordinal != expected_sequence_ordinal++) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "replay record sequence ordinal mismatch");
    }

    ++out_summary->record_count;
    switch (record.header.record_type) {
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
        ++out_summary->object_count;
        if (record.header.payload_type ==
            IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT) {
          IREE_RETURN_IF_ERROR(iree_hal_replay_dump_summary_scan_file_object(
              &record, out_summary));
        }
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
        ++out_summary->operation_count;
        IREE_RETURN_IF_ERROR(
            iree_hal_replay_dump_summary_scan_operation(&record, out_summary));
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
        ++out_summary->unsupported_count;
        break;
      default:
        break;
    }
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_replay_dump_file(iree_const_byte_span_t file_contents,
                          const iree_hal_replay_dump_options_t* options,
                          iree_hal_replay_dump_write_callback_t write_callback,
                          iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(write_callback.fn);
  if (IREE_UNLIKELY(options->format != IREE_HAL_REPLAY_DUMP_FORMAT_TEXT &&
                    options->format != IREE_HAL_REPLAY_DUMP_FORMAT_JSONL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported replay dump format");
  }

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_file_parse_header(file_contents, &file_header, &offset));

  iree_const_byte_span_t valid_contents = file_contents;
  if (file_header.file_length != 0) {
    valid_contents.data_length = (iree_host_size_t)file_header.file_length;
  } else {
    file_header.file_length = file_contents.data_length;
  }
  iree_hal_replay_dump_file_summary_t summary;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_dump_scan_summary(valid_contents, offset, &summary));

  iree_hal_replay_dump_context_t context = {
      .write_callback = write_callback,
      .host_allocator = host_allocator,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  iree_status_t status = iree_ok_status();
  if (options->format == IREE_HAL_REPLAY_DUMP_FORMAT_TEXT) {
    status = iree_hal_replay_dump_emit_text_file(&context, &builder,
                                                 &file_header, &summary);
  } else {
    status =
        iree_hal_replay_dump_emit_json_file(&context, &builder, &file_header);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_replay_dump_emit_json_summary(&context, &builder, &summary);
    }
  }

  uint64_t expected_sequence_ordinal = 0;
  while (iree_status_is_ok(status) && offset < valid_contents.data_length) {
    const iree_host_size_t record_offset = offset;
    iree_hal_replay_file_record_t record;
    status = iree_hal_replay_file_parse_record(valid_contents, record_offset,
                                               &record, &offset);
    if (!iree_status_is_ok(status)) break;

    if (record.header.sequence_ordinal != expected_sequence_ordinal) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay record sequence ordinal mismatch");
      break;
    }
    ++expected_sequence_ordinal;

    iree_hal_replay_file_range_t payload_range =
        iree_hal_replay_dump_record_payload_range(&record, record_offset);
    if (options->format == IREE_HAL_REPLAY_DUMP_FORMAT_TEXT) {
      status = iree_hal_replay_dump_emit_text_record(
          &context, &builder, &record, &payload_range, record_offset);
    } else {
      status = iree_hal_replay_dump_emit_json_record(
          &context, &builder, &record, &payload_range, record_offset);
    }
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}
