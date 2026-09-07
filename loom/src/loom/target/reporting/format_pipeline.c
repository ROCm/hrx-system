// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/format_pipeline.h"

#include <inttypes.h>
#include <stddef.h>

enum loom_target_compile_report_pipeline_numeric_kind_e {
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U32 = 0,
  LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U64 = 1,
};

// Compact descriptor for one numeric field in a pipeline report structure.
typedef struct loom_target_compile_report_pipeline_numeric_field_t {
  // JSON field name.
  const char* name;
  // Byte offset of the field in its containing structure.
  uint16_t offset;
  // Byte length of |name|.
  uint8_t name_length;
  // loom_target_compile_report_pipeline_numeric_kind_e value.
  uint8_t kind;
} loom_target_compile_report_pipeline_numeric_field_t;

#define LOOM_PIPELINE_NUMERIC_FIELD(struct_type, member, field_kind) \
  {#member, (uint16_t)offsetof(struct_type, member),                 \
   (uint8_t)(sizeof(#member) - 1), field_kind}
#define LOOM_PIPELINE_U32_FIELD(struct_type, member) \
  LOOM_PIPELINE_NUMERIC_FIELD(struct_type, member,   \
                              LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U32)
#define LOOM_PIPELINE_U64_FIELD(struct_type, member) \
  LOOM_PIPELINE_NUMERIC_FIELD(struct_type, member,   \
                              LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U64)
#define STORAGE_U32(member) \
  LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_pipeline_storage_t, member)
#define STORAGE_U64(member) \
  LOOM_PIPELINE_U64_FIELD(loom_target_compile_report_pipeline_storage_t, member)
#define TRANSFER_U32(member)                                              \
  LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_pipeline_transfer_t, \
                          member)
#define TRANSFER_U64(member)                                              \
  LOOM_PIPELINE_U64_FIELD(loom_target_compile_report_pipeline_transfer_t, \
                          member)
#define WORKER_U32(member)                                                  \
  LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_pipeline_worker_row_t, \
                          member)
#define CHANNEL_U32(member)                                                  \
  LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_pipeline_channel_row_t, \
                          member)
#define CHANNEL_U64(member)                                                  \
  LOOM_PIPELINE_U64_FIELD(loom_target_compile_report_pipeline_channel_row_t, \
                          member)
#define SUMMARY_U32(member)                                                   \
  LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_pipeline_plan_summary_t, \
                          member)
#define SUMMARY_U64(member)                                                   \
  LOOM_PIPELINE_U64_FIELD(loom_target_compile_report_pipeline_plan_summary_t, \
                          member)

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_storage_fields[] = {
        STORAGE_U32(schema_logical_element_count),
        STORAGE_U32(schema_record_byte_count),
        STORAGE_U32(schema_required_alignment),
        STORAGE_U32(schema_records_per_channel_record),
        STORAGE_U64(transform_extra_byte_count),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_transfer_fields[] = {
        TRANSFER_U32(binding_ordinal),
        TRANSFER_U32(partition_lane),
        TRANSFER_U32(partition_lane_count),
        TRANSFER_U64(binding_byte_offset),
        TRANSFER_U64(binding_span_byte_count),
        TRANSFER_U32(task_byte_count),
        TRANSFER_U32(task_repeat_count),
        TRANSFER_U64(activation_byte_count),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_worker_identity_fields[] = {
        WORKER_U32(worker_index),
        WORKER_U32(group_index),
        WORKER_U32(lane),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_worker_metric_fields[] = {
        WORKER_U32(input_channel_count),
        WORKER_U32(output_channel_count),
        WORKER_U32(ring_state_count),
        WORKER_U32(input_record_count),
        WORKER_U32(output_record_count),
        WORKER_U32(finite_loop_trip_count),
        WORKER_U32(code_byte_count),
        WORKER_U32(code_capacity_byte_count),
        WORKER_U32(worker_storage_byte_count),
        WORKER_U32(channel_storage_byte_count),
        WORKER_U32(local_memory_byte_count),
        WORKER_U32(local_memory_capacity_byte_count),
        WORKER_U32(maximum_bank_storage_byte_count),
        WORKER_U32(bank_storage_capacity_byte_count),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_channel_metric_fields[] = {
        CHANNEL_U32(capacity),
        CHANNEL_U32(record_count),
        CHANNEL_U32(record_byte_count),
        CHANNEL_U64(activation_byte_count),
        CHANNEL_U64(local_storage_byte_count),
        CHANNEL_U32(hardware_lock_count),
        CHANNEL_U32(dma_channel_count),
        CHANNEL_U32(dma_buffer_descriptor_count),
        CHANNEL_U32(route_count),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_summary_fields[] = {
        SUMMARY_U32(group_count),
        SUMMARY_U32(binding_count),
        SUMMARY_U32(channel_slot_count),
        SUMMARY_U32(hardware_lock_count),
        SUMMARY_U32(dma_channel_count),
        SUMMARY_U32(dma_buffer_descriptor_count),
        SUMMARY_U32(route_count),
        SUMMARY_U64(worker_code_byte_count),
        SUMMARY_U32(maximum_worker_code_byte_count),
        SUMMARY_U32(minimum_worker_code_headroom_byte_count),
        SUMMARY_U64(worker_storage_byte_count),
        SUMMARY_U64(channel_storage_byte_count),
        SUMMARY_U32(maximum_tile_local_memory_byte_count),
        SUMMARY_U32(minimum_tile_local_memory_headroom_byte_count),
        SUMMARY_U32(maximum_bank_storage_byte_count),
        SUMMARY_U32(bank_storage_capacity_byte_count),
        SUMMARY_U64(external_dma_byte_count),
        SUMMARY_U64(routed_dma_byte_count),
};

#undef SUMMARY_U64
#undef SUMMARY_U32
#undef CHANNEL_U64
#undef CHANNEL_U32
#undef WORKER_U32
#undef TRANSFER_U64
#undef TRANSFER_U32
#undef STORAGE_U64
#undef STORAGE_U32
#undef LOOM_PIPELINE_U64_FIELD
#undef LOOM_PIPELINE_U32_FIELD
#undef LOOM_PIPELINE_NUMERIC_FIELD

static iree_status_t loom_target_compile_report_format_pipeline_numeric_json(
    const void* value,
    const loom_target_compile_report_pipeline_numeric_field_t* fields,
    iree_host_size_t field_count, loom_json_object_writer_t* object) {
  const uint8_t* bytes = value;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    const loom_target_compile_report_pipeline_numeric_field_t* field =
        &fields[i];
    const iree_string_view_t name =
        iree_make_string_view(field->name, field->name_length);
    const void* field_value = bytes + field->offset;
    if (field->kind == LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U32) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          object, name, *(const uint32_t*)field_value));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
          object, name, *(const uint64_t*)field_value));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pipeline_endpoint_json(
    const loom_target_compile_report_pipeline_endpoint_t* endpoint,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("owner"),
      loom_target_compile_report_pipeline_endpoint_owner_name(
          endpoint->owner)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("owner_index"), endpoint->owner_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("port"), endpoint->port));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pipeline_storage_json(
    const loom_target_compile_report_pipeline_storage_t* storage,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("schema"), storage->schema_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("address_layout"), storage->address_layout));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("transform"), storage->transform));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      storage, loom_target_compile_report_pipeline_storage_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_storage_fields),
      &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pipeline_transfer_json(
    const loom_target_compile_report_pipeline_transfer_t* transfer,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      transfer, loom_target_compile_report_pipeline_transfer_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_transfer_fields),
      &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pipeline_worker_json(
    const loom_target_compile_report_pipeline_worker_row_t* row,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      row, loom_target_compile_report_pipeline_worker_identity_fields,
      IREE_ARRAYSIZE(
          loom_target_compile_report_pipeline_worker_identity_fields),
      &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("entry"), row->entry_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("resident"),
      iree_any_bit_set(row->flags,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_RESIDENT)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
      &object, IREE_SV("folded"),
      iree_any_bit_set(row->flags,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_FOLDED)));
  if (iree_any_bit_set(row->flags,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_PLACED)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("placement")));
    loom_json_object_writer_t placement;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &placement));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &placement, IREE_SV("rank"), row->placement.rank));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &placement, IREE_SV("x"), row->placement.x));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &placement, IREE_SV("y"), row->placement.y));
    if (row->placement.rank >= 3) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &placement, IREE_SV("z"), row->placement.z));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&placement));
  }
  if (!iree_string_view_is_empty(row->fold_kind)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("fold_kind"), row->fold_kind));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      row, loom_target_compile_report_pipeline_worker_metric_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_worker_metric_fields),
      &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pipeline_channel_json(
    const loom_target_compile_report_pipeline_channel_row_t* row,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("channel_index"), row->channel_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("transport"), row->transport));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("sender")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_endpoint_json(
      &row->sender, stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("receiver")));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_endpoint_json(
      &row->receiver, stream));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      row, loom_target_compile_report_pipeline_channel_metric_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_channel_metric_fields),
      &object));
  if (!iree_string_view_is_empty(row->storage.schema_name)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("storage")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_storage_json(&row->storage,
                                                                stream));
  }
  if (row->transfer.binding_ordinal != UINT32_MAX) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("external_transfer")));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_transfer_json(&row->transfer,
                                                                 stream));
  }
  return loom_json_object_end(&object);
}

iree_status_t loom_target_compile_report_format_pipeline_plan_json(
    const loom_target_compile_report_pipeline_plan_t* plan,
    loom_target_compile_report_format_mode_t mode,
    loom_output_stream_t* stream) {
  const loom_target_compile_report_pipeline_plan_summary_t* summary =
      &plan->summary;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("root"), summary->root_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("realization"), summary->realization));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      summary, loom_target_compile_report_pipeline_summary_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_summary_fields),
      &object));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("workers")));
  loom_json_object_writer_t workers;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &workers));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &workers, IREE_SV("count"), summary->worker_count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS &&
      plan->worker_row_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&workers, IREE_SV("rows")));
    loom_json_array_writer_t rows;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &rows));
    for (iree_host_size_t i = 0; i < plan->worker_row_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&rows));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_pipeline_worker_json(
              &plan->worker_rows[i], stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&rows));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&workers));

  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("channels")));
  loom_json_object_writer_t channels;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &channels));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &channels, IREE_SV("count"), summary->channel_count));
  if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS &&
      plan->channel_row_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&channels, IREE_SV("rows")));
    loom_json_array_writer_t rows;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &rows));
    for (iree_host_size_t i = 0; i < plan->channel_row_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&rows));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_pipeline_channel_json(
              &plan->channel_rows[i], stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&rows));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&channels));
  return loom_json_object_end(&object);
}

static iree_string_view_t loom_target_compile_report_pipeline_text_value(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

iree_status_t loom_target_compile_report_format_pipeline_plan_text(
    const loom_target_compile_report_pipeline_plan_t* plan,
    loom_target_compile_report_format_mode_t mode,
    iree_string_builder_t* builder) {
  const loom_target_compile_report_pipeline_plan_summary_t* summary =
      &plan->summary;
  const iree_string_view_t root =
      loom_target_compile_report_pipeline_text_value(summary->root_name);
  const iree_string_view_t realization =
      loom_target_compile_report_pipeline_text_value(summary->realization);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "COMPILE-REPORT: pipeline root=%.*s realization=%.*s groups=%" PRIu32
      " workers=%" PRIu32 " bindings=%" PRIu32 " channels=%" PRIu32
      " slots=%" PRIu32 " locks=%" PRIu32 " dma_channels=%" PRIu32
      " dma_bds=%" PRIu32 " routes=%" PRIu32 " code_bytes=%" PRIu64
      " max_worker_code_bytes=%" PRIu32 " min_code_headroom_bytes=%" PRIu32
      " worker_storage_bytes=%" PRIu64 " channel_storage_bytes=%" PRIu64
      " max_tile_local_bytes=%" PRIu32 " min_local_headroom_bytes=%" PRIu32
      " max_bank_storage_bytes=%" PRIu32 " bank_capacity_bytes=%" PRIu32
      " external_dma_bytes=%" PRIu64 " routed_dma_bytes=%" PRIu64 "\n",
      (int)root.size, root.data, (int)realization.size, realization.data,
      summary->group_count, summary->worker_count, summary->binding_count,
      summary->channel_count, summary->channel_slot_count,
      summary->hardware_lock_count, summary->dma_channel_count,
      summary->dma_buffer_descriptor_count, summary->route_count,
      summary->worker_code_byte_count, summary->maximum_worker_code_byte_count,
      summary->minimum_worker_code_headroom_byte_count,
      summary->worker_storage_byte_count, summary->channel_storage_byte_count,
      summary->maximum_tile_local_memory_byte_count,
      summary->minimum_tile_local_memory_headroom_byte_count,
      summary->maximum_bank_storage_byte_count,
      summary->bank_storage_capacity_byte_count,
      summary->external_dma_byte_count, summary->routed_dma_byte_count));
  if (mode != LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < plan->worker_row_count; ++i) {
    const loom_target_compile_report_pipeline_worker_row_t* row =
        &plan->worker_rows[i];
    const iree_string_view_t entry =
        loom_target_compile_report_pipeline_text_value(row->entry_name);
    const iree_string_view_t fold =
        loom_target_compile_report_pipeline_text_value(row->fold_kind);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: pipeline.worker[%" PRIhsz "] entry=%.*s group=%" PRIu32
        " lane=%" PRIu32
        " resident=%u folded=%u placement=%u,%u input_channels=%" PRIu32
        " output_channels=%" PRIu32 " ring_states=%" PRIu32
        " input_records=%" PRIu32 " output_records=%" PRIu32
        " finite_loop_trips=%" PRIu32 " fold=%.*s code_bytes=%" PRIu32
        " code_capacity_bytes=%" PRIu32 " worker_storage_bytes=%" PRIu32
        " channel_storage_bytes=%" PRIu32 " local_bytes=%" PRIu32
        " local_capacity_bytes=%" PRIu32 " max_bank_storage_bytes=%" PRIu32
        " bank_capacity_bytes=%" PRIu32 "\n",
        i, (int)entry.size, entry.data, row->group_index, row->lane,
        iree_any_bit_set(row->flags,
                         LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_RESIDENT),
        iree_any_bit_set(row->flags,
                         LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_FOLDED),
        row->placement.x, row->placement.y, row->input_channel_count,
        row->output_channel_count, row->ring_state_count,
        row->input_record_count, row->output_record_count,
        row->finite_loop_trip_count, (int)fold.size, fold.data,
        row->code_byte_count, row->code_capacity_byte_count,
        row->worker_storage_byte_count, row->channel_storage_byte_count,
        row->local_memory_byte_count, row->local_memory_capacity_byte_count,
        row->maximum_bank_storage_byte_count,
        row->bank_storage_capacity_byte_count));
  }

  for (iree_host_size_t i = 0; i < plan->channel_row_count; ++i) {
    const loom_target_compile_report_pipeline_channel_row_t* row =
        &plan->channel_rows[i];
    const iree_string_view_t transport =
        loom_target_compile_report_pipeline_text_value(row->transport);
    const iree_string_view_t sender =
        loom_target_compile_report_pipeline_endpoint_owner_name(
            row->sender.owner);
    const iree_string_view_t receiver =
        loom_target_compile_report_pipeline_endpoint_owner_name(
            row->receiver.owner);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: pipeline.channel[%" PRIhsz
        "] transport=%.*s sender=%.*s[%" PRIu32 "]:%" PRIu32
        " receiver=%.*s[%" PRIu32 "]:%" PRIu32 " capacity=%" PRIu32
        " records=%" PRIu32 " record_bytes=%" PRIu32
        " activation_bytes=%" PRIu64 " local_storage_bytes=%" PRIu64
        " locks=%" PRIu32 " dma_channels=%" PRIu32 " dma_bds=%" PRIu32
        " routes=%" PRIu32,
        i, (int)transport.size, transport.data, (int)sender.size, sender.data,
        row->sender.owner_index, row->sender.port, (int)receiver.size,
        receiver.data, row->receiver.owner_index, row->receiver.port,
        row->capacity, row->record_count, row->record_byte_count,
        row->activation_byte_count, row->local_storage_byte_count,
        row->hardware_lock_count, row->dma_channel_count,
        row->dma_buffer_descriptor_count, row->route_count));
    if (!iree_string_view_is_empty(row->storage.schema_name)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " schema=%.*s layout=%.*s transform=%.*s schema_elements=%" PRIu32
          " schema_record_bytes=%" PRIu32 " schema_alignment=%" PRIu32
          " schema_records_per_record=%" PRIu32
          " transform_extra_bytes=%" PRIu64,
          (int)row->storage.schema_name.size, row->storage.schema_name.data,
          (int)row->storage.address_layout.size,
          row->storage.address_layout.data, (int)row->storage.transform.size,
          row->storage.transform.data,
          row->storage.schema_logical_element_count,
          row->storage.schema_record_byte_count,
          row->storage.schema_required_alignment,
          row->storage.schema_records_per_channel_record,
          row->storage.transform_extra_byte_count));
    }
    if (row->transfer.binding_ordinal != UINT32_MAX) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " binding=%" PRIu32 " partition=%" PRIu32 "/%" PRIu32
          " binding_offset=%" PRIu64 " binding_span=%" PRIu64
          " task_bytes=%" PRIu32 " task_repeats=%" PRIu32
          " external_bytes=%" PRIu64,
          row->transfer.binding_ordinal, row->transfer.partition_lane,
          row->transfer.partition_lane_count, row->transfer.binding_byte_offset,
          row->transfer.binding_span_byte_count, row->transfer.task_byte_count,
          row->transfer.task_repeat_count,
          row->transfer.activation_byte_count));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}
