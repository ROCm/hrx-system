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
  // Required fact family, or zero for universally available physical facts.
  loom_target_compile_report_pipeline_facts_t facts;
} loom_target_compile_report_pipeline_numeric_field_t;

#define LOOM_PIPELINE_NUMERIC_FIELD(struct_type, member, field_kind, facts) \
  {#member, (uint16_t)offsetof(struct_type, member),                        \
   (uint8_t)(sizeof(#member) - 1), field_kind, facts}
#define LOOM_PIPELINE_U32_FIELD(struct_type, member) \
  LOOM_PIPELINE_NUMERIC_FIELD(                       \
      struct_type, member, LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U32, 0)
#define LOOM_PIPELINE_U64_FIELD(struct_type, member) \
  LOOM_PIPELINE_NUMERIC_FIELD(                       \
      struct_type, member, LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U64, 0)
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

#define LOGICAL_FIELD(type, member, kind)                               \
  LOOM_PIPELINE_NUMERIC_FIELD(                                          \
      type, member, LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_##kind, \
      LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_CHANNELS)
#define WORKER_LOGICAL(member) \
  LOGICAL_FIELD(loom_target_compile_report_pipeline_worker_row_t, member, U32)
#define SUMMARY_LOGICAL(member, kind)                                       \
  LOGICAL_FIELD(loom_target_compile_report_pipeline_plan_summary_t, member, \
                kind)
#define SUMMARY_INVENTORY(member, kind, family)                   \
  LOOM_PIPELINE_NUMERIC_FIELD(                                    \
      loom_target_compile_report_pipeline_plan_summary_t, member, \
      LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_##kind,         \
      LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_##family)
#define MEMORY_U32(member)                                                  \
  LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_pipeline_memory_row_t, \
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
        WORKER_LOGICAL(group_index),
        WORKER_LOGICAL(lane),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_worker_metric_fields[] = {
        WORKER_LOGICAL(input_channel_count),
        WORKER_LOGICAL(output_channel_count),
        WORKER_LOGICAL(ring_state_count),
        WORKER_LOGICAL(input_record_count),
        WORKER_LOGICAL(output_record_count),
        WORKER_LOGICAL(finite_loop_trip_count),
        WORKER_U32(code_byte_count),
        WORKER_U32(code_capacity_byte_count),
        WORKER_U32(worker_storage_byte_count),
        WORKER_LOGICAL(channel_storage_byte_count),
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
        SUMMARY_LOGICAL(group_count, U32),
        SUMMARY_U32(binding_count),
        SUMMARY_INVENTORY(program_count, U32, PROGRAMS),
        SUMMARY_INVENTORY(program_code_byte_count, U64, PROGRAMS),
        SUMMARY_INVENTORY(memory_count, U32, MEMORY),
        SUMMARY_INVENTORY(reserved_storage_byte_count, U64, MEMORY),
        SUMMARY_LOGICAL(channel_slot_count, U32),
        SUMMARY_LOGICAL(hardware_lock_count, U32),
        SUMMARY_LOGICAL(dma_channel_count, U32),
        SUMMARY_LOGICAL(dma_buffer_descriptor_count, U32),
        SUMMARY_LOGICAL(route_count, U32),
        SUMMARY_U64(worker_code_byte_count),
        SUMMARY_U32(maximum_worker_code_byte_count),
        SUMMARY_U32(minimum_worker_code_headroom_byte_count),
        SUMMARY_U64(worker_storage_byte_count),
        SUMMARY_LOGICAL(channel_storage_byte_count, U64),
        SUMMARY_U32(maximum_tile_local_memory_byte_count),
        SUMMARY_U32(minimum_tile_local_memory_headroom_byte_count),
        SUMMARY_U32(maximum_bank_storage_byte_count),
        SUMMARY_U32(bank_storage_capacity_byte_count),
        SUMMARY_LOGICAL(external_dma_byte_count, U64),
        SUMMARY_LOGICAL(routed_dma_byte_count, U64),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    loom_target_compile_report_pipeline_memory_fields[] = {
        MEMORY_U32(memory_index),
        MEMORY_U32(reserved_byte_count),
        MEMORY_U32(program_data_byte_count),
        MEMORY_U32(occupied_byte_count),
        MEMORY_U32(high_water_byte_count),
        MEMORY_U32(capacity_byte_count),
        MEMORY_U32(maximum_bank_storage_byte_count),
        MEMORY_U32(bank_storage_capacity_byte_count),
};

static const loom_target_compile_report_pipeline_numeric_field_t
    placed_function_fields[] = {
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_placed_function_t,
                                worker_index),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_placed_function_t,
                                group),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_placed_function_t,
                                address),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_placed_function_t,
                                code_byte_count),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_placed_function_t,
                                stored_packet_count),
};
static const loom_target_compile_report_pipeline_numeric_field_t
    packet_loop_fields[] = {
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_packet_loop_t,
                                byte_offset),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_packet_loop_t,
                                packet_count),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_packet_loop_t,
                                trip_count),
        LOOM_PIPELINE_U32_FIELD(loom_target_compile_report_packet_loop_t,
                                empty_packet_count),
};

#undef MEMORY_U32
#undef SUMMARY_INVENTORY
#undef SUMMARY_LOGICAL
#undef WORKER_LOGICAL
#undef LOGICAL_FIELD
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
    iree_host_size_t field_count,
    loom_target_compile_report_pipeline_facts_t available_facts,
    loom_json_object_writer_t* object) {
  const uint8_t* bytes = value;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    const loom_target_compile_report_pipeline_numeric_field_t* field =
        &fields[i];
    if (field->facts && !iree_any_bit_set(available_facts, field->facts)) {
      continue;
    }
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
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_storage_fields), 0,
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
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_transfer_fields), 0,
      &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pipeline_placement_json(
    const loom_target_compile_report_pipeline_placement_t* value,
    loom_json_object_writer_t* object) {
  loom_output_stream_t* stream = object->stream;
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("placement")));
  loom_json_object_writer_t placement;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &placement));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &placement, IREE_SV("rank"), value->rank));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&placement, IREE_SV("x"), value->x));
  IREE_RETURN_IF_ERROR(
      loom_json_object_write_uint32_field(&placement, IREE_SV("y"), value->y));
  if (value->rank >= 3) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &placement, IREE_SV("z"), value->z));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&placement));
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_pipeline_memory_json(
    const loom_target_compile_report_pipeline_memory_row_t* row,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_pipeline_placement_json(&row->placement,
                                                                &object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      row, loom_target_compile_report_pipeline_memory_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_memory_fields), 0,
      &object));
  return loom_json_object_end(&object);
}

static iree_status_t loom_target_compile_report_format_pipeline_worker_json(
    const loom_target_compile_report_pipeline_worker_row_t* row,
    loom_target_compile_report_pipeline_facts_t available_facts,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      row, loom_target_compile_report_pipeline_worker_identity_fields,
      IREE_ARRAYSIZE(
          loom_target_compile_report_pipeline_worker_identity_fields),
      available_facts, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("entry"), row->entry_name));
  if (iree_any_bit_set(available_facts,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_CHANNELS)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("resident"),
        iree_any_bit_set(row->flags,
                         LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_RESIDENT)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_bool_field(
        &object, IREE_SV("folded"),
        iree_any_bit_set(row->flags,
                         LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_FOLDED)));
  }
  if (iree_any_bit_set(row->flags,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_PLACED)) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_placement_json(
            &row->placement, &object));
  }
  if (!iree_string_view_is_empty(row->fold_kind)) {
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("fold_kind"), row->fold_kind));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      row, loom_target_compile_report_pipeline_worker_metric_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_worker_metric_fields),
      available_facts, &object));
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
      0, &object));
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

static iree_status_t loom_target_compile_report_format_placed_functions_json(
    const loom_target_compile_report_pipeline_plan_t* plan,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("placed_functions")));
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &array));
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    const loom_target_compile_report_placed_function_t* function =
        &plan->functions[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t row;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &row));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &row, IREE_SV("function"), function->name));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_placement_json(
            &function->placement, &row));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_numeric_json(
            function, placed_function_fields,
            IREE_ARRAYSIZE(placed_function_fields), 0, &row));
    IREE_RETURN_IF_ERROR(
        function->has_packets_per_call
            ? loom_json_object_write_uint64_field(
                  &row, IREE_SV("packets_per_call"), function->packets_per_call)
            : loom_json_object_write_null_field(&row,
                                                IREE_SV("packets_per_call")));
    IREE_RETURN_IF_ERROR(
        function->has_calls_per_entry
            ? loom_json_object_write_uint64_field(
                  &row, IREE_SV("calls_per_entry"), function->calls_per_entry)
            : loom_json_object_write_null_field(&row,
                                                IREE_SV("calls_per_entry")));
    if (function->loop_count) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&row, IREE_SV("packet_loops")));
      loom_json_array_writer_t loops;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(object->stream, &loops));
      for (uint32_t j = 0; j < function->loop_count; ++j) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&loops));
        loom_json_object_writer_t loop;
        IREE_RETURN_IF_ERROR(loom_json_object_begin(object->stream, &loop));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pipeline_numeric_json(
                &plan->packet_loops[function->first_loop + j],
                packet_loop_fields, IREE_ARRAYSIZE(packet_loop_fields), 0,
                &loop));
        IREE_RETURN_IF_ERROR(loom_json_object_end(&loop));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&loops));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&row));
  }
  return loom_json_array_end(&array);
}

iree_status_t loom_target_compile_report_format_pipeline_copies_json(
    const loom_target_compile_report_row_list_t* copies,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (const loom_target_compile_report_vec_t* vec = copies->head; vec;
       vec = vec->next) {
    const loom_target_compile_report_pipeline_copy_t* rows =
        loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_pipeline_copy_t* row = &rows[i];
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
      loom_json_object_writer_t object;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &object, IREE_SV("root"), row->root));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &object, IREE_SV("phase"), row->phase));
      IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
          &object, IREE_SV("occurrence"), row->occurrence));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("binding"), row->binding));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &object, IREE_SV("direction"),
          row->read ? IREE_SV("read") : IREE_SV("write")));
      if (row->bytes_per_copy >= 0) {
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &object, IREE_SV("bytes_per_copy"), (uint64_t)row->bytes_per_copy));
      } else {
        IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
            &object, IREE_SV("bytes_per_copy")));
      }
      if (row->visits_known) {
        IREE_RETURN_IF_ERROR(loom_json_object_write_uint64_field(
            &object, IREE_SV("copies_per_phase_call"), row->visits));
      } else {
        IREE_RETURN_IF_ERROR(loom_json_object_write_null_field(
            &object, IREE_SV("copies_per_phase_call")));
      }
      IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
    }
  }
  return loom_json_array_end(&array);
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
  if (plan->function_count) {
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_placed_functions_json(plan, &object));
  }
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_json(
      summary, loom_target_compile_report_pipeline_summary_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_summary_fields),
      summary->available_facts, &object));

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
              &plan->worker_rows[i], summary->available_facts, stream));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&rows));
  }
  IREE_RETURN_IF_ERROR(loom_json_object_end(&workers));

  if (iree_any_bit_set(summary->available_facts,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_CHANNELS)) {
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
  }
  if (iree_any_bit_set(summary->available_facts,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_MEMORY)) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("memories")));
    loom_json_object_writer_t memories;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &memories));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &memories, IREE_SV("count"), summary->memory_count));
    if (mode == LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS &&
        plan->memory_row_count != 0) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_begin_field(&memories, IREE_SV("rows")));
      loom_json_array_writer_t rows;
      IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &rows));
      for (iree_host_size_t i = 0; i < plan->memory_row_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&rows));
        IREE_RETURN_IF_ERROR(
            loom_target_compile_report_format_pipeline_memory_json(
                &plan->memory_rows[i], stream));
      }
      IREE_RETURN_IF_ERROR(loom_json_array_end(&rows));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&memories));
  }
  return loom_json_object_end(&object);
}

static iree_string_view_t loom_target_compile_report_pipeline_text_value(
    iree_string_view_t value) {
  return iree_string_view_is_empty(value) ? IREE_SV("-") : value;
}

static iree_status_t loom_target_compile_report_format_pipeline_numeric_text(
    const void* value,
    const loom_target_compile_report_pipeline_numeric_field_t* fields,
    iree_host_size_t field_count,
    loom_target_compile_report_pipeline_facts_t available_facts,
    iree_string_builder_t* builder) {
  const uint8_t* bytes = value;
  for (iree_host_size_t i = 0; i < field_count; ++i) {
    const loom_target_compile_report_pipeline_numeric_field_t* field =
        &fields[i];
    if (field->facts && !iree_any_bit_set(available_facts, field->facts)) {
      continue;
    }
    const void* field_value = bytes + field->offset;
    const uint64_t number =
        field->kind == LOOM_TARGET_COMPILE_REPORT_PIPELINE_NUMERIC_U32
            ? *(const uint32_t*)field_value
            : *(const uint64_t*)field_value;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " %s=%" PRIu64, field->name, number));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_append_optional_count(
    iree_string_builder_t* builder, bool exact, uint64_t count) {
  return exact ? iree_string_builder_append_format(builder, "%" PRIu64, count)
               : iree_string_builder_append_cstring(builder, "unknown");
}

iree_status_t loom_target_compile_report_format_pipeline_copies_text(
    const loom_target_compile_report_row_list_t* copies,
    iree_string_builder_t* builder) {
  for (const loom_target_compile_report_vec_t* vec = copies->head; vec;
       vec = vec->next) {
    const loom_target_compile_report_pipeline_copy_t* rows =
        loom_target_compile_report_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_target_compile_report_pipeline_copy_t* row = &rows[i];
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          "COMPILE-REPORT: pipeline.copy root=%.*s phase=%.*s "
          "occurrence=%" PRIhsz " binding=%u direction=%s bytes_per_copy=",
          (int)row->root.size, row->root.data, (int)row->phase.size,
          row->phase.data, row->occurrence, row->binding,
          row->read ? "read" : "write"));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_count(
          builder, row->bytes_per_copy >= 0, (uint64_t)row->bytes_per_copy));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          builder, " copies_per_phase_call="));
      IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_count(
          builder, row->visits_known, row->visits));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_format_placed_functions_text(
    const loom_target_compile_report_pipeline_plan_t* plan,
    iree_string_builder_t* builder) {
  for (iree_host_size_t i = 0; i < plan->function_count; ++i) {
    const loom_target_compile_report_placed_function_t* row =
        &plan->functions[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: placed_function @%.*s placement=%u,%u",
        (int)row->name.size, row->name.data, row->placement.x,
        row->placement.y));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_numeric_text(
            row, placed_function_fields, IREE_ARRAYSIZE(placed_function_fields),
            0, builder));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " packets_per_call="));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_count(
        builder, row->has_packets_per_call, row->packets_per_call));
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, " calls_per_entry="));
    IREE_RETURN_IF_ERROR(loom_target_compile_report_append_optional_count(
        builder, row->has_calls_per_entry, row->calls_per_entry));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    for (uint32_t j = 0; j < row->loop_count; ++j) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "COMPILE-REPORT: packet_loop @%.*s", (int)row->name.size,
          row->name.data));
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_format_pipeline_numeric_text(
              &plan->packet_loops[row->first_loop + j], packet_loop_fields,
              IREE_ARRAYSIZE(packet_loop_fields), 0, builder));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
    }
  }
  return iree_ok_status();
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
      builder, "COMPILE-REPORT: pipeline root=%.*s realization=%.*s workers=%u",
      (int)root.size, root.data, (int)realization.size, realization.data,
      summary->worker_count));
  IREE_RETURN_IF_ERROR(loom_target_compile_report_format_pipeline_numeric_text(
      summary, loom_target_compile_report_pipeline_summary_fields,
      IREE_ARRAYSIZE(loom_target_compile_report_pipeline_summary_fields),
      summary->available_facts, builder));
  if (iree_any_bit_set(summary->available_facts,
                       LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_CHANNELS)) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " channels=%u", summary->channel_count));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  if (mode != LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_format_placed_functions_text(plan, builder));

  for (iree_host_size_t i = 0; i < plan->worker_row_count; ++i) {
    const loom_target_compile_report_pipeline_worker_row_t* row =
        &plan->worker_rows[i];
    const iree_string_view_t entry =
        loom_target_compile_report_pipeline_text_value(row->entry_name);
    const iree_string_view_t fold =
        loom_target_compile_report_pipeline_text_value(row->fold_kind);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "COMPILE-REPORT: pipeline.worker[%" PRIhsz
        "] entry=%.*s placement=%u,%u",
        i, (int)entry.size, entry.data, row->placement.x, row->placement.y));
    if (iree_any_bit_set(summary->available_facts,
                         LOOM_TARGET_COMPILE_REPORT_PIPELINE_FACT_CHANNELS)) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " group=%u lane=%u resident=%u folded=%u fold=%.*s",
          row->group_index, row->lane,
          iree_any_bit_set(row->flags,
                           LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_RESIDENT),
          iree_any_bit_set(row->flags,
                           LOOM_TARGET_COMPILE_REPORT_PIPELINE_WORKER_FOLDED),
          (int)fold.size, fold.data));
    }
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_numeric_text(
            row, loom_target_compile_report_pipeline_worker_metric_fields,
            IREE_ARRAYSIZE(
                loom_target_compile_report_pipeline_worker_metric_fields),
            summary->available_facts, builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  for (iree_host_size_t i = 0; i < plan->memory_row_count; ++i) {
    const loom_target_compile_report_pipeline_memory_row_t* row =
        &plan->memory_rows[i];
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "COMPILE-REPORT: pipeline.memory[%" PRIhsz "] placement=%u,%u",
        i, row->placement.x, row->placement.y));
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_format_pipeline_numeric_text(
            row, loom_target_compile_report_pipeline_memory_fields,
            IREE_ARRAYSIZE(loom_target_compile_report_pipeline_memory_fields),
            0, builder));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
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
