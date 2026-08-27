// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_hazard_plan_json.h"

#include <inttypes.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/util/json.h"

static iree_string_view_t loom_low_packet_hazard_plan_json_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[symbol->name_id];
}

static iree_string_view_t loom_low_packet_hazard_plan_json_function_name(
    const loom_low_schedule_table_t* schedule) {
  if (loom_low_function_def_isa(schedule->function_op)) {
    return loom_low_packet_hazard_plan_json_symbol_name(
        schedule->module, loom_low_function_callee(schedule->function_op));
  }
  return IREE_SV("<unnamed>");
}

iree_string_view_t loom_low_packet_progress_action_name(
    loom_low_packet_progress_action_t action) {
  switch (action) {
    case LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE:
      return IREE_SV("advance");
    case LOOM_LOW_PACKET_PROGRESS_ACTION_RESET:
      return IREE_SV("reset");
    case LOOM_LOW_PACKET_PROGRESS_ACTION_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_low_packet_hazard_plan_record_kind_name(
    loom_low_packet_hazard_plan_record_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION:
      return IREE_SV("action");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA:
      return IREE_SV("missing_target_data");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNSUPPORTED_PRE_ALLOCATION:
      return IREE_SV("unsupported_pre_allocation");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_IMPOSSIBLE_SATISFACTION:
      return IREE_SV("impossible_satisfaction");
    case LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_low_packet_hazard_plan_write_nullable_host_size(
    iree_host_size_t value, iree_host_size_t sentinel,
    loom_output_stream_t* stream) {
  if (value == sentinel) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%zu", value);
}

static iree_status_t loom_low_packet_hazard_plan_write_nullable_u32(
    uint32_t value, uint32_t sentinel, loom_output_stream_t* stream) {
  if (value == sentinel) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu32, value);
}

static iree_status_t loom_low_packet_hazard_plan_write_nullable_u16(
    uint16_t value, uint16_t sentinel, loom_output_stream_t* stream) {
  if (value == sentinel) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_output_stream_write_format(stream, "%" PRIu16, value);
}

static iree_status_t loom_low_packet_hazard_plan_write_nullable_string(
    iree_string_view_t value, loom_output_stream_t* stream) {
  if (iree_string_view_is_empty(value)) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream, value);
}

static iree_status_t loom_low_packet_hazard_plan_validate_for_json(
    const loom_low_packet_hazard_plan_t* plan) {
  if (plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet hazard-plan JSON requires a plan");
  }
  if (plan->schedule == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet hazard-plan JSON requires a schedule");
  }
  if (plan->progress != NULL && plan->progress->schedule != plan->schedule) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "packet hazard-plan JSON progress table must use plan schedule");
  }
  if (plan->allocation != NULL && plan->progress != NULL &&
      plan->progress->allocation != plan->allocation) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "packet hazard-plan JSON progress table must use plan allocation");
  }
  return iree_ok_status();
}

iree_status_t loom_low_packet_progress_write_json_array(
    const loom_low_packet_progress_table_t* progress,
    loom_output_stream_t* stream) {
  if (stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet progress JSON stream is required");
  }
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  const iree_host_size_t record_count = progress ? progress->record_count : 0;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    const loom_low_packet_progress_record_t* record = &progress->records[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_host_size_field(&object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("packet"), record->packet_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("node"), record->node_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("block"), record->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("scheduled_ordinal"), record->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("class_id"), record->progress_class_id));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("class_name"), record->progress_class_name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("action"),
        loom_low_packet_progress_action_name(record->action)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("units"), record->units));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

iree_status_t loom_low_packet_hazard_plan_write_json_array(
    const loom_low_packet_hazard_plan_t* plan, loom_output_stream_t* stream) {
  if (stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet hazard-plan JSON stream is required");
  }
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  const iree_host_size_t record_count = plan ? plan->record_count : 0;
  for (iree_host_size_t i = 0; i < record_count; ++i) {
    const loom_low_packet_hazard_plan_record_t* record = &plan->records[i];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(
        loom_json_object_write_host_size_field(&object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("kind"),
        loom_low_packet_hazard_plan_record_kind_name(record->kind)));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("action_id")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u16(
        record->action_id, LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("action_name")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_string(
        record->action_name, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("reason_id"), record->reason_id));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("reason_name"), record->reason_name));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("producer_node")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u32(
        record->producer_node_index, LOOM_LOW_SCHEDULE_NODE_NONE, stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("producer_packet")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_host_size(
        record->producer_packet_index, LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE,
        stream));
    IREE_RETURN_IF_ERROR(loom_json_object_begin_field(
        &object, IREE_SV("producer_scheduled_ordinal")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u32(
        record->producer_scheduled_ordinal,
        LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE, stream));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("consumer_node"), record->consumer_node_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("insertion_packet"), record->insertion_packet_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("block"), record->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("scheduled_ordinal"), record->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("progress_class_id")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u16(
        record->progress_class_id, LOOM_LOW_PACKET_PROGRESS_CLASS_NONE,
        stream));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("progress_class_name")));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_string(
        record->progress_class_name, stream));
    if (record->progress_class_id == LOOM_LOW_PACKET_PROGRESS_CLASS_NONE) {
      IREE_RETURN_IF_ERROR(
          loom_json_object_write_null_field(&object, IREE_SV("required")));
      IREE_RETURN_IF_ERROR(
          loom_json_object_write_null_field(&object, IREE_SV("observed")));
      IREE_RETURN_IF_ERROR(
          loom_json_object_write_null_field(&object, IREE_SV("residual")));
    } else {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("required"), record->required_progress));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("observed"), record->observed_progress));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &object, IREE_SV("residual"), record->residual_progress));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_low_packet_hazard_plan_write_json_members(
    const loom_low_packet_hazard_plan_t* plan,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_for_json(plan));
  const loom_low_packet_progress_table_t* progress = plan->progress;
  const iree_host_size_t progress_count =
      progress != NULL ? progress->record_count : 0;
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("progress_count"), progress_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      object, IREE_SV("hazard_count"), plan->record_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("progress")));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_progress_write_json_array(progress, object->stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(object, IREE_SV("hazards")));
  return loom_low_packet_hazard_plan_write_json_array(plan, object->stream);
}

iree_status_t loom_low_packet_hazard_plan_format_json(
    const loom_low_packet_hazard_plan_t* plan, iree_string_builder_t* builder) {
  if (builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet hazard-plan JSON builder is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_for_json(plan));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  const loom_low_schedule_table_t* schedule = plan->schedule;
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.low.packet_hazard_plan.v1")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("function"),
      loom_low_packet_hazard_plan_json_function_name(schedule)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("target"), schedule->target.target_name));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("descriptor_set"), schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_hazard_plan_write_json_members(plan, &object));
  return loom_json_object_end(&object);
}
