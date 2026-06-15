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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  if (progress == NULL) {
    return loom_output_stream_write_char(stream, ']');
  }
  for (iree_host_size_t i = 0; i < progress->record_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_low_packet_progress_record_t* record = &progress->records[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        "{\"index\":%zu,\"packet\":%zu,\"node\":%" PRIu32 ",\"block\":%" PRIu32
        ",\"scheduled_ordinal\":%" PRIu32 ",\"class_id\":%" PRIu16
        ",\"class_name\":",
        i, record->packet_index, record->node_index, record->block_index,
        record->scheduled_ordinal, record->progress_class_id));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, record->progress_class_name));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"action\":"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_low_packet_progress_action_name(record->action)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"units\":%" PRIu32 "}", record->units));
  }
  return loom_output_stream_write_char(stream, ']');
}

iree_status_t loom_low_packet_hazard_plan_write_json_array(
    const loom_low_packet_hazard_plan_t* plan, loom_output_stream_t* stream) {
  if (stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet hazard-plan JSON stream is required");
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  if (plan == NULL) {
    return loom_output_stream_write_char(stream, ']');
  }
  for (iree_host_size_t i = 0; i < plan->record_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    const loom_low_packet_hazard_plan_record_t* record = &plan->records[i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, "{\"index\":%zu,\"kind\":", i));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        stream, loom_low_packet_hazard_plan_record_kind_name(record->kind)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"action_id\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u16(
        record->action_id, LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"action_name\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_string(
        record->action_name, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"reason_id\":%" PRIu16 ",\"reason_name\":", record->reason_id));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, record->reason_name));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"producer_node\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u32(
        record->producer_node_index, LOOM_LOW_SCHEDULE_NODE_NONE, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"producer_packet\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_host_size(
        record->producer_packet_index, LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE,
        stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, ",\"producer_scheduled_ordinal\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u32(
        record->producer_scheduled_ordinal,
        LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"consumer_node\":%" PRIu32
        ",\"insertion_packet\":%zu"
        ",\"block\":%" PRIu32 ",\"scheduled_ordinal\":%" PRIu32
        ",\"progress_class_id\":",
        record->consumer_node_index, record->insertion_packet_index,
        record->block_index, record->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_u16(
        record->progress_class_id, LOOM_LOW_PACKET_PROGRESS_CLASS_NONE,
        stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"progress_class_name\":"));
    IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_nullable_string(
        record->progress_class_name, stream));
    if (record->progress_class_id == LOOM_LOW_PACKET_PROGRESS_CLASS_NONE) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
          stream, ",\"required\":null,\"observed\":null,\"residual\":null"));
    } else {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
          stream,
          ",\"required\":%" PRIu32 ",\"observed\":%" PRIu32
          ",\"residual\":%" PRIu32,
          record->required_progress, record->observed_progress,
          record->residual_progress));
    }
    if (!iree_string_view_is_empty(record->target_detail)) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, ",\"target_detail\":"));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(stream, record->target_detail));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '}'));
  }
  return loom_output_stream_write_char(stream, ']');
}

iree_status_t loom_low_packet_hazard_plan_write_json_members(
    const loom_low_packet_hazard_plan_t* plan, loom_output_stream_t* stream) {
  if (stream == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "packet hazard-plan JSON stream is required");
  }
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_validate_for_json(plan));
  const loom_low_packet_progress_table_t* progress = plan->progress;
  const iree_host_size_t progress_count =
      progress != NULL ? progress->record_count : 0;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "\"progress_count\":%zu,\"hazard_count\":%zu,\"progress\":",
      progress_count, plan->record_count));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_progress_write_json_array(progress, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"hazards\":"));
  return loom_low_packet_hazard_plan_write_json_array(plan, stream);
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
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      &stream, "{\"format\":\"loom.low.packet_hazard_plan.v1\""));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"function\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, loom_low_packet_hazard_plan_json_function_name(schedule)));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"target\":"));
  IREE_RETURN_IF_ERROR(
      loom_json_write_escaped_string(&stream, schedule->target.target_name));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, ",\"descriptor_set\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      &stream, schedule->target.descriptor_set_key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_hazard_plan_write_json_members(plan, &stream));
  return loom_output_stream_write_char(&stream, '}');
}
