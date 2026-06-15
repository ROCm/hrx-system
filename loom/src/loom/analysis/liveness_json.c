// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_json.h"

#include <inttypes.h>

#include "loom/format/text/printer.h"
#include "loom/ir/module.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_string_view_t loom_liveness_type_kind_name(
    loom_type_kind_t type_kind) {
  switch (type_kind) {
    case LOOM_TYPE_NONE:
      return IREE_SV("none");
    case LOOM_TYPE_SCALAR:
      return IREE_SV("scalar");
    case LOOM_TYPE_TILE:
      return IREE_SV("tile");
    case LOOM_TYPE_TENSOR:
      return IREE_SV("tensor");
    case LOOM_TYPE_GROUP:
      return IREE_SV("group");
    case LOOM_TYPE_FUNCTION:
      return IREE_SV("function");
    case LOOM_TYPE_DIALECT:
      return IREE_SV("dialect");
    case LOOM_TYPE_ENCODING:
      return IREE_SV("encoding");
    case LOOM_TYPE_POOL:
      return IREE_SV("pool");
    case LOOM_TYPE_VECTOR:
      return IREE_SV("vector");
    case LOOM_TYPE_VIEW:
      return IREE_SV("view");
    case LOOM_TYPE_BUFFER:
      return IREE_SV("buffer");
    case LOOM_TYPE_REGISTER:
      return IREE_SV("register");
    case LOOM_TYPE_STORAGE:
      return IREE_SV("storage");
    case LOOM_TYPE_COUNT_:
      break;
  }
  return IREE_SV("unknown");
}

static iree_status_t loom_liveness_json_write_string_or_null(
    const loom_module_t* module, loom_string_id_t string_id,
    loom_output_stream_t* stream) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return loom_output_stream_write_cstring(stream, "null");
  }
  return loom_json_write_escaped_string(stream,
                                        module->strings.entries[string_id]);
}

static iree_status_t loom_liveness_json_write_scalar_name_or_null(
    loom_scalar_type_t scalar_type, loom_output_stream_t* stream) {
  const char* name = loom_scalar_type_name(scalar_type);
  if (!name) return loom_output_stream_write_cstring(stream, "null");
  return loom_json_write_escaped_cstring(stream, name);
}

static iree_status_t loom_liveness_json_write_type(
    const loom_module_t* module,
    const loom_text_print_options_t* type_print_options, loom_type_t type,
    loom_output_stream_t* stream) {
  loom_json_escape_stream_t escape_data;
  loom_output_stream_t escape_stream;
  loom_json_escape_stream_init(stream, &escape_data, &escape_stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '"'));
  if (type_print_options) {
    IREE_RETURN_IF_ERROR(loom_text_print_type_with_options(
        type, module, &escape_stream, type_print_options));
  } else {
    IREE_RETURN_IF_ERROR(loom_text_print_type(type, module, &escape_stream));
  }
  return loom_output_stream_write_char(stream, '"');
}

static iree_status_t loom_liveness_json_write_value(
    const loom_liveness_analysis_t* analysis,
    const loom_text_print_options_t* type_print_options,
    loom_value_id_t value_id, loom_output_stream_t* stream) {
  const loom_module_t* module = analysis->module;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"id\":%u,\"name\":", (unsigned)value_id));
  if (value_id < module->values.count) {
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(loom_liveness_json_write_string_or_null(
        module, value->name_id, stream));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"type\":"));
    IREE_RETURN_IF_ERROR(loom_liveness_json_write_type(
        module, type_print_options, value->type, stream));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, "null,\"type\":null"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_liveness_json_write_value_id_array(
    const loom_value_id_t* values, iree_host_size_t count,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, '['));
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(stream, ','));
    }
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_format(stream, "%u", (unsigned)values[i]));
  }
  return loom_output_stream_write_char(stream, ']');
}

static iree_status_t loom_liveness_json_write_value_class(
    const loom_liveness_analysis_t* analysis,
    loom_liveness_value_class_t value_class, loom_output_stream_t* stream) {
  (void)analysis;
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"type_kind\":%u,\"type_kind_name\":",
      (unsigned)value_class.type_kind));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      stream, loom_liveness_type_kind_name(value_class.type_kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"element_type\":%u,\"element_type_name\":",
      (unsigned)value_class.element_type));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_scalar_name_or_null(
      value_class.element_type, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
      stream, ",\"register_descriptor_set\":"));
  if (value_class.type_kind == LOOM_TYPE_REGISTER) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, "%" PRIu64 ",\"register_class_id\":%u",
        value_class.register_descriptor_set_stable_id,
        (unsigned)value_class.register_class_id));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, "null,\"register_class_id\":null"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static int32_t loom_liveness_json_block_index(
    const loom_liveness_analysis_t* analysis, const loom_block_t* block) {
  if (!block) return -1;
  for (iree_host_size_t i = 0; i < analysis->block_count; ++i) {
    if (analysis->blocks[i].block == block) return (int32_t)i;
  }
  return -1;
}

static int32_t loom_liveness_json_op_index(const loom_block_t* block,
                                           const loom_op_t* op) {
  if (!block || !op) return -1;
  int32_t op_index = 0;
  const loom_op_t* candidate = NULL;
  loom_block_for_each_op(block, candidate) {
    if (candidate == op) return op_index;
    ++op_index;
  }
  return -1;
}

static iree_status_t loom_liveness_json_write_block(
    const loom_liveness_analysis_t* analysis, iree_host_size_t block_index,
    loom_output_stream_t* stream) {
  const loom_liveness_block_info_t* block_info = &analysis->blocks[block_index];
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, "{\"index\":%zu,\"label\":", (size_t)block_index));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_string_or_null(
      analysis->module, block_info->block->label_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"start_point\":%u,\"end_point\":%u,\"live_in\":",
      block_info->start_point, block_info->end_point));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_value_id_array(
      block_info->live_in_values, block_info->live_in_count, stream));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"live_out\":"));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_value_id_array(
      block_info->live_out_values, block_info->live_out_count, stream));
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_liveness_json_write_interval(
    const loom_liveness_analysis_t* analysis,
    const loom_text_print_options_t* type_print_options,
    const loom_liveness_interval_t* interval, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"value\":"));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_value(
      analysis, type_print_options, interval->value_id, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ",\"class\":"));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_value_class(
      analysis, interval->value_class, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"start_point\":%u,\"end_point\":%u,\"unit_count\":%u}",
      interval->start_point, interval->end_point, interval->unit_count));
  return iree_ok_status();
}

static iree_status_t loom_liveness_json_write_nullable_i32(
    loom_output_stream_t* stream, int32_t value) {
  if (value < 0) return loom_output_stream_write_cstring(stream, "null");
  return loom_output_stream_write_format(stream, "%d", value);
}

static iree_status_t loom_liveness_json_write_pressure_summary(
    const loom_liveness_analysis_t* analysis,
    const loom_liveness_pressure_summary_t* summary,
    loom_output_stream_t* stream) {
  int32_t block_index =
      loom_liveness_json_block_index(analysis, summary->peak_block);
  int32_t op_index =
      block_index < 0
          ? -1
          : loom_liveness_json_op_index(summary->peak_block, summary->peak_op);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{\"class\":"));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_value_class(
      analysis, summary->value_class, stream));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"peak_live_units\":%u,\"peak_live_values\":%u,\"peak_block\":",
      summary->peak_live_units, summary->peak_live_values));
  IREE_RETURN_IF_ERROR(
      loom_liveness_json_write_nullable_i32(stream, block_index));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"peak_op\":"));
  IREE_RETURN_IF_ERROR(loom_liveness_json_write_nullable_i32(stream, op_index));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"peak_point\":%u}", summary->peak_point));
  return iree_ok_status();
}

iree_status_t loom_liveness_format_json(
    const loom_liveness_analysis_t* analysis,
    const loom_text_print_options_t* type_print_options,
    iree_string_builder_t* builder) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      &stream,
      "{\"format\":\"loom.liveness.v0\",\"is_cfg\":%s,\"block_count\":%zu,"
      "\"value_count\":%zu,\"blocks\":[",
      analysis->is_cfg ? "true" : "false", (size_t)analysis->block_count,
      (size_t)analysis->value_count));
  for (iree_host_size_t i = 0; i < analysis->block_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_liveness_json_write_block(analysis, i, &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"intervals\":["));
  for (iree_host_size_t i = 0; i < analysis->interval_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_liveness_json_write_interval(
        analysis, type_print_options, &analysis->intervals[i], &stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(&stream, "],\"pressure_summaries\":["));
  for (iree_host_size_t i = 0; i < analysis->pressure_summary_count; ++i) {
    if (i > 0) {
      IREE_RETURN_IF_ERROR(loom_output_stream_write_char(&stream, ','));
    }
    IREE_RETURN_IF_ERROR(loom_liveness_json_write_pressure_summary(
        analysis, &analysis->pressure_summaries[i], &stream));
  }
  return loom_output_stream_write_cstring(&stream, "]}");
}
