// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/report.h"

#include <inttypes.h>
#include <string.h>

#include "loom/pass/environment.h"
#include "loom/pass/program.h"
#include "loom/pass/registry.h"
#include "loom/util/json.h"

void loom_pass_report_initialize(iree_allocator_t allocator,
                                 loom_pass_report_t* out_report) {
  *out_report = (loom_pass_report_t){
      .allocator = allocator,
  };
}

static void loom_pass_report_free_detail_list(
    loom_pass_report_t* report, loom_pass_report_detail_t* detail) {
  while (detail) {
    loom_pass_report_detail_t* next = detail->next;
    for (uint16_t i = 0; i < detail->field_count; ++i) {
      loom_pass_report_detail_field_t* field = &detail->fields[i];
      iree_allocator_free(report->allocator, (void*)field->name.data);
      if (field->value_kind == LOOM_PASS_REPORT_DETAIL_VALUE_STRING) {
        iree_allocator_free(report->allocator, (void*)field->string_value.data);
      }
    }
    iree_allocator_free(report->allocator, detail->fields);
    iree_allocator_free(report->allocator, (void*)detail->category.data);
    iree_allocator_free(report->allocator, detail);
    detail = next;
  }
}

void loom_pass_report_deinitialize(loom_pass_report_t* report) {
  for (iree_host_size_t i = 0; i < report->invocation_count; ++i) {
    iree_allocator_free(report->allocator, report->invocations[i].statistics);
    loom_pass_report_free_detail_list(report, report->invocations[i].details);
  }
  iree_allocator_free(report->allocator, report->invocations);
  memset(report, 0, sizeof(*report));
}

static iree_status_t loom_pass_report_ensure_invocation_capacity(
    loom_pass_report_t* report) {
  if (report->invocation_count < report->invocation_capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity =
      report->invocation_capacity > 0 ? report->invocation_capacity * 2 : 16;
  IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
      report->allocator, new_capacity, sizeof(*report->invocations),
      (void**)&report->invocations));
  memset(report->invocations + report->invocation_capacity, 0,
         (new_capacity - report->invocation_capacity) *
             sizeof(*report->invocations));
  report->invocation_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_pass_report_copy_statistics(
    loom_pass_report_t* report, const loom_pass_info_t* pass_info,
    const void* statistic_storage,
    loom_pass_report_invocation_t* out_invocation) {
  const loom_pass_statistic_layout_t* layout = pass_info->statistic_layout;
  out_invocation->statistic_count = layout ? layout->field_count : 0;
  if (!layout || layout->field_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(report->allocator, layout->field_count,
                                  sizeof(*out_invocation->statistics),
                                  (void**)&out_invocation->statistics));
  for (uint16_t i = 0; i < layout->field_count; ++i) {
    const loom_pass_statistic_field_t* field = &layout->fields[i];
    int64_t value = 0;
    if (statistic_storage) {
      const uint8_t* storage = (const uint8_t*)statistic_storage;
      value = *(const int64_t*)(const void*)(storage + field->offset);
    }
    out_invocation->statistics[i] = (loom_pass_report_statistic_t){
        .name = field->name,
        .value = value,
    };
  }
  return iree_ok_status();
}

iree_status_t loom_pass_report_append_invocation(
    loom_pass_report_t* report,
    const loom_pass_report_invocation_options_t* options) {
  if (!report || !options || !options->instruction ||
      options->instruction->kind != LOOM_PASS_PROGRAM_INSTRUCTION_INVOKE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass report invocation requires an invoke instruction");
  }
  const loom_pass_program_invoke_t* invoke = &options->instruction->invoke;
  if (!invoke->descriptor || !invoke->info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass report invocation requires descriptor metadata");
  }

  IREE_RETURN_IF_ERROR(loom_pass_report_ensure_invocation_capacity(report));
  loom_pass_report_invocation_t invocation = {
      .descriptor = invoke->descriptor,
      .pass_key = invoke->descriptor->key,
      .pass_kind = invoke->info->kind,
      .anchor_kind = options->anchor_kind,
      .pipeline_symbol = options->pipeline_symbol,
      .symbol_name = options->symbol_name,
      .instruction_index = options->instruction_index,
      .duration_nanoseconds = options->duration_nanoseconds,
      .changed = options->changed,
      .status_code = options->status_code,
  };
  IREE_RETURN_IF_ERROR(loom_pass_report_copy_statistics(
      report, invoke->info, options->statistic_storage, &invocation));
  invocation.details = options->detail_head;
  invocation.detail_count = options->detail_count;
  report->invocations[report->invocation_count++] = invocation;
  return iree_ok_status();
}

static iree_status_t loom_pass_report_copy_string(
    loom_pass_report_t* report, iree_string_view_t source,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) {
    return iree_ok_status();
  }
  void* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      report->allocator, iree_make_const_byte_span(source.data, source.size),
      &data));
  *out_target = iree_make_string_view((const char*)data, source.size);
  return iree_ok_status();
}

static iree_status_t loom_pass_report_copy_detail_fields(
    loom_pass_report_t* report, const loom_pass_report_detail_field_t* fields,
    uint16_t field_count, loom_pass_report_detail_field_t** out_fields) {
  *out_fields = NULL;
  if (field_count == 0) {
    return iree_ok_status();
  }
  loom_pass_report_detail_field_t* copied_fields = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      report->allocator, field_count, sizeof(*copied_fields),
      (void**)&copied_fields));
  iree_status_t status = iree_ok_status();
  uint16_t copied_count = 0;
  for (uint16_t i = 0; i < field_count && iree_status_is_ok(status); ++i) {
    copied_fields[i] = fields[i];
    status = loom_pass_report_copy_string(report, fields[i].name,
                                          &copied_fields[i].name);
    if (!iree_status_is_ok(status)) break;
    ++copied_count;
    if (fields[i].value_kind == LOOM_PASS_REPORT_DETAIL_VALUE_STRING) {
      status = loom_pass_report_copy_string(report, fields[i].string_value,
                                            &copied_fields[i].string_value);
    }
  }
  if (!iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < copied_count; ++i) {
      iree_allocator_free(report->allocator, (void*)copied_fields[i].name.data);
      if (copied_fields[i].value_kind == LOOM_PASS_REPORT_DETAIL_VALUE_STRING) {
        iree_allocator_free(report->allocator,
                            (void*)copied_fields[i].string_value.data);
      }
    }
    iree_allocator_free(report->allocator, copied_fields);
    return status;
  }
  *out_fields = copied_fields;
  return iree_ok_status();
}

iree_status_t loom_pass_report_append_detail(
    loom_pass_t* pass, iree_string_view_t category,
    const loom_pass_report_detail_field_t* fields, uint16_t field_count) {
  if (!loom_pass_report_is_enabled(pass)) {
    return iree_ok_status();
  }
  if (field_count > 0 && !fields) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass report detail row requires fields when field_count is nonzero");
  }

  loom_pass_report_t* report = pass->report;
  loom_pass_report_detail_t* detail = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(report->allocator, sizeof(*detail),
                                             (void**)&detail));
  memset(detail, 0, sizeof(*detail));

  iree_status_t status =
      loom_pass_report_copy_string(report, category, &detail->category);
  if (iree_status_is_ok(status)) {
    status = loom_pass_report_copy_detail_fields(report, fields, field_count,
                                                 &detail->fields);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(report->allocator, (void*)detail->category.data);
    iree_allocator_free(report->allocator, detail);
    return status;
  }
  detail->field_count = field_count;

  if (pass->report_detail_tail) {
    pass->report_detail_tail->next = detail;
  } else {
    pass->report_detail_head = detail;
  }
  pass->report_detail_tail = detail;
  ++pass->report_detail_count;
  return iree_ok_status();
}

void loom_pass_report_discard_pass_details(loom_pass_t* pass) {
  if (!loom_pass_report_is_enabled(pass)) {
    return;
  }
  loom_pass_report_t* report = pass->report;
  loom_pass_report_free_detail_list(report, pass->report_detail_head);
  pass->report_detail_head = NULL;
  pass->report_detail_tail = NULL;
  pass->report_detail_count = 0;
}

static iree_string_view_t loom_pass_report_kind_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return IREE_SV("module");
    case LOOM_PASS_FUNCTION:
      return IREE_SV("func");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_pass_report_write_json_string_field(
    loom_output_stream_t* stream, const char* name, iree_string_view_t value) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_format(stream, "\"%s\":", name));
  return loom_json_write_escaped_string(stream, value);
}

static iree_string_view_t loom_pass_report_option_schema_kind_name(
    loom_pass_option_schema_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_OPTION_SCHEMA_STRING:
      return IREE_SV("string");
    case LOOM_PASS_OPTION_SCHEMA_UINT32:
      return IREE_SV("uint32");
    case LOOM_PASS_OPTION_SCHEMA_ENUM:
      return IREE_SV("enum");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_pass_report_format_detail_value_json(
    const loom_pass_report_detail_field_t* field,
    loom_output_stream_t* stream) {
  switch (field->value_kind) {
    case LOOM_PASS_REPORT_DETAIL_VALUE_STRING:
      return loom_json_write_escaped_string(stream, field->string_value);
    case LOOM_PASS_REPORT_DETAIL_VALUE_INT64:
      return loom_output_stream_write_format(stream, "%" PRId64,
                                             field->int64_value);
    case LOOM_PASS_REPORT_DETAIL_VALUE_UINT64:
      return loom_output_stream_write_format(stream, "%" PRIu64,
                                             field->uint64_value);
    case LOOM_PASS_REPORT_DETAIL_VALUE_BOOL:
      return loom_output_stream_write_cstring(
          stream, field->bool_value ? "true" : "false");
    case LOOM_PASS_REPORT_DETAIL_VALUE_NONE:
    default:
      return loom_output_stream_write_cstring(stream, "null");
  }
}

static iree_status_t loom_pass_report_format_details_json(
    const loom_pass_report_invocation_t* invocation,
    loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"details\":["));
  const loom_pass_report_detail_t* detail = invocation->details;
  for (iree_host_size_t i = 0; i < invocation->detail_count && detail;
       ++i, detail = detail->next) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "{" : ",{"));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "category", detail->category));
    for (uint16_t j = 0; j < detail->field_count; ++j) {
      const loom_pass_report_detail_field_t* field = &detail->fields[j];
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(stream, field->name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ":"));
      IREE_RETURN_IF_ERROR(
          loom_pass_report_format_detail_value_json(field, stream));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "]");
}

static iree_status_t loom_pass_report_format_option_schema_json(
    const loom_pass_option_schema_t* schema, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_pass_report_write_json_string_field(stream, "name", schema->name));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
      stream, "kind", loom_pass_report_option_schema_kind_name(schema->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"required\":%s",
      iree_all_bits_set(
          schema->flags,
          (loom_pass_option_schema_flags_t)LOOM_PASS_OPTION_SCHEMA_REQUIRED)
          ? "true"
          : "false"));
  if (schema->kind == LOOM_PASS_OPTION_SCHEMA_UINT32) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream, ",\"minimum\":%" PRIu32 ",\"maximum\":%" PRIu32,
        schema->minimum_uint32, schema->maximum_uint32));
  } else if (schema->kind == LOOM_PASS_OPTION_SCHEMA_ENUM) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"values\":["));
    for (uint16_t i = 0; i < schema->enum_value_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, i == 0 ? "" : ","));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(stream, schema->enum_values[i].value));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "]"));
  }
  return loom_output_stream_write_cstring(stream, "}");
}

static iree_status_t loom_pass_report_format_descriptor_json(
    const loom_pass_descriptor_t* descriptor, loom_output_stream_t* stream) {
  const loom_pass_info_t* info = descriptor->info();
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }

  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "{"));
  IREE_RETURN_IF_ERROR(
      loom_pass_report_write_json_string_field(stream, "key", descriptor->key));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
      stream, "kind", loom_pass_report_kind_name(info->kind)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
  IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
      stream, "description", info->description));
  bool available = loom_pass_descriptor_is_available(descriptor);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      stream, ",\"available\":%s", available ? "true" : "false"));
  if (!available) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"unavailable_reason\":"));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(stream, descriptor->unavailable_reason));
  }

  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, ",\"options\":["));
  for (uint16_t i = 0; i < descriptor->option_schema_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "" : ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_format_option_schema_json(
        &descriptor->option_schema[i], stream));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "],\"statistics\":["));
  const loom_pass_statistic_layout_t* statistic_layout = info->statistic_layout;
  uint16_t statistic_count =
      statistic_layout ? statistic_layout->field_count : 0;
  for (uint16_t i = 0; i < statistic_count; ++i) {
    const loom_pass_statistic_field_t* statistic = &statistic_layout->fields[i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "{" : ",{"));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "name", statistic->name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "description", statistic->description));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "],\"requirements\":["));
  for (uint16_t i = 0; i < descriptor->requirement_count; ++i) {
    const loom_pass_requirement_def_t* requirement =
        &descriptor->requirement_defs[i];
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, i == 0 ? "{" : ",{"));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "key", requirement->key));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "capability", requirement->capability_type->name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "description", requirement->description));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  return loom_output_stream_write_cstring(stream, "]}");
}

iree_status_t loom_pass_report_format_json(const loom_pass_report_t* report,
                                           loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "{\n  \"invocations\": ["));
  for (iree_host_size_t i = 0; i < report->invocation_count; ++i) {
    const loom_pass_report_invocation_t* invocation = &report->invocations[i];
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, i == 0 ? "\n    {" : ",\n    {"));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "pass", invocation->pass_key));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "kind", loom_pass_report_kind_name(invocation->pass_kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "anchor", loom_pass_report_kind_name(invocation->anchor_kind)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "pipeline", invocation->pipeline_symbol));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, ","));
    IREE_RETURN_IF_ERROR(loom_pass_report_write_json_string_field(
        stream, "symbol", invocation->symbol_name));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        stream,
        ",\"instruction_index\":%" PRIhsz ",\"duration_ns\":%" PRId64
        ",\"changed\":%s,\"status\":",
        invocation->instruction_index, invocation->duration_nanoseconds,
        invocation->changed ? "true" : "false"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_cstring(
        stream, iree_status_code_string(invocation->status_code)));
    IREE_RETURN_IF_ERROR(
        loom_output_stream_write_cstring(stream, ",\"statistics\":{"));
    for (uint16_t j = 0; j < invocation->statistic_count; ++j) {
      const loom_pass_report_statistic_t* statistic =
          &invocation->statistics[j];
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(stream, j == 0 ? "" : ","));
      IREE_RETURN_IF_ERROR(
          loom_json_write_escaped_string(stream, statistic->name));
      IREE_RETURN_IF_ERROR(loom_output_stream_write_format(stream, ":%" PRId64,
                                                           statistic->value));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
    IREE_RETURN_IF_ERROR(
        loom_pass_report_format_details_json(invocation, stream));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "}"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n  ]\n}\n"));
  return iree_ok_status();
}

iree_status_t loom_pass_report_format_registry_json(
    const loom_pass_registry_t* registry, loom_output_stream_t* stream) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(stream, "{\n  \"passes\": ["));
  for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(
        stream, i == 0 ? "\n    " : ",\n    "));
    IREE_RETURN_IF_ERROR(loom_pass_report_format_descriptor_json(
        &registry->descriptors[i], stream));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(stream, "\n  ]\n}\n"));
  return iree_ok_status();
}
