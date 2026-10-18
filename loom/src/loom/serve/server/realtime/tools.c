// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/tools.h"

#include "loom/serve/server/realtime/event.h"
#include "loom/serve/service/text/tool_schemas.h"

typedef struct loom_serve_realtime_function_range_t {
  // Offset of the complete native function object in storage.
  iree_host_size_t offset;
  // Encoded object length, including its braces.
  iree_host_size_t length;
} loom_serve_realtime_function_range_t;

struct loom_serve_realtime_tools_t {
  // Serialized owner references, including active response snapshots.
  iree_host_size_t references;
  // Owned native declaration array; allocator also owns this record.
  iree_string_builder_t storage;
  // Exact number of declarations, established before allocation.
  iree_host_size_t count;
  // Function objects indexed in source declaration order.
  loom_serve_realtime_function_range_t functions[];
};

static iree_status_t loom_serve_realtime_tools_visit(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tools requires function declaration objects");
  }
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"name", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"description", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"parameters", 1u << IREE_JSON_VALUE_TYPE_OBJECT},
      {"strict",
       (1u << IREE_JSON_VALUE_TYPE_TRUE) | (1u << IREE_JSON_VALUE_TYPE_FALSE)}};
  loom_serve_realtime_event_value_t fields[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      value, IREE_ARRAYSIZE(rules), rules, fields, &present));
  // Native function names and types use literal protocol spellings. Preserve
  // the original encoded values for the shared native schema validator.
  if (!(present & 1) ||
      !iree_string_view_equal(fields[0].text, IREE_SV("function"))) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "tools supports function declarations");
  }
  loom_serve_realtime_tools_t* tools = user_data;
  iree_string_builder_t* output = &tools->storage;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "%s{\"type\":\"function\",\"function\":", index ? "," : ""));
  tools->functions[index].offset = output->size;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "{"));
  bool comma = false;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 1;
       i < IREE_ARRAYSIZE(rules) && iree_status_is_ok(status); ++i) {
    if (!(present & (1u << i))) {
      continue;
    }
    const bool string = fields[i].type == IREE_JSON_VALUE_TYPE_STRING;
    status = iree_string_builder_append_format(output, "%s\"%s\":%s",
                                               comma ? "," : "", rules[i].name,
                                               string ? "\"" : "");
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_string(output, fields[i].text);
    }
    if (iree_status_is_ok(status) && string) {
      status = iree_string_builder_append_cstring(output, "\"");
    }
    comma = true;
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  tools->functions[index].length =
      output->size - tools->functions[index].offset;
  return iree_string_builder_append_cstring(output, "}");
}

iree_status_t loom_serve_realtime_tools_parse(
    iree_string_view_t array, iree_allocator_t host_allocator,
    loom_serve_realtime_tools_t** out_tools) {
  *out_tools = NULL;
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(iree_json_array_length(array, &count));
  if (!count) {
    return iree_ok_status();
  }
  if (count > LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many function declarations");
  }
  loom_serve_realtime_tools_t* tools = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*tools) + count * sizeof(tools->functions[0]),
      (void**)&tools));
  tools->references = 1;
  tools->count = count;
  iree_string_builder_initialize(host_allocator, &tools->storage);
  iree_status_t status =
      iree_string_builder_append_cstring(&tools->storage, "[");
  if (iree_status_is_ok(status)) {
    status = iree_json_enumerate_array_typed(
        array, loom_serve_realtime_tools_visit, tools);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(&tools->storage, "]");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_tool_schemas_t schemas;
    status = loom_serve_text_tool_schemas_initialize(
        iree_string_builder_view(&tools->storage), host_allocator, &schemas);
    loom_serve_text_tool_schemas_deinitialize(&schemas);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_tools_release(tools);
    return status;
  }
  *out_tools = tools;
  return iree_ok_status();
}

void loom_serve_realtime_tools_retain(loom_serve_realtime_tools_t* tools) {
  if (tools) {
    ++tools->references;
  }
}

void loom_serve_realtime_tools_release(loom_serve_realtime_tools_t* tools) {
  if (tools && !--tools->references) {
    const iree_allocator_t allocator = tools->storage.allocator;
    iree_string_builder_deinitialize(&tools->storage);
    iree_allocator_free(allocator, tools);
  }
}

iree_string_view_t loom_serve_realtime_tools_native(
    const loom_serve_realtime_tools_t* tools) {
  return tools ? iree_string_builder_view(&tools->storage)
               : iree_string_view_empty();
}

iree_status_t loom_serve_realtime_tools_write(
    const loom_serve_realtime_tools_t* tools, iree_string_builder_t* output) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "["));
  const iree_host_size_t count = tools ? tools->count : 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    status = iree_string_builder_append_format(
        output, "%s{\"type\":\"function\",", i ? "," : "");
    if (iree_status_is_ok(status)) {
      const loom_serve_realtime_function_range_t range = tools->functions[i];
      status = iree_string_builder_append_string(
          output,
          iree_make_string_view(tools->storage.buffer + range.offset + 1,
                                range.length - 1));
    }
  }
  IREE_RETURN_IF_ERROR(status);
  return iree_string_builder_append_cstring(output, "]");
}
