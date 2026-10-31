// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/tool_schemas.h"

#include <stdlib.h>
#include <string.h>

#include "loom/serve/service/text/chat_fields.h"

static iree_status_t loom_serve_text_tool_schemas_parse_entry(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_tool_schemas_t* schemas = user_data;
  if (index >= LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tools must contain at most %d entries",
                            LOOM_SERVE_TEXT_CHAT_MAX_TOOL_COUNT);
  }
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_ok_status();
  }
  loom_serve_text_chat_field_t fields[] = {
      {"type", LOOM_SERVE_TEXT_CHAT_ANY},
      {"function", LOOM_SERVE_TEXT_CHAT_ANY}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      value, IREE_ARRAYSIZE(fields), fields,
      LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN));
  if (fields[0].type != IREE_JSON_VALUE_TYPE_STRING ||
      !iree_string_view_equal(fields[0].value, IREE_SV("function")) ||
      fields[1].type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_ok_status();
  }
  loom_serve_text_chat_field_t function[] = {
      {"name", LOOM_SERVE_TEXT_CHAT_ANY},
      {"parameters", LOOM_SERVE_TEXT_CHAT_ANY}};
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_fields_read(
      fields[1].value, IREE_ARRAYSIZE(function), function,
      LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN));
  if (!function[0].value.data ||
      function[0].type != IREE_JSON_VALUE_TYPE_STRING) {
    return iree_ok_status();
  }
  const iree_host_size_t required_capacity = schemas->count + 1;
  if (required_capacity > schemas->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        schemas->host_allocator, required_capacity, sizeof(*schemas->values),
        &schemas->capacity, (void**)&schemas->values));
  }
  loom_serve_text_chat_tool_t* tool = &schemas->values[schemas->count++];
  *tool = (loom_serve_text_chat_tool_t){
      .name = function[0].value,
      .parameters = function[1].type == IREE_JSON_VALUE_TYPE_OBJECT
                        ? function[1].value
                        : iree_string_view_empty()};
  if (iree_string_view_find_char(tool->name, '\\', 0) !=
      IREE_STRING_VIEW_NPOS) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        schemas->host_allocator, tool->name.size, (void**)&tool->name_storage));
    iree_host_size_t length = 0;
    IREE_RETURN_IF_ERROR(iree_json_unescape_string(
        tool->name, tool->name.size, tool->name_storage, &length));
    tool->name = iree_make_string_view(tool->name_storage, length);
  }
  return iree_ok_status();
}

static int loom_serve_text_tool_schemas_compare(const void* left,
                                                const void* right) {
  const loom_serve_text_chat_tool_t* left_tool = left;
  const loom_serve_text_chat_tool_t* right_tool = right;
  return iree_string_view_compare(left_tool->name, right_tool->name);
}

iree_status_t loom_serve_text_tool_schemas_initialize(
    iree_string_view_t declarations, iree_allocator_t host_allocator,
    loom_serve_text_tool_schemas_t* out_schemas) {
  memset(out_schemas, 0, sizeof(*out_schemas));
  out_schemas->host_allocator = host_allocator;
  iree_status_t status = iree_json_enumerate_array_typed(
      declarations, loom_serve_text_tool_schemas_parse_entry, out_schemas);
  if (iree_status_is_ok(status) && out_schemas->count > 1) {
    qsort(out_schemas->values, out_schemas->count, sizeof(*out_schemas->values),
          loom_serve_text_tool_schemas_compare);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_text_tool_schemas_deinitialize(out_schemas);
  }
  return status;
}

void loom_serve_text_tool_schemas_deinitialize(
    loom_serve_text_tool_schemas_t* schemas) {
  for (iree_host_size_t i = 0; i < schemas->count; ++i) {
    iree_allocator_free(schemas->host_allocator,
                        schemas->values[i].name_storage);
  }
  iree_allocator_free(schemas->host_allocator, schemas->values);
  memset(schemas, 0, sizeof(*schemas));
}

const loom_serve_text_chat_tool_t* loom_serve_text_tool_schemas_find(
    const loom_serve_text_tool_schemas_t* schemas, iree_string_view_t name) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = schemas->count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    const int comparison =
        iree_string_view_compare(name, schemas->values[middle].name);
    if (comparison < 0) {
      end = middle;
    } else if (comparison > 0) {
      begin = middle + 1;
    } else {
      // Repeated names do not identify an unambiguous representation hint.
      // Preserve the generated text instead of selecting an arbitrary schema.
      if ((middle > 0 &&
           iree_string_view_equal(schemas->values[middle - 1].name, name)) ||
          (middle + 1 < schemas->count &&
           iree_string_view_equal(schemas->values[middle + 1].name, name))) {
        return NULL;
      }
      return &schemas->values[middle];
    }
  }
  return NULL;
}
