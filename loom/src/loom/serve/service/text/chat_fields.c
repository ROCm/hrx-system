// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/chat_fields.h"

typedef struct loom_serve_text_chat_fields_t {
  // Number of declared fields.
  iree_host_size_t count;
  // Mutable decoding destinations.
  loom_serve_text_chat_field_t* values;
  // Whether descriptive schema keys outside this view may pass through.
  loom_serve_text_chat_field_flags_t flags;
} loom_serve_text_chat_fields_t;

static iree_status_t loom_serve_text_chat_field_visit(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_text_chat_fields_t* fields = user_data;
  for (iree_host_size_t i = 0; i < fields->count; ++i) {
    loom_serve_text_chat_field_t* field = &fields->values[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(field->name))) {
      continue;
    }
    if (field->value.data || !(field->types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed chat field '%s'",
                              field->name);
    }
    field->value = value;
    field->type = type;
    return iree_ok_status();
  }
  if (iree_any_bit_set(fields->flags,
                       LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported chat field '%.*s'", (int)key.size,
                          key.data);
}

iree_status_t loom_serve_text_chat_fields_read(
    iree_string_view_t object, iree_host_size_t count,
    loom_serve_text_chat_field_t* fields,
    loom_serve_text_chat_field_flags_t flags) {
  loom_serve_text_chat_fields_t context = {count, fields, flags};
  return iree_json_enumerate_object_typed(
      object, loom_serve_text_chat_field_visit, &context);
}

iree_status_t loom_serve_text_chat_required(
    const loom_serve_text_chat_field_t* field) {
  if (!field->value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing chat field '%s'", field->name);
  }
  return iree_ok_status();
}

bool loom_serve_text_chat_is_identifier(iree_string_view_t name) {
  if (!name.size || name.size > 128) {
    return false;
  }
  for (iree_host_size_t i = 0; i < name.size; ++i) {
    const char c = name.data[i];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) {
      return false;
    }
  }
  return true;
}
