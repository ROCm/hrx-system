// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CHAT_FIELDS_H_
#define LOOM_SERVE_SERVICE_TEXT_CHAT_FIELDS_H_

#include "iree/base/api.h"
#include "iree/base/internal/json.h"

#ifdef __cplusplus
extern "C" {
#endif

// Accepted types for a field at the chat request boundary. JSON null remains
// distinct from the string "null".
enum loom_serve_text_chat_field_type_bits_e {
  LOOM_SERVE_TEXT_CHAT_STRING = 1u << IREE_JSON_VALUE_TYPE_STRING,
  LOOM_SERVE_TEXT_CHAT_NUMBER = 1u << IREE_JSON_VALUE_TYPE_NUMBER,
  LOOM_SERVE_TEXT_CHAT_OBJECT = 1u << IREE_JSON_VALUE_TYPE_OBJECT,
  LOOM_SERVE_TEXT_CHAT_ARRAY = 1u << IREE_JSON_VALUE_TYPE_ARRAY,
  LOOM_SERVE_TEXT_CHAT_NULL = 1u << IREE_JSON_VALUE_TYPE_NULL,
  LOOM_SERVE_TEXT_CHAT_BOOL =
      (1u << IREE_JSON_VALUE_TYPE_TRUE) | (1u << IREE_JSON_VALUE_TYPE_FALSE),
};

typedef struct loom_serve_text_chat_field_t {
  // Literal protocol key; escaped protocol keys are not accepted.
  const char* name;
  // Accepted JSON types as loom_serve_text_chat_field_type_bits_e bits.
  uint32_t types;
  // Borrowed value; null data denotes an absent optional field.
  iree_string_view_t value;
  // Actual JSON type, valid only when value.data is non-null.
  iree_json_value_type_t type;
} loom_serve_text_chat_field_t;

typedef enum loom_serve_text_chat_field_flag_bits_e {
  // Descriptive schema keys outside this view may pass through.
  LOOM_SERVE_TEXT_CHAT_FIELDS_IGNORE_UNKNOWN = 1u << 0,
} loom_serve_text_chat_field_flag_bits_t;
typedef uint32_t loom_serve_text_chat_field_flags_t;

// Decodes typed fields borrowing object. All destination values start empty;
// duplicate, wrongly typed, and (unless permitted) unknown fields are errors.
iree_status_t loom_serve_text_chat_fields_read(
    iree_string_view_t object, iree_host_size_t count,
    loom_serve_text_chat_field_t* fields,
    loom_serve_text_chat_field_flags_t flags);

// Requires the field to have been present in the decoded object.
iree_status_t loom_serve_text_chat_required(
    const loom_serve_text_chat_field_t* field);

// Function and XML parameter names use literal ASCII identifiers of 1..128
// bytes, consisting of letters, digits, underscores, hyphens, and periods.
bool loom_serve_text_chat_is_identifier(iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CHAT_FIELDS_H_
