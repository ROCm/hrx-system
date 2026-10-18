// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/event.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loom/util/json.h"

#define LOOM_REALTIME_TYPE(type) (1u << IREE_JSON_VALUE_TYPE_##type)
static const loom_serve_realtime_field_rule_t loom_serve_realtime_fields[] = {
    {"type", LOOM_REALTIME_TYPE(STRING)},
    {"event_id", LOOM_REALTIME_TYPE(STRING)},
    {"session", LOOM_REALTIME_TYPE(OBJECT)},
    {"audio", LOOM_REALTIME_TYPE(STRING)},
    {"response", LOOM_REALTIME_TYPE(OBJECT)},
    {"response_id", LOOM_REALTIME_TYPE(STRING)},
    {"item", LOOM_REALTIME_TYPE(OBJECT)},
    {"previous_item_id", LOOM_REALTIME_TYPE(STRING)},
    {"item_id", LOOM_REALTIME_TYPE(STRING)},
    {"content_index", LOOM_REALTIME_TYPE(NUMBER)},
    {"audio_end_ms", LOOM_REALTIME_TYPE(NUMBER)},
};
#undef LOOM_REALTIME_TYPE

typedef struct loom_serve_realtime_event_rule_t {
  // Decoded command name, indexed by loom_serve_realtime_event_kind_t.
  const char* name;
  // Required fields, excluding the common type field.
  uint32_t required;
  // Optional fields, excluding the common event_id field.
  uint32_t optional;
} loom_serve_realtime_event_rule_t;

#define LOOM_REALTIME_FIELD(field) (1u << LOOM_SERVE_REALTIME_FIELD_##field)
static const loom_serve_realtime_event_rule_t loom_serve_realtime_events[] = {
    {"session.update", LOOM_REALTIME_FIELD(SESSION), 0},
    {"input_audio_buffer.append", LOOM_REALTIME_FIELD(AUDIO), 0},
    {"input_audio_buffer.commit", 0, 0},
    {"input_audio_buffer.clear", 0, 0},
    {"response.create", 0, LOOM_REALTIME_FIELD(RESPONSE)},
    {"response.cancel", 0, LOOM_REALTIME_FIELD(RESPONSE_ID)},
    {"conversation.item.create", LOOM_REALTIME_FIELD(ITEM),
     LOOM_REALTIME_FIELD(PREVIOUS_ITEM_ID)},
    {"conversation.item.delete", LOOM_REALTIME_FIELD(ITEM_ID), 0},
    {"conversation.item.truncate",
     LOOM_REALTIME_FIELD(ITEM_ID) | LOOM_REALTIME_FIELD(CONTENT_INDEX) |
         LOOM_REALTIME_FIELD(AUDIO_END_MS),
     0},
};
#undef LOOM_REALTIME_FIELD

typedef struct loom_serve_realtime_object_t {
  // Immutable field descriptors established by the command's static schema.
  const loom_serve_realtime_field_rule_t* rules;
  // Number of descriptors and output values, at most 32.
  iree_host_size_t count;
  // Caller-owned indexed results, borrowing the parsed object.
  loom_serve_realtime_event_value_t* values;
  // Observed keys, also detecting duplicates of decoded spellings.
  uint32_t present;
} loom_serve_realtime_object_t;

static iree_status_t loom_serve_realtime_object_visit(
    void* user_data, iree_string_view_t key, iree_json_value_type_t type,
    iree_string_view_t value) {
  loom_serve_realtime_object_t* object = user_data;
  char decoded_key[32];
  iree_host_size_t key_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(key, sizeof(decoded_key),
                                                 decoded_key, &key_length));
  key = iree_make_string_view(decoded_key, key_length);
  for (iree_host_size_t i = 0; i < object->count; ++i) {
    const loom_serve_realtime_field_rule_t* rule = &object->rules[i];
    if (!iree_string_view_equal(key, iree_make_cstring_view(rule->name))) {
      continue;
    }
    if ((object->present & (1u << i)) || !(rule->types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed realtime field '%s'",
                              rule->name);
    }
    object->values[i] = (loom_serve_realtime_event_value_t){type, value};
    object->present |= 1u << i;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported realtime field '%.*s'", (int)key.size,
                          key.data);
}

iree_status_t loom_serve_realtime_object_parse(
    iree_string_view_t object, iree_host_size_t count,
    const loom_serve_realtime_field_rule_t* rules,
    loom_serve_realtime_event_value_t* values, uint32_t* present) {
  memset(values, 0, count * sizeof(*values));
  loom_serve_realtime_object_t parser = {rules, count, values, 0};
  const iree_status_t status = iree_json_enumerate_object_typed(
      object, loom_serve_realtime_object_visit, &parser);
  *present = parser.present;
  return status;
}

iree_status_t loom_serve_realtime_string_decode(
    iree_string_view_t encoded, iree_host_size_t limit,
    iree_string_builder_t* decoded) {
  iree_string_builder_reset(decoded);
  const iree_host_size_t capacity = iree_min(encoded.size, limit);
  IREE_RETURN_IF_ERROR(iree_string_builder_reserve(decoded, capacity + 1));
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(encoded, capacity, decoded->buffer, &length));
  if (!iree_unicode_utf8_validate(
          iree_make_string_view(decoded->buffer, length))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "realtime string requires valid UTF-8");
  }
  iree_string_builder_commit_append(decoded, length);
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_string_write(iree_string_builder_t* output,
                                               iree_string_view_t text) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  return loom_json_write_escaped_string(&stream, text);
}

iree_status_t loom_serve_realtime_event_parse(
    iree_string_view_t message, loom_serve_realtime_event_t* out_event) {
  *out_event = (loom_serve_realtime_event_t){0};
  iree_string_view_t cursor = message, object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing realtime message content");
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      object, IREE_ARRAYSIZE(loom_serve_realtime_fields),
      loom_serve_realtime_fields, out_event->fields, &out_event->present));
  if (!(out_event->present & (1u << LOOM_SERVE_REALTIME_FIELD_TYPE))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "realtime event requires type");
  }
  char decoded_type[64];
  iree_host_size_t type_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      out_event->fields[LOOM_SERVE_REALTIME_FIELD_TYPE].text,
      sizeof(decoded_type), decoded_type, &type_length));
  const iree_string_view_t type =
      iree_make_string_view(decoded_type, type_length);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_serve_realtime_events);
       ++i) {
    const loom_serve_realtime_event_rule_t* rule =
        &loom_serve_realtime_events[i];
    if (!iree_string_view_equal(type, iree_make_cstring_view(rule->name))) {
      continue;
    }
    const uint32_t allowed = rule->required | rule->optional |
                             (1u << LOOM_SERVE_REALTIME_FIELD_TYPE) |
                             (1u << LOOM_SERVE_REALTIME_FIELD_EVENT_ID);
    if ((out_event->present & rule->required) != rule->required ||
        (out_event->present & ~allowed)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing or inapplicable fields for '%s'",
                              rule->name);
    }
    out_event->kind = (loom_serve_realtime_event_kind_t)i;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "unsupported realtime event '%.*s'", (int)type.size,
                          type.data);
}
