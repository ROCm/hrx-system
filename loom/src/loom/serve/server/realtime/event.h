// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_EVENT_H_
#define LOOM_SERVE_SERVER_REALTIME_EVENT_H_

#include "iree/base/internal/json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_realtime_event_kind_e {
  LOOM_SERVE_REALTIME_SESSION_UPDATE,
  LOOM_SERVE_REALTIME_INPUT_APPEND,
  LOOM_SERVE_REALTIME_INPUT_COMMIT,
  LOOM_SERVE_REALTIME_INPUT_CLEAR,
  LOOM_SERVE_REALTIME_RESPONSE_CREATE,
  LOOM_SERVE_REALTIME_RESPONSE_CANCEL,
  LOOM_SERVE_REALTIME_ITEM_CREATE,
  LOOM_SERVE_REALTIME_ITEM_DELETE,
  LOOM_SERVE_REALTIME_ITEM_TRUNCATE,
} loom_serve_realtime_event_kind_t;

typedef enum loom_serve_realtime_event_field_e {
  LOOM_SERVE_REALTIME_FIELD_TYPE,
  LOOM_SERVE_REALTIME_FIELD_EVENT_ID,
  LOOM_SERVE_REALTIME_FIELD_SESSION,
  LOOM_SERVE_REALTIME_FIELD_AUDIO,
  LOOM_SERVE_REALTIME_FIELD_RESPONSE,
  LOOM_SERVE_REALTIME_FIELD_RESPONSE_ID,
  LOOM_SERVE_REALTIME_FIELD_ITEM,
  LOOM_SERVE_REALTIME_FIELD_PREVIOUS_ITEM_ID,
  LOOM_SERVE_REALTIME_FIELD_ITEM_ID,
  LOOM_SERVE_REALTIME_FIELD_CONTENT_INDEX,
  LOOM_SERVE_REALTIME_FIELD_AUDIO_END_MS,
  LOOM_SERVE_REALTIME_FIELD_COUNT,
} loom_serve_realtime_event_field_t;

typedef struct loom_serve_realtime_event_value_t {
  // Validated JSON type; inspect only for a present field.
  iree_json_value_type_t type;
  // Encoded value borrowing the message, with string quotes removed.
  iree_string_view_t text;
} loom_serve_realtime_event_value_t;

typedef struct loom_serve_realtime_field_rule_t {
  // Decoded member spelling; array order is the consumer's field index.
  const char* name;
  // Accepted JSON types as bit positions of iree_json_value_type_t.
  uint32_t types;
} loom_serve_realtime_field_rule_t;

// Parses an event-owned object once into directly indexed fields. Rule tables
// contain at most 32 distinct decoded keys. Duplicate/unknown/wrongly typed
// fields reject before the owning command mutates anything. present receives
// one bit per member; empty strings and null remain distinct from absence.
iree_status_t loom_serve_realtime_object_parse(
    iree_string_view_t object, iree_host_size_t count,
    const loom_serve_realtime_field_rule_t* rules,
    loom_serve_realtime_event_value_t* values, uint32_t* present);

// Decodes one checked JSON string into an owning builder with a byte bound.
// A failure leaves scratch deinitializable, never a partially applied setting.
iree_status_t loom_serve_realtime_string_decode(iree_string_view_t encoded,
                                                iree_host_size_t limit,
                                                iree_string_builder_t* decoded);
// Appends a quoted/escaped JSON string to a wire or native-request builder.
iree_status_t loom_serve_realtime_string_write(iree_string_builder_t* output,
                                               iree_string_view_t text);

typedef struct loom_serve_realtime_event_t {
  // Selected command after the entire outer object passes validation.
  loom_serve_realtime_event_kind_t kind;
  // Bit per field distinguishing absent values from empty strings and null.
  uint32_t present;
  // Directly indexed checked fields; nested policy belongs to that command.
  loom_serve_realtime_event_value_t fields[LOOM_SERVE_REALTIME_FIELD_COUNT];
} loom_serve_realtime_event_t;

// Parses one complete UTF-8 WebSocket text message without copying payloads.
// Unknown, repeated, wrongly typed, missing, and command-inapplicable fields
// reject before any session mutation. Escaped member names and event types
// compare by their decoded meaning. Nested settings/item/response objects are
// retained for their owning command's validation; no model lookup occurs here.
// All returned views borrow the message receipt until it is released.
iree_status_t loom_serve_realtime_event_parse(
    iree_string_view_t message, loom_serve_realtime_event_t* out_event);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_EVENT_H_
