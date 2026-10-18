// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_ITEM_H_
#define LOOM_SERVE_SERVER_REALTIME_ITEM_H_

#include "loom/serve/server/realtime/conversation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validated command value, independent of canonical identity allocation.
typedef struct loom_serve_realtime_item_input_t {
  // Decoded optional client identity; empty requests generated identity.
  iree_string_builder_t id;
  // Decoded function identity; optional only for a new function call.
  iree_string_builder_t call_id;
  // Owned message text, function arguments, or free-text result.
  iree_string_builder_t text;
  // Validated ASCII function name borrowing the command through consumption.
  iree_string_view_t name;
  // Disjoint message, call, or result semantics.
  loom_serve_realtime_item_kind_t kind;
  // Validated semantic speaker for message values.
  loom_serve_realtime_role_t role;
} loom_serve_realtime_item_input_t;

// Decodes without reserving a session identity. Deinitialize on either outcome;
// name borrows object, and all other variable-length bytes belong to the value.
iree_status_t loom_serve_realtime_item_input_initialize(
    iree_string_view_t object, iree_allocator_t host_allocator,
    loom_serve_realtime_item_input_t* input);
void loom_serve_realtime_item_input_deinitialize(
    loom_serve_realtime_item_input_t* input);

// Decodes a client item into reserved, owned conversation storage. Message
// text, function calls, and free-text function results have disjoint field
// contracts. Client status is descriptive and has no effect on native history;
// accepted input is complete. Success does not insert history; failure releases
// the reservation and leaves out_item NULL.
iree_status_t loom_serve_realtime_item_parse(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t object,
    loom_serve_realtime_item_t** out_item);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_ITEM_H_
