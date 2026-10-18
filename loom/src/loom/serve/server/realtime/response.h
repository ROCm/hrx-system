// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_RESPONSE_H_
#define LOOM_SERVE_SERVER_REALTIME_RESPONSE_H_

#include "iree/task/executor.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/server/realtime/conversation.h"
#include "loom/serve/server/realtime/history.h"
#include "loom/serve/server/realtime/settings.h"
#include "loom/serve/server/realtime/wire.h"

#ifdef __cplusplus
extern "C" {
#endif

// One default-conversation writer and independently retiring cancelled work.
// All calls belong to the serialized session owner. CPU work borrows only
// generation-owned state and stable immutable conversion coefficients.
typedef struct loom_serve_realtime_responses_t loom_serve_realtime_responses_t;

// Borrows every dependency through retirement. Only the response records and
// shared speech coefficients belong to this owner; no model work starts here.
iree_status_t loom_serve_realtime_responses_allocate(
    uint64_t session_id, const loom_serve_realtime_profile_t* profile,
    loom_serve_realtime_binding_t text, loom_serve_realtime_binding_t speech,
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_http_server_t* server, loom_serve_wake_callback_t observer,
    loom_serve_realtime_settings_t* settings,
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_history_t* history, loom_serve_realtime_wire_t* wire,
    iree_allocator_t host_allocator,
    loom_serve_realtime_responses_t** out_owner);
// Releases an owner after pending becomes false. NULL is permitted for partial
// session construction; accepted work must have retired before this call.
void loom_serve_realtime_responses_free(loom_serve_realtime_responses_t* owner);

bool loom_serve_realtime_responses_active(
    loom_serve_realtime_responses_t* owner);
bool loom_serve_realtime_responses_available(
    const loom_serve_realtime_responses_t* owner);
bool loom_serve_realtime_responses_pending(
    const loom_serve_realtime_responses_t* owner);
iree_status_t loom_serve_realtime_responses_create(
    loom_serve_realtime_responses_t* owner, iree_string_view_t overrides);
// encoded_id borrows the command's JSON string; empty means the active writer.
iree_status_t loom_serve_realtime_responses_cancel(
    loom_serve_realtime_responses_t* owner, iree_string_view_t encoded_id);
// Activity interruption also drops audio text without claiming playback
// alignment.
void loom_serve_realtime_responses_interrupt(
    loom_serve_realtime_responses_t* owner);
// Closing withdraws every response; accepted CPU/native storage still retires.
void loom_serve_realtime_responses_stop(loom_serve_realtime_responses_t* owner);

// Validates the response ownership boundary before a conversation edit. An
// accepted truncation cancels that response; rejected edits change no state.
iree_status_t loom_serve_realtime_responses_edit(
    loom_serve_realtime_responses_t* owner, loom_serve_realtime_item_t* item,
    bool truncate, bool changes_history);

// earliest_input is the earliest unfinished committed transcription sequence,
// or UINT64_MAX when none exists. Deferred responses consume only earlier
// input.
iree_status_t loom_serve_realtime_responses_advance(
    loom_serve_realtime_responses_t* owner, bool closing,
    uint64_t earliest_input, bool* out_progress);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_RESPONSE_H_
