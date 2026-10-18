// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/session.h"

#include <inttypes.h>

#include "loom/serve/server/realtime/capture.h"
#include "loom/serve/server/realtime/conversation.h"
#include "loom/serve/server/realtime/history.h"
#include "loom/serve/server/realtime/input.h"
#include "loom/serve/server/realtime/item.h"
#include "loom/serve/server/realtime/output.h"
#include "loom/serve/server/realtime/response.h"
#include "loom/serve/server/realtime/wire.h"

typedef enum loom_serve_realtime_transcription_phase_e {
  LOOM_SERVE_REALTIME_TRANSCRIPTION_FREE,
  LOOM_SERVE_REALTIME_TRANSCRIPTION_WAITING,
  LOOM_SERVE_REALTIME_TRANSCRIPTION_READING,
  LOOM_SERVE_REALTIME_TRANSCRIPTION_FINISHED,
} loom_serve_realtime_transcription_phase_t;

typedef struct loom_serve_realtime_transcription_t {
  // Independent input lifetime, including accepted conversion after clear.
  loom_serve_realtime_input_item_t* input;
  // Start of this utterance after all earlier encoded appends publish.
  loom_serve_realtime_capture_mark_t begin;
  // Sealed commit frontier, retained independently of canonical item deletion.
  loom_serve_realtime_capture_mark_t end;
  // One captured PCM loan, stable until conversion releases it on this owner.
  loom_serve_realtime_capture_span_t span;
  // Original-rate endpoint already offered to the independent converter.
  uint64_t position;
  // Ordered start readiness, input delivery, and explicit native EOF.
  loom_serve_realtime_transcription_phase_t phase;
  // Uncommitted delivery follows classified activity rather than capture end.
  bool automatic;
  // Commit seals the PCM end independently of canonical item deletion.
  bool committed;
  // Reserved or committed canonical consumer; NULL after explicit deletion.
  loom_serve_realtime_item_t* item;
} loom_serve_realtime_transcription_t;

struct loom_serve_realtime_session_t {
  // Allocator owning this session and each bounded child representation.
  iree_allocator_t host_allocator;
  // Shared transport diagnostics and connection lifetime owner.
  loom_serve_http_server_t* server;
  // Retained HTTP claim; released only after receipt borrowers retire.
  loom_serve_http_connection_t* connection;
  // Connection-owned framed protocol, retained by the HTTP claim.
  loom_serve_websocket_t* socket;
  // Immutable application-owned assistant composition.
  const loom_serve_realtime_profile_t* profile;
  // Unique server session identity, independent of native model request IDs.
  uint64_t id;
  // Effective settings for subsequent audio items and responses.
  loom_serve_realtime_settings_t settings;
  // Canonical committed conversation, without waveform or KV copies.
  loom_serve_realtime_conversation_t conversation;
  // Independent input items sharing one finite conversion worker and filter.
  loom_serve_realtime_input_t* input;
  // Continuous original microphone clock and bounded PCM for all utterances.
  loom_serve_realtime_capture_t* capture;
  // Optional continuous activity conversion and original-clock turn policy.
  loom_serve_realtime_turn_t* turn;
  // Bounded mapping of native transcription lifetimes to conversation items.
  loom_serve_realtime_transcription_t transcriptions[4];
  // Current uncommitted microphone item, or NULL until the next append.
  loom_serve_realtime_transcription_t* microphone;
  // Latest stopped turn requests a response after independent STT retirement.
  bool automatic_response;
  // Response execution, complete-call publication, and independent retirement.
  loom_serve_realtime_responses_t* responses;
  // Completed prefixes and accepted capture/release effects.
  loom_serve_realtime_history_t history;
  // Bounded ordered output, separate from model and client-playback progress.
  loom_serve_realtime_wire_t wire;
  // No further commands or model work may be admitted.
  bool closing;
  // Broken transport or server shutdown requires abort rather than close drain.
  bool aborted;
};

static iree_status_t loom_serve_realtime_session_event(
    loom_serve_realtime_session_t* session, const char* type,
    const loom_serve_realtime_settings_t* settings) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(&session->wire, type));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &session->wire.scratch, ",\"session\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_settings_write(
      settings, session->profile, session->id, &session->wire.scratch));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->wire.scratch, "}"));
  return loom_serve_realtime_wire_commit(&session->wire, 0, false);
}

static iree_status_t loom_serve_realtime_session_error(
    loom_serve_realtime_session_t* session, iree_status_t failure,
    const char* code, iree_string_view_t event_id, bool server_failure) {
  loom_serve_http_server_record_error(session->server, failure);
  iree_status_t status =
      loom_serve_realtime_wire_begin(&session->wire, "error");
  iree_string_builder_t* output = &session->wire.scratch;
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        output,
        ",\"error\":{\"type\":\"%s\",\"code\":\"%s\","
        "\"message\":",
        server_failure ? "server_error" : "invalid_request_error",
        code ? code : iree_status_code_string(iree_status_code(failure)));
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_realtime_string_write(output, iree_status_message(failure));
  }
  if (iree_status_is_ok(status) && event_id.size) {
    status = iree_string_builder_append_cstring(output, ",\"event_id\":");
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_string_write(output, event_id);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(output, "}}");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_wire_commit(&session->wire, 0, false);
  }
  iree_status_ignore(failure);
  return status;
}

static void loom_serve_realtime_transcription_close(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_transcription_t* transcription) {
  if (transcription->item) {
    transcription->item->input = NULL;
  }
  if (transcription->input) {
    loom_serve_realtime_input_close(transcription->input);
    transcription->input = NULL;
  }
  loom_serve_realtime_capture_mark_cancel(&transcription->begin);
  loom_serve_realtime_capture_mark_cancel(&transcription->end);
  if (transcription->item && !transcription->committed) {
    loom_serve_realtime_conversation_remove(&session->conversation,
                                            transcription->item);
  }
  transcription->item = NULL;
  transcription->phase = LOOM_SERVE_REALTIME_TRANSCRIPTION_FREE;
}

static void loom_serve_realtime_session_stop(
    loom_serve_realtime_session_t* session, bool abort) {
  session->closing = true;
  session->aborted |= abort;
  session->microphone = NULL;
  session->automatic_response = false;
  if (session->turn) {
    loom_serve_realtime_turn_clear(session->turn);
  }
  loom_serve_realtime_capture_close(session->capture);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
       ++i) {
    loom_serve_realtime_transcription_t* transcription =
        &session->transcriptions[i];
    loom_serve_realtime_transcription_close(session, transcription);
  }
  loom_serve_realtime_responses_stop(session->responses);
  loom_serve_realtime_history_invalidate(&session->history, 0);
  if (session->aborted) {
    loom_serve_realtime_wire_clear(&session->wire);
  }
}

static void loom_serve_realtime_session_fail(
    loom_serve_realtime_session_t* session, iree_status_t status) {
  if (!session->closing) {
    iree_status_t report = loom_serve_realtime_session_error(
        session, status, NULL, iree_string_view_empty(), true);
    loom_serve_realtime_session_stop(session, !iree_status_is_ok(report));
    iree_status_ignore(report);
    if (!session->aborted) {
      const uint8_t code[] = {0x03, 0xF3};  // 1011: failed server operation.
      iree_status_t close = loom_serve_realtime_wire_control(
          &session->wire, LOOM_SERVE_WEBSOCKET_CLOSE,
          iree_make_const_byte_span(code, sizeof(code)));
      if (!iree_status_is_ok(close)) {
        loom_serve_http_server_record_error(session->server, close);
        iree_status_ignore(close);
        loom_serve_realtime_session_stop(session, true);
      }
    }
  } else {
    loom_serve_http_server_record_error(session->server, status);
    iree_status_ignore(status);
  }
}

static iree_status_t loom_serve_realtime_command_session(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  if (loom_serve_realtime_responses_active(session->responses)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "session settings require an idle response boundary");
  }
  loom_serve_realtime_settings_t candidate;
  iree_status_t status = loom_serve_realtime_settings_prepare(
      &session->settings, session->profile,
      event->fields[LOOM_SERVE_REALTIME_FIELD_SESSION].text,
      session->microphone ||
          (session->turn && loom_serve_realtime_turn_is_open(session->turn)),
      &candidate);
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_session_event(session, "session.updated",
                                               &candidate);
  }
  if (iree_status_is_ok(status)) {
    if (!loom_serve_realtime_output_settings_text_equal(
            &session->settings.output, &candidate.output)) {
      loom_serve_realtime_history_invalidate(&session->history, 0);
    }
    loom_serve_realtime_settings_t previous = session->settings;
    session->settings = candidate;
    candidate = previous;
  }
  loom_serve_realtime_settings_deinitialize(&candidate);
  return status;
}

static iree_status_t loom_serve_realtime_transcription_begin(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_transcription_t* transcription) {
  const loom_serve_recognition_request_t request = {
      .language = iree_string_builder_view(&session->settings.language),
      .prompt = iree_string_builder_view(&session->settings.prompt),
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT};
  IREE_RETURN_IF_ERROR(loom_serve_realtime_input_begin(session->input, &request,
                                                       &transcription->input));
  transcription->item->input = transcription->input;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_microphone_reserve(
    loom_serve_realtime_session_t* session) {
  loom_serve_realtime_transcription_t* transcription = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
       ++i) {
    if (session->transcriptions[i].phase ==
            LOOM_SERVE_REALTIME_TRANSCRIPTION_FREE &&
        !session->transcriptions[i].span.capture) {
      transcription = &session->transcriptions[i];
      break;
    }
  }
  if (!transcription) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "realtime input item capacity is full");
  }
  loom_serve_realtime_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_reserve(
      &session->conversation, iree_string_view_empty(),
      LOOM_SERVE_REALTIME_ITEM_MESSAGE, LOOM_SERVE_REALTIME_USER, true, &item));
  *transcription = (loom_serve_realtime_transcription_t){
      .item = item,
      .phase = LOOM_SERVE_REALTIME_TRANSCRIPTION_WAITING,
      .automatic = session->settings.detect_turns};
  session->microphone = transcription;
  loom_serve_realtime_capture_mark(session->capture, &transcription->begin);
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_command_append(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  const bool created = !session->microphone;
  const bool begin_turn = session->settings.detect_turns &&
                          !loom_serve_realtime_turn_is_open(session->turn);
  iree_status_t status = iree_ok_status();
  if (created) {
    status = loom_serve_realtime_microphone_reserve(session);
  }
  if (iree_status_is_ok(status) && created && !session->settings.detect_turns) {
    status =
        loom_serve_realtime_transcription_begin(session, session->microphone);
  }
  if (iree_status_is_ok(status) && begin_turn) {
    status =
        loom_serve_realtime_turn_begin(session->turn, session->settings.turn);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_capture_append(
        session->capture, *message,
        event->fields[LOOM_SERVE_REALTIME_FIELD_AUDIO].text);
  }
  if (iree_status_is_ok(status)) {
    *message = NULL;
  } else {
    if (created && session->microphone) {
      loom_serve_realtime_transcription_close(session, session->microphone);
      session->microphone = NULL;
    }
    if (begin_turn) {
      loom_serve_realtime_turn_clear(session->turn);
    }
  }
  return status;
}

static iree_status_t loom_serve_realtime_microphone_commit(
    loom_serve_realtime_session_t* session, bool automatic, uint64_t end) {
  loom_serve_realtime_item_t* previous =
      loom_serve_realtime_conversation_last_item(&session->conversation);
  loom_serve_realtime_item_t* item = session->microphone->item;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_insert(
      &session->conversation, session->conversation.history.last, item));
  session->microphone->committed = true;
  if (automatic) {
    session->microphone->end =
        (loom_serve_realtime_capture_mark_t){.position = end, .reached = true};
  } else {
    loom_serve_realtime_capture_mark(session->capture,
                                     &session->microphone->end);
  }
  session->microphone = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      &session->wire, "input_audio_buffer.committed"));
  iree_string_builder_t* output = &session->wire.scratch;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"item_id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&item->id)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"previous_item_id\":"));
  if (previous) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
        output, iree_string_builder_view(&previous->id)));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_wire_commit(&session->wire, 0, false));
  return loom_serve_realtime_item_event(
      &session->wire, "conversation.item.added", item, previous);
}

static iree_status_t loom_serve_realtime_command_commit(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  if (!session->microphone) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot commit an empty input audio buffer");
  }
  if (!session->microphone->input) {
    uint64_t begin = 0;
    if (loom_serve_realtime_turn_prefix(session->turn, &begin)) {
      loom_serve_realtime_capture_mark_cancel(&session->microphone->begin);
      session->microphone->begin = (loom_serve_realtime_capture_mark_t){
          .position = begin, .reached = true};
    }
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_transcription_begin(session, session->microphone));
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_microphone_commit(session, false, 0));
  if (session->turn) {
    loom_serve_realtime_turn_clear(session->turn);
  }
  session->automatic_response = false;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_command_clear(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  if (session->microphone) {
    loom_serve_realtime_transcription_close(session, session->microphone);
    session->microphone = NULL;
  }
  if (session->turn) {
    loom_serve_realtime_turn_clear(session->turn);
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      &session->wire, "input_audio_buffer.cleared"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->wire.scratch, "}"));
  return loom_serve_realtime_wire_commit(&session->wire, 0, false);
}

static iree_status_t loom_serve_realtime_command_respond(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  bool default_response = false;
  iree_status_t status = loom_serve_realtime_responses_create(
      session->responses,
      event->present & (1u << LOOM_SERVE_REALTIME_FIELD_RESPONSE)
          ? event->fields[LOOM_SERVE_REALTIME_FIELD_RESPONSE].text
          : iree_string_view_empty(),
      &default_response);
  if (iree_status_is_ok(status) && default_response) {
    session->automatic_response = false;
  }
  return status;
}

static iree_status_t loom_serve_realtime_command_cancel(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  return loom_serve_realtime_responses_cancel(
      session->responses,
      event->present & (1u << LOOM_SERVE_REALTIME_FIELD_RESPONSE_ID)
          ? event->fields[LOOM_SERVE_REALTIME_FIELD_RESPONSE_ID].text
          : iree_string_view_empty());
}

static iree_status_t loom_serve_realtime_item_create(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event, iree_string_builder_t* id) {
  const bool explicit_previous =
      event->present & (1u << LOOM_SERVE_REALTIME_FIELD_PREVIOUS_ITEM_ID);
  loom_serve_realtime_item_t* previous =
      loom_serve_realtime_conversation_last_item(&session->conversation);
  if (explicit_previous) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        event->fields[LOOM_SERVE_REALTIME_FIELD_PREVIOUS_ITEM_ID].text, 128,
        id));
    if (iree_string_view_equal(iree_string_builder_view(id), IREE_SV("root"))) {
      previous = NULL;
    } else {
      previous = loom_serve_realtime_conversation_find(
          &session->conversation, iree_string_builder_view(id));
      if (!previous ||
          previous->group->owner != &session->conversation.history) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "previous item was not found");
      }
    }
  }
  const bool inserted =
      explicit_previous &&
      (previous ? previous->next ||
                      previous->group != session->conversation.history.last
                : session->conversation.history.first != NULL);
  loom_serve_realtime_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_item_parse(
      &session->conversation,
      event->fields[LOOM_SERVE_REALTIME_FIELD_ITEM].text, &item));
  iree_status_t status = iree_ok_status();
  if (loom_serve_realtime_responses_active(session->responses) &&
      (item->kind != LOOM_SERVE_REALTIME_ITEM_FUNCTION_OUTPUT || inserted)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "conversation edits require an idle response boundary");
  }
  loom_serve_realtime_wire_message_t* tail = session->wire.last;
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_item_event(
        &session->wire, "conversation.item.added", item, previous);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_item_event(
        &session->wire, "conversation.item.done", item, previous);
  }
  if (iree_status_is_ok(status)) {
    status = explicit_previous ? loom_serve_realtime_conversation_insert_after(
                                     &session->conversation, previous, item)
                               : loom_serve_realtime_conversation_insert(
                                     &session->conversation,
                                     session->conversation.history.last, item);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_wire_rewind(&session->wire, tail);
    loom_serve_realtime_conversation_remove(&session->conversation, item);
    return status;
  }
  if (inserted) {
    loom_serve_realtime_history_invalidate(&session->history, 0);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_command_create(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  iree_string_builder_t id;
  iree_string_builder_initialize(session->host_allocator, &id);
  const iree_status_t status =
      loom_serve_realtime_item_create(session, event, &id);
  iree_string_builder_deinitialize(&id);
  return status;
}

static iree_status_t loom_serve_realtime_item_mutate(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event, iree_string_builder_t* id) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
      event->fields[LOOM_SERVE_REALTIME_FIELD_ITEM_ID].text, 128, id));
  loom_serve_realtime_item_t* item = loom_serve_realtime_conversation_find(
      &session->conversation, iree_string_builder_view(id));
  if (!item) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "conversation item was not found");
  }
  const bool truncate = event->kind == LOOM_SERVE_REALTIME_ITEM_TRUNCATE;
  uint64_t end_ms = 0;
  if (truncate) {
    uint64_t content_index = 0;
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(
        event->fields[LOOM_SERVE_REALTIME_FIELD_CONTENT_INDEX].text,
        &content_index));
    IREE_RETURN_IF_ERROR(iree_json_parse_uint64(
        event->fields[LOOM_SERVE_REALTIME_FIELD_AUDIO_END_MS].text, &end_ms));
    if (content_index) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "audio content_index must be zero");
    }
    // Validate before changing any generation or cache ownership.
    if (!item->audio || item->role != LOOM_SERVE_REALTIME_ASSISTANT ||
        end_ms > item->generated_frames / 24) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncate exceeds assistant audio duration");
    }
  }
  const bool changes_history =
      item->group->owner == &session->conversation.history &&
      (!truncate || item->has_transcript);
  IREE_RETURN_IF_ERROR(loom_serve_realtime_responses_edit(
      session->responses, item, truncate, changes_history));
  if (truncate) {
    loom_serve_realtime_conversation_truncate(&session->conversation, item);
  } else {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
         ++i) {
      loom_serve_realtime_transcription_t* transcription =
          &session->transcriptions[i];
      if (transcription->item == item) {
        transcription->item = NULL;
        item->input = NULL;
        if (!loom_serve_realtime_input_has_readers(transcription->input)) {
          loom_serve_realtime_transcription_close(session, transcription);
        }
      }
    }
  }
  if (changes_history) {
    loom_serve_realtime_history_invalidate(&session->history,
                                           item->group->sequence);
    loom_serve_realtime_wire_cancel(&session->wire, item->group->sequence);
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      &session->wire,
      truncate ? "conversation.item.truncated" : "conversation.item.deleted"));
  iree_string_builder_t* output = &session->wire.scratch;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"item_id\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_string_write(output, iree_string_builder_view(id)));
  if (truncate) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"content_index\":0,\"audio_end_ms\":%" PRIu64, end_ms));
  } else {
    loom_serve_realtime_conversation_remove(&session->conversation, item);
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  return loom_serve_realtime_wire_commit(&session->wire, 0, false);
}

static iree_status_t loom_serve_realtime_command_mutate(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  iree_string_builder_t id;
  iree_string_builder_initialize(session->host_allocator, &id);
  const iree_status_t status =
      loom_serve_realtime_item_mutate(session, event, &id);
  iree_string_builder_deinitialize(&id);
  return status;
}

typedef iree_status_t (*loom_serve_realtime_command_fn_t)(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message);

typedef struct loom_serve_realtime_command_t {
  // Applies one validated command on the serialized session owner.
  loom_serve_realtime_command_fn_t execute;
  // Failure unwinds every mutation and queued acknowledgement, allowing an
  // admission error to preserve already accepted work even at capacity.
  bool transactional;
} loom_serve_realtime_command_t;

static const loom_serve_realtime_command_t loom_serve_realtime_commands[] = {
    {loom_serve_realtime_command_session, true},
    {loom_serve_realtime_command_append, false},
    {loom_serve_realtime_command_commit, false},
    {loom_serve_realtime_command_clear, false},
    {loom_serve_realtime_command_respond, true},
    {loom_serve_realtime_command_cancel, false},
    {loom_serve_realtime_command_create, true},
    {loom_serve_realtime_command_mutate, false},
    {loom_serve_realtime_command_mutate, false},
};

static iree_status_t loom_serve_realtime_session_receive(
    loom_serve_realtime_session_t* session, bool* out_progress) {
  iree_string_builder_t event_id;
  iree_string_builder_initialize(session->host_allocator, &event_id);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < 8 && !session->closing && iree_status_is_ok(status); ++i) {
    loom_serve_websocket_message_type_t type;
    iree_const_byte_span_t payload;
    loom_serve_websocket_message_t* message =
        loom_serve_websocket_take(session->socket, &type, &payload);
    if (!message) {
      break;
    }
    *out_progress = true;
    iree_string_builder_reset(&event_id);
    if (type == LOOM_SERVE_WEBSOCKET_TEXT) {
      loom_serve_realtime_event_t event;
      iree_status_t command = loom_serve_realtime_event_parse(
          iree_make_string_view((const char*)payload.data, payload.data_length),
          &event);
      if (iree_status_is_ok(command) &&
          (event.present & (1u << LOOM_SERVE_REALTIME_FIELD_EVENT_ID))) {
        command = loom_serve_realtime_string_decode(
            event.fields[LOOM_SERVE_REALTIME_FIELD_EVENT_ID].text, 128,
            &event_id);
      }
      bool transactional = false;
      const char* error_code = NULL;
      if (iree_status_is_ok(command)) {
        const loom_serve_realtime_command_t* operation =
            &loom_serve_realtime_commands[event.kind];
        transactional = operation->transactional;
        command = operation->execute(session, &event, &message);
        // Completion or automatic interruption may win an explicit cancel.
        // Clients distinguish this harmless race from other command failures.
        if (event.kind == LOOM_SERVE_REALTIME_RESPONSE_CANCEL &&
            iree_status_is_not_found(command)) {
          error_code = "response_cancel_not_active";
        }
      }
      if (!iree_status_is_ok(command)) {
        const bool terminal =
            !transactional && iree_status_is_resource_exhausted(command);
        status = loom_serve_realtime_session_error(
            session, command, error_code, iree_string_builder_view(&event_id),
            false);
        if (terminal) {
          loom_serve_realtime_session_stop(session, true);
        }
      }
    } else if (type == LOOM_SERVE_WEBSOCKET_PING) {
      status = loom_serve_realtime_wire_control(
          &session->wire, LOOM_SERVE_WEBSOCKET_PONG, payload);
    } else if (type == LOOM_SERVE_WEBSOCKET_CLOSE) {
      loom_serve_realtime_session_stop(session, false);
      loom_serve_realtime_wire_clear(&session->wire);
      status = loom_serve_realtime_wire_control(
          &session->wire, LOOM_SERVE_WEBSOCKET_CLOSE, payload);
    } else if (type != LOOM_SERVE_WEBSOCKET_PONG) {
      status = loom_serve_realtime_session_error(
          session,
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "realtime requires JSON text messages"),
          NULL, iree_string_view_empty(), false);
    }
    if (message) {
      loom_serve_websocket_message_release(session->socket, message);
    }
  }
  iree_string_builder_deinitialize(&event_id);
  return status;
}

static iree_status_t loom_serve_realtime_turn_event(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_turn_event_t event) {
  const bool started = event.kind == LOOM_SERVE_REALTIME_TURN_STARTED;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      &session->wire, started ? "input_audio_buffer.speech_started"
                              : "input_audio_buffer.speech_stopped"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &session->wire.scratch, ",\"item_id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      &session->wire.scratch,
      iree_string_builder_view(&session->microphone->item->id)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &session->wire.scratch, ",\"audio_%s_ms\":%" PRIu64 "}",
      started ? "start" : "end", event.position / 24));
  return loom_serve_realtime_wire_commit(&session->wire, 0, false);
}

static iree_status_t loom_serve_realtime_turns_advance(
    loom_serve_realtime_session_t* session, bool* out_progress) {
  if (!session->turn) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_turn_advance(
      session->turn, session->closing, out_progress));
  const loom_serve_realtime_turn_event_t event =
      loom_serve_realtime_turn_peek(session->turn);
  if (!event.kind) {
    return iree_ok_status();
  }
  if (event.kind == LOOM_SERVE_REALTIME_TURN_STARTED) {
    if (!session->microphone) {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_microphone_reserve(session));
    }
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_transcription_begin(session, session->microphone));
    loom_serve_realtime_capture_mark_cancel(&session->microphone->begin);
    session->microphone->begin = (loom_serve_realtime_capture_mark_t){
        .position = event.position, .reached = true};
    session->automatic_response = false;
    if (session->settings.interrupt_response) {
      loom_serve_realtime_responses_interrupt(session->responses);
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_turn_event(session, event));
  if (event.kind == LOOM_SERVE_REALTIME_TURN_STOPPED) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_microphone_commit(session, true, event.position));
    session->automatic_response = session->settings.create_response;
  }
  loom_serve_realtime_turn_consume(session->turn);
  *out_progress = true;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_transcriptions_advance(
    loom_serve_realtime_session_t* session, bool* out_progress) {
  loom_serve_realtime_input_advance(session->input, session->closing,
                                    out_progress);
  loom_serve_realtime_capture_advance(session->capture, out_progress);
  iree_status_t status =
      session->closing ? iree_ok_status()
                       : loom_serve_realtime_capture_status(session->capture);
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_turns_advance(session, out_progress);
  }
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(session->transcriptions) && iree_status_is_ok(status);
       ++i) {
    loom_serve_realtime_transcription_t* transcription =
        &session->transcriptions[i];
    if (!transcription->input) {
      continue;
    }
    if (!transcription->item &&
        !loom_serve_realtime_input_has_readers(transcription->input)) {
      loom_serve_realtime_transcription_close(session, transcription);
      *out_progress = true;
      continue;
    }
    status = loom_serve_realtime_input_status(transcription->input);
    if (!iree_status_is_ok(status)) {
      continue;
    }
    if (transcription->phase == LOOM_SERVE_REALTIME_TRANSCRIPTION_WAITING) {
      if (!transcription->begin.reached) {
        continue;
      }
      transcription->position = transcription->begin.position;
      transcription->phase = LOOM_SERVE_REALTIME_TRANSCRIPTION_READING;
    }
    uint64_t end = loom_serve_realtime_capture_end(session->capture);
    if (transcription->automatic && !transcription->committed) {
      end = iree_min(end, loom_serve_realtime_turn_covered(session->turn));
    }
    if (transcription->committed && transcription->end.reached) {
      end = iree_min(end, transcription->end.position);
    }
    if (!transcription->span.capture && transcription->position < end) {
      loom_serve_realtime_capture_borrow(
          session->capture, transcription->position,
          (iree_host_size_t)iree_min(4096, end - transcription->position),
          &transcription->span);
      const iree_host_size_t count =
          transcription->span.samples.data_length / sizeof(int16_t);
      status = loom_serve_realtime_input_append(
          session->input, transcription->input, transcription->span.samples,
          loom_serve_realtime_capture_release, &transcription->span);
      if (iree_status_is_ok(status)) {
        transcription->position += count;
        *out_progress = true;
      } else {
        loom_serve_realtime_capture_release(&transcription->span);
      }
    }
    if (!iree_status_is_ok(status)) {
      continue;
    }
    if (transcription->phase == LOOM_SERVE_REALTIME_TRANSCRIPTION_READING &&
        transcription->committed && transcription->end.reached &&
        transcription->position == transcription->end.position) {
      loom_serve_realtime_input_finish(transcription->input);
      transcription->phase = LOOM_SERVE_REALTIME_TRANSCRIPTION_FINISHED;
      *out_progress = true;
    }
    const iree_string_builder_t* text =
        loom_serve_realtime_input_transcript(transcription->input);
    if (!text || !transcription->committed) {
      continue;
    }
    if (!transcription->item) {
      loom_serve_realtime_transcription_close(session, transcription);
      *out_progress = true;
      continue;
    }
    loom_serve_realtime_item_t* item = transcription->item;
    status = loom_serve_realtime_conversation_append_text(
        &session->conversation, item, iree_string_builder_view(text));
    if (iree_status_is_ok(status)) {
      item->status = LOOM_SERVE_REALTIME_ITEM_COMPLETED;
      status = loom_serve_realtime_wire_begin(
          &session->wire,
          "conversation.item.input_audio_transcription.completed");
    }
    iree_string_builder_t* output = &session->wire.scratch;
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(output, ",\"item_id\":");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_string_write(
          output, iree_string_builder_view(&item->id));
    }
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_cstring(
          output, ",\"content_index\":0,\"transcript\":");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_string_write(
          output, iree_string_builder_view(&item->text));
    }
    if (iree_status_is_ok(status)) {
      const double seconds =
          (double)loom_serve_realtime_input_frames(transcription->input) /
          24000.0;
      status = iree_string_builder_append_format(
          output, ",\"usage\":{\"type\":\"duration\",\"seconds\":%.9g}}",
          seconds);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_wire_commit(&session->wire, 0, false);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_item_event(
          &session->wire, "conversation.item.done", item, NULL);
    }
    if (iree_status_is_ok(status)) {
      loom_serve_realtime_transcription_close(session, transcription);
      *out_progress = true;
    }
  }
  if (iree_status_is_ok(status)) {
    uint64_t retain = loom_serve_realtime_capture_end(session->capture);
    if (session->turn) {
      retain = iree_min(retain, loom_serve_realtime_turn_retain(session->turn));
    }
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
         ++i) {
      const loom_serve_realtime_transcription_t* transcription =
          &session->transcriptions[i];
      if (transcription->input) {
        if (transcription->phase == LOOM_SERVE_REALTIME_TRANSCRIPTION_READING) {
          retain = iree_min(retain, transcription->position);
        } else if (transcription->phase ==
                       LOOM_SERVE_REALTIME_TRANSCRIPTION_WAITING &&
                   transcription->begin.reached) {
          retain = iree_min(retain, transcription->begin.position);
        }
      }
    }
    loom_serve_realtime_capture_discard_before(session->capture, retain);
  }
  return status;
}

static iree_status_t loom_serve_realtime_automatic_response_advance(
    loom_serve_realtime_session_t* session, bool* out_progress) {
  if (!session->automatic_response || session->closing ||
      loom_serve_realtime_turn_is_speaking(session->turn) ||
      loom_serve_realtime_responses_active(session->responses)) {
    return iree_ok_status();
  }
  bool waiting = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
       ++i) {
    const loom_serve_realtime_item_t* item = session->transcriptions[i].item;
    waiting |= item && item->group;
  }
  if (waiting || !loom_serve_realtime_responses_available(session->responses)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_responses_create(
      session->responses, iree_string_view_empty(), NULL));
  session->automatic_response = false;
  *out_progress = true;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_session_allocate(
    uint64_t id, const loom_serve_realtime_profile_t* profile,
    loom_serve_realtime_binding_t text,
    loom_serve_realtime_binding_t recognition,
    loom_serve_realtime_binding_t speech,
    loom_serve_realtime_binding_t activity, iree_task_executor_t* executor,
    loom_serve_registry_t* registry, loom_serve_http_server_t* server,
    loom_serve_http_connection_t* connection, loom_serve_websocket_t* socket,
    loom_serve_wake_callback_t observer, iree_allocator_t host_allocator,
    loom_serve_realtime_session_t** out_session) {
  *out_session = NULL;
  loom_serve_realtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*session),
                                             (void**)&session));
  session->host_allocator = host_allocator;
  session->server = server;
  session->connection = connection;
  session->socket = socket;
  session->profile = profile;
  session->id = id;
  loom_serve_realtime_conversation_initialize(id, host_allocator,
                                              &session->conversation);
  loom_serve_realtime_history_initialize(registry, text, host_allocator,
                                         &session->history);
  loom_serve_realtime_wire_initialize(id, socket, connection, host_allocator,
                                      &session->wire);
  iree_status_t status = loom_serve_realtime_settings_initialize(
      profile, host_allocator, &session->settings);
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_responses_allocate(
        id, profile, text, speech, executor, registry, server, observer,
        &session->settings, &session->conversation, &session->history,
        &session->wire, host_allocator, &session->responses);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_input_allocate(executor, registry, recognition,
                                                observer, host_allocator,
                                                &session->input);
  }
  const iree_host_size_t capacity =
      profile->input_capacity ? profile->input_capacity
                              : LOOM_SERVE_REALTIME_INPUT_CAPACITY_DEFAULT;
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_capture_allocate(executor, socket, capacity,
                                                  observer, host_allocator,
                                                  &session->capture);
  }
  if (iree_status_is_ok(status) && profile->activity.size) {
    status = loom_serve_realtime_turn_allocate(
        executor, registry, activity, session->capture, capacity, observer,
        host_allocator, &session->turn);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_session_event(session, "session.created",
                                               &session->settings);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_session_free(session);
    return status;
  }
  *out_session = session;
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_session_advance(
    loom_serve_realtime_session_t* session, bool stopping, bool* out_progress,
    bool* out_complete) {
  *out_complete = false;
  if (stopping || loom_serve_http_connection_failed(session->connection)) {
    loom_serve_realtime_session_stop(session, true);
  }
  iree_status_t status = iree_ok_status();
  if (!session->closing) {
    status = loom_serve_realtime_session_receive(session, out_progress);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_http_server_record_error(session->server, status);
    iree_status_ignore(status);
    loom_serve_realtime_session_stop(session, true);
  }
  status = loom_serve_realtime_transcriptions_advance(session, out_progress);
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_responses_advance(
        session->responses, session->closing, out_progress);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_realtime_automatic_response_advance(session, out_progress);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_session_fail(session, status);
  }
  bool cleanup_failed = false;
  status = loom_serve_realtime_history_advance(
      &session->history, session->closing, out_progress, &cleanup_failed);
  if (!iree_status_is_ok(status)) {
    // A failed RELEASE still owns a possible native pin. Only that failure
    // needs outer model teardown; rejected captures have no effect to retire.
    if (cleanup_failed) {
      return status;
    }
    loom_serve_realtime_session_fail(session, status);
  }
  if (!session->aborted) {
    status = loom_serve_realtime_wire_flush(&session->wire, out_progress);
    if (!iree_status_is_ok(status)) {
      loom_serve_http_server_record_error(session->server, status);
      iree_status_ignore(status);
      loom_serve_realtime_session_stop(session, true);
    }
  }
  if (session->closing &&
      !loom_serve_realtime_responses_pending(session->responses) &&
      !loom_serve_realtime_capture_pending(session->capture) &&
      (!session->turn || !loom_serve_realtime_turn_pending(session->turn)) &&
      !loom_serve_realtime_input_pending(session->input) &&
      !loom_serve_realtime_history_pending(&session->history) &&
      !session->wire.first) {
    if (session->aborted) {
      loom_serve_http_connection_abort(session->connection);
    } else {
      loom_serve_http_connection_finish(session->connection);
    }
    session->connection = NULL;
    *out_complete = true;
    *out_progress = true;
  }
  return iree_ok_status();
}

void loom_serve_realtime_session_free(loom_serve_realtime_session_t* session) {
  if (!session) {
    return;
  }
  loom_serve_realtime_wire_deinitialize(&session->wire);
  loom_serve_realtime_history_deinitialize(&session->history);
  loom_serve_realtime_input_free(session->input);
  loom_serve_realtime_turn_free(session->turn);
  loom_serve_realtime_capture_free(session->capture);
  loom_serve_realtime_responses_free(session->responses);
  loom_serve_realtime_conversation_deinitialize(&session->conversation);
  loom_serve_realtime_settings_deinitialize(&session->settings);
  iree_allocator_free(session->host_allocator, session);
}
