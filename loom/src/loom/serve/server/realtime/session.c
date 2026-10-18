// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/session.h"

#include <inttypes.h>

#include "loom/serve/server/realtime/capture.h"
#include "loom/serve/server/realtime/conversation.h"
#include "loom/serve/server/realtime/generation.h"
#include "loom/serve/server/realtime/history.h"
#include "loom/serve/server/realtime/input.h"
#include "loom/serve/server/realtime/wire.h"
#include "loom/serve/service/text/tool_calls.h"

typedef enum loom_serve_realtime_transcription_phase_e {
  LOOM_SERVE_REALTIME_TRANSCRIPTION_FREE,
  LOOM_SERVE_REALTIME_TRANSCRIPTION_WAITING,
  LOOM_SERVE_REALTIME_TRANSCRIPTION_READING,
  LOOM_SERVE_REALTIME_TRANSCRIPTION_FINISHED,
} loom_serve_realtime_transcription_phase_t;

typedef struct loom_serve_realtime_transcription_t {
  // Reserved wire identity, available before the item joins the conversation.
  uint64_t identity;
  // Independent input lifetime, including accepted conversion after clear.
  loom_serve_realtime_input_item_t* input;
  // Start of this utterance after all earlier encoded appends publish.
  loom_serve_realtime_capture_mark_t begin;
  // Explicit commit frontier, valid after a conversation item is assigned.
  loom_serve_realtime_capture_mark_t end;
  // One captured PCM loan, stable until conversion releases it on this owner.
  loom_serve_realtime_capture_span_t span;
  // Original-rate endpoint already offered to the independent converter.
  uint64_t position;
  // Ordered start readiness, input delivery, and explicit native EOF.
  loom_serve_realtime_transcription_phase_t phase;
  // Uncommitted delivery follows classified activity rather than capture end.
  bool automatic;
  // Committed conversation destination; NULL while the microphone item is open.
  loom_serve_realtime_item_t* item;
} loom_serve_realtime_transcription_t;

typedef enum loom_serve_realtime_response_finish_e {
  LOOM_SERVE_REALTIME_FINISH_GENERATING,
  LOOM_SERVE_REALTIME_FINISH_TEXT,
  LOOM_SERVE_REALTIME_FINISH_AUDIO,
  LOOM_SERVE_REALTIME_FINISH_CONTENT,
  LOOM_SERVE_REALTIME_FINISH_OUTPUT_ITEM,
  LOOM_SERVE_REALTIME_FINISH_CONVERSATION_ITEM,
  LOOM_SERVE_REALTIME_FINISH_RESPONSE,
  LOOM_SERVE_REALTIME_FINISH_RETIRED,
} loom_serve_realtime_response_finish_t;

typedef enum loom_serve_realtime_response_stop_e {
  LOOM_SERVE_REALTIME_STOP_NONE,
  LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED,
  LOOM_SERVE_REALTIME_STOP_TURN_DETECTED,
  LOOM_SERVE_REALTIME_STOP_FAILED,
} loom_serve_realtime_response_stop_t;

typedef struct loom_serve_realtime_response_t {
  // Canonical assistant item; NULL denotes an available response slot.
  loom_serve_realtime_item_t* item;
  // Owned stable native session and wire response ID.
  iree_string_builder_t id;
  // Copied selected checkpoint, retained through deferred native admission.
  iree_string_builder_t checkpoint;
  // Immutable response settings, captured when response.create is accepted.
  loom_serve_realtime_response_settings_t settings;
  // Native composition, including a separately retiring output CPU borrower.
  loom_serve_realtime_generation_t* generation;
  // First terminal cause withdrawing generation, independent of retirement.
  loom_serve_realtime_response_stop_t stop;
  // Owned native failure when stop is FAILED, retained through final delivery.
  iree_status_t failure;
  // Completed text prefix capture has transferred to the history owner.
  bool captured;
  // Ordered final events, independent of native work and capture retirement.
  loom_serve_realtime_response_finish_t finish;
  // Native text ended at its token budget rather than the model's EOS.
  bool budget_exhausted;
  // Copied input token accounting after successful native text completion.
  iree_host_size_t input_tokens;
  // Copied compatible-prefix reuse, independent of output playback accounting.
  iree_host_size_t cached_tokens;
  // Copied generated output-token count, including source terminal tokens.
  iree_host_size_t output_tokens;
} loom_serve_realtime_response_t;

struct loom_serve_realtime_session_t {
  // Allocator owning this session and each bounded child representation.
  iree_allocator_t host_allocator;
  // Shared application workers, borrowed through all accepted CPU releases.
  iree_task_executor_t* executor;
  // Serialized registration owner, never borrowed by conversion workers.
  loom_serve_registry_t* registry;
  // Shared transport diagnostics and connection lifetime owner.
  loom_serve_http_server_t* server;
  // Retained HTTP claim; released only after receipt borrowers retire.
  loom_serve_http_connection_t* connection;
  // Connection-owned framed protocol, retained by the HTTP claim.
  loom_serve_websocket_t* socket;
  // Immutable application-owned assistant composition.
  const loom_serve_realtime_profile_t* profile;
  // Nonblocking shared-owner wake target, stable until final CPU retirement.
  loom_serve_wake_callback_t observer;
  // Unique server session identity, independent of native model request IDs.
  uint64_t id;
  // Reserved server item identities, independent of conversation insertion.
  uint64_t item_identity;
  // Bound text registration; never rebound to a replacement namesake.
  loom_serve_realtime_binding_t text;
  // Bound speech registration, independent of the selected voice.
  loom_serve_realtime_binding_t speech;
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
  // One logical response plus one cancelled response whose worker is retiring.
  loom_serve_realtime_response_t responses[2];
  // Reusable speech conversion coefficients; only one worker borrows at a time.
  loom_serve_realtime_audio_t audio;
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
    loom_serve_realtime_session_t* session, const char* type) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(&session->wire, type));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      &session->wire.scratch, ",\"session\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_settings_write(&session->settings, session->profile,
                                         session->id, &session->wire.scratch));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->wire.scratch, "}"));
  return loom_serve_realtime_wire_commit(&session->wire, 0, false);
}

static iree_status_t loom_serve_realtime_session_error(
    loom_serve_realtime_session_t* session, iree_status_t failure,
    iree_string_view_t event_id, bool server_failure) {
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
        iree_status_code_string(iree_status_code(failure)));
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

static iree_status_t loom_serve_realtime_item_event(
    loom_serve_realtime_session_t* session, const char* type,
    const loom_serve_realtime_item_t* item,
    const loom_serve_realtime_item_t* previous) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(&session->wire, type));
  iree_string_builder_t* output = &session->wire.scratch;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"previous_item_id\":"));
  if (previous) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
        output, iree_string_builder_view(&previous->id)));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"item\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_item_write(item, output));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  return loom_serve_realtime_wire_commit(&session->wire, 0, false);
}

static iree_status_t loom_serve_realtime_insert_item(
    loom_serve_realtime_session_t* session, uint64_t identity,
    loom_serve_realtime_role_t role, bool audio,
    loom_serve_realtime_item_t** out_item) {
  char id[96];
  const int length = iree_snprintf(id, sizeof(id), "item_%" PRIu64 "_%" PRIu64,
                                   session->id, identity);
  return loom_serve_realtime_conversation_insert(
      &session->conversation, session->conversation.last,
      iree_make_string_view(id, (iree_host_size_t)length), role, audio,
      out_item);
}

static iree_status_t loom_serve_realtime_new_item(
    loom_serve_realtime_session_t* session, loom_serve_realtime_role_t role,
    bool audio, loom_serve_realtime_item_t** out_item) {
  return loom_serve_realtime_insert_item(session, ++session->item_identity,
                                         role, audio, out_item);
}

static loom_serve_realtime_response_t* loom_serve_realtime_active_response(
    loom_serve_realtime_session_t* session) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    loom_serve_realtime_response_t* response = &session->responses[i];
    if (response->item && !response->stop &&
        response->finish != LOOM_SERVE_REALTIME_FINISH_RETIRED) {
      return response;
    }
  }
  return NULL;
}

static void loom_serve_realtime_response_stop(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_response_t* response,
    loom_serve_realtime_response_stop_t stop) {
  if (!response->stop) {
    response->stop = stop;
  }
  response->captured = true;
  response->finish =
      response->finish ? response->finish : LOOM_SERVE_REALTIME_FINISH_TEXT;
  response->item->status = LOOM_SERVE_REALTIME_ITEM_INCOMPLETE;
  loom_serve_realtime_wire_cancel(&session->wire, response->item->sequence);
  if (response->generation) {
    loom_serve_realtime_generation_cancel(response->generation);
  }
  // A capture already accepted for this response must still be observed and
  // released. Its predecessor remains valid until a conversation edit says so.
  loom_serve_realtime_history_invalidate(&session->history,
                                         response->item->sequence);
}

static void loom_serve_realtime_response_fail(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_response_t* response, iree_status_t failure) {
  loom_serve_http_server_record_error(session->server, failure);
  if (response->stop) {
    // A late conversion failure is still diagnosed, but cannot replace the
    // terminal cause already selected by cancellation or an earlier failure.
    iree_status_ignore(failure);
    return;
  }
  loom_serve_realtime_response_stop(session, response,
                                    LOOM_SERVE_REALTIME_STOP_FAILED);
  response->failure = failure;
  if (response->item->audio) {
    // Failed generation gives no alignment between partial text and playback.
    loom_serve_realtime_conversation_truncate(&session->conversation,
                                              response->item);
  }
}

static void loom_serve_realtime_transcription_close(
    loom_serve_realtime_transcription_t* transcription) {
  if (transcription->input) {
    loom_serve_realtime_input_close(transcription->input);
    transcription->input = NULL;
  }
  loom_serve_realtime_capture_mark_cancel(&transcription->begin);
  loom_serve_realtime_capture_mark_cancel(&transcription->end);
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
    loom_serve_realtime_transcription_close(transcription);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    if (session->responses[i].item) {
      loom_serve_realtime_response_stop(
          session, &session->responses[i],
          LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED);
    }
  }
  loom_serve_realtime_history_invalidate(&session->history, 0);
  if (session->aborted) {
    loom_serve_realtime_wire_clear(&session->wire);
  }
}

static void loom_serve_realtime_session_fail(
    loom_serve_realtime_session_t* session, iree_status_t status) {
  if (!session->closing) {
    iree_status_t report = loom_serve_realtime_session_error(
        session, status, iree_string_view_empty(), true);
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

static iree_status_t loom_serve_realtime_response_fields(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_response_t* response, const char* type) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(&session->wire, type));
  iree_string_builder_t* output = &session->wire.scratch;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"response_id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&response->id)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"item_id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&response->item->id)));
  return iree_string_builder_append_cstring(
      output, ",\"output_index\":0,\"content_index\":0");
}

static iree_status_t loom_serve_realtime_response_event(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_response_t* response, bool done) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      &session->wire, done ? "response.done" : "response.created"));
  iree_string_builder_t* output = &session->wire.scratch;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"response\":{\"id\":"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, iree_string_builder_view(&response->id)));
  static const char* const stopped_status[] = {NULL, "cancelled", "cancelled",
                                               "failed"};
  const char* status = !done            ? "in_progress"
                       : response->stop ? stopped_status[response->stop]
                       : response->budget_exhausted ? "incomplete"
                                                    : "completed";
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      ",\"object\":\"realtime.response\",\"status\":\"%s\","
      "\"status_details\":",
      status));
  if (done && response->stop == LOOM_SERVE_REALTIME_STOP_FAILED) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        "{\"type\":\"failed\",\"error\":{\"type\":\"server_error\","
        "\"code\":\"%s\",\"message\":",
        iree_status_code_string(iree_status_code(response->failure))));
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
        output, iree_status_message(response->failure)));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}}"));
  } else if (done && (response->stop || response->budget_exhausted)) {
    const char* reason =
        response->stop == LOOM_SERVE_REALTIME_STOP_TURN_DETECTED
            ? "turn_detected"
        : response->stop ? "client_cancelled"
                         : "max_output_tokens";
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, "{\"type\":\"%s\",\"reason\":\"%s\"}", status, reason));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"output\":["));
  if (done) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_item_write(response->item, output));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, "],\"output_modalities\":[\"%s\"],\"metadata\":",
      response->settings.output.audio ? "audio" : "text"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      output, response->settings.metadata.size
                  ? iree_string_builder_view(&response->settings.metadata)
                  : IREE_SV("null")));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(output, ",\"usage\":"));
  if (done && !response->stop) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output,
        "{\"total_tokens\":%" PRIhsz ",\"input_tokens\":%" PRIhsz
        ",\"output_tokens\":%" PRIhsz
        ",\"input_token_details\":{"
        "\"cached_tokens\":%" PRIhsz "}}",
        response->input_tokens + response->output_tokens,
        response->input_tokens, response->output_tokens,
        response->cached_tokens));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}}"));
  return loom_serve_realtime_wire_commit(&session->wire,
                                         response->item->sequence, false);
}

static iree_status_t loom_serve_realtime_response_item_event(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_response_t* response, bool done) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
      session, response,
      done ? "response.output_item.done" : "response.output_item.added"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->wire.scratch, ",\"item\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_item_write(response->item, &session->wire.scratch));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&session->wire.scratch, "}"));
  return loom_serve_realtime_wire_commit(&session->wire,
                                         response->item->sequence, false);
}

static iree_status_t loom_serve_realtime_response_part_event(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_response_t* response, bool done) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
      session, response,
      done ? "response.content_part.done" : "response.content_part.added"));
  iree_string_builder_t* output = &session->wire.scratch;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"part\":{\"type\":\"%s\",\"%s\":",
      response->settings.output.audio ? "audio" : "text",
      response->settings.output.audio ? "transcript" : "text"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, done ? iree_string_builder_view(&response->item->text)
                   : iree_string_view_empty()));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}}"));
  return loom_serve_realtime_wire_commit(&session->wire,
                                         response->item->sequence, false);
}

static iree_status_t loom_serve_realtime_response_finish(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_response_t* response, bool* out_progress) {
  if (!loom_serve_realtime_wire_can_publish(&session->wire)) {
    return iree_ok_status();
  }
  iree_string_builder_t* output = &session->wire.scratch;
  switch (response->finish) {
    case LOOM_SERVE_REALTIME_FINISH_GENERATING:
    case LOOM_SERVE_REALTIME_FINISH_RETIRED:
      return iree_ok_status();
    case LOOM_SERVE_REALTIME_FINISH_TEXT: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
          session, response,
          response->settings.output.audio
              ? "response.output_audio_transcript.done"
              : "response.output_text.done"));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          output,
          ",\"%s\":", response->settings.output.audio ? "transcript" : "text"));
      IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
          output, iree_string_builder_view(&response->item->text)));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
      IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
          &session->wire, response->item->sequence, false));
      if (!response->settings.output.audio) {
        ++response->finish;
      }
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_AUDIO: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
          session, response, "response.output_audio.done"));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
      IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
          &session->wire, response->item->sequence, false));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_CONTENT: {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_response_part_event(session, response, true));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_OUTPUT_ITEM: {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_response_item_event(session, response, true));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_CONVERSATION_ITEM: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_item_event(
          session, "conversation.item.done", response->item, NULL));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_RESPONSE: {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_response_event(session, response, true));
      break;
    }
  }
  ++response->finish;
  *out_progress = true;
  return iree_ok_status();
}

static bool loom_serve_realtime_response_ready(
    const loom_serve_realtime_session_t* session,
    const loom_serve_realtime_response_t* response) {
  if (loom_serve_realtime_history_capturing(&session->history)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    const loom_serve_realtime_response_t* other = &session->responses[i];
    if (other != response && other->generation &&
        (loom_serve_realtime_generation_pending(other->generation) ||
         (!other->stop && !other->captured))) {
      return false;
    }
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
       ++i) {
    if (session->transcriptions[i].item &&
        session->transcriptions[i].item->sequence < response->item->sequence) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_serve_realtime_response_start(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_response_t* response) {
  loom_serve_text_request_t request = {0};
  iree_string_builder_initialize(session->host_allocator, &request.body);
  iree_status_t status = loom_serve_realtime_conversation_request(
      &session->conversation, response->item, session->text.name,
      iree_string_builder_view(&response->settings.output.instructions),
      response->settings.output.max_output_tokens, session->profile->seed,
      &request.body);
  if (iree_status_is_ok(status) &&
      iree_string_view_equal(
          iree_string_builder_view(&response->settings.output.instructions),
          iree_string_builder_view(&session->settings.output.instructions))) {
    status = iree_string_builder_append_string(
        &response->checkpoint,
        loom_serve_realtime_history_checkpoint(&session->history));
  }
  if (iree_status_is_ok(status)) {
    request.session = iree_string_builder_view(&response->id);
    request.checkpoint = iree_string_builder_view(&response->checkpoint);
    status = loom_serve_realtime_generation_allocate(
        session->executor, session->registry, session->text, session->speech,
        &request,
        response->settings.output.audio
            ? iree_string_builder_view(&response->settings.output.voice)
            : iree_string_view_empty(),
        session->profile->seed, &session->audio, session->observer,
        session->host_allocator, &response->generation);
  }
  iree_string_builder_deinitialize(&request.body);
  return status;
}

static iree_status_t loom_serve_realtime_response_advance(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_response_t* response, bool* out_progress) {
  if (!response->item) {
    return iree_ok_status();
  }
  if (!response->generation && !response->stop &&
      loom_serve_realtime_response_ready(session, response)) {
    iree_status_t status =
        loom_serve_realtime_response_start(session, response);
    if (!iree_status_is_ok(status)) {
      loom_serve_realtime_response_fail(session, response, status);
    }
    *out_progress = true;
  }
  if (response->generation) {
    iree_status_t status = loom_serve_realtime_generation_advance(
        response->generation,
        !session->closing &&
            loom_serve_realtime_wire_can_publish(&session->wire),
        out_progress);
    if (!iree_status_is_ok(status)) {
      loom_serve_realtime_response_fail(session, response, status);
      *out_progress = true;
    }
    response->item->generated_frames =
        loom_serve_realtime_generation_frames(response->generation);
    if (!response->stop) {
      const iree_string_view_t audio =
          loom_serve_realtime_generation_audio(response->generation);
      if (audio.size && loom_serve_realtime_wire_can_publish(&session->wire)) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
            session, response, "response.output_audio.delta"));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            &session->wire.scratch, ",\"delta\":\""));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(&session->wire.scratch, audio));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(&session->wire.scratch, "\"}"));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
            &session->wire, response->item->sequence, true));
        loom_serve_realtime_generation_consume_audio(response->generation);
        *out_progress = true;
      }
      const iree_string_view_t text =
          loom_serve_realtime_generation_text(response->generation);
      if (text.size && loom_serve_realtime_wire_can_publish(&session->wire)) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_append_text(
            &session->conversation, response->item, text));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
            session, response,
            response->settings.output.audio
                ? "response.output_audio_transcript.delta"
                : "response.output_text.delta"));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            &session->wire.scratch, ",\"delta\":"));
        IREE_RETURN_IF_ERROR(
            loom_serve_realtime_string_write(&session->wire.scratch, text));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(&session->wire.scratch, "}"));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
            &session->wire, response->item->sequence, true));
        loom_serve_realtime_generation_consume_text(response->generation);
        *out_progress = true;
      }
      const loom_serve_text_stream_result_t* result =
          loom_serve_realtime_generation_result(response->generation);
      if (result && !response->captured) {
        if (result->calls->count) {
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "unexpected tool output in a tool-free session");
        }
        response->input_tokens = result->response.usage.input_tokens;
        response->cached_tokens = result->response.usage.cached_tokens;
        response->output_tokens = result->response.usage.output_tokens;
        response->budget_exhausted =
            iree_string_view_equal(result->finish_reason, IREE_SV("length"));
        if (iree_string_view_equal(
                iree_string_builder_view(
                    &response->settings.output.instructions),
                iree_string_builder_view(
                    &session->settings.output.instructions))) {
          IREE_RETURN_IF_ERROR(loom_serve_realtime_history_capture(
              &session->history, iree_string_builder_view(&response->id),
              response->item->sequence, &response->captured));
        } else {
          response->captured = true;
        }
        *out_progress |= response->captured;
      }
      if (!response->finish &&
          loom_serve_realtime_generation_complete(response->generation)) {
        response->item->status = response->budget_exhausted
                                     ? LOOM_SERVE_REALTIME_ITEM_INCOMPLETE
                                     : LOOM_SERVE_REALTIME_ITEM_COMPLETED;
        response->finish = LOOM_SERVE_REALTIME_FINISH_TEXT;
        *out_progress = true;
      }
    }
  }
  if (!session->closing) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_response_finish(session, response, out_progress));
  }
  if ((session->closing ||
       response->finish == LOOM_SERVE_REALTIME_FINISH_RETIRED) &&
      response->captured &&
      (!response->generation ||
       !loom_serve_realtime_generation_pending(response->generation))) {
    if (response->generation) {
      loom_serve_realtime_generation_cancel(response->generation);
      loom_serve_realtime_generation_free(response->generation);
    }
    iree_string_builder_deinitialize(&response->id);
    iree_string_builder_deinitialize(&response->checkpoint);
    loom_serve_realtime_response_settings_deinitialize(&response->settings);
    iree_status_ignore(response->failure);
    *response = (loom_serve_realtime_response_t){0};
    *out_progress = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_command_session(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  if (loom_serve_realtime_active_response(session)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "session settings require an idle response boundary");
  }
  bool instructions_changed = false;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_settings_update(
      &session->settings, session->profile,
      event->fields[LOOM_SERVE_REALTIME_FIELD_SESSION].text,
      session->microphone ||
          (session->turn && loom_serve_realtime_turn_is_open(session->turn)),
      &instructions_changed));
  if (instructions_changed) {
    loom_serve_realtime_history_invalidate(&session->history, 0);
  }
  return loom_serve_realtime_session_event(session, "session.updated");
}

static iree_status_t loom_serve_realtime_transcription_begin(
    loom_serve_realtime_session_t* session,
    loom_serve_realtime_transcription_t* transcription) {
  const loom_serve_recognition_request_t request = {
      .language = iree_string_builder_view(&session->settings.language),
      .prompt = iree_string_builder_view(&session->settings.prompt),
      .outputs = LOOM_SERVE_RECOGNITION_OUTPUT_TRANSCRIPT};
  return loom_serve_realtime_input_begin(session->input, &request,
                                         &transcription->input);
}

static iree_status_t loom_serve_realtime_microphone_reserve(
    loom_serve_realtime_session_t* session) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
       ++i) {
    if (session->transcriptions[i].phase ==
            LOOM_SERVE_REALTIME_TRANSCRIPTION_FREE &&
        !session->transcriptions[i].span.capture) {
      session->microphone = &session->transcriptions[i];
      *session->microphone = (loom_serve_realtime_transcription_t){
          .identity = ++session->item_identity,
          .phase = LOOM_SERVE_REALTIME_TRANSCRIPTION_WAITING,
          .automatic = session->settings.detect_turns};
      break;
    }
  }
  if (!session->microphone) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "realtime input item capacity is full");
  }
  loom_serve_realtime_capture_mark(session->capture,
                                   &session->microphone->begin);
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
      loom_serve_realtime_transcription_close(session->microphone);
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
  loom_serve_realtime_item_t* previous = session->conversation.last;
  loom_serve_realtime_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_insert_item(session, session->microphone->identity,
                                      LOOM_SERVE_REALTIME_USER, true, &item));
  if (automatic) {
    session->microphone->end =
        (loom_serve_realtime_capture_mark_t){.position = end, .reached = true};
  } else {
    loom_serve_realtime_capture_mark(session->capture,
                                     &session->microphone->end);
  }
  session->microphone->item = item;
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
  return loom_serve_realtime_item_event(session, "conversation.item.added",
                                        item, previous);
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
    loom_serve_realtime_transcription_close(session->microphone);
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

static iree_status_t loom_serve_realtime_response_create(
    loom_serve_realtime_session_t* session, iree_string_view_t overrides) {
  if (loom_serve_realtime_active_response(session)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cancel the active response before creating another");
  }
  loom_serve_realtime_response_t* response = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    if (!session->responses[i].item) {
      response = &session->responses[i];
      break;
    }
  }
  if (!response) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "cancelled response work is still retiring");
  }
  iree_string_builder_initialize(session->host_allocator, &response->id);
  iree_string_builder_initialize(session->host_allocator,
                                 &response->checkpoint);
  iree_status_t status = loom_serve_realtime_response_settings_initialize(
      &session->settings, overrides, &response->settings);
  loom_serve_realtime_item_t* previous = session->conversation.last;
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_new_item(
        session, LOOM_SERVE_REALTIME_ASSISTANT, response->settings.output.audio,
        &response->item);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_response_settings_deinitialize(&response->settings);
    iree_string_builder_deinitialize(&response->id);
    iree_string_builder_deinitialize(&response->checkpoint);
    *response = (loom_serve_realtime_response_t){0};
    return status;
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &response->id, "resp_%" PRIu64 "_%" PRIu64, session->id,
      response->item->sequence));
  session->automatic_response = false;
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_response_event(session, response, false));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_item_event(
      session, "conversation.item.added", response->item, previous));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_response_item_event(session, response, false));
  return loom_serve_realtime_response_part_event(session, response, false);
}

static iree_status_t loom_serve_realtime_command_respond(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  return loom_serve_realtime_response_create(
      session, event->present & (1u << LOOM_SERVE_REALTIME_FIELD_RESPONSE)
                   ? event->fields[LOOM_SERVE_REALTIME_FIELD_RESPONSE].text
                   : iree_string_view_empty());
}

static iree_status_t loom_serve_realtime_command_cancel(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  loom_serve_realtime_response_t* response =
      loom_serve_realtime_active_response(session);
  if (!response ||
      ((event->present & (1u << LOOM_SERVE_REALTIME_FIELD_RESPONSE_ID)) &&
       !loom_serve_realtime_string_is(
           event->fields[LOOM_SERVE_REALTIME_FIELD_RESPONSE_ID].text,
           iree_string_builder_view(&response->id)))) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "response is not active");
  }
  loom_serve_realtime_response_stop(session, response,
                                    LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED);
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_content_visit(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_OBJECT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "conversation content requires text-part objects");
  }
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"text", 1u << IREE_JSON_VALUE_TYPE_STRING}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      value, IREE_ARRAYSIZE(rules), rules, values, &present));
  if (present != 3 ||
      (!loom_serve_realtime_string_is(values[0].text, IREE_SV("input_text")) &&
       !loom_serve_realtime_string_is(values[0].text, IREE_SV("text")))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "conversation creation currently accepts text parts");
  }
  iree_string_builder_t* output = user_data;
  iree_string_builder_t decoded;
  iree_string_builder_initialize(output->allocator, &decoded);
  iree_status_t status = loom_serve_realtime_string_decode(
      values[1].text, 1024 * 1024 - output->size, &decoded);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(
        output, iree_string_builder_view(&decoded));
  }
  iree_string_builder_deinitialize(&decoded);
  return status;
}

static iree_status_t loom_serve_realtime_item_create(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event, iree_string_builder_t* id,
    iree_string_builder_t* text) {
  if (loom_serve_realtime_active_response(session)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "conversation edits require an idle response boundary");
  }
  static const loom_serve_realtime_field_rule_t rules[] = {
      {"type", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"id", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"role", 1u << IREE_JSON_VALUE_TYPE_STRING},
      {"content", 1u << IREE_JSON_VALUE_TYPE_ARRAY},
      {"status", 1u << IREE_JSON_VALUE_TYPE_STRING}};
  loom_serve_realtime_event_value_t values[IREE_ARRAYSIZE(rules)];
  uint32_t present = 0;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_object_parse(
      event->fields[LOOM_SERVE_REALTIME_FIELD_ITEM].text, IREE_ARRAYSIZE(rules),
      rules, values, &present));
  if ((present & 13) != 13 ||
      !loom_serve_realtime_string_is(values[0].text, IREE_SV("message")) ||
      ((present & 16) &&
       !loom_serve_realtime_string_is(values[4].text, IREE_SV("completed")))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "item requires a completed message with role and content");
  }
  loom_serve_realtime_role_t role;
  if (loom_serve_realtime_string_is(values[2].text, IREE_SV("user"))) {
    role = LOOM_SERVE_REALTIME_USER;
  } else if (loom_serve_realtime_string_is(values[2].text,
                                           IREE_SV("assistant"))) {
    role = LOOM_SERVE_REALTIME_ASSISTANT;
  } else if (loom_serve_realtime_string_is(values[2].text, IREE_SV("system"))) {
    role = LOOM_SERVE_REALTIME_SYSTEM;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported message role");
  }
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      values[3].text, loom_serve_realtime_content_visit, text));
  loom_serve_realtime_item_t* previous = session->conversation.last;
  if (event->present & (1u << LOOM_SERVE_REALTIME_FIELD_PREVIOUS_ITEM_ID)) {
    IREE_RETURN_IF_ERROR(loom_serve_realtime_string_decode(
        event->fields[LOOM_SERVE_REALTIME_FIELD_PREVIOUS_ITEM_ID].text, 128,
        id));
    if (iree_string_view_equal(iree_string_builder_view(id), IREE_SV("root"))) {
      previous = NULL;
    } else {
      previous = loom_serve_realtime_conversation_find(
          &session->conversation, iree_string_builder_view(id));
      if (!previous) {
        return iree_make_status(IREE_STATUS_NOT_FOUND,
                                "previous item was not found");
      }
    }
  }
  iree_string_builder_reset(id);
  if (present & 2) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_string_decode(values[1].text, 128, id));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        id, "item_%" PRIu64 "_%" PRIu64, session->id,
        session->conversation.sequence + 1));
  }
  const bool inserted = previous != session->conversation.last;
  loom_serve_realtime_item_t* item = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_insert(
      &session->conversation, previous, iree_string_builder_view(id), role,
      false, &item));
  iree_status_t status = loom_serve_realtime_conversation_append_text(
      &session->conversation, item, iree_string_builder_view(text));
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_conversation_remove(&session->conversation, item);
    return status;
  }
  item->status = LOOM_SERVE_REALTIME_ITEM_COMPLETED;
  if (inserted) {
    loom_serve_realtime_history_invalidate(&session->history, 0);
  }
  IREE_RETURN_IF_ERROR(loom_serve_realtime_item_event(
      session, "conversation.item.added", item, previous));
  return loom_serve_realtime_item_event(session, "conversation.item.done", item,
                                        previous);
}

static iree_status_t loom_serve_realtime_command_create(
    loom_serve_realtime_session_t* session,
    const loom_serve_realtime_event_t* event,
    loom_serve_websocket_message_t** message) {
  iree_string_builder_t id, text;
  iree_string_builder_initialize(session->host_allocator, &id);
  iree_string_builder_initialize(session->host_allocator, &text);
  const iree_status_t status =
      loom_serve_realtime_item_create(session, event, &id, &text);
  iree_string_builder_deinitialize(&id);
  iree_string_builder_deinitialize(&text);
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
  const bool changes_history = !truncate || item->has_transcript;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    loom_serve_realtime_response_t* response = &session->responses[i];
    if (response->item == item && !truncate) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "response item remains owned until cancellation retires");
    }
    if (changes_history && response->item && !response->stop &&
        response->finish != LOOM_SERVE_REALTIME_FINISH_RETIRED) {
      if (response->item != item || !truncate) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "conversation edits require an idle response boundary");
      }
    }
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    if (session->responses[i].item == item && !session->responses[i].stop &&
        session->responses[i].finish != LOOM_SERVE_REALTIME_FINISH_RETIRED) {
      loom_serve_realtime_response_stop(
          session, &session->responses[i],
          LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED);
    }
  }
  if (truncate) {
    loom_serve_realtime_conversation_truncate(&session->conversation, item);
  } else {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
         ++i) {
      if (session->transcriptions[i].item == item) {
        loom_serve_realtime_transcription_close(&session->transcriptions[i]);
      }
    }
  }
  if (changes_history) {
    loom_serve_realtime_history_invalidate(&session->history, item->sequence);
    loom_serve_realtime_wire_cancel(&session->wire, item->sequence);
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

static const loom_serve_realtime_command_fn_t loom_serve_realtime_commands[] = {
    loom_serve_realtime_command_session, loom_serve_realtime_command_append,
    loom_serve_realtime_command_commit,  loom_serve_realtime_command_clear,
    loom_serve_realtime_command_respond, loom_serve_realtime_command_cancel,
    loom_serve_realtime_command_create,  loom_serve_realtime_command_mutate,
    loom_serve_realtime_command_mutate,
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
      if (iree_status_is_ok(command)) {
        command =
            loom_serve_realtime_commands[event.kind](session, &event, &message);
      }
      if (!iree_status_is_ok(command)) {
        const bool terminal = iree_status_is_resource_exhausted(command);
        status = loom_serve_realtime_session_error(
            session, command, iree_string_builder_view(&event_id), false);
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
          iree_string_view_empty(), false);
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
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      &session->wire.scratch,
      ",\"item_id\":\"item_%" PRIu64 "_%" PRIu64 "\",\"audio_%s_ms\":%" PRIu64
      "}",
      session->id, session->microphone->identity, started ? "start" : "end",
      event.position / 24));
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
      loom_serve_realtime_response_t* response =
          loom_serve_realtime_active_response(session);
      if (response) {
        loom_serve_realtime_response_stop(
            session, response, LOOM_SERVE_REALTIME_STOP_TURN_DETECTED);
        if (response->item->audio) {
          // Cancellation says nothing about playback. Drop unaligned text
          // without inventing a client-owned played duration.
          loom_serve_realtime_conversation_truncate(&session->conversation,
                                                    response->item);
        }
      }
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
    if (transcription->automatic && !transcription->item) {
      end = iree_min(end, loom_serve_realtime_turn_covered(session->turn));
    }
    if (transcription->item && transcription->end.reached) {
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
        transcription->item && transcription->end.reached &&
        transcription->position == transcription->end.position) {
      loom_serve_realtime_input_finish(transcription->input);
      transcription->phase = LOOM_SERVE_REALTIME_TRANSCRIPTION_FINISHED;
      *out_progress = true;
    }
    const iree_string_builder_t* text =
        loom_serve_realtime_input_transcript(transcription->input);
    if (!text || !transcription->item) {
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
      status = loom_serve_realtime_item_event(session, "conversation.item.done",
                                              item, NULL);
    }
    if (iree_status_is_ok(status)) {
      loom_serve_realtime_transcription_close(transcription);
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
      loom_serve_realtime_active_response(session)) {
    return iree_ok_status();
  }
  bool waiting = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->transcriptions);
       ++i) {
    waiting |= session->transcriptions[i].item != NULL;
  }
  bool available = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    available |= session->responses[i].item == NULL;
  }
  if (waiting || !available) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_response_create(session, iree_string_view_empty()));
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
  session->executor = executor;
  session->registry = registry;
  session->server = server;
  session->connection = connection;
  session->socket = socket;
  session->profile = profile;
  session->observer = observer;
  session->id = id;
  session->text = text;
  session->speech = speech;
  loom_serve_realtime_conversation_initialize(host_allocator,
                                              &session->conversation);
  loom_serve_realtime_history_initialize(registry, text, host_allocator,
                                         &session->history);
  loom_serve_realtime_wire_initialize(id, socket, connection, host_allocator,
                                      &session->wire);
  iree_status_t status = loom_serve_realtime_settings_initialize(
      profile, host_allocator, &session->settings);
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
    status = loom_serve_realtime_session_event(session, "session.created");
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
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(session->responses) && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_realtime_response_advance(
        session, &session->responses[i], out_progress);
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
  bool responses = false;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(session->responses); ++i) {
    responses |= session->responses[i].item != NULL;
  }
  if (session->closing && !responses &&
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
  loom_serve_realtime_audio_deinitialize(&session->audio);
  loom_serve_realtime_conversation_deinitialize(&session->conversation);
  loom_serve_realtime_settings_deinitialize(&session->settings);
  iree_allocator_free(session->host_allocator, session);
}
