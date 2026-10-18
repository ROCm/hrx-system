// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/output.h"

#include <inttypes.h>
#include <string.h>

#include "loom/util/json.h"

typedef struct loom_serve_realtime_output_outcome_t {
  // Public response status for this outcome.
  const char* status;
  // Public stop reason, or NULL when the outcome has no reason field.
  const char* reason;
} loom_serve_realtime_output_outcome_t;

iree_status_t loom_serve_realtime_item_write(
    const loom_serve_realtime_item_t* item,
    loom_serve_realtime_item_status_t status, bool initial,
    loom_output_stream_t* output) {
  static const char* const statuses[] = {"in_progress", "completed",
                                         "incomplete"};
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "{\"id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      output, iree_string_builder_view(&item->id)));
  if (item->kind != LOOM_SERVE_REALTIME_ITEM_MESSAGE) {
    const bool call = item->kind == LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL;
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output, ",\"type\":\"%s\",\"status\":\"%s\",\"call_id\":",
        call ? "function_call" : "function_call_output", statuses[status]));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        output, iree_string_builder_view(&item->call_id)));
    if (call) {
      IREE_RETURN_IF_ERROR(
          loom_output_stream_write_cstring(output, ",\"name\":"));
      IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
          output, iree_string_builder_view(&item->name)));
    }
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output, ",\"%s\":", call ? "arguments" : "output"));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        output, call && initial ? iree_string_view_empty()
                                : iree_string_builder_view(&item->text)));
    return loom_output_stream_write_cstring(output, "}");
  }
  static const char* const loom_serve_realtime_roles[] = {"user", "assistant",
                                                          "system"};
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      ",\"type\":\"message\",\"role\":\"%s\",\"status\":\"%s\","
      "\"content\":[{\"type\":\"%s\",\"%s\":",
      loom_serve_realtime_roles[item->role], statuses[status],
      item->audio
          ? (item->role == LOOM_SERVE_REALTIME_USER ? "input_audio" : "audio")
          : (item->role == LOOM_SERVE_REALTIME_USER ? "input_text" : "text"),
      item->audio ? "transcript" : "text"));
  if (item->has_transcript) {
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        output, iree_string_builder_view(&item->text)));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
  }
  return loom_output_stream_write_cstring(output, "}]}");
}

iree_status_t loom_serve_realtime_output_write(
    const loom_serve_realtime_output_t* view, loom_output_stream_t* output) {
  static const loom_serve_realtime_output_outcome_t outcomes[] = {
      {"in_progress", NULL},
      {"completed", NULL},
      {"incomplete", "max_output_tokens"},
      {"cancelled", "client_cancelled"},
      {"cancelled", "turn_detected"},
      {"failed", NULL}};
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"response\":{\"id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(output, view->id));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output,
      ",\"object\":\"realtime.response\",\"status\":\"%s\","
      "\"status_details\":",
      outcomes[view->state].status));
  if (view->state == LOOM_SERVE_REALTIME_OUTPUT_FAILED) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output,
        "{\"type\":\"failed\",\"error\":{\"type\":\"server_error\","
        "\"code\":\"%s\",\"message\":",
        iree_status_code_string(iree_status_code(view->failure))));
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        output, iree_status_message(view->failure)));
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "}}"));
  } else if (outcomes[view->state].reason) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output, "{\"type\":\"%s\",\"reason\":\"%s\"}",
        outcomes[view->state].status, outcomes[view->state].reason));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"output\":["));
  iree_status_t status = iree_ok_status();
  const loom_serve_realtime_item_t* end = view->last ? view->last->next : NULL;
  for (const loom_serve_realtime_item_t* item = view->first;
       item != end && iree_status_is_ok(status); item = item->next) {
    if (item != view->first) {
      status = loom_output_stream_write_char(output, ',');
    }
    if (iree_status_is_ok(status)) {
      const loom_serve_realtime_item_status_t item_status =
          item->kind == LOOM_SERVE_REALTIME_ITEM_MESSAGE &&
                  view->state >= LOOM_SERVE_REALTIME_OUTPUT_CLIENT_CANCELLED
              ? LOOM_SERVE_REALTIME_ITEM_INCOMPLETE
              : item->status;
      status = loom_serve_realtime_item_write(item, item_status, false, output);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output, "],\"output_modalities\":[\"%s\"],\"metadata\":",
      view->audio ? "audio" : "text"));
  IREE_RETURN_IF_ERROR(loom_output_stream_write(
      output, view->metadata.size ? view->metadata : IREE_SV("null")));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, ",\"usage\":"));
  if (view->state == LOOM_SERVE_REALTIME_OUTPUT_COMPLETED ||
      view->state == LOOM_SERVE_REALTIME_OUTPUT_TOKEN_LIMIT) {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
        output,
        "{\"total_tokens\":%" PRIhsz ",\"input_tokens\":%" PRIhsz
        ",\"output_tokens\":%" PRIhsz
        ",\"input_token_details\":{"
        "\"cached_tokens\":%" PRIhsz "}}",
        view->input_tokens + view->output_tokens, view->input_tokens,
        view->output_tokens, view->cached_tokens));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
  }
  return loom_output_stream_write_cstring(output, "}}");
}

static iree_status_t loom_serve_realtime_output_fields(
    iree_string_view_t response_id, const loom_serve_realtime_item_t* item,
    iree_host_size_t index, loom_output_stream_t* output) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"response_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(output, response_id));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"item_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      output, iree_string_builder_view(&item->id)));
  return loom_output_stream_write_format(output, ",\"output_index\":%" PRIhsz,
                                         index);
}

iree_status_t loom_serve_realtime_output_item_write(
    iree_string_view_t response_id, const loom_serve_realtime_item_t* item,
    iree_host_size_t index, loom_serve_realtime_item_status_t status,
    bool initial, loom_output_stream_t* output) {
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_output_fields(response_id, item, index, output));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, ",\"item\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_item_write(item, status, initial, output));
  return loom_output_stream_write_char(output, '}');
}

iree_status_t loom_serve_realtime_output_arguments_write(
    iree_string_view_t response_id, const loom_serve_realtime_item_t* item,
    iree_host_size_t index, bool done, loom_output_stream_t* output) {
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_output_fields(response_id, item, index, output));
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"call_id\":"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      output, iree_string_builder_view(&item->call_id)));
  IREE_RETURN_IF_ERROR(loom_output_stream_write_format(
      output, ",\"%s\":", done ? "arguments" : "delta"));
  IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
      output, iree_string_builder_view(&item->text)));
  return loom_output_stream_write_char(output, '}');
}

iree_status_t loom_serve_realtime_output_conversation_write(
    const loom_serve_realtime_item_t* item,
    const loom_serve_realtime_item_t* previous, bool initial,
    loom_output_stream_t* output) {
  IREE_RETURN_IF_ERROR(
      loom_output_stream_write_cstring(output, ",\"previous_item_id\":"));
  if (previous) {
    IREE_RETURN_IF_ERROR(loom_json_write_escaped_string(
        output, iree_string_builder_view(&previous->id)));
  } else {
    IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, "null"));
  }
  IREE_RETURN_IF_ERROR(loom_output_stream_write_cstring(output, ",\"item\":"));
  IREE_RETURN_IF_ERROR(
      loom_serve_realtime_item_write(item, item->status, initial, output));
  return loom_output_stream_write_char(output, '}');
}

iree_status_t loom_serve_realtime_item_event(
    loom_serve_realtime_wire_t* wire, const char* type,
    const loom_serve_realtime_item_t* item,
    const loom_serve_realtime_item_t* previous) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(wire, type));
  loom_output_stream_t output;
  loom_output_stream_for_builder(&wire->scratch, &output);
  IREE_RETURN_IF_ERROR(loom_serve_realtime_output_conversation_write(
      item, previous,
      item->status == LOOM_SERVE_REALTIME_ITEM_IN_PROGRESS &&
          strcmp(type, "conversation.item.added") == 0,
      &output));
  return loom_serve_realtime_wire_commit(wire, 0, false);
}

static iree_status_t loom_serve_realtime_output_measure_begin(
    uint64_t session_id, const char* type, loom_output_stream_t* output) {
  loom_output_stream_null(output);
  return loom_serve_realtime_wire_prefix(session_id, UINT64_MAX, type, output);
}

iree_status_t loom_serve_realtime_output_admit(
    uint64_t session_id, const loom_serve_realtime_output_t* view) {
  const loom_serve_realtime_output_state_t states[] = {
      view->state, LOOM_SERVE_REALTIME_OUTPUT_CLIENT_CANCELLED,
      LOOM_SERVE_REALTIME_OUTPUT_TURN_DETECTED};
  iree_status_t status = iree_ok_status();
  loom_output_stream_t output;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(states) && iree_status_is_ok(status); ++i) {
    loom_serve_realtime_output_t terminal = *view;
    terminal.state = states[i];
    status = loom_serve_realtime_output_measure_begin(session_id,
                                                      "response.done", &output);
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_output_write(&terminal, &output);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_wire_check_size(output.offset);
    }
  }
  iree_host_size_t index = 0;
  const loom_serve_realtime_item_t* end = view->last ? view->last->next : NULL;
  for (const loom_serve_realtime_item_t* item = view->first;
       item != end && iree_status_is_ok(status); item = item->next, ++index) {
    if (item->kind != LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL) {
      continue;
    }
    static const char* const types[] = {
        "conversation.item.added",
        "conversation.item.done",
        "response.output_item.added",
        "response.output_item.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done"};
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(types) && iree_status_is_ok(status); ++i) {
      status = loom_serve_realtime_output_measure_begin(session_id, types[i],
                                                        &output);
      if (iree_status_is_ok(status)) {
        if (i < 2) {
          status = loom_serve_realtime_output_conversation_write(
              item,
              item->previous
                  ? item->previous
                  : loom_serve_realtime_conversation_previous(item->group),
              i == 0, &output);
        } else if (i < 4) {
          status = loom_serve_realtime_output_item_write(
              view->id, item, index, item->status, i == 2, &output);
        } else {
          status = loom_serve_realtime_output_arguments_write(
              view->id, item, index, i == 5, &output);
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_serve_realtime_wire_check_size(output.offset);
      }
    }
  }
  return status;
}
