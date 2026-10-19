// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/response.h"

#include <inttypes.h>

#include "loom/serve/server/realtime/context.h"
#include "loom/serve/server/realtime/generation.h"
#include "loom/serve/server/realtime/output.h"
#include "loom/serve/service/text/tool_calls.h"

typedef enum loom_serve_realtime_response_finish_e {
  LOOM_SERVE_REALTIME_FINISH_GENERATING,
  LOOM_SERVE_REALTIME_FINISH_TEXT,
  LOOM_SERVE_REALTIME_FINISH_AUDIO,
  LOOM_SERVE_REALTIME_FINISH_CONTENT,
  LOOM_SERVE_REALTIME_FINISH_OUTPUT_ITEM,
  LOOM_SERVE_REALTIME_FINISH_CONVERSATION_ITEM,
  LOOM_SERVE_REALTIME_FINISH_CALLS,
  LOOM_SERVE_REALTIME_FINISH_RESPONSE,
  LOOM_SERVE_REALTIME_FINISH_RETIRED,
} loom_serve_realtime_response_finish_t;

typedef enum loom_serve_realtime_response_stop_e {
  LOOM_SERVE_REALTIME_STOP_NONE,
  LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED,
  LOOM_SERVE_REALTIME_STOP_TURN_DETECTED,
  LOOM_SERVE_REALTIME_STOP_FAILED,
} loom_serve_realtime_response_stop_t;

typedef enum loom_serve_realtime_call_phase_e {
  LOOM_SERVE_REALTIME_CALL_CONVERSATION_ADDED,
  LOOM_SERVE_REALTIME_CALL_OUTPUT_ADDED,
  LOOM_SERVE_REALTIME_CALL_ARGUMENTS_DELTA,
  LOOM_SERVE_REALTIME_CALL_ARGUMENTS_DONE,
  LOOM_SERVE_REALTIME_CALL_OUTPUT_DONE,
  LOOM_SERVE_REALTIME_CALL_CONVERSATION_DONE,
} loom_serve_realtime_call_phase_t;

typedef struct loom_serve_realtime_response_t {
  // Retained native-message frontier; NULL denotes an available response slot.
  loom_serve_realtime_group_t* group;
  // Real output message, created only when content becomes available.
  loom_serve_realtime_item_t* item;
  // Last announced member; later reserved calls are invisible to the client.
  loom_serve_realtime_item_t* announced;
  // Next complete native call being published, independent of its arguments.
  loom_serve_realtime_item_t* call;
  // Current call's stable output ordinal, including an optional message at
  // zero.
  iree_host_size_t call_index;
  // Next call event; completed OUTPUT_DONE establishes executable history.
  loom_serve_realtime_call_phase_t call_phase;
  // Owned stable native session and wire response ID.
  iree_string_builder_t id;
  // Copied selected checkpoint, retained through deferred native admission.
  iree_string_builder_t checkpoint;
  // Immutable response settings, captured when response.create is accepted.
  loom_serve_realtime_response_settings_t settings;
  // Immutable accepted input, including retained pending transcription results.
  loom_serve_realtime_context_t context;
  // Default context and output permit native checkpoint selection/capture.
  bool uses_history;
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

struct loom_serve_realtime_responses_t {
  // Allocator owning this owner and response storage.
  iree_allocator_t host_allocator;
  // Session namespace for independent wire/native response identities.
  uint64_t session_id;
  // Immutable application-owned assistant composition.
  const loom_serve_realtime_profile_t* profile;
  // Bound text registration, independent of replacements with the same name.
  loom_serve_realtime_binding_t text;
  // Bound speech registration, independent of voice selection.
  loom_serve_realtime_binding_t speech;
  // Shared workers, borrowed through every accepted conversion release.
  iree_task_executor_t* executor;
  // Serialized native registration and admission owner.
  loom_serve_registry_t* registry;
  // Shared diagnostic owner, retained through response retirement.
  loom_serve_http_server_t* server;
  // Nonblocking owner wake target for CPU completion.
  loom_serve_wake_callback_t observer;
  // Current session defaults, borrowed for creation and checkpoint
  // compatibility.
  loom_serve_realtime_settings_t* settings;
  // Canonical native-message groups, independent of executable state.
  loom_serve_realtime_conversation_t* conversation;
  // Native checkpoint ownership and asynchronous capture/release effects.
  loom_serve_realtime_history_t* history;
  // Ordered, bounded event admission and network credit.
  loom_serve_realtime_wire_t* wire;
  // Bounded default/independent requests, including accepted retiring work.
  loom_serve_realtime_response_t responses[4];
  // Rotating first claimant of shared connection publication credit.
  iree_host_size_t cursor;
  // Immutable speech coefficients shared by independently owned audio cursors.
  loom_serve_realtime_generation_filter_t filter;
  // Session close has withdrawn all responses and suppresses further delivery.
  bool closing;
};

static loom_serve_realtime_response_t* loom_serve_realtime_active_response(
    loom_serve_realtime_responses_t* owner) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    loom_serve_realtime_response_t* response = &owner->responses[i];
    if (response->group && !response->settings.independent && !response->stop &&
        response->finish != LOOM_SERVE_REALTIME_FINISH_RETIRED) {
      return response;
    }
  }
  return NULL;
}

static void loom_serve_realtime_response_stop(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response,
    loom_serve_realtime_response_stop_t stop) {
  if (!response->stop) {
    response->stop = stop;
  }
  response->captured = true;
  loom_serve_realtime_context_deinitialize(&response->context);
  if (!response->finish) {
    response->finish = response->item ? LOOM_SERVE_REALTIME_FINISH_TEXT
                                      : LOOM_SERVE_REALTIME_FINISH_RESPONSE;
  }
  if (response->item) {
    response->item->status = LOOM_SERVE_REALTIME_ITEM_INCOMPLETE;
  }
  loom_serve_realtime_item_t* unannounced =
      response->announced ? response->announced->next : response->group->first;
  if (response->call != response->announced) {
    response->call = NULL;
  }
  while (unannounced) {
    loom_serve_realtime_item_t* next = unannounced->next;
    loom_serve_realtime_conversation_remove(owner->conversation, unannounced);
    unannounced = next;
  }
  if (response->call &&
      response->call->status != LOOM_SERVE_REALTIME_ITEM_COMPLETED) {
    response->call->status = LOOM_SERVE_REALTIME_ITEM_INCOMPLETE;
  }
  loom_serve_realtime_wire_cancel(owner->wire, response->group->sequence);
  if (response->generation) {
    loom_serve_realtime_generation_cancel(response->generation);
  }
  // A capture already accepted for this response must still be observed and
  // released. Its predecessor remains valid until a conversation edit says so.
  if (response->uses_history) {
    loom_serve_realtime_history_invalidate(owner->history,
                                           response->group->sequence);
  }
}

static void loom_serve_realtime_response_fail(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response, iree_status_t failure) {
  loom_serve_http_server_record_error(owner->server, failure);
  if (response->stop) {
    // A late conversion failure is still diagnosed, but cannot replace the
    // terminal cause already selected by cancellation or an earlier failure.
    iree_status_ignore(failure);
    return;
  }
  loom_serve_realtime_response_stop(owner, response,
                                    LOOM_SERVE_REALTIME_STOP_FAILED);
  response->failure = failure;
  if (response->item && response->item->audio) {
    // Failed generation gives no alignment between partial text and playback.
    loom_serve_realtime_conversation_truncate(owner->conversation,
                                              response->item);
  }
}

static iree_status_t loom_serve_realtime_response_fields(
    loom_serve_realtime_responses_t* owner,
    const loom_serve_realtime_response_t* response, const char* type) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(owner->wire, type));
  iree_string_builder_t* output = &owner->wire->scratch;
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

static loom_serve_realtime_output_t loom_serve_realtime_response_view(
    const loom_serve_realtime_responses_t* owner,
    const loom_serve_realtime_response_t* response, bool done) {
  static const loom_serve_realtime_output_state_t stopped[] = {
      LOOM_SERVE_REALTIME_OUTPUT_GENERATING,
      LOOM_SERVE_REALTIME_OUTPUT_CLIENT_CANCELLED,
      LOOM_SERVE_REALTIME_OUTPUT_TURN_DETECTED,
      LOOM_SERVE_REALTIME_OUTPUT_FAILED};
  return (loom_serve_realtime_output_t){
      .id = iree_string_builder_view(&response->id),
      .conversation_id = response->settings.independent ? 0 : owner->session_id,
      .first = done && response->announced ? response->group->first : NULL,
      .last = done ? response->announced : NULL,
      .metadata = iree_string_builder_view(&response->settings.metadata),
      .audio = response->settings.output.audio,
      .state = !done            ? LOOM_SERVE_REALTIME_OUTPUT_GENERATING
               : response->stop ? stopped[response->stop]
               : response->budget_exhausted
                   ? LOOM_SERVE_REALTIME_OUTPUT_TOKEN_LIMIT
                   : LOOM_SERVE_REALTIME_OUTPUT_COMPLETED,
      .failure = response->failure,
      .input_tokens = response->input_tokens,
      .cached_tokens = response->cached_tokens,
      .output_tokens = response->output_tokens};
}

static iree_status_t loom_serve_realtime_response_event(
    loom_serve_realtime_responses_t* owner,
    const loom_serve_realtime_response_t* response, bool done) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      owner->wire, done ? "response.done" : "response.created"));
  const loom_serve_realtime_output_t view =
      loom_serve_realtime_response_view(owner, response, done);
  loom_output_stream_t output;
  loom_output_stream_for_builder(&owner->wire->scratch, &output);
  IREE_RETURN_IF_ERROR(loom_serve_realtime_output_write(&view, &output));
  return loom_serve_realtime_wire_commit(owner->wire, response->group->sequence,
                                         false);
}

static iree_status_t loom_serve_realtime_response_item_event(
    loom_serve_realtime_responses_t* owner,
    const loom_serve_realtime_response_t* response,
    const loom_serve_realtime_item_t* item, iree_host_size_t index, bool done,
    loom_serve_realtime_item_status_t status) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
      owner->wire,
      done ? "response.output_item.done" : "response.output_item.added"));
  loom_output_stream_t output;
  loom_output_stream_for_builder(&owner->wire->scratch, &output);
  IREE_RETURN_IF_ERROR(loom_serve_realtime_output_item_write(
      iree_string_builder_view(&response->id), item, index, status, !done,
      &output));
  return loom_serve_realtime_wire_commit(owner->wire, response->group->sequence,
                                         false);
}

static iree_status_t loom_serve_realtime_response_part_event(
    loom_serve_realtime_responses_t* owner,
    const loom_serve_realtime_response_t* response, bool done) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
      owner, response,
      done ? "response.content_part.done" : "response.content_part.added"));
  iree_string_builder_t* output = &owner->wire->scratch;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output, ",\"part\":{\"type\":\"%s\",\"%s\":",
      response->settings.output.audio ? "audio" : "text",
      response->settings.output.audio ? "transcript" : "text"));
  IREE_RETURN_IF_ERROR(loom_serve_realtime_string_write(
      output, done ? iree_string_builder_view(&response->item->text)
                   : iree_string_view_empty()));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}}"));
  return loom_serve_realtime_wire_commit(owner->wire, response->group->sequence,
                                         false);
}

static iree_status_t loom_serve_realtime_response_message(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response) {
  IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_add(
      owner->conversation, response->group, LOOM_SERVE_REALTIME_ITEM_MESSAGE,
      LOOM_SERVE_REALTIME_ASSISTANT, response->settings.output.audio,
      &response->item));
  loom_serve_realtime_wire_message_t* tail = owner->wire->last;
  iree_status_t status = iree_ok_status();
  if (!response->settings.independent) {
    status = loom_serve_realtime_item_event(
        owner->wire, "conversation.item.added", response->item,
        loom_serve_realtime_conversation_previous(response->group));
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_response_item_event(
        owner, response, response->item, 0, false,
        LOOM_SERVE_REALTIME_ITEM_IN_PROGRESS);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_response_part_event(owner, response, false);
  }
  if (!iree_status_is_ok(status)) {
    loom_serve_realtime_wire_rewind(owner->wire, tail);
    loom_serve_realtime_conversation_remove(owner->conversation,
                                            response->item);
    response->item = NULL;
  }
  if (iree_status_is_ok(status)) {
    response->announced = response->item;
  }
  return status;
}

static iree_status_t loom_serve_realtime_response_call_advance(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response) {
  loom_serve_realtime_item_t* call = response->call;
  if (!call) {
    response->finish = LOOM_SERVE_REALTIME_FINISH_RESPONSE;
    return iree_ok_status();
  }
  switch (response->call_phase) {
    case LOOM_SERVE_REALTIME_CALL_CONVERSATION_ADDED: {
      if (!response->settings.independent) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_item_event(
            owner->wire, "conversation.item.added", call,
            call->previous
                ? call->previous
                : loom_serve_realtime_conversation_previous(call->group)));
      }
      response->announced = call;
      break;
    }
    case LOOM_SERVE_REALTIME_CALL_OUTPUT_ADDED: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_item_event(
          owner, response, call, response->call_index, false, call->status));
      response->announced = call;
      break;
    }
    case LOOM_SERVE_REALTIME_CALL_ARGUMENTS_DELTA:
    case LOOM_SERVE_REALTIME_CALL_ARGUMENTS_DONE: {
      const bool done =
          response->call_phase == LOOM_SERVE_REALTIME_CALL_ARGUMENTS_DONE;
      if (!response->stop || done) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_begin(
            owner->wire, done ? "response.function_call_arguments.done"
                              : "response.function_call_arguments.delta"));
        loom_output_stream_t output;
        loom_output_stream_for_builder(&owner->wire->scratch, &output);
        IREE_RETURN_IF_ERROR(loom_serve_realtime_output_arguments_write(
            iree_string_builder_view(&response->id), call, response->call_index,
            done, &output));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
            owner->wire, response->group->sequence, !done));
      }
      break;
    }
    case LOOM_SERVE_REALTIME_CALL_OUTPUT_DONE: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_item_event(
          owner, response, call, response->call_index, true,
          response->stop ? LOOM_SERVE_REALTIME_ITEM_INCOMPLETE
                         : LOOM_SERVE_REALTIME_ITEM_COMPLETED));
      if (!response->stop) {
        loom_serve_realtime_item_complete(call);
      }
      break;
    }
    case LOOM_SERVE_REALTIME_CALL_CONVERSATION_DONE: {
      if (!response->settings.independent) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_item_event(
            owner->wire, "conversation.item.done", call, NULL));
      }
      response->call = call->next;
      ++response->call_index;
      response->call_phase = response->settings.independent
                                 ? LOOM_SERVE_REALTIME_CALL_OUTPUT_ADDED
                                 : LOOM_SERVE_REALTIME_CALL_CONVERSATION_ADDED;
      return iree_ok_status();
    }
  }
  ++response->call_phase;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_response_finish(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response, bool* out_progress) {
  if (!loom_serve_realtime_wire_can_publish(owner->wire)) {
    return iree_ok_status();
  }
  iree_string_builder_t* output = &owner->wire->scratch;
  switch (response->finish) {
    case LOOM_SERVE_REALTIME_FINISH_GENERATING:
    case LOOM_SERVE_REALTIME_FINISH_RETIRED:
      return iree_ok_status();
    case LOOM_SERVE_REALTIME_FINISH_TEXT: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
          owner, response,
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
          owner->wire, response->group->sequence, false));
      if (!response->settings.output.audio) {
        ++response->finish;
      }
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_AUDIO: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
          owner, response, "response.output_audio.done"));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
      IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
          owner->wire, response->group->sequence, false));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_CONTENT: {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_response_part_event(owner, response, true));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_OUTPUT_ITEM: {
      IREE_RETURN_IF_ERROR(loom_serve_realtime_response_item_event(
          owner, response, response->item, 0, true, response->item->status));
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_CONVERSATION_ITEM: {
      if (!response->settings.independent) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_item_event(
            owner->wire, "conversation.item.done", response->item, NULL));
      }
      break;
    }
    case LOOM_SERVE_REALTIME_FINISH_CALLS: {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_response_call_advance(owner, response));
      *out_progress = true;
      return iree_ok_status();
    }
    case LOOM_SERVE_REALTIME_FINISH_RESPONSE: {
      IREE_RETURN_IF_ERROR(
          loom_serve_realtime_response_event(owner, response, true));
      break;
    }
  }
  ++response->finish;
  *out_progress = true;
  return iree_ok_status();
}

static bool loom_serve_realtime_response_ready(
    const loom_serve_realtime_responses_t* owner,
    const loom_serve_realtime_response_t* response) {
  if (response->settings.independent) {
    return true;
  }
  if (loom_serve_realtime_history_capturing(owner->history)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    const loom_serve_realtime_response_t* other = &owner->responses[i];
    if (other != response && !other->settings.independent &&
        other->generation && !other->stop && !other->captured) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_serve_realtime_response_start(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response) {
  loom_serve_text_request_t request = {.execution_weight =
                                           owner->profile->execution_weight};
  iree_status_t status =
      loom_serve_realtime_context_take(&response->context, &request.body);
  if (iree_status_is_ok(status) && response->uses_history &&
      loom_serve_realtime_output_settings_text_equal(
          &response->settings.output, &owner->settings->output)) {
    status = iree_string_builder_append_string(
        &response->checkpoint,
        loom_serve_realtime_history_checkpoint(owner->history));
  }
  if (iree_status_is_ok(status)) {
    if (response->uses_history) {
      request.session = iree_string_builder_view(&response->id);
    }
    request.checkpoint = iree_string_builder_view(&response->checkpoint);
    status = loom_serve_realtime_generation_allocate(
        owner->executor, owner->registry, owner->text, owner->speech, &request,
        response->settings.output.audio
            ? iree_string_builder_view(&response->settings.output.voice)
            : iree_string_view_empty(),
        owner->profile->seed, &owner->filter, owner->observer,
        owner->host_allocator, &response->generation);
  }
  iree_string_builder_deinitialize(&request.body);
  return status;
}

static iree_status_t loom_serve_realtime_response_complete(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response) {
  const loom_serve_text_stream_result_t* result =
      loom_serve_realtime_generation_result(response->generation);
  response->input_tokens = result->response.usage.input_tokens;
  response->cached_tokens = result->response.usage.cached_tokens;
  response->output_tokens = result->response.usage.output_tokens;
  response->budget_exhausted =
      iree_string_view_equal(result->finish_reason, IREE_SV("length"));
  if (response->item) {
    response->item->status = response->budget_exhausted
                                 ? LOOM_SERVE_REALTIME_ITEM_INCOMPLETE
                                 : LOOM_SERVE_REALTIME_ITEM_COMPLETED;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < result->calls->count && iree_status_is_ok(status); ++i) {
    const loom_serve_text_tool_call_t call =
        loom_serve_text_tool_calls_at(result->calls, i);
    loom_serve_realtime_item_t* item = NULL;
    status = loom_serve_realtime_conversation_add(
        owner->conversation, response->group,
        LOOM_SERVE_REALTIME_ITEM_FUNCTION_CALL, LOOM_SERVE_REALTIME_ASSISTANT,
        false, &item);
    if (iree_status_is_ok(status)) {
      status = loom_serve_realtime_conversation_function(
          owner->conversation, item, call.id, call.name, call.arguments);
    }
  }
  if (iree_status_is_ok(status) && result->calls->count) {
    loom_serve_realtime_output_t view =
        loom_serve_realtime_response_view(owner, response, true);
    view.first = response->group->first;
    view.last = response->group->last;
    status = loom_serve_realtime_output_admit(owner->session_id, &view);
  }
  // Failure leaves every new call unannounced. response_stop removes them as
  // one transaction before native storage is cancelled or a failure is emitted.
  if (!iree_status_is_ok(status)) {
    return status;
  }
  response->call =
      response->item ? response->item->next : response->group->first;
  response->call_index = response->item ? 1 : 0;
  response->call_phase = response->settings.independent
                             ? LOOM_SERVE_REALTIME_CALL_OUTPUT_ADDED
                             : LOOM_SERVE_REALTIME_CALL_CONVERSATION_ADDED;
  response->finish = response->item   ? LOOM_SERVE_REALTIME_FINISH_TEXT
                     : response->call ? LOOM_SERVE_REALTIME_FINISH_CALLS
                                      : LOOM_SERVE_REALTIME_FINISH_RESPONSE;
  return iree_ok_status();
}

static iree_status_t loom_serve_realtime_response_advance(
    loom_serve_realtime_responses_t* owner,
    loom_serve_realtime_response_t* response, bool* out_progress) {
  if (!response->group) {
    return iree_ok_status();
  }
  if (!response->generation && !response->stop &&
      loom_serve_realtime_response_ready(owner, response)) {
    bool ready = false;
    iree_status_t status =
        loom_serve_realtime_context_ready(&response->context, &ready);
    if (iree_status_is_ok(status) && ready) {
      status = loom_serve_realtime_response_start(owner, response);
      *out_progress = true;
    }
    if (!iree_status_is_ok(status)) {
      loom_serve_realtime_response_fail(owner, response, status);
      *out_progress = true;
    }
  }
  if (response->generation) {
    iree_status_t status = loom_serve_realtime_generation_advance(
        response->generation,
        !owner->closing && loom_serve_realtime_wire_can_publish(owner->wire),
        out_progress);
    if (!iree_status_is_ok(status)) {
      loom_serve_realtime_response_fail(owner, response, status);
      *out_progress = true;
    }
    if (!response->stop && !response->item &&
        loom_serve_realtime_wire_can_publish(owner->wire) &&
        (loom_serve_realtime_generation_text(response->generation).size ||
         loom_serve_realtime_generation_audio(response->generation).size)) {
      status = loom_serve_realtime_response_message(owner, response);
      if (!iree_status_is_ok(status)) {
        loom_serve_realtime_response_fail(owner, response, status);
      }
      *out_progress = true;
    }
    if (response->item) {
      response->item->generated_frames =
          loom_serve_realtime_generation_frames(response->generation);
    }
    if (!response->stop) {
      const iree_string_view_t audio =
          loom_serve_realtime_generation_audio(response->generation);
      if (audio.size && loom_serve_realtime_wire_can_publish(owner->wire)) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
            owner, response, "response.output_audio.delta"));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            &owner->wire->scratch, ",\"delta\":\""));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_string(&owner->wire->scratch, audio));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(&owner->wire->scratch, "\"}"));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
            owner->wire, response->group->sequence, true));
        loom_serve_realtime_generation_consume_audio(response->generation);
        *out_progress = true;
      }
      const iree_string_view_t text =
          loom_serve_realtime_generation_text(response->generation);
      if (text.size && loom_serve_realtime_wire_can_publish(owner->wire)) {
        IREE_RETURN_IF_ERROR(loom_serve_realtime_conversation_append_text(
            owner->conversation, response->item, text));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_response_fields(
            owner, response,
            response->settings.output.audio
                ? "response.output_audio_transcript.delta"
                : "response.output_text.delta"));
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
            &owner->wire->scratch, ",\"delta\":"));
        IREE_RETURN_IF_ERROR(
            loom_serve_realtime_string_write(&owner->wire->scratch, text));
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(&owner->wire->scratch, "}"));
        IREE_RETURN_IF_ERROR(loom_serve_realtime_wire_commit(
            owner->wire, response->group->sequence, true));
        loom_serve_realtime_generation_consume_text(response->generation);
        *out_progress = true;
      }
      if (!response->finish &&
          loom_serve_realtime_generation_complete(response->generation)) {
        status = loom_serve_realtime_response_complete(owner, response);
        if (!iree_status_is_ok(status)) {
          loom_serve_realtime_response_fail(owner, response, status);
        }
        *out_progress = true;
      }
      if (!response->stop && !response->captured &&
          response->finish >= LOOM_SERVE_REALTIME_FINISH_RESPONSE) {
        if (response->uses_history && response->group->first &&
            loom_serve_realtime_output_settings_text_equal(
                &response->settings.output, &owner->settings->output)) {
          IREE_RETURN_IF_ERROR(loom_serve_realtime_history_capture(
              owner->history, iree_string_builder_view(&response->id),
              response->group->sequence, &response->captured));
        } else {
          response->captured = true;
        }
        *out_progress |= response->captured;
      }
    }
  }
  if (!owner->closing) {
    IREE_RETURN_IF_ERROR(
        loom_serve_realtime_response_finish(owner, response, out_progress));
  }
  if ((owner->closing ||
       response->finish == LOOM_SERVE_REALTIME_FINISH_RETIRED) &&
      response->captured &&
      (!response->generation ||
       !loom_serve_realtime_generation_pending(response->generation))) {
    if (response->generation) {
      loom_serve_realtime_generation_cancel(response->generation);
      loom_serve_realtime_generation_free(response->generation);
    }
    loom_serve_realtime_context_deinitialize(&response->context);
    iree_string_builder_deinitialize(&response->id);
    iree_string_builder_deinitialize(&response->checkpoint);
    loom_serve_realtime_response_settings_deinitialize(&response->settings);
    iree_status_ignore(response->failure);
    loom_serve_realtime_conversation_release(owner->conversation,
                                             response->group);
    *response = (loom_serve_realtime_response_t){0};
    *out_progress = true;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_responses_create(
    loom_serve_realtime_responses_t* owner, iree_string_view_t overrides,
    bool* out_default) {
  if (out_default) {
    *out_default = false;
  }
  loom_serve_realtime_response_t* response = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    if (!owner->responses[i].group) {
      response = &owner->responses[i];
      break;
    }
  }
  if (!response) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "realtime response capacity is full, including retiring work");
  }
  iree_string_builder_initialize(owner->host_allocator, &response->id);
  iree_string_builder_initialize(owner->host_allocator, &response->checkpoint);
  iree_string_view_t input;
  iree_status_t status = loom_serve_realtime_response_settings_initialize(
      owner->settings, overrides, &response->settings, &input);
  if (iree_status_is_ok(status) && !response->settings.independent &&
      loom_serve_realtime_active_response(owner)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cancel the active default response before creating another");
  }
  if (iree_status_is_ok(status)) {
    response->uses_history = !response->settings.independent && !input.data;
    status = loom_serve_realtime_context_initialize(
        owner->conversation, input, owner->text.name,
        iree_string_builder_view(&response->settings.output.instructions),
        loom_serve_realtime_output_settings_tools(&response->settings.output),
        response->settings.output.max_output_tokens, owner->profile->seed,
        &response->context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_conversation_begin(
        owner->conversation, response->settings.independent, &response->group);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &response->id, "resp_%" PRIu64 "_%" PRIu64, owner->session_id,
        response->group->sequence);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_realtime_response_event(owner, response, false);
  }
  if (!iree_status_is_ok(status)) {
    if (response->group) {
      loom_serve_realtime_conversation_release(owner->conversation,
                                               response->group);
    }
    loom_serve_realtime_response_settings_deinitialize(&response->settings);
    loom_serve_realtime_context_deinitialize(&response->context);
    iree_string_builder_deinitialize(&response->id);
    iree_string_builder_deinitialize(&response->checkpoint);
    *response = (loom_serve_realtime_response_t){0};
    return status;
  }
  if (out_default) {
    *out_default = !response->settings.independent;
  }
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_responses_allocate(
    uint64_t session_id, const loom_serve_realtime_profile_t* profile,
    loom_serve_realtime_binding_t text, loom_serve_realtime_binding_t speech,
    iree_task_executor_t* executor, loom_serve_registry_t* registry,
    loom_serve_http_server_t* server, loom_serve_wake_callback_t observer,
    loom_serve_realtime_settings_t* settings,
    loom_serve_realtime_conversation_t* conversation,
    loom_serve_realtime_history_t* history, loom_serve_realtime_wire_t* wire,
    iree_allocator_t host_allocator,
    loom_serve_realtime_responses_t** out_owner) {
  *out_owner = NULL;
  loom_serve_realtime_responses_t* owner = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*owner), (void**)&owner));
  *owner = (loom_serve_realtime_responses_t){.host_allocator = host_allocator,
                                             .session_id = session_id,
                                             .profile = profile,
                                             .text = text,
                                             .speech = speech,
                                             .executor = executor,
                                             .registry = registry,
                                             .server = server,
                                             .observer = observer,
                                             .settings = settings,
                                             .conversation = conversation,
                                             .history = history,
                                             .wire = wire};
  *out_owner = owner;
  return iree_ok_status();
}

void loom_serve_realtime_responses_free(
    loom_serve_realtime_responses_t* owner) {
  if (owner) {
    loom_serve_realtime_generation_filter_deinitialize(&owner->filter);
    iree_allocator_free(owner->host_allocator, owner);
  }
}

bool loom_serve_realtime_responses_active(
    loom_serve_realtime_responses_t* owner) {
  return loom_serve_realtime_active_response(owner) != NULL;
}

bool loom_serve_realtime_responses_available(
    const loom_serve_realtime_responses_t* owner) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    if (!owner->responses[i].group) {
      return true;
    }
  }
  return false;
}

bool loom_serve_realtime_responses_pending(
    const loom_serve_realtime_responses_t* owner) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    if (owner->responses[i].group) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_serve_realtime_responses_cancel(
    loom_serve_realtime_responses_t* owner, iree_string_view_t encoded_id) {
  loom_serve_realtime_response_t* response = NULL;
  if (encoded_id.data) {
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
      loom_serve_realtime_response_t* candidate = &owner->responses[i];
      if (candidate->group && !candidate->stop &&
          candidate->finish != LOOM_SERVE_REALTIME_FINISH_RETIRED &&
          loom_serve_realtime_string_is(
              encoded_id, iree_string_builder_view(&candidate->id))) {
        response = candidate;
        break;
      }
    }
  } else {
    response = loom_serve_realtime_active_response(owner);
  }
  if (!response) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "response is not active");
  }
  loom_serve_realtime_response_stop(owner, response,
                                    LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED);
  return iree_ok_status();
}

void loom_serve_realtime_responses_interrupt(
    loom_serve_realtime_responses_t* owner) {
  loom_serve_realtime_response_t* response =
      loom_serve_realtime_active_response(owner);
  if (response) {
    loom_serve_realtime_response_stop(owner, response,
                                      LOOM_SERVE_REALTIME_STOP_TURN_DETECTED);
    if (response->item && response->item->audio) {
      loom_serve_realtime_conversation_truncate(owner->conversation,
                                                response->item);
    }
  }
}

void loom_serve_realtime_responses_stop(
    loom_serve_realtime_responses_t* owner) {
  owner->closing = true;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    if (owner->responses[i].group) {
      loom_serve_realtime_response_stop(
          owner, &owner->responses[i],
          LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED);
    }
  }
}

iree_status_t loom_serve_realtime_responses_edit(
    loom_serve_realtime_responses_t* owner, loom_serve_realtime_item_t* item,
    bool truncate, bool changes_history) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    loom_serve_realtime_response_t* response = &owner->responses[i];
    if (response->group == item->group && !truncate) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "response item remains owned until cancellation retires");
    }
    if (changes_history &&
        item->group->owner == &owner->conversation->history &&
        !response->settings.independent && response->group && !response->stop &&
        response->finish != LOOM_SERVE_REALTIME_FINISH_RETIRED &&
        (response->item != item || !truncate)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "conversation edits require an idle response boundary");
    }
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(owner->responses); ++i) {
    if (owner->responses[i].item == item && !owner->responses[i].stop &&
        owner->responses[i].finish != LOOM_SERVE_REALTIME_FINISH_RETIRED) {
      loom_serve_realtime_response_stop(
          owner, &owner->responses[i],
          LOOM_SERVE_REALTIME_STOP_CLIENT_CANCELLED);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_realtime_responses_advance(
    loom_serve_realtime_responses_t* owner, bool closing, bool* out_progress) {
  owner->closing = closing;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(owner->responses) && iree_status_is_ok(status); ++i) {
    const iree_host_size_t index =
        (owner->cursor + i) % IREE_ARRAYSIZE(owner->responses);
    status = loom_serve_realtime_response_advance(
        owner, &owner->responses[index], out_progress);
  }
  owner->cursor = (owner->cursor + 1) % IREE_ARRAYSIZE(owner->responses);
  return status;
}
