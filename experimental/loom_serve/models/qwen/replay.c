// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-trajectory replay of retained chat through the real model and planner.
// Predictions remain observable but never change a later input or arrival.

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/scheduling/packing.h"
#include "experimental/loom_serve/text/chat.h"
#include "experimental/loom_serve/text/flags.h"
#include "iree/base/internal/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, workload, "",
          "JSON sessions containing request/response turns.");
IREE_FLAG_LIST(string, window,
               "Comma-separated epoch indexes allowed in one replay window; "
               "repeat to interleave policies in the same residency.");

enum {
  QWEN_REPLAY_ROWS = LOOM_SERVE_TEXT_ROW_CAPACITY,
  QWEN_REPLAY_SHAPES = 8
};

static volatile sig_atomic_t qwen_replay_interrupted = 0;

static void qwen_replay_interrupt(int signal_number) {
  qwen_replay_interrupted = signal_number;
}

typedef struct qwen_replay_turn_t {
  // Consumed position at which the prompt selects the first output token.
  iree_host_size_t prompt_end;
  // Final consumed position, excluding the last selected, pending token.
  iree_host_size_t end;
} qwen_replay_turn_t;

typedef struct qwen_replay_row_t {
  // Owned fixed input trajectory, including each turn's final pending token.
  int32_t* tokens;
  // Number of populated trajectory IDs, bounded by the model context.
  iree_host_size_t token_count;
  // Owned first-window predictions indexed by consumed position.
  int32_t* reference;
  // Owned request boundaries into tokens.
  qwen_replay_turn_t* turns;
  // Number of retained requests in this session.
  iree_host_size_t turn_count;
  // Current request; turn_count means the session has finished.
  iree_host_size_t turn;
  // Consumed model position, independent of its selected prediction.
  iree_host_size_t position;
} qwen_replay_row_t;

typedef struct qwen_replay_window_t {
  // Allowed shapes copied from the model's immutable directory inventory.
  loom_serve_packing_shape_t shapes[QWEN_REPLAY_SHAPES];
  // Model-stage index corresponding to each allowed shape.
  iree_host_size_t indexes[QWEN_REPLAY_SHAPES];
  // Number of populated choices.
  iree_host_size_t count;
  // Largest allowed token capacity, also the per-row prompt chunk bound.
  iree_host_size_t chunk_size;
} qwen_replay_window_t;

static iree_status_t qwen_replay_window_parse(
    loom_serve_text_model_t* model, iree_string_view_t text,
    qwen_replay_window_t* out_window) {
  memset(out_window, 0, sizeof(*out_window));
  const iree_host_size_t shape_count = loom_serve_text_model_shape_count(model);
  const loom_serve_packing_shape_t* shapes =
      loom_serve_text_model_shapes(model);
  iree_string_view_t remaining = text;
  do {
    iree_string_view_t value;
    const intptr_t separator =
        iree_string_view_split(remaining, ',', &value, &remaining);
    uint32_t index = 0;
    if (!iree_string_view_atoi_uint32(value, &index) || index >= shape_count ||
        out_window->count == QWEN_REPLAY_SHAPES ||
        (separator >= 0 && !remaining.size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid epoch selection '%.*s'", (int)text.size,
                              text.data);
    }
    for (iree_host_size_t i = 0; i < out_window->count; ++i) {
      if (out_window->indexes[i] == index) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "repeated epoch index in '%.*s'",
                                (int)text.size, text.data);
      }
    }
    const iree_host_size_t next = out_window->count++;
    out_window->indexes[next] = index;
    out_window->shapes[next] = shapes[index];
    out_window->chunk_size =
        iree_max(out_window->chunk_size, shapes[index].token_capacity);
  } while (remaining.size);
  return iree_ok_status();
}

static iree_status_t qwen_replay_append(
    loom_serve_text_model_t* model, iree_string_view_t text,
    loom_serve_text_chat_boundary_t boundary, qwen_replay_row_t* row,
    iree_allocator_t allocator) {
  const iree_host_size_t capacity =
      loom_serve_text_model_context_capacity(model);
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_prepare_input(
      loom_serve_text_model_chat_policy(model), text,
      LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED, boundary,
      capacity - row->token_count, row->tokens + row->token_count, &count,
      allocator));
  row->token_count += count;
  return iree_ok_status();
}

static iree_status_t qwen_replay_prepare_turn(
    loom_serve_text_model_t* model, iree_string_view_t fixture,
    qwen_replay_row_t* row, qwen_replay_turn_t* turn,
    loom_serve_text_chat_completion_t* completion, iree_allocator_t allocator) {
  iree_string_view_t request, response;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(fixture, IREE_SV("request"), &request));
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_object_value(fixture, IREE_SV("response"), &response));
  char finish_reason[16];
  iree_host_size_t finish_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_lookup_string(
      fixture, IREE_SV("finish_reason"),
      iree_make_mutable_string_view(finish_reason, sizeof(finish_reason)),
      &finish_length));
  if (!iree_string_view_equal(
          iree_make_string_view(finish_reason, finish_length),
          IREE_SV("length"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "text replay requires length-capped responses; "
                            "EOS needs the original selected token IDs");
  }
  loom_serve_text_chat_t chat;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_chat_initialize(loom_serve_text_model_chat_policy(model),
                                      request, 192, allocator, &chat));
  iree_string_builder_t text, tool_calls;
  iree_string_builder_initialize(allocator, &text);
  iree_string_builder_initialize(allocator, &tool_calls);
  const iree_string_view_t prompt = iree_string_builder_view(&chat.prompt);
  const iree_string_view_t prior = completion->transcript;
  iree_status_t status = iree_ok_status();
  iree_string_view_t input = prompt;
  loom_serve_text_chat_boundary_t boundary =
      LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH;
  if (prior.size) {
    if (!iree_string_view_starts_with(prompt, prior)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "recorded request changes retained history");
    } else {
      // The previous pending token is already last in the fixed trajectory.
      // The live service prefixes that ID and then encodes this same suffix.
      boundary = LOOM_SERVE_TEXT_CHAT_BOUNDARY_OPEN;
      input = iree_string_view_substr(prompt, prior.size, IREE_HOST_SIZE_MAX);
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen_replay_append(model, input, boundary, row, allocator);
  }
  turn->prompt_end = row->token_count;
  iree_string_builder_reset(&text);
  char* response_data = NULL;
  iree_host_size_t response_capacity = 0;
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_reserve_for_append(
        &text, response.size, &response_data, &response_capacity);
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t length = 0;
    status = iree_json_unescape_string(response, response_capacity,
                                       response_data, &length);
    if (iree_status_is_ok(status)) {
      iree_string_builder_commit_append(&text, length);
      status = qwen_replay_append(model, iree_string_builder_view(&text),
                                  LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH, row,
                                  allocator);
    }
  }
  if (iree_status_is_ok(status) &&
      row->token_count - turn->prompt_end != chat.max_tokens) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "recorded text re-encodes to %zu tokens, expected %zu; "
        "use original token IDs for this trajectory",
        row->token_count - turn->prompt_end, chat.max_tokens);
  }
  if (iree_status_is_ok(status)) {
    turn->end = row->token_count - 1;
    status = loom_serve_text_chat_complete(
        &chat, iree_string_builder_view(&text), 0, 0, &tool_calls, completion);
  }
  iree_string_builder_deinitialize(&tool_calls);
  iree_string_builder_deinitialize(&text);
  loom_serve_text_chat_deinitialize(&chat);
  return status;
}

static iree_status_t qwen_replay_prepare(loom_serve_text_model_t* model,
                                         iree_string_view_t sessions,
                                         iree_host_size_t row_count,
                                         qwen_replay_row_t* rows,
                                         iree_allocator_t allocator) {
  const iree_host_size_t capacity =
      loom_serve_text_model_context_capacity(model);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
       ++i) {
    qwen_replay_row_t* row = &rows[i];
    iree_string_view_t fixture, turns;
    status = iree_json_array_get(sessions, i, &fixture);
    if (iree_status_is_ok(status)) {
      status = iree_json_lookup_object_value(fixture, IREE_SV("turns"), &turns);
    }
    if (iree_status_is_ok(status)) {
      status = iree_json_array_length(turns, &row->turn_count);
    }
    if (iree_status_is_ok(status) &&
        (!row->turn_count || row->turn_count > capacity)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "session needs nonempty turns fitting context");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(allocator,
                                     row->turn_count * sizeof(*row->turns),
                                     (void**)&row->turns);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(allocator, capacity * sizeof(int32_t),
                                     (void**)&row->tokens);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(allocator, capacity * sizeof(int32_t),
                                     (void**)&row->reference);
    }
    loom_serve_text_chat_completion_t completion = {0};
    for (iree_host_size_t j = 0;
         j < row->turn_count && iree_status_is_ok(status); ++j) {
      iree_string_view_t turn;
      status = iree_json_array_get(turns, j, &turn);
      if (iree_status_is_ok(status)) {
        status = qwen_replay_prepare_turn(model, turn, row, &row->turns[j],
                                          &completion, allocator);
      }
    }
    loom_serve_text_chat_completion_deinitialize(&completion);
    if (iree_status_is_ok(status)) {
      printf("{\"event\":\"trajectory\",\"row\":%zu,\"turns\":[", i);
      for (iree_host_size_t j = 0; j < row->turn_count; ++j) {
        printf("%s{\"prompt_end\":%zu,\"end\":%zu}", j ? "," : "",
               row->turns[j].prompt_end, row->turns[j].end);
      }
      printf("],\"tokens\":[");
      for (iree_host_size_t j = 0; j < row->token_count; ++j) {
        printf("%s%d", j ? "," : "", row->tokens[j]);
      }
      printf("]}\n");
      if (fflush(stdout) != 0 || ferror(stdout)) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS, "writing fixed trajectory");
      }
    }
  }
  return status;
}

static iree_status_t qwen_replay_run(loom_serve_text_model_t* model,
                                     const qwen_replay_window_t* policy,
                                     iree_host_size_t row_count,
                                     qwen_replay_row_t* rows,
                                     iree_host_size_t window) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
       ++i) {
    status = loom_serve_text_row_reset(loom_serve_text_model_row(model, i));
    rows[i].turn = 0;
    rows[i].position = 0;
  }
  IREE_RETURN_IF_ERROR(status);
  iree_host_size_t cursor = 0, epoch = 0, differences = 0;
  uint64_t prefill_tokens = 0, decode_tokens = 0, outputs = 0;
  iree_duration_t model_duration = 0;
  const iree_time_t start = iree_time_now();
  while (iree_status_is_ok(status) && !qwen_replay_interrupted) {
    loom_serve_ready_span_t ready[QWEN_REPLAY_ROWS] = {0};
    for (iree_host_size_t i = 0; i < row_count; ++i) {
      const qwen_replay_row_t* row = &rows[i];
      if (row->turn < row->turn_count) {
        const qwen_replay_turn_t* turn = &row->turns[row->turn];
        ready[i].token_count = row->position < turn->prompt_end
                                   ? turn->prompt_end - row->position
                                   : 1;
        ready[i].minimum_count = 1;
      }
    }
    loom_serve_packed_span_t scheduled[QWEN_REPLAY_ROWS];
    loom_serve_packed_span_t scratch[QWEN_REPLAY_ROWS];
    iree_host_size_t choice = 0;
    const iree_host_size_t count = loom_serve_pack_shapes(
        row_count, ready, policy->count, policy->shapes, policy->chunk_size,
        &cursor, scheduled, scratch, &choice);
    if (!count) {
      break;
    }
    loom_serve_text_span_t spans[QWEN_REPLAY_ROWS];
    iree_host_size_t prompt_count = 0, decode_count = 0, output_count = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      const iree_host_size_t index = scheduled[i].row_index;
      const qwen_replay_row_t* row = &rows[index];
      const qwen_replay_turn_t* turn = &row->turns[row->turn];
      const iree_host_size_t length = scheduled[i].token_count;
      const bool prompt = row->position < turn->prompt_end;
      const bool select = !prompt || row->position + length == turn->prompt_end;
      spans[i] = (loom_serve_text_span_t){
          index, length, row->tokens + row->position,
          select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0};
      prompt_count += prompt ? length : 0;
      decode_count += prompt ? 0 : length;
      output_count += select;
    }
    const iree_time_t issue = iree_time_now();
    status = loom_serve_text_model_epoch(model, policy->indexes[choice], count,
                                         spans);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const iree_duration_t duration = iree_time_now() - issue;
    model_duration += duration;
    prefill_tokens += prompt_count;
    decode_tokens += decode_count;
    outputs += output_count;
    printf(
        "{\"event\":\"epoch\",\"window\":%zu,\"epoch\":%zu,"
        "\"shape\":%zu,\"token_capacity\":%zu,\"spans\":%zu,"
        "\"prefill_tokens\":%zu,\"decode_tokens\":%zu,"
        "\"selected_tokens\":%zu,\"model_ns\":%" PRId64 ",\"rows\":[",
        window, epoch++, policy->indexes[choice],
        policy->shapes[choice].token_capacity, count, prompt_count,
        decode_count, output_count, duration);
    for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
      const loom_serve_text_span_t* span = &spans[i];
      qwen_replay_row_t* row = &rows[span->row_index];
      row->position += span->token_count;
      const bool selected =
          iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT);
      const loom_serve_text_row_t* device_row =
          loom_serve_text_model_row(model, span->row_index);
      if (loom_serve_text_row_position(device_row) != row->position) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "row %zu frontier differs", span->row_index);
      }
      const int32_t prediction =
          selected ? loom_serve_text_row_token(device_row) : -1;
      if (selected) {
        if (!window) {
          row->reference[row->position] = prediction;
        }
        differences += prediction != row->reference[row->position];
      }
      printf(
          "%s{\"row\":%zu,\"turn\":%zu,\"position\":%zu,"
          "\"tokens\":%zu,\"prediction\":%d,\"reference\":%d}",
          i ? "," : "", span->row_index, row->turn, row->position,
          span->token_count, prediction,
          selected ? row->reference[row->position] : -1);
      if (row->position == row->turns[row->turn].end) {
        ++row->turn;
      }
    }
    printf("]}\n");
    if (fflush(stdout) != 0 || ferror(stdout)) {
      status = iree_status_join(
          status,
          iree_make_status(IREE_STATUS_DATA_LOSS, "writing replay epoch"));
    }
  }
  IREE_RETURN_IF_ERROR(status);
  if (qwen_replay_interrupted) {
    return iree_make_status(IREE_STATUS_CANCELLED,
                            "replay interrupted at a completed epoch");
  }
  printf(
      "{\"event\":\"window\",\"window\":%zu,\"epochs\":%zu,"
      "\"prefill_tokens\":%" PRIu64 ",\"decode_tokens\":%" PRIu64
      ",\"selected_tokens\":%" PRIu64 ",\"model_ns\":%" PRId64
      ",\"window_ns\":%" PRId64 ",\"prediction_differences\":%zu}\n",
      window, epoch, prefill_tokens, decode_tokens, outputs, model_duration,
      iree_time_now() - start, differences);
  if (fflush(stdout) != 0 || ferror(stdout)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "writing replay summary");
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_host_size_t epoch_count =
      loom_serve_text_explicit_shape_count_from_flags();
  const iree_flag_string_list_t windows = FLAG_window_list();
  if (!FLAG_workload[0] || !epoch_count || epoch_count > QWEN_REPLAY_SHAPES ||
      !windows.count) {
    fprintf(stderr, "Invalid replay flags; see --help.\n");
    return EXIT_FAILURE;
  }
  if (signal(SIGINT, qwen_replay_interrupt) == SIG_ERR ||
      signal(SIGTERM, qwen_replay_interrupt) == SIG_ERR) {
    perror("installing replay signal handler");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_workload), allocator, &contents);
  iree_string_view_t sessions = iree_string_view_empty();
  iree_host_size_t row_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_json_lookup_object_value(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        IREE_SV("sessions"), &sessions);
  }
  if (iree_status_is_ok(status)) {
    status = iree_json_array_length(sessions, &row_count);
  }
  if (iree_status_is_ok(status) &&
      (!row_count || row_count > QWEN_REPLAY_ROWS)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "replay requires one through sixteen sessions");
  }
  const loom_serve_text_flag_defaults_t defaults = {.row_count = row_count};
  loom_serve_text_model_t* model = NULL;
  loom_serve_device_t* device = NULL;
  qwen_replay_row_t rows[QWEN_REPLAY_ROWS] = {0};
  qwen_replay_window_t* policies = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_device_create_from_flags(&device, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_model_create_from_flags(device, &defaults, &model,
                                                     allocator);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_replay_prepare(model, sessions, row_count, rows, allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, windows.count * sizeof(*policies),
                                   (void**)&policies);
  }
  for (iree_host_size_t i = 0; i < windows.count && iree_status_is_ok(status);
       ++i) {
    status = qwen_replay_window_parse(model, windows.values[i], &policies[i]);
  }
  for (iree_host_size_t i = 0; i < windows.count && iree_status_is_ok(status);
       ++i) {
    status = qwen_replay_run(model, &policies[i], row_count, rows, i);
  }
  status = iree_status_join(status, loom_serve_text_model_destroy(model));
  status = iree_status_join(status, loom_serve_device_destroy(device));
  iree_allocator_free(allocator, policies);
  for (iree_host_size_t i = 0; i < QWEN_REPLAY_ROWS; ++i) {
    iree_allocator_free(allocator, rows[i].turns);
    iree_allocator_free(allocator, rows[i].tokens);
    iree_allocator_free(allocator, rows[i].reference);
  }
  iree_io_file_contents_free(contents);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
