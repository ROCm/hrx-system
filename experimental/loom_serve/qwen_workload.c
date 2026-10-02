// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded retained-workload replay through the public model interface.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/qwen_flags.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG_LIST(string, prompt_file,
               "Rendered chat prompt; one file per independent resident row.");
IREE_FLAG(int32_t, retained_tokens, 1024,
          "Prefix consumed before timing for rows starting in prefill.");
IREE_FLAG(int32_t, prefill_rows, 2,
          "First N rows start in suffix prefill; others start in decode.");
IREE_FLAG(int32_t, max_tokens, 64,
          "Maximum selected tokens per row inside each measured window.");
IREE_FLAG(
    string, order, "ABABA",
    "Window order: A is isolated, B is packed. Prefixes replay each time.");
IREE_FLAG(string, baseline, "practical",
          "practical uses ordinary decode; matched uses the prefill math for "
          "length-one inputs and requires identical outputs.");

enum { QWEN_WORKLOAD_ROWS = 8 };

typedef struct qwen_workload_row_t {
  // Borrowed device residency, shared across every replay window.
  loom_serve_qwen_row_t* device_row;
  // Owned encoded prompt storage, allocated to context capacity.
  int32_t* input;
  // Actual encoded prompt count.
  iree_host_size_t input_count;
  // Identical prefix replayed outside each measured window.
  iree_host_size_t prefix_count;
  // Prompt tokens consumed in the current window, including its prefix.
  iree_host_size_t consumed_count;
  // Owned selected-token storage for this window.
  int32_t* output;
  // Selected tokens produced inside this window, including terminal EOS.
  iree_host_size_t output_count;
  // Owned copy of the first window's selected tokens.
  int32_t* reference;
  // First-window selected count used to compare work and greedy trajectories.
  iree_host_size_t reference_count;
  // Prediction made by prefix setup for a decode row; -1 for prefill rows.
  int32_t initial_prediction;
  // EOS or the requested output cap has completed this row.
  bool finished;
} qwen_workload_row_t;

static iree_status_t qwen_workload_prepare(loom_serve_qwen_model_t* model,
                                           iree_host_size_t row_count,
                                           qwen_workload_row_t* rows,
                                           iree_allocator_t allocator) {
  const iree_flag_string_list_t paths = FLAG_prompt_file_list();
  const iree_host_size_t capacity =
      loom_serve_qwen_model_context_capacity(model);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
       ++i) {
    qwen_workload_row_t* row = &rows[i];
    row->device_row = loom_serve_qwen_model_row(model, i);
    status = iree_allocator_malloc(allocator, capacity * sizeof(int32_t),
                                   (void**)&row->input);
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(
          allocator, FLAG_max_tokens * sizeof(int32_t), (void**)&row->output);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(allocator, FLAG_max_tokens * sizeof(int32_t),
                                (void**)&row->reference);
    }
    iree_io_file_contents_t* contents = NULL;
    if (iree_status_is_ok(status)) {
      status =
          iree_io_file_contents_read(paths.values[i], allocator, &contents);
    }
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_encode(
          loom_serve_qwen_model_tokenizer(model),
          iree_make_string_view((const char*)contents->const_buffer.data,
                                contents->const_buffer.data_length),
          IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(row->input, NULL, NULL, capacity),
          allocator, &row->input_count);
    }
    iree_io_file_contents_free(contents);
    if (iree_status_is_ok(status)) {
      row->prefix_count = i < (iree_host_size_t)FLAG_prefill_rows
                              ? (iree_host_size_t)FLAG_retained_tokens
                              : row->input_count;
      if (!row->input_count || row->prefix_count > row->input_count ||
          (i < (iree_host_size_t)FLAG_prefill_rows &&
           row->prefix_count == row->input_count) ||
          (iree_host_size_t)FLAG_max_tokens > capacity - row->input_count) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "row %zu prompt/prefix/output do not fit", i);
      }
    }
  }
  return status;
}

static iree_status_t qwen_workload_seed(loom_serve_qwen_model_t* model,
                                        iree_host_size_t row_count,
                                        qwen_workload_row_t* rows) {
  const iree_host_size_t capacity =
      loom_serve_qwen_model_prefill_capacity(model);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
       ++i) {
    qwen_workload_row_t* row = &rows[i];
    status = loom_serve_qwen_row_reset(row->device_row);
    row->consumed_count = 0;
    row->output_count = 0;
    row->finished = false;
    row->initial_prediction = -1;
    while (row->consumed_count < row->prefix_count &&
           iree_status_is_ok(status)) {
      const iree_host_size_t count =
          iree_min(capacity, row->prefix_count - row->consumed_count);
      status = loom_serve_qwen_row_prefill(row->device_row, count,
                                           row->input + row->consumed_count);
      row->consumed_count += count;
    }
    if (iree_status_is_ok(status) && row->prefix_count == row->input_count) {
      row->initial_prediction = loom_serve_qwen_row_token(row->device_row);
      row->finished = loom_serve_qwen_row_is_eos(row->device_row);
    }
  }
  return status;
}

// This is workload generation, not service policy: ready decode rows reserve
// one token each, and ready prefill rows split the remaining capacity. Both
// execution arms receive the same partition at each identical logical state.
static iree_host_size_t qwen_workload_plan(
    loom_serve_qwen_model_t* model, iree_host_size_t row_count,
    qwen_workload_row_t* rows, loom_serve_qwen_span_t spans[QWEN_WORKLOAD_ROWS],
    bool prefills[QWEN_WORKLOAD_ROWS],
    iree_host_size_t positions[QWEN_WORKLOAD_ROWS],
    int32_t decode_tokens[QWEN_WORKLOAD_ROWS]) {
  iree_host_size_t span_count = 0;
  iree_host_size_t prefill_count = 0;
  iree_host_size_t remaining =
      loom_serve_qwen_model_shapes(model)[0].token_capacity;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    qwen_workload_row_t* row = &rows[i];
    if (row->finished) {
      continue;
    }
    if (row->consumed_count < row->input_count) {
      ++prefill_count;
    } else {
      decode_tokens[i] = loom_serve_qwen_row_token(row->device_row);
      positions[span_count] = loom_serve_qwen_row_position(row->device_row);
      prefills[span_count] = false;
      spans[span_count++] = (loom_serve_qwen_span_t){
          i, 1, &decode_tokens[i], LOOM_SERVE_QWEN_SPAN_FLAG_SELECT};
      --remaining;
    }
  }
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    qwen_workload_row_t* row = &rows[i];
    if (row->finished || row->consumed_count == row->input_count) {
      continue;
    }
    const iree_host_size_t count = iree_min(
        row->input_count - row->consumed_count, remaining / prefill_count);
    positions[span_count] = loom_serve_qwen_row_position(row->device_row);
    prefills[span_count] = true;
    spans[span_count++] =
        (loom_serve_qwen_span_t){i, count, row->input + row->consumed_count,
                                 row->consumed_count + count == row->input_count
                                     ? LOOM_SERVE_QWEN_SPAN_FLAG_SELECT
                                     : 0};
    remaining -= count;
    --prefill_count;
  }
  return span_count;
}

static iree_status_t qwen_workload_window(loom_serve_qwen_model_t* model,
                                          iree_host_size_t row_count,
                                          qwen_workload_row_t* rows,
                                          iree_host_size_t window) {
  const char arm = FLAG_order[window];
  uint64_t prefill_tokens = 0;
  uint64_t decode_tokens = 0;
  uint64_t selected_tokens = 0;
  iree_duration_t execution_duration = 0;
  iree_host_size_t epochs = 0;
  iree_status_t status = iree_ok_status();
  const iree_time_t start = iree_time_now();
  while (iree_status_is_ok(status)) {
    loom_serve_qwen_span_t spans[QWEN_WORKLOAD_ROWS];
    bool prefills[QWEN_WORKLOAD_ROWS];
    iree_host_size_t positions[QWEN_WORKLOAD_ROWS];
    int32_t inputs[QWEN_WORKLOAD_ROWS];
    const iree_host_size_t span_count = qwen_workload_plan(
        model, row_count, rows, spans, prefills, positions, inputs);
    if (!span_count) {
      break;
    }
    const iree_time_t issue_start = iree_time_now();
    if (arm == 'B') {
      status = loom_serve_qwen_model_epoch(model, 0, span_count, spans);
    } else {
      for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
           ++i) {
        const loom_serve_qwen_span_t* span = &spans[i];
        loom_serve_qwen_row_t* row = rows[span->row_index].device_row;
        status = prefills[i] || strcmp(FLAG_baseline, "matched") == 0
                     ? loom_serve_qwen_row_prefill(row, span->token_count,
                                                   span->token_ids)
                     : loom_serve_qwen_row_decode(row);
      }
    }
    const iree_duration_t duration = iree_time_now() - issue_start;
    if (!iree_status_is_ok(status)) {
      break;
    }
    execution_duration += duration;
    printf(
        "{\"event\":\"epoch\",\"window\":%zu,\"arm\":\"%c\","
        "\"epoch\":%zu,\"completed_ns\":%" PRId64 ",\"work\":[",
        window, arm, epochs++, duration);
    for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
         ++i) {
      const loom_serve_qwen_span_t* span = &spans[i];
      qwen_workload_row_t* row = &rows[span->row_index];
      if (prefills[i]) {
        row->consumed_count += span->token_count;
        prefill_tokens += span->token_count;
      } else {
        decode_tokens += span->token_count;
      }
      const bool selected =
          iree_any_bit_set(span->flags, LOOM_SERVE_QWEN_SPAN_FLAG_SELECT);
      if (selected) {
        row->output[row->output_count++] =
            loom_serve_qwen_row_token(row->device_row);
        row->finished = loom_serve_qwen_row_is_eos(row->device_row) ||
                        row->output_count == (iree_host_size_t)FLAG_max_tokens;
        ++selected_tokens;
      }
      printf(
          "%s{\"row\":%zu,\"position\":%zu,\"prefill_tokens\":%zu,"
          "\"decode_tokens\":%zu,\"selected\":%d}",
          i ? "," : "", span->row_index, positions[i],
          prefills[i] ? span->token_count : 0,
          prefills[i] ? 0 : span->token_count, selected ? 1 : 0);
      if (loom_serve_qwen_row_position(row->device_row) !=
          positions[i] + span->token_count) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "row %zu did not consume its span",
                                  span->row_index);
      }
    }
    printf("]}\n");
    if (fflush(stdout) != 0 || ferror(stdout)) {
      status = iree_status_join(
          status, iree_make_status(IREE_STATUS_DATA_LOSS,
                                   "writing workload epoch measurements"));
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_duration_t duration = iree_time_now() - start;
  bool equal_counts = true;
  bool equal_outputs = true;
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    qwen_workload_row_t* row = &rows[i];
    if (!window) {
      memcpy(row->reference, row->output, row->output_count * sizeof(int32_t));
      row->reference_count = row->output_count;
    } else {
      equal_counts &= row->reference_count == row->output_count;
      equal_outputs &= row->reference_count == row->output_count &&
                       memcmp(row->reference, row->output,
                              row->output_count * sizeof(int32_t)) == 0;
    }
  }
  printf(
      "{\"event\":\"window\",\"window\":%zu,\"arm\":\"%c\","
      "\"epochs\":%zu,\"prefill_tokens\":%" PRIu64
      ","
      "\"decode_tokens\":%" PRIu64 ",\"selected_tokens\":%" PRIu64
      ","
      "\"execution_ns\":%" PRId64 ",\"window_ns\":%" PRId64
      ","
      "\"equal_output_counts\":%s,\"equal_outputs\":%s,\"rows\":[",
      window, arm, epochs, prefill_tokens, decode_tokens, selected_tokens,
      execution_duration, duration, equal_counts ? "true" : "false",
      equal_outputs ? "true" : "false");
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    const qwen_workload_row_t* row = &rows[i];
    printf(
        "%s{\"row\":%zu,\"consumed_position\":%zu,\"eos\":%s,"
        "\"initial_prediction\":%d,\"selected_ids\":[",
        i ? "," : "", i, loom_serve_qwen_row_position(row->device_row),
        loom_serve_qwen_row_is_eos(row->device_row) ? "true" : "false",
        row->initial_prediction);
    for (iree_host_size_t j = 0; j < row->output_count; ++j) {
      printf("%s%d", j ? "," : "", row->output[j]);
    }
    printf("]}");
  }
  printf("]}\n");
  if (fflush(stdout) != 0 || ferror(stdout)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "writing workload summary");
  }
  if (!equal_outputs && strcmp(FLAG_baseline, "matched") == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "matched-math workload changed selected tokens");
  }
  return iree_ok_status();
}

static iree_status_t qwen_workload_print(loom_serve_qwen_model_t* model,
                                         iree_host_size_t row_count,
                                         const qwen_workload_row_t* rows,
                                         iree_host_size_t window,
                                         iree_allocator_t allocator) {
  // Text is diagnostic, outside timing; JSONL measurements remain on stdout.
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    int32_t tokens[1025];
    iree_host_size_t token_count = 0;
    if (rows[i].initial_prediction >= 0) {
      tokens[token_count++] = rows[i].initial_prediction;
    }
    memcpy(tokens + token_count, rows[i].output,
           rows[i].output_count * sizeof(int32_t));
    token_count += rows[i].output_count;
    char text[65536];
    iree_host_size_t length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_decode(
        loom_serve_qwen_model_tokenizer(model),
        iree_tokenizer_make_token_id_list(tokens, token_count),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(text, sizeof(text)), allocator, &length));
    fprintf(stderr,
            "window %zu row %zu (%zu timed tokens, %d seed prediction): %.*s\n",
            window, i, rows[i].output_count,
            rows[i].initial_prediction >= 0 ? 1 : 0, (int)length, text);
  }
  if (ferror(stderr)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "writing generated text");
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_host_size_t row_count = FLAG_prompt_file_list().count;
  if (loom_serve_qwen_shape_count_from_flags() != 1 || row_count < 1 ||
      row_count > QWEN_WORKLOAD_ROWS || FLAG_retained_tokens < 0 ||
      FLAG_prefill_rows < 0 ||
      (iree_host_size_t)FLAG_prefill_rows > row_count || FLAG_max_tokens < 1 ||
      FLAG_max_tokens > 1024 || !FLAG_order[0] ||
      strspn(FLAG_order, "AB") != strlen(FLAG_order) ||
      (strcmp(FLAG_baseline, "practical") != 0 &&
       strcmp(FLAG_baseline, "matched") != 0)) {
    fprintf(stderr, "Invalid workload flags; see --help.\n");
    return EXIT_FAILURE;
  }
  const iree_allocator_t allocator = iree_allocator_system();
  loom_serve_qwen_model_t* model = NULL;
  qwen_workload_row_t rows[QWEN_WORKLOAD_ROWS] = {0};
  iree_status_t status =
      loom_serve_qwen_model_create_from_flags(row_count, allocator, &model);
  if (iree_status_is_ok(status) &&
      (loom_serve_qwen_model_shapes(model)[0].span_capacity < row_count ||
       loom_serve_qwen_model_shapes(model)[0].token_capacity < row_count ||
       loom_serve_qwen_model_shapes(model)[0].token_capacity !=
           loom_serve_qwen_model_prefill_capacity(model))) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "workload requires covering span capacity and equal token capacities");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_workload_prepare(model, row_count, rows, allocator);
  }
  for (iree_host_size_t window = 0;
       window < strlen(FLAG_order) && iree_status_is_ok(status); ++window) {
    const iree_time_t start = iree_time_now();
    status = qwen_workload_seed(model, row_count, rows);
    if (iree_status_is_ok(status)) {
      printf("{\"event\":\"seed\",\"window\":%zu,\"completed_ns\":%" PRId64
             ",\"token_capacity\":%zu,\"rows\":[",
             window, iree_time_now() - start,
             loom_serve_qwen_model_shapes(model)[0].token_capacity);
      for (iree_host_size_t i = 0; i < row_count; ++i) {
        printf("%s{\"row\":%zu,\"prompt_tokens\":%zu,\"retained_tokens\":%zu}",
               i ? "," : "", i, rows[i].input_count, rows[i].prefix_count);
      }
      printf("]}\n");
      if (fflush(stdout) != 0 || ferror(stdout)) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "writing workload seed measurements");
      }
    }
    if (iree_status_is_ok(status)) {
      status = qwen_workload_window(model, row_count, rows, window);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_workload_print(model, row_count, rows, window, allocator);
    }
  }
  status = iree_status_join(status, loom_serve_qwen_model_destroy(model));
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    iree_allocator_free(allocator, rows[i].input);
    iree_allocator_free(allocator, rows[i].output);
    iree_allocator_free(allocator, rows[i].reference);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
