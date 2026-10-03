// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete multi-row Qwen CLI and completed-stage timing witness.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/qwen_flags.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"

IREE_FLAG_LIST(string, prompt, "User prompt; repeat for independent rows.");
IREE_FLAG(string, prompt_file, "",
          "Already rendered chat text, instead of prompt.");
IREE_FLAG(string, followup, "",
          "Second user turn appended to each retained row.");
IREE_FLAG(int32_t, rows, 0,
          "Retained rows (1-16); zero uses the prompt count.");
IREE_FLAG(int32_t, max_tokens, 128,
          "Maximum selected tokens per turn, including EOS.");
IREE_FLAG(int32_t, chunk_size, 0,
          "Prefill chunk size; zero uses compiled capacity.");
IREE_FLAG(int32_t, iterations, 1,
          "Independent repetitions within one loaded model.");

typedef struct qwen_cli_row_t {
  // Borrowed retained device row.
  loom_serve_qwen_row_t* row;
  // Host input storage sized to the model context.
  int32_t* input;
  // Host generated-token storage sized to the model context.
  int32_t* output;
  // Number of encoded input tokens in this turn.
  iree_host_size_t input_count;
  // Number of input tokens already consumed in this turn.
  iree_host_size_t input_offset;
  // Number of selected output tokens, including EOS.
  iree_host_size_t output_count;
  // Admission-to-first-prediction duration for this turn.
  iree_duration_t first_token_duration;
  // Stage counters at the beginning of this turn.
  loom_serve_qwen_metrics_t initial_metrics;
} qwen_cli_row_t;

static iree_status_t qwen_print(iree_tokenizer_t* tokenizer,
                                iree_host_size_t count, const int32_t* tokens,
                                iree_allocator_t allocator) {
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
  void* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, state_size, &storage));
  iree_tokenizer_decode_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_decode_state_initialize(
      tokenizer, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
      iree_make_byte_span(storage, state_size), &state);
  iree_host_size_t offset = 0;
  char text[8192];
  while (offset < count && iree_status_is_ok(status)) {
    iree_host_size_t consumed = 0;
    iree_host_size_t length = 0;
    status = iree_tokenizer_decode_state_feed(
        state,
        iree_tokenizer_make_token_id_list(tokens + offset, count - offset),
        iree_make_mutable_string_view(text, sizeof(text)), &consumed, &length);
    offset += consumed;
    if (iree_status_is_ok(status) && !consumed && !length) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "token text exceeds the CLI output chunk");
    }
    if (iree_status_is_ok(status) &&
        fwrite(text, 1, length, stdout) != length) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS, "writing generated text");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t length = 0;
    status = iree_tokenizer_decode_state_finalize(
        state, iree_make_mutable_string_view(text, sizeof(text)), &length);
    if (iree_status_is_ok(status) &&
        (fwrite(text, 1, length, stdout) != length || fflush(stdout) != 0)) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS, "flushing generated text");
    }
  }
  if (state) {
    iree_tokenizer_decode_state_deinitialize(state);
  }
  iree_allocator_free(allocator, storage);
  return status;
}

static iree_status_t qwen_prepare_turn(loom_serve_qwen_model_t* model,
                                       iree_host_size_t row_count,
                                       qwen_cli_row_t* rows, int turn,
                                       iree_allocator_t allocator) {
  const iree_host_size_t capacity =
      loom_serve_qwen_model_context_capacity(model);
  iree_tokenizer_t* tokenizer = loom_serve_qwen_model_tokenizer(model);
  const iree_flag_string_list_t prompts = FLAG_prompt_list();
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_ok_status();
  if (!turn && FLAG_prompt_file[0]) {
    status = iree_io_file_contents_read(
        iree_make_cstring_view(FLAG_prompt_file), allocator, &contents);
  }
  for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
       ++i) {
    qwen_cli_row_t* row = &rows[i];
    iree_string_builder_t prompt;
    iree_string_builder_initialize(allocator, &prompt);
    if (turn) {
      // The last selected token was returned to the user, not consumed. Append
      // it before the role delimiter so retained state matches the transcript.
      row->input[0] = loom_serve_qwen_row_token(row->row);
      status = iree_string_builder_append_format(
          &prompt,
          "%s\n<|im_start|>user\n%s<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n",
          loom_serve_qwen_row_is_eos(row->row) ? "" : "<|im_end|>",
          FLAG_followup);
    } else if (contents) {
      status = iree_string_builder_append_string(
          &prompt,
          iree_make_string_view((const char*)contents->const_buffer.data,
                                contents->const_buffer.data_length));
    } else {
      const iree_string_view_t text =
          prompts.count ? prompts.values[i % prompts.count]
                        : IREE_SV("What is 2+2? Answer with one number.");
      status = iree_string_builder_append_format(
          &prompt,
          "<|im_start|>user\n%.*s<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n",
          (int)text.size, text.data);
    }
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_encode(
          tokenizer, iree_string_builder_view(&prompt),
          IREE_TOKENIZER_ENCODE_FLAG_NONE,
          iree_tokenizer_make_token_output(row->input + turn, NULL, NULL,
                                           capacity - turn),
          allocator, &row->input_count);
      row->input_count += turn;
    }
    iree_string_builder_deinitialize(&prompt);
    if (iree_status_is_ok(status) &&
        row->input_count + (iree_host_size_t)FLAG_max_tokens - 1 >
            capacity - loom_serve_qwen_row_position(row->row)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "row %zu input and output exceed remaining context", i);
    }
    row->input_offset = 0;
    row->output_count = 0;
    row->initial_metrics = loom_serve_qwen_row_metrics(row->row);
  }
  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t qwen_turn(iree_host_size_t row_count, qwen_cli_row_t* rows,
                               iree_host_size_t chunk_size) {
  const iree_time_t start = iree_time_now();
  iree_status_t status = iree_ok_status();
  bool active = true;
  while (active && iree_status_is_ok(status)) {
    active = false;
    for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
         ++i) {
      qwen_cli_row_t* row = &rows[i];
      if (row->input_offset < row->input_count) {
        const iree_host_size_t count =
            iree_min(chunk_size, row->input_count - row->input_offset);
        status = loom_serve_qwen_row_prefill(row->row, count,
                                             row->input + row->input_offset);
        if (iree_status_is_ok(status)) {
          row->input_offset += count;
          if (row->input_offset == row->input_count) {
            row->output[row->output_count++] =
                loom_serve_qwen_row_token(row->row);
            row->first_token_duration = iree_time_now() - start;
          }
        }
        active = true;
      } else if (!loom_serve_qwen_row_is_eos(row->row) &&
                 row->output_count < (iree_host_size_t)FLAG_max_tokens) {
        status = loom_serve_qwen_row_decode(row->row);
        if (iree_status_is_ok(status)) {
          row->output[row->output_count++] =
              loom_serve_qwen_row_token(row->row);
        }
        active = true;
      }
    }
  }
  return status;
}

static iree_status_t qwen_run(loom_serve_qwen_model_t* model,
                              iree_host_size_t row_count,
                              iree_allocator_t allocator) {
  const iree_host_size_t capacity =
      loom_serve_qwen_model_context_capacity(model);
  const iree_host_size_t prefill_capacity =
      loom_serve_qwen_model_prefill_capacity(model);
  const iree_host_size_t chunk_size =
      FLAG_chunk_size ? (iree_host_size_t)FLAG_chunk_size : prefill_capacity;
  if (chunk_size > prefill_capacity ||
      (iree_host_size_t)FLAG_max_tokens > capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "requested lengths exceed the compiled model capacity");
  }
  int32_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      allocator, 2 * capacity * row_count, sizeof(int32_t), (void**)&storage));
  qwen_cli_row_t rows[LOOM_SERVE_QWEN_ROW_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < row_count; ++i) {
    rows[i].row = loom_serve_qwen_model_row(model, i);
    rows[i].input = storage + i * capacity * 2;
    rows[i].output = rows[i].input + capacity;
  }
  iree_status_t status = iree_ok_status();
  for (int iteration = 0;
       iteration < FLAG_iterations && iree_status_is_ok(status); ++iteration) {
    for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
         ++i) {
      status = loom_serve_qwen_row_reset(rows[i].row);
    }
    const int turn_count = FLAG_followup[0] ? 2 : 1;
    for (int turn = 0; turn < turn_count && iree_status_is_ok(status); ++turn) {
      status = qwen_prepare_turn(model, row_count, rows, turn, allocator);
      const iree_time_t start = iree_time_now();
      if (iree_status_is_ok(status)) {
        status = qwen_turn(row_count, rows, chunk_size);
      }
      const iree_duration_t elapsed = iree_time_now() - start;
      for (iree_host_size_t i = 0; i < row_count && iree_status_is_ok(status);
           ++i) {
        const qwen_cli_row_t* row = &rows[i];
        const loom_serve_qwen_metrics_t metrics =
            loom_serve_qwen_row_metrics(row->row);
        fprintf(
            stderr,
            "{\"iteration\":%d,\"turn\":%d,\"row\":%zu,\"input_tokens\":%zu,"
            "\"output_tokens_including_eos\":%zu,\"consumed_position\":%zu,"
            "\"prefill_steps\":%" PRIu64
            ",\"prefill_ms\":%.3f,"
            "\"decode_steps\":%" PRIu64
            ",\"decode_ms\":%.3f,"
            "\"model_ttft_ms\":%.3f,\"round_robin_ms\":%.3f}\n",
            iteration, turn, i, row->input_count, row->output_count,
            loom_serve_qwen_row_position(row->row),
            metrics.prefill_steps - row->initial_metrics.prefill_steps,
            (metrics.prefill_duration - row->initial_metrics.prefill_duration) /
                1e6,
            metrics.decode_steps - row->initial_metrics.decode_steps,
            (metrics.decode_duration - row->initial_metrics.decode_duration) /
                1e6,
            row->first_token_duration / 1e6, elapsed / 1e6);
        if (row_count > 1 || turn_count > 1 || FLAG_iterations > 1) {
          printf("\n[iteration %d turn %d row %zu]\n", iteration, turn, i);
        }
        status = qwen_print(loom_serve_qwen_model_tokenizer(model),
                            row->output_count, row->output, allocator);
      }
    }
  }
  iree_allocator_free(allocator, storage);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_flag_string_list_t prompts = FLAG_prompt_list();
  const iree_host_size_t row_count =
      FLAG_rows ? (iree_host_size_t)FLAG_rows : iree_max(prompts.count, 1);
  if (FLAG_rows < 0 || row_count > LOOM_SERVE_QWEN_ROW_CAPACITY ||
      FLAG_max_tokens < 1 || FLAG_iterations < 1 || FLAG_chunk_size < 0 ||
      (FLAG_prompt_file[0] && prompts.count)) {
    fprintf(stderr,
            "Provide model paths and positive lengths; use 1-16 rows "
            "and either prompts or one rendered prompt file.\n");
    return EXIT_FAILURE;
  }
  iree_allocator_t allocator = iree_allocator_system();
  const loom_serve_qwen_flag_defaults_t defaults = {.row_count = row_count};
  loom_serve_qwen_model_t* model = NULL;
  iree_status_t status =
      loom_serve_qwen_model_create_from_flags(&defaults, &model, allocator);
  if (iree_status_is_ok(status)) {
    status = qwen_run(model, row_count, allocator);
  }
  status = iree_status_join(status, loom_serve_qwen_model_destroy(model));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
