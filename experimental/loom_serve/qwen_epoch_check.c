// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-weight differential through the runner's public model interface.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/qwen_model.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, prefill, "",
          "Isolated prefill artifacts, using the same schedule.");
IREE_FLAG(string, decode, "", "Isolated decode artifact directory.");
IREE_FLAG_LIST(
    string, epoch,
    "Packed epoch directory; repeat to cycle through cached shapes.");
IREE_FLAG(string, weights, "", "Canonical Qwen3.8-27B UD-Q5_K_XL GGUF.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json.");
IREE_FLAG(string, mtp, "",
          "Optional MTP bundle; checks private proposal isolation.");
IREE_FLAG(string, compare, "",
          "Optional completed-work ABABA comparison: single, mixed, full or "
          "decode. Empty runs the correctness witness.");

typedef struct qwen_check_row_t {
  // Nonzero, permuted resident row used by packed invocations.
  iree_host_size_t packed;
  // Disjoint resident row used by isolated invocations of the same model.
  iree_host_size_t isolated;
  // Rendered prompt IDs; all fixture storage is bounded and host-owned.
  int32_t input[512];
  // Active prompt ID count.
  iree_host_size_t input_count;
  // Prefix consumed before the first mixed epoch.
  iree_host_size_t prefix_count;
  // Selected continuation IDs, retained for human-readable output.
  int32_t output[16];
  // Number of selected continuation IDs.
  iree_host_size_t output_count;
} qwen_check_row_t;

static iree_status_t qwen_check_prefill(loom_serve_qwen_model_t* model,
                                        iree_host_size_t row_index,
                                        iree_host_size_t count,
                                        const int32_t* tokens) {
  loom_serve_qwen_row_t* row = loom_serve_qwen_model_row(model, row_index);
  const iree_host_size_t capacity =
      loom_serve_qwen_model_prefill_capacity(model);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t offset = 0;
       offset < count && iree_status_is_ok(status);) {
    const iree_host_size_t length = iree_min(capacity, count - offset);
    status = loom_serve_qwen_row_prefill(row, length, tokens + offset);
    offset += length;
  }
  return status;
}

static iree_status_t qwen_check_prediction(loom_serve_qwen_model_t* model,
                                           qwen_check_row_t* row) {
  const int32_t packed =
      loom_serve_qwen_row_token(loom_serve_qwen_model_row(model, row->packed));
  const int32_t isolated = loom_serve_qwen_row_token(
      loom_serve_qwen_model_row(model, row->isolated));
  if (packed != isolated) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "row %zu prediction %d differs from isolated %d",
                            row->packed, packed, isolated);
  }
  row->output[row->output_count++] = packed;
  return iree_ok_status();
}

static iree_status_t qwen_check_epoch(loom_serve_qwen_model_t* model,
                                      iree_host_size_t* next_shape,
                                      qwen_check_row_t rows[4],
                                      iree_host_size_t count,
                                      const iree_host_size_t logical_rows[4],
                                      const loom_serve_qwen_span_t* spans) {
  const iree_host_size_t shape_index =
      (*next_shape)++ % loom_serve_qwen_model_shape_count(model);
  IREE_RETURN_IF_ERROR(
      loom_serve_qwen_model_epoch(model, shape_index, count, spans));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    qwen_check_row_t* row = &rows[logical_rows[i]];
    // Length-one decode input uses the same dense schedule here. Comparing to
    // the separate GEMV decode family would mix numerical changes with routing.
    status = qwen_check_prefill(model, row->isolated, spans[i].token_count,
                                spans[i].token_ids);
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(spans[i].flags, LOOM_SERVE_QWEN_SPAN_FLAG_SELECT)) {
      status = qwen_check_prediction(model, row);
    }
  }
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    const iree_host_size_t packed = loom_serve_qwen_row_position(
        loom_serve_qwen_model_row(model, rows[i].packed));
    const iree_host_size_t isolated = loom_serve_qwen_row_position(
        loom_serve_qwen_model_row(model, rows[i].isolated));
    if (packed != isolated) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "row %zu position %zu differs from isolated %zu",
                           rows[i].packed, packed, isolated);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(stderr,
            "Matched packed epoch: shape %zu, %zu spans; positions "
            "[%zu,%zu,%zu,%zu].\n",
            shape_index, count,
            loom_serve_qwen_row_position(
                loom_serve_qwen_model_row(model, rows[0].packed)),
            loom_serve_qwen_row_position(
                loom_serve_qwen_model_row(model, rows[1].packed)),
            loom_serve_qwen_row_position(
                loom_serve_qwen_model_row(model, rows[2].packed)),
            loom_serve_qwen_row_position(
                loom_serve_qwen_model_row(model, rows[3].packed)));
  }
  return status;
}

static iree_status_t qwen_check_proposals(loom_serve_qwen_model_t* model,
                                          qwen_check_row_t rows[4]) {
  if (!FLAG_mtp[0]) {
    return iree_ok_status();
  }
  iree_host_size_t indices[3][4];
  const iree_host_size_t permutation[4] = {2, 0, 3, 1};
  iree_host_size_t positions[8];
  int32_t predictions[8];
  for (iree_host_size_t i = 0; i < 8; ++i) {
    const loom_serve_qwen_row_t* row = loom_serve_qwen_model_row(model, i);
    positions[i] = loom_serve_qwen_row_position(row);
    predictions[i] = loom_serve_qwen_row_token(row);
  }
  for (iree_host_size_t i = 0; i < 4; ++i) {
    indices[0][i] = rows[i].packed;
    indices[1][i] = rows[i].isolated;
    indices[2][i] = rows[permutation[i]].packed;
  }
  int32_t proposals[3][4][3] = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t issue = 0; issue < 3 && iree_status_is_ok(status);
       ++issue) {
    status = loom_serve_qwen_model_propose(model, 4, indices[issue],
                                           proposals[issue]);
    for (iree_host_size_t i = 0; i < 8 && iree_status_is_ok(status); ++i) {
      const loom_serve_qwen_row_t* row = loom_serve_qwen_model_row(model, i);
      if (loom_serve_qwen_row_position(row) != positions[i] ||
          loom_serve_qwen_row_token(row) != predictions[i]) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "proposal changed committed target row %zu", i);
      }
    }
  }
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    if (memcmp(proposals[0][i], proposals[1][i], sizeof(proposals[0][i])) ||
        memcmp(proposals[2][i], proposals[0][permutation[i]],
               sizeof(proposals[0][i]))) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "MTP proposal depends on resident or compact row placement");
    }
    if (iree_status_is_ok(status)) {
      fprintf(
          stderr,
          "Matched MTP row %zu: [%d,%d,%d], target position %zu unchanged.\n",
          rows[i].packed, proposals[0][i][0], proposals[0][i][1],
          proposals[0][i][2], positions[rows[i].packed]);
    }
  }
  return status;
}

static iree_status_t qwen_check_error(iree_status_t actual,
                                      iree_status_code_t expected) {
  if (iree_status_code(actual) == expected) {
    iree_status_free(actual);
    return iree_ok_status();
  }
  return iree_status_join(
      iree_make_status(IREE_STATUS_DATA_LOSS, "expected rejection code %d",
                       expected),
      actual);
}

static iree_status_t qwen_check_run(loom_serve_qwen_model_t* model,
                                    iree_allocator_t allocator) {
  for (iree_host_size_t i = 0; i < loom_serve_qwen_model_shape_count(model);
       ++i) {
    const loom_serve_qwen_shape_t shape =
        loom_serve_qwen_model_shapes(model)[i];
    if (shape.token_capacity < 10 || shape.span_capacity < 4) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "witness requires at least 10 tokens and four spans in every shape");
    }
  }
  iree_host_size_t next_shape = 0;
  qwen_check_row_t rows[4] = {
      {.packed = 6, .isolated = 0},
      {.packed = 1, .isolated = 2},
      {.packed = 7, .isolated = 4},
      {.packed = 3, .isolated = 5},
  };
  const char* phrases[] = {
      "amber cedar maple raven",
      "violet birch willow falcon",
      "silver pine oak robin",
      "golden elm ash eagle",
  };
  const iree_host_size_t withheld[] = {5, 4, 0, 0};
  iree_tokenizer_t* tokenizer = loom_serve_qwen_model_tokenizer(model);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    char prompt[512];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>user\nReturn exactly: %s<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
             phrases[i]);
    status = iree_tokenizer_encode(
        tokenizer, iree_make_cstring_view(prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(rows[i].input, NULL, NULL, 512),
        allocator, &rows[i].input_count);
    if (iree_status_is_ok(status) && rows[i].input_count <= withheld[i]) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "fixture prompt is too short");
    }
    if (iree_status_is_ok(status)) {
      rows[i].prefix_count = rows[i].input_count - withheld[i];
      status = qwen_check_prefill(model, rows[i].packed, rows[i].prefix_count,
                                  rows[i].input);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_check_prefill(model, rows[i].isolated, rows[i].prefix_count,
                                  rows[i].input);
    }
    if (iree_status_is_ok(status) && !withheld[i]) {
      status = qwen_check_prediction(model, &rows[i]);
    }
  }
  IREE_RETURN_IF_ERROR(status);

  IREE_RETURN_IF_ERROR(qwen_check_proposals(model, rows));
  int32_t decode_tokens[4] = {0};
  for (iree_host_size_t i = 2; i < 4; ++i) {
    decode_tokens[i] = loom_serve_qwen_row_token(
        loom_serve_qwen_model_row(model, rows[i].packed));
  }
  const iree_host_size_t first_order[] = {0, 1, 2, 3};
  const loom_serve_qwen_span_t first[] = {
      {rows[0].packed, 5, rows[0].input + rows[0].prefix_count,
       LOOM_SERVE_QWEN_SPAN_FLAG_SELECT},
      {rows[1].packed, 3, rows[1].input + rows[1].prefix_count, 0},
      {rows[2].packed, 1, &decode_tokens[2], LOOM_SERVE_QWEN_SPAN_FLAG_SELECT},
      {rows[3].packed, 1, &decode_tokens[3], LOOM_SERVE_QWEN_SPAN_FLAG_SELECT},
  };
  loom_serve_qwen_span_t repeated[] = {first[0], first[0]};
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_qwen_model_epoch(model, 0, 2, repeated),
                       IREE_STATUS_INVALID_ARGUMENT));
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 4, first_order, first));
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_qwen_row_decode(
                           loom_serve_qwen_model_row(model, rows[1].packed)),
                       IREE_STATUS_FAILED_PRECONDITION));

  decode_tokens[3] = loom_serve_qwen_row_token(
      loom_serve_qwen_model_row(model, rows[3].packed));
  const iree_host_size_t second_order[] = {3, 1};
  const loom_serve_qwen_span_t second[] = {
      {rows[3].packed, 1, &decode_tokens[3], LOOM_SERVE_QWEN_SPAN_FLAG_SELECT},
      {rows[1].packed, 1, rows[1].input + rows[1].input_count - 1,
       LOOM_SERVE_QWEN_SPAN_FLAG_SELECT},
  };
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 2, second_order, second));

  // Rows omitted by the previous epoch rejoin in different compact slots.
  const iree_host_size_t resumed_order[] = {2, 0, 3, 1};
  for (int step = 0; step < 2 && iree_status_is_ok(status); ++step) {
    loom_serve_qwen_span_t resumed[4];
    for (iree_host_size_t i = 0; i < 4; ++i) {
      const iree_host_size_t logical = resumed_order[i];
      decode_tokens[i] = loom_serve_qwen_row_token(
          loom_serve_qwen_model_row(model, rows[logical].packed));
      resumed[i] =
          (loom_serve_qwen_span_t){rows[logical].packed, 1, &decode_tokens[i],
                                   LOOM_SERVE_QWEN_SPAN_FLAG_SELECT};
    }
    status =
        qwen_check_epoch(model, &next_shape, rows, 4, resumed_order, resumed);
  }
  IREE_RETURN_IF_ERROR(status);
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    char text[1024];
    iree_host_size_t length = 0;
    status = iree_tokenizer_decode(
        tokenizer,
        iree_tokenizer_make_token_id_list(rows[i].output, rows[i].output_count),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(text, sizeof(text)), allocator, &length);
    if (iree_status_is_ok(status)) {
      printf("row %zu: %.*s\n", rows[i].packed, (int)length, text);
    }
  }
  IREE_RETURN_IF_ERROR(status);

  // No-output work still advances retained state, and the next selected
  // continuation must agree after a deliberately supplied additional input.
  decode_tokens[0] = loom_serve_qwen_row_token(
      loom_serve_qwen_model_row(model, rows[0].packed));
  const iree_host_size_t final_order[] = {0};
  const loom_serve_qwen_span_t hidden = {rows[0].packed, 1, decode_tokens, 0};
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 1, final_order, &hidden));
  const loom_serve_qwen_span_t visible = {rows[0].packed, 1, rows[0].input + 3,
                                          LOOM_SERVE_QWEN_SPAN_FLAG_SELECT};
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 1, final_order, &visible));

  // Crossing back into the isolated decode family must consume the packed
  // prediction and position, not the old device-local single-row control.
  IREE_RETURN_IF_ERROR(loom_serve_qwen_row_decode(
      loom_serve_qwen_model_row(model, rows[0].packed)));
  IREE_RETURN_IF_ERROR(loom_serve_qwen_row_decode(
      loom_serve_qwen_model_row(model, rows[0].isolated)));
  return qwen_check_prediction(model, &rows[0]);
}

// Each sample replays the same prefix outside timing. Both arms use the same
// physical resident rows, weights and workspace. The baseline uses the ordinary
// prefill/decode families, not four padded prefill calls for decode inputs.
static iree_status_t qwen_check_measure(
    loom_serve_qwen_model_t* model, const qwen_check_row_t rows[4],
    iree_host_size_t span_count, iree_host_size_t prefill_count,
    const iree_host_size_t lengths[4], bool packed,
    iree_duration_t* out_duration, int32_t outputs[4]) {
  int32_t decode_tokens[4] = {0};
  loom_serve_qwen_span_t spans[4];
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_qwen_row_t* row =
        loom_serve_qwen_model_row(model, rows[i].packed);
    const iree_host_size_t prefix_count =
        rows[i].input_count - (i < prefill_count ? lengths[i] : 0);
    status = loom_serve_qwen_row_reset(row);
    if (iree_status_is_ok(status)) {
      status = qwen_check_prefill(model, rows[i].packed, prefix_count,
                                  rows[i].input);
    }
    if (iree_status_is_ok(status)) {
      decode_tokens[i] = loom_serve_qwen_row_token(row);
      spans[i] = (loom_serve_qwen_span_t){
          rows[i].packed, lengths[i],
          i < prefill_count ? rows[i].input + prefix_count : &decode_tokens[i],
          LOOM_SERVE_QWEN_SPAN_FLAG_SELECT};
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_time_t start = iree_time_now();
  if (packed) {
    status = loom_serve_qwen_model_epoch(model, 0, span_count, spans);
  } else {
    for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
         ++i) {
      loom_serve_qwen_row_t* row =
          loom_serve_qwen_model_row(model, rows[i].packed);
      status = i < prefill_count ? loom_serve_qwen_row_prefill(
                                       row, lengths[i], spans[i].token_ids)
                                 : loom_serve_qwen_row_decode(row);
    }
  }
  *out_duration = iree_time_now() - start;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      loom_serve_qwen_row_t* row =
          loom_serve_qwen_model_row(model, rows[i].packed);
      const iree_host_size_t expected_position =
          rows[i].input_count + (i < prefill_count ? 0 : 1);
      if (loom_serve_qwen_row_position(row) != expected_position) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "comparison row %zu has the wrong position",
                                rows[i].packed);
      }
      outputs[i] = loom_serve_qwen_row_token(row);
    }
  }
  return status;
}

static iree_status_t qwen_check_compare(loom_serve_qwen_model_t* model,
                                        iree_allocator_t allocator) {
  iree_host_size_t span_count = 4;
  iree_host_size_t prefill_count = 2;
  iree_host_size_t lengths[4] = {5, 3, 1, 1};
  if (strcmp(FLAG_compare, "single") == 0) {
    span_count = 1;
    prefill_count = 1;
  } else if (strcmp(FLAG_compare, "decode") == 0) {
    prefill_count = 0;
    lengths[0] = lengths[1] = 1;
  } else if (strcmp(FLAG_compare, "full") == 0) {
    lengths[0] = 17;
    lengths[1] = 13;
  }
  iree_host_size_t token_count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    token_count += lengths[i];
  }
  if (token_count > loom_serve_qwen_model_shapes(model)[0].token_capacity ||
      span_count > loom_serve_qwen_model_shapes(model)[0].span_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "comparison exceeds packed stage capacities");
  }
  qwen_check_row_t rows[4] = {
      {.packed = 6}, {.packed = 1}, {.packed = 7}, {.packed = 3}};
  const char* phrases[] = {"amber cedar maple raven",
                           "violet birch willow falcon",
                           "silver pine oak robin", "golden elm ash eagle"};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
       ++i) {
    char prompt[512];
    snprintf(prompt, sizeof(prompt),
             "<|im_start|>user\nReturn exactly: %s<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\n",
             phrases[i]);
    status = iree_tokenizer_encode(
        loom_serve_qwen_model_tokenizer(model), iree_make_cstring_view(prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(rows[i].input, NULL, NULL, 512),
        allocator, &rows[i].input_count);
    if (iree_status_is_ok(status) && i < prefill_count &&
        rows[i].input_count <= lengths[i]) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "comparison prompt is too short");
    }
  }
  IREE_RETURN_IF_ERROR(status);

  // One untimed warm-up of each arm, then three samples in each ABABA window.
  // Every sample also checks the same final tokens and consumed positions.
  int32_t expected[4] = {0};
  for (int window = -2; window < 5 && iree_status_is_ok(status); ++window) {
    const bool packed = (window + 2) % 2 != 0;
    const int sample_count = window < 0 ? 1 : 3;
    for (int sample = 0; sample < sample_count && iree_status_is_ok(status);
         ++sample) {
      iree_duration_t duration = 0;
      int32_t outputs[4] = {0};
      status = qwen_check_measure(model, rows, span_count, prefill_count,
                                  lengths, packed, &duration, outputs);
      if (iree_status_is_ok(status)) {
        if (window == -2) {
          memcpy(expected, outputs, sizeof(expected));
        } else if (memcmp(expected, outputs, sizeof(expected)) != 0) {
          status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "comparison changed the selected tokens");
        }
      }
      if (iree_status_is_ok(status) && window >= 0) {
        printf(
            "{\"case\":\"%s\",\"arm\":\"%s\",\"window\":%d,"
            "\"sample\":%d,\"tokens\":%zu,\"spans\":%zu,"
            "\"completed_ns\":%" PRId64 "}\n",
            FLAG_compare, packed ? "packed" : "separate", window, sample,
            token_count, span_count, duration);
        fflush(stdout);
      }
    }
  }
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  const iree_flag_string_list_t epochs = FLAG_epoch_list();
  if (!FLAG_prefill[0] || !FLAG_decode[0] || !epochs.count ||
      !FLAG_weights[0] || !FLAG_tokenizer[0]) {
    fprintf(stderr,
            "Provide prefill, decode, epoch, weights and tokenizer paths.\n");
    return EXIT_FAILURE;
  }
  if (FLAG_compare[0] && epochs.count != 1) {
    fprintf(stderr,
            "The isolated comparison requires exactly one epoch shape.\n");
    return EXIT_FAILURE;
  }
  if (FLAG_compare[0] && strcmp(FLAG_compare, "single") != 0 &&
      strcmp(FLAG_compare, "mixed") != 0 && strcmp(FLAG_compare, "full") != 0 &&
      strcmp(FLAG_compare, "decode") != 0) {
    fprintf(stderr, "compare must be single, mixed, full or decode.\n");
    return EXIT_FAILURE;
  }
  const loom_serve_qwen_options_t options = {
      .prefill_directory = iree_make_cstring_view(FLAG_prefill),
      .decode_directory = iree_make_cstring_view(FLAG_decode),
      .epoch_count = epochs.count,
      .epoch_directories = epochs.values,
      .weights_path = iree_make_cstring_view(FLAG_weights),
      .tokenizer_path = iree_make_cstring_view(FLAG_tokenizer),
      .mtp_directory = iree_make_cstring_view(FLAG_mtp),
      .row_count = 8,
  };
  const iree_allocator_t allocator = iree_allocator_system();
  loom_serve_qwen_model_t* model = NULL;
  iree_status_t status =
      loom_serve_qwen_model_create(&options, allocator, &model);
  if (iree_status_is_ok(status)) {
    status = FLAG_compare[0] ? qwen_check_compare(model, allocator)
                             : qwen_check_run(model, allocator);
  }
  status = iree_status_join(status, loom_serve_qwen_model_destroy(model));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  fprintf(stderr, FLAG_compare[0]
                      ? "PASS: completed-work comparison and output checks.\n"
                      : "PASS: packed outputs, omitted outputs, row "
                        "continuation and schedule transitions.\n");
  return EXIT_SUCCESS;
}
