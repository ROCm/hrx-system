// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-weight differential through the runner's public model interface.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/text/flags.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, compare, "",
          "Optional completed-work ABABA comparison: single, mixed, full or "
          "decode. Empty runs the correctness witness.");
IREE_FLAG(int32_t, continuation_epochs, 1,
          "One or two device-fed epochs in the mixed MTP witness.");

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

static iree_status_t qwen_check_prefill(loom_serve_text_model_t* model,
                                        iree_host_size_t row_index,
                                        iree_host_size_t count,
                                        const int32_t* tokens) {
  loom_serve_text_row_t* row = loom_serve_text_model_row(model, row_index);
  const iree_host_size_t capacity =
      loom_serve_text_model_prefill_capacity(model);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t offset = 0;
       offset < count && iree_status_is_ok(status);) {
    const iree_host_size_t length = iree_min(capacity, count - offset);
    status = loom_serve_text_row_prefill(row, length, tokens + offset);
    offset += length;
  }
  return status;
}

static iree_status_t qwen_check_prediction(loom_serve_text_model_t* model,
                                           qwen_check_row_t* row) {
  const int32_t packed =
      loom_serve_text_row_token(loom_serve_text_model_row(model, row->packed));
  const int32_t isolated = loom_serve_text_row_token(
      loom_serve_text_model_row(model, row->isolated));
  if (packed != isolated) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "row %zu prediction %d differs from isolated %d",
                            row->packed, packed, isolated);
  }
  row->output[row->output_count++] = packed;
  return iree_ok_status();
}

// Returns the number of consecutive spans fitting one legal epoch. Each
// fixture span fits the smallest accepted token capacity. Partitioning changes
// compact slots without changing any resident row history.
static iree_host_size_t qwen_check_batch_count(
    loom_serve_packing_shape_t shape, iree_host_size_t count,
    const loom_serve_text_span_t* spans) {
  iree_host_size_t batch_count = 0;
  iree_host_size_t remaining = shape.token_capacity;
  while (batch_count < iree_min(count, shape.span_capacity) &&
         spans[batch_count].token_count <= remaining) {
    remaining -= spans[batch_count++].token_count;
  }
  return batch_count;
}

static iree_status_t qwen_check_epoch(loom_serve_text_model_t* model,
                                      iree_host_size_t* next_shape,
                                      qwen_check_row_t rows[4],
                                      iree_host_size_t count,
                                      const iree_host_size_t logical_rows[4],
                                      const loom_serve_text_span_t* spans) {
  const iree_host_size_t shape_index =
      (*next_shape)++ % loom_serve_text_model_shape_count(model);
  iree_host_size_t epoch_count = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t offset = 0;
       offset < count && iree_status_is_ok(status);) {
    const iree_host_size_t batch_count =
        qwen_check_batch_count(loom_serve_text_model_shapes(model)[shape_index],
                               count - offset, spans + offset);
    status = loom_serve_text_model_epoch(model, shape_index, batch_count,
                                         spans + offset);
    ++epoch_count;
    offset += batch_count;
  }
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    qwen_check_row_t* row = &rows[logical_rows[i]];
    // Length-one decode input uses the same dense schedule here. Comparing to
    // the separate GEMV decode family would mix numerical changes with routing.
    status = qwen_check_prefill(model, row->isolated, spans[i].token_count,
                                spans[i].token_ids);
    if (iree_status_is_ok(status) &&
        iree_any_bit_set(spans[i].flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT)) {
      status = qwen_check_prediction(model, row);
    }
  }
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    const iree_host_size_t packed = loom_serve_text_row_position(
        loom_serve_text_model_row(model, rows[i].packed));
    const iree_host_size_t isolated = loom_serve_text_row_position(
        loom_serve_text_model_row(model, rows[i].isolated));
    if (packed != isolated) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "row %zu position %zu differs from isolated %zu",
                           rows[i].packed, packed, isolated);
    }
  }
  if (iree_status_is_ok(status)) {
    fprintf(
        stderr,
        "Matched packed fixture: shape %zu, %zu spans in %zu epochs; positions "
        "[%zu,%zu,%zu,%zu].\n",
        shape_index, count, epoch_count,
        loom_serve_text_row_position(
            loom_serve_text_model_row(model, rows[0].packed)),
        loom_serve_text_row_position(
            loom_serve_text_model_row(model, rows[1].packed)),
        loom_serve_text_row_position(
            loom_serve_text_model_row(model, rows[2].packed)),
        loom_serve_text_row_position(
            loom_serve_text_model_row(model, rows[3].packed)));
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

static iree_status_t qwen_check_verified_epoch(
    loom_serve_text_model_t* model, iree_host_size_t shape_index,
    iree_host_size_t epoch_count, qwen_check_row_t rows[4],
    const iree_host_size_t order[4], const loom_serve_text_span_t spans[4],
    const uint32_t limits[4], const iree_host_size_t* expected_counts) {
  loom_serve_text_result_t results[4];
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t offset = 0; offset < 4 && iree_status_is_ok(status);) {
    const iree_host_size_t batch_count =
        qwen_check_batch_count(loom_serve_text_model_shapes(model)[shape_index],
                               4 - offset, spans + offset);
    loom_serve_text_span_t next[4];
    uint32_t next_limits[4];
    iree_host_size_t next_count = 0;
    for (iree_host_size_t i = 0; epoch_count == 2 && i < batch_count; ++i) {
      if (!limits[offset + i]) {
        continue;
      }
      next[next_count] = spans[offset + i];
      next[next_count].flags |= LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE;
      next[next_count].token_ids = NULL;
      next_limits[next_count++] = limits[offset + i];
    }
    const loom_serve_text_continuation_t continuation = {
        .shape_index = shape_index,
        .span_count = next_count,
        .spans = next,
        .output_limits = next_limits,
    };
    status = loom_serve_text_model_verify(
        model, shape_index, batch_count, spans + offset, limits + offset,
        next_count ? &continuation : NULL, results + offset);
    offset += batch_count;
  }
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    qwen_check_row_t* fixture = &rows[order[i]];
    loom_serve_text_row_t* reference =
        loom_serve_text_model_row(model, fixture->isolated);
    const loom_serve_text_result_t* result = &results[i];
    const bool selects =
        iree_any_bit_set(spans[i].flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT);
    const bool generates =
        iree_any_bit_set(spans[i].flags, LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE);
    if ((limits[i] &&
         (!result->output_count || result->output_count > limits[i] ||
          result->consumed_count != result->output_count)) ||
        (!limits[i] && (result->consumed_count != spans[i].token_count ||
                        result->output_count != (selects ? 1u : 0u))) ||
        (expected_counts && result->consumed_count != expected_counts[i])) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "verified row %zu has unexpected consumed/output counts %zu/%zu",
          fixture->packed, result->consumed_count, result->output_count);
      continue;
    }
    if (limits[i]) {
      for (iree_host_size_t j = 0;
           j < result->output_count && iree_status_is_ok(status); ++j) {
        const int32_t pending = loom_serve_text_row_token(reference);
        const loom_serve_text_span_t next = {fixture->isolated, 1, &pending,
                                             LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
        status = loom_serve_text_model_epoch(model, shape_index, 1, &next);
        if (iree_status_is_ok(status) &&
            result->tokens[j] != loom_serve_text_row_token(reference)) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "verified row %zu output %zu is %d, ordinary target selected %d",
              fixture->packed, j, result->tokens[j],
              loom_serve_text_row_token(reference));
        }
        if (iree_status_is_ok(status) && j + 1 < result->output_count &&
            ((!generates && result->tokens[j] != spans[i].token_ids[j + 1]) ||
             loom_serve_text_row_is_eos(reference))) {
          status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                    "verification crossed rejection or EOS");
        }
      }
      if (iree_status_is_ok(status) && !generates &&
          result->output_count < limits[i] &&
          !loom_serve_text_row_is_eos(reference) &&
          result->tokens[result->output_count - 1] ==
              spans[i].token_ids[result->output_count]) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "verification stopped before its frontier");
      }
    } else {
      loom_serve_text_span_t known = spans[i];
      known.row_index = fixture->isolated;
      status = loom_serve_text_model_epoch(model, shape_index, 1, &known);
      if (iree_status_is_ok(status) && selects &&
          result->tokens[0] != loom_serve_text_row_token(reference)) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "known span changed during verification");
      }
    }
    if (iree_status_is_ok(status)) {
      const loom_serve_text_row_t* actual =
          loom_serve_text_model_row(model, fixture->packed);
      if (loom_serve_text_row_position(actual) !=
              loom_serve_text_row_position(reference) ||
          (selects && (loom_serve_text_row_token(actual) !=
                           loom_serve_text_row_token(reference) ||
                       loom_serve_text_row_is_eos(actual) !=
                           loom_serve_text_row_is_eos(reference)))) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "verified retained frontier differs");
      }
    }
    if (iree_status_is_ok(status)) {
      memcpy(fixture->output + fixture->output_count, result->tokens,
             result->output_count * sizeof(int32_t));
      fixture->output_count += result->output_count;
      fprintf(stderr,
              "Matched verification: shape %zu, row %zu, limit %u, "
              "consumed %zu, outputs %zu [%d,%d,%d,%d].\n",
              shape_index, fixture->packed, limits[i], result->consumed_count,
              result->output_count, result->tokens[0], result->tokens[1],
              result->tokens[2], result->tokens[3]);
    }
  }
  return status;
}

static iree_status_t qwen_check_mtp_verification(loom_serve_text_model_t* model,
                                                 iree_allocator_t allocator) {
  if (!loom_serve_text_mtp_from_flags()) {
    return iree_ok_status();
  }
  qwen_check_row_t rows[4] = {
      {.packed = 6, .isolated = 0},
      {.packed = 1, .isolated = 2},
      {.packed = 7, .isolated = 4},
      {.packed = 3, .isolated = 5},
  };
  const char* phrases[] = {"amber cedar maple raven",
                           "violet birch willow falcon",
                           "silver pine oak robin", "golden elm ash eagle"};
  int32_t inputs[4][4];
  int32_t padding[32];
  iree_host_size_t padding_count = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      loom_serve_text_model_tokenizer(model),
      IREE_SV("Background context unrelated to the requested list.\n"),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(padding, NULL, NULL,
                                       IREE_ARRAYSIZE(padding)),
      allocator, &padding_count));
  if (!padding_count) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "verification padding tokenized to no inputs");
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    char prompt[512];
    snprintf(
        prompt, sizeof(prompt),
        "<|im_start|>user\nReturn exactly: %s copper hazel sparrow "
        "juniper bronze larch heron poplar crimson beech kestrel "
        "spruce.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        phrases[i]);
    status = iree_tokenizer_encode(
        loom_serve_text_model_tokenizer(model), iree_make_cstring_view(prompt),
        IREE_TOKENIZER_ENCODE_FLAG_NONE,
        iree_tokenizer_make_token_output(rows[i].input, NULL, NULL, 512),
        allocator, &rows[i].input_count);
    if (iree_status_is_ok(status)) {
      // Unequal retained prefixes end immediately before a page boundary.
      // Verification must provision a second page even when acceptance keeps
      // only the anchor in the first page. These are ordinary model inputs;
      // both independent runs consume the same complete history.
      const iree_host_size_t length =
          ((rows[i].input_count + 64) / 64 + i) * 64 - 1;
      if (length > IREE_ARRAYSIZE(rows[i].input)) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "verification prefix exceeds fixture storage");
      } else {
        const iree_host_size_t count = length - rows[i].input_count;
        memmove(rows[i].input + count, rows[i].input,
                rows[i].input_count * sizeof(int32_t));
        for (iree_host_size_t j = 0; j < count; ++j) {
          rows[i].input[j] = padding[j % padding_count];
        }
        rows[i].input_count = length;
      }
    }
    const iree_host_size_t indices[] = {rows[i].packed, rows[i].isolated};
    for (iree_host_size_t j = 0; j < 2 && iree_status_is_ok(status); ++j) {
      status = loom_serve_text_row_reset(
          loom_serve_text_model_row(model, indices[j]));
      if (iree_status_is_ok(status)) {
        status = qwen_check_prefill(model, indices[j], rows[i].input_count,
                                    rows[i].input);
      }
    }
    // Derive matching proposals from ordinary target execution, then restore
    // the reference through its actual input history. No state copies or
    // fabricated prediction oracle are involved.
    loom_serve_text_row_t* reference =
        loom_serve_text_model_row(model, rows[i].isolated);
    for (iree_host_size_t j = 0; j < 4 && iree_status_is_ok(status); ++j) {
      if (loom_serve_text_row_is_eos(reference)) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "verification fixture ended before drafts");
      } else {
        inputs[i][j] = loom_serve_text_row_token(reference);
        status = loom_serve_text_row_decode(reference);
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_row_reset(reference);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_check_prefill(model, rows[i].isolated, rows[i].input_count,
                                  rows[i].input);
    }
  }
  IREE_RETURN_IF_ERROR(status);
  inputs[0][1] = inputs[0][1] == 0 ? 1 : 0;
  inputs[1][2] = inputs[1][2] == 0 ? 1 : 0;
  const iree_host_size_t order[] = {2, 0, 3, 1};
  const uint32_t limits[] = {4, 4, 2, 4};
  const iree_host_size_t expected[] = {4, 1, 2, 2};
  loom_serve_text_span_t spans[4];
  for (iree_host_size_t i = 0; i < 4; ++i) {
    spans[i] =
        (loom_serve_text_span_t){rows[order[i]].packed, 4, inputs[order[i]],
                                 LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
  }
  uint32_t invalid_limits[] = {5, 4, 2, 4};
  loom_serve_text_result_t rejected[4];
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_text_model_verify(
                           model, 0, 1, spans, invalid_limits, NULL, rejected),
                       IREE_STATUS_INVALID_ARGUMENT));
  IREE_RETURN_IF_ERROR(qwen_check_verified_epoch(model, 0, 1, rows, order,
                                                 spans, limits, expected));

  // Natural proposals share the next epoch with an intermediate prompt and a
  // known decode. Captured/replayed slots now contain holes and reordered rows.
  for (iree_host_size_t i = 0; i < 4; ++i) {
    inputs[i][0] = loom_serve_text_row_token(
        loom_serve_text_model_row(model, rows[i].packed));
  }
  const iree_host_size_t mixed_order[] = {3, 0, 2, 1};
  const loom_serve_text_span_t mixed[] = {
      {rows[3].packed, 4, inputs[3],
       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE},
      {rows[0].packed, 3, rows[0].input + 4, 0},
      {rows[2].packed, 4, inputs[2],
       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE},
      {rows[1].packed, 1, inputs[1], LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  const uint32_t mixed_limits[] = {(uint32_t)FLAG_continuation_epochs * 4, 0,
                                   (uint32_t)FLAG_continuation_epochs * 4, 0};
  const iree_host_size_t mixed_shape =
      loom_serve_text_model_shape_count(model) - 1;
  IREE_RETURN_IF_ERROR(
      qwen_check_verified_epoch(model, mixed_shape, FLAG_continuation_epochs,
                                rows, mixed_order, mixed, mixed_limits, NULL));
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_text_row_decode(
                           loom_serve_text_model_row(model, rows[0].packed)),
                       IREE_STATUS_FAILED_PRECONDITION));
  IREE_RETURN_IF_ERROR(
      qwen_check_prefill(model, rows[0].packed, 1, rows[0].input + 7));
  IREE_RETURN_IF_ERROR(
      qwen_check_prefill(model, rows[0].isolated, 1, rows[0].input + 7));
  IREE_RETURN_IF_ERROR(qwen_check_prediction(model, &rows[0]));
  iree_host_size_t next_shape = 0;
  for (int step = 0; step < 3 && iree_status_is_ok(status); ++step) {
    for (iree_host_size_t i = 0; i < 4; ++i) {
      inputs[i][0] = loom_serve_text_row_token(
          loom_serve_text_model_row(model, rows[order[i]].packed));
      spans[i] = (loom_serve_text_span_t){rows[order[i]].packed, 1, inputs[i],
                                          LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
    }
    status = qwen_check_epoch(model, &next_shape, rows, 4, order, spans);
  }
  for (iree_host_size_t i = 0; i < 4 && iree_status_is_ok(status); ++i) {
    char text[1024];
    iree_host_size_t length = 0;
    status = iree_tokenizer_decode(
        loom_serve_text_model_tokenizer(model),
        iree_tokenizer_make_token_id_list(rows[i].output, rows[i].output_count),
        IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        iree_make_mutable_string_view(text, sizeof(text)), allocator, &length);
    if (iree_status_is_ok(status)) {
      printf("verified row %zu: %.*s\n", rows[i].packed, (int)length, text);
    }
  }
  return status;
}

static iree_status_t qwen_check_run(loom_serve_text_model_t* model,
                                    iree_allocator_t allocator) {
  iree_host_size_t widest_shape = 0;
  for (iree_host_size_t i = 0; i < loom_serve_text_model_shape_count(model);
       ++i) {
    const loom_serve_packing_shape_t shape =
        loom_serve_text_model_shapes(model)[i];
    if (shape.token_capacity < 5) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "witness requires at least five tokens in every shape");
    }
    if (shape.span_capacity >
        loom_serve_text_model_shapes(model)[widest_shape].span_capacity) {
      widest_shape = i;
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
  iree_tokenizer_t* tokenizer = loom_serve_text_model_tokenizer(model);
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

  int32_t decode_tokens[4] = {0};
  for (iree_host_size_t i = 2; i < 4; ++i) {
    decode_tokens[i] = loom_serve_text_row_token(
        loom_serve_text_model_row(model, rows[i].packed));
  }
  const iree_host_size_t first_order[] = {0, 1, 2, 3};
  const loom_serve_text_span_t first[] = {
      {rows[0].packed, 5, rows[0].input + rows[0].prefix_count,
       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {rows[1].packed, 3, rows[1].input + rows[1].prefix_count, 0},
      {rows[2].packed, 1, &decode_tokens[2], LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {rows[3].packed, 1, &decode_tokens[3], LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  loom_serve_text_span_t repeated[] = {first[2], first[2]};
  // A one-span catalog rejects this call at the span-count boundary, before
  // inspecting duplicate rows. Wider catalogs reach input validation.
  const iree_status_code_t repeated_code =
      loom_serve_text_model_shapes(model)[widest_shape].span_capacity > 1
          ? IREE_STATUS_INVALID_ARGUMENT
          : IREE_STATUS_OUT_OF_RANGE;
  IREE_RETURN_IF_ERROR(qwen_check_error(
      loom_serve_text_model_epoch(model, widest_shape, 2, repeated),
      repeated_code));
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 4, first_order, first));
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_text_row_decode(
                           loom_serve_text_model_row(model, rows[1].packed)),
                       IREE_STATUS_FAILED_PRECONDITION));

  decode_tokens[3] = loom_serve_text_row_token(
      loom_serve_text_model_row(model, rows[3].packed));
  const iree_host_size_t second_order[] = {3, 1};
  const loom_serve_text_span_t second[] = {
      {rows[3].packed, 1, &decode_tokens[3], LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {rows[1].packed, 1, rows[1].input + rows[1].input_count - 1,
       LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
  };
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 2, second_order, second));

  // Rows omitted by the previous epoch rejoin in different compact slots.
  const iree_host_size_t resumed_order[] = {2, 0, 3, 1};
  for (int step = 0; step < 2 && iree_status_is_ok(status); ++step) {
    loom_serve_text_span_t resumed[4];
    for (iree_host_size_t i = 0; i < 4; ++i) {
      const iree_host_size_t logical = resumed_order[i];
      decode_tokens[i] = loom_serve_text_row_token(
          loom_serve_text_model_row(model, rows[logical].packed));
      resumed[i] =
          (loom_serve_text_span_t){rows[logical].packed, 1, &decode_tokens[i],
                                   LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
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
  decode_tokens[0] = loom_serve_text_row_token(
      loom_serve_text_model_row(model, rows[0].packed));
  const iree_host_size_t final_order[] = {0};
  const loom_serve_text_span_t hidden = {rows[0].packed, 1, decode_tokens, 0};
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 1, final_order, &hidden));
  const loom_serve_text_span_t visible = {rows[0].packed, 1, rows[0].input + 3,
                                          LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
  IREE_RETURN_IF_ERROR(
      qwen_check_epoch(model, &next_shape, rows, 1, final_order, &visible));

  // Crossing back into the isolated decode family must consume the packed
  // prediction and position, not the old device-local single-row control.
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(
      loom_serve_text_model_row(model, rows[0].packed)));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(
      loom_serve_text_model_row(model, rows[0].isolated)));
  return qwen_check_prediction(model, &rows[0]);
}

// Each sample replays the same prefix outside timing. Both arms use the same
// physical resident rows, weights and workspace. The baseline uses the ordinary
// prefill/decode families, not four padded prefill calls for decode inputs.
static iree_status_t qwen_check_measure(
    loom_serve_text_model_t* model, const qwen_check_row_t rows[4],
    iree_host_size_t span_count, iree_host_size_t prefill_count,
    const iree_host_size_t lengths[4], bool packed,
    iree_duration_t* out_duration, int32_t outputs[4]) {
  int32_t decode_tokens[4] = {0};
  loom_serve_text_span_t spans[4];
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
       ++i) {
    loom_serve_text_row_t* row =
        loom_serve_text_model_row(model, rows[i].packed);
    const iree_host_size_t prefix_count =
        rows[i].input_count - (i < prefill_count ? lengths[i] : 0);
    status = loom_serve_text_row_reset(row);
    if (iree_status_is_ok(status)) {
      status = qwen_check_prefill(model, rows[i].packed, prefix_count,
                                  rows[i].input);
    }
    if (iree_status_is_ok(status)) {
      decode_tokens[i] = loom_serve_text_row_token(row);
      spans[i] = (loom_serve_text_span_t){
          rows[i].packed, lengths[i],
          i < prefill_count ? rows[i].input + prefix_count : &decode_tokens[i],
          LOOM_SERVE_TEXT_SPAN_FLAG_SELECT};
    }
  }
  IREE_RETURN_IF_ERROR(status);
  const iree_time_t start = iree_time_now();
  if (packed) {
    status = loom_serve_text_model_epoch(model, 0, span_count, spans);
  } else {
    for (iree_host_size_t i = 0; i < span_count && iree_status_is_ok(status);
         ++i) {
      loom_serve_text_row_t* row =
          loom_serve_text_model_row(model, rows[i].packed);
      status = i < prefill_count ? loom_serve_text_row_prefill(
                                       row, lengths[i], spans[i].token_ids)
                                 : loom_serve_text_row_decode(row);
    }
  }
  *out_duration = iree_time_now() - start;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      loom_serve_text_row_t* row =
          loom_serve_text_model_row(model, rows[i].packed);
      const iree_host_size_t expected_position =
          rows[i].input_count + (i < prefill_count ? 0 : 1);
      if (loom_serve_text_row_position(row) != expected_position) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "comparison row %zu has the wrong position",
                                rows[i].packed);
      }
      outputs[i] = loom_serve_text_row_token(row);
    }
  }
  return status;
}

static iree_status_t qwen_check_compare(loom_serve_text_model_t* model,
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
  if (token_count > loom_serve_text_model_shapes(model)[0].token_capacity ||
      span_count > loom_serve_text_model_shapes(model)[0].span_capacity) {
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
        loom_serve_text_model_tokenizer(model), iree_make_cstring_view(prompt),
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
  if (FLAG_continuation_epochs < 1 || FLAG_continuation_epochs > 2) {
    fprintf(stderr, "continuation_epochs must be one or two.\n");
    return EXIT_FAILURE;
  }
  const iree_host_size_t epoch_count =
      loom_serve_text_explicit_shape_count_from_flags();
  if (!epoch_count) {
    fprintf(stderr, "Provide at least one --epoch=tokens:spans shape.\n");
    return EXIT_FAILURE;
  }
  if (FLAG_compare[0] && epoch_count != 1) {
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
  const iree_allocator_t allocator = iree_allocator_system();
  const loom_serve_text_flag_defaults_t defaults = {.row_count = 8};
  loom_serve_text_model_t* model = NULL;
  iree_status_t status =
      loom_serve_text_model_create_from_flags(&defaults, &model, allocator);
  if (iree_status_is_ok(status)) {
    status = FLAG_compare[0] ? qwen_check_compare(model, allocator)
                             : qwen_check_run(model, allocator);
  }
  if (iree_status_is_ok(status) && !FLAG_compare[0]) {
    status = qwen_check_mtp_verification(model, allocator);
  }
  status = iree_status_join(status, loom_serve_text_model_destroy(model));
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
