// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Real-weight differential through the runner's public model interface.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/loom_serve/runtime/device_flags.h"
#include "experimental/loom_serve/runtime/residency.h"
#include "experimental/loom_serve/text/flags.h"
#include "iree/base/tooling/flags.h"

IREE_FLAG(string, compare, "",
          "Optional completed-work ABABA comparison: single, mixed, full or "
          "decode. Empty runs the correctness witness.");
IREE_FLAG(int32_t, continuation_epochs, 1,
          "One or two device-fed epochs in the mixed MTP witness.");
IREE_FLAG(bool, trim, false,
          "Also compare live compaction, physical trim and regrowth with an "
          "uncompacted continuation; requires pooled state and context >=512.");
IREE_FLAG(bool, reload_weights, false,
          "Run the trim witness with weight eviction/reload before target/MTP "
          "continuation; requires elastic backing.");
IREE_FLAG(bool, checkpoints, false,
          "Check shared non-page-aligned forks, rewind, COW and reclamation; "
          "requires checkpoint_capacity=1 and pooled state.");
IREE_FLAG(bool, suspend_checkpoints, false,
          "Extend checkpoints with independent DRAM endpoints, live-branch "
          "preservation and denied restore; requires elastic backing and pool "
          "capacity in [2048, 3584].");
IREE_FLAG(
    bool, suspend_rows, false,
    "Extend the trim witness with retained-row DRAM capture, block reuse, "
    "denied restore and fresh-ID resume; requires elastic pooled state "
    "and pool capacity in [2048, 3584].");
IREE_FLAG(
    bool, shared_residency, false,
    "Alternate two independent retained models on one device and physical "
    "budget using automatic eviction; requires elastic backing and a budget "
    "too small for both parameter sets.");

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

// Captures uninterrupted continuation, then rewinds to its starting frontier.
static iree_status_t qwen_check_residency_reference(
    loom_serve_text_model_t* model, iree_string_view_t prompt,
    int32_t expected[8], iree_allocator_t allocator) {
  int32_t tokens[256];
  iree_host_size_t token_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode(loom_serve_text_model_tokenizer(model), prompt,
                            IREE_TOKENIZER_ENCODE_FLAG_NONE,
                            iree_tokenizer_make_token_output(
                                tokens, NULL, NULL, IREE_ARRAYSIZE(tokens)),
                            allocator, &token_count));
  loom_serve_text_row_t* row = loom_serve_text_model_row(model, 0);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(row));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 0, token_count, tokens));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 8 && iree_status_is_ok(status); ++i) {
    expected[i] = loom_serve_text_row_token(row);
    status = loom_serve_text_row_decode(row);
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(row));
  return qwen_check_prefill(model, 0, token_count, tokens);
}

static iree_status_t qwen_check_residency_continuation(
    loom_serve_text_model_t* model, const int32_t expected[8]) {
  loom_serve_text_row_t* row = loom_serve_text_model_row(model, 0);
  const iree_host_size_t initial_position = loom_serve_text_row_position(row);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < 8 && iree_status_is_ok(status); ++i) {
    if (loom_serve_text_row_token(row) != expected[i]) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "shared residency changed token %zu", i);
    } else {
      status = loom_serve_text_row_decode(row);
    }
  }
  if (iree_status_is_ok(status) &&
      loom_serve_text_row_position(row) != initial_position + 8) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "shared residency changed retained frontier");
  }
  return status;
}

static iree_status_t qwen_check_shared_residency(loom_serve_device_t* device,
                                                 loom_serve_text_model_t* first,
                                                 iree_allocator_t allocator) {
  if (!loom_serve_device_memory_pool(device)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "shared residency witness requires elastic backing");
  }
  const loom_serve_text_flag_defaults_t defaults = {.row_count = 1};
  loom_serve_text_model_t* second = NULL;
  iree_status_t status = loom_serve_text_model_create_from_flags(
      device, &defaults, &second, allocator);
  if (iree_status_is_ok(status) &&
      loom_serve_text_model_weight_statistics(second).committed_bytes) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "cold model creation materialized parameters");
  }
  int32_t expected[2][8] = {{0}};
  if (iree_status_is_ok(status)) {
    status = qwen_check_residency_reference(
        first,
        IREE_SV("<|im_start|>user\nRepeat: amber cedar maple violet birch "
                "willow falcon.<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n"),
        expected[0], allocator);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_residency_t* first_residency =
        loom_serve_text_model_residency(first);
    status = loom_serve_residency_acquire(first_residency);
    if (iree_status_is_ok(status)) {
      bool admitted = false;
      status = loom_serve_residency_try_acquire(
          loom_serve_text_model_residency(second), &admitted);
      if (iree_status_is_ok(status) && admitted) {
        loom_serve_residency_release(loom_serve_text_model_residency(second));
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "shared residency witness needs a smaller physical budget");
      }
      if (iree_status_is_ok(status) &&
          loom_serve_text_model_weight_statistics(second).committed_bytes) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "denied admission partially loaded weights");
      }
      loom_serve_residency_release(first_residency);
    }
  }
  if (iree_status_is_ok(status)) {
    status = qwen_check_residency_reference(
        second,
        IREE_SV("<|im_start|>user\nRepeat: silver pine oak robin copper "
                "hazel sparrow.<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n"),
        expected[1], allocator);
  }
  if (iree_status_is_ok(status) &&
      loom_serve_text_model_weight_statistics(first).committed_bytes) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "second model did not evict idle first model");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_check_residency_continuation(first, expected[0]);
  }
  if (iree_status_is_ok(status) &&
      loom_serve_text_model_weight_statistics(second).committed_bytes) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "first model did not evict idle second model");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_check_residency_continuation(second, expected[1]);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_memory_statistics_t memory =
        loom_serve_memory_pool_statistics(
            loom_serve_device_memory_pool(device));
    fprintf(stderr,
            "{\"event\":\"shared_residency\",\"models\":2,\"automatic\":true,"
            "\"reserved_bytes\":"
            "%" PRIu64 ",\"committed_bytes\":%" PRIu64
            ",\"peak_bytes\":%" PRIu64 ",\"released_bytes\":%" PRIu64 "}\n",
            memory.reserved_bytes, memory.committed_bytes, memory.peak_bytes,
            memory.released_bytes);
    fprintf(stderr,
            "PASS: two cached models share one physical budget and preserve "
            "independent continuations across automatic eviction/reload; "
            "retention pins deny admission without partial loading.\n");
  }
  return iree_status_join(status, loom_serve_text_model_destroy(second));
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

  // Every entry point honors the same request boundary, including isolated
  // dense calls. Rejection leaves the prediction usable after credit release.
  loom_serve_text_row_t* reserved =
      loom_serve_text_model_row(model, rows[0].packed);
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      reserved, LOOM_SERVE_TEXT_RESERVE_CONTINUE, NULL,
      loom_serve_text_row_position(reserved), &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "retained-frontier reservation refused");
  }
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_text_row_prefill(reserved, 1, rows[0].input),
                       IREE_STATUS_OUT_OF_RANGE));
  IREE_RETURN_IF_ERROR(qwen_check_error(loom_serve_text_row_decode(reserved),
                                        IREE_STATUS_OUT_OF_RANGE));
  IREE_RETURN_IF_ERROR(
      qwen_check_error(loom_serve_text_model_epoch(model, 0, 1, &visible),
                       IREE_STATUS_OUT_OF_RANGE));
  loom_serve_text_row_release_reservation(reserved);

  // Crossing back into the isolated decode family must consume the packed
  // prediction and position, not the old device-local single-row control.
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(
      loom_serve_text_model_row(model, rows[0].packed)));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(
      loom_serve_text_model_row(model, rows[0].isolated)));
  return qwen_check_prediction(model, &rows[0]);
}

static iree_status_t qwen_check_suspend_roundtrip(
    loom_serve_text_model_t* model, const int32_t input[512]) {
  loom_serve_text_row_t* row = loom_serve_text_model_row(model, 7);
  const iree_host_size_t position = loom_serve_text_row_position(row);
  const int32_t prediction = loom_serve_text_row_token(row);
  const loom_serve_text_metrics_t metrics = loom_serve_text_row_metrics(row);
  const uint64_t before =
      loom_serve_text_model_memory_statistics(model).committed_bytes;
  const iree_time_t capture_start = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_row_suspend(row));
  const iree_duration_t capture_duration = iree_time_now() - capture_start;
  const iree_host_size_t saved = loom_serve_text_row_suspended_bytes(row);
  if (!saved || loom_serve_text_row_pool_usage(row) ||
      loom_serve_text_row_position(row) != position ||
      loom_serve_text_row_token(row) != prediction) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "suspension changed frontier or retained IDs");
  }
  IREE_RETURN_IF_ERROR(qwen_check_error(loom_serve_text_row_decode(row),
                                        IREE_STATUS_FAILED_PRECONDITION));
  loom_serve_text_trim_result_t trimmed;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  const uint64_t after =
      loom_serve_text_model_memory_statistics(model).committed_bytes;
  if (!before || after || !trimmed.released_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "suspended-only model retained physical state");
  }
  // The remaining rows fill the whole logical pool, overwriting the old IDs.
  // Row zero stays resident during restore, so the resumed row cannot get its
  // original compact prefix back even after the other fillers are reset.
  for (iree_host_size_t i = 0; i < 7; ++i) {
    const iree_host_size_t available =
        loom_serve_text_model_pool_usage(model).available;
    if (!available) {
      break;
    }
    IREE_RETURN_IF_ERROR(
        qwen_check_prefill(model, i, iree_min(available, 512u), input));
  }
  bool resumed = true;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_resume(row, &resumed));
  if (resumed || loom_serve_text_model_pool_usage(model).available ||
      loom_serve_text_row_suspended_bytes(row) != saved ||
      loom_serve_text_row_position(row) != position ||
      loom_serve_text_row_token(row) != prediction) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "denied resume changed the saved session");
  }
  for (iree_host_size_t i = 1; i < 7; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  const iree_time_t restore_start = iree_time_now();
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_resume(row, &resumed));
  const iree_duration_t restore_duration = iree_time_now() - restore_start;
  const loom_serve_text_metrics_t restored = loom_serve_text_row_metrics(row);
  if (!resumed || loom_serve_text_row_suspended_bytes(row) ||
      loom_serve_text_row_position(row) != position ||
      loom_serve_text_row_token(row) != prediction ||
      memcmp(&metrics, &restored, sizeof(metrics))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "resume changed the saved host frontier");
  }
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_reset(loom_serve_text_model_row(model, 0)));
  fprintf(stderr,
          "{\"event\":\"row_restored\",\"host_snapshot_bytes\":%zu,"
          "\"state_before_bytes\":%" PRIu64
          ",\"state_suspended_bytes\":%" PRIu64 ",\"capture_ns\":%" PRId64
          ",\"restore_ns\":%" PRId64 "}\n",
          saved, before, after, capture_duration, restore_duration);
  return iree_ok_status();
}

static iree_status_t qwen_check_trim(loom_serve_text_model_t* model,
                                     iree_allocator_t allocator) {
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(model);
  if (loom_serve_text_model_context_capacity(model) < 512 ||
      pool.capacity < 2048 || (FLAG_suspend_rows && pool.capacity > 3584)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trim requires context >=512, pool >=2048; "
                            "suspend_rows requires pool <=3584");
  }
  for (iree_host_size_t i = 0; i < 8; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  loom_serve_text_trim_result_t trimmed;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_memory_statistics(model).committed_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "empty state still has physical backing");
  }
  int32_t input[512], prompt[128], padding[32];
  iree_host_size_t prompt_count = 0, padding_count = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      loom_serve_text_model_tokenizer(model),
      IREE_SV("<|im_start|>user\nReturn exactly: amber cedar maple raven "
              "copper hazel sparrow juniper bronze larch heron poplar crimson "
              "beech kestrel spruce.<|im_end|>\n<|im_start|>assistant\n"
              "<think>\n\n</think>\n\n"),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(prompt, NULL, NULL,
                                       IREE_ARRAYSIZE(prompt)),
      allocator, &prompt_count));
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      loom_serve_text_model_tokenizer(model), IREE_SV("Background context.\n"),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(padding, NULL, NULL,
                                       IREE_ARRAYSIZE(padding)),
      allocator, &padding_count));
  if (!padding_count) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "empty fixture padding");
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(input); ++i) {
    input[i] = padding[i % padding_count];
  }
  // Sixteen low blocks pin at least one physical slab per plane. The survivor
  // then starts above those blocks and must move when their rows are reset.
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 0, 512, input));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 1, 512, input));
  const iree_host_size_t prefix_count = 193;
  memcpy(input + prefix_count - prompt_count, prompt,
         prompt_count * sizeof(*prompt));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, prefix_count, input));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 6, prefix_count, input));
  loom_serve_text_row_t* survivor = loom_serve_text_model_row(model, 7);
  loom_serve_text_row_t* reference = loom_serve_text_model_row(model, 6);
  int32_t expected[8];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    expected[i] = loom_serve_text_row_token(reference);
    IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  }
  const int32_t expected_pending = loom_serve_text_row_token(reference);
  loom_serve_text_result_t expected_verification = {0};
  const uint32_t limit = 4;
  if (loom_serve_text_mtp_from_flags()) {
    const loom_serve_text_span_t span = {
        6, 4, &expected_pending,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE};
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(
        model, 0, 1, &span, &limit, NULL, &expected_verification));
  }
  const iree_host_size_t expected_position =
      loom_serve_text_row_position(reference);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(reference));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_reset(loom_serve_text_model_row(model, 0)));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_reset(loom_serve_text_model_row(model, 1)));
  const loom_serve_memory_statistics_t before =
      loom_serve_text_model_memory_statistics(model);
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  const loom_serve_memory_statistics_t after =
      loom_serve_text_model_memory_statistics(model);
  if (before.reserved_bytes &&
      (!trimmed.moved_blocks || !trimmed.released_bytes ||
       after.committed_bytes >= before.committed_bytes)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "live compaction did not move blocks and free backing");
  }
  fprintf(stderr,
          "{\"event\":\"memory_trimmed\",\"moved_blocks\":%u,"
          "\"copied_bytes\":%" PRIu64 ",\"released_bytes\":%" PRIu64
          ",\"committed_bytes\":%" PRIu64 "}\n",
          trimmed.moved_blocks, trimmed.copied_bytes, trimmed.released_bytes,
          after.committed_bytes);
  if (FLAG_suspend_rows) {
    IREE_RETURN_IF_ERROR(qwen_check_suspend_roundtrip(model, input));
  }
  if (FLAG_reload_weights) {
    const loom_serve_memory_statistics_t resident =
        loom_serve_text_model_weight_statistics(model);
    IREE_RETURN_IF_ERROR(loom_serve_text_model_deactivate(model));
    const loom_serve_memory_statistics_t inactive =
        loom_serve_text_model_weight_statistics(model);
    const loom_serve_memory_statistics_t state =
        loom_serve_text_model_memory_statistics(model);
    if (!resident.committed_bytes || inactive.committed_bytes ||
        inactive.reserved_bytes != resident.reserved_bytes ||
        state.committed_bytes != after.committed_bytes) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "parameter eviction changed retained state or "
                              "failed to release physical weights");
    }
    fprintf(stderr,
            "{\"event\":\"weights_deactivated\",\"released_bytes\":%" PRIu64
            ",\"state_bytes\":%" PRIu64 "}\n",
            resident.committed_bytes, state.committed_bytes);
    // The next decode activates on demand through the existing model API.
    // Its commands, target KV, recurrent state and draft cache are unchanged.
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    if (loom_serve_text_row_token(survivor) != expected[i]) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "compaction changed continuation token %zu", i);
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(survivor));
  }
  if (loom_serve_text_row_token(survivor) != expected_pending) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "compaction changed pending token");
  }
  if (loom_serve_text_mtp_from_flags()) {
    const loom_serve_text_span_t span = {
        7, 4, &expected_pending,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE};
    loom_serve_text_result_t actual;
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(model, 0, 1, &span,
                                                      &limit, NULL, &actual));
    if (actual.consumed_count != expected_verification.consumed_count ||
        actual.output_count != expected_verification.output_count ||
        memcmp(actual.tokens, expected_verification.tokens,
               actual.output_count * sizeof(*actual.tokens))) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "compaction changed draft verification");
    }
  }
  if (loom_serve_text_row_position(survivor) != expected_position) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "compaction changed frontier");
  }
  if (FLAG_reload_weights) {
    fprintf(stderr,
            "{\"event\":\"weights_reactivated\",\"committed_bytes\":%" PRIu64
            "}\n",
            loom_serve_text_model_weight_statistics(model).committed_bytes);
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(survivor));
  if (FLAG_suspend_rows) {
    // An unselected prefill chunk has no pending token but still owns both
    // recurrence and KV. Restore it, finish the prompt and compare the normal
    // selected prediction before testing reset of a suspended row.
    const iree_host_size_t first = 32;
    const loom_serve_text_span_t span = {7, first, input, 0};
    IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(model, 0, 1, &span));
    IREE_RETURN_IF_ERROR(loom_serve_text_row_suspend(survivor));
    IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
    bool resumed = false;
    IREE_RETURN_IF_ERROR(loom_serve_text_row_try_resume(survivor, &resumed));
    if (!resumed) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "empty pool rejected unselected row restore");
    }
    IREE_RETURN_IF_ERROR(
        qwen_check_prefill(model, 7, prefix_count - first, input + first));
    if (loom_serve_text_row_token(survivor) != expected[0]) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "unselected row restore changed prediction");
    }
    IREE_RETURN_IF_ERROR(loom_serve_text_row_suspend(survivor));
    IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
    IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(survivor));
    if (loom_serve_text_row_position(survivor) ||
        loom_serve_text_row_suspended_bytes(survivor)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "reset retained the suspended row");
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_memory_statistics(model).committed_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "final trim retained backing");
  }
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, prefix_count, input));
  if (loom_serve_text_row_token(survivor) != expected[0]) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "regrowth changed prediction");
  }
  fprintf(stderr,
          "PASS: live relocation, target/draft continuation, full trim and "
          "regrowth.\n");
  return iree_ok_status();
}

// Each sample replays the same prefix outside timing. Both arms use the same
// resident rows, weights and workspace. The separate arm issues one public row
// call per span: pooled/MTP calls use packed epoch stages, while dense target-
// only calls use the isolated prefill/decode families. Results must retain the
// backing/speculation configuration to distinguish these comparisons.
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

static iree_status_t qwen_check_endpoint(loom_serve_text_model_t* model,
                                         iree_host_size_t actual_index,
                                         iree_host_size_t reference_index) {
  const loom_serve_text_row_t* actual =
      loom_serve_text_model_row(model, actual_index);
  const loom_serve_text_row_t* reference =
      loom_serve_text_model_row(model, reference_index);
  if (loom_serve_text_row_position(actual) !=
          loom_serve_text_row_position(reference) ||
      loom_serve_text_row_token(actual) !=
          loom_serve_text_row_token(reference) ||
      loom_serve_text_row_is_eos(actual) !=
          loom_serve_text_row_is_eos(reference)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "checkpoint row %zu differs from replay row %zu at "
                            "position %zu/%zu, token %d/%d",
                            actual_index, reference_index,
                            loom_serve_text_row_position(actual),
                            loom_serve_text_row_position(reference),
                            loom_serve_text_row_token(actual),
                            loom_serve_text_row_token(reference));
  }
  return iree_ok_status();
}

static iree_status_t qwen_check_restore(
    loom_serve_text_model_t* model, iree_host_size_t index,
    const loom_serve_text_checkpoint_t* checkpoint) {
  bool restored = false;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_restore(
      loom_serve_text_model_row(model, index), checkpoint, &restored));
  return restored ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_RESOURCE_EXHAUSTED,
                        "checkpoint witness could not admit row %zu", index);
}

static iree_status_t qwen_check_checkpoint_suspension(
    loom_serve_text_model_t* model, const int32_t input[512]) {
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(model);
  if (pool.capacity < 2048 || pool.capacity > 3584 ||
      !loom_serve_text_model_memory_statistics(model).reserved_bytes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "suspend_checkpoints requires elastic pool capacity in [2048, 3584]");
  }
  loom_serve_text_row_t* source = loom_serve_text_model_row(model, 7);
  loom_serve_text_row_t* selected = loom_serve_text_model_row(model, 0);
  loom_serve_text_row_t* branch = loom_serve_text_model_row(model, 1);
  loom_serve_text_row_t* restored = loom_serve_text_model_row(model, 3);
  loom_serve_text_row_t* reference = loom_serve_text_model_row(model, 6);
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 0, 512, input));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, 127, input));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 6, 127, input));
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(source, &checkpoint));
  if (!checkpoint) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "cold endpoint pin refused");
  }
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 1, checkpoint));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(source));
  // Reuse the original execution row while another branch owns the prefix.
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, 512, input));
  const iree_host_size_t available =
      loom_serve_text_model_pool_usage(model).available;
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_suspend(checkpoint));
  const iree_host_size_t saved =
      loom_serve_text_checkpoint_suspended_bytes(checkpoint);
  if (!saved ||
      loom_serve_text_model_pool_usage(model).available != available) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "suspended endpoint released a live branch");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_suspend(checkpoint));
  loom_serve_text_trim_result_t trimmed;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(branch));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 6));
  for (iree_host_size_t i = 0; i < 8; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_memory_statistics(model).committed_bytes ||
      loom_serve_text_model_pool_usage(model).available != pool.capacity) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "cold endpoint retained device backing");
  }
  // Fill every logical slot, starting with the old source row. The endpoint
  // cannot borrow selected-row IDs before its replacement is safely admitted.
  for (iree_host_size_t i = 0; i < 7; ++i) {
    const iree_host_size_t remaining =
        loom_serve_text_model_pool_usage(model).available;
    if (!remaining) {
      break;
    }
    IREE_RETURN_IF_ERROR(qwen_check_prefill(model, (i + 7) % 8,
                                            iree_min(remaining, 512u), input));
  }
  const iree_host_size_t position = loom_serve_text_row_position(selected);
  const int32_t token = loom_serve_text_row_token(selected);
  bool admitted = true;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_try_restore(selected, checkpoint, &admitted));
  if (admitted || loom_serve_text_model_pool_usage(model).available ||
      loom_serve_text_checkpoint_suspended_bytes(checkpoint) != saved ||
      loom_serve_text_row_position(selected) != position ||
      loom_serve_text_row_token(selected) != token) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "cold restore refusal changed an owner");
  }
  // Rows 7 and 0 keep the low IDs occupied, forcing a different physical map.
  for (iree_host_size_t i = 1; i < 7; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 3, checkpoint));
  if (loom_serve_text_checkpoint_suspended_bytes(checkpoint)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "warm endpoint retained its DRAM image");
  }
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 6, 127, input));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 3, 6));
  if (loom_serve_text_mtp_from_flags()) {
    const int32_t anchor = loom_serve_text_row_token(restored);
    const loom_serve_text_span_t span = {
        3, 4, &anchor,
        LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE};
    loom_serve_text_span_t next = span;
    next.token_ids = NULL;
    const uint32_t limit = 8;
    const loom_serve_text_continuation_t continuation = {0, 1, &next, &limit};
    loom_serve_text_result_t result;
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(
        model, 0, 1, &span, &limit, &continuation, &result));
    if (!result.output_count || result.verification_count != 2) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "cold endpoint did not execute both MTP epochs");
    }
    for (iree_host_size_t i = 0; i < result.output_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
      if (result.tokens[i] != loom_serve_text_row_token(reference)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "cold endpoint MTP differs from replay");
      }
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(restored));
    IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  }
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 3, 6));
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_suspend(checkpoint));
  loom_serve_text_checkpoint_release(checkpoint);
  checkpoint = NULL;
  for (iree_host_size_t i = 0; i < 8; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  // An unselected cold endpoint preserves the absence of a pending token.
  const loom_serve_text_span_t hidden = {7, 3, input, 0};
  IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(model, 0, 1, &hidden));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(source, &checkpoint));
  if (!checkpoint) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "cold pin capacity leaked");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_suspend(checkpoint));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(source));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 3, checkpoint));
  IREE_RETURN_IF_ERROR(qwen_check_error(loom_serve_text_row_decode(restored),
                                        IREE_STATUS_FAILED_PRECONDITION));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 3, 1, input + 3));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 6, 4, input));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 3, 6));
  // Model destruction invalidates remaining endpoint handles and must own
  // their cold images too; the first endpoint exercised explicit cold release.
  IREE_RETURN_IF_ERROR(loom_serve_text_checkpoint_suspend(checkpoint));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(restored));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(reference));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_memory_statistics(model).committed_bytes ||
      loom_serve_text_model_pool_usage(model).available != pool.capacity) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "cold endpoint lifecycle leaked backing");
  }
  fprintf(stderr,
          "{\"event\":\"cold_checkpoint\",\"host_snapshot_bytes\":%zu,"
          "\"source_row\":7,\"restored_row\":3,\"committed_bytes\":0}\n",
          saved);
  return iree_ok_status();
}

static iree_status_t qwen_check_reserved_relocation(
    loom_serve_text_model_t* model, const int32_t* input,
    iree_host_size_t position, iree_host_size_t extent,
    uint64_t* out_copied_bytes) {
  loom_serve_text_row_t* filler = loom_serve_text_model_row(model, 0);
  loom_serve_text_row_t* survivor = loom_serve_text_model_row(model, 1);
  loom_serve_text_row_t* reference = loom_serve_text_model_row(model, 7);
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      filler, LOOM_SERVE_TEXT_RESERVE_FRESH, NULL, extent, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "filler reservation refused");
  }
  // Establish a compact low-ID filler independently of previous free order.
  // The subsequent survivor must lie above it until the filler is released.
  loom_serve_text_trim_result_t trimmed;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      survivor, LOOM_SERVE_TEXT_RESERVE_FRESH, NULL, extent, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "survivor reservation refused");
  }
  if (position) {
    IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 1, position, input));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(filler));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  const loom_serve_memory_statistics_t memory =
      loom_serve_text_model_memory_statistics(model);
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(model);
  const iree_host_size_t consumed =
      ((position + pool.block_size - 1) / pool.block_size) * pool.block_size;
  if (pool.available != pool.capacity - extent ||
      pool.pending != extent - consumed ||
      loom_serve_text_row_position(survivor) != position ||
      (memory.reserved_bytes &&
       trimmed.moved_blocks != extent / pool.block_size)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "compaction changed reserved page ownership");
  }
  fprintf(
      stderr,
      "{\"event\":\"reservation_relocated\",\"position\":%zu,"
      "\"reserved_tokens\":%zu,\"moved_blocks\":%u,\"copied_bytes\":%" PRIu64
      ",\"committed_bytes\":%" PRIu64 "}\n",
      position, extent, trimmed.moved_blocks, trimmed.copied_bytes,
      memory.committed_bytes);
  *out_copied_bytes = trimmed.copied_bytes;
  // Cross from retained KV into a relocated, previously unwritten page while
  // keeping the completion reservation. No new backing may be needed.
  IREE_RETURN_IF_ERROR(
      qwen_check_prefill(model, 1, 129 - position, input + position));
  if (loom_serve_text_model_memory_statistics(model).committed_bytes !=
      memory.committed_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "relocated reservation lost physical backing");
  }
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, 129, input));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 7));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(survivor));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 7));
  if (loom_serve_text_mtp_from_flags()) {
    const int32_t token = loom_serve_text_row_token(survivor);
    const loom_serve_text_span_t spans[] = {
        {1, 4, &token,
         LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE},
        {7, 4, &token,
         LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE}};
    const uint32_t limits[] = {4, 4};
    loom_serve_text_result_t results[2];
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(model, 0, 2, spans,
                                                      limits, NULL, results));
    if (results[0].consumed_count != results[1].consumed_count ||
        results[0].output_count != results[1].output_count ||
        memcmp(results[0].tokens, results[1].tokens,
               results[0].output_count * sizeof(*results[0].tokens))) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "relocated reservation changed draft output");
    }
    IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 7));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(survivor));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(reference));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_pool_usage(model).available != pool.capacity ||
      loom_serve_text_model_memory_statistics(model).committed_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "relocated reservation leaked backing");
  }
  return iree_ok_status();
}

static iree_status_t qwen_check_reservations(loom_serve_text_model_t* model,
                                             const int32_t* input) {
  loom_serve_text_row_t* row = loom_serve_text_model_row(model, 0);
  loom_serve_text_row_t* reference = loom_serve_text_model_row(model, 7);
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, 127, input));
  bool admitted = false;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_FRESH, NULL, 256, &admitted));
  if (!admitted || loom_serve_text_model_pool_usage(model).pending != 256) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fresh completion reservation was not retained");
  }
  IREE_RETURN_IF_ERROR(qwen_check_error(loom_serve_text_model_deactivate(model),
                                        IREE_STATUS_FAILED_PRECONDITION));
  loom_serve_text_trim_result_t trimmed;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  const uint64_t committed =
      loom_serve_text_model_memory_statistics(model).committed_bytes;
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 0, 127, input));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 0, 7));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(row));
  if (loom_serve_text_model_memory_statistics(model).committed_bytes !=
      committed) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "reserved inference changed physical commitment");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 0, 7));
  loom_serve_text_row_release_reservation(row);
  if (loom_serve_text_model_pool_usage(model).pending) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "completed request retained unused page credit");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      row, LOOM_SERVE_TEXT_RESERVE_CONTINUE, NULL, 256, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "continuation reservation refused");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(row));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 0, 7));
  loom_serve_text_row_release_reservation(row);
  // Use a partial tail and release a reserved fork before its first epoch.
  // Its private writer has no state yet: continuation must reattach the reader.
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(reference));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, 127, input));
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(reference, &checkpoint));
  if (!checkpoint) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "reservation pin refused");
  }
  loom_serve_text_row_t* fork = loom_serve_text_model_row(model, 1);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      fork, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint, 256, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "fork reservation refused");
  }
  loom_serve_text_row_release_reservation(fork);
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 7));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(fork));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 7));
  fork = loom_serve_text_model_row(model, 2);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_reserve(
      fork, LOOM_SERVE_TEXT_RESERVE_CHECKPOINT, checkpoint, 256, &admitted));
  if (!admitted) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "second fork reservation refused");
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 2, 1, input + 127));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 3, 128, input));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 2, 3));
  // Reset is also a complete reservation owner; it must release the weight pin.
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(fork));
  loom_serve_text_checkpoint_release(checkpoint);
  for (iree_host_size_t i = 0; i < 8; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(model);
  if (pool.available != pool.capacity || pool.pending ||
      loom_serve_text_model_memory_statistics(model).committed_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "request reservations leaked retained backing");
  }
  uint64_t empty_bytes = 0, short_bytes = 0, long_bytes = 0;
  IREE_RETURN_IF_ERROR(
      qwen_check_reserved_relocation(model, input, 0, 512, &empty_bytes));
  IREE_RETURN_IF_ERROR(
      qwen_check_reserved_relocation(model, input, 127, 192, &short_bytes));
  IREE_RETURN_IF_ERROR(
      qwen_check_reserved_relocation(model, input, 127, 512, &long_bytes));
  if (empty_bytes || short_bytes != long_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "compaction copied unconsumed completion credit");
  }
  if (loom_serve_text_model_weight_statistics(model).reserved_bytes) {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_deactivate(model));
  }
  fprintf(stderr,
          "PASS: request reservation, allocation-free advancement, "
          "unadvanced fork release and full reclamation.\n");
  return iree_ok_status();
}

static iree_status_t qwen_check_checkpoints(loom_serve_text_model_t* model,
                                            iree_allocator_t allocator) {
  const loom_serve_text_pool_usage_t pool =
      loom_serve_text_model_pool_usage(model);
  if (pool.block_size != 64 || pool.capacity < 1024 ||
      loom_serve_text_model_context_capacity(model) < 512 ||
      loom_serve_text_model_shapes(model)[0].token_capacity < 8 ||
      loom_serve_text_model_shapes(model)[0].span_capacity < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "checkpoint witness needs paged context >=512 and shape >=8:2");
  }
  for (iree_host_size_t i = 0; i < 8; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  loom_serve_text_trim_result_t trimmed;
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  int32_t input[512], prompt[128], padding[32];
  iree_host_size_t prompt_count = 0, padding_count = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      loom_serve_text_model_tokenizer(model),
      IREE_SV(
          "<|im_start|>user\nReturn exactly: amber cedar maple raven copper "
          "hazel sparrow juniper bronze larch heron poplar crimson beech "
          "kestrel "
          "spruce.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(prompt, NULL, NULL,
                                       IREE_ARRAYSIZE(prompt)),
      allocator, &prompt_count));
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode(
      loom_serve_text_model_tokenizer(model), IREE_SV("Background context.\n"),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      iree_tokenizer_make_token_output(padding, NULL, NULL,
                                       IREE_ARRAYSIZE(padding)),
      allocator, &padding_count));
  if (!padding_count || prompt_count > 127) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid checkpoint fixture");
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(input); ++i) {
    input[i] = padding[i % padding_count];
  }
  memcpy(input + 127 - prompt_count, prompt, prompt_count * sizeof(*input));
  // Occupy low IDs first, so the shared prefix must relocate during trim.
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 0, 512, input));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 7, 127, input));
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 6, 127, input));
  loom_serve_text_row_t* source = loom_serve_text_model_row(model, 7);
  loom_serve_text_checkpoint_t* checkpoint = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(source, &checkpoint));
  if (!checkpoint) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "checkpoint witness requires checkpoint_capacity=1");
  }
  loom_serve_text_checkpoint_t* refused = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(source, &refused));
  if (refused) {
    loom_serve_text_checkpoint_release(refused);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint witness requires exactly one pin slot");
  }
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 7, 6));
  const iree_host_size_t before_fork =
      loom_serve_text_model_pool_usage(model).available;
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 1, checkpoint));
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 3, checkpoint));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 6));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 3, 6));
  if (loom_serve_text_row_pool_reclaimable(source) != 0 ||
      loom_serve_text_row_pool_reclaimable(
          loom_serve_text_model_row(model, 6)) != 2 * pool.block_size ||
      loom_serve_text_row_pool_growth(loom_serve_text_model_row(model, 1),
                                      134) != 2 * pool.block_size) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "shared admission queries miscounted unique pages or tail COW");
  }
  if (loom_serve_text_model_pool_usage(model).available != before_fork) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "fork copied immutable KV");
  }
  fprintf(stderr,
          "{\"event\":\"checkpoint_shared\",\"prefix_tokens\":127,"
          "\"branches\":2,\"added_kv_blocks\":0}\n");
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_reset(loom_serve_text_model_row(model, 0)));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_reset(source));
  // The pin alone must preserve this prefix while its physical pages move.
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_reset(loom_serve_text_model_row(model, 1)));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_reset(loom_serve_text_model_row(model, 3)));
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_memory_statistics(model).reserved_bytes &&
      !trimmed.moved_blocks) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "shared prefix did not relocate");
  }
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 1, checkpoint));
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 3, checkpoint));
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 2, checkpoint));
  const bool elastic =
      loom_serve_text_model_memory_statistics(model).reserved_bytes != 0;
  if (elastic) {
    // This row has never advanced: its effective recurrent state is the anchor.
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_suspend(loom_serve_text_model_row(model, 2)));
  }
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 5, 127, input));
  const loom_serve_text_span_t first[] = {{1, 4, input, 0},
                                          {3, 3, input + 1, 0}};
  const loom_serve_text_span_t second[] = {
      {1, 3, input + 4, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {3, 2, input + 7, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}};
  const uint32_t known_limits[] = {0, 0};
  const loom_serve_text_continuation_t known = {0, 2, second, known_limits};
  const iree_host_size_t before_append =
      loom_serve_text_model_pool_usage(model).available;
  if (loom_serve_text_mtp_from_flags()) {
    loom_serve_text_result_t results[2];
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(
        model, 0, 2, first, known_limits, &known, results));
  } else {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(model, 0, 2, first));
    IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(model, 0, 2, second));
  }
  if (before_append - loom_serve_text_model_pool_usage(model).available !=
      4 * pool.block_size) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "fork did not detach two partial tails");
  }
  fprintf(stderr,
          "{\"event\":\"checkpoint_detached\",\"partial_tails\":2,"
          "\"new_pages\":2}\n");
  for (iree_host_size_t i = 0; i < 2; ++i) {
    const iree_host_size_t reference = i ? 5 : 6;
    IREE_RETURN_IF_ERROR(qwen_check_prefill(
        model, reference, first[i].token_count, first[i].token_ids));
    IREE_RETURN_IF_ERROR(qwen_check_prefill(
        model, reference, second[i].token_count, second[i].token_ids));
    IREE_RETURN_IF_ERROR(
        qwen_check_endpoint(model, first[i].row_index, reference));
  }
  if (elastic) {
    // Every live writer has left the original recurrent slot; only the pin
    // now owns it. Maintenance must retain that anchor for the later rewind.
    IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
    bool resumed = false;
    IREE_RETURN_IF_ERROR(loom_serve_text_row_try_resume(
        loom_serve_text_model_row(model, 2), &resumed));
    if (!resumed) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "unadvanced fork resume refused");
    }
  }
  IREE_RETURN_IF_ERROR(qwen_check_prefill(model, 4, 127, input));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 2, 4));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_decode(loom_serve_text_model_row(model, 2)));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_decode(loom_serve_text_model_row(model, 4)));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 2, 4));
  // Rewind both divergent writers to the same still-immutable endpoint.
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_RETURN_IF_ERROR(
        qwen_check_restore(model, first[i].row_index, checkpoint));
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i ? 5 : 6)));
    IREE_RETURN_IF_ERROR(qwen_check_prefill(model, i ? 5 : 6, 127, input));
    IREE_RETURN_IF_ERROR(
        qwen_check_endpoint(model, first[i].row_index, i ? 5 : 6));
  }
  if (loom_serve_text_mtp_from_flags()) {
    int32_t anchors[2];
    loom_serve_text_span_t spans[2];
    loom_serve_text_span_t next[2];
    const uint32_t limits[] = {8, 8};
    for (iree_host_size_t i = 0; i < 2; ++i) {
      anchors[i] = loom_serve_text_row_token(
          loom_serve_text_model_row(model, first[i].row_index));
      spans[i] = (loom_serve_text_span_t){
          first[i].row_index, 4, &anchors[i],
          LOOM_SERVE_TEXT_SPAN_FLAG_SELECT | LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE};
      next[i] = spans[i];
      next[i].token_ids = NULL;
    }
    const loom_serve_text_continuation_t continuation = {0, 2, next, limits};
    loom_serve_text_result_t results[2];
    IREE_RETURN_IF_ERROR(loom_serve_text_model_verify(
        model, 0, 2, spans, limits, &continuation, results));
    for (iree_host_size_t i = 0; i < 2; ++i) {
      loom_serve_text_row_t* reference =
          loom_serve_text_model_row(model, i ? 5 : 6);
      for (iree_host_size_t j = 0; j < results[i].output_count; ++j) {
        IREE_RETURN_IF_ERROR(loom_serve_text_row_decode(reference));
        if (results[i].tokens[j] != loom_serve_text_row_token(reference)) {
          return iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "forked MTP differs from replay");
        }
      }
      IREE_RETURN_IF_ERROR(
          qwen_check_endpoint(model, spans[i].row_index, i ? 5 : 6));
      fprintf(stderr,
              "{\"event\":\"checkpoint_mtp\",\"row\":%zu,\"outputs\":%zu,"
              "\"epochs\":%zu}\n",
              spans[i].row_index, results[i].output_count,
              results[i].verification_count);
    }
  }
  loom_serve_text_checkpoint_release(checkpoint);
  checkpoint = NULL;
  // A short-lived pin released without rewind must not leave a hidden owner.
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(
      loom_serve_text_model_row(model, 1), &checkpoint));
  if (!checkpoint) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "pin capacity leaked");
  }
  loom_serve_text_checkpoint_release(checkpoint);
  checkpoint = NULL;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_decode(loom_serve_text_model_row(model, 1)));
  IREE_RETURN_IF_ERROR(
      loom_serve_text_row_decode(loom_serve_text_model_row(model, 6)));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 1, 6));
  // An unselected endpoint has no recoverable prediction, but known-input
  // continuation can fork it without evaluating its prefix again.
  const loom_serve_text_span_t hidden = {7, 3, input, 0};
  IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(model, 0, 1, &hidden));
  IREE_RETURN_IF_ERROR(loom_serve_text_row_try_pin(source, &checkpoint));
  if (!checkpoint) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "unselected pin refused");
  }
  IREE_RETURN_IF_ERROR(qwen_check_restore(model, 2, checkpoint));
  IREE_RETURN_IF_ERROR(qwen_check_error(
      loom_serve_text_row_decode(loom_serve_text_model_row(model, 2)),
      IREE_STATUS_FAILED_PRECONDITION));
  // Both remaining readers advance together after the explicit pin is gone.
  // Releasing the first reader during cohort assembly would reuse live state.
  loom_serve_text_checkpoint_release(checkpoint);
  checkpoint = NULL;
  const loom_serve_text_span_t visible[] = {
      {7, 1, input + 3, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT},
      {2, 1, input + 3, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT}};
  IREE_RETURN_IF_ERROR(loom_serve_text_model_epoch(model, 0, 2, visible));
  IREE_RETURN_IF_ERROR(qwen_check_endpoint(model, 2, 7));
  for (iree_host_size_t i = 0; i < 8; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_serve_text_row_reset(loom_serve_text_model_row(model, i)));
  }
  IREE_RETURN_IF_ERROR(loom_serve_text_model_trim(model, &trimmed));
  if (loom_serve_text_model_pool_usage(model).available != pool.capacity ||
      loom_serve_text_model_memory_statistics(model).committed_bytes) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "checkpoint lifecycle leaked retained backing");
  }
  fprintf(stderr,
          "PASS: shared prefix, divergent forks, queued continuation, rewind, "
          "pin release and full reclamation.\n");
  IREE_RETURN_IF_ERROR(qwen_check_reservations(model, input));
  return FLAG_suspend_checkpoints
             ? qwen_check_checkpoint_suspension(model, input)
             : iree_ok_status();
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
  loom_serve_device_t* device = NULL;
  iree_status_t status =
      loom_serve_device_create_from_flags(&device, allocator);
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_model_create_from_flags(device, &defaults, &model,
                                                     allocator);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_memory_statistics_t memory =
        loom_serve_text_model_memory_statistics(model);
    fprintf(stderr,
            "{\"event\":\"memory_created\",\"reserved_bytes\":%" PRIu64
            ",\"committed_bytes\":%" PRIu64 "}\n",
            memory.reserved_bytes, memory.committed_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = FLAG_compare[0] ? qwen_check_compare(model, allocator)
                             : qwen_check_run(model, allocator);
  }
  if (iree_status_is_ok(status) && !FLAG_compare[0]) {
    status = qwen_check_mtp_verification(model, allocator);
  }
  if (iree_status_is_ok(status) &&
      (FLAG_trim || FLAG_reload_weights || FLAG_suspend_rows)) {
    status = qwen_check_trim(model, allocator);
  }
  if (iree_status_is_ok(status) && FLAG_shared_residency) {
    status = qwen_check_shared_residency(device, model, allocator);
  }
  if (iree_status_is_ok(status) &&
      (FLAG_checkpoints || FLAG_suspend_checkpoints)) {
    status = qwen_check_checkpoints(model, allocator);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_memory_statistics_t memory =
        loom_serve_text_model_memory_statistics(model);
    fprintf(stderr,
            "{\"event\":\"memory_completed\",\"reserved_bytes\":%" PRIu64
            ",\"committed_bytes\":%" PRIu64 ",\"peak_bytes\":%" PRIu64 "}\n",
            memory.reserved_bytes, memory.committed_bytes, memory.peak_bytes);
  }
  status = iree_status_join(status, loom_serve_text_model_destroy(model));
  status = iree_status_join(status, loom_serve_device_destroy(device));
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
