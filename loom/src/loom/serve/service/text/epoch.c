// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch.h"

#include <string.h>

// Only the leased first cohort can contribute to its device-fed continuation.
// Known progress is exact; the device resolves speculative acceptance and EOS.
static void loom_serve_text_epoch_prepare_continuation(
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_row_t* rows, iree_host_size_t count,
    const loom_serve_text_span_t* spans, const uint32_t* output_limits,
    loom_serve_text_span_t* next_spans, uint32_t* next_limits,
    loom_serve_text_continuation_t* out_continuation) {
  *out_continuation = (loom_serve_text_continuation_t){0};
  if (options->continuation_epochs != 2) {
    return;
  }
  const iree_host_size_t width = options->draft_depth + 1;
  loom_serve_ready_span_t ready[LOOM_SERVE_TEXT_ROW_CAPACITY] = {0};
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    const loom_serve_text_epoch_row_t* row = &rows[span->row_index];
    const iree_host_size_t remaining_output = row->output_count;
    const iree_host_size_t remaining_input =
        row->input_count ? row->input_count - span->token_count : 0;
    if (remaining_input) {
      ready[i] = (loom_serve_ready_span_t){remaining_input, 1};
    } else {
      const iree_host_size_t minimum_consumed =
          output_limits[i] ? 1 : span->token_count;
      // The first selected result consumes at least one output. Another
      // verifier needs at least two left, as in ordinary request scheduling.
      if (options->draft_depth && remaining_output > 2 &&
          options->context_capacity - row->position >=
              minimum_consumed + width) {
        ready[i] = (loom_serve_ready_span_t){width, width};
      }
    }
  }
  // Rotate within the leased cohort instead of repeatedly giving its first
  // long prompt all of the expansion slots.
  iree_host_size_t cursor = count > 1 ? 1 : 0;
  if (options->packing_mode == LOOM_SERVE_TEXT_PACKING_SEPARATE) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      const iree_host_size_t first = (cursor + i) % count;
      if (!ready[first].token_count) {
        continue;
      }
      const iree_host_size_t minimum = ready[first].minimum_count;
      for (iree_host_size_t j = 0; j < count; ++j) {
        if (ready[j].minimum_count != minimum) {
          ready[j] = (loom_serve_ready_span_t){0};
        }
      }
      break;
    }
  }
  loom_serve_packed_span_t scheduled[LOOM_SERVE_TEXT_ROW_CAPACITY];
  loom_serve_packed_span_t scratch[LOOM_SERVE_TEXT_ROW_CAPACITY];
  iree_host_size_t shape = 0;
  const iree_host_size_t next_count = loom_serve_pack_shapes(
      count, ready, options->shape_count, options->shapes, options->chunk_size,
      &cursor, scheduled, scratch, &shape);
  for (iree_host_size_t i = 0; i < next_count; ++i) {
    const iree_host_size_t first = scheduled[i].row_index;
    const iree_host_size_t row = spans[first].row_index;
    const bool speculative = ready[first].minimum_count > 1;
    const bool select =
        speculative || scheduled[i].token_count == ready[first].token_count;
    next_spans[i] = (loom_serve_text_span_t){
        .row_index = row,
        .token_count = scheduled[i].token_count,
        .token_ids = speculative
                         ? NULL
                         : rows[row].input_tokens + spans[first].token_count,
        .flags = (select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0) |
                 (speculative ? LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE : 0),
    };
    next_limits[i] =
        speculative
            ? (uint32_t)iree_min(output_limits[first] ? 2 * width : width + 1,
                                 rows[row].output_count)
            : 0;
  }
  *out_continuation = (loom_serve_text_continuation_t){
      .shape_index = shape,
      .span_count = next_count,
      .spans = next_spans,
      .output_limits = next_limits,
  };
}

void loom_serve_text_epoch_prepare(
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_row_t* rows, iree_host_size_t shape_index,
    iree_host_size_t count, const loom_serve_packed_span_t* scheduled,
    loom_serve_text_epoch_t* epoch,
    loom_serve_text_epoch_continuation_t* continuation) {
  memset(epoch, 0, sizeof(*epoch));
  epoch->count = count;
  epoch->shape_index = shape_index;
  loom_serve_text_span_t* spans = epoch->spans;
  int32_t* decode_tokens = epoch->decode_tokens;
  iree_host_size_t* positions = epoch->positions;
  uint32_t* output_limits = epoch->output_limits;
  iree_host_size_t proposal_count = 0;
  iree_host_size_t input_count = 0;
  const iree_host_size_t width = options->draft_depth + 1;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t row = scheduled[i].row_index;
    const loom_serve_text_epoch_row_t* input = &rows[row];
    const bool prefill = input->input_count != 0;
    epoch->prefill[i] = prefill;
    const bool select =
        !prefill || scheduled[i].token_count == input->input_count;
    positions[i] = input->position;
    if (!prefill) {
      decode_tokens[i] = input->token;
      if (options->draft_depth && scheduled[i].token_count == width) {
        ++proposal_count;
        output_limits[i] = (uint32_t)iree_min(
            width * options->continuation_epochs, input->output_count);
      }
    }
    spans[i] = (loom_serve_text_span_t){
        .row_index = row,
        .token_count = scheduled[i].token_count,
        .token_ids = prefill ? input->input_tokens : &decode_tokens[i],
        .flags = (select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0) |
                 (output_limits[i] ? LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE : 0),
    };
    input_count += spans[i].token_count;
  }
  loom_serve_text_epoch_prepare_continuation(
      options, rows, count, spans, output_limits, continuation->spans,
      continuation->output_limits, &continuation->continuation);
  if (!continuation->continuation.span_count) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      output_limits[i] = (uint32_t)iree_min(output_limits[i], width);
    }
  }
  iree_host_size_t continued_prefill_count = 0;
  for (iree_host_size_t i = 0; i < continuation->continuation.span_count; ++i) {
    continued_prefill_count +=
        continuation->output_limits[i] ? 0 : continuation->spans[i].token_count;
  }
  epoch->input_count = input_count;
  epoch->proposal_count = proposal_count;
  epoch->device_epochs = continuation->continuation.span_count ? 2 : 1;
  if (options->schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    epoch->token_slots = options->shapes[shape_index].token_capacity;
    if (continuation->continuation.span_count) {
      epoch->token_slots +=
          options->shapes[continuation->continuation.shape_index]
              .token_capacity;
    }
  }
  epoch->continued_prefill_count = continued_prefill_count;
}

iree_status_t loom_serve_text_epoch_submit(
    loom_serve_text_model_t* model, loom_serve_text_schedule_mode_t mode,
    const loom_serve_text_epoch_t* epoch,
    const loom_serve_text_epoch_continuation_t* continuation,
    loom_serve_execution_frontier_t* out_frontier) {
  if (mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    const bool verifies = epoch->proposal_count || epoch->device_epochs == 2;
    return loom_serve_text_model_submit_epoch(
        model, epoch->shape_index, epoch->count, epoch->spans,
        verifies ? epoch->output_limits : NULL,
        continuation->continuation.span_count ? &continuation->continuation
                                              : NULL,
        out_frontier);
  }
  const loom_serve_text_span_t* span = &epoch->spans[epoch->cursor];
  loom_serve_text_row_t* row =
      loom_serve_text_model_row(model, span->row_index);
  if (epoch->prefill[epoch->cursor] ||
      mode == LOOM_SERVE_TEXT_SCHEDULE_MATCHED) {
    return loom_serve_text_row_submit_prefill(row, span->token_count,
                                              span->token_ids, out_frontier);
  }
  return loom_serve_text_row_submit_decode(row, out_frontier);
}

iree_status_t loom_serve_text_epoch_complete(
    loom_serve_text_model_t* model, loom_serve_text_schedule_mode_t mode,
    loom_serve_text_epoch_t* epoch) {
  if (mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    return loom_serve_text_model_complete_epoch(model, epoch->results);
  }
  const loom_serve_text_span_t* span = &epoch->spans[epoch->cursor];
  loom_serve_text_row_t* row =
      loom_serve_text_model_row(model, span->row_index);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_complete_step(row));
  loom_serve_text_result_t* result = &epoch->results[epoch->cursor];
  result->consumed_count = span->token_count;
  result->known_count = span->token_count;
  if (iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT)) {
    result->output_count = 1;
    result->tokens[0] = loom_serve_text_row_token(row);
  }
  ++epoch->cursor;
  return iree_ok_status();
}
