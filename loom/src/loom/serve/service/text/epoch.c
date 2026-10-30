// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch.h"

#include <string.h>

iree_status_t loom_serve_text_epoch_initialize(
    iree_host_size_t span_capacity, iree_allocator_t host_allocator,
    loom_serve_text_epoch_t* out_epoch) {
  *out_epoch = (loom_serve_text_epoch_t){.host_allocator = host_allocator};
  iree_host_size_t spans = 0, results = 0, tokens = 0, positions = 0;
  iree_host_size_t limits = 0, prefill = 0, rows = 0, ready = 0;
  iree_host_size_t scheduled = 0, candidate = 0, next_spans = 0,
                   next_limits = 0;
  iree_host_size_t total = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total,
      IREE_STRUCT_FIELD(span_capacity, loom_serve_text_span_t, &spans),
      IREE_STRUCT_FIELD(span_capacity, loom_serve_text_result_t, &results),
      IREE_STRUCT_FIELD_ALIGNED(span_capacity, iree_host_size_t,
                                iree_alignof(iree_host_size_t), &positions),
      IREE_STRUCT_FIELD(span_capacity, loom_serve_text_epoch_row_t, &rows),
      IREE_STRUCT_FIELD(span_capacity, loom_serve_ready_span_t, &ready),
      IREE_STRUCT_FIELD(span_capacity, loom_serve_packed_span_t, &scheduled),
      IREE_STRUCT_FIELD(span_capacity, loom_serve_packed_span_t, &candidate),
      IREE_STRUCT_FIELD(span_capacity, loom_serve_text_span_t, &next_spans),
      IREE_STRUCT_FIELD(span_capacity, int32_t, &tokens),
      IREE_STRUCT_FIELD(span_capacity, uint32_t, &limits),
      IREE_STRUCT_FIELD(span_capacity, uint32_t, &next_limits),
      IREE_STRUCT_FIELD(span_capacity, bool, &prefill)));
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total, (void**)&storage));
  out_epoch->storage = storage;
  out_epoch->spans = (loom_serve_text_span_t*)(storage + spans);
  out_epoch->results = (loom_serve_text_result_t*)(storage + results);
  out_epoch->decode_tokens = (int32_t*)(storage + tokens);
  out_epoch->positions = (iree_host_size_t*)(storage + positions);
  out_epoch->output_limits = (uint32_t*)(storage + limits);
  out_epoch->prefill = (bool*)(storage + prefill);
  out_epoch->planning.rows = (loom_serve_text_epoch_row_t*)(storage + rows);
  out_epoch->planning.ready = (loom_serve_ready_span_t*)(storage + ready);
  out_epoch->planning.scheduled =
      (loom_serve_packed_span_t*)(storage + scheduled);
  out_epoch->planning.candidate =
      (loom_serve_packed_span_t*)(storage + candidate);
  out_epoch->planning.spans = (loom_serve_text_span_t*)(storage + next_spans);
  out_epoch->planning.output_limits = (uint32_t*)(storage + next_limits);
  return iree_ok_status();
}

void loom_serve_text_epoch_deinitialize(loom_serve_text_epoch_t* epoch) {
  iree_allocator_free(epoch->host_allocator, epoch->storage);
  memset(epoch, 0, sizeof(*epoch));
}

// Only the leased first cohort can contribute to its device-fed continuation.
// Known progress is exact; the device resolves speculative acceptance and EOS.
static iree_host_size_t loom_serve_text_epoch_prepare_continuation(
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_row_t* rows, iree_host_size_t row_count,
    const loom_serve_packing_plan_t* plan, loom_serve_text_epoch_t* epoch,
    loom_serve_text_continuation_t* out_continuation) {
  *out_continuation = (loom_serve_text_continuation_t){0};
  if (options->continuation_epochs != 2) {
    return plan->next_row;
  }
  const iree_host_size_t count = epoch->count;
  const loom_serve_text_span_t* spans = epoch->spans;
  const uint32_t* output_limits = epoch->output_limits;
  loom_serve_text_span_t* next_spans = epoch->planning.spans;
  uint32_t* next_limits = epoch->planning.output_limits;
  const iree_host_size_t width = options->draft_depth + 1;
  loom_serve_ready_span_t* ready = epoch->planning.ready;
  memset(ready, 0, count * sizeof(*ready));
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    const loom_serve_text_epoch_row_t* row = &rows[i];
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
  const iree_host_size_t cursor = plan->next_span;
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
  loom_serve_packed_span_t* scheduled = epoch->planning.scheduled;
  loom_serve_packed_span_t* scratch = epoch->planning.candidate;
  const loom_serve_packing_plan_t next = loom_serve_pack_shapes(
      count, ready, options->shape_count, options->shapes, options->chunk_size,
      cursor, scheduled, scratch);
  const iree_host_size_t next_count = next.span_count;
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
                         : rows[first].input_tokens + spans[first].token_count,
        .flags = (select ? LOOM_SERVE_TEXT_SPAN_FLAG_SELECT : 0) |
                 (speculative ? LOOM_SERVE_TEXT_SPAN_FLAG_PROPOSE : 0),
        .input_offset =
            speculative ? 0
                        : rows[first].input_offset + spans[first].token_count,
        .media = speculative ? NULL : rows[first].media,
    };
    next_limits[i] =
        speculative
            ? (uint32_t)iree_min(output_limits[first] ? 2 * width : width + 1,
                                 rows[first].output_count)
            : 0;
  }
  *out_continuation = (loom_serve_text_continuation_t){
      .shape_index = next.shape_index,
      .span_count = next_count,
      .spans = next_spans,
      .output_limits = next_limits,
  };
  if (!next_count || plan->deferred) {
    return plan->next_row;
  }
  if (next.deferred) {
    return spans[next.next_row].row_index;
  }
  // The continuation's row domain is the first cohort. Advance after its
  // rotation owner in the full resident domain, preserving newly ready gaps.
  const iree_host_size_t previous =
      next.next_row ? next.next_row - 1 : count - 1;
  const iree_host_size_t following = spans[previous].row_index + 1;
  return following == row_count ? 0 : following;
}

iree_host_size_t loom_serve_text_epoch_prepare(
    const loom_serve_text_epoch_options_t* options,
    const loom_serve_text_epoch_row_t* rows, iree_host_size_t row_count,
    const loom_serve_packing_plan_t* plan,
    const loom_serve_packed_span_t* scheduled, loom_serve_text_epoch_t* epoch,
    loom_serve_text_continuation_t* continuation) {
  const iree_host_size_t count = plan->span_count;
  epoch->count = count;
  epoch->shape_index = plan->shape_index;
  epoch->cursor = 0;
  epoch->start.serial = 0;
  epoch->start.time = 0;
  epoch->token_slots = 0;
  memset(epoch->results, 0, count * sizeof(*epoch->results));
  memset(epoch->output_limits, 0, count * sizeof(*epoch->output_limits));
  loom_serve_text_span_t* spans = epoch->spans;
  int32_t* decode_tokens = epoch->decode_tokens;
  iree_host_size_t* positions = epoch->positions;
  uint32_t* output_limits = epoch->output_limits;
  iree_host_size_t proposal_count = 0;
  const iree_host_size_t width = options->draft_depth + 1;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t row = scheduled[i].row_index;
    const loom_serve_text_epoch_row_t* input = &rows[i];
    const bool prefill = input->input_count != 0;
    epoch->prefill[i] = prefill;
    const bool select =
        options->model_mode == LOOM_SERVE_TEXT_MODE_GENERATE &&
        (!prefill || scheduled[i].token_count == input->input_count);
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
        .input_offset = prefill ? input->input_offset : 0,
        .media = prefill ? input->media : NULL,
    };
  }
  const iree_host_size_t next_row = loom_serve_text_epoch_prepare_continuation(
      options, rows, row_count, plan, epoch, continuation);
  if (!continuation->span_count) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      output_limits[i] = (uint32_t)iree_min(output_limits[i], width);
    }
  }
  iree_host_size_t continued_prefill_count = 0;
  for (iree_host_size_t i = 0; i < continuation->span_count; ++i) {
    continued_prefill_count +=
        continuation->output_limits[i] ? 0 : continuation->spans[i].token_count;
  }
  epoch->input_count = plan->token_count;
  epoch->proposal_count = proposal_count;
  epoch->device_epochs = continuation->span_count ? 2 : 1;
  if (options->schedule_mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    epoch->token_slots = options->shapes[plan->shape_index].token_capacity;
    if (continuation->span_count) {
      epoch->token_slots +=
          options->shapes[continuation->shape_index].token_capacity;
    }
  }
  epoch->continued_prefill_count = continued_prefill_count;
  return next_row;
}

iree_status_t loom_serve_text_epoch_submit(
    loom_serve_text_model_t* model, loom_serve_text_schedule_mode_t mode,
    const loom_serve_text_epoch_t* epoch,
    const loom_serve_text_continuation_t* continuation,
    loom_serve_text_capture_t** out_capture,
    loom_serve_execution_frontier_t* out_frontier) {
  if (mode == LOOM_SERVE_TEXT_SCHEDULE_PACKED) {
    const bool verifies = epoch->proposal_count || epoch->device_epochs == 2;
    return loom_serve_text_model_submit_epoch(
        model, epoch->shape_index, epoch->count, epoch->spans,
        verifies ? epoch->output_limits : NULL,
        continuation->span_count ? continuation : NULL, out_capture,
        out_frontier);
  }
  const loom_serve_text_span_t* span = &epoch->spans[epoch->cursor];
  if (span->media) {
    return loom_serve_text_model_submit_epoch(
        model, epoch->shape_index, 1, span, NULL, NULL, NULL, out_frontier);
  }
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
  if (span->media) {
    IREE_RETURN_IF_ERROR(loom_serve_text_model_complete_epoch(
        model, &epoch->results[epoch->cursor]));
    ++epoch->cursor;
    return iree_ok_status();
  }
  loom_serve_text_row_t* row =
      loom_serve_text_model_row(model, span->row_index);
  IREE_RETURN_IF_ERROR(loom_serve_text_row_complete_step(row));
  loom_serve_text_result_t* result = &epoch->results[epoch->cursor];
  result->consumed_count = (uint32_t)span->token_count;
  result->known_count = (uint32_t)span->token_count;
  if (iree_any_bit_set(span->flags, LOOM_SERVE_TEXT_SPAN_FLAG_SELECT)) {
    result->output_count = 1;
    result->tokens[0] = loom_serve_text_row_token(row);
  }
  ++epoch->cursor;
  return iree_ok_status();
}
