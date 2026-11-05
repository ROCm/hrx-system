// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/epoch_plan.h"

#include <string.h>

#include "loom/serve/service/text/state.h"

// The semantic host record is {length, position, source_position, row,
// input_begin, input_count, flags, output_credit, first_span}. Known IDs are
// copied once into a contiguous host stream. Source owns all device packet
// offsets, logical positions and selection policy.
iree_host_size_t loom_serve_text_epoch_plan_write(
    loom_serve_text_state_row_t* const* rows, iree_byte_span_t plan,
    iree_byte_span_t input, iree_host_size_t span_count,
    const loom_serve_text_span_t* spans, const uint32_t* output_limits,
    const uint32_t* first_indices, iree_host_size_t record_begin,
    iree_host_size_t input_begin) {
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_serve_text_span_t* span = &spans[i];
    const bool generates =
        iree_any_bit_set(span->flags, LOOM_SERVICE_TEXT_SPAN_FLAG_PROPOSE);
    const iree_host_size_t input_count =
        span->token_ids && !(first_indices && generates)
            ? (generates ? 1 : span->token_count)
            : 0;
    const loom_serve_text_state_row_t* row = rows[span->row_index];
    const uint32_t fields[TEXT_PLAN_FIELD_COUNT] = {
        [TEXT_PLAN_FIELD_LENGTH] = (uint32_t)span->token_count,
        [TEXT_PLAN_FIELD_POSITION] = row->position,
        [TEXT_PLAN_FIELD_SOURCE_POSITION] = row->source_position,
        [TEXT_PLAN_FIELD_ROW] = (uint32_t)span->row_index,
        [TEXT_PLAN_FIELD_INPUT_BEGIN] = (uint32_t)input_begin,
        [TEXT_PLAN_FIELD_INPUT_COUNT] = (uint32_t)input_count,
        [TEXT_PLAN_FIELD_FLAGS] = span->flags,
        [TEXT_PLAN_FIELD_OUTPUT_LIMIT] = output_limits ? output_limits[i] : 0,
        [TEXT_PLAN_FIELD_FIRST_SPAN] =
            first_indices ? first_indices[i] : (uint32_t)i,
    };
    uint8_t* record = plan.data + (record_begin + i) * sizeof(fields);
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(fields); ++j) {
      iree_unaligned_store_le_u32(record + j * sizeof(uint32_t), fields[j]);
    }
    if (input_count) {
      memcpy(input.data + input_begin * 4, span->token_ids,
             input_count * sizeof(int32_t));
    }
    input_begin += input_count;
  }
  return input_begin;
}

void loom_serve_text_epoch_plan_read_progress(
    const uint8_t* record, loom_serve_text_result_t* out_result,
    uint32_t* out_source_position) {
  *out_result = (loom_serve_text_result_t){
      .consumed_count = iree_unaligned_load_le_u32(
          record + TEXT_PROGRESS_FIELD_CONSUMED_COUNT * sizeof(int32_t)),
      .known_count = iree_unaligned_load_le_u32(
          record + TEXT_PROGRESS_FIELD_KNOWN_COUNT * sizeof(int32_t)),
      .output_count = iree_unaligned_load_le_u32(
          record + TEXT_PROGRESS_FIELD_OUTPUT_COUNT * sizeof(int32_t)),
      .verification_count = iree_unaligned_load_le_u32(
          record + TEXT_PROGRESS_FIELD_VERIFICATION_COUNT * sizeof(int32_t)),
  };
  *out_source_position = iree_unaligned_load_le_u32(
      record + TEXT_PROGRESS_FIELD_SOURCE_POSITION * sizeof(int32_t));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(out_result->tokens); ++i) {
    out_result->tokens[i] = (int32_t)iree_unaligned_load_le_u32(
        record + (TEXT_PROGRESS_FIELD_TOKENS + i) * sizeof(int32_t));
  }
}
