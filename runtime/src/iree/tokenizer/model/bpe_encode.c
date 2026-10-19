// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE complete-segment encoding. Partial spans remain caller-owned until the
// segmenter establishes a boundary; output backpressure retains computed
// tokens.

#include "iree/tokenizer/model/bpe_state.h"

static bool iree_tokenizer_bpe_emit_results(
    iree_tokenizer_bpe_state_t* state, const iree_tokenizer_bpe_model_t* model,
    iree_tokenizer_bpe_output_cursor_t* cursor) {
  if (state->phase == IREE_TOKENIZER_BPE_PHASE_BACKTRACK_EMIT) {
    iree_tokenizer_bpe_backtrack_entry_t* stack =
        iree_tokenizer_bpe_state_backtrack_stack(state, model);
    while (state->backtrack.emit_index < state->backtrack.stack_count) {
      iree_tokenizer_bpe_backtrack_entry_t entry =
          stack[state->backtrack.emit_index];
      if (!iree_tokenizer_bpe_emit(cursor, entry.token_id, entry.start_byte,
                                   entry.end_byte)) {
        return false;
      }
      ++state->backtrack.emit_index;
    }
  } else if (state->phase == IREE_TOKENIZER_BPE_PHASE_SYMBOL_EMIT) {
    while (state->symbol_emit_index != UINT32_MAX) {
      iree_tokenizer_bpe_symbol_t* symbol =
          &state->symbols[state->symbol_emit_index];
      if (!iree_tokenizer_bpe_emit(cursor, symbol->token_id, symbol->start_byte,
                                   symbol->end_byte)) {
        return false;
      }
      state->symbol_emit_index = symbol->next;
    }
  }
  state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
  return true;
}

static iree_status_t iree_tokenizer_bpe_encode_segment(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment, iree_tokenizer_token_id_t* out_tokens,
    iree_tokenizer_offset_t* out_offsets, iree_host_size_t max_tokens,
    iree_host_size_t segment_base_offset, bool is_partial,
    iree_host_size_t* out_token_count, bool* out_segment_complete) {
  *out_token_count = 0;
  *out_segment_complete = false;
  if (is_partial) {
    return iree_ok_status();
  }
  iree_tokenizer_bpe_output_cursor_t cursor =
      iree_tokenizer_bpe_output_cursor_make(out_tokens, out_offsets,
                                            segment_base_offset, max_tokens);
  if (state->phase == IREE_TOKENIZER_BPE_PHASE_SEGMENT_START) {
    state->segment.base_offset = segment_base_offset;
    if (!segment.size) {
      *out_segment_complete = true;
      return iree_ok_status();
    }
    const bool ignore_merges =
        iree_any_bit_set(model->flags, IREE_TOKENIZER_BPE_FLAG_IGNORE_MERGES);
    if (!model->end_of_word_suffix_length || ignore_merges) {
      int32_t token = -1;
      iree_host_size_t length = 0;
      iree_tokenizer_bpe_trie_longest_match_byte_level(model, segment, &token,
                                                       &length);
      if (token >= 0 && length == segment.size &&
          (ignore_merges || iree_tokenizer_bpe_is_first_token_reachable(
                                model, (uint32_t)token))) {
        // The caller only enters with output credit.
        iree_tokenizer_bpe_emit(&cursor, token, 0, (uint32_t)segment.size);
        *out_token_count = 1;
        *out_segment_complete = true;
        return iree_ok_status();
      }
    }
    if (segment.size <= model->max_backtrack_segment_bytes &&
        !model->end_of_word_suffix_length &&
        iree_tokenizer_bpe_backtrack_encode(
            model, state, (const uint8_t*)segment.data, segment.size)) {
      iree_tokenizer_bpe_cache_populate(
          model, state, segment,
          iree_tokenizer_bpe_state_backtrack_stack(state, model),
          state->backtrack.stack_count);
      state->phase = IREE_TOKENIZER_BPE_PHASE_BACKTRACK_EMIT;
    } else {
      IREE_RETURN_IF_ERROR(
          iree_tokenizer_bpe_encode_symbols(model, state, segment));
      state->phase = IREE_TOKENIZER_BPE_PHASE_SYMBOL_EMIT;
    }
  }
  *out_segment_complete =
      iree_tokenizer_bpe_emit_results(state, model, &cursor);
  *out_token_count =
      iree_tokenizer_bpe_output_cursor_count(&cursor, out_tokens);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BPE State Encode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_bpe_state_encode(
    iree_tokenizer_model_state_t* state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)state->model;

  *out_segments_consumed = 0;
  *out_token_count = 0;

  iree_host_size_t total_tokens_written = 0;
  iree_host_size_t remaining_capacity = output.capacity;
  iree_tokenizer_token_id_t* current_token_output = output.token_ids;
  iree_tokenizer_offset_t* current_offset_output = output.token_offsets;
  iree_status_t status = iree_ok_status();

  // Process segments from the start of the provided list.
  // The caller is responsible for slicing the segment list when resuming
  // after a partial return - we always start from index 0.
  // Computed tokens and the output cursor are retained across calls.
  for (iree_host_size_t i = 0; i < segments.count && iree_status_is_ok(status);
       ++i) {
    // Early exit when output buffer is full - no point processing more
    // segments.
    if (remaining_capacity == 0) {
      break;
    }

    const iree_tokenizer_segment_t* segment = &segments.values[i];

    if (segment->end > transform_buffer.data_length ||
        segment->start > segment->end) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "segment bounds out of range: [%" PRIhsz
                                ", %" PRIhsz ") in buffer of %" PRIhsz " bytes",
                                segment->start, segment->end,
                                transform_buffer.data_length);
      break;
    }

    iree_string_view_t segment_text = iree_make_string_view(
        (const char*)transform_buffer.data + segment->start,
        segment->end - segment->start);

    bool is_partial = segments.last_is_partial && (i == segments.count - 1);

    // Inline cache fast-path: check the word cache before entering the BPE
    // state machine. On hit (60-68% of segments for English text), this
    // skips the bpe_encode_segment function call, phase dispatch, cursor
    // construction, and all fast-path/backtrack setup.
    //
    // Guards: state machine must be ready for a new segment, the segment
    // must be complete (not partial), and no offset tracking (cache doesn't
    // store per-token byte positions).
    if (!is_partial &&
        bpe_state->phase == IREE_TOKENIZER_BPE_PHASE_SEGMENT_START &&
        !current_offset_output && segment_text.size > 0) {
      iree_tokenizer_bpe_output_cursor_t cache_cursor =
          iree_tokenizer_bpe_output_cursor_make(
              current_token_output, NULL, segment->start, remaining_capacity);
      if (iree_tokenizer_bpe_cache_lookup(model, bpe_state, segment_text,
                                          &cache_cursor)) {
        iree_host_size_t count = iree_tokenizer_bpe_output_cursor_count(
            &cache_cursor, current_token_output);
        total_tokens_written += count;
        remaining_capacity -= count;
        current_token_output += count;
        (*out_segments_consumed)++;
        continue;
      }
    }

    iree_host_size_t tokens_for_segment = 0;
    bool segment_complete = false;
    status = iree_tokenizer_bpe_encode_segment(
        model, bpe_state, segment_text, current_token_output,
        current_offset_output, remaining_capacity, segment->start, is_partial,
        &tokens_for_segment, &segment_complete);

    total_tokens_written += tokens_for_segment;
    remaining_capacity -= tokens_for_segment;
    current_token_output += tokens_for_segment;
    if (current_offset_output) {
      current_offset_output += tokens_for_segment;
    }

    if (segment_complete) {
      (*out_segments_consumed)++;
    } else {
      // Output full mid-segment - stop processing.
      // State preserves position for next call.
      break;
    }
  }

  *out_token_count = total_tokens_written;
  return status;
}

iree_status_t iree_tokenizer_bpe_state_finalize(
    iree_tokenizer_model_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)state->model;
  iree_tokenizer_bpe_output_cursor_t cursor =
      iree_tokenizer_bpe_output_cursor_make(
          output.token_ids, output.token_offsets,
          bpe_state->segment.base_offset, output.capacity);
  iree_tokenizer_bpe_emit_results(bpe_state, model, &cursor);
  *out_token_count =
      iree_tokenizer_bpe_output_cursor_count(&cursor, output.token_ids);
  return iree_ok_status();
}

bool iree_tokenizer_bpe_state_has_pending(
    const iree_tokenizer_model_state_t* state) {
  const iree_tokenizer_bpe_state_t* bpe_state =
      (const iree_tokenizer_bpe_state_t*)state;
  return bpe_state->phase != IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
}

iree_host_size_t iree_tokenizer_bpe_state_reclaim(
    iree_tokenizer_model_state_t* state) {
  // Incomplete input stays with the caller until its segment is complete.
  return 0;
}
