// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/regex_replace.h"

#include <string.h>

#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/regex/exec.h"

// Normalizer struct. Content data follows immediately after this struct.
typedef struct iree_tokenizer_normalizer_regex_replace_t {
  iree_tokenizer_normalizer_t base;
  iree_allocator_t allocator;
  iree_tokenizer_regex_dfa_t dfa;
  iree_tokenizer_regex_stride_t* stride;
  uint8_t* dfa_data;
  uint8_t content_length;
  // Trailing data: [content_data]
} iree_tokenizer_normalizer_regex_replace_t;

// Matching continues the streaming state until output credit is exhausted.
// Re-entry then replays only the accepted suffix after committed output.
typedef enum iree_tokenizer_regex_replace_phase_e {
  IREE_TOKENIZER_REGEX_REPLACE_CONTINUE = 0,
  IREE_TOKENIZER_REGEX_REPLACE_REPLAY,
  IREE_TOKENIZER_REGEX_REPLACE_DONE,
} iree_tokenizer_regex_replace_phase_t;

// State structure. Uses inline callback processing like split.c.
typedef struct iree_tokenizer_normalizer_regex_replace_state_t {
  iree_tokenizer_normalizer_state_t base;

  // Position tracking (all absolute byte offsets).
  // Total bytes fed to regex.
  iree_host_size_t bytes_processed;
  // Bytes committed to output stream.
  iree_host_size_t output_position;

  // Pending match: when we've consumed a match's input but haven't finished
  // emitting its replacement content. This happens when output buffer fills
  // mid-replacement.
  // End of match needing content emission.
  iree_host_size_t pending_match_end;
  // True if content emission is pending.
  bool has_pending_match;

  // Partial content emission (when output smaller than content).
  uint8_t content_emitted;

  // Cached from normalizer for hot path.
  const uint8_t* content;
  uint8_t content_length;

  // Whether matching continues, replays after output pressure, or is finalized.
  uint8_t phase;

  // Regex execution state.
  iree_tokenizer_regex_exec_state_t regex_state;
} iree_tokenizer_normalizer_regex_replace_state_t;

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_regex_replace_vtable;

iree_status_t iree_tokenizer_normalizer_regex_replace_allocate(
    iree_string_view_t pattern, iree_string_view_t content,
    iree_allocator_t allocator, iree_tokenizer_normalizer_t** out_normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_normalizer);
  *out_normalizer = NULL;

  // Validate inputs before any allocations.
  if (pattern.size == 0) {
    IREE_RETURN_AND_END_ZONE(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "regex replace normalizer requires non-empty "
                             "pattern"));
  }
  if (content.size > IREE_TOKENIZER_REGEX_REPLACE_MAX_CONTENT) {
    IREE_RETURN_AND_END_ZONE(
        z0,
        iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "regex replace content exceeds maximum length (%" PRIhsz " > %d)",
            content.size, IREE_TOKENIZER_REGEX_REPLACE_MAX_CONTENT));
  }

  // Calculate total allocation size.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              iree_sizeof_struct(iree_tokenizer_normalizer_regex_replace_t),
              &total_size, IREE_STRUCT_FIELD(content.size, uint8_t, NULL)));

  // Compile regex pattern (first allocation — status chaining from here).
  iree_tokenizer_regex_dfa_t dfa = {0};
  uint8_t* dfa_data = NULL;
  iree_tokenizer_regex_stride_t* stride = NULL;
  iree_tokenizer_normalizer_regex_replace_t* normalizer = NULL;

  iree_tokenizer_regex_compile_error_t compile_error;
  iree_status_t status = iree_tokenizer_regex_compile_and_load(
      pattern, IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, allocator, &dfa,
      &dfa_data, &compile_error);
  if (!iree_status_is_ok(status) && compile_error.message[0] != '\0') {
    status =
        iree_status_annotate_f(status, "regex compile error at %zu: %s",
                               compile_error.position, compile_error.message);
  }

  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_regex_stride_allocate(&dfa, allocator, &stride);
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, total_size, (void**)&normalizer);
  }

  if (iree_status_is_ok(status)) {
    iree_tokenizer_normalizer_initialize(
        &normalizer->base, &iree_tokenizer_normalizer_regex_replace_vtable,
        sizeof(iree_tokenizer_normalizer_regex_replace_state_t));
    normalizer->allocator = allocator;
    normalizer->dfa = dfa;
    normalizer->stride = stride;
    normalizer->dfa_data = dfa_data;
    normalizer->content_length = (uint8_t)content.size;
    if (content.size > 0) {
      memcpy((uint8_t*)(normalizer + 1), content.data, content.size);
    }
    *out_normalizer = &normalizer->base;
  } else {
    iree_tokenizer_regex_stride_free(stride, allocator);
    iree_allocator_free(allocator, dfa_data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static inline const uint8_t* iree_tokenizer_normalizer_regex_replace_content(
    const iree_tokenizer_normalizer_regex_replace_t* normalizer) {
  return (const uint8_t*)(normalizer + 1);
}

static void iree_tokenizer_normalizer_regex_replace_destroy(
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_normalizer_regex_replace_t* self =
      (iree_tokenizer_normalizer_regex_replace_t*)normalizer;
  iree_allocator_t allocator = self->allocator;
  if (self->stride) {
    iree_tokenizer_regex_stride_free(self->stride, allocator);
  }
  if (self->dfa_data) {
    iree_allocator_free(allocator, self->dfa_data);
  }
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_normalizer_regex_replace_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_normalizer_regex_replace_t* self =
      (const iree_tokenizer_normalizer_regex_replace_t*)normalizer;
  iree_tokenizer_normalizer_regex_replace_state_t* state =
      (iree_tokenizer_normalizer_regex_replace_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.normalizer = normalizer;
  state->content = iree_tokenizer_normalizer_regex_replace_content(self);
  state->content_length = self->content_length;
  iree_tokenizer_regex_exec_initialize(&state->regex_state, &self->dfa);

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_normalizer_regex_replace_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

// Callback context for inline match processing.
typedef struct iree_tokenizer_regex_replace_context_t {
  // Incoming bytes, or the complete retained tail during finalization.
  const uint8_t* input_base;
  // Absolute offset of input_base.
  iree_host_size_t chunk_base;
  // Previously accepted bytes ending at chunk_base, stable during the call.
  iree_string_view_t preceding_input;
  // Cursor advanced as bytes are committed to the caller's output.
  uint8_t** out_ptr;
  // End of the caller's output credit.
  uint8_t* out_end;

  // Mutable state (updated per-match on successful emit).
  iree_host_size_t output_position;
  uint8_t content_emitted;

  // Pending match: set when we start emitting content but can't finish.
  // output_position stays at match.start; pending_match_end records match.end.
  iree_host_size_t pending_match_end;
  bool has_pending_match;

  // Immutable config.
  const uint8_t* content;
  uint8_t content_length;
} iree_tokenizer_regex_replace_context_t;

// Copies confirmed non-matching bytes from the retained prefix or incoming
// chunk. A consumer must never index the incoming chunk with an older offset.
static iree_status_t iree_tokenizer_regex_replace_emit_passthrough(
    iree_tokenizer_regex_replace_context_t* context,
    iree_host_size_t target_position) {
  while (context->output_position < target_position &&
         *context->out_ptr < context->out_end) {
    const uint8_t* source = context->input_base;
    iree_host_size_t source_base = context->chunk_base;
    iree_host_size_t source_end = target_position;
    if (context->output_position < context->chunk_base) {
      source_base -= context->preceding_input.size;
      if (context->output_position < source_base) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "regex replacement requires passthrough "
                                "before the retained input");
      }
      source = (const uint8_t*)context->preceding_input.data;
      source_end = iree_min(source_end, context->chunk_base);
    }
    iree_host_size_t to_write =
        iree_min(source_end - context->output_position,
                 (iree_host_size_t)(context->out_end - *context->out_ptr));
    memcpy(*context->out_ptr, source + context->output_position - source_base,
           to_write);
    *context->out_ptr += to_write;
    context->output_position += to_write;
  }
  return iree_ok_status();
}

// Emits replacement content. Returns true if all content was emitted.
static bool iree_tokenizer_regex_replace_emit_content(
    iree_tokenizer_regex_replace_context_t* context) {
  if (context->content_length == 0) {
    return true;
  }
  iree_host_size_t output_available =
      (iree_host_size_t)(context->out_end - *context->out_ptr);
  iree_host_size_t remaining =
      context->content_length - context->content_emitted;
  iree_host_size_t to_write =
      remaining < output_available ? remaining : output_available;
  if (to_write > 0) {
    memcpy(*context->out_ptr, context->content + context->content_emitted,
           to_write);
    *context->out_ptr += to_write;
    context->content_emitted += (uint8_t)to_write;
  }
  if (context->content_emitted >= context->content_length) {
    context->content_emitted = 0;
    return true;
  }
  return false;
}

// Inline callback: processes each match immediately.
// Emits passthrough bytes up to match start, then emits replacement content.
// Returns RESOURCE_EXHAUSTED if output buffer is full.
static iree_status_t iree_tokenizer_regex_replace_match_callback(
    void* user_data, iree_tokenizer_regex_match_t match) {
  iree_tokenizer_regex_replace_context_t* context =
      (iree_tokenizer_regex_replace_context_t*)user_data;

  // Emit passthrough bytes from output_position to match.start.
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_regex_replace_emit_passthrough(context, match.start));
  if (context->output_position < match.start) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  // Emit replacement content.
  if (!iree_tokenizer_regex_replace_emit_content(context)) {
    // Partial or no content written. Record the pending match so we can finish
    // emitting content and then advance to match.end on resume.
    // CRITICAL: Do NOT advance output_position here - it stays at match.start
    // so we know content emission is pending.
    context->has_pending_match = true;
    context->pending_match_end = match.end;
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  // Advance past matched region.
  context->output_position = match.end;
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_regex_replace_state_process(
    iree_tokenizer_normalizer_state_t* base_state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  (void)flags;  // Regex replace handles boundaries via exec state.

  iree_tokenizer_normalizer_regex_replace_state_t* state =
      (iree_tokenizer_normalizer_regex_replace_state_t*)base_state;
  const iree_tokenizer_normalizer_regex_replace_t* normalizer =
      (const iree_tokenizer_normalizer_regex_replace_t*)base_state->normalizer;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;
  iree_host_size_t chunk_base = state->bytes_processed;

  *out_consumed = 0;
  *out_written = 0;

  if (input.size == 0 || output.size == 0) {
    return iree_ok_status();
  }

  // Resume partial content emission from previous call.
  if (state->has_pending_match) {
    // We have a pending match: content emission was interrupted.
    // content_emitted tracks how much was written; may be 0 if output was full.
    iree_host_size_t remaining = state->content_length - state->content_emitted;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write =
        remaining < output_available ? remaining : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, state->content + state->content_emitted, to_write);
      out_ptr += to_write;
      state->content_emitted += (uint8_t)to_write;
    }
    if (state->content_emitted < state->content_length) {
      // Still more content to emit; output full.
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
    // Content fully emitted. Now advance output_position past the match.
    state->output_position = state->pending_match_end;
    state->content_emitted = 0;
    state->has_pending_match = false;
  }

  // The executor refreshes its retained tail after feeding the new chunk.
  // Preserve any older accepted bytes still awaiting passthrough so output
  // callbacks and post-scan emission see the same immutable input.
  uint8_t preceding_bytes[IREE_TOKENIZER_UTIL_REGEX_REWIND_BUFFER_SIZE];
  iree_string_view_t preceding_input = iree_string_view_empty();
  if (state->output_position < chunk_base) {
    memcpy(preceding_bytes, state->regex_state.rewind_buffer,
           state->regex_state.rewind_buffer_length);
    preceding_input = iree_make_string_view(
        (const char*)preceding_bytes, state->regex_state.rewind_buffer_length);
  }
  iree_tokenizer_regex_replace_context_t context = {
      .input_base = (const uint8_t*)input.data,
      .chunk_base = chunk_base,
      .preceding_input = preceding_input,
      .out_ptr = &out_ptr,
      .out_end = out_end,
      .output_position = state->output_position,
      .content_emitted = 0,
      .pending_match_end = 0,
      .has_pending_match = false,
      .content = state->content,
      .content_length = state->content_length,
  };

  iree_status_t feed_status = iree_ok_status();
  if (state->phase == IREE_TOKENIZER_REGEX_REPLACE_REPLAY) {
    iree_tokenizer_regex_exec_initialize(&state->regex_state, &normalizer->dfa);
    if (state->output_position < chunk_base) {
      iree_host_size_t prefix_start = chunk_base - preceding_input.size;
      if (state->output_position < prefix_start) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "regex replacement requires replay before "
                                "the retained input");
      }
      feed_status = iree_tokenizer_regex_exec_feed(
          &normalizer->dfa, &state->regex_state,
          iree_make_string_view(
              preceding_input.data + state->output_position - prefix_start,
              chunk_base - state->output_position),
          state->output_position, normalizer->stride,
          iree_tokenizer_regex_replace_match_callback, &context);
    }
  }
  if (iree_status_is_ok(feed_status)) {
    feed_status = iree_tokenizer_regex_exec_feed(
        &normalizer->dfa, &state->regex_state, input, chunk_base,
        normalizer->stride, iree_tokenizer_regex_replace_match_callback,
        &context);
  }
  if (!iree_status_is_ok(feed_status) &&
      !iree_status_is_resource_exhausted(feed_status)) {
    return feed_status;
  }
  bool output_full = iree_status_is_resource_exhausted(feed_status);
  iree_status_ignore(feed_status);
  iree_host_size_t input_end = chunk_base + input.size;
  if (!output_full) {
    iree_host_size_t confirmed_end = state->regex_state.in_match
                                         ? state->regex_state.match_start
                                         : input_end;
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_regex_replace_emit_passthrough(&context, confirmed_end));
    output_full = context.output_position < confirmed_end;
  }

  state->output_position = context.output_position;
  state->content_emitted = context.content_emitted;
  state->has_pending_match = context.has_pending_match;
  state->pending_match_end = context.pending_match_end;
  if (output_full) {
    // Already accepted input stays owned by this state even if output stopped
    // inside the prior chunk. Only newly accepted bytes count as consumption.
    iree_host_size_t resume_position = context.has_pending_match
                                           ? context.pending_match_end
                                           : context.output_position;
    state->bytes_processed = iree_max(chunk_base, resume_position);
    state->phase = IREE_TOKENIZER_REGEX_REPLACE_REPLAY;
    iree_tokenizer_regex_exec_initialize(&state->regex_state, &normalizer->dfa);
    if (resume_position < chunk_base) {
      memcpy(state->regex_state.rewind_buffer, preceding_input.data,
             preceding_input.size);
      state->regex_state.rewind_buffer_start =
          chunk_base - preceding_input.size;
      state->regex_state.rewind_buffer_length = preceding_input.size;
    }
  } else {
    state->bytes_processed = input_end;
    state->phase = IREE_TOKENIZER_REGEX_REPLACE_CONTINUE;
  }
  *out_consumed = state->bytes_processed - chunk_base;

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static iree_status_t iree_tokenizer_normalizer_regex_replace_state_finalize(
    iree_tokenizer_normalizer_state_t* base_state,
    iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  iree_tokenizer_normalizer_regex_replace_state_t* state =
      (iree_tokenizer_normalizer_regex_replace_state_t*)base_state;
  const iree_tokenizer_normalizer_regex_replace_t* normalizer =
      (const iree_tokenizer_normalizer_regex_replace_t*)base_state->normalizer;

  uint8_t* out_ptr = (uint8_t*)output.data;
  uint8_t* out_end = out_ptr + output.size;
  *out_written = 0;

  // Resume partial content emission from pending match.
  if (state->has_pending_match) {
    iree_host_size_t remaining = state->content_length - state->content_emitted;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write =
        remaining < output_available ? remaining : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, state->content + state->content_emitted, to_write);
      out_ptr += to_write;
      state->content_emitted += (uint8_t)to_write;
    }
    if (state->content_emitted < state->content_length) {
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
    // Content fully emitted. Advance past the match.
    state->output_position = state->pending_match_end;
    state->content_emitted = 0;
    state->has_pending_match = false;
  }

  if (state->phase != IREE_TOKENIZER_REGEX_REPLACE_DONE) {
    iree_tokenizer_regex_replace_context_t context = {
        .input_base = state->regex_state.rewind_buffer,
        .chunk_base = state->regex_state.rewind_buffer_start,
        .out_ptr = &out_ptr,
        .out_end = out_end,
        .output_position = state->output_position,
        .content_emitted = 0,
        .pending_match_end = 0,
        .has_pending_match = false,
        .content = state->content,
        .content_length = state->content_length,
    };

    // Retained bytes remain owned by the original state. Re-entry uses a
    // transient executor so scanning cannot overwrite pending passthrough.
    iree_tokenizer_regex_exec_state_t replay_state;
    iree_tokenizer_regex_exec_state_t* exec_state = &state->regex_state;
    iree_status_t finalize_status = iree_ok_status();
    if (state->phase == IREE_TOKENIZER_REGEX_REPLACE_REPLAY &&
        state->output_position < state->bytes_processed) {
      if (state->output_position < state->regex_state.rewind_buffer_start) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "regex replacement requires input before "
                                "the retained finalization buffer");
      }
      iree_tokenizer_regex_exec_initialize(&replay_state, &normalizer->dfa);
      exec_state = &replay_state;
      iree_host_size_t offset =
          state->output_position - state->regex_state.rewind_buffer_start;
      finalize_status = iree_tokenizer_regex_exec_feed(
          &normalizer->dfa, exec_state,
          iree_make_string_view(
              (const char*)state->regex_state.rewind_buffer + offset,
              state->bytes_processed - state->output_position),
          state->output_position, normalizer->stride,
          iree_tokenizer_regex_replace_match_callback, &context);
    }
    if (iree_status_is_ok(finalize_status) &&
        state->output_position < state->bytes_processed) {
      finalize_status = iree_tokenizer_regex_exec_finalize(
          &normalizer->dfa, exec_state, state->bytes_processed,
          iree_string_view_empty(), iree_tokenizer_regex_replace_match_callback,
          &context);
    }
    if (!iree_status_is_ok(finalize_status) &&
        !iree_status_is_resource_exhausted(finalize_status)) {
      return finalize_status;
    }
    state->output_position = context.output_position;
    state->content_emitted = context.content_emitted;
    state->has_pending_match = context.has_pending_match;
    state->pending_match_end = context.pending_match_end;
    if (iree_status_is_resource_exhausted(finalize_status)) {
      state->phase = IREE_TOKENIZER_REGEX_REPLACE_REPLAY;
      iree_status_ignore(finalize_status);
      *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
      return iree_ok_status();
    }
    state->phase = IREE_TOKENIZER_REGEX_REPLACE_DONE;
  }

  // Emit trailing passthrough bytes that process() couldn't emit because the
  // regex DFA was evaluating a partial match. The gap between output_position
  // and bytes_processed represents bytes consumed by process() but held by the
  // DFA. After exec_finalize resolves the partial match (as non-matching), the
  // bytes must be emitted as passthrough. The data is in the regex exec state's
  // rewind buffer, which exec_feed populates for partial matches and
  // exec_finalize does not modify.
  if (state->output_position < state->bytes_processed) {
    iree_host_size_t gap = state->bytes_processed - state->output_position;
    if (state->output_position < state->regex_state.rewind_buffer_start ||
        state->output_position - state->regex_state.rewind_buffer_start + gap >
            state->regex_state.rewind_buffer_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "regex replace finalize: trailing passthrough [%" PRIhsz ", %" PRIhsz
          ") not in rewind buffer [%" PRIhsz ", %" PRIhsz ")",
          state->output_position, state->bytes_processed,
          state->regex_state.rewind_buffer_start,
          state->regex_state.rewind_buffer_start +
              state->regex_state.rewind_buffer_length);
    }
    iree_host_size_t rewind_offset =
        state->output_position - state->regex_state.rewind_buffer_start;
    iree_host_size_t output_available = (iree_host_size_t)(out_end - out_ptr);
    iree_host_size_t to_write = gap < output_available ? gap : output_available;
    if (to_write > 0) {
      memcpy(out_ptr, state->regex_state.rewind_buffer + rewind_offset,
             to_write);
      out_ptr += to_write;
      state->output_position += to_write;
    }
  }

  *out_written = (iree_host_size_t)(out_ptr - (uint8_t*)output.data);
  return iree_ok_status();
}

static bool iree_tokenizer_normalizer_regex_replace_state_has_pending(
    const iree_tokenizer_normalizer_state_t* base_state) {
  const iree_tokenizer_normalizer_regex_replace_state_t* state =
      (const iree_tokenizer_normalizer_regex_replace_state_t*)base_state;
  if (state->has_pending_match || state->content_emitted > 0) {
    return true;
  }
  // Trailing passthrough gap: process() consumed bytes the regex DFA was
  // evaluating but couldn't emit. After finalize resolves the partial match,
  // the gap bytes are emitted from the rewind buffer.
  if (state->output_position < state->bytes_processed) {
    return true;
  }
  return false;
}

static const iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_regex_replace_vtable = {
        .destroy = iree_tokenizer_normalizer_regex_replace_destroy,
        .state_initialize =
            iree_tokenizer_normalizer_regex_replace_state_initialize,
        .state_deinitialize =
            iree_tokenizer_normalizer_regex_replace_state_deinitialize,
        .state_process = iree_tokenizer_normalizer_regex_replace_state_process,
        .state_finalize =
            iree_tokenizer_normalizer_regex_replace_state_finalize,
        .state_has_pending =
            iree_tokenizer_normalizer_regex_replace_state_has_pending,
};
