// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/tokenizer.h"

#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/decode_table.h"
#include "iree/tokenizer/decoder.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/postprocessor.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/special_tokens.h"
#include "iree/tokenizer/tokenizer_instance.h"
#include "iree/tokenizer/vocab/vocab.h"

//===----------------------------------------------------------------------===//
// Configuration Constants
//===----------------------------------------------------------------------===//

// Default segment batch size. This determines how many segments are buffered
// between the segmenter and model stages. Larger values amortize vtable
// call overhead but use more state storage.
#define IREE_TOKENIZER_DEFAULT_SEGMENT_BATCH_SIZE 64

// Default string batch size for decode. This determines how many token strings
// are looked up from vocab before feeding to the decoder. Larger values
// amortize vtable call overhead but use more state storage (16 bytes per slot
// on 64-bit systems for iree_string_view_t).
#define IREE_TOKENIZER_DEFAULT_STRING_BATCH_SIZE 64

//===----------------------------------------------------------------------===//
// Internal Structures
//===----------------------------------------------------------------------===//

// Completion advances only after the preceding producer can emit no more.
// MATCH drains normalized token matching before the segmenter receives EOF.
typedef enum iree_tokenizer_encode_phase_e {
  IREE_TOKENIZER_ENCODE_PHASE_PROCESS = 0,
  IREE_TOKENIZER_ENCODE_PHASE_NORMALIZE,
  IREE_TOKENIZER_ENCODE_PHASE_MATCH,
  IREE_TOKENIZER_ENCODE_PHASE_SEGMENT,
  IREE_TOKENIZER_ENCODE_PHASE_MODEL,
  IREE_TOKENIZER_ENCODE_PHASE_SUFFIX,
  IREE_TOKENIZER_ENCODE_PHASE_DONE,
} iree_tokenizer_encode_phase_t;

// Internal encode state structure.
// Layout in state_storage:
//   [encode_state_t]
//   [normalizer_state (normalizer->state_size bytes)]
//   [segmenter_state (segmenter->state_size bytes)]
//   [model_state (model->state_size bytes)]
//   [segment_buffer (segment_batch_size * sizeof(segment_t))]
struct iree_tokenizer_encode_state_t {
  const iree_tokenizer_t* tokenizer;

  // User-provided buffers.
  iree_byte_span_t transform_buffer;
  iree_tokenizer_offset_run_list_t offset_runs;

  // Flags from initialization.
  iree_tokenizer_encode_flags_t flags;

  // Special token partial match state for streaming. Runs BEFORE the normalizer
  // to match literal tokens like <|endoftext|> in raw input. Tracks partial
  // matches that span chunk boundaries so they can be completed or recovered.
  iree_tokenizer_special_tokens_encode_state_t special_token_match;

  // Post-normalization special token match state. Runs AFTER the normalizer
  // to match tokens with normalized=true in the transformed output.
  iree_tokenizer_special_tokens_encode_state_t special_token_match_post;

  // Ring buffer position tracking. All positions are logical (cumulative).
  // Physical buffer index = logical_position & capacity_mask.
  // Invariants:
  //   read_position <= segmenter_view_start <= write_position
  //   write_position - read_position <= (capacity_mask + 1)
  //
  // Buffer capacity - 1, for fast & instead of %. Must be power of two minus 1.
  iree_host_size_t capacity_mask;
  // Oldest byte still needed. Bytes before this position can be overwritten.
  iree_host_size_t read_position;
  // Where normalizer writes next.
  iree_host_size_t write_position;
  // Start of segmenter's view. The segmenter sees [segmenter_view_start,
  // write_position). Used for view management and reclaim tracking. Advances
  // when segmenter consumes bytes or when post-norm special tokens skip bytes.
  iree_host_size_t segmenter_view_start;

  // Segment buffer (points into state_storage after stage states).
  iree_tokenizer_segment_t* segments;
  iree_host_size_t segment_count;
  iree_host_size_t segments_consumed;

  // Pipeline stage states (point into state_storage).
  iree_tokenizer_normalizer_state_t* normalizer_state;
  iree_tokenizer_segmenter_state_t* segmenter_state;
  iree_tokenizer_model_state_t* model_state;

  // Postprocessor state. When phase is IDLE, all postprocessor operations are
  // no-ops (either ADD_SPECIAL_TOKENS not set or template has no tokens).
  iree_tokenizer_postprocessor_encode_state_t postprocessor;

  // The model owns an unfinished span fed directly from the ring. Reclaiming
  // its bytes does not retire the span; a producer boundary must complete it.
  bool has_partial_segment;

  // Deferred pre-normalization special token emission.
  //
  // When a special token is matched but the pipeline still has content (ring
  // buffer or pending segments), we defer emission until that content is
  // tokenized. This ensures correct output ordering: "hello<|endoftext|>world"
  // produces [hello_tokens..., <|endoftext|>, world_tokens...], not
  // [<|endoftext|>, hello_tokens..., world_tokens...].
  //
  // -1 means no pending token. When >= 0, emit after pipeline flushes.
  iree_tokenizer_token_id_t pending_special_token;

  // True when a special token consumed position 0 of the input. Used to signal
  // the prepend normalizer to skip prepending for prepend_scheme="first"
  // semantics: text following a position-0 special token is NOT "first".
  bool first_consumed_by_special_token;

  // Producer completion frontier, retained across output backpressure.
  iree_tokenizer_encode_phase_t phase;
};

//===----------------------------------------------------------------------===//
// Ring Buffer Helpers
//===----------------------------------------------------------------------===//

// Converts a logical (cumulative) position to a physical buffer index.
// Uses bitwise AND instead of modulo for performance (capacity must be power
// of two).
static inline iree_host_size_t iree_tokenizer_ring_to_physical(
    iree_host_size_t logical_position, iree_host_size_t capacity_mask) {
  return logical_position & capacity_mask;
}

// Returns the number of bytes currently in the ring buffer.
// This is the distance between read and write positions.
static inline iree_host_size_t iree_tokenizer_ring_used(
    const iree_tokenizer_encode_state_t* state) {
  return state->write_position - state->read_position;
}

// Returns the total available space in the ring buffer.
// This is capacity minus used bytes.
static inline iree_host_size_t iree_tokenizer_ring_available(
    const iree_tokenizer_encode_state_t* state) {
  iree_host_size_t capacity = state->capacity_mask + 1;
  return capacity - iree_tokenizer_ring_used(state);
}

// Returns the available ring space for writing. Writes may extend into the
// mirror region [capacity, 2*capacity) since the buffer is allocated 2x
// capacity. After writing, callers must call ring_fixup_write_wrap to copy
// any overflow from the mirror back to the buffer start.
static inline iree_host_size_t iree_tokenizer_ring_writable_space(
    const iree_tokenizer_encode_state_t* state) {
  return iree_tokenizer_ring_available(state);
}

// After writing |bytes_written| bytes starting at |physical_write_start|,
// copies any bytes that extended past the buffer capacity from the mirror
// region [capacity, capacity + overflow) back to the buffer start [0,
// overflow). This maintains the ring invariant that logical positions map
// correctly via (position & capacity_mask).
static inline void iree_tokenizer_ring_fixup_write_wrap(
    iree_tokenizer_encode_state_t* state, iree_host_size_t physical_write_start,
    iree_host_size_t bytes_written) {
  iree_host_size_t capacity = state->capacity_mask + 1;
  if (physical_write_start + bytes_written > capacity) {
    iree_host_size_t overflow = physical_write_start + bytes_written - capacity;
    memcpy(state->transform_buffer.data,
           state->transform_buffer.data + capacity, overflow);
  }
}

// Returns a contiguous view of [start, end) in logical ring coordinates.
// The mirrored bytes remain valid until the next write to their source region.
static iree_string_view_t iree_tokenizer_ring_get_view(
    iree_tokenizer_encode_state_t* state, iree_host_size_t start,
    iree_host_size_t end) {
  iree_host_size_t length = end - start;
  iree_host_size_t capacity = state->capacity_mask + 1;
  iree_host_size_t physical_start = start & state->capacity_mask;
  if (physical_start + length > capacity) {
    iree_host_size_t wrap_length = physical_start + length - capacity;
    memcpy(state->transform_buffer.data + capacity,
           state->transform_buffer.data, wrap_length);
  }
  return iree_make_string_view(
      (const char*)state->transform_buffer.data + physical_start, length);
}

//===----------------------------------------------------------------------===//
// Builder Implementation
//===----------------------------------------------------------------------===//

void iree_tokenizer_builder_initialize(iree_allocator_t allocator,
                                       iree_tokenizer_builder_t* out_builder) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_builder, 0, sizeof(*out_builder));
  out_builder->allocator = allocator;
  out_builder->string_batch_size = IREE_TOKENIZER_DEFAULT_STRING_BATCH_SIZE;
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_builder_deinitialize(iree_tokenizer_builder_t* builder) {
  if (!builder) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Free any components still owned by builder.
  // After successful build(), components are transferred to tokenizer,
  // so this becomes a no-op. Only frees on error path.
  iree_tokenizer_normalizer_free(builder->normalizer);
  iree_tokenizer_segmenter_free(builder->segmenter);
  // Free model BEFORE vocab since model references but doesn't own vocab.
  iree_tokenizer_model_free(builder->model);
  iree_tokenizer_decoder_free(builder->decoder);
  iree_tokenizer_normalizer_free(builder->decode_transform);
  iree_tokenizer_postprocessor_deinitialize(&builder->postprocessor);
  iree_tokenizer_special_tokens_deinitialize(&builder->special_tokens);
  iree_tokenizer_special_tokens_deinitialize(
      &builder->special_tokens_post_norm);
  iree_tokenizer_vocab_free(builder->vocab);

  memset(builder, 0, sizeof(*builder));
  IREE_TRACE_ZONE_END(z0);
}

void iree_tokenizer_builder_set_normalizer(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_normalizer_t* normalizer) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing normalizer if any.
  iree_tokenizer_normalizer_free(builder->normalizer);
  builder->normalizer = normalizer;
}

void iree_tokenizer_builder_set_segmenter(
    iree_tokenizer_builder_t* builder, iree_tokenizer_segmenter_t* segmenter) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing segmenter if any.
  iree_tokenizer_segmenter_free(builder->segmenter);
  builder->segmenter = segmenter;
}

void iree_tokenizer_builder_set_model(iree_tokenizer_builder_t* builder,
                                      iree_tokenizer_model_t* model) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing model if any.
  iree_tokenizer_model_free(builder->model);
  builder->model = model;
}

void iree_tokenizer_builder_set_decoder(iree_tokenizer_builder_t* builder,
                                        iree_tokenizer_decoder_t* decoder) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing decoder if any.
  iree_tokenizer_decoder_free(builder->decoder);
  builder->decoder = decoder;
}

void iree_tokenizer_builder_set_decode_transform(
    iree_tokenizer_builder_t* builder, iree_tokenizer_normalizer_t* transform) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_tokenizer_normalizer_free(builder->decode_transform);
  builder->decode_transform = transform;
}

void iree_tokenizer_builder_set_postprocessor(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_postprocessor_t postprocessor) {
  IREE_ASSERT_ARGUMENT(builder);
  iree_tokenizer_postprocessor_deinitialize(&builder->postprocessor);
  builder->postprocessor = postprocessor;
}

void iree_tokenizer_builder_set_special_tokens(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_special_tokens_t* special_tokens) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(special_tokens);
  iree_tokenizer_special_tokens_deinitialize(&builder->special_tokens);
  // Move ownership: copy struct and clear source.
  builder->special_tokens = *special_tokens;
  memset(special_tokens, 0, sizeof(*special_tokens));
}

void iree_tokenizer_builder_set_special_tokens_post_norm(
    iree_tokenizer_builder_t* builder,
    iree_tokenizer_special_tokens_t* special_tokens) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(special_tokens);
  iree_tokenizer_special_tokens_deinitialize(
      &builder->special_tokens_post_norm);
  // Move ownership: copy struct and clear source.
  builder->special_tokens_post_norm = *special_tokens;
  memset(special_tokens, 0, sizeof(*special_tokens));
}

void iree_tokenizer_builder_set_vocab(iree_tokenizer_builder_t* builder,
                                      iree_tokenizer_vocab_t* vocab) {
  IREE_ASSERT_ARGUMENT(builder);
  // Free existing vocab if any.
  iree_tokenizer_vocab_free(builder->vocab);
  builder->vocab = vocab;
}

void iree_tokenizer_builder_set_string_batch_size(
    iree_tokenizer_builder_t* builder, iree_host_size_t batch_size) {
  IREE_ASSERT_ARGUMENT(builder);
  builder->string_batch_size = batch_size;
}

iree_status_t iree_tokenizer_builder_build(iree_tokenizer_builder_t* builder,
                                           iree_tokenizer_t** out_tokenizer) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_tokenizer);
  *out_tokenizer = NULL;

  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate required components.
  // Note: segmenter is optional - NULL means passthrough (entire input as one
  // segment). This is used by tokenizers with null pre_tokenizer (TinyLlama,
  // Phi-3, etc.).
  iree_status_t status = iree_ok_status();
  if (!builder->model) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "model is required");
  }
  if (iree_status_is_ok(status) && builder->string_batch_size == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "string_batch_size must be > 0");
  }

  if (iree_status_is_ok(status) && builder->decode_transform &&
      (!builder->decoder || !builder->vocab)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "decode text transform requires a decoder and vocabulary");
  }

  // Allocate tokenizer struct.
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(builder->allocator, sizeof(*tokenizer),
                                   (void**)&tokenizer);
  }

  if (iree_status_is_ok(status)) {
    tokenizer->allocator = builder->allocator;
    tokenizer->segment_batch_size = IREE_TOKENIZER_DEFAULT_SEGMENT_BATCH_SIZE;
    tokenizer->string_batch_size = builder->string_batch_size;

    // Transfer ownership of components from builder to tokenizer.
    tokenizer->normalizer = builder->normalizer;
    tokenizer->segmenter = builder->segmenter;
    tokenizer->model = builder->model;
    tokenizer->decoder = builder->decoder;
    tokenizer->decode_transform = builder->decode_transform;
    tokenizer->postprocessor = builder->postprocessor;
    tokenizer->special_tokens = builder->special_tokens;
    tokenizer->special_tokens_post_norm = builder->special_tokens_post_norm;
    tokenizer->vocab = builder->vocab;

    // Clear builder (ownership transferred).
    builder->normalizer = NULL;
    builder->segmenter = NULL;
    builder->model = NULL;
    builder->decoder = NULL;
    builder->decode_transform = NULL;
    memset(&builder->postprocessor, 0, sizeof(builder->postprocessor));
    memset(&builder->special_tokens, 0, sizeof(builder->special_tokens));
    memset(&builder->special_tokens_post_norm, 0,
           sizeof(builder->special_tokens_post_norm));
    builder->vocab = NULL;

    // Build pre-decoded token table for fast-path decode.
    status = iree_tokenizer_decode_table_initialize(
        tokenizer->decoder, tokenizer->vocab, tokenizer->allocator,
        &tokenizer->pre_decoded);
    if (iree_status_is_ok(status) && tokenizer->decode_transform) {
      // UTF-8 repair can expand each vocabulary byte to U+FFFD. Reserve that
      // bound plus a separator and the byte accumulator's unfinished scalar.
      iree_host_size_t capacity = 0;
      if (!iree_host_size_checked_mul(
              iree_tokenizer_vocab_max_token_length(tokenizer->vocab), 3,
              &capacity) ||
          !iree_host_size_checked_add(capacity, 16, &capacity)) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "decoded token size exceeds address space");
      } else {
        tokenizer->decode_transform_capacity = iree_max(
            (iree_host_size_t)256,
            iree_max(capacity, tokenizer->pre_decoded.maximum_token_length));
      }
    }
  }

  if (iree_status_is_ok(status)) {
    *out_tokenizer = tokenizer;
  } else {
    iree_tokenizer_free(tokenizer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Tokenizer
//===----------------------------------------------------------------------===//

void iree_tokenizer_free(iree_tokenizer_t* tokenizer) {
  if (!tokenizer) {
    return;
  }

  iree_allocator_t allocator = tokenizer->allocator;

  // Free pre-decoded table (single slab covers offsets + data).
  iree_allocator_free(allocator, tokenizer->pre_decoded.slab);

  // Free owned components.
  iree_tokenizer_normalizer_free(tokenizer->normalizer);
  iree_tokenizer_segmenter_free(tokenizer->segmenter);
  // Free model BEFORE vocab since model references but doesn't own vocab.
  iree_tokenizer_model_free(tokenizer->model);
  iree_tokenizer_decoder_free(tokenizer->decoder);
  iree_tokenizer_normalizer_free(tokenizer->decode_transform);
  iree_tokenizer_postprocessor_deinitialize(&tokenizer->postprocessor);
  iree_tokenizer_special_tokens_deinitialize(&tokenizer->special_tokens);
  iree_tokenizer_special_tokens_deinitialize(
      &tokenizer->special_tokens_post_norm);
  iree_tokenizer_vocab_free(tokenizer->vocab);

  iree_allocator_free(allocator, tokenizer);
}

const iree_tokenizer_vocab_t* iree_tokenizer_vocab(
    const iree_tokenizer_t* tokenizer) {
  return tokenizer->vocab;
}

iree_host_size_t iree_tokenizer_max_special_token_count(
    const iree_tokenizer_t* tokenizer) {
  // Return the larger of single/pair template counts (pair may have more
  // tokens due to infix separators).
  iree_host_size_t single_count =
      iree_tokenizer_postprocessor_template_total_count(
          &tokenizer->postprocessor.single);
  iree_host_size_t pair_count =
      iree_tokenizer_postprocessor_template_total_count(
          &tokenizer->postprocessor.pair);
  return single_count > pair_count ? single_count : pair_count;
}

iree_string_view_t iree_tokenizer_model_type_name(
    const iree_tokenizer_t* tokenizer) {
  return tokenizer->model->type_name;
}

//===----------------------------------------------------------------------===//
// Batch Encode/Decode
//===----------------------------------------------------------------------===//

// Complete-input callers can diagnose a stall as an exhausted output span or
// incomplete final input. Incremental callers may supply more of either.
static iree_status_t iree_tokenizer_encode_state_feed_internal(
    iree_tokenizer_encode_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_output_t output, bool is_final_input,
    iree_host_size_t* out_bytes_consumed, iree_host_size_t* out_token_count);

iree_status_t iree_tokenizer_encode(const iree_tokenizer_t* tokenizer,
                                    iree_string_view_t text,
                                    iree_tokenizer_encode_flags_t flags,
                                    iree_tokenizer_token_output_t output,
                                    iree_allocator_t allocator,
                                    iree_host_size_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(output.token_ids);
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate combined buffer for state + transform.
  iree_host_size_t state_size = 0;
  iree_status_t status =
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size);

  // One-shot encoding needs a transform buffer proportional to the input since
  // pre-tokenizers with infrequent splits (e.g., DeepSeek-V3's compound
  // Sequence pre-tokenizer on English text) can produce segments spanning tens
  // of KB, exceeding the streaming-optimized 64KB cap. The streaming
  // recommended_size() caps at 64KB for bounded memory; one-shot is uncapped
  // since we already allocate O(text_size) for output tokens.
  iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_oneshot_size(text.size);
  iree_host_size_t total_size = 0;
  iree_host_size_t transform_offset = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        state_size, &total_size,
        IREE_STRUCT_ARRAY_FIELD(transform_size, 1, uint8_t, &transform_offset));
  }

  uint8_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, total_size, (void**)&buffer);
  }

  // Initialize and run pipeline. AT_INPUT_START is always set for fresh
  // encodes. TRACK_OFFSETS is inferred from the output having an offsets
  // buffer.
  iree_tokenizer_encode_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    iree_tokenizer_encode_flags_t effective_flags =
        flags | IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
    if (output.token_offsets) {
      effective_flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
    }
    status = iree_tokenizer_encode_state_initialize(
        tokenizer, iree_make_byte_span(buffer, state_size),
        iree_make_byte_span(buffer + transform_offset, transform_size),
        iree_tokenizer_offset_run_list_empty(), effective_flags, allocator,
        &state);
  }

  // Feed all input, collecting tokens.
  iree_host_size_t total_tokens = 0;
  while (iree_status_is_ok(status) && text.size > 0) {
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t tokens_written = 0;
    iree_tokenizer_token_output_t sub_output = {
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    status = iree_tokenizer_encode_state_feed_internal(
        state, text, sub_output, /*is_final_input=*/true, &bytes_consumed,
        &tokens_written);
    total_tokens += tokens_written;
    text.data += bytes_consumed;
    text.size -= bytes_consumed;
  }

  // Finalize to flush remaining tokens.
  if (iree_status_is_ok(status)) {
    iree_host_size_t final_tokens = 0;
    iree_tokenizer_token_output_t final_output = {
        .capacity = output.capacity - total_tokens,
        .token_ids = &output.token_ids[total_tokens],
        .token_offsets =
            output.token_offsets ? &output.token_offsets[total_tokens] : NULL,
        .type_ids = output.type_ids ? &output.type_ids[total_tokens] : NULL,
    };
    status = iree_tokenizer_encode_state_finalize(state, final_output,
                                                  &final_tokens);
    total_tokens += final_tokens;
    if (iree_status_is_ok(status) &&
        iree_tokenizer_encode_state_has_pending(state)) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "encode output capacity exhausted during finalization");
    }
  }

  if (iree_status_is_ok(status)) {
    *out_token_count = total_tokens;
  }

  // Cleanup.
  iree_tokenizer_encode_state_deinitialize(state);
  iree_allocator_free(allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Multi-Item Batch Encode/Decode
//===----------------------------------------------------------------------===//

typedef uint32_t iree_tokenizer_encode_state_reset_flags_t;
enum iree_tokenizer_encode_state_reset_flag_bits_e {
  IREE_TOKENIZER_ENCODE_STATE_RESET_FLAG_NONE = 0u,
  // Preserve postprocessor phase/template while resetting the rest of the
  // encode pipeline. Used between pair sequence A and sequence B.
  IREE_TOKENIZER_ENCODE_STATE_RESET_FLAG_PRESERVE_POSTPROCESSOR = 1u << 0,
};

typedef uint32_t iree_tokenizer_encode_state_finalize_flags_t;
enum iree_tokenizer_encode_state_finalize_flag_bits_e {
  IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_NONE = 0u,
  // Finalize normalizer/segmenter/model state without transitioning to or
  // emitting the postprocessor suffix. Used after pair sequence A so the
  // caller can emit the pair infix and continue with sequence B.
  IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_OMIT_SUFFIX = 1u << 0,
};

static iree_status_t iree_tokenizer_encode_state_finalize_internal(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_tokenizer_encode_state_finalize_flags_t finalize_flags,
    iree_host_size_t* out_token_count);

static iree_status_t iree_tokenizer_encode_state_reset_internal(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_encode_flags_t flags,
    iree_tokenizer_encode_state_reset_flags_t reset_flags) {
  if (!state) {
    return iree_ok_status();
  }
  const iree_tokenizer_t* tokenizer = state->tokenizer;

  // Reset model progress while retaining its workspace capacity.
  if (state->model_state) {
    iree_tokenizer_model_state_reset(state->model_state);
  }
  // Reinitialize the fixed-size transform stages.
  iree_tokenizer_segmenter_state_deinitialize(state->segmenter_state);
  iree_tokenizer_normalizer_state_deinitialize(state->normalizer_state);

  // Reset ring buffer positions.
  state->flags = flags;
  state->read_position = 0;
  state->write_position = 0;
  state->segmenter_view_start = 0;
  state->segment_count = 0;
  state->segments_consumed = 0;

  // Re-initialize pipeline stage states (storage layout unchanged).
  if (state->normalizer_state) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_initialize(
        tokenizer->normalizer, (void*)state->normalizer_state,
        &state->normalizer_state));
  }
  if (state->segmenter_state) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_initialize(
        tokenizer->segmenter, (void*)state->segmenter_state,
        &state->segmenter_state));
  }

  // Reset special token match states.
  iree_tokenizer_special_tokens_encode_state_initialize(
      &state->special_token_match);
  iree_tokenizer_special_tokens_encode_state_initialize(
      &state->special_token_match_post);

  if (!iree_all_bits_set(
          reset_flags,
          IREE_TOKENIZER_ENCODE_STATE_RESET_FLAG_PRESERVE_POSTPROCESSOR)) {
    if (iree_all_bits_set(flags,
                          IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS)) {
      iree_tokenizer_postprocessor_encode_state_initialize(
          &tokenizer->postprocessor, &tokenizer->postprocessor.single,
          &state->postprocessor);
    } else {
      memset(&state->postprocessor, 0, sizeof(state->postprocessor));
    }
  }

  state->pending_special_token = -1;
  state->first_consumed_by_special_token = false;
  state->phase = IREE_TOKENIZER_ENCODE_PHASE_PROCESS;
  state->has_partial_segment = false;
  return iree_ok_status();
}

static iree_tokenizer_token_output_t iree_tokenizer_sub_token_output(
    iree_tokenizer_token_output_t output, iree_host_size_t token_offset) {
  iree_tokenizer_token_output_t sub_output = {
      .capacity = output.capacity - token_offset,
      .token_ids = output.token_ids ? &output.token_ids[token_offset] : NULL,
      .token_offsets =
          output.token_offsets ? &output.token_offsets[token_offset] : NULL,
      .type_ids = output.type_ids ? &output.type_ids[token_offset] : NULL,
  };
  return sub_output;
}

static iree_status_t iree_tokenizer_encode_batch_feed_sequence(
    iree_tokenizer_encode_state_t* state, iree_string_view_t text,
    iree_tokenizer_token_output_t output,
    iree_tokenizer_encode_state_finalize_flags_t finalize_flags,
    iree_host_size_t* total_tokens) {
  while (text.size > 0) {
    if (*total_tokens >= output.capacity) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "batch encode: output buffer full before input was consumed "
          "(wrote %" PRIhsz " of %" PRIhsz " capacity)",
          *total_tokens, output.capacity);
    }
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t tokens_written = 0;
    iree_tokenizer_token_output_t sub_output =
        iree_tokenizer_sub_token_output(output, *total_tokens);
    iree_status_t status = iree_tokenizer_encode_state_feed_internal(
        state, text, sub_output, /*is_final_input=*/true, &bytes_consumed,
        &tokens_written);
    *total_tokens += tokens_written;
    if (!iree_status_is_ok(status)) {
      return status;
    }
    text.data += bytes_consumed;
    text.size -= bytes_consumed;
  }

  if (*total_tokens > output.capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "batch encode: output count exceeded capacity "
                            "(wrote %" PRIhsz " of %" PRIhsz " capacity)",
                            *total_tokens, output.capacity);
  }
  iree_host_size_t final_tokens = 0;
  iree_tokenizer_token_output_t final_output =
      iree_tokenizer_sub_token_output(output, *total_tokens);
  iree_status_t status = iree_tokenizer_encode_state_finalize_internal(
      state, final_output, finalize_flags, &final_tokens);
  *total_tokens += final_tokens;
  if (iree_status_is_ok(status) &&
      state->phase != IREE_TOKENIZER_ENCODE_PHASE_DONE) {
    status = iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "batch encode output capacity exhausted during finalization");
  }
  return status;
}

iree_status_t iree_tokenizer_encode_batch(
    const iree_tokenizer_t* tokenizer,
    iree_tokenizer_encode_batch_item_t* items, iree_host_size_t item_count,
    iree_tokenizer_encode_flags_t flags, iree_byte_span_t state_storage,
    iree_byte_span_t transform_buffer,
    iree_tokenizer_offset_run_list_t offset_runs,
    iree_allocator_t workspace_allocator) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(items || item_count == 0);

  // Initialize all output counts to zero.
  for (iree_host_size_t i = 0; i < item_count; ++i) {
    items[i].out_token_count = 0;
  }

  iree_status_t status = iree_ok_status();
  iree_tokenizer_encode_state_t* state = NULL;

  // Process each item, reusing state storage across items. Loop exits early on
  // error; prior items retain their results.
  for (iree_host_size_t i = 0; i < item_count && iree_status_is_ok(status);
       ++i) {
    iree_tokenizer_encode_batch_item_t* item = &items[i];

    // AT_INPUT_START always set for fresh encodes. TRACK_OFFSETS inferred
    // per-item from whether the output has an offsets buffer.
    iree_tokenizer_encode_flags_t effective_flags =
        flags | IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
    if (item->output.token_offsets) {
      effective_flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
    }

    bool has_text_pair = iree_all_bits_set(
        item->flags, IREE_TOKENIZER_ENCODE_BATCH_ITEM_FLAG_HAS_TEXT_PAIR);
    iree_tokenizer_encode_batch_item_flags_t unknown_flags =
        item->flags & ~IREE_TOKENIZER_ENCODE_BATCH_ITEM_FLAG_HAS_TEXT_PAIR;
    if (unknown_flags != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "batch encode item has unknown flags 0x%08x",
                                (unsigned)unknown_flags);
    } else if (!has_text_pair && item->text_pair.size > 0) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "batch encode item has non-empty text_pair but HAS_TEXT_PAIR flag "
          "is not set");
    }

    // First item: full initialize. Subsequent items: lightweight reset.
    if (iree_status_is_ok(status) && state == NULL) {
      status = iree_tokenizer_encode_state_initialize(
          tokenizer, state_storage, transform_buffer, offset_runs,
          effective_flags, workspace_allocator, &state);
    } else if (iree_status_is_ok(status)) {
      status = iree_tokenizer_encode_state_reset(state, effective_flags);
    }

    // For pair encoding, re-initialize the postprocessor with the pair
    // template. encode_state_initialize/reset always selects the single
    // template; pair template selection is an encode_batch concern since the
    // streaming API does not support pair encoding.
    if (iree_status_is_ok(status) && has_text_pair &&
        iree_all_bits_set(effective_flags,
                          IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS)) {
      if (!iree_tokenizer_postprocessor_supports_pair(
              &tokenizer->postprocessor)) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "pair encoding requested but tokenizer has no pair template");
      } else {
        iree_tokenizer_postprocessor_encode_state_initialize(
            &tokenizer->postprocessor, &tokenizer->postprocessor.pair,
            &state->postprocessor);
      }
    }

    // Feed all text (sequence A), collecting tokens.
    iree_host_size_t total_tokens = 0;

    if (iree_status_is_ok(status)) {
      iree_tokenizer_encode_state_finalize_flags_t finalize_flags =
          has_text_pair ? IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_OMIT_SUFFIX
                        : IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_NONE;
      status = iree_tokenizer_encode_batch_feed_sequence(
          state, item->text, item->output, finalize_flags, &total_tokens);
    }

    // Pair encoding: emit infix tokens, reset pipeline for sequence B, then
    // feed and finalize the second text.
    if (iree_status_is_ok(status) && has_text_pair) {
      // Transition SEQUENCE_A → INFIX (or directly to SEQUENCE_B).
      iree_tokenizer_postprocessor_begin_infix(&state->postprocessor);

      // Emit infix special tokens (e.g., [SEP] between sequences).
      total_tokens += iree_tokenizer_postprocessor_emit_infix(
          &state->postprocessor, item->output, total_tokens);
      if (iree_tokenizer_postprocessor_encode_state_has_pending(
              &state->postprocessor)) {
        status = iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "batch encode: output buffer full while emitting pair infix "
            "(wrote %" PRIhsz " of %" PRIhsz " capacity)",
            total_tokens, item->output.capacity);
      }
    }

    if (iree_status_is_ok(status) && has_text_pair) {
      // Reset the normalizer/segmenter/model pipeline for sequence B while
      // preserving the postprocessor state (now in SEQUENCE_B phase).
      // AT_INPUT_START is set because sequence B is a fresh input to the
      // normalizer (e.g., prepend_scheme="first" applies per-sequence).
      status = iree_tokenizer_encode_state_reset_internal(
          state, effective_flags,
          IREE_TOKENIZER_ENCODE_STATE_RESET_FLAG_PRESERVE_POSTPROCESSOR);
      if (iree_status_is_ok(status)) {
        status = iree_tokenizer_encode_batch_feed_sequence(
            state, item->text_pair, item->output,
            IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_NONE, &total_tokens);
      }
    }

    item->out_token_count = total_tokens;
  }

  iree_tokenizer_encode_state_deinitialize(state);
  return status;
}

//===----------------------------------------------------------------------===//
// Streaming Encode
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_encode_state_calculate_size(
    const iree_tokenizer_t* tokenizer, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  iree_host_size_t normalizer_state_size =
      tokenizer->normalizer
          ? iree_tokenizer_normalizer_state_size(tokenizer->normalizer)
          : 0;
  iree_host_size_t segmenter_state_size =
      tokenizer->segmenter
          ? iree_tokenizer_segmenter_state_size(tokenizer->segmenter)
          : 0;
  iree_host_size_t model_state_size =
      tokenizer->model ? iree_tokenizer_model_state_size(tokenizer->model) : 0;

  return IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_encode_state_t), out_size,
      IREE_STRUCT_FIELD_ALIGNED(normalizer_state_size, uint8_t,
                                iree_max_align_t, NULL),
      IREE_STRUCT_FIELD_ALIGNED(segmenter_state_size, uint8_t, iree_max_align_t,
                                NULL),
      IREE_STRUCT_FIELD_ALIGNED(model_state_size, uint8_t, iree_max_align_t,
                                NULL),
      IREE_STRUCT_FIELD(tokenizer->segment_batch_size, iree_tokenizer_segment_t,
                        NULL));
}

iree_status_t iree_tokenizer_encode_state_initialize(
    const iree_tokenizer_t* tokenizer, iree_byte_span_t state_storage,
    iree_byte_span_t transform_buffer,
    iree_tokenizer_offset_run_list_t offset_runs,
    iree_tokenizer_encode_flags_t flags, iree_allocator_t workspace_allocator,
    iree_tokenizer_encode_state_t** out_state) {
  IREE_ASSERT_ARGUMENT(tokenizer);
  IREE_ASSERT_ARGUMENT(state_storage.data);
  IREE_ASSERT_ARGUMENT(transform_buffer.data);
  IREE_ASSERT_ARGUMENT(out_state);

  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate layout with offsets. Must match calculate_size exactly.
  iree_host_size_t normalizer_state_size =
      tokenizer->normalizer
          ? iree_tokenizer_normalizer_state_size(tokenizer->normalizer)
          : 0;
  iree_host_size_t segmenter_state_size =
      tokenizer->segmenter
          ? iree_tokenizer_segmenter_state_size(tokenizer->segmenter)
          : 0;
  iree_host_size_t model_state_size =
      tokenizer->model ? iree_tokenizer_model_state_size(tokenizer->model) : 0;

  iree_host_size_t total_size = 0;
  iree_host_size_t normalizer_offset = 0;
  iree_host_size_t segmenter_offset = 0;
  iree_host_size_t model_offset = 0;
  iree_host_size_t segments_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_encode_state_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(normalizer_state_size, uint8_t,
                                iree_max_align_t, &normalizer_offset),
      IREE_STRUCT_FIELD_ALIGNED(segmenter_state_size, uint8_t, iree_max_align_t,
                                &segmenter_offset),
      IREE_STRUCT_FIELD_ALIGNED(model_state_size, uint8_t, iree_max_align_t,
                                &model_offset),
      IREE_STRUCT_FIELD(tokenizer->segment_batch_size, iree_tokenizer_segment_t,
                        &segments_offset));

  if (iree_status_is_ok(status) && state_storage.data_length < total_size) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "state_storage too small: need %" PRIhsz
                              " bytes, got %" PRIhsz,
                              total_size, state_storage.data_length);
  }

  // Validate transform buffer allocation for double-buffer mode.
  // Allocation must be a power of two. Logical capacity is half the allocation:
  // first half is the working buffer, second half is the mirror region for
  // contiguous wrap-spanning access (see
  // iree_tokenizer_ring_get_view).
  //
  // Minimum allocation is 2 * IREE_UNICODE_UTF8_MAX_BYTE_LENGTH = 8 bytes,
  // ensuring the logical capacity can hold at least one complete UTF-8
  // codepoint. In practice, 8KB-16KB is recommended for streaming.
  iree_host_size_t allocation = transform_buffer.data_length;
  iree_host_size_t min_allocation = 2 * IREE_UNICODE_UTF8_MAX_BYTE_LENGTH;
  if (iree_status_is_ok(status) &&
      (allocation < min_allocation ||
       !iree_host_size_is_power_of_two(allocation))) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transform buffer allocation must be a power of two >= %" PRIhsz
        ", got %" PRIhsz,
        min_allocation, allocation);
  }
  iree_host_size_t capacity = allocation / 2;

  // Validate buffer is large enough for the frozen token theorem to apply.
  // BPE's streaming mode freezes tokens when: current_byte > token_end +
  // max_token_length - 1. With capacity <= max_token_length, tokens at the
  // start of the buffer can never freeze, causing deadlock when the ring fills.
  // Require capacity > max_token_length (strictly greater) to guarantee
  // progress.
  iree_host_size_t max_token_length =
      iree_tokenizer_vocab_max_token_length(tokenizer->vocab);
  if (iree_status_is_ok(status) && capacity <= max_token_length) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "transform buffer too small for this tokenizer: logical capacity "
        "%" PRIhsz " must be > max_token_length %" PRIhsz
        " (need physical allocation > %" PRIhsz ")",
        capacity, max_token_length, max_token_length * 2);
  }

  // Initialize state struct at base of storage.
  iree_tokenizer_encode_state_t* state = NULL;
  if (iree_status_is_ok(status)) {
    uint8_t* base = (uint8_t*)state_storage.data;
    state = (iree_tokenizer_encode_state_t*)base;
    memset(state, 0, sizeof(*state));

    state->tokenizer = tokenizer;
    state->transform_buffer = transform_buffer;
    state->offset_runs = offset_runs;
    state->flags = flags;
    state->capacity_mask = capacity - 1;
    state->read_position = 0;
    state->write_position = 0;
    state->segmenter_view_start = 0;
    state->segment_count = 0;
    state->segments_consumed = 0;
  }

  // Initialize normalizer state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->normalizer) {
    uint8_t* base = (uint8_t*)state_storage.data;
    status = iree_tokenizer_normalizer_state_initialize(
        tokenizer->normalizer, base + normalizer_offset,
        &state->normalizer_state);
  }

  // Initialize segmenter state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->segmenter) {
    uint8_t* base = (uint8_t*)state_storage.data;
    status = iree_tokenizer_segmenter_state_initialize(
        tokenizer->segmenter, base + segmenter_offset, &state->segmenter_state);
  }

  // Initialize model state at computed offset.
  if (iree_status_is_ok(status) && tokenizer->model) {
    uint8_t* base = (uint8_t*)state_storage.data;
    iree_tokenizer_model_state_initialize(tokenizer->model, base + model_offset,
                                          workspace_allocator,
                                          &state->model_state);
  }

  if (iree_status_is_ok(status)) {
    // Set up segment buffer at computed offset.
    uint8_t* base = (uint8_t*)state_storage.data;
    state->segments = (iree_tokenizer_segment_t*)(base + segments_offset);

    // Initialize special token match state (runs before normalizer).
    iree_tokenizer_special_tokens_encode_state_initialize(
        &state->special_token_match);

    // Initialize post-normalization special token match state (runs after
    // normalizer, before segmenter).
    iree_tokenizer_special_tokens_encode_state_initialize(
        &state->special_token_match_post);

    // Set up postprocessor state if ADD_SPECIAL_TOKENS is requested.
    // The initializer is a no-op when the template has no special tokens.
    if (iree_all_bits_set(flags,
                          IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS)) {
      iree_tokenizer_postprocessor_encode_state_initialize(
          &tokenizer->postprocessor, &tokenizer->postprocessor.single,
          &state->postprocessor);
    }

    state->pending_special_token = -1;
    state->first_consumed_by_special_token = false;
    state->phase = IREE_TOKENIZER_ENCODE_PHASE_PROCESS;

    *out_state = state;
  } else {
    // Deinitialize any partially-initialized sub-states.
    iree_tokenizer_encode_state_deinitialize(state);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_tokenizer_encode_state_deinitialize(
    iree_tokenizer_encode_state_t* state) {
  if (!state) {
    return;
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  // Deinitialize stage states (does not free storage).
  iree_tokenizer_model_state_deinitialize(state->model_state);
  iree_tokenizer_segmenter_state_deinitialize(state->segmenter_state);
  iree_tokenizer_normalizer_state_deinitialize(state->normalizer_state);

  memset(state, 0, sizeof(*state));
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_tokenizer_encode_state_reset(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_encode_flags_t flags) {
  return iree_tokenizer_encode_state_reset_internal(
      state, flags, IREE_TOKENIZER_ENCODE_STATE_RESET_FLAG_NONE);
}

// Returns true if the encode pipeline has content that must be emitted before
// a special token: ring buffer data, pending segments, or normalizer-buffered
// data (e.g., NFC combining sequences).
static inline bool iree_tokenizer_encode_state_pipeline_has_content(
    const iree_tokenizer_encode_state_t* state) {
  if (state->write_position > state->read_position) {
    return true;
  }
  if (state->segment_count > 0) {
    return true;
  }
  if (state->has_partial_segment) {
    return true;
  }
  if (state->normalizer_state &&
      iree_tokenizer_normalizer_state_has_pending(state->normalizer_state)) {
    return true;
  }
  return false;
}

bool iree_tokenizer_encode_state_has_pending(
    const iree_tokenizer_encode_state_t* state) {
  IREE_ASSERT_ARGUMENT(state);

  if (state->phase == IREE_TOKENIZER_ENCODE_PHASE_DONE) {
    return false;
  }
  if (state->phase != IREE_TOKENIZER_ENCODE_PHASE_PROCESS) {
    return true;
  }

  // Check if any stage has pending data (ordered by pipeline position).
  // Deferred special token waiting for pipeline to flush.
  if (state->pending_special_token >= 0) {
    return true;
  }
  // Special token matching runs first, before normalizer.
  if (iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match)) {
    return true;
  }
  if (state->segment_count > state->segments_consumed) {
    return true;
  }
  if (state->write_position > state->segmenter_view_start) {
    return true;
  }
  if (state->normalizer_state &&
      iree_tokenizer_normalizer_state_has_pending(state->normalizer_state)) {
    return true;
  }
  // Post-normalization special token matching runs after normalizer output.
  if (iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match_post)) {
    return true;
  }
  if (state->segmenter_state &&
      iree_tokenizer_segmenter_state_has_pending(state->segmenter_state)) {
    return true;
  }
  if (state->model_state &&
      iree_tokenizer_model_state_has_pending(state->model_state)) {
    return true;
  }

  if (iree_tokenizer_postprocessor_encode_state_has_pending(
          &state->postprocessor)) {
    return true;
  }

  return false;
}

// Feeds ring buffer data [read_position, end_position) directly to the
// model as a single segment, bypassing the segmenter. Handles physical
// wrap-around via mirror copy. When |is_partial| is true, the model preserves
// state for continuation (more bytes will follow); when false, the model
// processes through to completion (FLUSH transition).
static iree_status_t iree_tokenizer_encode_ring_as_segment(
    iree_tokenizer_encode_state_t* state, iree_host_size_t end_position,
    bool is_partial, iree_tokenizer_token_output_t sub_output,
    iree_host_size_t* out_tokens_written, bool* out_segment_complete) {
  *out_tokens_written = 0;
  *out_segment_complete = false;

  // An empty final span still retires model state after all accepted bytes
  // have been reclaimed. Storage ownership and span lifetime are independent.
  iree_string_view_t view =
      iree_tokenizer_ring_get_view(state, state->read_position, end_position);
  iree_tokenizer_segment_t segment = {
      .start = state->read_position,
      .end = end_position,
  };
  iree_tokenizer_segment_list_t segment_list = {
      .count = 1,
      .values = &segment,
      .last_is_partial = is_partial,
      .buffer_offset = state->read_position,
  };

  iree_host_size_t segments_consumed = 0;
  IREE_RETURN_IF_ERROR(iree_tokenizer_model_state_encode(
      state->model_state, iree_make_const_byte_span(view.data, view.size),
      segment_list, sub_output, &segments_consumed, out_tokens_written));
  *out_segment_complete = (segments_consumed > 0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// LSTRIP Whitespace Stripping Helpers
//===----------------------------------------------------------------------===//

// Returns the flags for a token given its ID by searching the special tokens
// collection. Returns NONE if not found (which shouldn't happen for valid
// matches).
static iree_tokenizer_special_token_flags_t
iree_tokenizer_special_tokens_get_flags_for_id(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_tokenizer_token_id_t token_id) {
  for (iree_host_size_t i = 0; i < special_tokens->count; ++i) {
    if (special_tokens->ids[i] == token_id) {
      return special_tokens->flags[i];
    }
  }
  return IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE;
}

// Strips trailing whitespace from a safe prefix length when a potential
// special token at position |safe| has the LSTRIP flag. This prevents
// the whitespace immediately preceding an LSTRIP token from being tokenized.
//
// |special_tokens|: the special tokens collection to check
// |input|: the input text being processed
// |safe|: the position where a special token could start (may be modified)
// |check_result|: the result from a speculative match at position |safe|
// |check_id|: the matched token ID (only valid if check_result is MATCHED)
// |check_state|: the match state (contains partial_token_index for NEED_MORE)
//
// Returns the adjusted safe prefix length with trailing whitespace removed
// if the potential token has LSTRIP.
static iree_host_size_t iree_tokenizer_strip_lstrip_whitespace(
    const iree_tokenizer_special_tokens_t* special_tokens,
    iree_string_view_t input, iree_host_size_t safe,
    iree_tokenizer_special_tokens_match_result_t check_result,
    iree_tokenizer_token_id_t check_id,
    const iree_tokenizer_special_tokens_encode_state_t* check_state) {
  // Get flags for the potential token.
  iree_tokenizer_special_token_flags_t flags = 0;
  if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
    flags = iree_tokenizer_special_tokens_get_flags_for_id(special_tokens,
                                                           check_id);
  } else if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
    // For partial matches, check the token being matched.
    flags = special_tokens->flags[check_state->partial_token_index];
  }

  // If LSTRIP is set, strip trailing Unicode whitespace from the safe prefix.
  if (iree_any_bit_set(flags, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_LSTRIP)) {
    while (safe > 0) {
      // Walk back over UTF-8 continuation bytes (10xxxxxx) to find the
      // leading byte. ASCII bytes (0xxxxxxx) don't match the continuation
      // pattern, so the loop is a no-op for single-byte characters.
      iree_host_size_t start = safe - 1;
      while (start > 0 && ((uint8_t)input.data[start] & 0xC0) == 0x80) {
        --start;
      }
      iree_host_size_t pos = 0;
      iree_string_view_t tail =
          iree_make_string_view(input.data + start, safe - start);
      uint32_t codepoint = iree_unicode_utf8_decode(tail, &pos);
      if (!iree_unicode_is_whitespace(codepoint)) {
        break;
      }
      safe = start;
    }
  }

  return safe;
}

//===----------------------------------------------------------------------===//
// Post-Normalization Special Token Matching
//===----------------------------------------------------------------------===//

// Result from post-normalization special token matching.
typedef enum iree_tokenizer_post_norm_match_result_e {
  // No special token could match anywhere in the view. Segmenter can consume
  // the entire normalized text.
  IREE_TOKENIZER_POST_NORM_NO_MATCH = 0,
  // A special token was matched and emitted. Caller should re-enter loop.
  IREE_TOKENIZER_POST_NORM_MATCHED,
  // Normalized input remains in the ring until token output has credit.
  // Stop the pump without transferring those bytes to the segmenter/model.
  IREE_TOKENIZER_POST_NORM_OUTPUT_FULL,
  // Partial match in progress - waiting for more normalized output. Block
  // segmenter from consuming partial match bytes.
  IREE_TOKENIZER_POST_NORM_NEED_MORE,
  // A special token could start at |out_segment_limit| bytes into the view.
  // Segmenter should only consume bytes up to that limit, then re-check.
  IREE_TOKENIZER_POST_NORM_LIMIT,
  // A confirmed special token ends the ordinary span at out_segment_limit.
  // Earlier model work must retire before the token can be emitted.
  IREE_TOKENIZER_POST_NORM_BOUNDARY,
} iree_tokenizer_post_norm_match_result_t;

// Matches post-normalization special tokens in the ring buffer.
// Called after normalizer writes output, before segmenter reads.
//
// Post-norm tokens have normalized=true in HuggingFace added_tokens.
// They are matched against NORMALIZED text, not raw input. For example, a token
// "yesterday" with a lowercasing normalizer would match input "Yesterday".
//
// The matching region is [segmenter_view_start, write_position) - the
// normalized bytes that haven't been processed by the segmenter yet.
//
// Returns:
//   NO_MATCH: No special token in view. Segmenter can process all bytes.
//   MATCHED: Token emitted. Re-enter loop.
//   NEED_MORE: Partial match. Block segmenter, wait for input.
//   LIMIT: Potential match at *out_segment_limit. Segmenter should only
//          process that many bytes, then re-check.
//   BOUNDARY: Confirmed token at *out_segment_limit ends the ordinary span.
static iree_tokenizer_post_norm_match_result_t
iree_tokenizer_encode_state_match_post_norm_special_tokens(
    iree_tokenizer_encode_state_t* state, const iree_tokenizer_t* tokenizer,
    bool normalized_span_complete, iree_tokenizer_token_output_t output,
    iree_host_size_t* total_tokens, iree_host_size_t* out_segment_limit) {
  *out_segment_limit = IREE_HOST_SIZE_MAX;

  // Skip if no post-norm tokens configured or matching disabled.
  if (iree_tokenizer_special_tokens_is_empty(
          &tokenizer->special_tokens_post_norm) ||
      iree_any_bit_set(state->flags,
                       IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING)) {
    return IREE_TOKENIZER_POST_NORM_NO_MATCH;
  }
  // Skip if no unprocessed normalized output available.
  if (state->write_position <= state->segmenter_view_start) {
    return IREE_TOKENIZER_POST_NORM_NO_MATCH;
  }

  // Get the normalized text view (segmenter_view_start to write_position).
  iree_string_view_t normalized_text = iree_tokenizer_ring_get_view(
      state, state->segmenter_view_start, state->write_position);
  if (normalized_text.size == 0) {
    return IREE_TOKENIZER_POST_NORM_NO_MATCH;
  }
  if (*total_tokens == output.capacity) {
    return IREE_TOKENIZER_POST_NORM_OUTPUT_FULL;
  }

  // When continuing a partial match (NEED_MORE), the match API contract
  // requires that we pass only new bytes, not bytes already consumed. The
  // match_position tracks how many bytes have been matched so far.
  iree_host_size_t already_matched =
      state->special_token_match_post.match_position;
  if (already_matched > 0) {
    // Slice to only new bytes.
    if (already_matched >= normalized_text.size) {
      if (normalized_span_complete) {
        iree_tokenizer_special_tokens_encode_state_clear_partial(
            &state->special_token_match_post);
        return IREE_TOKENIZER_POST_NORM_NO_MATCH;
      }
      // The normalizer still owns input or output that may complete this match.
      return IREE_TOKENIZER_POST_NORM_NEED_MORE;
    }
    normalized_text.data += already_matched;
    normalized_text.size -= already_matched;
  }

  // O(1) rejection: check if any special token could start at this position.
  // If safe_prefix_length > 0, no token starts in the first `safe` bytes.
  iree_host_size_t safe = 0;
  if (already_matched == 0) {
    safe = iree_tokenizer_special_tokens_safe_prefix_length(
        &tokenizer->special_tokens_post_norm, normalized_text);
    if (safe == normalized_text.size) {
      return IREE_TOKENIZER_POST_NORM_NO_MATCH;  // No possible match.
    }
    // If safe > 0, a special token could start at position `safe`. Check if it
    // actually matches/could match before reporting LIMIT. This avoids false
    // LIMITs for characters like `<` in `<=` that look like they could start a
    // token like `<|EOT|>` but don't.
    if (safe > 0) {
      iree_string_view_t check_view = {
          .data = normalized_text.data + safe,
          .size = normalized_text.size - safe,
      };
      iree_host_size_t check_length = 0;
      iree_tokenizer_token_id_t check_id = 0;
      iree_tokenizer_special_tokens_encode_state_t check_state = {0};
      iree_tokenizer_special_tokens_match_result_t check_result =
          iree_tokenizer_special_tokens_match(
              &tokenizer->special_tokens_post_norm, check_view, &check_length,
              &check_id, &check_state);
      if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH) {
        // Not a special token (e.g., `<=` is not `<|EOT|>`). Let segmenter see
        // all bytes including the `<`.
        return IREE_TOKENIZER_POST_NORM_NO_MATCH;
      }
      // After normalization completes, NEED_MORE means the partial prefix
      // will never become a complete special token. Treat it as
      // NO_MATCH so the segmenter can process all bytes including the potential
      // prefix. This prevents " <" at end-of-input from being split into " " +
      // "<".
      if (check_result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE &&
          normalized_span_complete) {
        return IREE_TOKENIZER_POST_NORM_NO_MATCH;
      }
      // A special token matches or could match at position `safe`. Strip
      // trailing whitespace if the token has LSTRIP - this prevents the
      // whitespace from being tokenized when it should be consumed by the
      // special token match (RoBERTa/BART <mask> with lstrip=true).
      iree_host_size_t original_safe = safe;
      safe = iree_tokenizer_strip_lstrip_whitespace(
          &tokenizer->special_tokens_post_norm, normalized_text, safe,
          check_result, check_id, &check_state);
      bool confirmed = check_result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED;
      if (safe == 0 && original_safe > 0 && confirmed &&
          !state->has_partial_segment) {
        // All prefix was whitespace. Skip those bytes and let the match
        // proceed on the next call. Update prev_byte context so LSTRIP check
        // succeeds when using the real state.
        uint8_t last_whitespace =
            (uint8_t)normalized_text.data[original_safe - 1];
        state->special_token_match_post.prev_byte_plus_one =
            last_whitespace + 1;
        state->special_token_match_post.at_start_of_input = false;
        state->segmenter_view_start += original_safe;
        // Advance read_position if no segments are pending.
        if (state->segment_count == state->segments_consumed) {
          state->read_position = state->segmenter_view_start;
        }
      }
      // Let the segmenter process the prefix first. Only a confirmed token
      // completes that span; a possible prefix only limits available bytes.
      *out_segment_limit = safe;
      return confirmed ? IREE_TOKENIZER_POST_NORM_BOUNDARY
                       : IREE_TOKENIZER_POST_NORM_LIMIT;
    }
  }

  // Attempt to match (handles both fresh and continued partial matches).
  iree_host_size_t match_length = 0;
  iree_tokenizer_token_id_t match_id = 0;
  iree_tokenizer_special_tokens_encode_state_t match_state =
      state->special_token_match_post;
  iree_tokenizer_special_tokens_match_result_t result =
      iree_tokenizer_special_tokens_match(&tokenizer->special_tokens_post_norm,
                                          normalized_text, &match_length,
                                          &match_id, &match_state);

  if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED &&
      state->has_partial_segment) {
    *out_segment_limit = 0;
    return IREE_TOKENIZER_POST_NORM_BOUNDARY;
  }
  state->special_token_match_post = match_state;

  if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
    // Emit the special token.
    output.token_ids[*total_tokens] = match_id;
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor, output,
                                                 *total_tokens, 1);
    // Post-norm special tokens don't map directly to raw input bytes (they
    // matched normalized text), so use zero-length offsets like the
    // postprocessor does for added special tokens.
    if (output.token_offsets) {
      output.token_offsets[*total_tokens].start = 0;
      output.token_offsets[*total_tokens].end = 0;
    }
    (*total_tokens)++;
    // Advance segmenter_view_start past the matched bytes - the segmenter
    // won't see them. Include any bytes from a partial match continuation.
    iree_host_size_t skipped_bytes = already_matched + match_length;
    state->segmenter_view_start += skipped_bytes;
    // Advance read_position if no segments are pending. Pending segments
    // reference physical offsets starting at read_position - advancing it
    // prematurely would "free" memory the model still needs, corrupting data
    // if the ring wraps.
    if (state->segment_count == state->segments_consumed) {
      state->read_position = state->segmenter_view_start;
    }
    return IREE_TOKENIZER_POST_NORM_MATCHED;
  }

  if (result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
    // Partial match in progress.
    if (normalized_span_complete) {
      // After normalization completes, this cannot become a complete special
      // token. Clear any partial state and return NO_MATCH so the
      // segmenter processes these bytes normally.
      if (state->special_token_match_post.match_position > 0) {
        iree_tokenizer_special_tokens_encode_state_clear_partial(
            &state->special_token_match_post);
      }
      return IREE_TOKENIZER_POST_NORM_NO_MATCH;
    }
    // Wait for more normalized output. Don't advance any positions - the bytes
    // are still in the ring and will be re-examined once more input arrives.
    return IREE_TOKENIZER_POST_NORM_NEED_MORE;
  }

  // NO_MATCH: Let the segmenter process these bytes normally.
  // If there was a partial match, clear it - those bytes aren't a special
  // token.
  if (state->special_token_match_post.match_position > 0) {
    iree_tokenizer_special_tokens_encode_state_clear_partial(
        &state->special_token_match_post);
  }
  return IREE_TOKENIZER_POST_NORM_NO_MATCH;
}

// Attempts to emit a deferred special token.
// Returns true if a token was emitted (caller should re-pump).
static bool iree_tokenizer_try_emit_pending_special_token(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* total_tokens) {
  if (state->pending_special_token < 0) {
    return false;
  }

  // Only emit when pipeline is flushed: no ring buffer content, no segments,
  // and no normalizer-buffered data (e.g., NFC combining sequences).
  bool pipeline_flushed =
      !iree_tokenizer_encode_state_pipeline_has_content(state);
  if (!pipeline_flushed) {
    return false;
  }

  // Emit the deferred token.
  if (output.capacity > *total_tokens) {
    output.token_ids[*total_tokens] = state->pending_special_token;
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor, output,
                                                 *total_tokens, 1);
    (*total_tokens)++;
    state->pending_special_token = -1;
    return true;
  }
  return false;
}

// Internal helper: run one iteration of the encode pipeline.
// Sets |out_made_progress| to true if data moved through the pipeline.
static iree_status_t iree_tokenizer_encode_state_pump(
    iree_tokenizer_encode_state_t* state, iree_string_view_t* chunk,
    iree_tokenizer_token_output_t* output, iree_host_size_t* total_tokens,
    bool* out_made_progress) {
  const iree_tokenizer_t* tokenizer = state->tokenizer;
  *out_made_progress = false;

  // Reclaim bytes consumed without producing segments (for example whitespace).
  // Logical positions continue across an empty ring; only stream reset starts
  // a new coordinate space. Mirrored storage permits a full-capacity write at
  // any physical position.
  if (state->segment_count == state->segments_consumed &&
      state->segmenter_view_start > state->read_position) {
    state->read_position = state->segmenter_view_start;
  }

  // Tokenize pending segments. Segments are produced by the segmenter and
  // queued for the model. Process them first to make room in the segment
  // buffer.
  if (state->segment_count > state->segments_consumed &&
      output->capacity > *total_tokens) {
    // Keep logical segment positions intact across output backpressure. The
    // segmenter's contiguous view has already prepared any mirrored bytes.
    iree_host_size_t pending_count =
        state->segment_count - state->segments_consumed;
    iree_tokenizer_segment_t* pending_base =
        &state->segments[state->segments_consumed];
    iree_tokenizer_segment_list_t pending_segments = {
        .count = pending_count,
        .values = pending_base,
        .buffer_offset = state->read_position & ~state->capacity_mask,
    };
    iree_tokenizer_token_output_t sub_output = {
        .capacity = output->capacity - *total_tokens,
        .token_ids = &output->token_ids[*total_tokens],
        .token_offsets = output->token_offsets
                             ? &output->token_offsets[*total_tokens]
                             : NULL,
        .type_ids = output->type_ids ? &output->type_ids[*total_tokens] : NULL,
    };

    iree_host_size_t segments_consumed = 0;
    iree_host_size_t tokens_written = 0;
    // The allocation's logical origin maps segments to storage, including the
    // mirror region for segments crossing the physical wrap boundary.
    IREE_RETURN_IF_ERROR(iree_tokenizer_model_state_encode(
        state->model_state,
        iree_make_const_byte_span(state->transform_buffer.data,
                                  state->transform_buffer.data_length),
        pending_segments, sub_output, &segments_consumed, &tokens_written));

    // Assign type_ids to model-produced tokens based on sequence phase.
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor,
                                                 sub_output, 0, tokens_written);
    // Trim offsets on model tokens if requested (ByteLevel/RoBERTa behavior).
    iree_tokenizer_postprocessor_trim_token_offsets(
        &state->postprocessor, state->tokenizer->vocab, sub_output, 0,
        tokens_written);

    state->segments_consumed += segments_consumed;
    *total_tokens += tokens_written;
    if (segments_consumed > 0 || tokens_written > 0) {
      *out_made_progress = true;
    }

    // If all segments consumed, advance read pointer to reclaim space.
    // No data movement - ring buffer just advances the read position.
    if (state->segments_consumed >= state->segment_count) {
      state->segment_count = 0;
      state->segments_consumed = 0;
      // Reclaim bytes up to segmenter_view_start. This includes both bytes
      // that were converted to segments and bytes consumed by the segmenter
      // without producing segments (buffering for word boundaries, etc.).
      state->read_position = state->segmenter_view_start;
    }
  }

  // Queued model output precedes every later span and added token.
  if (state->segment_count > state->segments_consumed) {
    return iree_ok_status();
  }

  // A raw token boundary or final input completes normalization only after
  // every retained raw prefix byte has been accepted.
  bool finishing_normalizer =
      state->phase == IREE_TOKENIZER_ENCODE_PHASE_NORMALIZE &&
      !iree_tokenizer_special_tokens_encode_state_has_partial(
          &state->special_token_match);
  if (state->pending_special_token >= 0 || finishing_normalizer) {
    if (state->normalizer_state) {
      iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
          state->write_position, state->capacity_mask);
      iree_mutable_string_view_t norm_output = iree_make_mutable_string_view(
          (char*)state->transform_buffer.data + physical_write,
          iree_tokenizer_ring_writable_space(state));
      iree_host_size_t bytes_written = 0;
      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_finalize(
          state->normalizer_state, norm_output, &bytes_written));
      iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                           bytes_written);
      state->write_position += bytes_written;
      *out_made_progress |= bytes_written > 0;
    }
    if (finishing_normalizer && (!state->normalizer_state ||
                                 !iree_tokenizer_normalizer_state_has_pending(
                                     state->normalizer_state))) {
      state->phase = IREE_TOKENIZER_ENCODE_PHASE_MATCH;
      *out_made_progress = true;
    }
  }

  // Emit the retained token after the preceding span and pipeline have drained.
  if (iree_tokenizer_try_emit_pending_special_token(state, *output,
                                                    total_tokens)) {
    *out_made_progress = true;
    return iree_ok_status();
  }

  // Once the segmenter receives EOF, only its queued segments may reach the
  // model. Re-entering process would reopen a completed producer.
  if (state->phase >= IREE_TOKENIZER_ENCODE_PHASE_SEGMENT) {
    return iree_ok_status();
  }

  bool normalized_span_complete =
      state->phase >= IREE_TOKENIZER_ENCODE_PHASE_MATCH ||
      (state->pending_special_token >= 0 &&
       (!state->normalizer_state ||
        !iree_tokenizer_normalizer_state_has_pending(state->normalizer_state)));

  // Match post-normalization special tokens before segmenter runs.
  // These tokens have normalized=true and are matched after the normalizer
  // transforms the input. If matched, the token is emitted directly and we
  // re-enter the loop to check for more matches or continue processing.
  //
  // IMPORTANT: Skip post-norm matching when the segmenter has pending state.
  // When the segmenter returns consumed=0 with pending internal state (e.g.,
  // sequence segmenter mid-expansion), it stores absolute byte offsets into
  // the input view. Post-norm matching can advance segmenter_view_start
  // (LSTRIP whitespace skip or matched token skip), which shrinks the ring
  // view. The segmenter's stored offsets then exceed the new view size,
  // causing unsigned underflow when computing remaining bytes. Let the
  // segmenter finish its pending work first; post-norm matching will run on
  // the next pump iteration after the segmenter commits its consumed bytes.
  iree_host_size_t post_norm_segment_limit = IREE_HOST_SIZE_MAX;
  bool post_norm_boundary = false;
  if (!(state->segmenter_state &&
        iree_tokenizer_segmenter_state_has_pending(state->segmenter_state))) {
    iree_tokenizer_post_norm_match_result_t post_norm_result =
        iree_tokenizer_encode_state_match_post_norm_special_tokens(
            state, tokenizer, normalized_span_complete, *output, total_tokens,
            &post_norm_segment_limit);
    switch (post_norm_result) {
      case IREE_TOKENIZER_POST_NORM_MATCHED:
        // Token emitted. Re-enter loop to check for more matches.
        *out_made_progress = true;
        return iree_ok_status();
      case IREE_TOKENIZER_POST_NORM_OUTPUT_FULL:
        return iree_ok_status();
      case IREE_TOKENIZER_POST_NORM_NEED_MORE:
        // Partial match - waiting for more input. Block segmenter by setting
        // limit to 0, but continue to normalize step so we can accumulate more
        // bytes in the ring buffer. The next pump iteration will retry the
        // match with more data.
        post_norm_segment_limit = 0;
        break;
      case IREE_TOKENIZER_POST_NORM_LIMIT:
        // Special token could start at post_norm_segment_limit. Let segmenter
        // process only that many bytes (handled below).
        break;
      case IREE_TOKENIZER_POST_NORM_BOUNDARY:
        post_norm_boundary = true;
        break;
      case IREE_TOKENIZER_POST_NORM_NO_MATCH:
        // No special token. Segmenter can process all bytes.
        break;
    }
  }

  // Segment normalized text. Requires: segment buffer has room, ring has data.
  // get_segment_view handles wrap-around by copying to mirror region if needed.
  // Truncate the view to exclude any incomplete UTF-8 trailing sequence. The
  // ring may contain partial codepoints when input chunks split mid-sequence
  // (passthrough path) - the segmenter must only see complete codepoints.
  //
  // When partial mode is active, the segmenter is bypassed: those bytes are
  // being fed directly to the model. Continuing to feed the segmenter would
  // cause it to produce segments for bytes already processed, with positions
  // referencing reclaimed ring buffer space.
  iree_string_view_t normalized_text = iree_tokenizer_ring_get_view(
      state, state->segmenter_view_start, state->write_position);
  if (normalized_text.size > 0) {
    normalized_text.size -= iree_unicode_utf8_incomplete_tail_length(
        normalized_text.data, normalized_text.size);
  }
  // Apply post-norm segment limit: if a special token could start at position
  // N, only let the segmenter see bytes [0, N). After processing, post-norm
  // matching will get another chance to match at the new segmenter_view_start.
  if (post_norm_segment_limit < normalized_text.size) {
    normalized_text.size = post_norm_segment_limit;
  }
  iree_host_size_t normalized_end =
      state->segmenter_view_start + normalized_text.size;
  bool span_complete = post_norm_boundary || normalized_span_complete;
  if (!state->has_partial_segment &&
      state->segment_count < tokenizer->segment_batch_size &&
      normalized_text.size > 0) {
    iree_tokenizer_segment_output_t segment_output = {
        .capacity = tokenizer->segment_batch_size - state->segment_count,
        .values = &state->segments[state->segment_count],
    };

    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t segments_produced = 0;
    if (state->segmenter_state) {
      IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_process(
          state->segmenter_state, normalized_text, segment_output,
          &bytes_consumed, &segments_produced));
    } else if (span_complete) {
      // Without a pre-tokenizer the entire unmatched span is one segment.
      // Producer output chunks do not establish segmentation boundaries.
      segment_output.values[0].start = 0;
      segment_output.values[0].end = normalized_text.size;
      bytes_consumed = normalized_text.size;
      segments_produced = 1;
    }

    // Only an established producer boundary can finish a pending segment.
    // A possible added-token prefix reserves bytes without ending the span.
    if (bytes_consumed == 0 && segments_produced == 0 && span_complete &&
        segment_output.capacity > 0 && state->segmenter_state) {
      IREE_RETURN_IF_ERROR(iree_tokenizer_segmenter_state_finalize(
          state->segmenter_state, normalized_text, segment_output,
          &segments_produced));
      bytes_consumed = normalized_text.size;
    }

    // Capture segmenter_view_start before mutation for offset adjustment.
    // Segments use this captured value to convert from relative to absolute
    // offsets. This makes segments immune to later state mutations (e.g.,
    // post-norm special token matching advancing segmenter_view_start).
    iree_host_size_t view_start = state->segmenter_view_start;

    // Convert segment offsets from relative to absolute (logical cumulative).
    // Models receive these unchanged with the logical origin of their buffer.
    for (iree_host_size_t i = 0; i < segments_produced; ++i) {
      state->segments[state->segment_count + i].start += view_start;
      state->segments[state->segment_count + i].end += view_start;
    }

    // Advance segmenter_view_start for next call.
    state->segmenter_view_start += bytes_consumed;
    state->segment_count += segments_produced;
    if (bytes_consumed > 0 || segments_produced > 0) {
      *out_made_progress = true;
    }
  }

  // Match special tokens in raw input (runs BEFORE normalizer).
  // Special tokens like <|endoftext|> must be matched literally before the
  // normalizer or pre-tokenizer could transform them.
  //
  // The match() function handles continuation: if state->match_position > 0,
  // we're continuing a partial match and |chunk| contains only NEW bytes.
  // On NEED_MORE, we consume the bytes and track progress in state. The token
  // content itself serves as the "buffer" for reconstruction via get_partial().
  iree_host_size_t normalize_limit = IREE_HOST_SIZE_MAX;
  if (!iree_tokenizer_special_tokens_is_empty(&tokenizer->special_tokens) &&
      !iree_any_bit_set(state->flags,
                        IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING) &&
      chunk->size > 0 && state->pending_special_token < 0 &&
      state->special_token_match.drain_position == 0) {
    // Continuing a partial match? Pass chunk directly to match().
    // Starting fresh? Check if first byte could start a special token.
    bool should_match = state->special_token_match.match_position > 0;
    if (!should_match) {
      iree_host_size_t safe = iree_tokenizer_special_tokens_safe_prefix_length(
          &tokenizer->special_tokens,
          iree_make_string_view(chunk->data, chunk->size));
      if (safe == 0) {
        should_match = true;
      } else {
        // Check if the potential token at position safe has LSTRIP.
        // If so, strip trailing whitespace from the normalize limit to prevent
        // the whitespace from being tokenized (RoBERTa/BART <mask> with
        // lstrip=true).
        if (safe < chunk->size) {
          iree_string_view_t check_view = {
              .data = chunk->data + safe,
              .size = chunk->size - safe,
          };
          iree_host_size_t check_length = 0;
          iree_tokenizer_token_id_t check_id = 0;
          iree_tokenizer_special_tokens_encode_state_t check_state = {0};
          iree_tokenizer_special_tokens_match_result_t check_result =
              iree_tokenizer_special_tokens_match(&tokenizer->special_tokens,
                                                  check_view, &check_length,
                                                  &check_id, &check_state);
          if (check_result != IREE_TOKENIZER_SPECIAL_TOKENS_NO_MATCH) {
            iree_host_size_t original_safe = safe;
            safe = iree_tokenizer_strip_lstrip_whitespace(
                &tokenizer->special_tokens,
                iree_make_string_view(chunk->data, chunk->size), safe,
                check_result, check_id, &check_state);
            if (safe == 0 && original_safe > 0) {
              // All prefix was whitespace. Skip those bytes and match the
              // special token directly. The whitespace is consumed by LSTRIP.
              // Update prev_byte context to be the last whitespace byte so
              // the LSTRIP match succeeds when using the real state.
              uint8_t last_whitespace = (uint8_t)chunk->data[original_safe - 1];
              state->special_token_match.prev_byte_plus_one =
                  last_whitespace + 1;
              state->special_token_match.at_start_of_input = false;
              chunk->data += original_safe;
              chunk->size -= original_safe;
              *out_made_progress = true;
              should_match = true;
            }
            // If safe > 0 (partial whitespace stripped), we just reduce the
            // normalize limit to exclude trailing whitespace before the token.
          }
        }
        if (!should_match) {
          normalize_limit = safe;
        }
      }
    }

    if (should_match) {
      iree_host_size_t match_length = 0;
      iree_tokenizer_token_id_t match_id = 0;
      iree_tokenizer_special_tokens_match_result_t result =
          iree_tokenizer_special_tokens_match(
              &tokenizer->special_tokens,
              iree_make_string_view(chunk->data, chunk->size), &match_length,
              &match_id, &state->special_token_match);

      if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
        // Track if this special token matched at position 0 of the original
        // input. This affects prepend_scheme="first" semantics: text following
        // a position-0 special token should NOT get metaspace prepended.
        if (state->special_token_match.at_start_of_input) {
          state->first_consumed_by_special_token = true;
        }

        // Consume matched bytes from input.
        chunk->data += match_length;
        chunk->size -= match_length;
        *out_made_progress = true;

        // rstrip: consume trailing Unicode whitespace after the matched token.
        if (iree_any_bit_set(state->special_token_match.matched_flags,
                             IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_RSTRIP)) {
          while (chunk->size > 0) {
            iree_host_size_t pos = 0;
            iree_string_view_t remaining =
                iree_make_string_view(chunk->data, chunk->size);
            uint32_t codepoint = iree_unicode_utf8_decode(remaining, &pos);
            if (!iree_unicode_is_whitespace(codepoint)) {
              break;
            }
            chunk->data += pos;
            chunk->size -= pos;
          }
        }

        // Retain the accepted match until earlier content has drained and an
        // output slot is available. Consuming input transfers ownership even
        // when the caller currently has no output credit.
        bool pipeline_has_content =
            iree_tokenizer_encode_state_pipeline_has_content(state);
        if (state->normalizer_state || pipeline_has_content ||
            output->capacity == *total_tokens) {
          state->pending_special_token = match_id;
        } else {
          output->token_ids[*total_tokens] = match_id;
          iree_tokenizer_postprocessor_assign_type_ids(
              &state->postprocessor, *output, *total_tokens, 1);
          (*total_tokens)++;
        }
        return iree_ok_status();
      } else if (result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
        // Partial match. Consume bytes (tracked in state), wait for more.
        chunk->data += chunk->size;
        chunk->size = 0;
        *out_made_progress = true;
        return iree_ok_status();
      }
      // NO_MATCH. If we had a partial match in progress, recover via finalize
      // or drain the partial bytes through the normalizer.
      if (state->special_token_match.match_position > 0) {
        // Recover the partial match through the normalizer below.
        // get_partial() reconstructs the bytes from the token data itself.
        normalize_limit = 0;
      } else {
        // Match at position 0 definitively failed (not NEED_MORE). Find where
        // the next potential special token could start and normalize up to it.
        // This allows normalizing e.g. "<=" fully when we know it's not
        // "<|...".
        //
        // Without this, we'd normalize only 1 byte (the '<'), causing post-norm
        // matching to see just '<' and return NEED_MORE (could be start of
        // special token), which incorrectly splits tokens like " <=" into " "
        // and "<=".
        if (chunk->size > 1) {
          iree_host_size_t next_safe =
              iree_tokenizer_special_tokens_safe_prefix_length(
                  &tokenizer->special_tokens,
                  iree_make_string_view(chunk->data + 1, chunk->size - 1));
          normalize_limit = 1 + next_safe;
        } else {
          normalize_limit = 1;
        }
      }
    }
  }

  // A failed raw match is ordinary input. Its existing drain cursor owns
  // accepted bytes until normalization consumes them, including during EOF.
  if (state->special_token_match.match_position > 0 &&
      (normalize_limit == 0 || state->special_token_match.drain_position > 0 ||
       state->phase == IREE_TOKENIZER_ENCODE_PHASE_NORMALIZE)) {
    iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
    if (write_space > 0) {
      uint8_t partial_buffer[256 + 4];
      iree_host_size_t partial_size =
          iree_tokenizer_special_tokens_encode_state_get_partial(
              &state->special_token_match, &tokenizer->special_tokens,
              partial_buffer);
      iree_host_size_t drain_offset = state->special_token_match.drain_position;
      iree_host_size_t retained = partial_size - drain_offset;
      // A raw match may end inside a codepoint. Borrow only enough following
      // input to complete it; those bytes advance with normalizer acceptance.
      iree_host_size_t appended = 0;
      iree_host_size_t incomplete = iree_unicode_utf8_incomplete_tail_length(
          (const char*)partial_buffer + drain_offset, retained);
      if (incomplete > 0) {
        iree_host_size_t required =
            iree_unicode_utf8_sequence_length(
                partial_buffer[partial_size - incomplete]) -
            incomplete;
        appended = iree_min(chunk->size, required);
        memcpy(partial_buffer + partial_size, chunk->data, appended);
      }
      iree_string_view_t recovered = iree_make_string_view(
          (const char*)partial_buffer + drain_offset, retained + appended);
      recovered.size -= iree_unicode_utf8_incomplete_tail_length(
          recovered.data, recovered.size);
      if (recovered.size == 0 &&
          state->phase == IREE_TOKENIZER_ENCODE_PHASE_NORMALIZE) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "encode input ends with incomplete UTF-8");
      }
      iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
          state->write_position, state->capacity_mask);
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t bytes_written = 0;
      if (state->normalizer_state) {
        iree_mutable_string_view_t norm_output = iree_make_mutable_string_view(
            (char*)state->transform_buffer.data + physical_write, write_space);
        IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
            state->normalizer_state, recovered, norm_output,
            IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &bytes_consumed,
            &bytes_written));
      } else {
        bytes_consumed = iree_min(recovered.size, write_space);
        bytes_written = bytes_consumed;
        memcpy(state->transform_buffer.data + physical_write, recovered.data,
               bytes_written);
      }
      iree_tokenizer_ring_fixup_write_wrap(state, physical_write,
                                           bytes_written);
      state->write_position += bytes_written;
      state->special_token_match.drain_position +=
          (uint16_t)iree_min(bytes_consumed, retained);
      if (bytes_consumed > retained) {
        *chunk = iree_string_view_substr(*chunk, bytes_consumed - retained,
                                         IREE_HOST_SIZE_MAX);
      }
      if (state->special_token_match.drain_position == partial_size) {
        iree_tokenizer_special_tokens_encode_state_clear_partial(
            &state->special_token_match);
      }
      *out_made_progress |= bytes_consumed > 0 || bytes_written > 0;
      return iree_ok_status();
    }
  }

  // Normalize input and write to ring buffer. Requires: ring has write space.
  // If no normalizer is configured, copy input directly (passthrough).
  //
  // Skip normalization when there's a pending special token waiting to emit.
  // This forces the pump to drain existing ring content first, ensuring correct
  // output order: [content before special token, special token, content after].
  iree_host_size_t write_space = iree_tokenizer_ring_writable_space(state);
  if (write_space > 0 && chunk->size > 0 && normalize_limit > 0 &&
      state->pending_special_token < 0) {
    iree_host_size_t physical_write = iree_tokenizer_ring_to_physical(
        state->write_position, state->capacity_mask);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t bytes_written = 0;

    if (state->normalizer_state) {
      // Normalizer requires complete UTF-8 codepoints (see normalizer.h
      // contract). Truncate the view to exclude any incomplete trailing
      // sequence so the normalizer always receives valid boundaries.
      // Also respect normalize_limit from special token matching.
      iree_host_size_t max_consume =
          iree_min(iree_min(chunk->size, write_space), normalize_limit);
      iree_host_size_t incomplete_tail =
          iree_unicode_utf8_incomplete_tail_length(chunk->data, max_consume);
      max_consume -= incomplete_tail;
      if (max_consume == 0) {
        // Entire consumable region is an incomplete sequence. No progress
        // until more input arrives to complete the codepoint.
        return iree_ok_status();
      }
      iree_string_view_t safe_chunk =
          iree_make_string_view(chunk->data, max_consume);
      iree_mutable_string_view_t norm_output = {
          .data = (char*)state->transform_buffer.data + physical_write,
          .size = write_space,
      };
      iree_tokenizer_normalizer_flags_t flags =
          IREE_TOKENIZER_NORMALIZER_FLAG_NONE;

      // Signal to prepend normalizer that position 0 was consumed by a special
      // token, so prepend_scheme="first" should skip prepending for this text.
      if (state->first_consumed_by_special_token) {
        flags |= IREE_TOKENIZER_NORMALIZER_FLAG_FIRST_CONSUMED;
        // Clear after passing: only the first normalization gets this flag.
        state->first_consumed_by_special_token = false;
      }

      IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_state_process(
          state->normalizer_state, safe_chunk, norm_output, flags,
          &bytes_consumed, &bytes_written));
    } else {
      // Passthrough: copy raw bytes directly to the ring buffer. Partial
      // UTF-8 sequences at the end are fine here - the ring accumulates bytes
      // across feed() calls, and Step 2 truncates the segmenter view to
      // exclude any incomplete trailing codepoint.
      // Also respect normalize_limit from special token matching.
      bytes_consumed =
          iree_min(iree_min(chunk->size, write_space), normalize_limit);
      memcpy((char*)state->transform_buffer.data + physical_write, chunk->data,
             bytes_consumed);
      bytes_written = bytes_consumed;
    }

    iree_tokenizer_ring_fixup_write_wrap(state, physical_write, bytes_written);
    chunk->data += bytes_consumed;
    chunk->size -= bytes_consumed;
    state->write_position += bytes_written;
    if (bytes_consumed > 0 || bytes_written > 0) {
      *out_made_progress = true;
    }
  }

  // Handle ring full with no segment boundaries. When no progress was made,
  // has data but no segment boundaries (split=false), enter partial segment
  // mode. The model processes ring data directly, emitting frozen tokens and
  // reclaiming bytes to free ring space for more normalization.
  if (!*out_made_progress &&
      (chunk->size > 0 ||
       state->phase == IREE_TOKENIZER_ENCODE_PHASE_NORMALIZE) &&
      normalized_end > state->read_position &&
      state->segment_count <= state->segments_consumed) {
    state->has_partial_segment = true;
  }
  if (state->has_partial_segment &&
      (normalized_end > state->read_position || span_complete) &&
      output->capacity > *total_tokens) {
    iree_tokenizer_token_output_t sub_output = {
        .capacity = output->capacity - *total_tokens,
        .token_ids = &output->token_ids[*total_tokens],
        .token_offsets = output->token_offsets
                             ? &output->token_offsets[*total_tokens]
                             : NULL,
        .type_ids = output->type_ids ? &output->type_ids[*total_tokens] : NULL,
    };
    iree_host_size_t tokens_written = 0;
    bool segment_complete = false;
    IREE_RETURN_IF_ERROR(iree_tokenizer_encode_ring_as_segment(
        state, normalized_end, /*is_partial=*/!span_complete, sub_output,
        &tokens_written, &segment_complete));
    iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor,
                                                 sub_output, 0, tokens_written);
    iree_tokenizer_postprocessor_trim_token_offsets(
        &state->postprocessor, state->tokenizer->vocab, sub_output, 0,
        tokens_written);
    *total_tokens += tokens_written;

    iree_host_size_t reclaimed =
        iree_tokenizer_model_state_reclaim(state->model_state);
    if (reclaimed > 0) {
      state->read_position += reclaimed;
      // Keep segmenter_view_start in sync: those bytes were processed by the
      // model directly, not through the segmenter. Without this, the segmenter
      // view (segmenter_view_start to write_position) would include reclaimed
      // bytes, producing segments with out-of-bounds positions.
      if (state->segmenter_view_start < state->read_position) {
        state->segmenter_view_start = state->read_position;
      }
    }
    // When a forced-complete segment finishes, retire the entire ring region
    // that was just fed directly to the model. Unlike the partial reclaim path
    // above, a non-partial completion resets the model to SEGMENT_START, so
    // state_reclaim() returns 0 even though all bytes were fully processed.
    // If we leave the ring positions unchanged, the same bytes are re-encoded
    // on the next pump iteration and a pending special token can never flush.
    if (span_complete && segment_complete) {
      state->read_position = normalized_end;
      state->segmenter_view_start = normalized_end;
      state->has_partial_segment = false;
    }
    if (reclaimed > 0 || tokens_written > 0 || segment_complete) {
      *out_made_progress = true;
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_tokenizer_encode_state_feed_internal(
    iree_tokenizer_encode_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_output_t output, bool is_final_input,
    iree_host_size_t* out_bytes_consumed, iree_host_size_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(output.token_ids);
  IREE_ASSERT_ARGUMENT(out_bytes_consumed);
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_bytes_consumed = 0;
  *out_token_count = 0;

  if (state->phase != IREE_TOKENIZER_ENCODE_PHASE_PROCESS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "encode input is already finalized; reset before feed");
  }

  // Verify required pipeline stages are configured.
  // Note: segmenter_state may be NULL (passthrough mode).
  if (!state->model_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "encode pipeline not fully configured");
  }

  iree_host_size_t original_chunk_size = chunk.size;
  iree_host_size_t total_tokens = 0;

  // Emit prefix special tokens before any model tokens.
  total_tokens += iree_tokenizer_postprocessor_emit_prefix(
      &state->postprocessor, output, total_tokens);

  // Pull-based processing: pump the pipeline until no more progress or error.
  bool made_progress = true;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && made_progress) {
    status = iree_tokenizer_encode_state_pump(state, &chunk, &output,
                                              &total_tokens, &made_progress);
  }

  *out_bytes_consumed = original_chunk_size - chunk.size;
  *out_token_count = total_tokens;

  // Only complete-input callers know that no further input or output capacity
  // will arrive. A streaming caller retains unconsumed bytes while waiting for
  // more input (including a split UTF-8 codepoint) or output credit.
  if (iree_status_is_ok(status) && is_final_input && chunk.size > 0 &&
      *out_bytes_consumed == 0 && total_tokens == 0) {
    if (output.capacity == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "encode: output buffer full with input remaining "
                              "(pending_input=%" PRIhsz " bytes)",
                              chunk.size);
    }
    if (iree_unicode_utf8_incomplete_tail_length(chunk.data, chunk.size) ==
        chunk.size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "encode input ends with incomplete UTF-8 sequence "
          "(%" PRIhsz " trailing bytes)",
          chunk.size);
    }
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "encode pump invariant violated: no progress with output room "
        "and no pending pipeline data "
        "(logical_capacity=%" PRIhsz " bytes, used=%" PRIhsz
        " bytes, pending_input=%" PRIhsz " bytes, has_partial=%" PRIu32 ")",
        state->capacity_mask + 1, state->write_position - state->read_position,
        chunk.size, (uint32_t)state->has_partial_segment);
  }

  return status;
}

iree_status_t iree_tokenizer_encode_state_feed(
    iree_tokenizer_encode_state_t* state, iree_string_view_t chunk,
    iree_tokenizer_token_output_t output, iree_host_size_t* out_bytes_consumed,
    iree_host_size_t* out_token_count) {
  return iree_tokenizer_encode_state_feed_internal(
      state, chunk, output, /*is_final_input=*/false, out_bytes_consumed,
      out_token_count);
}

static iree_status_t iree_tokenizer_encode_state_finalize_internal(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_tokenizer_encode_state_finalize_flags_t finalize_flags,
    iree_host_size_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(state);
  IREE_ASSERT_ARGUMENT(output.token_ids);
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;
  if (!state->model_state) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "encode pipeline not fully configured");
  }
  if (state->phase == IREE_TOKENIZER_ENCODE_PHASE_PROCESS) {
    state->phase = IREE_TOKENIZER_ENCODE_PHASE_NORMALIZE;
  }

  iree_host_size_t total_tokens = iree_tokenizer_postprocessor_emit_prefix(
      &state->postprocessor, output, 0);
  iree_status_t status = iree_ok_status();
  bool made_progress = true;
  while (iree_status_is_ok(status) && made_progress &&
         state->phase != IREE_TOKENIZER_ENCODE_PHASE_DONE) {
    made_progress = false;
    if (state->postprocessor.phase ==
        IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX) {
      break;
    }
    if (state->phase <= IREE_TOKENIZER_ENCODE_PHASE_MODEL) {
      iree_string_view_t empty = iree_string_view_empty();
      status = iree_tokenizer_encode_state_pump(state, &empty, &output,
                                                &total_tokens, &made_progress);
      if (!iree_status_is_ok(status) || made_progress) {
        continue;
      }
      if (state->segment_count > state->segments_consumed) {
        break;
      }
      if (state->pending_special_token >= 0 &&
          total_tokens == output.capacity) {
        break;
      }
    }

    if (state->phase == IREE_TOKENIZER_ENCODE_PHASE_MATCH) {
      if (total_tokens == output.capacity &&
          state->write_position > state->segmenter_view_start) {
        break;
      }
      if (state->has_partial_segment) {
        break;
      }
      state->phase = IREE_TOKENIZER_ENCODE_PHASE_SEGMENT;
      made_progress = true;
    }
    if (state->phase == IREE_TOKENIZER_ENCODE_PHASE_SEGMENT) {
      if (state->segmenter_state && !state->has_partial_segment) {
        iree_tokenizer_segment_output_t segment_output = {
            .capacity = state->tokenizer->segment_batch_size,
            .values = state->segments,
        };
        iree_host_size_t segments_produced = 0;
        status = iree_tokenizer_segmenter_state_finalize(
            state->segmenter_state,
            iree_tokenizer_ring_get_view(state, state->segmenter_view_start,
                                         state->write_position),
            segment_output, &segments_produced);
        if (iree_status_is_ok(status)) {
          for (iree_host_size_t i = 0; i < segments_produced; ++i) {
            state->segments[i].start += state->segmenter_view_start;
            state->segments[i].end += state->segmenter_view_start;
          }
          state->segment_count = segments_produced;
          made_progress |= segments_produced > 0;
        }
      }
      if (iree_status_is_ok(status) &&
          (state->has_partial_segment || !state->segmenter_state ||
           !iree_tokenizer_segmenter_state_has_pending(
               state->segmenter_state))) {
        state->segmenter_view_start = state->write_position;
        state->phase = IREE_TOKENIZER_ENCODE_PHASE_MODEL;
        made_progress = true;
      }
      continue;
    }
    if (state->phase == IREE_TOKENIZER_ENCODE_PHASE_MODEL) {
      iree_tokenizer_token_output_t remaining =
          iree_tokenizer_sub_token_output(output, total_tokens);
      iree_host_size_t written = 0;
      status = iree_tokenizer_model_state_finalize(state->model_state,
                                                   remaining, &written);
      if (iree_status_is_ok(status)) {
        iree_tokenizer_postprocessor_assign_type_ids(&state->postprocessor,
                                                     remaining, 0, written);
        iree_tokenizer_postprocessor_trim_token_offsets(&state->postprocessor,
                                                        state->tokenizer->vocab,
                                                        remaining, 0, written);
        total_tokens += written;
        made_progress |= written > 0;
        if (!iree_tokenizer_model_state_has_pending(state->model_state)) {
          if (iree_all_bits_set(
                  finalize_flags,
                  IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_OMIT_SUFFIX)) {
            state->phase = IREE_TOKENIZER_ENCODE_PHASE_DONE;
          } else {
            iree_tokenizer_postprocessor_begin_suffix(&state->postprocessor);
            state->phase = IREE_TOKENIZER_ENCODE_PHASE_SUFFIX;
          }
          made_progress = true;
        }
      }
    }
    if (state->phase == IREE_TOKENIZER_ENCODE_PHASE_SUFFIX) {
      iree_host_size_t written = iree_tokenizer_postprocessor_emit_suffix(
          &state->postprocessor, output, total_tokens);
      total_tokens += written;
      made_progress |= written > 0;
      if (!iree_tokenizer_postprocessor_encode_state_has_pending(
              &state->postprocessor)) {
        state->phase = IREE_TOKENIZER_ENCODE_PHASE_DONE;
      }
    }
  }
  *out_token_count = total_tokens;
  return status;
}

iree_status_t iree_tokenizer_encode_state_finalize(
    iree_tokenizer_encode_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count) {
  return iree_tokenizer_encode_state_finalize_internal(
      state, output, IREE_TOKENIZER_ENCODE_STATE_FINALIZE_FLAG_NONE,
      out_token_count);
}
