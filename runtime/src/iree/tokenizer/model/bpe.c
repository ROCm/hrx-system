// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE construction and state ownership. Canonical merge histories accelerate
// complete words; bounded explicit-symbol workspace handles other segments.

#include "iree/tokenizer/model/bpe.h"

#include <string.h>

#include "iree/tokenizer/model/bpe_heap.h"
#include "iree/tokenizer/model/bpe_state.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

static const iree_tokenizer_model_vtable_t iree_tokenizer_bpe_model_vtable;
static void iree_tokenizer_bpe_model_destroy(
    iree_tokenizer_model_t* base_model);

//===----------------------------------------------------------------------===//
// BPE Model Allocation
//===----------------------------------------------------------------------===//

// Byte fallback spellings are model-owned constants; look them up once.
static void iree_tokenizer_bpe_build_byte_fallback(
    iree_tokenizer_bpe_model_t* model) {
  for (int byte_value = 0; byte_value < 256; ++byte_value) {
    char spelling[8];
    iree_snprintf(spelling, sizeof(spelling), "<0x%02X>", byte_value);
    model->byte_fallback_token[byte_value] = iree_tokenizer_vocab_lookup(
        model->vocab, iree_make_cstring_view(spelling));
  }
}

iree_status_t iree_tokenizer_bpe_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_bpe_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;

  iree_tokenizer_bpe_model_t* model = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*model), (void**)&model));
  memset(model, 0, sizeof(*model));

  model->allocator = allocator;
  model->vocab = vocab;
  model->flags = flags;
  iree_tokenizer_bpe_build_byte_fallback(model);

  // Maximum spelling length bounds cold table-construction workspace.
  model->max_token_length = iree_tokenizer_vocab_max_token_length(vocab);
  if (model->max_token_length == 0) {
    model->max_token_length = 1;  // Safety minimum.
  }

  // Vocab capacity for backtracking table sizing.
  model->vocab_capacity = iree_tokenizer_vocab_capacity(vocab);

  // Backtracking path capacities.
  // Segments up to this size use the O(n) backtracking algorithm.
  // Larger segments use the indexed explicit-symbol heap.
  // Capped at 4095 so the bitfield (ceil_div(4096, 64) = 64 words) fits in
  // a single uint64_t dirty mask for O(1) amortized per-segment init.
  iree_host_size_t min_backtrack = 0;
  if (!iree_host_size_checked_mul(16, model->max_token_length,
                                  &min_backtrack)) {
    // Overflow implies very large token length; clamp to maximum.
    min_backtrack = 4095;
  }
  if (min_backtrack < 2048) {
    min_backtrack = 2048;
  }
  if (min_backtrack > 4095) {
    min_backtrack = 4095;
  }
  model->max_backtrack_segment_bytes = min_backtrack;
  model->backtrack_stack_capacity = model->max_backtrack_segment_bytes;
  // Bitfield tracks one bit per byte position, including the end position
  // (positions 0..segment_size inclusive, so segment_size+1 bits needed).
  model->backtrack_bitfield_capacity =
      iree_host_size_ceil_div(model->max_backtrack_segment_bytes + 1, 64);

  // Word cache: only allocate when the caller has signaled that the
  // pre-tokenizer produces word-level segments (ENABLE_WORD_CACHE flag).
  // Long unsplit inputs exceed its key size. The cache exploits Zipf's law
  // which only applies with sufficient vocabulary diversity (>= 256 tokens).
  if (iree_any_bit_set(flags, IREE_TOKENIZER_BPE_FLAG_ENABLE_WORD_CACHE) &&
      model->vocab_capacity >= IREE_TOKENIZER_BPE_CACHE_MIN_VOCAB_SIZE) {
    model->cache_capacity = IREE_TOKENIZER_BPE_CACHE_CAPACITY;
  } else {
    model->cache_capacity = 0;
  }
  model->cache_capacity_mask =
      model->cache_capacity > 0 ? model->cache_capacity - 1 : 0;

  // Pair validation cache: always enabled for the backtracking path.
  // Memoizes exact boundary validity for both accepted and rejected pairs.
  model->pair_cache_capacity = IREE_TOKENIZER_BPE_PAIR_CACHE_CAPACITY_DEFAULT;
  // Compute state size and offsets with overflow-checked math.
  // Memory layout:
  //   [state struct]
  //   [backtrack stack]     - backtracking path
  //   [backtrack bitfield]  - backtracking path
  //   [word cache]          - repeated-word acceleration (0 if disabled)
  //   [pair cache]          - exact boundary validation results
  iree_host_size_t state_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_bpe_state_t), &state_size,
      IREE_STRUCT_FIELD(model->backtrack_stack_capacity,
                        iree_tokenizer_bpe_backtrack_entry_t,
                        &model->backtrack_stack_offset),
      IREE_STRUCT_FIELD(model->backtrack_bitfield_capacity, uint64_t,
                        &model->backtrack_bitfield_offset),
      IREE_STRUCT_FIELD(model->cache_capacity, iree_tokenizer_bpe_cache_entry_t,
                        &model->cache_offset),
      IREE_STRUCT_FIELD(model->pair_cache_capacity,
                        iree_tokenizer_bpe_pair_cache_entry_t,
                        &model->pair_cache_offset));

  if (iree_status_is_ok(status)) {
    iree_tokenizer_model_initialize(&model->base,
                                    &iree_tokenizer_bpe_model_vtable,
                                    state_size, IREE_SV("BPE"));

    // Build prefix trie for token lookup.
    const iree_tokenizer_token_t* tokens = iree_tokenizer_vocab_tokens(vocab);
    iree_host_size_t token_count = iree_tokenizer_vocab_capacity(vocab);
    iree_const_byte_span_t string_table =
        iree_tokenizer_vocab_string_table(vocab);
    status = iree_tokenizer_vocab_trie_build(tokens, token_count, string_table,
                                             allocator, &model->trie);
  }

  // Build merge hash table.
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_vocab_merge_hash_build(vocab, allocator,
                                                   &model->merge_hash);
  }

  // Build backtracking tables (split_table, next_prefix_match, effective_rank).
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_bpe_build_backtrack_tables(model);
  }

  if (iree_status_is_ok(status)) {
    *out_model = &model->base;
  } else {
    iree_tokenizer_bpe_model_destroy(&model->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_tokenizer_bpe_model_set_end_of_word_suffix(
    iree_tokenizer_model_t* base_model, iree_string_view_t suffix) {
  IREE_ASSERT_ARGUMENT(base_model);
  iree_tokenizer_bpe_model_t* model = (iree_tokenizer_bpe_model_t*)base_model;
  if (suffix.size > sizeof(model->end_of_word_suffix)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "end_of_word_suffix too long (%" PRIhsz
                            " bytes, max %zu)",
                            suffix.size, sizeof(model->end_of_word_suffix));
  }
  memcpy(model->end_of_word_suffix, suffix.data, suffix.size);
  model->end_of_word_suffix_length = suffix.size;
  return iree_ok_status();
}

static void iree_tokenizer_bpe_model_destroy(
    iree_tokenizer_model_t* base_model) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_tokenizer_bpe_model_t* model = (iree_tokenizer_bpe_model_t*)base_model;
  iree_allocator_t allocator = model->allocator;

  iree_tokenizer_vocab_trie_free(model->trie);
  iree_tokenizer_vocab_merge_hash_free(model->merge_hash);
  iree_allocator_free(allocator, model->backtrack_tables.slab);
  iree_allocator_free(allocator, model);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// BPE State Operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_bpe_state_initialize(
    const iree_tokenizer_model_t* base_model, void* storage,
    iree_tokenizer_model_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)base_model;
  iree_tokenizer_bpe_state_t* state = (iree_tokenizer_bpe_state_t*)storage;

  memset(state, 0, sizeof(*state));
  state->base.model = base_model;
  state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;

  // Mark all bitfield words as dirty so the first backtrack_encode call
  // initializes them to UINT64_MAX (the bitfield storage is uninitialized).
  state->backtrack.dirty_mask =
      (model->backtrack_bitfield_capacity >= 64)
          ? UINT64_MAX
          : ((uint64_t)1 << model->backtrack_bitfield_capacity) - 1;

  // Zero the word cache so key_hash == 0 correctly marks empty slots.
  if (model->cache_capacity > 0) {
    memset(iree_tokenizer_bpe_state_cache(state, model), 0,
           model->cache_capacity * sizeof(iree_tokenizer_bpe_cache_entry_t));
  }

  // Initialize pair validation cache: set all entries to empty (0xFF bytes).
  // token1 = 0xFFFFFFFF can never match a real token ID (bounded by
  // vocab_capacity), so all entries start as cache misses regardless of the
  // token2 validity bit.
  if (model->pair_cache_capacity > 0) {
    memset(iree_tokenizer_bpe_state_pair_cache(state, model), 0xFF,
           model->pair_cache_capacity *
               sizeof(iree_tokenizer_bpe_pair_cache_entry_t));
  }

  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_bpe_state_deinitialize(
    iree_tokenizer_model_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)state->model;
  iree_allocator_free(model->allocator, bpe_state->symbol_storage);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_tokenizer_bpe_state_reset(
    iree_tokenizer_model_state_t* state) {
  iree_tokenizer_bpe_state_t* bpe_state = (iree_tokenizer_bpe_state_t*)state;
  bpe_state->phase = IREE_TOKENIZER_BPE_PHASE_SEGMENT_START;
}

//===----------------------------------------------------------------------===//
// BPE Token String Lookup (for decode)
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_bpe_get_token_string(
    const iree_tokenizer_model_t* base_model,
    iree_tokenizer_token_id_t token_id, iree_string_view_t* out_string) {
  const iree_tokenizer_bpe_model_t* model =
      (const iree_tokenizer_bpe_model_t*)base_model;

  iree_string_view_t text =
      iree_tokenizer_vocab_token_text(model->vocab, token_id);
  if (text.size == 0 && token_id >= 0) {
    iree_host_size_t capacity = iree_tokenizer_vocab_capacity(model->vocab);
    if ((iree_host_size_t)token_id >= capacity) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "token ID %d out of range (capacity %" PRIhsz ")",
                              token_id, capacity);
    }
  }

  *out_string = text;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// BPE Model VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_model_vtable_t iree_tokenizer_bpe_model_vtable = {
    .destroy = iree_tokenizer_bpe_model_destroy,
    .state_initialize = iree_tokenizer_bpe_state_initialize,
    .state_deinitialize = iree_tokenizer_bpe_state_deinitialize,
    .state_reset = iree_tokenizer_bpe_state_reset,
    .state_encode = iree_tokenizer_bpe_state_encode,
    .state_finalize = iree_tokenizer_bpe_state_finalize,
    .state_has_pending = iree_tokenizer_bpe_state_has_pending,
    .state_reclaim = iree_tokenizer_bpe_state_reclaim,
    .get_token_string = iree_tokenizer_bpe_get_token_string,
};
