// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/bpe_state.h"

// Each merge removes a live symbol and queues at most two candidates. Starting
// with n-1 candidates, the heap holds at most 2*n entries, including stale
// ones.
static iree_status_t iree_tokenizer_bpe_symbols_reserve(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_host_size_t input_size, iree_tokenizer_bpe_heap_t* out_heap) {
  iree_host_size_t multiplier =
      iree_any_bit_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT)
          ? 2
          : 1;
  iree_host_size_t bytes_per_symbol =
      sizeof(iree_tokenizer_bpe_symbol_t) +
      2 * sizeof(iree_tokenizer_bpe_heap_entry_t);
  iree_host_size_t inline_size = model->backtrack_stack_capacity *
                                 sizeof(iree_tokenizer_bpe_backtrack_entry_t);
  iree_host_size_t workspace_limit = iree_max(
      inline_size, (iree_host_size_t)IREE_TOKENIZER_BPE_SCRATCH_MAX_SIZE);
  iree_host_size_t limit = workspace_limit / bytes_per_symbol;
  if (input_size > (limit - sizeof(model->end_of_word_suffix)) / multiplier) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "BPE segment exceeds %" PRIhsz
                            "-byte symbol workspace limit",
                            workspace_limit);
  }
  iree_host_size_t capacity =
      input_size * multiplier + sizeof(model->end_of_word_suffix);
  iree_host_size_t required = capacity * bytes_per_symbol;
  uint8_t* storage =
      (uint8_t*)iree_tokenizer_bpe_state_backtrack_stack(state, model);
  if (required > inline_size) {
    if (required > state->symbol_storage_size) {
      iree_host_size_t allocation =
          iree_min(iree_host_size_next_power_of_two(required),
                   (iree_host_size_t)IREE_TOKENIZER_BPE_SCRATCH_MAX_SIZE);
      IREE_RETURN_IF_ERROR(iree_allocator_realloc(model->allocator, allocation,
                                                  &state->symbol_storage));
      state->symbol_storage_size = allocation;
    }
    storage = (uint8_t*)state->symbol_storage;
  }
  state->symbols = (iree_tokenizer_bpe_symbol_t*)storage;
  iree_host_size_t heap_offset = capacity * sizeof(iree_tokenizer_bpe_symbol_t);
  iree_tokenizer_bpe_heap_initialize(
      out_heap, (iree_tokenizer_bpe_heap_entry_t*)(storage + heap_offset),
      2 * capacity);
  return iree_ok_status();
}

static void iree_tokenizer_bpe_symbols_append(
    iree_tokenizer_bpe_symbol_t* symbols, uint32_t* count, int32_t token_id,
    uint32_t start_byte, uint32_t end_byte) {
  uint32_t index = (*count)++;
  symbols[index] = (iree_tokenizer_bpe_symbol_t){
      .token_id = token_id,
      .start_byte = start_byte,
      .end_byte = end_byte,
      .previous = index == 0 ? UINT32_MAX : index - 1,
      .next = UINT32_MAX,
  };
  if (index) {
    symbols[index - 1].next = index;
  }
}

static void iree_tokenizer_bpe_symbols_queue(
    const iree_tokenizer_bpe_model_t* model,
    const iree_tokenizer_bpe_symbol_t* symbols, uint32_t index,
    iree_tokenizer_bpe_heap_t* heap) {
  if (index == UINT32_MAX || symbols[index].next == UINT32_MAX) {
    return;
  }
  iree_tokenizer_merge_hash_result_t merge =
      iree_tokenizer_vocab_merge_hash_lookup(
          model->merge_hash, symbols[index].token_id,
          symbols[symbols[index].next].token_id);
  if (iree_tokenizer_merge_hash_result_is_valid(merge)) {
    // Stable symbol order gives the leftmost tie break, including fallback
    // symbols that cover the same input byte range.
    iree_tokenizer_bpe_heap_push(
        heap, iree_tokenizer_make_bpe_heap_entry(merge.rank, index));
  }
}

iree_status_t iree_tokenizer_bpe_encode_symbols(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment) {
  iree_tokenizer_bpe_heap_t heap;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_bpe_symbols_reserve(model, state, segment.size, &heap));
  iree_tokenizer_bpe_symbol_t* symbols = state->symbols;
  uint32_t count = 0;
  bool byte_level =
      iree_any_bit_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);
  bool previous_unknown = false;
  int32_t unknown = iree_tokenizer_vocab_special_ids(model->vocab).unk;
  for (iree_host_size_t position = 0; position < segment.size;) {
    char spelling[IREE_UNICODE_UTF8_MAX_BYTE_LENGTH +
                  sizeof(model->end_of_word_suffix)];
    uint8_t first = (uint8_t)segment.data[position];
    iree_host_size_t input_length = 1;
    iree_host_size_t spelling_length = 0;
    if (byte_level) {
      spelling_length = iree_tokenizer_byte_level_utf8[first].length;
      memcpy(spelling, iree_tokenizer_byte_level_utf8[first].bytes,
             spelling_length);
    } else {
      input_length = iree_tokenizer_bpe_character_length(
          (const uint8_t*)segment.data + position, segment.size - position);
      spelling_length = input_length;
      memcpy(spelling, segment.data + position, input_length);
    }
    iree_host_size_t end = position + input_length;
    if (end == segment.size) {
      memcpy(spelling + spelling_length, model->end_of_word_suffix,
             model->end_of_word_suffix_length);
      spelling_length += model->end_of_word_suffix_length;
    }
    int32_t token = iree_tokenizer_vocab_lookup(
        model->vocab, iree_make_string_view(spelling, spelling_length));
    if (token >= 0) {
      iree_tokenizer_bpe_symbols_append(symbols, &count, token,
                                        (uint32_t)position, (uint32_t)end);
      previous_unknown = false;
    } else {
      bool byte_fallback = !iree_any_bit_set(
          model->flags, IREE_TOKENIZER_BPE_FLAG_NO_BYTE_FALLBACK);
      for (iree_host_size_t i = 0; i < spelling_length && byte_fallback; ++i) {
        byte_fallback = model->byte_fallback_token[(uint8_t)spelling[i]] >= 0;
      }
      if (byte_fallback) {
        for (iree_host_size_t i = 0; i < spelling_length; ++i) {
          uint32_t start_byte =
              byte_level ? (uint32_t)position
                         : (uint32_t)(position + iree_min(i, input_length));
          uint32_t end_byte =
              byte_level ? (uint32_t)end
                         : (uint32_t)(position + iree_min(i + 1, input_length));
          iree_tokenizer_bpe_symbols_append(
              symbols, &count, model->byte_fallback_token[(uint8_t)spelling[i]],
              start_byte, end_byte);
        }
        previous_unknown = false;
      } else if (unknown >= 0) {
        if (previous_unknown &&
            iree_any_bit_set(model->flags, IREE_TOKENIZER_BPE_FLAG_FUSE_UNK)) {
          symbols[count - 1].end_byte = (uint32_t)end;
        } else {
          iree_tokenizer_bpe_symbols_append(symbols, &count, unknown,
                                            (uint32_t)position, (uint32_t)end);
        }
        previous_unknown = true;
      }
    }
    position = end;
  }
  for (uint32_t i = 0; i < count; ++i) {
    iree_tokenizer_bpe_symbols_queue(model, symbols, i, &heap);
  }
  while (!iree_tokenizer_bpe_heap_is_empty(&heap)) {
    iree_tokenizer_bpe_heap_entry_t entry = iree_tokenizer_bpe_heap_pop(&heap);
    uint32_t left = entry.position;
    uint32_t right = symbols[left].next;
    if (symbols[left].token_id < 0 || right == UINT32_MAX) {
      continue;
    }
    iree_tokenizer_merge_hash_result_t merge =
        iree_tokenizer_vocab_merge_hash_lookup(
            model->merge_hash, symbols[left].token_id, symbols[right].token_id);
    if (merge.rank != entry.rank) {
      continue;
    }
    symbols[left].token_id = merge.result_id;
    symbols[left].end_byte = symbols[right].end_byte;
    symbols[left].next = symbols[right].next;
    if (symbols[right].next != UINT32_MAX) {
      symbols[symbols[right].next].previous = left;
    }
    symbols[right].token_id = -1;
    iree_tokenizer_bpe_symbols_queue(model, symbols, symbols[left].previous,
                                     &heap);
    iree_tokenizer_bpe_symbols_queue(model, symbols, left, &heap);
  }
  state->symbol_emit_index = count ? 0 : UINT32_MAX;

  if (segment.size <= IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES) {
    iree_tokenizer_bpe_backtrack_entry_t
        entries[IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS];
    uint32_t index = state->symbol_emit_index;
    iree_host_size_t cached_count = 0;
    while (index != UINT32_MAX && cached_count < IREE_ARRAYSIZE(entries)) {
      iree_tokenizer_bpe_backtrack_entry_set(
          &entries[cached_count++], symbols[index].token_id,
          symbols[index].start_byte, symbols[index].end_byte);
      index = symbols[index].next;
    }
    if (index == UINT32_MAX) {
      iree_tokenizer_bpe_cache_populate(model, state, segment, entries,
                                        cached_count);
    }
  }
  return iree_ok_status();
}
