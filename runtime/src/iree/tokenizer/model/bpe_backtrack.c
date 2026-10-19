// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "iree/tokenizer/model/bpe_state.h"

// Reverses the independent canonical histories at the boundary. At equal
// priorities, merges inside the left token preceded the boundary, and merges
// inside the right token followed it. The stored barriers also cover rules
// whose operands form at later priorities than the rule itself.
static bool iree_tokenizer_bpe_is_valid_token_pair(
    const iree_tokenizer_bpe_model_t* model, uint32_t left, uint32_t right) {
  uint64_t limit = UINT64_MAX;
  for (;;) {
    iree_tokenizer_merge_hash_result_t merge =
        iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, left, right);
    if (merge.rank != UINT32_MAX && merge.rank + 1 < limit) {
      return false;
    }
    uint32_t left_rank = model->backtrack_tables.effective_rank[left];
    uint32_t right_rank = model->backtrack_tables.effective_rank[right];
    if (left_rank == 0 && right_rank == 0) {
      return true;
    }
    if (left_rank > right_rank) {
      iree_tokenizer_bpe_split_entry_t split =
          model->backtrack_tables.split_table[left];
      limit = split.right_boundary_rank;
      left = split.right_id;
    } else {
      iree_tokenizer_bpe_split_entry_t split =
          model->backtrack_tables.split_table[right];
      limit = (uint64_t)split.left_boundary_rank + 1;
      right = split.left_id;
    }
  }
}

static bool iree_tokenizer_bpe_cached_pair(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    uint32_t left, uint32_t right) {
  iree_tokenizer_bpe_pair_cache_entry_t* cache =
      iree_tokenizer_bpe_state_pair_cache(state, model);
  uint32_t index = iree_tokenizer_bpe_pair_cache_index(
      left, right, (uint32_t)model->pair_cache_capacity - 1);
  iree_tokenizer_bpe_pair_cache_entry_t* entry = &cache[index];
  if (entry->token1 == left &&
      (entry->token2 & IREE_TOKENIZER_BPE_PAIR_CACHE_TOKEN_MASK) == right) {
    return (entry->token2 & IREE_TOKENIZER_BPE_PAIR_CACHE_VALID_BIT) != 0;
  }
  bool valid = iree_tokenizer_bpe_is_valid_token_pair(model, left, right);
  entry->token1 = left;
  entry->token2 = right | (valid ? IREE_TOKENIZER_BPE_PAIR_CACHE_VALID_BIT : 0);
  return valid;
}

static iree_host_size_t iree_tokenizer_bpe_token_raw_length(
    const iree_tokenizer_bpe_model_t* model, uint32_t token,
    const uint8_t* data) {
  iree_string_view_t text =
      iree_tokenizer_vocab_token_text(model->vocab, (int32_t)token);
  if (!iree_any_bit_set(model->flags,
                        IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT)) {
    return text.size;
  }
  iree_host_size_t length = 0;
  for (iree_host_size_t encoded = 0; encoded < text.size; ++length) {
    encoded += iree_tokenizer_byte_level_utf8[data[length]].length;
  }
  return length;
}

bool iree_tokenizer_bpe_backtrack_encode(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    const uint8_t* segment_data, iree_host_size_t segment_size) {
  iree_tokenizer_bpe_backtrack_entry_t* stack =
      iree_tokenizer_bpe_state_backtrack_stack(state, model);
  uint64_t* bitfield =
      iree_tokenizer_bpe_state_backtrack_bitfield(state, model);
  uint64_t dirty = state->backtrack.dirty_mask;
  while (dirty) {
    bitfield[iree_math_count_trailing_zeros_u64(dirty)] = UINT64_MAX;
    dirty &= dirty - 1;
  }
  state->backtrack.dirty_mask = 0;

  iree_host_size_t count = 0;
  iree_host_size_t position = 0;
  int32_t candidate = -1;
  iree_host_size_t length = 0;
  iree_tokenizer_bpe_backtrack_longest_match(model, segment_data, segment_size,
                                             &candidate, &length);
  while (position < segment_size) {
    if (candidate < 0) {
      return false;
    }
    bool accepted = false;
    while (candidate >= 0) {
      iree_host_size_t end = position + length;
      bool pair_matches =
          count == 0 || iree_tokenizer_bpe_cached_pair(
                            model, state, stack[count - 1].token_id, candidate);
      if (pair_matches && iree_tokenizer_bpe_bitfield_is_set(bitfield, end)) {
        iree_tokenizer_bpe_backtrack_entry_set(
            &stack[count++], candidate, (uint32_t)position, (uint32_t)end);
        position = end;
        accepted = true;
        break;
      }
      uint32_t prefix =
          model->backtrack_tables.next_prefix_match[(uint32_t)candidate];
      candidate = prefix == UINT32_MAX ? -1 : (int32_t)prefix;
      if (candidate >= 0) {
        length = iree_tokenizer_bpe_token_raw_length(model, candidate,
                                                     segment_data + position);
      }
    }
    if (!accepted) {
      iree_tokenizer_bpe_bitfield_clear(bitfield, &state->backtrack.dirty_mask,
                                        position);
      if (count) {
        --count;
        position = iree_tokenizer_bpe_backtrack_entry_start_byte(&stack[count]);
        candidate = stack[count].token_id;
        length = iree_tokenizer_bpe_token_raw_length(model, candidate,
                                                     segment_data + position);
        continue;
      }
      return false;
    }
    if (position < segment_size) {
      iree_tokenizer_bpe_backtrack_longest_match(model, segment_data + position,
                                                 segment_size - position,
                                                 &candidate, &length);
    }
  }
  state->backtrack.stack_count = count;
  state->backtrack.emit_index = 0;
  return true;
}
