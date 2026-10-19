// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/tokenizer/model/bpe_state.h"

bool iree_tokenizer_bpe_is_first_token_reachable(
    const iree_tokenizer_bpe_model_t* model, uint32_t token) {
  return iree_any_bit_set(model->backtrack_tables.token_reachable[token / 64],
                          1ull << (token % 64));
}

// Records the actual final merge, not the first dictionary rule producing the
// spelling. A surviving token boundary cannot have been crossed by a merge, so
// the canonical history of each operand is independent of its surroundings.
static void iree_tokenizer_bpe_record_merge(iree_tokenizer_bpe_model_t* model,
                                            uint32_t left, uint32_t right,
                                            uint32_t result, uint32_t rank) {
  uint32_t left_rank = model->backtrack_tables.effective_rank[left];
  uint32_t right_rank = model->backtrack_tables.effective_rank[right];
  model->backtrack_tables.split_table[result] =
      (iree_tokenizer_bpe_split_entry_t){
          .left_id = left,
          .right_id = right,
          .left_boundary_rank =
              iree_max(rank, right_rank >= left_rank ? right_rank : 0),
          .right_boundary_rank =
              iree_max(rank, left_rank > right_rank ? left_rank : 0),
      };
  model->backtrack_tables.effective_rank[result] =
      iree_max(rank, iree_max(left_rank, right_rank));
  model->backtrack_tables.token_reachable[result / 64] |= 1ull << (result % 64);
}

// Compiles canonical histories by executing BPE on vocabulary spellings once.
// The work array is reused for every spelling. Only character atoms are initial
// symbols; an arbitrary multi-character merge operand is not itself an atom.
static void iree_tokenizer_bpe_build_token_history(
    iree_tokenizer_bpe_model_t* model, iree_string_view_t text,
    int32_t* tokens) {
  iree_host_size_t count = 0;
  for (iree_host_size_t position = 0; position < text.size;) {
    iree_host_size_t length = iree_tokenizer_bpe_character_length(
        (const uint8_t*)text.data + position, text.size - position);
    iree_host_size_t end = position + length;
    int32_t token = iree_tokenizer_vocab_lookup(
        model->vocab,
        iree_make_string_view(text.data + position, end - position));
    if (token < 0) {
      return;
    }
    model->backtrack_tables.token_reachable[(uint32_t)token / 64] |=
        1ull << ((uint32_t)token % 64);
    tokens[count++] = token;
    position = end;
  }

  while (count > 1) {
    iree_host_size_t best_position = 0;
    iree_tokenizer_merge_hash_result_t best =
        iree_tokenizer_merge_hash_result_none();
    for (iree_host_size_t i = 0; i + 1 < count; ++i) {
      iree_tokenizer_merge_hash_result_t merge =
          iree_tokenizer_vocab_merge_hash_lookup(model->merge_hash, tokens[i],
                                                 tokens[i + 1]);
      if (merge.rank < best.rank) {
        best = merge;
        best_position = i;
      }
    }
    if (!iree_tokenizer_merge_hash_result_is_valid(best)) {
      break;
    }
    iree_tokenizer_bpe_record_merge(model, tokens[best_position],
                                    tokens[best_position + 1], best.result_id,
                                    best.rank + 1);
    tokens[best_position] = best.result_id;
    memmove(tokens + best_position + 1, tokens + best_position + 2,
            (count - best_position - 2) * sizeof(*tokens));
    --count;
  }
}

iree_status_t iree_tokenizer_bpe_build_backtrack_tables(
    iree_tokenizer_bpe_model_t* model) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_host_size_t capacity = model->vocab_capacity;
  iree_host_size_t reachable_words = iree_host_size_ceil_div(capacity, 64);
  iree_host_size_t slab_size = 0;
  iree_host_size_t split_offset = 0;
  iree_host_size_t prefix_offset = 0;
  iree_host_size_t rank_offset = 0;
  iree_host_size_t reachable_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &slab_size,
              IREE_STRUCT_FIELD(capacity, iree_tokenizer_bpe_split_entry_t,
                                &split_offset),
              IREE_STRUCT_FIELD(capacity, uint32_t, &prefix_offset),
              IREE_STRUCT_FIELD(capacity, uint32_t, &rank_offset),
              IREE_STRUCT_FIELD(reachable_words, uint64_t, &reachable_offset)));
  void* slab = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(model->allocator, slab_size, &slab));

  iree_tokenizer_bpe_model_t builder = *model;
  builder.backtrack_tables = (iree_tokenizer_bpe_backtrack_tables_t){
      .slab = slab,
      .split_table =
          (iree_tokenizer_bpe_split_entry_t*)((uint8_t*)slab + split_offset),
      .next_prefix_match = (uint32_t*)((uint8_t*)slab + prefix_offset),
      .effective_rank = (uint32_t*)((uint8_t*)slab + rank_offset),
      .token_reachable = (uint64_t*)((uint8_t*)slab + reachable_offset),
  };
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    builder.backtrack_tables.split_table[i] =
        (iree_tokenizer_bpe_split_entry_t){
            .left_id = (uint32_t)i,
            .right_id = (uint32_t)i,
        };
    builder.backtrack_tables.next_prefix_match[i] = UINT32_MAX;
    builder.backtrack_tables.effective_rank[i] = 0;
  }
  memset(builder.backtrack_tables.token_reachable, 0,
         reachable_words * sizeof(uint64_t));

  int32_t* tokens = NULL;
  iree_host_size_t work_size = 0;
  iree_status_t status = iree_ok_status();
  if (!iree_host_size_checked_mul(model->max_token_length, sizeof(*tokens),
                                  &work_size)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "BPE canonical history workspace overflow");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(model->allocator, work_size, (void**)&tokens);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < capacity; ++i) {
      if (iree_tokenizer_bpe_is_first_token_reachable(&builder, (uint32_t)i)) {
        continue;
      }
      iree_tokenizer_bpe_build_token_history(
          &builder, iree_tokenizer_vocab_token_text(model->vocab, (int32_t)i),
          tokens);
    }
    for (iree_host_size_t i = 0; i < capacity; ++i) {
      iree_string_view_t text =
          iree_tokenizer_vocab_token_text(model->vocab, (int32_t)i);
      iree_tokenizer_trie_cursor_t cursor;
      iree_tokenizer_trie_cursor_reset(&cursor, model->trie);
      iree_host_size_t character_end = 0;
      for (iree_host_size_t j = 0; j + 1 < text.size; ++j) {
        if (j == character_end) {
          character_end += iree_tokenizer_bpe_character_length(
              (const uint8_t*)text.data + j, text.size - j);
        }
        if (!iree_tokenizer_trie_cursor_advance(&cursor,
                                                (uint8_t)text.data[j])) {
          break;
        }
        if (j + 1 != character_end) {
          continue;
        }
        int32_t prefix = iree_tokenizer_trie_cursor_token_id(&cursor);
        if (prefix >= 0 && iree_tokenizer_bpe_is_first_token_reachable(
                               &builder, (uint32_t)prefix)) {
          builder.backtrack_tables.next_prefix_match[i] = (uint32_t)prefix;
        }
      }
    }
    iree_allocator_free(model->allocator, model->backtrack_tables.slab);
    model->backtrack_tables = builder.backtrack_tables;
  } else {
    iree_allocator_free(model->allocator, slab);
  }
  iree_allocator_free(model->allocator, tokens);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
