// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Word cache and whole-segment lookup for BPE tokenization.
//
// The word cache provides direct-mapped hash table memoization for
// previously-tokenized segments. This exploits Zipf's law: common words
// (the, a, is, etc.) appear frequently and can bypass the BPE merge
// computation entirely on cache hit.

#include <string.h>

#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/model/bpe_state.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

//===----------------------------------------------------------------------===//
// Word Cache Helpers
//===----------------------------------------------------------------------===//

// Attempts to serve a segment from the word cache. Returns true if the cache
// contained this segment AND all tokens were successfully emitted. Returns
// false on cache miss, oversized segments, or insufficient output capacity.
//
// When byte offsets are requested (cursor->offset_ptr != NULL), the cache is
// bypassed because it does not store per-token byte positions within segments.
bool iree_tokenizer_bpe_cache_lookup(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment, iree_tokenizer_bpe_output_cursor_t* cursor) {
  if (model->cache_capacity == 0) {
    return false;
  }
  if (cursor->offset_ptr) {
    return false;
  }
  if (segment.size > IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES) {
    return false;
  }

  uint32_t hash =
      iree_tokenizer_bpe_cache_hash((const uint8_t*)segment.data, segment.size);
  iree_tokenizer_bpe_cache_entry_t* cache =
      iree_tokenizer_bpe_state_cache(state, model);
  iree_tokenizer_bpe_cache_entry_t* entry =
      &cache[hash & model->cache_capacity_mask];

  if (entry->key_hash != hash || entry->key_length != (uint16_t)segment.size ||
      memcmp(entry->key, segment.data, segment.size) != 0) {
    return false;
  }

  // Verify capacity before emitting to avoid partial emission.
  if (cursor->remaining < entry->token_count) {
    return false;
  }
  for (uint16_t t = 0; t < entry->token_count; ++t) {
    iree_tokenizer_bpe_emit(cursor, entry->tokens[t], 0,
                            (uint32_t)segment.size);
  }
  return true;
}

// Populates a word cache entry from the backtrack stack after encoding.
// Silently skipped when the segment or token count exceeds cache limits.
void iree_tokenizer_bpe_cache_populate(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment,
    const iree_tokenizer_bpe_backtrack_entry_t* stack,
    iree_host_size_t token_count) {
  if (model->cache_capacity == 0) {
    return;
  }
  if (segment.size > IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES) {
    return;
  }
  if (token_count > IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS) {
    return;
  }

  uint32_t hash =
      iree_tokenizer_bpe_cache_hash((const uint8_t*)segment.data, segment.size);
  iree_tokenizer_bpe_cache_entry_t* cache =
      iree_tokenizer_bpe_state_cache(state, model);
  iree_tokenizer_bpe_cache_entry_t* entry =
      &cache[hash & model->cache_capacity_mask];

  entry->key_hash = hash;
  entry->key_length = (uint16_t)segment.size;
  entry->token_count = (uint16_t)token_count;
  memcpy(entry->key, segment.data, segment.size);
  for (iree_host_size_t i = 0; i < token_count; ++i) {
    entry->tokens[i] = stack[i].token_id;
  }
}

// Finds the longest token in the trie with ByteLevel transformation applied.
// Transforms segment bytes via the ByteLevel lookup table when the
// BYTE_LEVEL_INPUT flag is set; character models use their input bytes
// directly.
//
// Used for IGNORE_MERGES mode where we need longest-match vocab lookup but
// still need to respect ByteLevel byte-to-character mapping.
void iree_tokenizer_bpe_trie_longest_match_byte_level(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment,
    int32_t* out_token_id, iree_host_size_t* out_length) {
  *out_token_id = -1;
  *out_length = 0;

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  for (iree_host_size_t i = 0; i < segment.size; ++i) {
    uint8_t byte = (uint8_t)segment.data[i];
    bool advanced = false;

    if (!byte_level || (byte >= 0x21 && byte <= 0x7E)) {
      // Non-ByteLevel or printable ASCII: identity mapping.
      advanced = iree_tokenizer_trie_cursor_advance(&cursor, byte);
    } else {
      // ByteLevel with non-printable or high byte: use pre-computed UTF-8.
      const iree_tokenizer_byte_level_utf8_t* utf8 =
          &iree_tokenizer_byte_level_utf8[byte];
      advanced = iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[0]);
      if (advanced && utf8->length == 2) {
        advanced = iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[1]);
      }
    }

    if (!advanced) {
      break;  // No more matches possible.
    }

    int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token_id >= 0) {
      // Found a token ending here - remember it (greedy longest match).
      *out_token_id = token_id;
      *out_length = i + 1;
    }
  }
}
