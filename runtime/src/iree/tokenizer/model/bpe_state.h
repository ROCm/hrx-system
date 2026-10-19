// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BPE model tables, reusable encoding state, and their representation
// contracts.

#ifndef IREE_TOKENIZER_MODEL_BPE_STATE_H_
#define IREE_TOKENIZER_MODEL_BPE_STATE_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/byte_level_tables.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/model/bpe_heap.h"
#include "iree/tokenizer/vocab/vocab_merge_hash.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

#ifdef __cplusplus
extern "C" {
#endif

// Stable-index list entry for complete initial-symbol merging. Removed entries
// are tombstones, so queued merge candidates never depend on shifted indices.
typedef struct iree_tokenizer_bpe_symbol_t {
  // Vocabulary ID, or -1 after this node is merged into its predecessor.
  int32_t token_id;
  // Inclusive input byte position within the segment.
  uint32_t start_byte;
  // Exclusive input byte position within the segment.
  uint32_t end_byte;
  // Previous live node, or UINT32_MAX at the head.
  uint32_t previous;
  // Next live node, or UINT32_MAX at the tail.
  uint32_t next;
} iree_tokenizer_bpe_symbol_t;

// Inverse merge entry: for a token formed by a merge, stores its constituents.
// Character atoms and unreachable dictionary entries store (self, self).
typedef struct iree_tokenizer_bpe_split_entry_t {
  // Left operand of the canonical token's final merge, or self for an atom.
  uint32_t left_id;
  // Right operand of the canonical token's final merge, or self for an atom.
  uint32_t right_id;
  // One-based priority barrier before the left boundary reaches this token.
  uint32_t left_boundary_rank;
  // One-based priority barrier before the right boundary reaches this token.
  uint32_t right_boundary_rank;
} iree_tokenizer_bpe_split_entry_t;

// Complete-segment token workspace shared by boundary validation and explicit
// atom merging. Both offsets fit in 16 bits because complete segments routed
// here are bounded by max_backtrack_segment_bytes (at most 4095).
typedef struct iree_tokenizer_bpe_backtrack_entry_t {
  // Vocabulary token ID.
  int32_t token_id;
  // Inclusive offset in the original segment.
  uint16_t start_byte;
  // Exclusive offset; explicit because unknown omission can leave gaps.
  uint16_t end_byte;
} iree_tokenizer_bpe_backtrack_entry_t;

static inline uint32_t iree_tokenizer_bpe_backtrack_entry_start_byte(
    const iree_tokenizer_bpe_backtrack_entry_t* entry) {
  return entry->start_byte;
}

static inline void iree_tokenizer_bpe_backtrack_entry_set(
    iree_tokenizer_bpe_backtrack_entry_t* entry, int32_t token_id,
    uint32_t start_byte, uint32_t end_byte) {
  entry->token_id = token_id;
  entry->start_byte = (uint16_t)start_byte;
  entry->end_byte = (uint16_t)end_byte;
}

//===----------------------------------------------------------------------===//
// Backtrack Tables (Model-Side)
//===----------------------------------------------------------------------===//

// Precomputed tables for the O(n) backtracking algorithm.
// Built once during model allocation, read-only after construction.
// All tables are indexed by token ID and sized to vocab_capacity.
typedef struct iree_tokenizer_bpe_backtrack_tables_t {
  // Single allocation for all tables below.
  void* slab;
  // Canonical final merge and priority barriers per token.
  iree_tokenizer_bpe_split_entry_t* split_table;
  // Longest proper prefix token, or UINT32_MAX.
  uint32_t* next_prefix_match;
  // Maximum one-based merge priority in the canonical history; zero for atoms.
  uint32_t* effective_rank;
  // Bit per token reachable from the model's initial symbols.
  uint64_t* token_reachable;
} iree_tokenizer_bpe_backtrack_tables_t;

//===----------------------------------------------------------------------===//
// Backtrack State (State-Side)
//===----------------------------------------------------------------------===//

// Runtime state for the O(n) backtracking algorithm.
// Retained while computed output waits for credit.
typedef struct iree_tokenizer_bpe_backtrack_state_t {
  // Number of computed compact tokens.
  iree_host_size_t stack_count;
  // Next token to emit from stack.
  iree_host_size_t emit_index;
  // Dirty mask for lazy bitfield reset. Each bit corresponds to a uint64_t
  // word in the backtrack bitfield that was modified (cleared) during the
  // previous segment's backtracking. On segment start, only these words are
  // reset to UINT64_MAX, giving O(backtracks) init instead of O(capacity).
  uint64_t dirty_mask;
} iree_tokenizer_bpe_backtrack_state_t;

//===----------------------------------------------------------------------===//
// BPE Model Structure
//===----------------------------------------------------------------------===//

typedef struct iree_tokenizer_bpe_model_t {
  // Shared model interface, at offset zero.
  iree_tokenizer_model_t base;
  // Owns immutable tables and per-state overflow allocations.
  iree_allocator_t allocator;
  // Borrowed vocabulary, which outlives the model and its states.
  const iree_tokenizer_vocab_t* vocab;

  // Prefix trie for token lookup (owned).
  iree_tokenizer_vocab_trie_t* trie;

  // Merge hash table for O(1) merge lookup (owned).
  iree_tokenizer_vocab_merge_hash_t* merge_hash;

  // Configuration flags (see iree_tokenizer_bpe_flag_bits_t).
  iree_tokenizer_bpe_flags_t flags;

  // End-of-word suffix (e.g., "</w>" for CLIP).
  // Empty if not used. Appended to each segment before tokenization.
  char end_of_word_suffix[16];
  // Number of valid suffix bytes.
  iree_host_size_t end_of_word_suffix_length;

  // Precomputed byte fallback token IDs for each byte value.
  // Maps byte N to the token ID for "<0xNN>" (the byte-level fallback token).
  // -1 if no such token exists. Built during model construction to avoid
  // snprintf + vocab hash lookup per unknown byte during encoding.
  int32_t byte_fallback_token[256];

  // Maximum encoded vocabulary spelling length in bytes.
  iree_host_size_t max_token_length;

  // Backtracking algorithm tables (owned).
  // Each table is indexed by token ID and sized to vocab_capacity.
  iree_tokenizer_bpe_backtrack_tables_t backtrack_tables;
  // Number of addressable token IDs in the table arrays.
  iree_host_size_t vocab_capacity;

  // Backtracking encode configuration and state buffer offsets.
  // Segments <= max_backtrack_segment_bytes use the backtracking path;
  // longer segments use the indexed explicit-symbol heap.
  iree_host_size_t max_backtrack_segment_bytes;
  // Number of compact token slots in fixed state storage.
  iree_host_size_t backtrack_stack_capacity;
  // Number of uint64_t words in the boundary-search bitfield.
  iree_host_size_t backtrack_bitfield_capacity;
  // Byte offset of compact tokens from the state base.
  iree_host_size_t backtrack_stack_offset;
  // Byte offset of boundary-search bits from the state base.
  iree_host_size_t backtrack_bitfield_offset;

  // Word cache capacity (power of 2, or 0 if disabled).
  // Caching exploits Zipf's law: frequent words dominate token positions.
  // Only useful with sufficient vocabulary diversity (>= 256 tokens).
  iree_host_size_t cache_capacity;
  // Hash mask, cache_capacity - 1, or zero when disabled.
  iree_host_size_t cache_capacity_mask;
  // Byte offset of word cache entries from the state base.
  iree_host_size_t cache_offset;

  // Number of cached canonical boundary validity results.
  iree_host_size_t pair_cache_capacity;
  // Byte offset of pair cache entries from the state base.
  iree_host_size_t pair_cache_offset;
} iree_tokenizer_bpe_model_t;

//===----------------------------------------------------------------------===//
// Segment Context (State-Side)
//===----------------------------------------------------------------------===//

// Input origin retained while computed tokens wait for output credit.
typedef struct iree_tokenizer_bpe_segment_context_t {
  // Offset of the segment in the caller's transform buffer.
  iree_host_size_t base_offset;
} iree_tokenizer_bpe_segment_context_t;

//===----------------------------------------------------------------------===//
// BPE State Structure
//===----------------------------------------------------------------------===//

// Input is borrowed until its segment completes; only computed token results
// remain pending under output backpressure.
typedef enum iree_tokenizer_bpe_phase_e {
  IREE_TOKENIZER_BPE_PHASE_SEGMENT_START = 0,
  IREE_TOKENIZER_BPE_PHASE_BACKTRACK_EMIT,
  IREE_TOKENIZER_BPE_PHASE_SYMBOL_EMIT,
} iree_tokenizer_bpe_phase_t;

// Caller-provided fixed state with optional owned overflow workspace.
typedef struct iree_tokenizer_bpe_state_t {
  // Shared state interface and borrowed model, at offset zero.
  iree_tokenizer_model_state_t base;

  // Current encoding phase (see iree_tokenizer_bpe_phase_t).
  iree_tokenizer_bpe_phase_t phase;

  // Origin used when finalizing computed output.
  iree_tokenizer_bpe_segment_context_t segment;

  // Backtracking state (used in BACKTRACK/BACKTRACK_EMIT phases).
  iree_tokenizer_bpe_backtrack_state_t backtrack;

  // Reusable overflow storage owned by the model allocator; NULL before growth.
  void* symbol_storage;
  // Allocated overflow bytes, bounded by IREE_TOKENIZER_BPE_SCRATCH_MAX_SIZE.
  iree_host_size_t symbol_storage_size;
  // Active symbols, borrowing the backtrack slab or symbol_storage.
  iree_tokenizer_bpe_symbol_t* symbols;
  // Next live symbol to emit, or UINT32_MAX when complete.
  uint32_t symbol_emit_index;

  // Trailing buffers (accessed via model offsets):
  // - iree_tokenizer_bpe_backtrack_entry_t backtrack_stack[stack_capacity]
  // - uint64_t backtrack_bitfield[bitfield_capacity]
  // - iree_tokenizer_bpe_cache_entry_t cache_entries[cache_capacity]
  // - iree_tokenizer_bpe_pair_cache_entry_t pair_cache[pair_cache_capacity]
} iree_tokenizer_bpe_state_t;

//===----------------------------------------------------------------------===//
// Pair Validation Cache
//===----------------------------------------------------------------------===//

// Direct-mapped cache of exact canonical boundary validity.
// Default capacity for the pair validation cache. Must be a power of 2.
// 4096 entries × 8 bytes = 32 KB, which fits in L1 cache on most CPUs.
// Larger caches (16K+) spill into L2 and regress ASCII/Code throughput;
// smaller caches (256) have too many collisions for CJK pair sets.
// Embedded systems with tight memory budgets can reduce this to 256.

#define IREE_TOKENIZER_BPE_PAIR_CACHE_CAPACITY_DEFAULT 4096

typedef struct iree_tokenizer_bpe_pair_cache_entry_t {
  // Left token ID, or UINT32_MAX for an empty entry.
  uint32_t token1;
  // Right token ID with validity in bit 31 (IDs are below 2^31).
  uint32_t token2;
} iree_tokenizer_bpe_pair_cache_entry_t;

// Bit flag stored in token2 to indicate a valid (TRUE) pair cache result.
// When this bit is clear, the cached result is FALSE (invalid pair).
#define IREE_TOKENIZER_BPE_PAIR_CACHE_VALID_BIT (1u << 31)

// Mask to extract the actual token ID from the token2 field.
#define IREE_TOKENIZER_BPE_PAIR_CACHE_TOKEN_MASK \
  (~IREE_TOKENIZER_BPE_PAIR_CACHE_VALID_BIT)

// Computes the pair cache index for a (token1, token2) pair.
// Uses a Fibonacci-hashing-style multiplicative hash for good dispersion.
// |capacity_mask| must be (capacity - 1) where capacity is a power of 2.
static inline uint32_t iree_tokenizer_bpe_pair_cache_index(
    uint32_t token1, uint32_t token2, uint32_t capacity_mask) {
  return ((token1 * 0x9E3779B9u) ^ token2) & capacity_mask;
}

//===----------------------------------------------------------------------===//
// Whole-Word Cache
//===----------------------------------------------------------------------===//

// Direct-mapped cache for repeated segment tokenization results.
// Segments that produce the same byte sequence will hash to the same slot
// and return cached token IDs, skipping all trie/backtracking work.
//
// Capacity is a power of 2 for fast modulo via bitmask. 512 entries provides
// 60-68% hit rate on English prose and code (whitespace-segmented words follow
// Zipf's law — a small set of frequent words dominates token positions).

#define IREE_TOKENIZER_BPE_CACHE_CAPACITY 512

// Minimum vocabulary size for caching to be worthwhile. Below this threshold,
// there aren't enough distinct token sequences for the direct-mapped cache to
// achieve meaningful hit rates — the vocabulary simply can't produce the
// variety of multi-token word tokenizations that caching accelerates.
#define IREE_TOKENIZER_BPE_CACHE_MIN_VOCAB_SIZE 256

// Maximum segment byte length that can be cached. Segments longer than this
// bypass the cache (they're rare enough not to affect overall hit rate).
// 32 bytes covers identifiers and words up to the 99.9th percentile of
// English prose and code (mean ~5 bytes, long identifiers ~20-30 bytes).
#define IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES 32

// Maximum token count per cached segment. Segments producing more tokens than
// this are not cached (rare for natural language — most words produce 1-5
// tokens, long/rare words up to ~10).
#define IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS 10

typedef struct iree_tokenizer_bpe_cache_entry_t {
  // Hash of segment bytes (0 = empty slot).
  uint32_t key_hash;
  // Byte length of the cached segment.
  uint16_t key_length;
  // Number of tokens produced for this segment.
  uint16_t token_count;
  // Exact input bytes used to disambiguate hash collisions.
  uint8_t key[IREE_TOKENIZER_BPE_CACHE_MAX_KEY_BYTES];
  // Cached output IDs, valid through token_count.
  iree_tokenizer_token_id_t tokens[IREE_TOKENIZER_BPE_CACHE_MAX_TOKENS];
} iree_tokenizer_bpe_cache_entry_t;
// Size: 4 + 2 + 2 + 32 + 40 = 80 bytes per entry (1.25 cache lines).
// Total: 512 * 80 = 40KB.

// Word-at-a-time hash with Murmur3 finalizer for cache keys.
// Processes 8 bytes per iteration (1 load + 1 XOR + 1 multiply), finishing
// with the Murmur3 64-bit finalizer for thorough bit mixing. For a typical
// 8-byte word this is ~7 ALU ops total vs FNV-1a's 16 per-byte ops.
// Returns non-zero (0 is the empty-slot sentinel).
static inline uint32_t iree_tokenizer_bpe_cache_hash(const uint8_t* data,
                                                     iree_host_size_t length) {
  uint64_t accumulator = length;
  iree_host_size_t offset = 0;
  // Process 8-byte chunks.
  while (offset + 8 <= length) {
    uint64_t word;
    memcpy(&word, data + offset, 8);
    accumulator ^= word;
    accumulator *= 0x9E3779B97F4A7C15ULL;
    offset += 8;
  }
  // Handle remaining 1-7 bytes.
  if (offset < length) {
    uint64_t tail = 0;
    memcpy(&tail, data + offset, length - offset);
    accumulator ^= tail;
  }
  // Murmur3 64-bit finalizer.
  accumulator ^= accumulator >> 33;
  accumulator *= 0xff51afd7ed558ccdULL;
  accumulator ^= accumulator >> 33;
  accumulator *= 0xc4ceb9fe1a85ec53ULL;
  accumulator ^= accumulator >> 33;
  uint32_t result = (uint32_t)accumulator;
  return result ? result : 1;
}

//===----------------------------------------------------------------------===//
// State Buffer Accessors
//===----------------------------------------------------------------------===//

static inline iree_tokenizer_bpe_backtrack_entry_t*
iree_tokenizer_bpe_state_backtrack_stack(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_backtrack_entry_t*)((uint8_t*)state +
                                                 model->backtrack_stack_offset);
}

static inline uint64_t* iree_tokenizer_bpe_state_backtrack_bitfield(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (uint64_t*)((uint8_t*)state + model->backtrack_bitfield_offset);
}

static inline iree_tokenizer_bpe_cache_entry_t* iree_tokenizer_bpe_state_cache(
    iree_tokenizer_bpe_state_t* state,
    const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_cache_entry_t*)((uint8_t*)state +
                                             model->cache_offset);
}

static inline iree_tokenizer_bpe_pair_cache_entry_t*
iree_tokenizer_bpe_state_pair_cache(iree_tokenizer_bpe_state_t* state,
                                    const iree_tokenizer_bpe_model_t* model) {
  return (iree_tokenizer_bpe_pair_cache_entry_t*)((uint8_t*)state +
                                                  model->pair_cache_offset);
}

//===----------------------------------------------------------------------===//
// Output Cursor
//===----------------------------------------------------------------------===//

// Tracks position and remaining capacity in the output buffer.
// When offset_ptr is non-NULL, byte ranges are written for each emitted token.
typedef struct iree_tokenizer_bpe_output_cursor_t {
  iree_tokenizer_token_id_t* ptr;  // Next write position.
  // Next offset write position (or NULL).
  iree_tokenizer_offset_t* offset_ptr;
  // Added to per-token byte positions.
  iree_host_size_t segment_base_offset;
  iree_host_size_t remaining;  // Slots available.
} iree_tokenizer_bpe_output_cursor_t;

// Creates an output cursor for a segment at |segment_base_offset| within the
// transform buffer. Pass NULL for |offsets| to skip offset tracking.
static inline iree_tokenizer_bpe_output_cursor_t
iree_tokenizer_bpe_output_cursor_make(iree_tokenizer_token_id_t* base,
                                      iree_tokenizer_offset_t* offsets,
                                      iree_host_size_t segment_base_offset,
                                      iree_host_size_t capacity) {
  iree_tokenizer_bpe_output_cursor_t cursor = {base, offsets,
                                               segment_base_offset, capacity};
  return cursor;
}

// Returns the number of tokens written so far.
static inline iree_host_size_t iree_tokenizer_bpe_output_cursor_count(
    const iree_tokenizer_bpe_output_cursor_t* cursor,
    const iree_tokenizer_token_id_t* base) {
  return (iree_host_size_t)(cursor->ptr - base);
}

// Emits a token to the output buffer with its byte range within the segment.
// The offset written is {segment_base_offset + start_byte,
//                        segment_base_offset + end_byte}.
// Returns false if buffer is full.
static inline bool iree_tokenizer_bpe_emit(
    iree_tokenizer_bpe_output_cursor_t* cursor,
    iree_tokenizer_token_id_t token_id, uint32_t start_byte,
    uint32_t end_byte) {
  if (cursor->remaining == 0) {
    return false;
  }
  *cursor->ptr++ = token_id;
  if (cursor->offset_ptr) {
    cursor->offset_ptr->start = cursor->segment_base_offset + start_byte;
    cursor->offset_ptr->end = cursor->segment_base_offset + end_byte;
    cursor->offset_ptr++;
  }
  --cursor->remaining;
  return true;
}

//===----------------------------------------------------------------------===//
// Bitfield Helpers
//===----------------------------------------------------------------------===//

// Bitfield helpers: the bitfield uses one bit per byte position.
// Initialized to all-ones (optimistic: all positions assumed reachable).
// Bits are cleared when backtracking proves a position is unreachable.

static inline bool iree_tokenizer_bpe_bitfield_is_set(
    const uint64_t* bitfield, iree_host_size_t position) {
  return (bitfield[position / 64] >> (position % 64)) & 1;
}

static inline void iree_tokenizer_bpe_bitfield_clear(
    uint64_t* bitfield, uint64_t* dirty_mask, iree_host_size_t position) {
  iree_host_size_t word_index = position / 64;
  bitfield[word_index] &= ~((uint64_t)1 << (position % 64));
  *dirty_mask |= (uint64_t)1 << word_index;
}

//===----------------------------------------------------------------------===//
// Shared Trie Helpers
//===----------------------------------------------------------------------===//

// Character models start with complete UTF-8 codepoints. Invalid bytes remain
// individual atoms; raw byte vocabulary entries cannot split a valid codepoint.
static inline iree_host_size_t iree_tokenizer_bpe_character_length(
    const uint8_t* data, iree_host_size_t size) {
  iree_host_size_t length = iree_unicode_utf8_sequence_length(data[0]);
  return length <= size && iree_unicode_utf8_is_valid_sequence(data, length)
             ? length
             : 1;
}

// Finds the longest token matching raw input bytes starting at data[0..size].
// For ByteLevel mode, each raw byte is converted to its ByteLevel UTF-8 form
// (1-2 bytes) before feeding to the trie cursor.
// Returns the matched token_id and the count of raw bytes consumed.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_tokenizer_bpe_backtrack_longest_match(
    const iree_tokenizer_bpe_model_t* model, const uint8_t* data,
    iree_host_size_t size, int32_t* out_token_id,
    iree_host_size_t* out_raw_length) {
  *out_token_id = -1;
  *out_raw_length = 0;

  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);

  const bool byte_level =
      iree_all_bits_set(model->flags, IREE_TOKENIZER_BPE_FLAG_BYTE_LEVEL_INPUT);

  for (iree_host_size_t raw_consumed = 0; raw_consumed < size;) {
    uint8_t byte = data[raw_consumed];
    if (!byte_level) {
      iree_host_size_t length = iree_tokenizer_bpe_character_length(
          data + raw_consumed, size - raw_consumed);
      for (iree_host_size_t i = 0; i < length; ++i) {
        if (!iree_tokenizer_trie_cursor_advance(&cursor,
                                                data[raw_consumed + i])) {
          return;
        }
      }
      raw_consumed += length;
    } else if (byte >= 0x21 && byte <= 0x7E) {
      // The ByteLevel mapping is identity for printable ASCII.
      if (!iree_tokenizer_trie_cursor_advance(&cursor, byte)) {
        break;
      }
      ++raw_consumed;
    } else {
      // Non-printable or high byte: use pre-computed UTF-8 from lookup table.
      // All non-identity ByteLevel mappings produce 2-byte UTF-8.
      const iree_tokenizer_byte_level_utf8_t* utf8 =
          &iree_tokenizer_byte_level_utf8[byte];
      if (!iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[0])) {
        break;
      }
      if (utf8->length == 2 &&
          !iree_tokenizer_trie_cursor_advance(&cursor, utf8->bytes[1])) {
        break;
      }
      ++raw_consumed;
    }

    // Only initial-symbol boundaries can end a canonical token. Reachability
    // excludes dictionary spellings that BPE cannot actually construct.
    int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
    if (token_id >= 0) {
      if (iree_any_bit_set(
              model->backtrack_tables.token_reachable[(uint32_t)token_id / 64],
              1ull << ((uint32_t)token_id % 64))) {
        *out_token_id = token_id;
        *out_raw_length = raw_consumed;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Table Construction (bpe_tables.c)
//===----------------------------------------------------------------------===//

// Builds the backtracking tables (split_table, effective_rank,
// token_reachable). Called during model allocation.
iree_status_t iree_tokenizer_bpe_build_backtrack_tables(
    iree_tokenizer_bpe_model_t* model);

// Checks if a token is reachable from its input bytes via BPE merge ordering.
// Uses the precomputed token_reachable bitmap from model load.
bool iree_tokenizer_bpe_is_first_token_reachable(
    const iree_tokenizer_bpe_model_t* model, uint32_t token);

//===----------------------------------------------------------------------===//
// Backtracking Algorithm (bpe_backtrack.c)
//===----------------------------------------------------------------------===//

// Runs the O(n) backtracking BPE encode algorithm on a segment.
// Populates the backtrack_stack with the resulting token sequence.
bool iree_tokenizer_bpe_backtrack_encode(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    const uint8_t* segment_data, iree_host_size_t segment_size);

iree_status_t iree_tokenizer_bpe_encode_symbols(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment);

//===----------------------------------------------------------------------===//
// Word Cache and Trie Lookup (bpe_cache.c)
//===----------------------------------------------------------------------===//

// Attempts to serve a segment from the word cache.
// Returns true if cache hit and all tokens emitted.
bool iree_tokenizer_bpe_cache_lookup(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment, iree_tokenizer_bpe_output_cursor_t* cursor);

// Populates a word cache entry from the backtrack stack after encoding.
void iree_tokenizer_bpe_cache_populate(
    const iree_tokenizer_bpe_model_t* model, iree_tokenizer_bpe_state_t* state,
    iree_string_view_t segment,
    const iree_tokenizer_bpe_backtrack_entry_t* stack,
    iree_host_size_t token_count);

// Finds the longest token with ByteLevel transformation applied.
void iree_tokenizer_bpe_trie_longest_match_byte_level(
    const iree_tokenizer_bpe_model_t* model, iree_string_view_t segment,
    int32_t* out_token_id, iree_host_size_t* out_length);

//===----------------------------------------------------------------------===//
// Encoding State Machine (bpe_encode.c)
//===----------------------------------------------------------------------===//

// Encodes segments to tokens using the resumable state machine.
// These functions implement the iree_tokenizer_model_vtable_t interface.
iree_status_t iree_tokenizer_bpe_state_encode(
    iree_tokenizer_model_state_t* state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed, iree_host_size_t* out_token_count);

iree_status_t iree_tokenizer_bpe_state_finalize(
    iree_tokenizer_model_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* out_token_count);

bool iree_tokenizer_bpe_state_has_pending(
    const iree_tokenizer_model_state_t* state);

iree_host_size_t iree_tokenizer_bpe_state_reclaim(
    iree_tokenizer_model_state_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_BPE_STATE_H_
