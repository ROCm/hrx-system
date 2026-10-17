// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/unigram.h"

#include <string.h>

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/vocab/vocab_trie.h"

// Best paths form a tree rooted at the last published token. Active nodes own
// the best path to each unprocessed end position. An inactive root with one
// child proves that every possible continuation shares that child. Vocabulary
// width bounds the frontier, but ambiguous predecessor paths can be arbitrarily
// long. Those paths belong to the encoder's explicit workspace allocator.
#define IREE_TOKENIZER_UNIGRAM_NO_NODE UINT32_MAX

typedef struct iree_tokenizer_unigram_node_t {
  // Cumulative score, preserving the original addition order across
  // publication.
  double score;
  // Logical transformed-input position after this piece.
  iree_host_size_t end;
  // Predecessor index; free nodes reuse this field as a free-list link.
  uint32_t parent;
  // Number of retained children.
  uint32_t child_count;
  // XOR of child indices, directly identifying the only child when count is
  // one.
  uint32_t child_xor;
  // Selected vocabulary ID, including the configured unknown ID.
  iree_tokenizer_token_id_t token_id;
  // Original scalar bytes for an unknown transition, independent of input
  // storage.
  uint8_t unknown_bytes[IREE_UNICODE_UTF8_MAX_BYTE_LENGTH];
  // Nonzero for an unknown transition; known pieces borrow vocabulary spelling.
  uint8_t unknown_length;
  // Whether a future-end frontier slot owns this node.
  bool active;
} iree_tokenizer_unigram_node_t;
static_assert(sizeof(iree_tokenizer_unigram_node_t) <= 40,
              "Unigram path nodes must fit in 40 bytes");

typedef struct iree_tokenizer_unigram_model_t {
  // Immutable model interface.
  iree_tokenizer_model_t base;
  // Allocator owning the model and its trie.
  iree_allocator_t allocator;
  // Borrowed immutable vocabulary, which outlives the model.
  const iree_tokenizer_vocab_t* vocab;
  // Owned prefix lookup table.
  iree_tokenizer_vocab_trie_t* trie;
  // Configured unknown ID, or INVALID when unknown transitions are forbidden.
  iree_tokenizer_token_id_t unknown_id;
  // Score of a character-level unknown transition.
  double unknown_score;
  // Output policy for selected unknown pieces.
  iree_tokenizer_unigram_flags_t flags;
  // Byte value to fallback token ID, or INVALID for absent byte tokens.
  iree_tokenizer_token_id_t byte_to_token[256];
  // Maximum token bytes, with room for one complete Unicode scalar.
  iree_host_size_t lookahead;
  // Number of future-end slots and initial inline path nodes.
  iree_host_size_t frontier_capacity;
  // State slab offset of future-end node indices.
  iree_host_size_t frontier_offset;
  // State slab offset of the inline node pool.
  iree_host_size_t nodes_offset;
  // State slab offset of retained input suffix storage (twice lookahead).
  iree_host_size_t suffix_offset;
  // State slab offset of inline fused-unknown storage (lookahead bytes).
  iree_host_size_t unknown_offset;
} iree_tokenizer_unigram_model_t;

typedef struct iree_tokenizer_unigram_state_t {
  // Model dispatch interface; the model outlives this state.
  iree_tokenizer_model_state_t base;
  // Caller-controlled policy for path and output storage growth.
  iree_allocator_t workspace_allocator;
  // Inline or owned pool; all tree links are indices across reallocation.
  iree_tokenizer_unigram_node_t* nodes;
  // Allocated node slots, retained across reset.
  iree_host_size_t node_capacity;
  // Initialized slots at the high-water mark of this segment.
  uint32_t node_count;
  // First reusable node, or NO_NODE.
  uint32_t free_head;
  // Last published path node; owns all surviving future paths.
  uint32_t root;
  // Whether a segment still owns input, paths, or output.
  bool active;
  // Whether the final frontier owner has been retired at the segment boundary.
  bool ended;
  // Next logical input byte not yet processed or copied into suffix storage.
  iree_host_size_t input_position;
  // Logical position of the next Unicode scalar to process.
  iree_host_size_t position;
  // First accepted byte not yet reported through state_reclaim.
  iree_host_size_t reclaim_position;
  // Start of retained suffix bytes in the bounded inline suffix allocation.
  iree_host_size_t suffix_start;
  // Number of retained input bytes awaiting further lookahead.
  iree_host_size_t suffix_length;
  // Inline or owned selected unknown spelling, retained until its run ends.
  uint8_t* unknown_bytes;
  // Allocated unknown bytes, retained across reset.
  iree_host_size_t unknown_capacity;
  // Selected unknown spelling length.
  iree_host_size_t unknown_length;
  // First byte of the selected unknown run in logical input coordinates.
  iree_host_size_t unknown_start;
  // Next fallback byte to emit after output selection.
  iree_host_size_t unknown_cursor;
  // Single output ID, or INVALID when emitting individual fallback bytes.
  iree_tokenizer_token_id_t unknown_token;
  // Whether the run is closed and its output representation has been selected.
  bool unknown_ready;
} iree_tokenizer_unigram_state_t;

static const iree_tokenizer_model_vtable_t iree_tokenizer_unigram_model_vtable;

static const iree_tokenizer_unigram_model_t* iree_tokenizer_unigram_model(
    const iree_tokenizer_unigram_state_t* state) {
  return (const iree_tokenizer_unigram_model_t*)state->base.model;
}

static uint32_t* iree_tokenizer_unigram_frontier(
    iree_tokenizer_unigram_state_t* state) {
  return (uint32_t*)((uint8_t*)state +
                     iree_tokenizer_unigram_model(state)->frontier_offset);
}

static iree_tokenizer_unigram_node_t* iree_tokenizer_unigram_inline_nodes(
    iree_tokenizer_unigram_state_t* state) {
  return (iree_tokenizer_unigram_node_t*)((uint8_t*)state +
                                          iree_tokenizer_unigram_model(state)
                                              ->nodes_offset);
}

static uint8_t* iree_tokenizer_unigram_suffix(
    iree_tokenizer_unigram_state_t* state) {
  return (uint8_t*)state + iree_tokenizer_unigram_model(state)->suffix_offset;
}

static uint8_t* iree_tokenizer_unigram_inline_unknown(
    iree_tokenizer_unigram_state_t* state) {
  return (uint8_t*)state + iree_tokenizer_unigram_model(state)->unknown_offset;
}

// Reset invalidates all live indices without walking or clearing the pool.
// Grown allocations remain owned and reusable by this encoder.
static void iree_tokenizer_unigram_state_reset(
    iree_tokenizer_model_state_t* base_state) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  memset(iree_tokenizer_unigram_frontier(state), 0xFF,
         model->frontier_capacity * sizeof(uint32_t));
  state->node_count = 0;
  state->free_head = IREE_TOKENIZER_UNIGRAM_NO_NODE;
  state->root = IREE_TOKENIZER_UNIGRAM_NO_NODE;
  state->active = false;
  state->ended = false;
  state->input_position = 0;
  state->position = 0;
  state->reclaim_position = 0;
  state->suffix_start = 0;
  state->suffix_length = 0;
  state->unknown_length = 0;
  state->unknown_cursor = 0;
  state->unknown_ready = false;
}

static void iree_tokenizer_unigram_state_initialize(
    const iree_tokenizer_model_t* base_model, void* storage,
    iree_allocator_t workspace_allocator,
    iree_tokenizer_model_state_t** out_state) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.model = base_model;
  state->workspace_allocator = workspace_allocator;
  state->nodes = iree_tokenizer_unigram_inline_nodes(state);
  state->node_capacity = iree_tokenizer_unigram_model(state)->frontier_capacity;
  state->unknown_bytes = iree_tokenizer_unigram_inline_unknown(state);
  state->unknown_capacity = iree_tokenizer_unigram_model(state)->lookahead;
  iree_tokenizer_unigram_state_reset(&state->base);
  *out_state = &state->base;
}

static void iree_tokenizer_unigram_state_deinitialize(
    iree_tokenizer_model_state_t* base_state) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  if (state->nodes != iree_tokenizer_unigram_inline_nodes(state)) {
    iree_allocator_free(state->workspace_allocator, state->nodes);
  }
  if (state->unknown_bytes != iree_tokenizer_unigram_inline_unknown(state)) {
    iree_allocator_free(state->workspace_allocator, state->unknown_bytes);
  }
}

static void iree_tokenizer_unigram_begin(iree_tokenizer_unigram_state_t* state,
                                         iree_host_size_t position) {
  state->node_count = 1;
  state->free_head = IREE_TOKENIZER_UNIGRAM_NO_NODE;
  state->root = 0;
  state->nodes[0] = (iree_tokenizer_unigram_node_t){
      .end = position,
      .parent = IREE_TOKENIZER_UNIGRAM_NO_NODE,
      .token_id = IREE_TOKENIZER_TOKEN_ID_INVALID,
      .active = true,
  };
  iree_tokenizer_unigram_frontier(
      state)[position %
             iree_tokenizer_unigram_model(state)->frontier_capacity] = 0;
  state->active = true;
  state->ended = false;
  state->input_position = position;
  state->position = position;
  state->reclaim_position = position;
}

static iree_status_t iree_tokenizer_unigram_allocate_node(
    iree_tokenizer_unigram_state_t* state, uint32_t* out_index) {
  if (state->free_head != IREE_TOKENIZER_UNIGRAM_NO_NODE) {
    *out_index = state->free_head;
    state->free_head = state->nodes[*out_index].parent;
    return iree_ok_status();
  }
  if (state->node_count == state->node_capacity) {
    if (state->node_capacity == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Unigram path node index capacity exhausted");
    }
    iree_host_size_t capacity = state->node_capacity > UINT32_MAX / 2
                                    ? UINT32_MAX
                                    : state->node_capacity * 2;
    if (state->nodes == iree_tokenizer_unigram_inline_nodes(state)) {
      iree_tokenizer_unigram_node_t* nodes = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
          state->workspace_allocator, capacity, sizeof(*nodes),
          (void**)&nodes));
      memcpy(nodes, state->nodes, state->node_count * sizeof(*nodes));
      state->nodes = nodes;
    } else {
      IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
          state->workspace_allocator, capacity, sizeof(*state->nodes),
          (void**)&state->nodes));
    }
    state->node_capacity = capacity;
  }
  *out_index = state->node_count++;
  return iree_ok_status();
}

static void iree_tokenizer_unigram_detach(iree_tokenizer_unigram_state_t* state,
                                          uint32_t parent, uint32_t child) {
  --state->nodes[parent].child_count;
  state->nodes[parent].child_xor ^= child;
}

static void iree_tokenizer_unigram_free_node(
    iree_tokenizer_unigram_state_t* state, uint32_t index) {
  state->nodes[index].parent = state->free_head;
  state->free_head = index;
}

static void iree_tokenizer_unigram_retire(iree_tokenizer_unigram_state_t* state,
                                          uint32_t index) {
  while (index != state->root && !state->nodes[index].active &&
         state->nodes[index].child_count == 0) {
    uint32_t parent = state->nodes[index].parent;
    iree_tokenizer_unigram_detach(state, parent, index);
    iree_tokenizer_unigram_free_node(state, index);
    index = parent;
  }
}

// Only unprocessed destinations can change predecessor. They have no children
// yet; updating one cannot invalidate a path already used by another start.
static iree_status_t iree_tokenizer_unigram_edge(
    iree_tokenizer_unigram_state_t* state, uint32_t parent,
    iree_host_size_t end, iree_tokenizer_token_id_t token_id, double score,
    const uint8_t* unknown_bytes, uint8_t unknown_length) {
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  uint32_t* slot =
      &iree_tokenizer_unigram_frontier(state)[end % model->frontier_capacity];
  double candidate = state->nodes[parent].score + score;
  uint32_t index = *slot;
  if (index != IREE_TOKENIZER_UNIGRAM_NO_NODE &&
      !(candidate > state->nodes[index].score)) {
    return iree_ok_status();
  }
  if (token_id == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Unigram encountered unknown input without an "
                            "unknown token at byte %" PRIhsz,
                            state->position);
  }
  uint32_t previous = IREE_TOKENIZER_UNIGRAM_NO_NODE;
  if (index == IREE_TOKENIZER_UNIGRAM_NO_NODE) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_unigram_allocate_node(state, &index));
    *slot = index;
  } else {
    previous = state->nodes[index].parent;
    iree_tokenizer_unigram_detach(state, previous, index);
  }
  state->nodes[index] = (iree_tokenizer_unigram_node_t){
      .score = candidate,
      .end = end,
      .parent = parent,
      .token_id = token_id,
      .unknown_length = unknown_length,
      .active = true,
  };
  if (unknown_length) {
    memcpy(state->nodes[index].unknown_bytes, unknown_bytes, unknown_length);
  }
  ++state->nodes[parent].child_count;
  state->nodes[parent].child_xor ^= index;
  if (previous != IREE_TOKENIZER_UNIGRAM_NO_NODE) {
    iree_tokenizer_unigram_retire(state, previous);
  }
  return iree_ok_status();
}

// Both spans are borrowed for this call. The first can be a retained suffix;
// the second then supplies the new arrival without copying it into that suffix.
static iree_status_t iree_tokenizer_unigram_step(
    iree_tokenizer_unigram_state_t* state, iree_string_view_t first,
    iree_string_view_t second, iree_host_size_t* out_width) {
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  uint8_t bytes[IREE_UNICODE_UTF8_MAX_BYTE_LENGTH];
  iree_host_size_t width =
      iree_unicode_utf8_sequence_length((uint8_t)first.data[0]);
  iree_host_size_t prefix = iree_min(width, first.size);
  memcpy(bytes, first.data, prefix);
  if (width > first.size + second.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Unigram input ends inside UTF-8 at byte %" PRIhsz,
                            state->position);
  }
  if (prefix < width) {
    memcpy(bytes + prefix, second.data, width - prefix);
  }
  if (!iree_unicode_utf8_is_valid_sequence(bytes, width)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Unigram input is not UTF-8 at byte %" PRIhsz,
                            state->position);
  }
  uint32_t* frontier = iree_tokenizer_unigram_frontier(state);
  iree_host_size_t slot = state->position % model->frontier_capacity;
  uint32_t parent = frontier[slot];
  iree_tokenizer_trie_cursor_t cursor;
  iree_tokenizer_trie_cursor_reset(&cursor, model->trie);
  bool has_scalar = false;
  bool matching = true;
  iree_string_view_t spans[] = {first, second};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t part = 0;
       part < IREE_ARRAYSIZE(spans) && matching && iree_status_is_ok(status);
       ++part) {
    for (iree_host_size_t i = 0;
         i < spans[part].size && matching && iree_status_is_ok(status); ++i) {
      matching = iree_tokenizer_trie_cursor_advance(
          &cursor, (uint8_t)spans[part].data[i]);
      if (!matching) {
        break;
      }
      int32_t token_id = iree_tokenizer_trie_cursor_token_id(&cursor);
      if (token_id >= 0) {
        iree_host_size_t length = iree_tokenizer_trie_cursor_depth(&cursor);
        status = iree_tokenizer_unigram_edge(
            state, parent, state->position + length, token_id,
            iree_tokenizer_vocab_token_score(model->vocab, token_id), NULL, 0);
        has_scalar |= length == width;
      }
    }
  }
  if (iree_status_is_ok(status) && !has_scalar) {
    status = iree_tokenizer_unigram_edge(
        state, parent, state->position + width, model->unknown_id,
        model->unknown_score, bytes, (uint8_t)width);
  }
  if (iree_status_is_ok(status)) {
    frontier[slot] = IREE_TOKENIZER_UNIGRAM_NO_NODE;
    state->nodes[parent].active = false;
    iree_tokenizer_unigram_retire(state, parent);
    state->position += width;
    *out_width = width;
  }
  return status;
}

static iree_status_t iree_tokenizer_unigram_append_unknown(
    iree_tokenizer_unigram_state_t* state, iree_string_view_t piece,
    iree_host_size_t start) {
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_add(state->unknown_length, piece.size, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Unigram unknown spelling size overflow");
  }
  if (length > state->unknown_capacity) {
    if (state->unknown_bytes == iree_tokenizer_unigram_inline_unknown(state)) {
      iree_host_size_t capacity = iree_max(length, state->unknown_capacity * 2);
      uint8_t* bytes = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
          state->workspace_allocator, capacity, (void**)&bytes));
      memcpy(bytes, state->unknown_bytes, state->unknown_length);
      state->unknown_bytes = bytes;
      state->unknown_capacity = capacity;
    } else {
      IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
          state->workspace_allocator, length, 1, &state->unknown_capacity,
          (void**)&state->unknown_bytes));
    }
  }
  if (state->unknown_length == 0) {
    state->unknown_start = start;
  }
  memcpy(state->unknown_bytes + state->unknown_length, piece.data, piece.size);
  state->unknown_length = length;
  return iree_ok_status();
}

static void iree_tokenizer_unigram_close_unknown(
    iree_tokenizer_unigram_state_t* state) {
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  state->unknown_token = iree_tokenizer_vocab_lookup(
      model->vocab, iree_make_string_view((const char*)state->unknown_bytes,
                                          state->unknown_length));
  if (state->unknown_token == IREE_TOKENIZER_TOKEN_ID_INVALID) {
    bool use_bytes = !iree_any_bit_set(
        model->flags, IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK);
    for (iree_host_size_t i = 0; i < state->unknown_length && use_bytes; ++i) {
      use_bytes = model->byte_to_token[state->unknown_bytes[i]] !=
                  IREE_TOKENIZER_TOKEN_ID_INVALID;
    }
    if (!use_bytes) {
      state->unknown_token = model->unknown_id;
    }
  }
  state->unknown_cursor = 0;
  state->unknown_ready = true;
}

static void iree_tokenizer_unigram_emit(iree_tokenizer_token_output_t output,
                                        iree_host_size_t* count,
                                        iree_tokenizer_token_id_t token_id,
                                        iree_host_size_t start,
                                        iree_host_size_t end) {
  output.token_ids[*count] = token_id;
  if (output.token_offsets) {
    output.token_offsets[*count] = (iree_tokenizer_offset_t){start, end};
  }
  ++*count;
}

static void iree_tokenizer_unigram_drain_unknown(
    iree_tokenizer_unigram_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* count) {
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  iree_host_size_t end = state->unknown_start + state->unknown_length;
  while (state->unknown_cursor < state->unknown_length &&
         *count < output.capacity) {
    int32_t token_id = state->unknown_token;
    if (token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
      state->unknown_cursor = state->unknown_length;
    } else {
      token_id =
          model->byte_to_token[state->unknown_bytes[state->unknown_cursor++]];
    }
    iree_tokenizer_unigram_emit(output, count, token_id, state->unknown_start,
                                end);
  }
  if (state->unknown_cursor == state->unknown_length) {
    state->unknown_length = 0;
    state->unknown_ready = false;
  }
}

static iree_status_t iree_tokenizer_unigram_publish(
    iree_tokenizer_unigram_state_t* state, iree_tokenizer_token_output_t output,
    iree_host_size_t* count) {
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  iree_status_t status = iree_ok_status();
  while (*count < output.capacity && iree_status_is_ok(status)) {
    if (state->unknown_ready) {
      iree_tokenizer_unigram_drain_unknown(state, output, count);
      continue;
    }
    iree_tokenizer_unigram_node_t* root = &state->nodes[state->root];
    if (root->active || root->child_count != 1) {
      break;
    }
    uint32_t child = root->child_xor;
    const iree_tokenizer_unigram_node_t* node = &state->nodes[child];
    bool unknown = node->token_id == model->unknown_id;
    if (!unknown && state->unknown_length) {
      iree_tokenizer_unigram_close_unknown(state);
      continue;
    }
    iree_string_view_t piece =
        node->unknown_length
            ? iree_make_string_view((const char*)node->unknown_bytes,
                                    node->unknown_length)
            : iree_tokenizer_vocab_token_text(model->vocab, node->token_id);
    if (unknown) {
      status = iree_tokenizer_unigram_append_unknown(state, piece,
                                                     node->end - piece.size);
      if (iree_status_is_ok(status) &&
          iree_any_bit_set(model->flags,
                           IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK)) {
        iree_tokenizer_unigram_close_unknown(state);
      }
    } else {
      iree_tokenizer_unigram_emit(output, count, node->token_id,
                                  node->end - piece.size, node->end);
    }
    if (iree_status_is_ok(status)) {
      uint32_t previous = state->root;
      state->root = child;
      state->nodes[child].parent = IREE_TOKENIZER_UNIGRAM_NO_NODE;
      iree_tokenizer_unigram_free_node(state, previous);
    }
  }
  return status;
}

// Only the unresolved suffix is copied. Full input and new arrivals remain
// borrowed while traversing; no whole-segment buffer or prefix replay exists.
static void iree_tokenizer_unigram_retain_suffix(
    iree_tokenizer_unigram_state_t* state, iree_string_view_t input) {
  uint8_t* suffix = iree_tokenizer_unigram_suffix(state);
  iree_host_size_t capacity =
      iree_tokenizer_unigram_model(state)->lookahead * 2;
  if (state->suffix_start + state->suffix_length + input.size > capacity) {
    memmove(suffix, suffix + state->suffix_start, state->suffix_length);
    state->suffix_start = 0;
  }
  if (input.size) {
    memcpy(suffix + state->suffix_start + state->suffix_length, input.data,
           input.size);
    state->suffix_length += input.size;
    state->input_position += input.size;
  }
}

// EOF releases the last frontier owner; the remaining tree is the exact path.
// Output delivery can continue independently after this transition.
static void iree_tokenizer_unigram_end_input(
    iree_tokenizer_unigram_state_t* state) {
  if (state->ended) {
    return;
  }
  iree_host_size_t slot =
      state->position % iree_tokenizer_unigram_model(state)->frontier_capacity;
  uint32_t terminal = iree_tokenizer_unigram_frontier(state)[slot];
  iree_tokenizer_unigram_frontier(state)[slot] = IREE_TOKENIZER_UNIGRAM_NO_NODE;
  state->nodes[terminal].active = false;
  state->ended = true;
}

static iree_status_t iree_tokenizer_unigram_process(
    iree_tokenizer_unigram_state_t* state, iree_string_view_t input, bool final,
    iree_tokenizer_token_output_t output, iree_host_size_t* count) {
  const iree_tokenizer_unigram_model_t* model =
      iree_tokenizer_unigram_model(state);
  iree_status_t status = iree_ok_status();
  while (*count < output.capacity && iree_status_is_ok(status)) {
    iree_host_size_t available = state->suffix_length + input.size;
    if (final && available == 0) {
      iree_tokenizer_unigram_end_input(state);
    }
    status = iree_tokenizer_unigram_publish(state, output, count);
    if (!iree_status_is_ok(status) || *count == output.capacity) {
      break;
    }
    if (state->ended) {
      if (state->unknown_length) {
        iree_tokenizer_unigram_close_unknown(state);
        continue;
      }
      break;
    }
    if (!final && available < model->lookahead) {
      iree_tokenizer_unigram_retain_suffix(state, input);
      break;
    }
    iree_string_view_t first = input;
    iree_string_view_t second = iree_string_view_empty();
    if (state->suffix_length) {
      first = iree_make_string_view(
          (const char*)iree_tokenizer_unigram_suffix(state) +
              state->suffix_start,
          state->suffix_length);
      second = input;
    }
    iree_host_size_t width = 0;
    status = iree_tokenizer_unigram_step(state, first, second, &width);
    if (iree_status_is_ok(status)) {
      iree_host_size_t retained = iree_min(width, state->suffix_length);
      state->suffix_start += retained;
      state->suffix_length -= retained;
      if (!state->suffix_length) {
        state->suffix_start = 0;
      }
      iree_host_size_t borrowed = width - retained;
      if (borrowed) {
        input.data += borrowed;
      }
      input.size -= borrowed;
      state->input_position += borrowed;
    }
  }
  // Input retirement and segment completion need no additional output credit.
  // In particular, an exactly full output can still consume the final segment.
  if (iree_status_is_ok(status) && final && input.size == 0 &&
      state->suffix_length == 0) {
    iree_tokenizer_unigram_end_input(state);
    if (state->nodes[state->root].child_count == 0 &&
        state->unknown_length == 0) {
      state->active = false;
    }
  }
  return status;
}

static iree_status_t iree_tokenizer_unigram_state_encode(
    iree_tokenizer_model_state_t* base_state,
    iree_const_byte_span_t transform_buffer,
    iree_tokenizer_segment_list_t segments,
    iree_tokenizer_token_output_t output,
    iree_host_size_t* out_segments_consumed,
    iree_host_size_t* out_token_count) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  iree_host_size_t consumed = 0;
  iree_host_size_t count = 0;
  iree_status_t status = iree_ok_status();
  while (consumed < segments.count && count < output.capacity &&
         iree_status_is_ok(status)) {
    const iree_tokenizer_segment_t* segment = &segments.values[consumed];
    if (segment->start < segments.buffer_offset ||
        segment->start > segment->end ||
        segment->end - segments.buffer_offset > transform_buffer.data_length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Unigram segment is outside the input buffer");
      break;
    }
    if (!state->active) {
      iree_tokenizer_unigram_begin(state, segment->start);
    }
    if (state->input_position < segment->start ||
        state->input_position > segment->end) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "Unigram continuation omits retained input");
      break;
    }
    bool partial = segments.last_is_partial && consumed + 1 == segments.count;
    iree_string_view_t input = iree_make_string_view(
        (const char*)transform_buffer.data +
            (state->input_position - segments.buffer_offset),
        segment->end - state->input_position);
    status =
        iree_tokenizer_unigram_process(state, input, !partial, output, &count);
    if (iree_status_is_ok(status) && !state->active) {
      ++consumed;
    } else {
      break;
    }
  }
  *out_segments_consumed = consumed;
  *out_token_count = count;
  return status;
}

static iree_status_t iree_tokenizer_unigram_state_finalize(
    iree_tokenizer_model_state_t* base_state,
    iree_tokenizer_token_output_t output, iree_host_size_t* out_token_count) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  *out_token_count = 0;
  if (!state->active) {
    return iree_ok_status();
  }
  return iree_tokenizer_unigram_process(state, iree_string_view_empty(), true,
                                        output, out_token_count);
}

static bool iree_tokenizer_unigram_state_has_pending(
    const iree_tokenizer_model_state_t* base_state) {
  return ((const iree_tokenizer_unigram_state_t*)base_state)->active;
}

static iree_host_size_t iree_tokenizer_unigram_state_reclaim(
    iree_tokenizer_model_state_t* base_state) {
  iree_tokenizer_unigram_state_t* state =
      (iree_tokenizer_unigram_state_t*)base_state;
  if (!state->active) {
    return 0;
  }
  iree_host_size_t reclaimed = state->input_position - state->reclaim_position;
  state->reclaim_position = state->input_position;
  return reclaimed;
}

static void iree_tokenizer_unigram_model_destroy(
    iree_tokenizer_model_t* base_model) {
  iree_tokenizer_unigram_model_t* model =
      (iree_tokenizer_unigram_model_t*)base_model;
  iree_tokenizer_vocab_trie_free(model->trie);
  iree_allocator_free(model->allocator, model);
}

iree_status_t iree_tokenizer_unigram_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_token_id_t unk_token_id,
    double unk_score, iree_tokenizer_unigram_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_model);
  *out_model = NULL;
  const iree_tokenizer_token_t* tokens = iree_tokenizer_vocab_tokens(vocab);
  iree_host_size_t token_count = iree_tokenizer_vocab_capacity(vocab);
  iree_const_byte_span_t strings = iree_tokenizer_vocab_string_table(vocab);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < token_count && iree_status_is_ok(status);
       ++i) {
    if (tokens[i].string_length &&
        !iree_unicode_utf8_validate(iree_make_string_view(
            (const char*)strings.data + tokens[i].string_offset,
            tokens[i].string_length))) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Unigram vocabulary token %" PRIhsz " is not UTF-8", i);
    }
  }
  if (!iree_status_is_ok(status)) {
    return status;
  }
  iree_host_size_t lookahead =
      iree_max(iree_tokenizer_vocab_max_token_length(vocab),
               (iree_host_size_t)IREE_UNICODE_UTF8_MAX_BYTE_LENGTH);
  iree_host_size_t frontier_capacity = lookahead + 1;
  iree_host_size_t state_size = 0;
  iree_host_size_t frontier_offset = 0, nodes_offset = 0;
  iree_host_size_t suffix_offset = 0, unknown_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_tokenizer_unigram_state_t), &state_size,
      IREE_STRUCT_FIELD(frontier_capacity, uint32_t, &frontier_offset),
      IREE_STRUCT_FIELD(frontier_capacity, iree_tokenizer_unigram_node_t,
                        &nodes_offset),
      IREE_STRUCT_FIELD(lookahead * 2, uint8_t, &suffix_offset),
      IREE_STRUCT_FIELD(lookahead, uint8_t, &unknown_offset)));
  iree_tokenizer_unigram_model_t* model = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*model), (void**)&model));
  model->allocator = allocator;
  model->vocab = vocab;
  model->unknown_id = unk_token_id;
  model->unknown_score = unk_score;
  model->flags = flags;
  model->lookahead = lookahead;
  model->frontier_capacity = frontier_capacity;
  model->frontier_offset = frontier_offset;
  model->nodes_offset = nodes_offset;
  model->suffix_offset = suffix_offset;
  model->unknown_offset = unknown_offset;
  for (int byte = 0; byte < 256; ++byte) {
    char spelling[7];
    iree_snprintf(spelling, sizeof(spelling), "<0x%02X>", byte);
    model->byte_to_token[byte] =
        iree_tokenizer_vocab_lookup(vocab, iree_make_string_view(spelling, 6));
  }
  status = iree_tokenizer_vocab_trie_build(tokens, token_count, strings,
                                           allocator, &model->trie);
  if (iree_status_is_ok(status)) {
    iree_tokenizer_model_initialize(&model->base,
                                    &iree_tokenizer_unigram_model_vtable,
                                    state_size, IREE_SV("Unigram"));
    *out_model = &model->base;
  } else {
    iree_tokenizer_unigram_model_destroy(&model->base);
  }
  return status;
}

static iree_status_t iree_tokenizer_unigram_get_token_string(
    const iree_tokenizer_model_t* base_model,
    iree_tokenizer_token_id_t token_id, iree_string_view_t* out_string) {
  const iree_tokenizer_unigram_model_t* model =
      (const iree_tokenizer_unigram_model_t*)base_model;
  *out_string = iree_tokenizer_vocab_token_text(model->vocab, token_id);
  if (out_string->size == 0 && token_id >= 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "token ID %" PRId32 " not in vocabulary", token_id);
  }
  return iree_ok_status();
}

static const iree_tokenizer_model_vtable_t iree_tokenizer_unigram_model_vtable =
    {
        .destroy = iree_tokenizer_unigram_model_destroy,
        .state_initialize = iree_tokenizer_unigram_state_initialize,
        .state_reset = iree_tokenizer_unigram_state_reset,
        .state_deinitialize = iree_tokenizer_unigram_state_deinitialize,
        .state_encode = iree_tokenizer_unigram_state_encode,
        .state_finalize = iree_tokenizer_unigram_state_finalize,
        .state_has_pending = iree_tokenizer_unigram_state_has_pending,
        .state_reclaim = iree_tokenizer_unigram_state_reclaim,
        .get_token_string = iree_tokenizer_unigram_get_token_string,
};
