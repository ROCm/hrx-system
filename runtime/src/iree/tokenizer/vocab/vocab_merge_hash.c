// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/vocab/vocab_merge_hash.h"

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab.h"

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_vocab_merge_hash_build(
    const struct iree_tokenizer_vocab_t* vocab, iree_allocator_t allocator,
    iree_tokenizer_vocab_merge_hash_t** out_hash) {
  IREE_ASSERT_ARGUMENT(vocab);
  IREE_ASSERT_ARGUMENT(out_hash);
  *out_hash = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t merge_count = iree_tokenizer_vocab_merge_count(vocab);
  if (merge_count > UINT32_MAX || merge_count > (IREE_HOST_SIZE_MAX - 6) / 10) {
    IREE_RETURN_AND_END_ZONE(
        z0,
        iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                         "too many BPE merge rules: %" PRIhsz, merge_count));
  }
  // Size table for ~70% load factor, rounded up to power of 2.
  // For empty merge lists, capacity=0 is valid (lookup handles it).
  iree_host_size_t min_capacity =
      merge_count > 0 ? (merge_count * 10 + 6) / 7 : 0;  // ceil(n / 0.7)
  iree_host_size_t capacity = merge_count > 0 ? 1 : 0;
  while (capacity < min_capacity) {
    capacity *= 2;
  }

  // Calculate total allocation size using IREE_STRUCT_LAYOUT for overflow
  // checking and proper FAM alignment.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(
          sizeof(iree_tokenizer_vocab_merge_hash_t), &total_size,
          IREE_STRUCT_FIELD_FAM(capacity, iree_tokenizer_merge_hash_entry_t)));

  iree_tokenizer_vocab_merge_hash_t* hash = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&hash));

  hash->allocator = allocator;
  hash->capacity = capacity;
  hash->count = merge_count;

  if (capacity == 0) {
    // No merges - return empty hash table (FAM has zero elements).
    *out_hash = hash;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Initialize all entries as empty (UINT64_MAX sentinel).
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    hash->entries[i].key = IREE_TOKENIZER_MERGE_HASH_EMPTY_KEY;
  }

  // A valid merge result is a vocabulary token, so one maximum-length token
  // buffer can resolve every rule. Keep the scratch allocation out of the loop.
  iree_host_size_t max_length = iree_tokenizer_vocab_max_token_length(vocab);
  char* concat_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, iree_max(max_length, 1), (void**)&concat_buffer);

  // Validate the model's rules before publishing the immutable table. A found
  // entry must always contain a valid result ID for the encoding hot path.
  iree_host_size_t mask = capacity - 1;
  for (iree_host_size_t rank = 0;
       rank < merge_count && iree_status_is_ok(status); ++rank) {
    iree_tokenizer_merge_t merge = iree_tokenizer_vocab_merge(vocab, rank);
    iree_string_view_t left_text =
        iree_tokenizer_vocab_token_text(vocab, merge.left_id);
    iree_string_view_t right_text =
        iree_tokenizer_vocab_token_text(vocab, merge.right_id);
    if (left_text.size == 0 || right_text.size == 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "BPE merge %" PRIhsz
                                " references missing or empty token (%" PRIu32
                                ", %" PRIu32 ")",
                                rank, merge.left_id, merge.right_id);
      continue;
    }

    int32_t result_id = -1;
    if (left_text.size <= max_length - right_text.size) {
      iree_host_size_t concat_length = left_text.size + right_text.size;
      memcpy(concat_buffer, left_text.data, left_text.size);
      memcpy(concat_buffer + left_text.size, right_text.data, right_text.size);
      result_id = iree_tokenizer_vocab_lookup(
          vocab, iree_make_string_view(concat_buffer, concat_length));
    }
    if (result_id < 0) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "BPE merge %" PRIhsz " result for tokens (%" PRIu32
                           ", %" PRIu32 ") is absent from the vocabulary",
                           rank, merge.left_id, merge.right_id);
      continue;
    }

    uint64_t key =
        iree_tokenizer_merge_hash_pack_key(merge.left_id, merge.right_id);
    iree_host_size_t index = iree_tokenizer_merge_hash_fn(key) & mask;
    while (hash->entries[index].key != IREE_TOKENIZER_MERGE_HASH_EMPTY_KEY) {
      index = (index + 1) & mask;
    }
    hash->entries[index].key = key;
    hash->entries[index].rank = (uint32_t)rank;
    hash->entries[index].result_id = result_id;
  }

  iree_allocator_free(allocator, concat_buffer);
  if (iree_status_is_ok(status)) {
    *out_hash = hash;
  } else {
    iree_tokenizer_vocab_merge_hash_free(hash);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Lookup is defined as inline in the header for performance.

void iree_tokenizer_vocab_merge_hash_free(
    iree_tokenizer_vocab_merge_hash_t* hash) {
  if (!hash) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = hash->allocator;
  // Entries are part of the same allocation as hash, so just free hash.
  iree_allocator_free(allocator, hash);
  IREE_TRACE_ZONE_END(z0);
}
