// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable vocabulary storage transferred from the vocabulary builder.

#ifndef IREE_TOKENIZER_VOCAB_STORAGE_H_
#define IREE_TOKENIZER_VOCAB_STORAGE_H_

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_hash.h"

#ifdef __cplusplus
extern "C" {
#endif

// Token and optional score arrays share one owned slab. The builder transfers
// this slab without copying when the vocabulary is finalized.
struct iree_tokenizer_vocab_t {
  // Allocator owning all vocabulary storage.
  iree_allocator_t allocator;

  // Owned combined [tokens][optional scores] allocation.
  void* token_slab;
  // Slab token array indexed by ID; sparse gap slots have ATTR_UNUSED set.
  iree_tokenizer_token_t* tokens;
  // Slab score array indexed by ID, or NULL when every score is zero.
  double* scores;
  // Actual number of tokens (excludes gaps).
  iree_host_size_t token_count;
  // Highest valid token ID (-1 if empty).
  int32_t max_token_id;

  // String table (owned).
  uint8_t* string_data;
  // Length of the string table in bytes.
  iree_host_size_t string_size;

  // Hash table embedded in the vocabulary allocation.
  iree_tokenizer_vocab_hash_t* hash;

  // Special token IDs.
  iree_tokenizer_special_ids_t special_ids;

  // BPE merge rules (owned, NULL for non-BPE vocabs).
  // Stored in rank order: index 0 = highest priority merge.
  iree_tokenizer_merge_t* merges;
  // Number of merge rules.
  iree_host_size_t merge_count;

  // Maximum token text length in bytes across all tokens.
  // Computed during vocab build by tracking max(string_length).
  // Useful for buffer sizing and BPE window bounds.
  iree_host_size_t max_token_length;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_VOCAB_STORAGE_H_
