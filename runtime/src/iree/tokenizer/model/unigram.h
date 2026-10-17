// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_MODEL_UNIGRAM_H_
#define IREE_TOKENIZER_MODEL_UNIGRAM_H_

#include "iree/base/api.h"
#include "iree/tokenizer/model.h"
#include "iree/tokenizer/types.h"
#include "iree/tokenizer/vocab/vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Unigram Model
//===----------------------------------------------------------------------===//

// Configuration flags for Unigram tokenization.
typedef enum iree_tokenizer_unigram_flag_bits_e {
  IREE_TOKENIZER_UNIGRAM_FLAG_NONE = 0,
  // Disable byte expansion of selected unknown pieces. A piece first resolves
  // by vocabulary lookup; otherwise it emits the configured unknown ID.
  IREE_TOKENIZER_UNIGRAM_FLAG_NO_BYTE_FALLBACK = 1u << 0,
  // Preserve individual selected unknown pieces instead of fusing adjacent
  // ones before vocabulary lookup and byte expansion.
  IREE_TOKENIZER_UNIGRAM_FLAG_NO_FUSE_UNK = 1u << 1,
} iree_tokenizer_unigram_flag_bits_t;
typedef uint32_t iree_tokenizer_unigram_flags_t;

// Allocates a Unigram (SentencePiece) model from a vocabulary.
//
// Unigram uses Viterbi dynamic programming to find the segmentation with the
// highest total log-probability. Unlike BPE (merge-based) or WordPiece (greedy
// longest-match), Unigram considers ALL valid segmentations and picks the
// optimal one.
//
// A forward Viterbi pass visits each Unicode start once and preserves F64
// score order. The active frontier retains each future end's best path. Tokens
// are published only from the common prefix of all surviving paths. A segment
// boundary resolves the final path. Unknown transitions compete as complete
// scalars; their selected spellings are fused before output lookup or byte
// expansion. Every fallback byte carries the entire selected piece's span.
//
// Vocabulary width bounds input lookahead and frontier slots, but ambiguous
// paths and fused unknown spellings can grow with input length. Encoder states
// retain that workspace through their caller-supplied allocator and reuse its
// capacity on reset. Allocation failure is explicit; no chunk boundary changes
// the selected path. Input bytes can be reclaimed before their tokens resolve.
//
// |vocab| is the vocabulary containing tokens with scores. The model stores
//   a reference but does not take ownership — the caller must ensure vocab
//   outlives the model. Token spellings must be valid UTF-8. A prefix trie is
//   built from the vocabulary at allocation time for efficient DP traversal.
// |unk_token_id| is the resolved UNK token ID (from JSON's unk_id field).
//   Pass IREE_TOKENIZER_TOKEN_ID_INVALID if no UNK token is configured
//   (an improving unknown transition then fails during encoding).
// |unk_score| is the log-probability score used when inserting UNK candidates
//   in the Viterbi forward pass. Hugging Face import uses the minimum original
//   vocabulary score minus ten. Native callers choose this score explicitly.
// |flags| controls byte_fallback and UNK fusion behavior.
// |allocator| is used for model and trie allocation.
// |out_model| receives the allocated model on success.
iree_status_t iree_tokenizer_unigram_model_allocate(
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_token_id_t unk_token_id,
    double unk_score, iree_tokenizer_unigram_flags_t flags,
    iree_allocator_t allocator, iree_tokenizer_model_t** out_model);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_MODEL_UNIGRAM_H_
