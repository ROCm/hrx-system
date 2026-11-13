// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Normalizer interface for tokenizer preprocessing.
//
// Normalizers transform text before segmentation: NFC normalization,
// lowercasing, stripping, etc. Each normalizer type implements the vtable
// interface for pull-based streaming processing.
//
// Design principles:
// - Pull-based: caller controls flow via output buffer size
// - Batched: output buffer size determines processing batch size
// - Zero allocation: state lives in caller-provided storage
// - Composable: Sequence normalizer chains multiple normalizers

#ifndef IREE_TOKENIZER_NORMALIZER_H_
#define IREE_TOKENIZER_NORMALIZER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_tokenizer_normalizer_t iree_tokenizer_normalizer_t;
typedef struct iree_tokenizer_normalizer_state_t
    iree_tokenizer_normalizer_state_t;
typedef struct iree_tokenizer_normalizer_vtable_t
    iree_tokenizer_normalizer_vtable_t;

//===----------------------------------------------------------------------===//
// Normalizer Flags
//===----------------------------------------------------------------------===//

// Properties of the original input passed to state_process(). Input boundaries
// are communicated by state_finalize() after all input has been consumed.
typedef uint32_t iree_tokenizer_normalizer_flags_t;
enum iree_tokenizer_normalizer_flag_bits_e {
  IREE_TOKENIZER_NORMALIZER_FLAG_NONE = 0,

  // Position 0 of the original input was consumed by a special token. This
  // signals to the prepend normalizer that it should skip prepending for this
  // segment even though it's the first text the normalizer sees.
  //
  // For example, with prepend_scheme="first" and input "<s>hello":
  //   - Special token "<s>" matches at position 0
  //   - Text "hello" follows, but it's NOT at the "first" position
  //   - Without this flag, prepend would emit "▁hello" (incorrect)
  //   - With this flag, prepend knows to skip, emitting "hello" (correct)
  IREE_TOKENIZER_NORMALIZER_FLAG_FIRST_CONSUMED = 1u << 1,
};

//===----------------------------------------------------------------------===//
// Normalizer Base Type
//===----------------------------------------------------------------------===//

// Base normalizer structure. All concrete normalizer types embed this at
// offset 0. The vtable provides type-specific operations; state_size is
// cached at creation time for efficient state allocation.
struct iree_tokenizer_normalizer_t {
  const iree_tokenizer_normalizer_vtable_t* vtable;  // Must be at offset 0.
  // Size of state struct, cached at creation.
  iree_host_size_t state_size;
};

// Base streaming state structure. All concrete state types embed this at
// offset 0. The normalizer back-pointer enables vtable dispatch from state.
struct iree_tokenizer_normalizer_state_t {
  const iree_tokenizer_normalizer_t* normalizer;
};

//===----------------------------------------------------------------------===//
// Normalizer VTable
//===----------------------------------------------------------------------===//

// VTable for normalizer implementations.
//
// Pull-based processing model:
// - state_process() consumes input and writes to output
// - Caller controls batch size via output buffer capacity
// - Accepted input and pending output remain owned by the state
// - state_finalize() flushes remaining buffered data
struct iree_tokenizer_normalizer_vtable_t {
  void (*destroy)(iree_tokenizer_normalizer_t* normalizer);
  iree_status_t (*state_initialize)(
      const iree_tokenizer_normalizer_t* normalizer, void* storage,
      iree_tokenizer_normalizer_state_t** out_state);
  void (*state_deinitialize)(iree_tokenizer_normalizer_state_t* state);
  iree_status_t (*state_process)(iree_tokenizer_normalizer_state_t* state,
                                 iree_string_view_t input,
                                 iree_mutable_string_view_t output,
                                 iree_tokenizer_normalizer_flags_t flags,
                                 iree_host_size_t* IREE_RESTRICT out_consumed,
                                 iree_host_size_t* IREE_RESTRICT out_written);
  iree_status_t (*state_finalize)(iree_tokenizer_normalizer_state_t* state,
                                  iree_mutable_string_view_t output,
                                  iree_host_size_t* IREE_RESTRICT out_written);
  bool (*state_has_pending)(const iree_tokenizer_normalizer_state_t* state);
};

//===----------------------------------------------------------------------===//
// Normalizer Public API
//===----------------------------------------------------------------------===//

// Initializes base normalizer fields. Called by concrete implementations
// after allocating the normalizer struct.
void iree_tokenizer_normalizer_initialize(
    iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_normalizer_vtable_t* vtable,
    iree_host_size_t state_size);

// Frees a normalizer. Safe to call with NULL.
void iree_tokenizer_normalizer_free(iree_tokenizer_normalizer_t* normalizer);

// Returns the size of state storage required for this normalizer.
static inline iree_host_size_t iree_tokenizer_normalizer_state_size(
    const iree_tokenizer_normalizer_t* normalizer) {
  return normalizer->state_size;
}

// Initializes streaming state in caller-provided storage.
// |storage| must be at least iree_tokenizer_normalizer_state_size() bytes.
static inline iree_status_t iree_tokenizer_normalizer_state_initialize(
    const iree_tokenizer_normalizer_t* normalizer, void* storage,
    iree_tokenizer_normalizer_state_t** out_state) {
  return normalizer->vtable->state_initialize(normalizer, storage, out_state);
}

// Deinitializes streaming state (does not free storage).
static inline void iree_tokenizer_normalizer_state_deinitialize(
    iree_tokenizer_normalizer_state_t* state) {
  if (state && state->normalizer) {
    state->normalizer->vtable->state_deinitialize(state);
  }
}

// Processes input text, writing normalized output.
//
// Reads up to |input.size| bytes and writes up to |output.size| bytes.
// |flags| communicates properties of the original input.
// Returns bytes consumed in |out_consumed| and bytes written in |out_written|.
//
// May consume less than the full input if the output buffer is full or if the
// normalizer is waiting for more input to complete a normalization unit (e.g.,
// combining characters for NFC composition). Partial UTF-8 sequences are not
// handled here — the caller is responsible for ensuring input chunks end on
// codepoint boundaries. Malformed sequences are processed one byte at a time;
// Unicode-aware normalizers interpret each malformed byte as U+FFFD.
//
// Output credit never establishes an input boundary. The caller retains any
// unconsumed input and calls state_finalize() only after all input is accepted.
static inline iree_status_t iree_tokenizer_normalizer_state_process(
    iree_tokenizer_normalizer_state_t* state, iree_string_view_t input,
    iree_mutable_string_view_t output, iree_tokenizer_normalizer_flags_t flags,
    iree_host_size_t* IREE_RESTRICT out_consumed,
    iree_host_size_t* IREE_RESTRICT out_written) {
  return state->normalizer->vtable->state_process(state, input, output, flags,
                                                  out_consumed, out_written);
}

// Finalizes the stream, flushing all buffered data.
// Call after all input has been provided via state_process().
// May need multiple calls if output buffer is too small for buffered data.
static inline iree_status_t iree_tokenizer_normalizer_state_finalize(
    iree_tokenizer_normalizer_state_t* state, iree_mutable_string_view_t output,
    iree_host_size_t* IREE_RESTRICT out_written) {
  return state->normalizer->vtable->state_finalize(state, output, out_written);
}

// Returns true if state has pending data that would produce output on
// finalize. Useful for checking if more finalize() calls are needed.
static inline bool iree_tokenizer_normalizer_state_has_pending(
    const iree_tokenizer_normalizer_state_t* state) {
  return state->normalizer->vtable->state_has_pending(state);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_H_
