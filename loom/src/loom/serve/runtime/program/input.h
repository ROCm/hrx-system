// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_INPUT_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_INPUT_H_

#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Synchronous model-input capabilities, with no model policy or device access.
// The "input" module exports:
//   require(i32 condition, buffer diagnostic)
//   encode(buffer text, i32 flags, i64 capacity)
//       -> (buffer little_endian_ids, i64 count)
//   decode(buffer little_endian_ids, i32 flags, i64 byte_capacity)
//       -> (buffer text, i64 byte_length)
//   lookup(buffer token) -> i32 id
// encode retains at most capacity IDs and returns a zero-padded capacity*4 byte
// buffer plus its live count. Flags use iree_tokenizer_encode_flags_t:
// ADD_SPECIAL_TOKENS (4) enables tokenizer-defined framing, and
// NO_SPECIAL_TOKEN_MATCHING (8) encodes literal special-token spellings as
// ordinary text. Zero selects raw encoding with special-token matching. Every
// call starts a fresh input; AT_INPUT_START is implicit and TRACK_OFFSETS has
// no result channel in this ID-only capability. Other flag bits are rejected.
// A caller requiring the entire text requests one more ID than its limit and
// rejects a returned count above that limit instead of accepting truncation.
// decode consumes every ID and returns a zero-padded byte_capacity buffer with
// its live UTF-8 byte length. Insufficient capacity fails with
// RESOURCE_EXHAUSTED; it never returns partial text. Flags use
// iree_tokenizer_decode_flags_t: SKIP_SPECIAL_TOKENS (1) omits special tokens;
// zero retains them. Other flag bits, partial I32 words, and IDs absent from
// the vocabulary are rejected. Every call starts fresh decoder state, including
// tokenizer-defined leading space handling. Callers pass only live IDs,
// excluding encode's padding. lookup returns -1 for missing vocabulary tokens.
// require propagates INVALID_ARGUMENT with the source diagnostic when condition
// is zero. The environment and optional tokenizer are borrowed for the module
// lifetime. With a NULL tokenizer, encode/decode/lookup and codec creation fail
// with FAILED_PRECONDITION. Input validation and operations on existing codec
// objects remain available. No request bytes are retained by finite calls.
// On failure *out_module is NULL.
//
// Streaming capabilities use managed input.encoder/input.decoder references:
//   encoder_create(i32 flags, i64 transform_capacity, i64 token_capacity)
//       -> input.encoder
//   decoder_create(i32 flags, i64 token_capacity) -> input.decoder
//   encoder_reset(input.encoder state, i32 flags)
//   decoder_reset(input.decoder state, i32 flags)
//   encoder_feed(input.encoder state, buffer input, i64 input_offset,
//                i64 input_length, buffer output, i64 output_offset,
//                i64 output_capacity) -> (i64 consumed, i64 written)
//   decoder_feed(input.decoder state, buffer input, i64 input_offset,
//                i64 input_length, buffer output, i64 output_offset,
//                i64 output_capacity) -> (i64 consumed, i64 written)
//   encoder_finalize(input.encoder state, buffer output, i64 output_offset,
//                    i64 output_capacity) -> (i64 written, i32 complete)
//   decoder_finalize(input.decoder state, buffer output, i64 output_offset,
//                    i64 output_capacity) -> (i64 written, i32 complete)
//
// Offsets and transform_capacity are bytes. Encoder input/consumed units are
// bytes and output/written units are I32 IDs; decoder units are the reverse.
// Source buffers need no alignment. Only the written prefix changes. Each call
// handles at most token_capacity IDs; this positive creation-time capacity owns
// reusable native staging and does not limit total stream length. Transform
// capacity follows the runtime tokenizer's power-of-two storage contract.
// Source chooses input windows, output credit, normalization, and packet
// policy. Streaming encoding follows the runtime's partial-input byte contract;
// callers retain unconsumed input and supply complete UTF-8 through their
// ingress policy. Flags match finite calls, and reset restores independent
// first-input behavior. A call can accept input without producing output,
// including with zero output credit; the returned consumed count always
// advances the caller's input cursor.
//
// Finalize begins only after all input is accepted. Output exhaustion retains
// pending work; repeated calls drain it until complete is true. Zero output
// credit is valid. Feed requires reset after finalization starts. A decoder may
// require more than one output byte to emit a UTF-8 scalar or vocabulary piece.
// Codec state owns its storage and may outlive the module/program; its borrowed
// immutable tokenizer must outlive every codec. Calls on one object are
// serialized by the source owner. Creating this module installs the static
// input type family in the environment before source compilation, including
// cold preparation.
iree_status_t loom_serve_input_module_create(iree_vm_environment_t* environment,
                                             const iree_tokenizer_t* tokenizer,
                                             iree_allocator_t host_allocator,
                                             iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_INPUT_H_
