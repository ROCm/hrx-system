// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_INPUT_CODEC_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_INPUT_CODEC_H_

#include "iree/tokenizer/tokenizer.h"
#include "iree/vm/ref.h"

#ifdef __cplusplus
extern "C" {
#endif

// Managed codec state is independent of its creating module. The immutable
// tokenizer is borrowed and must outlive every codec. Each object is mutated
// exclusively by its source program's serialized CPU owner.
typedef struct loom_serve_input_encoder_t loom_serve_input_encoder_t;
typedef struct loom_serve_input_decoder_t loom_serve_input_decoder_t;

enum loom_serve_input_codec_type_ordinal_e {
  LOOM_SERVE_INPUT_CODEC_TYPE_DECODER = 0,
  LOOM_SERVE_INPUT_CODEC_TYPE_ENCODER = 1,
};

// Static descriptors remain valid after native module/program destruction.
extern const iree_vm_ref_type_table_t loom_serve_input_codec_type_table;

// Common validation for the finite and streaming input capabilities.
iree_status_t loom_serve_input_validate_encode_flags(
    iree_tokenizer_encode_flags_t flags);
iree_status_t loom_serve_input_validate_decode_flags(
    iree_tokenizer_decode_flags_t flags);

// Reads whole little-endian I32 words into caller-owned native staging. The
// input capability establishes the byte range and whole-word size before this
// call. IDs are checked before any tokenizer state is advanced.
iree_status_t loom_serve_input_read_token_ids(const iree_tokenizer_t* tokenizer,
                                              iree_const_byte_span_t bytes,
                                              int32_t* out_tokens);

// Allocates one reference-counted encoder with fixed transform/token staging
// storage. Runtime model workspace may grow through host_allocator and is
// retained across reset. AT_INPUT_START is implicit in create/reset.
iree_status_t loom_serve_input_encoder_create(
    const iree_tokenizer_t* tokenizer, iree_tokenizer_encode_flags_t flags,
    iree_host_size_t transform_capacity, iree_host_size_t token_capacity,
    iree_allocator_t host_allocator, iree_vm_ref_t* out_ref);
iree_status_t loom_serve_input_encoder_reset(
    loom_serve_input_encoder_t* encoder, iree_tokenizer_encode_flags_t flags);

// Output ranges contain whole I32 words. Only the written prefix is modified;
// work is capped by both output credit and the encoder's staging capacity.
iree_status_t loom_serve_input_encoder_feed(loom_serve_input_encoder_t* encoder,
                                            iree_string_view_t input,
                                            iree_byte_span_t output,
                                            iree_host_size_t* out_consumed,
                                            iree_host_size_t* out_written);
iree_status_t loom_serve_input_encoder_finalize(
    loom_serve_input_encoder_t* encoder, iree_byte_span_t output,
    iree_host_size_t* out_written, bool* out_complete);

// Allocates one decoder with reusable state and native token staging. Reset
// restores first-token/leading-space handling for an independent input.
iree_status_t loom_serve_input_decoder_create(
    const iree_tokenizer_t* tokenizer, iree_tokenizer_decode_flags_t flags,
    iree_host_size_t token_capacity, iree_allocator_t host_allocator,
    iree_vm_ref_t* out_ref);
iree_status_t loom_serve_input_decoder_reset(
    loom_serve_input_decoder_t* decoder, iree_tokenizer_decode_flags_t flags);

// Consumes at most token_capacity whole I32 IDs per call. Byte output credit
// may further constrain progress. Feed is invalid after finalization begins.
// Finalization drains remaining bytes with explicit completion; insufficient
// output credit leaves the remaining tail owned by the decoder.
iree_status_t loom_serve_input_decoder_feed(loom_serve_input_decoder_t* decoder,
                                            iree_const_byte_span_t input,
                                            iree_byte_span_t output,
                                            iree_host_size_t* out_consumed,
                                            iree_host_size_t* out_written);
iree_status_t loom_serve_input_decoder_finalize(
    loom_serve_input_decoder_t* decoder, iree_byte_span_t output,
    iree_host_size_t* out_written, bool* out_complete);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_INPUT_CODEC_H_
