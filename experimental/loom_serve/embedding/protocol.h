// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_EMBEDDING_PROTOCOL_H_
#define EXPERIMENTAL_LOOM_SERVE_EMBEDDING_PROTOCOL_H_

#include "iree/base/api.h"
#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_serve_embedding_encoding_e {
  LOOM_SERVE_EMBEDDING_ENCODING_FLOAT = 0,
  LOOM_SERVE_EMBEDDING_ENCODING_BASE64,
} loom_serve_embedding_encoding_t;

typedef struct loom_serve_embedding_request_t {
  // Owned read-only source input: i64 [dimensions, count], then count records
  // [kind, absolute byte offset, byte length]. Kind 0 is UTF-8 and kind 1 is
  // nonnegative LE i32 IDs. Payloads are aligned to eight bytes.
  iree_vm_buffer_t* input;
  // Number of ordered input sequences and expected result rows.
  iree_host_size_t count;
  // Requested output dimensions, or zero for the source-defined default.
  uint64_t dimensions;
  // Wire encoding; base64 contains little-endian IEEE754 F32 elements.
  loom_serve_embedding_encoding_t encoding;
} loom_serve_embedding_request_t;

// Parses a complete /v1/embeddings request. The router has selected model.
// Supports string, string[], integer[] and integer[][] input, positive optional
// dimensions, float/base64 and optional user metadata. Rejects mixed/empty
// outer arrays, duplicate/unknown fields and invalid UTF-8/IDs. Empty strings
// and token lists are passed to source policy. No chat framing is applied.
// Failure releases partial ownership and leaves a zeroed request.
iree_status_t loom_serve_embedding_request_initialize(
    iree_string_view_t body, loom_serve_embedding_request_t* out_request,
    iree_allocator_t allocator);
void loom_serve_embedding_request_deinitialize(
    loom_serve_embedding_request_t* request);

// Appends one JSON response in original input order. Rows are borrowed LE F32
// byte spans from the model's checked compact result plan. Values must be
// finite, and an explicit dimensions request must match every row; violations
// report DATA_LOSS, never native truncation or normalization. Partial output
// on failure is discarded by the caller.
iree_status_t loom_serve_embedding_response(
    const loom_serve_embedding_request_t* request, iree_string_view_t model,
    const iree_const_byte_span_t* rows, uint64_t input_tokens,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_EMBEDDING_PROTOCOL_H_
