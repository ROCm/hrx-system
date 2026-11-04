// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_HTTP_TOKENIZATION_H_
#define LOOM_SERVE_SERVER_HTTP_TOKENIZATION_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// CPU-only raw tokenizer protocol. The caller has already routed the required
// model selector. Requests accept content/string, add_special=false,
// parse_special=true and with_pieces=false; exact raw pieces are not exposed by
// the tokenizer decoder. Detokenization accepts tokens/integer array, preserves
// special tokens and returns content/string. Neither path applies chat policy.
// Both borrow the immutable tokenizer and JSON only for the call, use
// independent encode/decode state, and append a JSON response to output on
// success. Partial output on failure is discarded by the caller.
// Transport admission owns request-body and connection limits. These utilities
// impose no separate token or decoded-text quota, and drain complete output
// through reusable scratch. INVALID_ARGUMENT denotes bad input. Allocation,
// representation, output-sink, and tokenizer failures retain their diagnostics.
// No device/model call occurs.
iree_status_t loom_serve_http_tokenize(const iree_tokenizer_t* tokenizer,
                                       iree_string_view_t body,
                                       iree_string_builder_t* output,
                                       iree_allocator_t host_allocator);
iree_status_t loom_serve_http_detokenize(const iree_tokenizer_t* tokenizer,
                                         iree_string_view_t body,
                                         iree_string_builder_t* output,
                                         iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_HTTP_TOKENIZATION_H_
