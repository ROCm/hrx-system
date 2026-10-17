// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_INPUT_H_
#define LOOM_SERVE_SERVICE_SPEECH_INPUT_H_

#include "iree/base/api.h"
#include "iree/base/string_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_speech_input_t loom_serve_speech_input_t;

// Immutable complete-scalar prefix offered to one source CPU task. Appending
// writes disjoint free bytes; the snapshot remains valid until CPU retirement.
typedef struct loom_serve_speech_input_block_t {
  // Absolute position of the first offered UTF-8 byte.
  uint64_t position;
  // Complete UTF-8 bytes in spans, bounded by the model's text capacity.
  iree_host_size_t byte_count;
  // Borrowed spans in input order, permitting one ring wrap.
  iree_const_byte_span_t spans[2];
  // No text follows this window. An empty final window is meaningful.
  bool complete;
} loom_serve_speech_input_block_t;

// Creates one serving-owner reference. For complete input, transfers the
// validated request builder's allocation without copying; otherwise allocates
// a fixed ring and leaves the empty builder unchanged. Failure leaves the
// builder untouched. Geometry comes from checked model metadata: capacity >= 4
// bytes, and limit is a positive total request byte bound. Neither snapshots
// nor source VM buffers retain a mutable alias to this object.
iree_status_t loom_serve_speech_input_allocate(
    iree_host_size_t capacity, uint64_t limit, iree_string_builder_t* text,
    bool complete, iree_allocator_t host_allocator,
    loom_serve_speech_input_t** out_input);

// All ownership and mutation operations run on the serialized serving owner.
// A service reference outlives every worker borrowing one of its snapshots.
void loom_serve_speech_input_retain(loom_serve_speech_input_t* input);
void loom_serve_speech_input_release(loom_serve_speech_input_t* input);

// Copies at most available byte credit. Only that prefix is UTF-8 validated;
// up to three bytes of an incomplete scalar may remain buffered. Invalid UTF-8,
// exceeding the total request bound, or writing after EOF accepts nothing and
// leaves all earlier state intact. Zero accepted bytes is ordinary pressure.
iree_status_t loom_serve_speech_input_write(loom_serve_speech_input_t* input,
                                            iree_const_byte_span_t text,
                                            iree_host_size_t* out_byte_count);
iree_host_size_t loom_serve_speech_input_credit(
    const loom_serve_speech_input_t* input);

// Publishes EOF only after a complete scalar. Repeated completion is harmless;
// incomplete UTF-8 fails without changing EOF or the retained input prefix.
iree_status_t loom_serve_speech_input_finish(loom_serve_speech_input_t* input);

// Captures at most the declared text capacity, ending on a scalar boundary.
// An empty nonfinal snapshot cannot wake a source waiting for more input.
loom_serve_speech_input_block_t loom_serve_speech_input_peek(
    const loom_serve_speech_input_t* input);

// Acknowledges a source-validated complete-scalar prefix after CPU retirement,
// independently of numerical completion. The count fits the captured snapshot.
void loom_serve_speech_input_consume(loom_serve_speech_input_t* input,
                                     iree_host_size_t byte_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_INPUT_H_
