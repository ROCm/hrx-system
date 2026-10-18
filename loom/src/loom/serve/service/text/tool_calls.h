// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_TOOL_CALLS_H_
#define LOOM_SERVE_SERVICE_TEXT_TOOL_CALLS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One validated function call. Views borrow the completed call table, not the
// source VM, request body, or transport encoding.
typedef struct loom_serve_text_tool_call_t {
  // Stable request-derived identifier shared by every output representation.
  iree_string_view_t id;
  // Decoded function name.
  iree_string_view_t name;
  // Decoded JSON argument document, without an enclosing JSON string escape.
  iree_string_view_t arguments;
} loom_serve_text_tool_call_t;

// Reusable owned output from the complete source-tool transaction. Ranges stay
// valid while storage grows; borrowed views remain valid until reset, append,
// or deinitialize. Moving this value transfers ownership of both allocations.
// Canonical retained conversation history is independent of this output.
typedef struct loom_serve_text_tool_calls_t {
  // Allocator and concatenated owned identifiers, names, and argument bytes.
  iree_string_builder_t storage;
  // Compact string ranges, allocated only when a response produces calls.
  struct loom_serve_text_tool_call_range_t* ranges;
  // Number of complete records; publication waits for the whole transaction.
  iree_host_size_t count;
  // Allocated record capacity, retained across reset for shared CPU scratch.
  iree_host_size_t capacity;
} loom_serve_text_tool_calls_t;

void loom_serve_text_tool_calls_initialize(
    iree_allocator_t host_allocator, loom_serve_text_tool_calls_t* out_calls);
void loom_serve_text_tool_calls_deinitialize(
    loom_serve_text_tool_calls_t* calls);
void loom_serve_text_tool_calls_reset(loom_serve_text_tool_calls_t* calls);

// Copies one decoded call after the source boundary establishes its validity
// and the protocol count bound. Allocation failure can leave scratch bytes but
// never exposes an incomplete record; the transaction owner resets on failure.
iree_status_t loom_serve_text_tool_calls_append(
    loom_serve_text_tool_calls_t* calls, uint64_t request_id,
    iree_string_view_t name, iree_string_view_t arguments);

// Returns the indexed record from a successfully completed transaction.
// index is less than count, established by the consuming iteration.
loom_serve_text_tool_call_t loom_serve_text_tool_calls_at(
    const loom_serve_text_tool_calls_t* calls, iree_host_size_t index);

// Encodes the standard function-call array. Streaming deltas additionally
// carry each call's stable index; IDs and decoded values are unchanged.
iree_status_t loom_serve_text_tool_calls_write_json(
    const loom_serve_text_tool_calls_t* calls, bool indexed,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_TOOL_CALLS_H_
