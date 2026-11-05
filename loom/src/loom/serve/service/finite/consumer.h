// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_CONSUMER_H_
#define LOOM_SERVE_SERVICE_FINITE_CONSUMER_H_

#include "loom/serve/service/finite/protocol.h"
#include "loom/serve/service/finite/result.h"

// One accepted request's interest, input ownership, and output destination.
// Owner operations never run concurrently. encode alone runs on a CPU loan,
// disjoint from interest queries; end follows the final loan's release.
typedef struct loom_serve_finite_consumer_vtable_t {
  // Owner-only interest query. Once false, it never becomes true again.
  bool (*is_open)(void* self);
  // Optional immediate peer cancellation, without consuming the receipt or
  // invalidating input borrowed by a CPU task. end still follows retirement.
  // Device-only consumers need no peer action and may leave this null.
  void (*cancel)(void* self);
  // Optional owner publication immediately after device submission. The result
  // is borrowed for this call; retaining it permits dependency-based device
  // composition without a host wait. Publication does not imply retirement.
  void (*publish)(void* self, loom_serve_finite_result_t* result,
                  iree_host_size_t request_index, iree_host_size_t row_count);
  // Optional host output on a finite CPU loan. Its presence requests one shared
  // cohort download; device-only consumers leave it null. No row borrow escapes
  // this call. Input and output storage stay live until end.
  iree_status_t (*encode)(void* self,
                          const loom_serve_finite_protocol_t* protocol,
                          const loom_serve_finite_request_t* request,
                          iree_string_view_t name,
                          const iree_const_byte_span_t* rows,
                          uint64_t input_tokens, iree_string_builder_t* output);
  // Consumes the receipt after source/CPU borrowers retire. status is borrowed;
  // CANCELLED drops interest, rejected identifies a request-scoped rejection.
  // Output is borrowed reusable storage, containing successful encoding or
  // available for an error response. out_delivered reports
  // acceptance by a still-interested consumer, not physical I/O completion.
  // Delivery failures are request-local and never fail the model's peers.
  iree_status_t (*end)(void* self, const loom_serve_finite_protocol_t* protocol,
                       const iree_status_t status, bool rejected,
                       iree_string_builder_t* output, bool* out_delivered);
} loom_serve_finite_consumer_vtable_t;

typedef struct loom_serve_finite_consumer_t {
  // Owned receipt preserving input bytes until end, even after lost interest.
  void* self;
  // Immutable process-lifetime operations; nonnull for an accepted receipt.
  const loom_serve_finite_consumer_vtable_t* vtable;
} loom_serve_finite_consumer_t;

#endif  // LOOM_SERVE_SERVICE_FINITE_CONSUMER_H_
