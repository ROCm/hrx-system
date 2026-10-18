// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_CAPTURE_H_
#define LOOM_SERVE_SERVER_REALTIME_CAPTURE_H_

#include "iree/task/executor.h"
#include "loom/serve/runtime/host/wake.h"
#include "loom/serve/transport/websocket/connection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_realtime_capture_t loom_serve_realtime_capture_t;

// Caller-owned loan record, stable until release on the serialized owner.
// A numerical worker may borrow samples but never inspects the record links.
typedef struct loom_serve_realtime_capture_span_t {
  // Owner-only link among outstanding loans into the capture ring.
  struct loom_serve_realtime_capture_span_t* next;
  // Retained capture, or NULL when this record owns no samples.
  loom_serve_realtime_capture_t* capture;
  // First original-rate frame retained by this loan.
  uint64_t position;
  // Immutable contiguous PCM16LE mono samples at 24 kHz.
  iree_const_byte_span_t samples;
} loom_serve_realtime_capture_span_t;

// Owns original-rate microphone storage and one finite decoding worker.
// capacity is a positive count of PCM16 mono frames, independent of message
// partitioning. The HTTP claim keeps socket alive through all accepted work.
iree_status_t loom_serve_realtime_capture_allocate(
    iree_task_executor_t* executor, loom_serve_websocket_t* socket,
    iree_host_size_t capacity, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator,
    loom_serve_realtime_capture_t** out_capture);

// Success transfers a receipt and its escaped JSON audio view. Decode and PCM
// validation run on shared workers. The receipt stays charged to the transport
// until its final PCM block publishes; failure leaves it with the caller.
iree_status_t loom_serve_realtime_capture_append(
    loom_serve_realtime_capture_t* capture,
    loom_serve_websocket_message_t* message, iree_string_view_t audio);
// Seals the ordered append frontier. An empty or already sealed capture fails.
iree_status_t loom_serve_realtime_capture_finish(
    loom_serve_realtime_capture_t* capture);

// Publishes released decode work and schedules at most one finite block.
// Full retained storage is normal backpressure; no worker waits for a reader.
void loom_serve_realtime_capture_advance(loom_serve_realtime_capture_t* capture,
                                         bool* out_progress);
iree_status_t loom_serve_realtime_capture_status(
    const loom_serve_realtime_capture_t* capture);
// Published input endpoint, in original 24 kHz frames from this capture's
// start.
uint64_t loom_serve_realtime_capture_end(
    const loom_serve_realtime_capture_t* capture);
// Every sealed append has published, independently of outstanding PCM loans.
bool loom_serve_realtime_capture_is_complete(
    const loom_serve_realtime_capture_t* capture);

// Borrows up to maximum frames at a retained position before capture_end.
// The caller supplies an unused span record. Ring wrap can shorten the loan.
void loom_serve_realtime_capture_borrow(
    loom_serve_realtime_capture_t* capture, uint64_t position,
    iree_host_size_t maximum, loom_serve_realtime_capture_span_t* span);
// Releases a loan on the serialized owner after its final worker retires.
// The void* signature permits direct use as a PCM input release callback.
void loom_serve_realtime_capture_release(void* span);
// Advances the caller's retention frontier within published input. Outstanding
// loans continue to pin their samples even when they precede this frontier.
void loom_serve_realtime_capture_discard_before(
    loom_serve_realtime_capture_t* capture, uint64_t position);

// Ends publication interest. Accepted workers and loans still retain storage.
void loom_serve_realtime_capture_close(loom_serve_realtime_capture_t* capture);
bool loom_serve_realtime_capture_pending(
    const loom_serve_realtime_capture_t* capture);
// Requires no receipts, workers, or loans. NULL is accepted.
void loom_serve_realtime_capture_free(loom_serve_realtime_capture_t* capture);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_CAPTURE_H_
