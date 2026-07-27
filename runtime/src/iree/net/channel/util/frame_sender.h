// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Frame sender: scatter-gather send utility for channels.
//
// iree_net_frame_sender_t is a formatting utility that assembles scatter-gather
// lists from headers and payloads, retains copied storage, and tracks in-flight
// sends. It sits below the channel layer and above the carrier layer.
//
// Channels embed a frame_sender and use it for all send operations. The channel
// handles protocol semantics (frame types, stream IDs, etc.) while the
// frame_sender handles the mechanics of constructing and submitting sends.
//
// ## Ownership model
//
// The frame_sender does NOT own the optional storage pool or submit_fn
// resources; they must outlive the sender. The sender owns in-flight send
// contexts until completion.
//
// ## Callback contract
//
// Matches the carrier pattern:
//   - If send()/flush() returns non-OK: callback is NEVER called
//   - If send()/flush() returns OK: callback WILL be called (errors via
//   callback)
//
// ## Send modes
//
// Three modes are supported:
//
// 1. **send()**: Scatter-gather send with retained header, payload zero-copy.
//    Typical headers are copied to send-context storage; unusually large
//    headers use the optional storage pool or heap storage. Payload spans are
//    passed directly. Used for all sends with application payloads (queue
//    command payloads are typically 64KB-512KB; control DATA payloads can be
//    many megabytes).
//
// 2. **send_copy()**: Single-buffer send with header and payload copied to
//    retained sender storage before submission. Used for small messages whose
//    source storage cannot outlive the asynchronous send.
//
// 3. **queue() + flush()**: Batched send for self-delimiting frames.
//    Encoded frame bytes are copied to a batch buffer and concatenated into
//    one endpoint message on flush(). The receiving protocol must be able to
//    recover every frame boundary within that message. This is not valid for
//    protocols where the endpoint message length defines the frame length.
//
// ## Backpressure
//
// Frame sender propagates RESOURCE_EXHAUSTED from the carrier or configured
// storage pool. It does NOT manage backpressure policy; the embedding channel
// interprets RESOURCE_EXHAUSTED and handles drain callbacks.
//
// ## Thread safety
//
// **send()** is thread-safe and may be called from any thread. This is the
// primary API for application threads sending data. The underlying carrier
// is assumed to be thread-safe for sends (io_uring, RDMA, etc.). If a carrier
// implementation is not naturally thread-safe, it handles serialization
// internally - this is not the channel's or frame_sender's concern.
//
// **queue() and flush()** are NOT thread-safe and must only be called from
// a single thread (the proactor thread). These optimize protocols where the
// proactor assembles multiple self-delimiting frames.
//
// **Mixing send() with queue()/flush()**: If the proactor thread uses both
// APIs and needs ordering between them, it must call flush() explicitly
// before send(). send() does NOT auto-flush to maintain thread-safety.
// Off-thread callers only use send() and have no batched data to flush.
//
// **has_pending() and pending_count()** use atomic operations and are safe
// to call from any thread.
//
// **handle_completion()** may be called from the carrier's completion thread,
// which may differ from the submission thread. It uses atomic operations for
// the in-flight count and releases retained pool or heap storage.

#ifndef IREE_NET_CHANNEL_UTIL_FRAME_SENDER_H_
#define IREE_NET_CHANNEL_UTIL_FRAME_SENDER_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Frame sender
//===----------------------------------------------------------------------===//

typedef struct iree_net_frame_sender_t iree_net_frame_sender_t;
typedef struct iree_net_frame_send_context_t iree_net_frame_send_context_t;

// Submit callback for sending data through the transport.
//
// Called by the frame_sender to submit scatter-gather data for async delivery.
// Must follow the same completion semantics as iree_net_carrier_send():
//   - Returns OK: a completion callback will fire with params->user_data.
//   - Returns non-OK: no completion fires.
//
// The params->user_data is an opaque pointer to the frame_send_context_t and
// must be passed through to the underlying carrier's completion callback.
//
// For transports without a mux layer (loopback, shm), this calls
// iree_net_carrier_send() directly. For transports with mux/framing layers
// (TCP), this routes through the per-stream message endpoint which adds
// transport headers before reaching the carrier.
typedef iree_status_t (*iree_net_frame_send_submit_fn_t)(
    void* user_data, iree_async_span_list_t data, uint64_t send_user_data);

// Maximum scatter-gather spans per send operation (header + payloads).
// Validated against carrier->max_iov at send time.
#define IREE_NET_FRAME_SENDER_MAX_SPANS 8

// Maximum copied frame size retained inline in the send context.
#define IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY 2048

// Completion callback fired when a send completes (success or failure).
// Channel uses this for resource cleanup and drain signaling.
//
// |callback_user_data| is from the callback struct (set at initialize).
// |operation_user_data| is from the send()/flush() call (identifies the op).
// |status| is OK on success, or an error status on failure.
typedef void (*iree_net_frame_send_complete_fn_t)(void* callback_user_data,
                                                  uint64_t operation_user_data,
                                                  iree_status_t status);

// Callback struct for send completions.
typedef struct iree_net_frame_send_complete_callback_t {
  iree_net_frame_send_complete_fn_t fn;
  void* user_data;
} iree_net_frame_send_complete_callback_t;

// Acquires one reusable send context from a caller-owned pool.
typedef iree_status_t (*iree_net_frame_sender_context_acquire_fn_t)(
    void* user_data, iree_net_frame_send_context_t** out_context);

// Releases one send context back to a caller-owned pool.
typedef void (*iree_net_frame_sender_context_release_fn_t)(
    void* user_data, iree_net_frame_send_context_t* context);

// Caller-owned send context pool callbacks.
typedef struct iree_net_frame_sender_context_pool_t {
  // Acquires one context or returns RESOURCE_EXHAUSTED for backpressure.
  iree_net_frame_sender_context_acquire_fn_t acquire;

  // Releases one previously acquired context.
  iree_net_frame_sender_context_release_fn_t release;

  // User data passed to acquire/release.
  void* user_data;
} iree_net_frame_sender_context_pool_t;

// Per-send context owned by the frame sender until completion fires.
//
// Layout keeps hot metadata early and retains typical copied frames inline so
// stack-built control packets do not need a separate pool lease.
struct iree_net_frame_send_context_t {
  // Sender that owns this context and receives completion recycling.
  iree_net_frame_sender_t* sender;

  // Header or copied-frame buffer lease held until send completion.
  iree_async_buffer_lease_t buffer_lease;

  // Heap-backed copied frame storage held until send completion.
  void* heap_frame;

  // Caller-provided value echoed to the sender completion callback.
  uint64_t operation_user_data;

  // Number of valid entries in |spans|.
  iree_host_size_t span_count;

  // Inline storage for small stack-built frames.
  uint8_t inline_frame[IREE_NET_FRAME_SENDER_INLINE_FRAME_CAPACITY];

  // Scatter-gather spans submitted to the carrier.
  iree_async_span_t spans[IREE_NET_FRAME_SENDER_MAX_SPANS];

  // Intrusive free-list pointer used while the context is idle.
  iree_atomic_slist_intrusive_ptr_t slist_next;
};

// Embeddable sender structure.
// Use initialize/deinitialize pattern - do not allocate separately.
struct iree_net_frame_sender_t {
  // Send submit callback and context. The callback submits scatter-gather data
  // for async delivery. For carrier-direct sends, this wraps carrier_send().
  // For endpoint-routed sends, this goes through the message endpoint which
  // may add transport headers (e.g., TCP mux stream_id).
  iree_net_frame_send_submit_fn_t submit_fn;
  void* submit_fn_user_data;

  // Maximum scatter-gather spans per send. Validated at send time.
  // Accounts for any overhead added by the submit path (e.g., TCP mux adds
  // one span for its stream header, so max_send_spans = carrier->max_iov - 1).
  iree_host_size_t max_send_spans;

  // Optional pool used for unusually large frame headers, copied frames, and
  // batches. When NULL, retained sends use heap storage and batching is
  // unavailable. Not owned.
  iree_async_buffer_pool_t* storage_pool;

  // Completion callback (provided by channel).
  iree_net_frame_send_complete_callback_t callback;

  // Batch buffer for self-delimiting frames (queue/flush mode).
  iree_async_buffer_lease_t batch_lease;

  // True when |batch_lease| is live and holds queued bytes.
  bool has_batch_lease;

  // Number of bytes currently used in |batch_lease|.
  iree_host_size_t batch_used;

  // Number of send operations submitted without a matching completion.
  iree_atomic_int32_t sends_in_flight;

  // Free list of reusable send contexts in block-pool mode.
  iree_atomic_slist_t context_free_list;

  // Blocks backing the block-pool mode; freed during deinitialize.
  iree_atomic_slist_t context_block_list;

  // Optional caller-owned context pool; empty uses the internal block pool.
  iree_net_frame_sender_context_pool_t context_pool;

  // Allocator used when growing the internal block pool.
  iree_allocator_t context_allocator;

  // Host allocator used for heap-backed retained storage.
  iree_allocator_t host_allocator;
};

// Initializes a frame sender in pre-allocated memory.
//
// |submit_fn| is called to submit send operations. |submit_fn_user_data| is
// passed as the first argument to |submit_fn| on each call.
//
// |max_send_spans| is the maximum number of scatter-gather spans per send,
// accounting for any overhead added by the submit path. For carrier-direct
// sends, this is carrier->max_iov. For endpoint-routed sends that add
// transport headers (e.g., TCP mux), subtract the header span count.
//
// |storage_pool| optionally provides retained send storage and must outlive the
// sender. When NULL, large sends use heap storage and queue() is unavailable.
// |callback| is fired for each completed send operation.
// |context_allocator| is used to grow the internal send context pool.
// |host_allocator| is used for other dynamic allocations.
iree_status_t iree_net_frame_sender_initialize(
    iree_net_frame_sender_t* sender, iree_net_frame_send_submit_fn_t submit_fn,
    void* submit_fn_user_data, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* storage_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_allocator_t context_allocator, iree_allocator_t host_allocator);

// Initializes a frame sender using an external allocation-free context pool.
//
// The |context_pool| must return storage for iree_net_frame_send_context_t from
// a caller-owned pool and recycle it on release. It may return
// RESOURCE_EXHAUSTED to express intentional backpressure, but must not call
// general-purpose malloc/free per send in the steady state.
//
// |storage_pool| has the same optional retained-storage contract as
// iree_net_frame_sender_initialize().
iree_status_t iree_net_frame_sender_initialize_with_context_pool(
    iree_net_frame_sender_t* sender, iree_net_frame_send_submit_fn_t submit_fn,
    void* submit_fn_user_data, iree_host_size_t max_send_spans,
    iree_async_buffer_pool_t* storage_pool,
    iree_net_frame_send_complete_callback_t callback,
    iree_net_frame_sender_context_pool_t context_pool,
    iree_allocator_t host_allocator);

// Deinitializes a frame sender and discards any queued batch.
// Asserts that no sends are in flight. Caller must drain completions first.
void iree_net_frame_sender_deinitialize(iree_net_frame_sender_t* sender);

// Sends a frame with scatter-gather. THREAD-SAFE.
//
// This is the primary send API and may be called from any thread. The
// underlying carrier and configured storage pool are assumed to be thread-safe.
//
// |header| is copied into sender-owned storage (typically inline in the send
// context).
// |payload| is a span list sent zero-copy (may be hundreds of KB or MBs).
// |operation_user_data| is passed to the completion callback.
//
// Does NOT auto-flush batched frames. If the proactor thread is mixing
// send() with queue()/flush() and needs ordering, call flush() first.
//
// Returns OK if the operation was submitted (callback will fire).
// Returns RESOURCE_EXHAUSTED if carrier is backpressured or pool is empty.
// Returns OUT_OF_RANGE if 1 + payload.count > min(MAX_SPANS, carrier->max_iov).
//
// On non-OK return, the callback is NOT called.
iree_status_t iree_net_frame_sender_send(iree_net_frame_sender_t* sender,
                                         iree_const_byte_span_t header,
                                         iree_async_span_list_t payload,
                                         uint64_t operation_user_data);

// Sends a frame with header and payload copied into sender-owned storage.
// THREAD-SAFE.
//
// Unlike send(), all payload bytes are copied before this function returns.
// Callers may pass stack or otherwise short-lived payload spans. This is
// intended only for small control messages; large payloads should use send()
// and keep their backing storage alive until completion.
//
// Returns OK if the operation was submitted (callback will fire). The sender
// may allocate heap storage when the combined header+payload exceeds inline
// storage and the configured storage pool's buffer capacity.
// Returns FAILED_PRECONDITION if any payload span is not CPU-accessible.
//
// On non-OK return, the callback is NOT called.
iree_status_t iree_net_frame_sender_send_copy(iree_net_frame_sender_t* sender,
                                              iree_const_byte_span_t header,
                                              iree_async_span_list_t payload,
                                              uint64_t operation_user_data);

// Queues one self-delimiting frame for batched send. NOT THREAD-SAFE.
//
// Must only be called from the proactor thread. This is an optimization for
// protocols that can decode multiple concatenated frames from one endpoint
// message.
//
// |frame| (header + payload combined) is copied into the batch buffer. The
// frame encoding must include enough information for the receiver to determine
// its length without consulting the enclosing endpoint message length. Call
// flush() to send all queued bytes as one endpoint message.
//
// Returns OK if the frame was queued successfully.
// Returns INVALID_ARGUMENT if |frame| is empty.
// Returns FAILED_PRECONDITION if no storage pool was configured.
// Returns RESOURCE_EXHAUSTED if the batch buffer is full (call flush first).
iree_status_t iree_net_frame_sender_queue(iree_net_frame_sender_t* sender,
                                          iree_const_byte_span_t frame);

// Flushes all queued frames as a single send operation. NOT THREAD-SAFE.
//
// Must only be called from the proactor thread.
//
// |operation_user_data| is passed to the completion callback for this batch.
//
// Returns OK if the operation was submitted or if nothing was queued.
// Returns RESOURCE_EXHAUSTED if carrier is backpressured.
//
// On RESOURCE_EXHAUSTED, the batch data is preserved. The caller owns its
// disposition and must either retry flush() without queuing the frames again
// or explicitly discard the batch.
//
// If nothing is queued, returns OK immediately without firing a callback.
iree_status_t iree_net_frame_sender_flush(iree_net_frame_sender_t* sender,
                                          uint64_t operation_user_data);

// Discards all queued frame bytes without sending them. NOT THREAD-SAFE.
//
// Releases the batch buffer and returns the sender to its empty state. This is
// a no-op when no frames are queued.
void iree_net_frame_sender_discard_batch(iree_net_frame_sender_t* sender);

// Returns the number of bytes currently queued for batching.
iree_host_size_t iree_net_frame_sender_queued_bytes(
    const iree_net_frame_sender_t* sender);

// Returns true if any sends are currently in flight.
bool iree_net_frame_sender_has_pending(const iree_net_frame_sender_t* sender);

// Returns the number of sends currently in flight.
int32_t iree_net_frame_sender_pending_count(
    const iree_net_frame_sender_t* sender);

// Handles completion of a send operation.
// Called by the carrier's completion callback dispatcher.
//
// |context| is the send context passed as user_data to carrier->send().
// |status| is OK on success, or an error status on failure.
//
// This function:
//   1. Releases the buffer lease back to the pool
//   2. Decrements sends_in_flight
//   3. Fires the user callback
//   4. Returns the context to the sender pool
void iree_net_frame_sender_handle_completion(
    iree_net_frame_send_context_t* context, iree_status_t status);

// Generic carrier completion callback that dispatches SEND completions to a
// frame_sender.
//
// Use as the iree_net_carrier_callback_t.fn when all sends through a carrier
// use frame_sender. The callback extracts the frame_send_context_t from
// |operation_user_data| and calls iree_net_frame_sender_handle_completion().
//
// IREE_NET_CARRIER_COMPLETION_SEND_READY is a transport readiness edge and is
// ignored here. Other completion kinds require a carrier-level router rather
// than a frame_sender-only callback.
//
// This is the standard carrier callback for connections that use frame_sender
// in their channel layers.
void iree_net_frame_sender_dispatch_carrier_completion(
    void* callback_user_data, iree_net_carrier_completion_kind_t kind,
    uint64_t operation_user_data, iree_status_t status,
    iree_host_size_t bytes_transferred, iree_async_buffer_lease_t* recv_lease);

//===----------------------------------------------------------------------===//
// Carrier-direct submit helper
//===----------------------------------------------------------------------===//

// Submit function that calls iree_net_carrier_send() directly.
//
// Use as the submit_fn when the frame_sender should bypass any endpoint/mux
// layers and send to the carrier without modification. Typical for transports
// that don't multiplex (loopback, shm) or where the carrier already includes
// any necessary framing.
//
// |user_data| must be an iree_net_carrier_t*.
iree_status_t iree_net_frame_sender_carrier_submit(void* user_data,
                                                   iree_async_span_list_t data,
                                                   uint64_t send_user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_UTIL_FRAME_SENDER_H_
