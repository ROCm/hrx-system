// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Frontier-ordered queue messages over a borrowed message endpoint.
//
// A queue channel adds an 8-byte typed envelope and zero or more causal
// frontier entries to complete endpoint messages. COMMAND messages carry
// application-defined queue operations. ADVANCE messages report completed
// causal positions and may carry application-defined completion data.
//
// ## Wire format
//
// All integers are little-endian. Each message begins with:
//
//   byte 0:    version (currently 1)
//   byte 1:    type (COMMAND or ADVANCE)
//   byte 2:    wait frontier entry count
//   byte 3:    signal frontier entry count
//   bytes 4-7: queue ID, or IREE_NET_QUEUE_ID_NONE
//
// Wait entries immediately follow the header, then signal entries, then the
// opaque application payload. Each frontier entry is an explicit 16-byte
// (axis, epoch) pair. Received entries remain borrowed wire views and are read
// with unaligned accessors; they are never cast to host frontier structs.
//
// ## Ownership
//
// The channel borrows its endpoint. Allocate it before protocol handoff, call
// attach before endpoint activation or from the final bootstrap callback, and
// free it only after endpoint deactivation has completed. The channel owns no
// endpoint lifecycle, send tracker, or completion storage.
//
// A send builder runs synchronously after endpoint storage is admitted and
// before publication. This allows a queue implementation to assign an epoch,
// install completion waiters, and encode transient command data atomically
// with transport admission. Borrowed trailing spans remain caller-owned until
// the native endpoint completion callback fires.

#ifndef IREE_NET_CHANNEL_QUEUE_QUEUE_CHANNEL_H_
#define IREE_NET_CHANNEL_QUEUE_QUEUE_CHANNEL_H_

#include "iree/async/api.h"
#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_NET_QUEUE_MESSAGE_VERSION 1u
#define IREE_NET_QUEUE_MESSAGE_HEADER_SIZE 8u
#define IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE 16u

// Queue ID used by protocol operations that are not directed at one queue.
#define IREE_NET_QUEUE_ID_NONE UINT32_MAX

// Queue message type values encoded on the wire.
typedef enum iree_net_queue_message_type_e {
  // Frontier-ordered application command.
  IREE_NET_QUEUE_MESSAGE_TYPE_COMMAND = 0x01,
  // Completed frontier notification.
  IREE_NET_QUEUE_MESSAGE_TYPE_ADVANCE = 0x02,
} iree_net_queue_message_type_t;

// Borrowed encoded frontier entries from a structurally validated message.
typedef struct iree_net_queue_frontier_view_t {
  // Complete encoded entries in their original wire storage.
  iree_const_byte_span_t encoded_entries;
  // Number of fixed-size entries in |encoded_entries|.
  uint8_t count;
} iree_net_queue_frontier_view_t;

// Returns an aligned host value for frontier entry |index|.
//
// |frontier| must come from a queue channel callback and |index| must be less
// than |frontier->count|.
static inline iree_async_frontier_entry_t iree_net_queue_frontier_view_get(
    const iree_net_queue_frontier_view_t* frontier, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(index < frontier->count);
  const uint8_t* entry = frontier->encoded_entries.data +
                         index * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  return (iree_async_frontier_entry_t){
      /*.axis=*/iree_unaligned_load_le_u64(entry),
      /*.epoch=*/iree_unaligned_load_le_u64(entry + 8),
  };
}

// Writable encoded frontier entries provided to a send builder.
typedef struct iree_net_queue_frontier_builder_t {
  // Exact writable entry storage owned by the endpoint send.
  iree_byte_span_t encoded_entries;
  // Number of fixed-size entries in |encoded_entries|.
  uint8_t count;
} iree_net_queue_frontier_builder_t;

// Encodes one aligned host frontier entry at |index|.
//
// |frontier| must come from the active send builder callback and |index| must
// be less than |frontier->count|.
static inline void iree_net_queue_frontier_builder_set(
    const iree_net_queue_frontier_builder_t* frontier, iree_host_size_t index,
    iree_async_frontier_entry_t entry) {
  IREE_ASSERT_ARGUMENT(frontier);
  IREE_ASSERT(index < frontier->count);
  uint8_t* target = frontier->encoded_entries.data +
                    index * IREE_NET_QUEUE_FRONTIER_ENTRY_SIZE;
  iree_unaligned_store_le_u64(target, entry.axis);
  iree_unaligned_store_le_u64(target + 8, entry.epoch);
}

// Writable regions composing the generated portion of one queue message.
//
// Every frontier entry and generated payload byte must be initialized before
// the builder callback returns. Frontier entries must have nonzero epochs and
// strictly increasing unique axes. The target storage must not be retained.
typedef struct iree_net_queue_message_builder_t {
  // Wait frontier entries encoded before signal entries.
  iree_net_queue_frontier_builder_t wait_frontier;
  // Signal frontier entries encoded after wait entries.
  iree_net_queue_frontier_builder_t signal_frontier;
  // Exact application payload prefix encoded after both frontiers.
  iree_byte_span_t generated_payload;
} iree_net_queue_message_builder_t;

// Builds transient queue message data after endpoint send admission.
//
// Returning a non-OK status completes the accepted send with that status and
// zero transferred bytes. Any application state transferred by the builder
// remains owned until that terminal completion. The callback runs synchronously
// at most once before the public send function returns and without transport
// locks held.
typedef iree_status_t(IREE_API_PTR* iree_net_queue_message_build_fn_t)(
    void* user_data, const iree_net_queue_message_builder_t* builder);

// Parameters shared by COMMAND and ADVANCE sends.
typedef struct iree_net_queue_channel_send_params_t {
  // Number of wait frontier entries generated by |build|.
  uint8_t wait_frontier_count;
  // Number of signal frontier entries generated by |build|.
  uint8_t signal_frontier_count;
  // Exact number of transient application payload bytes generated by |build|.
  iree_host_size_t generated_payload_length;
  // Synchronous builder for frontiers and transient application payload.
  iree_net_queue_message_build_fn_t build;
  // Opaque value passed to |build|.
  void* build_user_data;
  // Stable application payload spans following the generated payload.
  iree_async_span_list_t payload;
  // Required callback invoked when the complete message send terminates.
  iree_net_send_completion_callback_t completion_callback;
} iree_net_queue_channel_send_params_t;

typedef struct iree_net_queue_channel_t iree_net_queue_channel_t;

// Handles one received COMMAND message.
//
// Frontier views and |payload| borrow the endpoint-provided storage. To retain
// any view after this callback, move |lease| by copying it and clearing the
// callback's lease value. Returning a non-OK status terminates the endpoint and
// delivers that status through |on_error|.
typedef iree_status_t(IREE_API_PTR* iree_net_queue_channel_command_fn_t)(
    void* user_data, uint32_t queue_id,
    const iree_net_queue_frontier_view_t* wait_frontier,
    const iree_net_queue_frontier_view_t* signal_frontier,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

// Handles one received ADVANCE message.
//
// ADVANCE messages always carry a nonempty signal frontier and may carry an
// opaque application payload. Borrowed storage follows the same lease contract
// as COMMAND messages.
typedef iree_status_t(IREE_API_PTR* iree_net_queue_channel_advance_fn_t)(
    void* user_data, const iree_net_queue_frontier_view_t* signal_frontier,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

// Handles the endpoint's one terminal error.
//
// Malformed input, application callback failure, and transport failure all
// converge here. Status ownership transfers to the callback.
typedef void(IREE_API_PTR* iree_net_queue_channel_error_fn_t)(
    void* user_data, iree_status_t status);

// Application callbacks installed on a queue channel.
typedef struct iree_net_queue_channel_callbacks_t {
  // Required callback for COMMAND messages.
  iree_net_queue_channel_command_fn_t on_command;
  // Required callback for ADVANCE messages.
  iree_net_queue_channel_advance_fn_t on_advance;
  // Required callback for the terminal endpoint error.
  iree_net_queue_channel_error_fn_t on_error;
  // Opaque pointer passed to every callback.
  void* user_data;
  // Optional progress edge when another endpoint returns shared send capacity.
  void (*on_capacity)(void* user_data);
} iree_net_queue_channel_callbacks_t;

// Allocates a queue channel over |endpoint| without changing its callbacks.
//
// The endpoint and callback user data must outlive the channel. COMMAND,
// ADVANCE and error callbacks are required. Call iree_net_queue_channel_attach
// to perform protocol handoff.
IREE_API_EXPORT iree_status_t iree_net_queue_channel_allocate(
    iree_net_message_endpoint_t endpoint,
    iree_net_queue_channel_callbacks_t callbacks,
    iree_allocator_t host_allocator, iree_net_queue_channel_t** out_channel);

// Frees a queue channel after its endpoint has fully deactivated.
//
// This does not operate on the borrowed endpoint or clear its callbacks.
IREE_API_EXPORT void iree_net_queue_channel_free(
    iree_net_queue_channel_t* channel);

// Installs the queue channel callback bundle on its borrowed endpoint.
//
// Call before endpoint activation or from inside the final bootstrap message
// callback. Serialized endpoint delivery guarantees all later messages use the
// queue channel callbacks.
IREE_API_EXPORT void iree_net_queue_channel_attach(
    iree_net_queue_channel_t* channel);

// Returns the endpoint's advisory send admission budget.
//
// A later terminal send completion is the readiness edge after backpressure.
// The send operation remains the authoritative admission check.
IREE_API_EXPORT iree_net_carrier_send_budget_t
iree_net_queue_channel_query_send_budget(iree_net_queue_channel_t* channel);

// Sends one COMMAND directed at |queue_id|.
//
// |queue_id| may be IREE_NET_QUEUE_ID_NONE for protocol commands that are not
// directed at one queue. The combined generated and borrowed application
// payload must be nonempty. An OK return guarantees exactly one terminal
// completion; a non-OK return guarantees none.
IREE_API_EXPORT iree_status_t iree_net_queue_channel_send_command(
    iree_net_queue_channel_t* channel, uint32_t queue_id,
    const iree_net_queue_channel_send_params_t* params);

// Sends one ADVANCE carrying a nonempty signal frontier.
//
// ADVANCE does not target one queue and cannot carry a wait frontier. Its
// opaque application payload may be empty. Completion semantics match
// iree_net_queue_channel_send_command.
IREE_API_EXPORT iree_status_t iree_net_queue_channel_send_advance(
    iree_net_queue_channel_t* channel,
    const iree_net_queue_channel_send_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_QUEUE_QUEUE_CHANNEL_H_
