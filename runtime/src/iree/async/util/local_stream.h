// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_UTIL_LOCAL_STREAM_H_
#define IREE_ASYNC_UTIL_LOCAL_STREAM_H_

#include "iree/async/primitive.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Owner-thread exact byte/resource transfers on a local native stream.
//
// This optional helper composes native resource passing with proactor
// readiness. It is intended for sequential connection bootstrap and subsequent
// EOF observation, not concurrent reads/writes or ordinary bulk/message
// traffic. There is one reusable transfer slot and no worker, retry timer, or
// idle polling. Every method is serialized with poll on the proactor's owner
// thread (including before the first poll). Callers marshal cross-thread
// requests themselves.
typedef struct iree_async_local_stream_t iree_async_local_stream_t;

typedef struct iree_async_local_stream_callback_t {
  // Receives owned terminal status on the poll owner. May start the next
  // transfer or deactivate the stream. Failure poisons the stream; only
  // deactivation is then permitted. Never invoked inline by send, receive, or
  // pipe_accept.
  void (*fn)(void* user_data, iree_status_t status);
  // Borrowed through the callback.
  void* user_data;
} iree_async_local_stream_callback_t;

typedef struct iree_async_local_stream_deactivated_callback_t {
  // Invoked after native operations, cancellation receipts, and
  // observer/control registrations retire. May destroy the stream and release
  // its borrowed channel.
  void (*fn)(void* user_data);
  // Borrowed through the callback.
  void* user_data;
} iree_async_local_stream_deactivated_callback_t;

// Allocates a helper borrowing |channel| exclusively through deactivation and
// destruction.
// The caller retains the channel's owner; no channel duplication is performed.
// Retains |proactor|. Allocates all resource scratch and native control state
// here; transfers do not allocate helper storage per fragment.
//
// POSIX requires a connected, nonblocking AF_UNIX SOCK_STREAM descriptor (for
// example, borrowed from a managed async socket). Windows requires an
// overlapped byte named pipe not associated with an IOCP. Server endpoints must
// complete pipe_accept before transferring; client endpoints are already
// connected. Windows observation is registered before any native I/O can be
// issued. Rejected creation leaves the channel with the caller and owes no
// callback.
iree_status_t iree_async_local_stream_allocate(
    iree_async_proactor_t* proactor, iree_async_primitive_t channel,
    iree_host_size_t handle_capacity, iree_allocator_t host_allocator,
    iree_async_local_stream_t** out_stream);

// Sends exactly |data.data_length| bytes and |handle_count| resources. Requires
// nonempty data and handle_count <= the configured capacity. Borrows data and
// handles through the callback; rejection owes no callback. Only one transfer
// may be active. The callback may start another transfer.
//
// Send completion does NOT establish import completion. In particular, Windows
// transmits source handle values for peer-side duplication. Source handles must
// remain stable until the caller's higher-level import acknowledgement. POSIX
// attaches SCM_RIGHTS only until the first successful byte is sent.
// Windows resources must support DuplicateHandle; sockets, completion ports,
// and pseudo-handles are not transferable through this helper.
iree_status_t iree_async_local_stream_send(
    iree_async_local_stream_t* stream, iree_const_byte_span_t data,
    iree_host_size_t handle_count, const iree_async_primitive_t* handles,
    iree_async_local_stream_callback_t callback);

// Receives exactly the declared byte and resource extents, known from the
// caller's protocol phase. There is no implicit framing header or import ACK.
// Borrows data and out_handles through the callback; rejection owes no
// callback. On admission, clears out_handles to NONE. Success transfers
// ownership of the complete resource bundle to the caller. Failure closes every
// acquired local resource and leaves all outputs NONE; data may contain a
// partial prefix.
//
// EOF before the byte extent completes reports OUT_OF_RANGE. Receiving one byte
// with zero resources can therefore monitor peer closure after bootstrap.
// Unexpected data is then a protocol violation for the caller to interpret.
// POSIX receives ancillary data on every fragment, including
// overflow/truncation cleanup. Windows duplicates noninheritable handles from
// the kernel-identified pipe peer. Imported resources remain tentative until
// the caller validates its protocol and authorizes use.
iree_status_t iree_async_local_stream_receive(
    iree_async_local_stream_t* stream, iree_byte_span_t data,
    iree_host_size_t handle_count, iree_async_local_stream_callback_t callback,
    iree_async_primitive_t* out_handles);

// Closes admission and joins all ownership. An active transfer receives its
// terminal callback before the deactivation callback. Call exactly once after
// successful create, even if no transfer was started. Callbacks are deferred
// until poll. Duplicate deactivation returns FAILED_PRECONDITION without a
// callback. Native cancellation failure propagates through poll while storage
// remains owned; it does not authorize freeing kernel-owned memory.
// Cancellation is advisory: a completed native transfer may still succeed.
iree_status_t iree_async_local_stream_deactivate(
    iree_async_local_stream_t* stream,
    iree_async_local_stream_deactivated_callback_t callback);

// Frees a deactivated helper. NULL is permitted. The borrowed channel remains
// the caller's responsibility. May be called from the deactivation callback.
void iree_async_local_stream_free(iree_async_local_stream_t* stream);

#if defined(IREE_PLATFORM_WINDOWS)

enum iree_async_local_stream_pipe_flag_bits_e {
  IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_NONE = 0u,
  // Atomically claims a new pipe name; fails if that name is already in use.
  // Subsequent instances of the same listener omit this flag.
  IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE = 1u << 0,
};
typedef uint32_t iree_async_local_stream_pipe_flags_t;

// Creates one overlapped, duplex, byte-mode server pipe instance. |name| is a
// UTF-8 local pipe name, without the \\.\pipe\ prefix or path separators. Uses
// the process's default security descriptor and rejects remote clients.
// |instance_limit| is in [1, 255] (255 is the native unlimited-instance value).
// Returns an owned primitive, closed by the caller after helper deactivation.
iree_status_t iree_async_local_stream_pipe_create(
    iree_string_view_t name, uint32_t instance_limit,
    iree_async_local_stream_pipe_flags_t flags,
    iree_async_primitive_t* out_channel);

// Opens one local pipe client endpoint in overlapped mode, without waiting for
// capacity. A busy server returns UNAVAILABLE; absent/access errors retain
// their native classification. No retry or worker is created. Name/ownership
// match pipe_create. This call precedes admission of the caller's asynchronous
// setup.
iree_status_t iree_async_local_stream_pipe_open(
    iree_string_view_t name, iree_async_primitive_t* out_channel);

// Accepts a client on the helper's server pipe using the reusable native slot.
// Call once before send/receive, including when the client opened before this
// call. Deactivation cancels and joins a pending accept without peer activity.
iree_status_t iree_async_local_stream_pipe_accept(
    iree_async_local_stream_t* stream,
    iree_async_local_stream_callback_t callback);

#endif  // IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_LOCAL_STREAM_H_
