// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// rdma_cm event channel integration with the IREE proactor.
//
// rdma_cm exposes connection-manager events through a pollable fd. This
// component owns the event channel, registers it with an IREE proactor, and
// drains/acknowledges events from the proactor callback. Higher-level
// connect/listen state machines consume the borrowed event view while the
// underlying rdma_cm event is still valid.

#ifndef IREE_NET_CARRIER_RDMA_CM_CHANNEL_H_
#define IREE_NET_CARRIER_RDMA_CM_CHANNEL_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier/rdma/librdmacm.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_cm_channel_t iree_net_rdma_cm_channel_t;

// Borrowed view of one rdma_cm event.
typedef struct iree_net_rdma_cm_event_t {
  // Native rdma_cm event. Valid only during the callback; do not acknowledge
  // it.
  struct rdma_cm_event* native_event;

  // Native connection ID associated with the event.
  struct rdma_cm_id* id;

  // Native listener ID for connect-request events, if present.
  struct rdma_cm_id* listen_id;

  // rdma_cm event type.
  enum rdma_cm_event_type type;

  // rdma_cm event status. Non-zero values are transport/provider errors.
  int status;

  // Private data borrowed from native_event->param.conn.
  iree_const_byte_span_t private_data;
} iree_net_rdma_cm_event_t;

// Handles rdma_cm events and channel-level errors.
//
// |status| is OK when |event| is non-NULL. The event is borrowed and must not
// be retained after the callback returns; the channel acknowledges the native
// event immediately afterward. If |status| is non-OK then |event| is NULL and
// the callback owns the status.
typedef void (*iree_net_rdma_cm_channel_callback_fn_t)(
    void* user_data, iree_status_t status,
    const iree_net_rdma_cm_event_t* event);

typedef struct iree_net_rdma_cm_channel_callback_t {
  // Callback function invoked from the proactor poll thread.
  iree_net_rdma_cm_channel_callback_fn_t fn;

  // Opaque user data passed to fn.
  void* user_data;
} iree_net_rdma_cm_channel_callback_t;

// Creates an rdma_cm event channel registered with |proactor|.
//
// |librdmacm| is borrowed and must outlive the channel. The channel retains
// |proactor| and unregisters its event source during release.
IREE_API_EXPORT iree_status_t iree_net_rdma_cm_channel_create(
    const iree_net_librdmacm_t* librdmacm, iree_async_proactor_t* proactor,
    iree_net_rdma_cm_channel_callback_t callback,
    iree_allocator_t host_allocator, iree_net_rdma_cm_channel_t** out_channel);

// Releases the channel and unregisters it from the proactor.
//
// Must not be called from the channel callback.
IREE_API_EXPORT void iree_net_rdma_cm_channel_release(
    iree_net_rdma_cm_channel_t* channel);

// Returns the native rdma_cm event channel.
IREE_API_EXPORT struct rdma_event_channel*
iree_net_rdma_cm_channel_native_event_channel(
    const iree_net_rdma_cm_channel_t* channel);

// Drains all currently queued rdma_cm events.
//
// This is normally called automatically when the proactor reports the channel
// fd readable. It is exposed for deterministic tests and for setup paths that
// need to process already-queued events before returning to the poll loop.
IREE_API_EXPORT iree_status_t
iree_net_rdma_cm_channel_drain(iree_net_rdma_cm_channel_t* channel);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_CM_CHANNEL_H_
