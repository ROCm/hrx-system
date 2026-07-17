// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_NET_CHANNEL_UTIL_SEND_GATE_H_
#define IREE_NET_CHANNEL_UTIL_SEND_GATE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/notification.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Synchronizes channel send admission with endpoint detachment.
//
// The gate packs its admission count and lifecycle bits into one atomic word.
// An admission either increments the count before close or observes the closed
// bit and fails; close either observes every admitted sender or prevents it
// from entering. This lets the send path remain lock-free while close sleeps on
// a notification until the exact quiescence transition.
//
// Callers must keep the gate live from a successful try_enter through leave.
// Close callers must keep it live until close completion has been observed.
typedef struct iree_net_channel_send_gate_t {
  // Packed admission count and lifecycle bits.
  iree_atomic_uint32_t state;

  // Posted when admitted sends quiesce or close finishes.
  iree_notification_t notification;
} iree_net_channel_send_gate_t;

// Initializes an open gate with no admitted sends.
void iree_net_channel_send_gate_initialize(
    iree_net_channel_send_gate_t* out_gate);

// Deinitializes a gate with no admitted sends or close callers.
//
// An open gate may be deinitialized when its owning channel was never attached.
// A gate that began closing must have finished closing before deinitialization.
void iree_net_channel_send_gate_deinitialize(
    iree_net_channel_send_gate_t* gate);

// Attempts to admit one send operation.
//
// Returns true when admitted. The caller must call leave exactly once after it
// no longer accesses the protected endpoint. Returns false after close begins.
bool iree_net_channel_send_gate_try_enter(iree_net_channel_send_gate_t* gate);

// Releases one successful admission.
void iree_net_channel_send_gate_leave(iree_net_channel_send_gate_t* gate);

// Begins closing the gate and prevents all future admissions.
//
// Exactly one concurrent caller returns true and owns endpoint cleanup. Other
// callers return false and must call await_closed instead.
bool iree_net_channel_send_gate_begin_close(iree_net_channel_send_gate_t* gate);

// Blocks the close owner until every admitted send has left.
void iree_net_channel_send_gate_await_quiescence(
    iree_net_channel_send_gate_t* gate);

// Marks endpoint cleanup complete and releases concurrent close callers.
//
// Must be called exactly once by the close owner after quiescence.
void iree_net_channel_send_gate_finish_close(
    iree_net_channel_send_gate_t* gate);

// Blocks a non-owning close caller until endpoint cleanup is complete.
void iree_net_channel_send_gate_await_closed(
    iree_net_channel_send_gate_t* gate);

// Returns the number of currently admitted sends.
uint32_t iree_net_channel_send_gate_pending_count(
    const iree_net_channel_send_gate_t* gate);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_UTIL_SEND_GATE_H_
