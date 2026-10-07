// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_IOCP_EVENT_SOURCE_H_
#define IREE_ASYNC_PLATFORM_IOCP_EVENT_SOURCE_H_

#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_async_proactor_iocp_t;

// Completion key used for event source wait completions. The completion
// overlapped value carries the iree_async_event_source_t pointer.
#define IREE_ASYNC_IOCP_EVENT_SOURCE_COMPLETION_KEY ((uintptr_t)3)

enum iree_async_iocp_event_source_flag_bits_e {
  // A native association may still publish one completion packet.
  IREE_ASYNC_IOCP_EVENT_SOURCE_FLAG_ARMED = 1u << 0,
  // Cleanup failed and the complete source/proactor graph must be retained.
  IREE_ASYNC_IOCP_EVENT_SOURCE_FLAG_RETAINED = 1u << 1,
};
typedef uint32_t iree_async_iocp_event_source_flags_t;

// Tracks a wait completion packet monitoring a caller-owned HANDLE.
struct iree_async_event_source_t {
  // Next source in the proactor-owned list.
  struct iree_async_event_source_t* next;

  // Previous source in the proactor-owned list.
  struct iree_async_event_source_t* previous;

  // Caller-owned waitable HANDLE monitored by the wait packet.
  uintptr_t target_handle;

  // Wait completion packet HANDLE owned by this source.
  uintptr_t wait_packet_handle;

  // Native association and terminal-retention state.
  iree_async_iocp_event_source_flags_t flags;

  // User callback dispatched by the polling thread; cleared on unregistration.
  iree_async_event_source_callback_t callback;

  // Final ownership return, dispatched only after native packet retirement.
  iree_async_event_source_unregistered_callback_t unregistered_callback;
};

// Registers a caller-owned waitable HANDLE with the IOCP proactor.
iree_status_t iree_async_iocp_event_source_register(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source);

// Stops callback admission and joins the source's native packet before
// callback.
void iree_async_iocp_event_source_unregister(
    iree_async_proactor_t* base_proactor,
    iree_async_event_source_t* event_source,
    iree_async_event_source_unregistered_callback_t callback);

// Dispatches one event source completion and re-arms its wait packet.
// Returns an operational rearm failure through the owning poll call.
iree_status_t iree_async_iocp_event_source_dispatch(
    struct iree_async_proactor_iocp_t* proactor,
    iree_async_event_source_t* event_source);

// Cancels all sources and drains admitted retirements during owner destruction.
// Failure retains the proactor because native callback reachability remains.
iree_status_t iree_async_iocp_event_source_deinitialize_all(
    struct iree_async_proactor_iocp_t* proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IOCP_EVENT_SOURCE_H_
