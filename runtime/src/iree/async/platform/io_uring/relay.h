// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for io_uring relay implementation.
//
// io_uring-specific state enum and implementation functions. The relay struct
// is defined in the shared iree/async/relay.h with a platform union.
// Relays use multishot POLL_ADD for persistent monitoring and callback-based
// sink execution during poll().

#ifndef IREE_ASYNC_PLATFORM_IO_URING_RELAY_H_
#define IREE_ASYNC_PLATFORM_IO_URING_RELAY_H_

#include "iree/async/relay.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_io_uring_t iree_async_proactor_io_uring_t;

//===----------------------------------------------------------------------===//
// Relay state
//===----------------------------------------------------------------------===//

// Internal state for relay lifecycle management.
typedef enum iree_async_io_uring_relay_state_e {
  // Relay is active and monitoring source.
  IREE_ASYNC_IO_URING_RELAY_STATE_ACTIVE = 0,

  // Terminal unregistration was requested but an SQE was not available for
  // the cancellation operation. The sink will not fire in this state.
  IREE_ASYNC_IO_URING_RELAY_STATE_UNREGISTRATION_PENDING = 1,

  // The terminal cancellation operation was submitted and the relay is
  // waiting for its final source CQE. The sink will not fire in this state.
  IREE_ASYNC_IO_URING_RELAY_STATE_UNREGISTRATION_SUBMITTED = 2,

  // Relay needs to re-arm its source monitoring but failed due to SQ pressure.
  // The proactor will retry submission on the next poll cycle.
  IREE_ASYNC_IO_URING_RELAY_STATE_REARM_PENDING = 3,

  // Relay faulted while its multishot source was still active, but an SQE was
  // not available to cancel it. The sink will not fire in this state.
  IREE_ASYNC_IO_URING_RELAY_STATE_FAULT_CANCELLATION_PENDING = 4,

  // Relay faulted and cancellation of its active multishot source was
  // submitted. The sink will not fire in this state.
  IREE_ASYNC_IO_URING_RELAY_STATE_FAULT_CANCELLATION_SUBMITTED = 5,

  // Relay faulted and has no remaining kernel references. The caller-visible
  // handle remains valid until terminal unregistration.
  IREE_ASYNC_IO_URING_RELAY_STATE_FAULTED = 6,

  // A persistent poll source terminated without a relay fault. The
  // caller-visible handle remains valid until terminal unregistration.
  IREE_ASYNC_IO_URING_RELAY_STATE_TERMINAL = 7,
} iree_async_io_uring_relay_state_t;

//===----------------------------------------------------------------------===//
// Implementation functions
//===----------------------------------------------------------------------===//

iree_status_t iree_async_io_uring_register_relay(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay);

void iree_async_io_uring_unregister_relay(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay,
    iree_async_relay_unregistered_callback_t callback);

// Called from CQE processing when a relay's source fires.
// Executes the sink action and handles re-arming or cleanup.
void iree_async_io_uring_handle_relay_cqe(
    iree_async_proactor_io_uring_t* proactor, iree_async_relay_t* relay,
    int32_t result, uint32_t cqe_flags);

// Retries terminal unregistration and source re-arming operations deferred by
// SQ pressure. Called from poll() after processing CQEs when space may be
// available.
void iree_async_io_uring_retry_pending_relays(
    iree_async_proactor_io_uring_t* proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_RELAY_H_
