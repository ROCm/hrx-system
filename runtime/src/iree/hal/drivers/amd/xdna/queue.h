// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/hal/drivers/amd/xdna/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates one finite execution owner. Context and proactor are borrowed from
// the device. Ready operations publish directly from their caller; the proactor
// owns deferred causal admission and unattended checked native completion.
// Private native-publication and host-transfer services isolate blocking work.
// Unresolved consumers occupy no native execution slot.
iree_status_t iree_hal_amd_xdna_queue_create(
    const iree_hal_queue_family_t* family, iree_hal_amd_xdna_context_t* context,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_hal_queue_t** out_queue);

// Assigns or clears an idle queue's causal domain. The tracker is retained.
// Topology assignment is externally serialized against submission and teardown.
iree_status_t iree_hal_amd_xdna_queue_assign_frontier(
    iree_hal_queue_t* queue, iree_async_frontier_tracker_t* tracker,
    iree_async_axis_t axis);

// Releases currently unused operation and payload capture blocks.
void iree_hal_amd_xdna_queue_trim(iree_hal_queue_t* queue);

// Completion invoked after the queue no longer borrows its native context or
// proactor. The callback runs from a placed proactor operation and may release
// the queue and the proactor-owning device. It must not access the queue after
// releasing either owner.
typedef void (*iree_hal_amd_xdna_queue_shutdown_fn_t)(void* user_data);

// Callback receiving successful terminal queue shutdown.
typedef struct iree_hal_amd_xdna_queue_shutdown_callback_t {
  // Function invoked after native and observer ownership has retired.
  iree_hal_amd_xdna_queue_shutdown_fn_t fn;
  // Borrowed callback context retained by the queue until invocation.
  void* user_data;
} iree_hal_amd_xdna_queue_shutdown_callback_t;

// Transfers an idle queue into poll-owned terminal shutdown. The caller has
// exclusive access after all public queue users and operation-held device
// references retire. A placed operation stops and joins both progress services,
// destroys the native queue, and unregisters the persistent event source. The
// terminal unregistration callback schedules a second placed operation before
// invoking |callback| so no proactor-owning storage is released inline.
//
// Any handoff, native BUSY, or observer retirement failure is diagnosed and
// retains the complete reachable ownership graph. In those cases |callback|
// does not fire.
void iree_hal_amd_xdna_queue_begin_shutdown(
    iree_hal_queue_t* queue,
    iree_hal_amd_xdna_queue_shutdown_callback_t callback);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_H_
