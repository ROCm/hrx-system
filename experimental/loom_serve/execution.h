// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_EXECUTION_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_EXECUTION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One ordered submission domain, shared by the model and its I/O.
// All operations explicitly wait on the preceding accepted submission. This
// serializes scratch reuse even across distinct transfer and dispatch queues;
// neither queue order nor buffer aliasing establishes dependencies implicitly.
// A submission lock protects timeline assignment and enqueue as one operation.
// Values returned by this object are meaningful only with this same object.
typedef struct loom_serve_execution_t loom_serve_execution_t;

// Retains the device and both exact queues. The device must belong to a live
// device group whose lifetime encloses the execution domain and all its work.
iree_status_t loom_serve_execution_create(
    iree_hal_queue_t* dispatch_queue, iree_hal_queue_t* transfer_queue,
    iree_allocator_t host_allocator, loom_serve_execution_t** out_execution);

void loom_serve_execution_retain(loom_serve_execution_t* execution);

// Releases ownership without an implicit host wait. The owner must drain and
// observe failures before final release or recycling borrowed host I/O storage.
void loom_serve_execution_release(loom_serve_execution_t* execution);

// Enqueues reusable commands. HAL captures the binding table and retains its
// buffers before returning; the table itself may immediately be reused.
// Success returns an accepted submission value, not a completed result.
iree_status_t loom_serve_execution_execute(
    loom_serve_execution_t* execution,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t bindings, uint64_t* out_value);

// Enqueues transfers on the same dependency chain. Host payload memory remains
// borrowed through completion of the returned value, including error cleanup.
iree_status_t loom_serve_execution_transfer(
    loom_serve_execution_t* execution, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, uint64_t* out_value);

// Waits for a value returned by this execution domain and propagates failure.
iree_status_t loom_serve_execution_wait(loom_serve_execution_t* execution,
                                        uint64_t value);

// Waits only for the last successfully accepted submission. A synchronous
// rejection never advances that frontier to an unsignaled value.
iree_status_t loom_serve_execution_drain(loom_serve_execution_t* execution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_EXECUTION_H_
