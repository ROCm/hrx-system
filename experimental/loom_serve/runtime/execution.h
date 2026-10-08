// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_EXECUTION_H_
#define IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_EXECUTION_H_

#include "experimental/loom_serve/runtime/workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

// Ordered model work with an independent feedback branch. Commands and input
// transfers serialize scratch reuse across the exact queues. Feedback waits
// on its producer and previous feedback, but never orders later model work.
// Neither queue order nor buffer aliasing establishes dependencies implicitly.
// A submission lock protects timeline assignment and enqueue as one operation.
// Completion values belong to this object and the method's named timeline.
typedef struct loom_serve_execution_t loom_serve_execution_t;

// Retains the device, both exact queues and the shared workspace. Queues must
// use the workspace's device memory domain. The device must belong to a live
// device group whose lifetime encloses the execution domain and all its work.
iree_status_t loom_serve_execution_create(
    iree_hal_queue_t* dispatch_queue, iree_hal_queue_t* transfer_queue,
    loom_serve_workspace_t* workspace, iree_allocator_t host_allocator,
    loom_serve_execution_t** out_execution);

void loom_serve_execution_retain(loom_serve_execution_t* execution);

// Releases ownership without an implicit host wait. The owner must drain both
// branches and observe failures before final release or recycling borrowed
// host I/O storage.
void loom_serve_execution_release(loom_serve_execution_t* execution);

// Cold registration of a command's private workspace requirement. Grows the
// shared pool's slab geometry without draining any domain or committing device
// memory. Existing roots retain their original pool through queue retirement.
iree_status_t loom_serve_execution_reserve_workspace(
    loom_serve_execution_t* execution, iree_device_size_t length,
    iree_device_size_t alignment);

// Enqueues a private workspace allocation on the work timeline. The caller
// owns the returned root, orders its consumers through this execution domain,
// enqueues dealloca, and releases the root before draining the domain. The pool
// belongs to the shared workspace; this domain tracks final root ownership.
iree_status_t loom_serve_execution_alloca(
    loom_serve_execution_t* execution,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_buffer_t** out_buffer);

// Enqueues release after all preceding accepted work. Caller ownership is not
// consumed. A synchronous rejection leaves the allocation live until its final
// queue/caller release; it does not tear down shared queues or pools.
iree_status_t loom_serve_execution_dealloca(loom_serve_execution_t* execution,
                                            iree_hal_buffer_t* buffer,
                                            uint64_t* out_value);

// Coherent snapshot of accepted work/feedback frontiers. Semaphores are
// borrowed through execution release. Owners may register asynchronous ANY/ALL
// waits on these values; subsequent submissions do not change this snapshot.
// Failure readiness is not proof that queue-captured buffers have retired.
typedef struct loom_serve_execution_frontier_t {
  // Model work followed by its independent host feedback branch.
  iree_hal_semaphore_t* semaphores[2];
  // Last accepted submission on each corresponding branch.
  uint64_t values[2];
} loom_serve_execution_frontier_t;

loom_serve_execution_frontier_t loom_serve_execution_frontier(
    loom_serve_execution_t* execution);

// Imports external dependencies into the model timeline without a host wait.
// Used at cross-domain boundaries such as parameter activation and composed
// model stages. Subsequent model work depends on both prior work and waits.
// Semaphore storage is borrowed only during this call; HAL retains handles.
iree_status_t loom_serve_execution_barrier(loom_serve_execution_t* execution,
                                           iree_hal_semaphore_list_t waits);

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

// Enqueues feedback downloads after all preceding accepted work and feedback.
// Returns a feedback completion, without advancing the model work timeline.
// Operations must be downloads; host destinations remain borrowed and device
// sources must not be overwritten until this completion retires. Independent
// later commands may read the same sources or consume other model state.
iree_status_t loom_serve_execution_feedback(
    loom_serve_execution_t* execution, iree_host_size_t operation_count,
    const iree_hal_transfer_operation_t* operations, uint64_t* out_value);

// Waits for a command/input-transfer value and propagates failure.
iree_status_t loom_serve_execution_wait(loom_serve_execution_t* execution,
                                        uint64_t value);

// Waits for a feedback value and propagates failure. This does not join work
// submitted after the feedback fork.
iree_status_t loom_serve_execution_feedback_wait(
    loom_serve_execution_t* execution, uint64_t value);

// Joins both accepted frontiers and their failures. A synchronous rejection
// never advances either frontier to an unsignaled value. A failed semaphore is
// not proof of final queue resource retirement; terminal owners also join their
// tracked buffer views before freeing borrowed payloads or virtual mappings.
// Private workspace roots are joined here after all callers release them.
iree_status_t loom_serve_execution_drain(loom_serve_execution_t* execution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_LOOM_SERVE_RUNTIME_EXECUTION_H_
