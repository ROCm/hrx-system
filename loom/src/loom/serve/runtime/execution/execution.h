// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_EXECUTION_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_EXECUTION_H_

#include "loom/serve/runtime/execution/workspace.h"

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

// Retains the device, both exact queues and the shared workspace. Both queues
// belong to the same device and must be admitted by the workspace's backing
// scope. Other execution domains may use different devices in that scope.
// The device must belong to the workspace's live group, whose lifetime
// encloses the execution domain and all its work.
iree_status_t loom_serve_execution_create(
    iree_hal_queue_t* dispatch_queue, iree_hal_queue_t* transfer_queue,
    loom_serve_workspace_t* workspace, iree_allocator_t host_allocator,
    loom_serve_execution_t** out_execution);

// Retains the execution domain. NULL is accepted.
void loom_serve_execution_retain(loom_serve_execution_t* execution);

// Releases ownership without an implicit host wait. The owner first retires
// all model-visible consumers and observes their terminal outcome. Raw
// allocation/deallocation bookkeeping can remain pending: it contains no model
// payloads and stays owned by the shared workspace. External buffers and host
// payloads require their own retirement when not covered by execution_alloca.
// NULL is accepted.
void loom_serve_execution_release(loom_serve_execution_t* execution);

// Terminal model-consumer observation, independent of semaphore readiness.
// Stop submissions, attach the nonblocking wake, release caller-owned views,
// and query is_retired. A callback only schedules the owner, never reenters
// this object or tears down resources. Detaching joins its final payload
// access. This covers roots returned by execution_alloca and their dependent
// views; weights and other externally owned buffers have their own retirement
// scopes. Raw alloca/dealloca bookkeeping may outlive the model at the shared
// workspace.
void loom_serve_execution_set_retirement_observer(
    loom_serve_execution_t* execution, void (*wake)(void* user_data),
    void* user_data);
bool loom_serve_execution_is_retired(loom_serve_execution_t* execution);

// Cold registration of a command's private workspace requirement. Grows the
// shared pool's slab geometry without draining any domain or committing device
// memory. Existing roots retain their original pool through queue retirement.
iree_status_t loom_serve_execution_reserve_workspace(
    loom_serve_execution_t* execution, iree_device_size_t length,
    iree_device_size_t alignment);

// Enqueues a private workspace allocation on the work timeline. The caller
// owns the returned tracked view, orders its consumers through this domain,
// enqueues execution_dealloca, and releases the view before draining the
// domain. The shared workspace separately retains the original root and source
// pool. A failed view allocation after accepted alloca leaves no caller buffer,
// but the accepted timeline and allocation bookkeeping remain protected.
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

// Enqueues a buffer-only native stage without a command buffer. The executable
// belongs to the exact dispatch queue family. HAL captures bindings and retains
// the executable and buffers until terminal completion; no host wait occurs.
iree_status_t loom_serve_execution_dispatch(
    loom_serve_execution_t* execution, iree_hal_executable_t* executable,
    iree_hal_executable_function_t function, iree_hal_dispatch_config_t config,
    iree_hal_buffer_ref_list_t bindings, uint64_t* out_value);

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
// Model-visible workspace views are joined after all callers release them;
// raw allocation bookkeeping transfers on release to the shared workspace.
iree_status_t loom_serve_execution_drain(loom_serve_execution_t* execution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_EXECUTION_EXECUTION_H_
