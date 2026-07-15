// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_COMPLETION_QUEUE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_COMPLETION_QUEUE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

typedef struct iree_hal_amdxdna_completion_queue_t
    iree_hal_amdxdna_completion_queue_t;
typedef struct iree_hal_amdxdna_completion_batch_t
    iree_hal_amdxdna_completion_batch_t;

typedef iree_status_t (*iree_hal_amdxdna_completion_action_fn_t)(
    void* user_data);
typedef void (*iree_hal_amdxdna_completion_cleanup_fn_t)(void* user_data);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a per-device native-completion queue. The async queue owns issue
// ordering; this queue usually owns native waits, post-completion actions,
// cleanup, and HAL semaphore signaling after native completion. Destroy drains
// all submitted batches before returning.
iree_status_t iree_hal_amdxdna_completion_queue_create(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_completion_queue_t** out_queue);

void iree_hal_amdxdna_completion_queue_set_frontier(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_atomic_uint64_t* epoch_counter);

void iree_hal_amdxdna_completion_queue_destroy(
    iree_hal_amdxdna_completion_queue_t* queue);

iree_hal_amdxdna_completion_batch_t* iree_hal_amdxdna_completion_batch_retain(
    iree_hal_amdxdna_completion_batch_t* batch);

// Creates a completion batch with its own clone of |signal_list|. If the batch
// is later submitted after native work/actions are added, it takes over
// signaling/failing the HAL queue operation. If no native work is added and the
// batch was never published, submit destroys it without signaling so the async
// queue can use its normal signal path. A published batch always owns the
// signal list, even when it completes without native work.
iree_status_t iree_hal_amdxdna_completion_batch_create(
    iree_hal_amdxdna_completion_queue_t* queue,
    iree_hal_semaphore_list_t signal_list,
    iree_hal_amdxdna_completion_batch_t** out_batch);

// Publishes this batch as the native completion producer for its signal list.
// Host waits may retain and wait on the batch before it has been submitted by
// the async worker. This is intentionally idempotent so submit paths can
// publish as a fallback when they receive a never-published batch.
void iree_hal_amdxdna_completion_batch_publish_signals(
    iree_hal_amdxdna_completion_batch_t* batch);

// Retains |command_buffer| until the batch completes. Used by direct command
// buffers whose native commands/buffers live in command-buffer-owned storage.
void iree_hal_amdxdna_completion_batch_retain_command_buffer(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_command_buffer_t* command_buffer);

// Adds a native submission wait. The batch owns |submission| after this call.
iree_status_t iree_hal_amdxdna_completion_batch_add_submission(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_amdxdna_native_submission_t* submission);

// Adds an ordered post-completion action. Actions with |run_on_error=false|
// are skipped after an earlier native wait/action fails. Actions and cleanups
// run on the completion worker and must not submit new queue work.
iree_status_t iree_hal_amdxdna_completion_batch_add_action(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_amdxdna_completion_action_fn_t action_fn,
    iree_hal_amdxdna_completion_cleanup_fn_t cleanup_fn, void* user_data,
    bool run_on_error);

// Adds cleanup that always runs after native waits/actions and before the HAL
// signal list is signaled or failed.
iree_status_t iree_hal_amdxdna_completion_batch_add_cleanup(
    iree_hal_amdxdna_completion_batch_t* batch,
    iree_hal_amdxdna_completion_cleanup_fn_t cleanup_fn, void* user_data);

void iree_hal_amdxdna_completion_batch_record_error(
    iree_hal_amdxdna_completion_batch_t* batch, iree_status_t status);

bool iree_hal_amdxdna_completion_batch_has_work(
    const iree_hal_amdxdna_completion_batch_t* batch);

// Submits the batch to the completion worker. Returns IREE_STATUS_DEFERRED when
// the batch took ownership of signaling. Returns OK when the batch had no
// native work/actions and was destroyed without signaling.
iree_status_t iree_hal_amdxdna_completion_batch_submit(
    iree_hal_amdxdna_completion_batch_t* batch);

// Waits for the completion worker to finish a submitted batch. Finite timeouts
// only bound the host wait; native waits/actions continue on the worker until
// the batch is signaled or failed.
iree_status_t iree_hal_amdxdna_completion_batch_wait(
    iree_hal_amdxdna_completion_batch_t* batch, iree_timeout_t timeout,
    iree_async_wait_flags_t flags);

void iree_hal_amdxdna_completion_batch_destroy(
    iree_hal_amdxdna_completion_batch_t* batch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_COMPLETION_QUEUE_H_
