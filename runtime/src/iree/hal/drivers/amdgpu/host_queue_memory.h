// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_MEMORY_H_
#define IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_MEMORY_H_

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_amdgpu_alloca_reservation_readiness_e {
  // Every reservation can be materialized and submitted immediately.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY = 0,
  // The transaction must wait for a pool death frontier before materialization.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT = 1,
  // The pool needs cold backing growth before another queue-locked reservation
  // attempt can succeed.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_GROWTH = 2,
  // The pool is exhausted or over budget and needs a release notification
  // retry.
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION = 3,
} iree_hal_amdgpu_alloca_reservation_readiness_t;

typedef struct iree_hal_amdgpu_alloca_transaction_t {
  // Scheduler action required before the transaction can be submitted.
  iree_hal_amdgpu_alloca_reservation_readiness_t readiness;
  // Pool acquisition result that produced |reservations|.
  iree_hal_pool_acquire_result_t acquire_result;
  // Number of sibling requests in the all-or-nothing transaction.
  iree_host_size_t request_count;
  // Canonical allocation requests retained for the transaction lifetime.
  const iree_hal_pool_reservation_request_t* requests;
  // Transient wrappers corresponding one-to-one with |requests|.
  iree_hal_buffer_t* const* buffers;
  // Caller-provided reservation storage with |request_count| elements.
  iree_hal_pool_reservation_t* reservations;
  // Caller-provided acquisition metadata with |request_count| elements.
  iree_hal_pool_acquire_info_t* acquire_infos;
  // Caller-provided materialization storage with |request_count| elements.
  iree_hal_buffer_t** backing_buffers;
  // Caller-provided union of every reservation wait frontier.
  iree_async_frontier_t* wait_frontier;
  // Queue wait resolution to use when publishing the alloca signal.
  iree_hal_amdgpu_wait_resolution_t wait_resolution;
  // True while the transaction owns every entry in |reservations|.
  bool reservations_held;
  // True while the transaction owns every entry in |backing_buffers|.
  bool backing_buffers_held;
} iree_hal_amdgpu_alloca_transaction_t;

typedef struct iree_hal_amdgpu_dealloca_transaction_t {
  // Number of sibling buffers in the all-or-nothing transaction.
  iree_host_size_t buffer_count;
  // Transient allocation roots captured by the transaction.
  iree_hal_buffer_t* const* buffers;
  // Source pool shared by every buffer in the transaction.
  iree_hal_pool_t* pool;
  // Caller-provided storage receiving detached reservation tokens.
  iree_hal_pool_reservation_t* reservations;
} iree_hal_amdgpu_dealloca_transaction_t;

// Validates/canonicalizes an allocation transaction against the exact source
// pool and creates the transient wrappers returned from queue_alloca.
iree_status_t iree_hal_amdgpu_host_queue_prepare_alloca_buffers(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    iree_host_size_t request_count,
    const iree_hal_pool_reservation_request_t* requests,
    iree_hal_pool_reservation_request_t* out_canonical_requests,
    iree_hal_buffer_t** out_buffers);

// Attempts to reserve bytes from |allocation_pool| and classifies the result
// as immediate, death-frontier-waitable, or notification-retry-required.
iree_status_t iree_hal_amdgpu_host_queue_acquire_alloca_transaction(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_pool_t* allocation_pool,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_alloca_transaction_t* transaction);

// Materializes every reservation in a ready transaction. Does not require
// submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_materialize_alloca_transaction(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction);

// Releases every materialized reservation in a transaction that was not
// submitted.
void iree_hal_amdgpu_host_queue_release_alloca_transaction(
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_alloca_transaction_t* transaction);

// Stages every materialized reservation on its transient buffer and submits the
// queue barrier that commits the transaction on completion. Caller must hold
// submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_materialization(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Materializes a ready transaction, stages every reservation on its transient
// buffer, and submits the queue barrier that commits the transaction on
// completion. Caller must hold submission_mutex.
iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_transaction(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_alloca_transaction_t* transaction,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

// Submits the queue barrier that decommits every transient buffer on
// completion.
iree_status_t iree_hal_amdgpu_host_queue_submit_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_dealloca_transaction_t* transaction,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_HOST_QUEUE_MEMORY_H_
