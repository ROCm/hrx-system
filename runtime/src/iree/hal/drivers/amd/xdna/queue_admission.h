// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Semaphore capture, causal admission and terminal operation ownership.
#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_ADMISSION_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_ADMISSION_H_

#include "iree/hal/drivers/amd/xdna/queue_operation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers signals awaiting native acceptance in the queue-local index.
// Admission and index mutation hold the queue state mutex.
void iree_hal_amd_xdna_operation_register_producers(
    iree_hal_amd_xdna_operation_t* operation);

// Removes indexed signals and transfers the list of acceptance waiters.
iree_hal_amd_xdna_operation_t* iree_hal_amd_xdna_operation_unregister_producers(
    iree_hal_amd_xdna_operation_t* operation);

// Reconsiders consumers after their producer changes acceptance state.
void iree_hal_amd_xdna_operation_wake_acceptance_waiters(
    iree_hal_amd_xdna_operation_t* operation);

// Publishes captured signals using the resolved operation frontier. The caller
// keeps the operation and device alive through inline signal callbacks.
iree_status_t iree_hal_amd_xdna_operation_signal(
    iree_hal_amd_xdna_operation_t* operation, iree_hal_device_t* queue_device);

// Releases terminal resources, publishes signals and recycles the operation.
// Native users must have retired or been rejected before this call.
void iree_hal_amd_xdna_operation_complete(
    iree_hal_amd_xdna_operation_t* operation);

// Resolves the unproven suffix of the captured wait list. Native operations
// hold the state mutex to read FIFO causality; memory operations have no FIFO
// state and resolve only their captured semaphore dependencies.
iree_status_t iree_hal_amd_xdna_operation_resolve_waits(
    iree_hal_amd_xdna_operation_t* operation,
    iree_hal_amd_xdna_wait_resolution_t* out_resolution,
    iree_host_size_t* out_deferred_wait_index);

// Consumes a causal admission result while the state mutex is held.
void iree_hal_amd_xdna_operation_admit(iree_hal_amd_xdna_operation_t* operation,
                                       iree_status_t status);

// Resolves memory dependencies directly on the caller or last wait callback.
// No queue state lock or native admission owner is required.
void iree_hal_amd_xdna_operation_admit_memory(
    iree_hal_amd_xdna_operation_t* operation, iree_status_t status);

// Reconsiders consumers attached while a caller owned native publication.
void iree_hal_amd_xdna_queue_retry_acceptance_waiters(
    iree_hal_amd_xdna_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_ADMISSION_H_
