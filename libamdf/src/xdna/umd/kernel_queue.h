// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_KERNEL_QUEUE_H_
#define AMDF_SRC_XDNA_UMD_KERNEL_QUEUE_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/wait.h"
#include "libamdf/src/xdna/umd/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_xdna_umd_kernel_queue_t amdf_xdna_umd_kernel_queue_t;

// Acquires one native kernel-mediated queue from a scheduling context.
// Cold path: all packet storage, completion resources and mandatory bootstrap
// are ready before return. Submit, status and wait cannot lazily prepare them.
amdf_status_t amdf_xdna_umd_kernel_queue_create(
    amdf_xdna_umd_context_t* context, uint32_t capacity,
    amdf_xdna_umd_kernel_queue_t** out_queue);

// Returns native event types qualified during cold queue construction. This
// immutable query performs no native call, allocation or initialization.
amdf_native_event_types_t amdf_xdna_umd_kernel_queue_query_notification_types(
    const amdf_xdna_umd_kernel_queue_t* queue);

// Requests one native-progress hint into a validated supported event. Zero
// means the shared owner already checked the public point and requires an
// immediate hint. No event descriptor or subscription is retained. Explicit
// native registration/signaling calls are allowed; result consumption,
// library allocation, waits, locks, retries and lazy setup are not.
amdf_status_t amdf_xdna_umd_kernel_queue_request_notification(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_native_event_t* event);

// Frames one validated context-qualified instruction range in the queue's
// preallocated native packet at slot. No instruction bytes are read or
// modified. Common code owns this idle slot through publication and checked
// retirement. Hot path: common code exclusively owns publication. No provider
// lock, allocation, resource initialization, indirect BO list or native-submit
// retry is permitted. Native packet cache publication and driver entry are
// part of this boundary; instruction and data publication remain caller-owned.
amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot,
    uint64_t instruction_address, uint32_t instruction_byte_length,
    uint64_t* out_native_submission);

// Samples the native progress fence with device-to-host acquire semantics.
// No locks, allocation, initialization, system calls or active polling. A
// provider without a mapped fence returns progress cached by waits or refresh.
uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue);

// Refreshes native progress without waiting. Native query/poll syscalls are
// permitted when no mapped fence exists. This does not consume command results
// and may run concurrently with submission and waits. No host allocation,
// resource creation, lazy preparation or native submission occurs here.
amdf_status_t amdf_xdna_umd_kernel_queue_refresh_progress(
    amdf_xdna_umd_kernel_queue_t* queue);

// Observes command-owned completion data after native progress proves
// retirement. The caller exclusively owns the pending slot and retains the
// native packet throughout this call. Records execution failure without
// delaying retirement.
// Called by capacity reclamation, explicit synchronization or exclusive
// teardown, never by a status query. It takes no lock and performs no
// allocation, initialization or wait; packet inspection and atomic terminal
// status publication are its only native-state work.
void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue, uint32_t slot);

// Returns the observed terminal device failure without native queries. An
// operation error alone does not fail the queue or establish retirement.
// Atomic observation only; no locks, allocation, initialization or polling.
amdf_status_t amdf_xdna_umd_kernel_queue_query_terminal_status(
    const amdf_xdna_umd_kernel_queue_t* queue);

// Waits for one native progress value with caller-selected polling.
// Explicit synchronization path: clocks, polling, yielding, native waits and
// wait-event serialization are permitted within the same deadline. All reusable
// resources already exist; no provider allocation or lazy setup occurs here.
amdf_status_t amdf_xdna_umd_kernel_queue_wait(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline);

// Consumes an idle native queue lease even when native cleanup fails. The
// shared layer has already established retirement. Native errors retain their
// domains; API-domain BUSY is reserved for the shared layer's precondition
// rejection.
amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_KERNEL_QUEUE_H_
