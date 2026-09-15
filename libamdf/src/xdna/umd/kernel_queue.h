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
    amdf_xdna_umd_context_t* context, amdf_xdna_umd_kernel_queue_t** out_queue);

// Frames one validated context-qualified instruction range in the queue's
// preallocated native packet. No instruction bytes are read or modified.
// Hot path: common code exclusively owns the submission slot. No provider
// lock, allocation, resource initialization, indirect BO list or native-submit
// retry is permitted. Native packet cache publication and driver entry are
// part of this boundary; instruction and data publication remain caller-owned.
amdf_status_t amdf_xdna_umd_kernel_queue_submit(
    amdf_xdna_umd_kernel_queue_t* queue, uint64_t instruction_address,
    uint32_t instruction_byte_length, uint64_t* out_native_submission);

// Samples the native progress fence with device-to-host acquire semantics.
// No locks, allocation, initialization, system calls or active polling. A
// provider without a mapped fence returns progress cached by explicit waits.
uint64_t amdf_xdna_umd_kernel_queue_query_progress(
    const amdf_xdna_umd_kernel_queue_t* queue);

// Observes command-owned completion data after native progress proves
// retirement. The caller exclusively owns the pending slot and retains the
// native packet throughout this call. Records execution failure without
// delaying retirement.
// This can run from the no-syscall status path. It takes no lock and performs
// no allocation, initialization or wait; packet inspection and atomic terminal
// status publication are its only native-state work.
void amdf_xdna_umd_kernel_queue_retire_command(
    amdf_xdna_umd_kernel_queue_t* queue);

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

// Releases an idle native queue lease.
amdf_status_t amdf_xdna_umd_kernel_queue_destroy(
    amdf_xdna_umd_kernel_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_KERNEL_QUEUE_H_
