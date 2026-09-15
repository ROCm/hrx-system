// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_KERNEL_EXECUTION_H_
#define AMDF_SRC_XDNA_UMD_MCDM_KERNEL_EXECUTION_H_

#include <stdint.h>

#include "libamdf/src/wait.h"
#include "libamdf/src/xdna/umd/context.h"
#include "libamdf/src/xdna/umd/mcdm/legacy_submission.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_windows_xdna_kernel_execution_t
    amdf_windows_xdna_kernel_execution_t;

// The native context binds one allocation. Its bootstrap prefix is reserved
// for provider initialization; the remaining logical range belongs to callers.
#define AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE UINT64_C(0x4000000)
#define AMDF_WINDOWS_XDNA_PRIVATE_BOOTSTRAP_SIZE UINT64_C(0x8000)

// Realizes and binds caller-owned instruction backing to this context. The
// allocation owns partial progress on failure. The context admits one such
// binding in its lifetime and records failed bootstrap as terminal.
amdf_status_t amdf_windows_xdna_kernel_execution_prepare_memory(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_windows_xdna_private_allocation_t* allocation);

// Observes any accepted bootstrap before private memory can be released. This
// does not wait for application work: the caller has already retired that work.
amdf_status_t amdf_windows_xdna_kernel_execution_release_memory(
    amdf_windows_xdna_kernel_execution_t* execution);

// Allocates inert host state for one context-owned execution path.
amdf_status_t amdf_windows_xdna_kernel_execution_create(
    amdf_xdna_umd_context_t* context,
    amdf_windows_xdna_kernel_execution_t** out_execution);

// Proves that no native work or public child remains and destroys the hardware
// queue before its parent KMT context is destroyed.
amdf_status_t amdf_windows_xdna_kernel_execution_prepare_context_destroy(
    amdf_windows_xdna_kernel_execution_t* execution);

// Releases context-private allocations after the parent KMT context is gone.
amdf_status_t amdf_windows_xdna_kernel_execution_destroy(
    amdf_windows_xdna_kernel_execution_t* execution);

// Acquires the context's single known-correct public KMQ lease.
amdf_status_t amdf_windows_xdna_kernel_execution_acquire_queue(
    amdf_windows_xdna_kernel_execution_t* execution);

// Releases one queue lease after all accepted work has retired.
void amdf_windows_xdna_kernel_execution_release_queue(
    amdf_windows_xdna_kernel_execution_t* execution);

// Frames and publishes one validated instruction range without touching its
// bytes or allocating submission storage.
amdf_status_t amdf_windows_xdna_kernel_execution_submit(
    amdf_windows_xdna_kernel_execution_t* execution,
    uint64_t instruction_address, uint32_t instruction_byte_length,
    uint64_t* out_native_submission);

// Samples the mapped native retirement watermark.
uint64_t amdf_windows_xdna_kernel_execution_query_progress(
    const amdf_windows_xdna_kernel_execution_t* execution);

// Consumes the single pending command's result after native fence proof. The
// caller exclusively owns software retirement and prevents response reuse.
void amdf_windows_xdna_kernel_execution_retire_command(
    amdf_windows_xdna_kernel_execution_t* execution);

// Returns the observed context or device terminal failure without polling.
amdf_status_t amdf_windows_xdna_kernel_execution_query_terminal_status(
    const amdf_windows_xdna_kernel_execution_t* execution);

// Waits for one native submission with caller-selected active polling.
amdf_status_t amdf_windows_xdna_kernel_execution_wait(
    amdf_windows_xdna_kernel_execution_t* execution, uint64_t native_submission,
    const amdf_wait_deadline_t* deadline);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MCDM_KERNEL_EXECUTION_H_
