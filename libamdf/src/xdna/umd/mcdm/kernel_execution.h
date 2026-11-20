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
#include "libamdf/src/xdna/umd/kernel_queue.h"
#include "libamdf/src/xdna/umd/mcdm/private_allocation.h"
#include "libamdf/src/xdna/umd/result_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_windows_xdna_kernel_execution_t
    amdf_windows_xdna_kernel_execution_t;

// The native context binds one instruction allocation of this exact size.
#define AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE UINT64_C(0x4000000)

// Byte offset of the caller's first instruction after the retained CU setup
// PDI, command-list cells, and independently aligned ELF admission program.
uint64_t amdf_windows_xdna_instruction_prefix_size(
    uint64_t instruction_alignment);

// Realizes and binds caller-owned instruction backing to this context. The
// allocation owns partial progress on failure. The context admits one such
// binding in its lifetime and records failed bootstrap as terminal.
amdf_status_t amdf_windows_xdna_kernel_execution_prepare_memory(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_windows_xdna_private_allocation_t* allocation);

// Observes any accepted bootstrap before private memory can be released. This
// does not wait for application work: the caller has already retired that work.
amdf_status_t amdf_windows_xdna_kernel_execution_release_memory(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_windows_xdna_private_allocation_t* allocation);

// Submits and waits for one Direct-protocol result control. The caller supplies
// a validated range in the bound aperture, excludes ordinary publication and
// has retired its result writers. Native points remain separate from public
// queue points. No backing pointer or attachment state is retained here.
amdf_status_t amdf_windows_xdna_kernel_execution_result_buffer(
    amdf_windows_xdna_kernel_execution_t* execution,
    amdf_xdna_result_buffer_operation_t operation,
    const amdf_windows_xdna_private_allocation_t* allocation,
    uint64_t native_byte_offset, uint64_t byte_length);

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

// Acquires the context's exclusive public KMQ lease and prepares every native
// packet/result slot. One resident fixed-stride packet ring and one response
// allocation serve all slots; native creation and teardown counts are constant
// in capacity. Responses use a separate native role for kernel CPU access.
// Submission derives host/GPU addresses and cache ranges from the slot offset,
// without allocation or storage locks. Checked retirement owns slot reuse.
// A required ELF transport is admitted before publishing the lease. A failure
// after native acceptance retains its storage until retirement.
amdf_status_t amdf_windows_xdna_kernel_execution_acquire_queue(
    amdf_windows_xdna_kernel_execution_t* execution, uint32_t capacity,
    amdf_xdna_instruction_features_t required_features);

// Consumes one idle queue lease and its packet storage, including on native
// cleanup failure. No packet metadata remains for a later retry.
amdf_status_t amdf_windows_xdna_kernel_execution_release_queue(
    amdf_windows_xdna_kernel_execution_t* execution);

// Returns the command-array limit supported by this native protocol.
uint32_t amdf_windows_xdna_kernel_execution_maximum_command_count(
    const amdf_windows_xdna_kernel_execution_t* execution);

// Tests list-cell credits without reservation. The exclusive publisher calls
// this function; checked retirement may return credits concurrently.
bool amdf_windows_xdna_kernel_execution_has_capacity(
    const amdf_windows_xdna_kernel_execution_t* execution,
    uint32_t command_count);

// Frames a validated ordered array without touching instruction bytes or
// allocating storage. The caller owns the slot and has checked list capacity.
// Acceptance borrows the instruction allocation and its provider prefix until
// checked retirement. Rejection consumes no slot or list-cell credit.
amdf_status_t amdf_windows_xdna_kernel_execution_submit(
    amdf_windows_xdna_kernel_execution_t* execution, uint32_t slot,
    uint32_t command_count, const amdf_xdna_umd_kernel_command_t* commands,
    uint64_t* out_native_submission);

// Samples the mapped native retirement watermark.
uint64_t amdf_windows_xdna_kernel_execution_query_progress(
    const amdf_windows_xdna_kernel_execution_t* execution);

// Requests one asynchronous native-fence wake into a validated caller event.
// Zero or an already-completed point signals immediately. This neither uses
// the synchronous wait event nor retains any notification state.
amdf_status_t amdf_windows_xdna_kernel_execution_request_notification(
    amdf_windows_xdna_kernel_execution_t* execution, uint64_t native_submission,
    const amdf_native_event_t* event);

// Consumes this slot's result after native fence proof. The caller exclusively
// owns checked retirement and prevents response reuse.
void amdf_windows_xdna_kernel_execution_retire_command(
    amdf_windows_xdna_kernel_execution_t* execution, uint32_t slot);

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
