// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_SUBMISSION_H_
#define AMDF_SRC_XDNA_UMD_MCDM_SUBMISSION_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/xdna/transaction_interpreter.h"
#include "libamdf/src/xdna/umd/mcdm/adapter_info.h"
#include "libamdf/src/xdna/umd/mcdm/private_allocation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum private record accepted by the native submission interface.
#define AMDF_WINDOWS_XDNA_SUBMISSION_CAPACITY 640u

// Bounded private record passed beside one KMT command submission.
typedef struct amdf_windows_xdna_submission_t {
  // Native submission record bytes.
  uint8_t bytes[AMDF_WINDOWS_XDNA_SUBMISSION_CAPACITY];
  // Number of initialized bytes in `bytes`.
  uint32_t byte_length;
} amdf_windows_xdna_submission_t;

// Size of the native prefix preceding the command copy for this interface.
uint32_t amdf_windows_xdna_submission_header_size(
    amdf_windows_xdna_protocol_t protocol);

// Builds the context-aperture publication record.
void amdf_windows_xdna_submission_build_aperture(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    amdf_windows_xdna_submission_t* out_submission);

// Builds the program-independent context initialization record.
void amdf_windows_xdna_submission_build_context_initialize(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    amdf_windows_xdna_submission_t* out_submission);

// Reports live instruction bytes for native accounting; it is not retirement.
void amdf_windows_xdna_submission_build_accounting(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    uint64_t live_byte_length, amdf_windows_xdna_submission_t* out_submission);

// Builds the private execution record adjoining one ERT packet.
void amdf_windows_xdna_submission_build_execute(
    amdf_windows_xdna_protocol_t protocol,
    const amdf_windows_xdna_private_allocation_t* execution_allocation,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    const amdf_xdna_transaction_interpreter_packet_t* packet,
    amdf_windows_xdna_submission_t* out_submission);

// Reads the context-initialization Boolean response after native retirement.
// A false response leaves this context unusable and returns DEVICE_LOST; it
// does not establish failure of other contexts on the native device.
amdf_status_t amdf_windows_xdna_submission_query_initialize_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation);

// Reads the execution response after native retirement, preserving failed ERT
// states in the firmware status domain. The driver writes this response cell,
// not the caller's execution packet. The caller holds the allocation borrow
// and prevents response reuse until this read; the result is not fence proof.
amdf_status_t amdf_windows_xdna_submission_query_execute_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MCDM_SUBMISSION_H_
