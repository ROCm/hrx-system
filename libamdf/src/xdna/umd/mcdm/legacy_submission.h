// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_LEGACY_SUBMISSION_H_
#define AMDF_SRC_XDNA_UMD_MCDM_LEGACY_SUBMISSION_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/xdna/transaction_interpreter.h"
#include "libamdf/src/xdna/umd/mcdm/private_allocation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Maximum private record accepted by the retained native submission ABI.
#define AMDF_WINDOWS_XDNA_LEGACY_SUBMISSION_CAPACITY 624u

// Bounded private record passed beside one KMT command submission.
typedef struct amdf_windows_xdna_legacy_submission_t {
  // Opaque installed-ABI bytes.
  uint8_t bytes[AMDF_WINDOWS_XDNA_LEGACY_SUBMISSION_CAPACITY];
  // Number of initialized bytes in `bytes`.
  uint32_t byte_length;
} amdf_windows_xdna_legacy_submission_t;

// Builds the context-aperture publication record.
void amdf_windows_xdna_legacy_submission_build_aperture(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    amdf_windows_xdna_legacy_submission_t* out_submission);

// Builds the program-independent context initialization record.
void amdf_windows_xdna_legacy_submission_build_context_initialize(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    amdf_windows_xdna_legacy_submission_t* out_submission);

// Reports live instruction bytes for native accounting; it is not retirement.
void amdf_windows_xdna_legacy_submission_build_accounting(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* instruction_allocation,
    uint64_t live_byte_length,
    amdf_windows_xdna_legacy_submission_t* out_submission);

// Builds the private execution record adjoining one ERT packet.
void amdf_windows_xdna_legacy_submission_build_execute(
    uint32_t header_byte_length,
    const amdf_windows_xdna_private_allocation_t* execution_allocation,
    const amdf_windows_xdna_private_allocation_t* command_allocation,
    const amdf_xdna_transaction_interpreter_packet_t* packet,
    amdf_windows_xdna_legacy_submission_t* out_submission);

// Reads the context-initialization Boolean response after native retirement.
// A false response leaves this context unusable and returns DEVICE_LOST; it
// does not establish failure of other contexts on the native device.
amdf_status_t amdf_windows_xdna_legacy_submission_query_initialize_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation);

// Reads the execution response after native retirement, preserving failed ERT
// states in the firmware status domain. The driver writes this response cell,
// not the caller's execution packet. The caller holds the allocation borrow
// and prevents response reuse until this read; the result is not fence proof.
amdf_status_t amdf_windows_xdna_legacy_submission_query_execute_result(
    const amdf_windows_xdna_private_allocation_t* command_allocation);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MCDM_LEGACY_SUBMISSION_H_
