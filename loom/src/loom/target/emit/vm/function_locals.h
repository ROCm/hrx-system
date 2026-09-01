// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM function-local storage layout and transfer encoding.

#ifndef LOOM_TARGET_EMIT_VM_FUNCTION_LOCALS_H_
#define LOOM_TARGET_EMIT_VM_FUNCTION_LOCALS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/packet.h"
#include "loom/format/bytecode/writer/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

// VM frame-local storage bank selected for one Low storage reservation.
typedef uint8_t loom_vm_function_local_bank_t;
enum loom_vm_function_local_bank_e {
  // Reservation has no runtime spill/reload traffic.
  LOOM_VM_FUNCTION_LOCAL_BANK_NONE = 0,
  // Raw value bits stored in the local-byte bank.
  LOOM_VM_FUNCTION_LOCAL_BANK_VALUE = 1,
  // VM ref states stored in the local-ref bank.
  LOOM_VM_FUNCTION_LOCAL_BANK_REF = 2,
  // Non-owning function values stored in the local-function bank.
  LOOM_VM_FUNCTION_LOCAL_BANK_FUNCTION = 3,
};

// Aggregate VM frame-local storage requirements.
typedef struct loom_vm_function_local_counts_t {
  // Number of addressable bytes in the local-byte bank.
  uint16_t byte_length;
  // Number of entries in the local-ref bank.
  uint32_t ref_count;
  // Number of entries in the local-function bank.
  uint32_t function_count;
} loom_vm_function_local_counts_t;

// VM placement of one Low storage reservation.
typedef struct loom_vm_function_local_reservation_t {
  // VM frame-local bank containing the reservation.
  loom_vm_function_local_bank_t bank;
  // First byte or slot ordinal in |bank|.
  uint32_t base;
} loom_vm_function_local_reservation_t;

// Exact VM projection of function-local Low storage.
typedef struct loom_vm_function_local_layout_t {
  // Complete frame-local counts, including the reusable call ABI prefix.
  loom_vm_function_local_counts_t counts;
  // Arena-owned placements in Low reservation declaration order.
  const loom_vm_function_local_reservation_t* reservations;
  // Number of entries in |reservations|.
  iree_host_size_t reservation_count;
} loom_vm_function_local_layout_t;

// Builds the VM projection of function-local Low storage. |call_prefix| is the
// maximum reusable caller-local prefix required by any call in the function;
// stable spill reservations are packed after that prefix.
iree_status_t loom_vm_function_local_layout_build(
    const loom_low_emission_frame_t* frame,
    loom_vm_function_local_counts_t call_prefix, iree_arena_allocator_t* arena,
    loom_vm_function_local_layout_t* out_layout);

// Returns true when |packet| is structural low.spill/low.reload traffic.
bool loom_vm_function_local_transfer_is_packet(
    const loom_low_packet_view_t* packet);

// Returns the exact encoded byte length of one verified local transfer.
uint32_t loom_vm_function_local_transfer_byte_length(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet);

// Encodes one verified local transfer using |layout|.
iree_status_t loom_vm_function_local_transfer_encode(
    const loom_low_emission_frame_t* frame,
    const loom_vm_function_local_layout_t* layout,
    const loom_low_packet_view_t* packet, loom_bytecode_page_writer_t* writer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_FUNCTION_LOCALS_H_
