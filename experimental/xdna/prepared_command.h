// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cold XDNA instruction instances in caller-owned memory.

#ifndef IREE_EXPERIMENTAL_XDNA_PREPARED_COMMAND_H_
#define IREE_EXPERIMENTAL_XDNA_PREPARED_COMMAND_H_

#include "amdf/xdna.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// One direct HAL buffer binding resolved into an exact XDNA address domain.
//
// The command-buffer implementation validates and resolves the HAL buffer and
// its AMD attachment before constructing this record. All fields are borrowed
// during preparation. Success retains |buffer_ref.buffer| until
// prepared-command destruction. The attachment owner keeps |memory| alive for
// that lifetime; libamdf receives only the instruction memory range and cannot
// retain indirectly referenced bindings.
typedef struct iree_hal_amd_xdna_prepared_command_binding_t {
  // Direct logical HAL buffer range whose access contract is validated.
  iree_hal_buffer_ref_t buffer_ref;
  // XDNA memory attachment backing |buffer_ref.buffer|.
  amdf_memory_t* memory;
  // Byte offset of the bound range within |memory|.
  uint64_t memory_byte_offset;
  // Exact shim DMA address of the first bound byte.
  uint64_t device_address;
} iree_hal_amd_xdna_prepared_command_binding_t;

// One cold instruction instance retaining its HAL executable and buffers.
// Native memory, mappings, contexts and queues remain caller-owned.
typedef struct iree_hal_amd_xdna_prepared_command_t
    iree_hal_amd_xdna_prepared_command_t;

// Returns the storage required for initialization and steady-state execution.
// |instruction_alignment| is the endpoint's instruction-address alignment.
iree_status_t iree_hal_amd_xdna_prepared_command_query_storage_size(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t instruction_alignment, iree_host_size_t* out_byte_length);

// Instantiates one executable function into caller-owned mapped memory.
//
// |storage_range| identifies the exact mapped instruction range. |storage| maps
// its first byte and has the same length. The caller supplies an aligned
// firmware address, exclusive writable storage, and fully resolved DMA binding
// addresses. Success retains the executable and logical HAL buffers, and writes
// fixed instruction bytes. The caller publishes those bytes with the memory
// cache API before submission and preserves them until all uses have retired.
// No native objects are created or retained here. On failure, storage and
// output are unchanged and no resources are retained.
iree_status_t iree_hal_amd_xdna_prepared_command_create(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    iree_host_size_t instruction_alignment,
    const amdf_xdna_kernel_command_t* storage_range, iree_byte_span_t storage,
    iree_host_size_t binding_count,
    const iree_hal_amd_xdna_prepared_command_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_amd_xdna_prepared_command_t** out_prepared_command);

// Releases retained HAL resources. The caller has retired every native use and
// remains responsible for unmapping and destroying instruction memory.
void iree_hal_amd_xdna_prepared_command_destroy(
    iree_hal_amd_xdna_prepared_command_t* prepared_command);

// Returns the cold initialization plus execution range. The caller submits this
// once when establishing the instance's array configuration.
const amdf_xdna_kernel_command_t*
iree_hal_amd_xdna_prepared_command_initialization(
    const iree_hal_amd_xdna_prepared_command_t* prepared_command);

// Returns the immutable execution range for repeated use with fixed bindings.
const amdf_xdna_kernel_command_t* iree_hal_amd_xdna_prepared_command_execution(
    const iree_hal_amd_xdna_prepared_command_t* prepared_command);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_XDNA_PREPARED_COMMAND_H_
