// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_COMMAND_ARENA_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_COMMAND_ARENA_H_

#include "iree/hal/drivers/amd/xdna/memory.h"
#include "iree/hal/memory/tlsf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device-owned context-private instruction storage shared by all executables.
typedef struct iree_hal_amd_xdna_command_arena_t
    iree_hal_amd_xdna_command_arena_t;
typedef struct iree_hal_amd_xdna_command_slab_t
    iree_hal_amd_xdna_command_slab_t;

// One exclusively owned range within a command arena. The arena and its native
// storage outlive the allocation. The owner releases the range only after all
// native uses have retired.
typedef struct iree_hal_amd_xdna_command_allocation_t {
  // Borrowed native memory containing the range.
  amdf_memory_t* memory;
  // Borrowed persistent host mapping containing the range.
  amdf_host_mapping_t* mapping;
  // Writable host view of the exact requested range.
  iree_byte_span_t contents;
  // Byte position of the requested range within |memory|.
  uint64_t memory_byte_offset;
  // Firmware address of the first requested byte.
  uint64_t device_address;
  // Borrowed arena slab containing this allocation.
  iree_hal_amd_xdna_command_slab_t* slab;
  // TLSF block owning the range and any alignment padding.
  iree_hal_memory_tlsf_block_index_t block_index;
} iree_hal_amd_xdna_command_allocation_t;

// Prepares context-private instruction storage for concurrent HAL
// suballocation. Indivisible native apertures are acquired immediately;
// ordinary backing grows in persistent slabs during cold executable loading.
// The context is borrowed and must outlive the arena. Failure publishes no
// output or cleanup obligation.
iree_status_t iree_hal_amd_xdna_command_arena_create(
    iree_hal_amd_xdna_context_t* context, iree_allocator_t host_allocator,
    iree_hal_amd_xdna_command_arena_t** out_arena);

// Releases an empty arena and its native instruction extent.
void iree_hal_amd_xdna_command_arena_destroy(
    iree_hal_amd_xdna_command_arena_t* arena);

// Acquires an exclusive range satisfying |alignment|. Allocation is a cold
// executable-growth operation; concurrent callers are serialized internally.
iree_status_t iree_hal_amd_xdna_command_arena_allocate(
    iree_hal_amd_xdna_command_arena_t* arena, uint64_t byte_length,
    uint64_t alignment, iree_hal_amd_xdna_command_allocation_t* out_allocation);

// Returns a range after all native users have retired. A zero allocation is a
// no-op. This performs no native operation or host allocation.
void iree_hal_amd_xdna_command_arena_release(
    iree_hal_amd_xdna_command_arena_t* arena,
    iree_hal_amd_xdna_command_allocation_t* allocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_COMMAND_ARENA_H_
