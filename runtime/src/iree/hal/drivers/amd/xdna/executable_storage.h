// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct loading and binding of native XDNA executable storage.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_STORAGE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_STORAGE_H_

#include "amdf/xdna.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/image/image.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loading policy for a prepared allocation view.
typedef enum iree_hal_amd_xdna_executable_storage_flag_bits_e {
  IREE_HAL_AMD_XDNA_EXECUTABLE_STORAGE_FLAG_NONE = 0,
  // Already loaded and statically relocated immutable backing. Loading another
  // invocation leaves this allocation untouched. Its static relocation sources
  // must also be shared; the executable establishes that closure before reuse.
  IREE_HAL_AMD_XDNA_EXECUTABLE_STORAGE_FLAG_SHARED = 1u << 0,
} iree_hal_amd_xdna_executable_storage_flag_bits_t;
typedef uint32_t iree_hal_amd_xdna_executable_storage_flags_t;

// Borrowed backing in entry-relative allocation-use order. The caller resolves
// each allocation's declared command or DMA address domain and owns its memory,
// mapping and context. Loading needs exclusive write access; execution retains
// all backing through terminal completion, including indirectly referenced DMA
// storage. No operation below allocates or retains native resources.
typedef struct iree_hal_amd_xdna_executable_storage_t {
  // Writable mapping beginning at memory_byte_offset.
  iree_byte_span_t mapping;
  // Native memory owning the mapped range.
  amdf_memory_t* memory;
  // Native device-access ordinal associated with the command address.
  uint32_t access_ordinal;
  // Whether loading may write this view or borrows initialized shared backing.
  iree_hal_amd_xdna_executable_storage_flags_t flags;
  // Byte offset of mapping within memory.
  uint64_t memory_byte_offset;
  // Address of mapping in the allocation's declared native address domain.
  uint64_t device_address;
} iree_hal_amd_xdna_executable_storage_t;

// Captured external binding whose native address is published after queue
// prerequisites resolve. Rows corresponding to image-declared NONE slots are
// ignored and may remain zero. The caller keeps each active logical buffer and
// its native backing alive through terminal completion.
typedef struct iree_hal_amd_xdna_executable_binding_t {
  // Direct logical HAL buffer range validated at queue capture.
  iree_hal_buffer_ref_t buffer_ref;
  // Stable native table slot selected for the exact program family.
  iree_hal_buffer_native_binding_slot_t slot;
  // Validated logical range length in bytes.
  iree_device_size_t byte_length;
  // Shim DMA address loaded once after queue prerequisites resolve.
  uint64_t device_address;
} iree_hal_amd_xdna_executable_binding_t;

// Copies declared load ranges directly into final backing and applies static
// allocation-address relocations. Only explicit load tails are zeroed; gaps are
// untouched. Argument checks precede writes. A source IO failure may leave
// partially loaded storage, which the caller cannot submit. Shared immutable
// backing may be published only after all loading and static relocation ends.
iree_status_t iree_hal_amd_xdna_executable_storage_load(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage);

// Validates one externally supplied address against an admitted relocation.
// This remains fallible because queue-ordered native binding publication is a
// public sequencing boundary.
iree_status_t iree_hal_amd_xdna_executable_storage_validate_relocation(
    const iree_xdna_elf_relocation_record_t* relocation, uint64_t base_address);

// Applies validated external binding addresses to invocation-private backing.
// The image, storage, binding counts, logical ranges, and every relocation
// address have already been validated. Prior users of mutable backing have
// drained. This trusted transform performs no allocation or native operation.
void iree_hal_amd_xdna_executable_storage_patch(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    const iree_hal_amd_xdna_executable_binding_t* bindings);

// Resolves an independent invocation to a native command over caller-owned
// backing. The entry's invocation zero establishes its tile state on every
// submission. The command may be reused after terminal completion while its
// backing and bindings remain valid; no host reload or relocation is required.
// Time-sliced contexts do not guarantee resident state between submissions,
// so this finite execution adapter does not follow image continuations.
iree_status_t iree_hal_amd_xdna_executable_storage_query_invocation(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    amdf_xdna_kernel_command_t* out_command);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_EXECUTABLE_STORAGE_H_
