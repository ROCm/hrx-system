// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct loading and binding of native XDNA executable storage.

#ifndef IREE_EXPERIMENTAL_XDNA_EXECUTABLE_H_
#define IREE_EXPERIMENTAL_XDNA_EXECUTABLE_H_

#include "amdf/xdna.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amd/xdna/image/image.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  // Byte offset of mapping within memory.
  uint64_t memory_byte_offset;
  // Address of mapping in the allocation's declared native address domain.
  uint64_t device_address;
} iree_hal_amd_xdna_executable_storage_t;

// Borrowed external binding resolved by the caller into a shim DMA address.
// Rows corresponding to image-declared NONE slots are ignored and may remain
// zero. The caller keeps each active logical buffer and native backing alive
// until every invocation using the binding has reached terminal completion.
typedef struct iree_hal_amd_xdna_executable_binding_t {
  // Direct logical HAL buffer range whose access contract is validated.
  iree_hal_buffer_ref_t buffer_ref;
  // XDNA memory attachment backing buffer_ref.buffer.
  amdf_memory_t* memory;
  // Byte offset of the bound range within memory.
  uint64_t memory_byte_offset;
  // Exact shim DMA address of the first bound byte.
  uint64_t device_address;
} iree_hal_amd_xdna_executable_binding_t;

// Copies declared load ranges directly into final backing and applies static
// allocation-address relocations. Only explicit load tails are zeroed; gaps are
// untouched. Argument checks precede writes. A source IO failure may leave
// partially loaded storage, which the caller cannot submit. Shared immutable
// backing may be published only after all loading and static relocation ends.
iree_status_t iree_hal_amd_xdna_executable_load(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage);

// Validates external binding ranges and patches their declared address fields
// in loaded backing. Prior users of mutable backing must have drained. The
// caller publishes mapped writes through the native cache API before
// submission. Independent storage ranges can bind the same or different images
// to different addresses while other ranges remain pending. Binding modifies
// only the supplied storage; it establishes no queue-global argument state.
iree_status_t iree_hal_amd_xdna_executable_bind(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    iree_host_size_t binding_count,
    const iree_hal_amd_xdna_executable_binding_t* bindings);

// Resolves an independent invocation to a native command over caller-owned
// backing. The entry's invocation zero establishes its tile state on every
// submission. The command may be reused after terminal completion while its
// backing and bindings remain valid; no host reload or relocation is required.
// Time-sliced contexts do not guarantee resident state between submissions,
// so this finite execution adapter does not follow image continuations.
iree_status_t iree_hal_amd_xdna_executable_query_invocation(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    amdf_xdna_kernel_command_t* out_command);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_XDNA_EXECUTABLE_H_
