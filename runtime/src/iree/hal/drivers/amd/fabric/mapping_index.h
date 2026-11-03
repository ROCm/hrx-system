// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_MAPPING_INDEX_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_MAPPING_INDEX_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/virtual_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// One installed range and its original prepared source facts. Both are borrowed
// under the caller's physical backing lifetime, including through subranges.
typedef struct iree_hal_amd_fabric_virtual_mapping_t {
  // Native source coordinates and exact target range.
  amdf_memory_mapping_update_t native;
  // Provider-owned source metadata, live through the source's final alias.
  const iree_hal_amd_fabric_mapping_source_t* source;
} iree_hal_amd_fabric_virtual_mapping_t;

typedef struct iree_hal_amd_fabric_mapping_node_t
    iree_hal_amd_fabric_mapping_node_t;
typedef struct iree_hal_amd_fabric_mapping_block_t
    iree_hal_amd_fabric_mapping_block_t;

// Ordered nonoverlapping source runs for one native page-table projection.
// Zero initialization is empty. Mutations are caller-serialized. Prepare may
// overlap queries; apply and deinitialize require exclusive access. Readers
// copy one result and never borrow a node. Physical sources remain
// caller-owned.
typedef struct iree_hal_amd_fabric_mapping_index_t {
  // Root of the balanced tree of installed ranges.
  iree_hal_amd_fabric_mapping_node_t* root;
  // Owned blocks of reusable nodes, retained until the index is deinitialized.
  iree_hal_amd_fabric_mapping_block_t* blocks;
  // Uninstalled nodes linked through their left child field.
  iree_hal_amd_fabric_mapping_node_t* available;
  // Number of nodes available without allocating.
  iree_host_size_t available_count;
} iree_hal_amd_fabric_mapping_index_t;

// Releases all metadata without allocating. Does not release physical sources.
void iree_hal_amd_fabric_mapping_index_deinitialize(
    iree_hal_amd_fabric_mapping_index_t* index,
    iree_allocator_t host_allocator);

// Reserves metadata for a validated, sorted, nonoverlapping batch before native
// mutation. Failure preserves both the installed runs and retained capacity.
// Success permits apply with no allocation or failure. Changes only node-pool
// state; queries of installed ranges may continue while allocation blocks.
iree_status_t iree_hal_amd_fabric_mapping_index_prepare(
    iree_hal_amd_fabric_mapping_index_t* index, iree_host_size_t update_count,
    const amdf_memory_mapping_update_t* updates,
    iree_allocator_t host_allocator);

// Commits the prepared batch after successful native mutation. Updates only
// affected ranges and coalesces adjacent identical source coordinates. Requires
// exclusive access for the whole batch so readers cannot see partial edits.
void iree_hal_amd_fabric_mapping_index_apply(
    iree_hal_amd_fabric_mapping_index_t* index, iree_host_size_t update_count,
    const iree_hal_memory_mapping_update_t* updates,
    const amdf_memory_mapping_update_t* native_updates);

// Copies the mapped prefix beginning at position, clamped to length. Returns
// false and zeroes the output for a hole. No index pointer escapes this query.
bool iree_hal_amd_fabric_mapping_index_query(
    const iree_hal_amd_fabric_mapping_index_t* index, uint64_t position,
    uint64_t length, iree_hal_amd_fabric_virtual_mapping_t* out_mapping);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_MAPPING_INDEX_H_
