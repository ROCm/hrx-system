// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_PRODUCER_INDEX_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_PRODUCER_INDEX_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Intrusive AVL node for one unaccepted queue signal.
//
// Nodes are ordered by semaphore identity and timeline value. Equal keys are
// ordered by node address so malformed duplicate signals remain representable
// until normal timeline completion reports their error.
typedef struct iree_hal_amd_xdna_queue_producer_node_t {
  // Smaller indexed signal, or NULL.
  struct iree_hal_amd_xdna_queue_producer_node_t* left;
  // Larger indexed signal, or NULL.
  struct iree_hal_amd_xdna_queue_producer_node_t* right;
  // Parent node, or NULL for the root.
  struct iree_hal_amd_xdna_queue_producer_node_t* parent;
  // Borrowed semaphore retained by the owning queue operation.
  iree_hal_semaphore_t* semaphore;
  // Timeline value published by the owning queue operation.
  uint64_t value;
  // Opaque owning producer used to exclude self dependencies.
  void* producer;
  // Height of this node including itself, or zero while unindexed.
  uint8_t height;
} iree_hal_amd_xdna_queue_producer_node_t;

// Proactor-owned index of unaccepted signals on one physical queue.
typedef struct iree_hal_amd_xdna_queue_producer_index_t {
  // Root of the intrusive AVL tree, or NULL when empty.
  iree_hal_amd_xdna_queue_producer_node_t* root;
} iree_hal_amd_xdna_queue_producer_index_t;

// Initializes an empty producer index.
void iree_hal_amd_xdna_queue_producer_index_initialize(
    iree_hal_amd_xdna_queue_producer_index_t* out_index);

// Inserts one previously unindexed producer node.
void iree_hal_amd_xdna_queue_producer_index_insert(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node);

// Removes one currently indexed producer node.
void iree_hal_amd_xdna_queue_producer_index_erase(
    iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_amd_xdna_queue_producer_node_t* node);

// Finds the lowest-value indexed signal on |semaphore| that reaches
// |minimum_value| and is not owned by |excluded_producer|. Returns NULL when
// no pending signal can satisfy the wait.
iree_hal_amd_xdna_queue_producer_node_t*
iree_hal_amd_xdna_queue_producer_index_find(
    const iree_hal_amd_xdna_queue_producer_index_t* index,
    iree_hal_semaphore_t* semaphore, uint64_t minimum_value,
    const void* excluded_producer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_QUEUE_PRODUCER_INDEX_H_
