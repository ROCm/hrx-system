// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_STORAGE_RELOCATION_H_
#define EXPERIMENTAL_LOOM_SERVE_STORAGE_RELOCATION_H_

#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-defined, validated storage geometry. A block has the same physical ID
// in each plane; planes and blocks occupy disjoint ranges in the reservation.
typedef struct loom_serve_block_region_t {
  // Byte origin of the first plane in its allocation.
  iree_device_size_t origin;
  // Number of identical planes addressed by every block ID.
  iree_host_size_t count;
  // Byte distance between adjacent plane origins.
  iree_device_size_t stride;
  // Contiguous bytes occupied by one block in each plane.
  iree_device_size_t block_bytes;
} loom_serve_block_region_t;

// Commits destinations and queues copies on the execution dependency chain.
// The map has block_count entries: UINT32_MAX denotes an unowned source;
// otherwise the entry names its destination. Moved destinations are unowned
// and disjoint from every source. No logical owner changes here. The caller
// excludes model work, drains accepted copies even on failure, then publishes
// new logical maps before reclaiming old storage. Host metadata is captured
// before return. out_copied_bytes counts successfully submitted copy bytes.
// reservation owns physical commitment; buffer is its whole exported view,
// including the caller's retirement tracking. Copies retain that exact view,
// not the reservation's raw root, through terminal queue completion.
iree_status_t loom_serve_block_region_relocate(
    loom_serve_execution_t* execution, loom_serve_virtual_buffer_t* reservation,
    iree_hal_buffer_t* buffer, const loom_serve_block_region_t* region,
    uint32_t block_count, const uint32_t* destinations,
    uint64_t* out_copied_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_STORAGE_RELOCATION_H_
