// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_MEMORY_GEOMETRY_H_
#define LOOM_RUNTIME_STORAGE_MEMORY_GEOMETRY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Virtual replacement and physical backing geometry qualified against one
// mapping source and its targets. Logical tensor/block sizes are independent.
typedef struct loom_runtime_memory_geometry_t {
  // Maximum independently allocated and replaced virtual extent in bytes.
  iree_device_size_t mapping_size;
  // Physical parent extent from which mapping sources are allocated.
  iree_device_size_t backing_size;
  // Common source/target alignment for physical extents and virtual addresses.
  iree_device_size_t granularity;
  // Minimum native extent, including source/target alignment.
  iree_device_size_t minimum_size;
} loom_runtime_memory_geometry_t;

// Qualifies an explicit physical backing extent, or the source recommendation
// when zero, against the native source and all targets in its scope. Mapping
// and backing sizes are initially equal; pool policy may divide backing into
// smaller independently replaced mappings.
iree_status_t loom_runtime_memory_pool_query_geometry(
    const iree_hal_virtual_buffer_capabilities_t* capabilities,
    iree_hal_pool_t* storage_pool, iree_device_size_t backing_size,
    loom_runtime_memory_geometry_t* out_geometry);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_STORAGE_MEMORY_GEOMETRY_H_
