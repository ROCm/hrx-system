// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_HOST_MAPPING_H_
#define AMDF_SRC_HOST_MAPPING_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_host_mapping_vtable_t {
  // Transfers host cache ownership over a mapping-relative range.
  amdf_status_t (*cache_control)(amdf_host_mapping_t* mapping,
                                 amdf_host_cache_operation_t operation,
                                 uint64_t byte_offset, uint64_t byte_length);
  // Releases the exact native state owned by a mapping implementation.
  amdf_status_t (*destroy_native)(amdf_host_mapping_t* mapping);
} amdf_host_mapping_vtable_t;

struct amdf_host_mapping_t {
  // Host allocator copied for direct terminal teardown.
  amdf_allocator_t host_allocator;
  // Implementation operations selected before the mapping is published.
  const amdf_host_mapping_vtable_t* vtable;
  // Memory borrowed for the lifetime of this mapping.
  amdf_memory_t* memory;
  // Immutable properties established before publication.
  amdf_host_mapping_info_t info;
};

// Initializes an unpublished mapping base and borrows its memory.
amdf_status_t amdf_host_mapping_initialize(
    amdf_host_mapping_t* mapping, const amdf_host_mapping_vtable_t* vtable,
    amdf_memory_t* memory);

// Releases the memory borrow held by an unpublished or torn-down mapping.
void amdf_host_mapping_deinitialize(amdf_host_mapping_t* mapping);

// Copies immutable host mapping properties.
amdf_status_t AMDF_CALL amdf_host_mapping_query_info(
    amdf_host_mapping_t* mapping, amdf_host_mapping_info_t* out_info);

// Performs one explicit host cache transition.
amdf_status_t AMDF_CALL amdf_host_mapping_cache_control(
    amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
    uint64_t byte_offset, uint64_t byte_length);

// Destroys one host mapping.
amdf_status_t AMDF_CALL amdf_host_mapping_destroy(amdf_host_mapping_t* mapping);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_HOST_MAPPING_H_
