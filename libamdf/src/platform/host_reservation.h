// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_HOST_RESERVATION_H_
#define AMDF_SRC_PLATFORM_HOST_RESERVATION_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif

// Queries CPU virtual-memory geometry without opening an accelerator.
amdf_status_t amdf_platform_host_reservation_query_capabilities(
    amdf_memory_reservation_capabilities_t* out_capabilities);

// Reserves an inaccessible range with native alignment when minimum_alignment
// is zero, or the stronger supplied alignment. The entire interval lies within
// the inclusive address bounds. Failure publishes no address.
amdf_status_t amdf_platform_host_reservation_create(uint64_t byte_length,
                                                    uint64_t minimum_alignment,
                                                    uint64_t minimum_address,
                                                    uint64_t maximum_address,
                                                    void** out_pointer);

// Replaces a validated range using a prepared native source. Platforms with
// whole-view removal install one view per mapping unit; others map the range
// in one operation. The caller has retired all accesses.
amdf_status_t amdf_platform_host_reservation_replace(
    void* pointer, uint64_t byte_length, uint64_t mapping_granularity,
    uintptr_t source, uint64_t source_byte_offset, amdf_memory_access_t access);

// Removes every mapping in a validated unit-aligned range, preserving VA.
// Success completes native translation invalidation before return.
amdf_status_t amdf_platform_host_reservation_clear(void* pointer,
                                                   uint64_t byte_length);

// Clears and releases a complete reservation, including holes. Failure leaves
// unreleased native addresses in place; no later cleanup owner is introduced.
amdf_status_t amdf_platform_host_reservation_destroy(void* pointer,
                                                     uint64_t byte_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_PLATFORM_HOST_RESERVATION_H_
