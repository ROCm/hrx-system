// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_INSTANCE_H_
#define AMDF_SRC_PLATFORM_INSTANCE_H_

#include <stdint.h>

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_platform_instance_t amdf_platform_instance_t;

// Creates the compile-time-selected platform instance.
amdf_status_t amdf_platform_instance_create(
    amdf_allocator_t host_allocator, amdf_platform_instance_t** out_instance);

// Destroys a platform instance after all platform children are closed.
amdf_status_t amdf_platform_instance_destroy(
    amdf_platform_instance_t* instance);

// Returns the system virtual-memory allocation granularity in bytes. This
// queries host facts only and initializes no accelerator or address space.
uint64_t amdf_platform_instance_host_allocation_granularity(
    const amdf_platform_instance_t* instance);

// Serializes cold shared-native preparation and native device teardown.
// The nonrecursive lock may span native IO. Queue and allocation hot paths do
// not acquire it. The instance must remain live through the matching unlock.
void amdf_platform_instance_lock_native(amdf_platform_instance_t* instance);

// Releases the cold native-state lock held by the calling thread.
void amdf_platform_instance_unlock_native(amdf_platform_instance_t* instance);

// Enumerates a bounded snapshot of normalized execution endpoints. Success and
// BUFFER_TOO_SMALL publish the output prefix and total together; every other
// result leaves both outputs unchanged.
amdf_status_t amdf_platform_endpoint_enumerate(
    amdf_platform_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_INSTANCE_H_
