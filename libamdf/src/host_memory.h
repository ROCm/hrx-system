// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_HOST_MEMORY_H_
#define AMDF_SRC_HOST_MEMORY_H_

#include "libamdf/src/memory_profile.h"
#include "libamdf/src/platform/host_export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Describes cache maintenance for platform-owned write-back pages. Profile
// qualification and concrete host views use the same native host facts.
amdf_status_t amdf_host_memory_query_description(
    amdf_memory_host_description_t* out_description);

// Prepares system backing in the resource's private host backing slot.
// Borrowed pages remain caller-owned; allocation owns only the host VM range.
amdf_status_t amdf_host_memory_prepare(
    amdf_memory_t* memory, const amdf_memory_native_profile_t* profile,
    const amdf_memory_create_info_t* create_info, amdf_memory_info_t* out_info);

// Returns the persistent range of prepared platform-owned host backing.
// The caller holds the memory resource and has already prepared its host slot.
void* amdf_host_memory_pointer(const amdf_memory_t* memory);

// Exports the already prepared immutable host source once for one attachment.
// The first export also makes the newly pinned pages ready for device writers;
// subsequent exports reuse that readiness within the unpublished transaction.
amdf_status_t amdf_host_memory_export(
    amdf_memory_t* memory, const amdf_platform_host_export_t* service,
    bool first_export, amdf_external_memory_t* out_memory);

// Creates a lightweight WRITE_BACK view borrowing a persistent host range.
// memory is NULL for a reservation-owned range. Cache operations need only
// the retained address, geometry and direct CPU recipe, never source lookup.
amdf_status_t amdf_host_memory_create_view(
    amdf_allocator_t host_allocator, amdf_memory_t* memory, void* base_pointer,
    const amdf_memory_map_info_t* map_info, amdf_host_mapping_t** out_mapping);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_HOST_MEMORY_H_
