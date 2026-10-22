// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_ENDPOINT_H_
#define AMDF_SRC_ENDPOINT_H_

#include <stddef.h>

#include "amdf/amdf.h"
#include "libamdf/src/platform/endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum { AMDF_ENDPOINT_QUEUE_FAMILY_CAPACITY = 4 };

// Opens one query-only endpoint identity directly.
amdf_status_t AMDF_CALL amdf_endpoint_open(amdf_instance_t* instance,
                                           const amdf_endpoint_id_t* id,
                                           amdf_endpoint_t** out_endpoint);

// Copies the immutable queue-family records selected during provider open.
void amdf_endpoint_set_queue_families(
    amdf_endpoint_t* endpoint, uint32_t queue_family_count,
    const amdf_queue_family_info_t* queue_families);

// Returns immutable passive families borrowed for the endpoint's lifetime.
const amdf_queue_family_info_t* amdf_endpoint_get_queue_families(
    const amdf_endpoint_t* endpoint);

// Returns the embedded local descriptor after its storage has been qualified.
amdf_memory_scope_t* amdf_endpoint_local_memory_scope(
    amdf_endpoint_t* endpoint);

// Copies immutable endpoint properties cached during open.
amdf_status_t AMDF_CALL amdf_endpoint_query_info(
    amdf_endpoint_t* endpoint, amdf_endpoint_info_t* out_info);

// Copies one immutable endpoint-local queue-family record.
amdf_status_t AMDF_CALL amdf_endpoint_query_queue_family_info(
    amdf_endpoint_t* endpoint, uint32_t queue_family_ordinal,
    amdf_queue_family_info_t* out_info);

// Closes an endpoint and releases its instance borrow.
amdf_status_t AMDF_CALL amdf_endpoint_close(amdf_endpoint_t* endpoint);

// Returns immutable endpoint information borrowed from `endpoint`.
const amdf_endpoint_info_t* amdf_endpoint_get_cached_info(
    const amdf_endpoint_t* endpoint);

// Returns the borrowed platform endpoint.
amdf_platform_endpoint_t* amdf_endpoint_get_platform(amdf_endpoint_t* endpoint);

// Returns the host allocator owned by the endpoint's provider instance.
amdf_allocator_t amdf_endpoint_host_allocator(const amdf_endpoint_t* endpoint);

// Returns the provider instance borrowed by `endpoint`.
amdf_instance_t* amdf_endpoint_get_instance(const amdf_endpoint_t* endpoint);

// Adopts one complete immutable profile allocated with the endpoint allocator.
// Its first member is the engine's public information structure, permitting
// consumers to read retained target facts without a private family dependency.
// The full metadata extent belongs to this allocation, freed on close.
void amdf_endpoint_store_engine_profile(amdf_endpoint_t* endpoint,
                                        amdf_output_structure_t* profile);

// Borrows the public prefix of qualified engine metadata, or NULL when no
// profile was established. This reads retained facts without native work.
const amdf_output_structure_t* amdf_endpoint_get_cached_engine_info(
    const amdf_endpoint_t* endpoint);

// Caches a terminal engine-profile qualification failure.
void amdf_endpoint_store_engine_profile_error(amdf_endpoint_t* endpoint,
                                              amdf_status_t status);

// Returns the cached profile for |expected_engine_kind| or its exact failure.
amdf_status_t amdf_endpoint_query_engine_profile(
    const amdf_endpoint_t* endpoint, amdf_engine_kind_t expected_engine_kind,
    const void** out_profile);

// Registers a materialized device borrowing this endpoint.
amdf_status_t amdf_endpoint_register_device(amdf_endpoint_t* endpoint);

// Unregisters one previously registered device.
void amdf_endpoint_unregister_device(amdf_endpoint_t* endpoint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_ENDPOINT_H_
