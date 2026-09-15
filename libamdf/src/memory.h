// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_H_
#define AMDF_SRC_MEMORY_H_

#include "amdf/amdf.h"
#include "libamdf/src/memory_resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Obtains one scoped backing and all requested live-device accesses.
amdf_status_t AMDF_CALL amdf_memory_create(
    amdf_memory_scope_t* scope, const amdf_memory_create_info_t* create_info,
    amdf_memory_t** out_memory);

// Imports one backing and establishes all requested live-device accesses.
amdf_status_t AMDF_CALL amdf_memory_import(
    amdf_memory_scope_t* scope, const amdf_memory_import_info_t* import_info,
    amdf_external_memory_t* inout_external_memory, amdf_memory_t** out_memory);

// Copies immutable memory properties.
amdf_status_t AMDF_CALL amdf_memory_query_info(amdf_memory_t* memory,
                                               amdf_memory_info_t* out_info);

// Copies immutable properties of one established device access.
amdf_status_t AMDF_CALL
amdf_memory_query_access_info(amdf_memory_t* memory, uint32_t access_ordinal,
                              amdf_memory_access_info_t* out_info);

// Returns a cached address for an established consuming interface.
amdf_status_t AMDF_CALL amdf_memory_query_address(
    amdf_memory_t* memory, uint32_t access_ordinal,
    amdf_memory_address_kind_t kind, uint64_t* out_address);

// Exports one logical range as a move-owned external value.
amdf_status_t AMDF_CALL amdf_memory_export(
    amdf_memory_t* memory, const amdf_memory_export_info_t* export_info,
    amdf_external_memory_t* out_value);

// Releases and zeros one move-owned external value.
void AMDF_CALL amdf_external_memory_release(amdf_external_memory_t* value);

// Copies exact directional facts for two concrete attachment sites.
amdf_status_t AMDF_CALL amdf_memory_query_pair_info(
    const amdf_memory_site_t* producer_site,
    const amdf_memory_site_t* consumer_site, amdf_memory_pair_info_t* out_info);

// Creates an explicit host mapping of one memory range.
amdf_status_t AMDF_CALL amdf_memory_map(amdf_memory_t* memory,
                                        const amdf_memory_map_info_t* map_info,
                                        amdf_host_mapping_t** out_mapping);

// Destroys memory with no remaining children.
amdf_status_t AMDF_CALL amdf_memory_destroy(amdf_memory_t* memory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_MEMORY_H_
